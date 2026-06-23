"""Standalone RFNBO Hybrid Solar-Ratio Sweep CLI.

This script evaluates how the internal technology mix of a hybrid Power Purchase 
Agreement (PPA) impacts the overall Renewable Fuels of Non-Biological Origin (RFNBO) 
compliance for a hydrogen electrolyser. 

It sweeps the solar share (from 0% to 100%) inside combined PPAs (Solar + Offshore Wind, 
and Solar + Onshore Wind) across multiple overall PPA sizes, generating dedicated datasets 
and interactive plots.

### Sizing Logic (Installed MW Capacity Split)
This script divides the physical **Installed Capacity (MW)** of the PPA, not the energy 
volume. It does not use capacity factors. 
* Total PPA MW = `Electrolyser MW` * `ppa_ratio`
* Solar MW = `Total PPA MW` * `solar_ratio`
* Wind MW = `Total PPA MW` * (1 - `solar_ratio`)
* *Example:* For a 100 MW electrolyser, a `ppa_ratio` of 1.0, and a `solar_ratio` of 0.4, 
  the script sizes the PPA as exactly 40 MW Solar and 60 MW Wind.

### Expected Inputs
* **Market Data:** ENTSO-E generation, prices, and installed capacity CSVs.
* **Electrolyser Capacity:** Required baseline load defined via `--electrolyser-mw`.
* **Sweep Parameters:** * `--ppa-ratios`: The overall size of the PPA relative to the electrolyser (Default: 0.75, 1.0, 1.25).
  * `--solar-ratios`: The percentage of that PPA dedicated to solar (Default: 0.0 to 1.0 in 200 steps).

### Expected Outputs
For each combination of Temporal Correlation (Hourly/Monthly) and Technology Scenario 
(Offshore/Onshore), the script generates:
1.  **Interval CSV (`*_solar_ratio_sweep.csv`):** A granular dataset containing the specific 
    RFNBO percentage for every tested solar split and base PPA ratio.
2.  **Summary Plot (`*_solar_ratio_sweep.html`):** An interactive line chart displaying the 
    Ratio of Solar Production (X-axis) against the overall RFNBO % (Y-axis), with separate 
    lines representing the different overall PPA-to-electrolyser sizes.

### Example CLI Usage

**1. Standard Sweep (Defaults to Belgium, Both Modes):**
Runs the default 201-point solar sweep across base PPA ratios of 0.75, 1.0, and 1.25 for a 50 MW electrolyser. Evaluates both hourly and monthly correlation and opens all 4 plots.
> python rfnbo_solar_ratio_sweep.py --country Belgium --electrolyser-mw 50 --open-html

**2. Targeted Hourly Sweep with Custom Ratios:**
Sweeps a tighter range of solar ratios (0.2 to 0.8) for an oversized PPA (1.5x and 2.0x the electrolyser MW), restricting the output to strictly hourly correlation.
> python rfnbo_solar_ratio_sweep.py \
    --country Germany \
    --electrolyser-mw 100 \
    --ppa-ratios 1.5 2.0 \
    --solar-ratios 0.2 0.4 0.5 0.6 0.8 \
    --temporal-correlation hourly \
    --open-html

**3. Specific Date Range:**
Runs the default sweep for a specific historical year.
> python rfnbo_solar_ratio_sweep.py \
    --country France \
    --electrolyser-mw 20 \
    --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from RFNBO_orchestrator import (
    _filter_date_range,
    _find_latest_file,
    _load_generation,
    _load_installed_capacity,
    _load_prices,
)
from rfnbo_calculations import (
    aggregate_to_monthly,
    calculate_renewable_share,
    calculate_rfnbo_compliance,
    get_grid_emission_factor,
)


TECHNOLOGY_SCENARIOS = [
    ("Solar + Wind Offshore", "offshore"),
    ("Solar + Wind Onshore", "onshore"),
]

BASE_PPA_RATIOS = [0.75, 1.0, 1.25]
SCALE = 1
DEFAULT_PPA_RATIOS = [round(value * SCALE, 3) for value in BASE_PPA_RATIOS]
DEFAULT_SOLAR_RATIOS = [round(index / 200, 3) for index in range(201)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse how the solar share inside a combined PPA changes RFNBO compliance "
            "for offshore and onshore wind combinations."
        )
    )
    parser.add_argument("--country", default="Belgium", help="Country folder name under entsoe_data")
    parser.add_argument("--data-dir", default="entsoe_data", help="Base directory containing country subfolders")
    parser.add_argument(
        "--temporal-correlation",
        choices=["hourly", "monthly", "all"],
        default="all",
        help="Generate one mode or both hourly and monthly graphs",
    )
    parser.add_argument(
        "--ppa-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_PPA_RATIOS,
        help="PPA-to-electrolyser ratios to plot as separate lines",
    )
    parser.add_argument(
        "--solar-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_SOLAR_RATIOS,
        help="Solar shares to sweep from 0.0 to 1.0",
    )
    parser.add_argument("--prices-file", help="Optional explicit path to prices CSV")
    parser.add_argument("--generation-file", help="Optional explicit path to generation CSV")
    parser.add_argument("--capacity-file", help="Optional explicit path to installed capacity CSV")
    parser.add_argument("--start", help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--electrolyser-mw", type=float, required=True, help="Electrolyser capacity in MW")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store the generated CSV and HTML files",
    )
    parser.add_argument(
        "--open-html",
        action="store_true",
        help="Open generated HTML plots in the default browser",
    )
    return parser


def _load_input_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    country_dir = Path(args.data_dir) / args.country
    if not country_dir.exists():
        raise FileNotFoundError(f"Country folder not found: {country_dir}")

    prices_file = (
        Path(args.prices_file)
        if args.prices_file
        else _find_latest_file(country_dir, f"{args.country}_prices_*.csv")
    )
    generation_file = (
        Path(args.generation_file)
        if args.generation_file
        else _find_latest_file(country_dir, f"{args.country}_generation_*.csv")
    )
    capacity_file = (
        Path(args.capacity_file)
        if args.capacity_file
        else _find_latest_file(country_dir, f"{args.country}_installed_capacity_*.csv")
    )

    prices_df = _load_prices(prices_file)
    generation_df = _load_generation(generation_file)
    capacity_df = _load_installed_capacity(capacity_file)

    prices_df, generation_df = _filter_date_range(prices_df, generation_df, args.start, args.end)

    if prices_df.empty:
        raise ValueError("No prices data left after applying date filters.")
    if generation_df.empty:
        raise ValueError("No generation data left after applying date filters.")

    return prices_df, generation_df, capacity_df


def _selected_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["hourly", "monthly"]
    return [mode]


def _sweep_solar_ratio(
    prices_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    country: str,
    electrolyser_mw: float,
    ppa_ratio: float,
    solar_ratio: float,
    technology: str,
    mode: str,
    renewable_share: float,
) -> pd.DataFrame:
    ppa_capacity_mw = electrolyser_mw * ppa_ratio
    wind_fraction = 1.0 - solar_ratio

    compliance_df = calculate_rfnbo_compliance(
        electrolyser_mw=electrolyser_mw,
        ppa_capacity_mw=ppa_capacity_mw,
        prices_df=prices_df,
        renewable_share=renewable_share,
        zone_name=country,
        temporal_correlation=mode,
        use_price_threshold=True,
        ppa_technology=technology,
        generation_df=generation_df,
        installed_capacity_df=capacity_df,
        solar_fraction=solar_ratio,
        wind_fraction=wind_fraction,
    )

    if compliance_df.empty:
        return compliance_df

    if mode == "monthly":
        country_emission_factor = get_grid_emission_factor(country)
        compliance_df = aggregate_to_monthly(compliance_df, country_emission_factor)

    total_consumption_mwh = compliance_df["electrolyser_consumption_mwh"].sum()
    total_rfnbo_mwh = compliance_df["rfnbo_energy_mwh"].sum()
    overall_rfnbo_fraction = (total_rfnbo_mwh / total_consumption_mwh) if total_consumption_mwh > 0 else 0.0

    return pd.DataFrame([
        {
            "country": country,
            "technology": technology,
            "temporal_correlation": mode,
            "ppa_to_electrolyser_ratio": ppa_ratio,
            "ppa_capacity_mw": ppa_capacity_mw,
            "electrolyser_mw": electrolyser_mw,
            "solar_ratio": solar_ratio,
            "wind_ratio": wind_fraction,
            "overall_rfnbo_fraction": overall_rfnbo_fraction,
            "overall_rfnbo_pct": overall_rfnbo_fraction * 100,
            "total_consumption_mwh": total_consumption_mwh,
            "total_rfnbo_mwh": total_rfnbo_mwh,
            "interval_count": len(compliance_df),
        }
    ])


def _build_ratio_plot(df: pd.DataFrame, country: str, mode: str, technology: str, output_path: Path) -> None:
    if df.empty:
        return

    fig = go.Figure()
    palette = {
        0.75: "#9DC3E6",
        1.0: "#5B9BD5",
        1.25: "#2F5597",
    }

    for ppa_ratio in sorted(df["ppa_to_electrolyser_ratio"].unique()):
        ratio_df = df[df["ppa_to_electrolyser_ratio"] == ppa_ratio].sort_values("solar_ratio")
        if ratio_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=ratio_df["solar_ratio"],
                y=ratio_df["overall_rfnbo_pct"],
                mode="lines",
                name=f"Prod. to cons. ratio = {ppa_ratio:g}",
                line=dict(color=palette.get(float(ppa_ratio), "#1f77b4"), width=3),
                hovertemplate=(
                    "Solar ratio=%{x:.3f}<br>RFNBO=%{y:.1f}%"
                    f"<extra>Prod. to cons. ratio = {ppa_ratio:g}</extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Combined {technology} - solar ratio influence - {country} ({mode})",
        xaxis_title="Ratio of solar production",
        yaxis_title="% RFNBO H2",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=70, r=30, t=70, b=100),
    )
    fig.update_xaxes(tickmode="linear", dtick=0.2)
    fig.update_yaxes(dtick=20)
    fig.write_html(output_path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.electrolyser_mw <= 0:
        raise ValueError("`--electrolyser-mw` must be greater than 0.")

    solar_ratios = sorted(set(args.solar_ratios))
    ppa_ratios = sorted(set(args.ppa_ratios))
    if not solar_ratios:
        raise ValueError("`--solar-ratios` must contain at least one value.")
    if not ppa_ratios:
        raise ValueError("`--ppa-ratios` must contain at least one value.")

    for solar_ratio in solar_ratios:
        if solar_ratio < 0.0 or solar_ratio > 1.0:
            raise ValueError("All `--solar-ratios` values must be between 0 and 1.")

    prices_df, generation_df, capacity_df = _load_input_data(args)
    renewable_share = calculate_renewable_share(generation_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = _selected_modes(args.temporal_correlation)

    all_frames: list[pd.DataFrame] = []
    generated_html_paths: list[Path] = []
    for mode in modes:
        for technology, technology_slug in TECHNOLOGY_SCENARIOS:
            scenario_frames: list[pd.DataFrame] = []
            for ppa_ratio in ppa_ratios:
                total_solar_ratios = len(solar_ratios)
                for solar_index, solar_ratio in enumerate(solar_ratios, start=1):
                    if solar_index == 1 or solar_index % 10 == 0 or solar_index == total_solar_ratios:
                        print(
                            f"[{mode} | {technology_slug} | base PPA ratio {ppa_ratio:g}] "
                            f"solar ratio {solar_index}/{total_solar_ratios}: {solar_ratio:.3f}"
                        )

                    frame = _sweep_solar_ratio(
                        prices_df=prices_df,
                        generation_df=generation_df,
                        capacity_df=capacity_df,
                        country=args.country,
                        electrolyser_mw=args.electrolyser_mw,
                        ppa_ratio=ppa_ratio,
                        solar_ratio=solar_ratio,
                        technology=technology,
                        mode=mode,
                        renewable_share=renewable_share,
                    )
                    if not frame.empty:
                        scenario_frames.append(frame)

            if not scenario_frames:
                continue

            scenario_df = pd.concat(scenario_frames, ignore_index=True)
            scenario_df = scenario_df.sort_values(["ppa_to_electrolyser_ratio", "solar_ratio"])

            csv_path = output_dir / f"{args.country}_{mode}_{technology_slug}_solar_ratio_sweep.csv"
            html_path = output_dir / f"{args.country}_{mode}_{technology_slug}_solar_ratio_sweep.html"
            scenario_df.to_csv(csv_path, index=False)
            _build_ratio_plot(scenario_df, args.country, mode, technology, html_path)

            all_frames.append(scenario_df)
            generated_html_paths.append(html_path)
            print(csv_path)
            print(html_path)

    if not all_frames:
        raise ValueError("RFNBO solar-ratio sweep produced no rows.")

    if args.open_html:
        for html_path in generated_html_paths:
            webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()