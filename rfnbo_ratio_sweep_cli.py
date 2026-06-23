"""Standalone RFNBO Capacity Ratio Sweep CLI.

This script evaluates how oversizing or undersizing a Power Purchase Agreement (PPA) 
impacts the Renewable Fuels of Non-Biological Origin (RFNBO) compliance of a hydrogen 
electrolyser. 

It sweeps the approved technology set across a range of PPA-to-electrolyser ratios 
and writes a detailed CSV containing the interval-level overall RFNBO percentage for 
each combination, alongside an aggregated visualization.

### Sizing Logic (Installed Capacity)
Unlike energy-based models, the ratio in this script directly scales the **installed 
capacity (MW)**. 
* A ratio of 1.0 means: `PPA MW = Electrolyser MW`.
* For combined/hybrid technologies (e.g., "Solar + Wind Offshore"), the script strictly 
  applies a 50/50 MW capacity split (e.g., a 100 MW hybrid PPA consists of 50 MW Solar 
  and 50 MW Wind).

### Expected Inputs
* **Market Data:** ENTSO-E generation, prices, and installed capacity CSVs (loaded from the specified `--data-dir`).
* **Electrolyser Capacity:** Required baseline load defined via `--electrolyser-mw`.
* **Sweep Parameters:** Technologies to test (defaults to all five approved profiles) and 
  the specific ratios to sweep (defaults to 200 points between 0.0 and 2.0).

### Expected Outputs
1.  **Interval CSV (`*_ratio_sweep.csv`):** A granular dataset containing the specific RFNBO 
    percentage for every single time interval (hourly or monthly), for every technology, 
    at every tested ratio.
2.  **Summary Plot (`*_ratio_sweep.html`):** An interactive line chart displaying the 
    Production-to-Consumption Ratio (X-axis) against the overall RFNBO % (Y-axis), 
    categorized by technology type.

### Example CLI Usage

**1. Standard Sweep (Defaults to Belgium):**
Runs the default 200-point sweep (0.0 to 2.0) across all 5 technologies for a 50 MW electrolyser, using hourly correlation.
> python rfnbo_ratio_sweep_cli.py --country Belgium --electrolyser-mw 50

**2. Targeted Technology and Monthly Correlation:**
Sweeps only pure Solar and pure Onshore Wind using custom ratios (0.5, 1.0, 1.5, 2.0) and evaluates compliance on a monthly matching basis.
> python rfnbo_ratio_sweep_cli.py \
    --country Germany \
    --electrolyser-mw 100 \
    --technologies "Solar" "Wind Onshore" \
    --ratios 0.5 1.0 1.5 2.0 \
    --temporal-correlation monthly

**3. Specific Date Range and Fast Progress Logging:**
Runs the default sweep for a specific historical year, printing a progress update to the console after every single ratio calculated.
> python rfnbo_ratio_sweep_cli.py \
    --country France \
    --electrolyser-mw 20 \
    --start 2023-01-01 --end 2023-12-31 \
    --progress-every 1
"""

from __future__ import annotations

import argparse
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


TECHNOLOGY_SWEEP = [
    "Solar",
    "Wind Onshore",
    "Wind Offshore",
    "Solar + Wind Offshore",
    "Solar + Wind Onshore",
]

TECHNOLOGY_LABELS = {
    "Solar": "Solar",
    "Wind Onshore": "OnWind",
    "Wind Offshore": "OffWind",
    "Solar + Wind Offshore": "Solar + OffWind",
    "Solar + Wind Onshore": "Solar + OnWind",
}

DEFAULT_RATIOS = [round(index * 2.0 / 199, 6) for index in range(200)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved RFNBO ratio sweep across the five technologies and "
            "export CSV-only interval RFNBO percentages."
        )
    )
    parser.add_argument("--country", default="Belgium", help="Country folder name under entsoe_data")
    parser.add_argument("--data-dir", default="entsoe_data", help="Base directory containing country subfolders")
    parser.add_argument(
        "--temporal-correlation",
        choices=["hourly", "monthly"],
        default="hourly",
        help="Interval aggregation mode for the sweep output",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=DEFAULT_RATIOS,
        help="PPA-to-electrolyser ratios to sweep (default: 200 points from 0.0 to 2.0)",
    )
    parser.add_argument(
        "--technologies",
        nargs="+",
        default=TECHNOLOGY_SWEEP,
        choices=TECHNOLOGY_SWEEP,
        help="Technologies to include in the sweep (default: all five approved technologies)",
    )
    parser.add_argument("--prices-file", help="Optional explicit path to prices CSV")
    parser.add_argument("--generation-file", help="Optional explicit path to generation CSV")
    parser.add_argument("--capacity-file", help="Optional explicit path to installed capacity CSV")
    parser.add_argument("--start", help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--electrolyser-mw", type=float, required=True, help="Electrolyser capacity in MW")
    parser.add_argument(
        "--output-file",
        help="Optional output CSV path. Defaults to outputs/<country>_<mode>_ratio_sweep.csv",
    )
    parser.add_argument(
        "--plot-file",
        help="Optional HTML plot path. Defaults to outputs/<country>_<mode>_ratio_sweep.html",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress update every N ratios (default: 10; set to 0 to disable)",
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


def _resolve_combined_split(technology: str) -> tuple[float | None, float | None]:
    if "+" not in technology:
        return None, None
    return 0.5, 0.5


def _build_interval_summary(df: pd.DataFrame, country: str, mode: str) -> pd.DataFrame:
    if df.empty:
        return df

    summary = df.copy()
    summary["datetime"] = pd.to_datetime(summary["datetime"], utc=True)
    summary["interval_start"] = summary["datetime"]
    summary["interval_label"] = summary["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    if mode == "monthly":
        summary["interval_label"] = summary["datetime"].dt.strftime("%Y-%m")

    summary["overall_rfnbo_pct"] = summary["rfnbo_fraction"] * 100
    summary["country"] = country
    summary["temporal_correlation"] = mode
    return summary


def _build_overall_summary(
    df: pd.DataFrame,
    country: str,
    technology: str,
    ratio: float,
    electrolyser_mw: float,
    ppa_capacity_mw: float,
    mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    total_consumption_mwh = df["electrolyser_consumption_mwh"].sum()
    total_rfnbo_mwh = df["rfnbo_energy_mwh"].sum()
    overall_rfnbo_fraction = (total_rfnbo_mwh / total_consumption_mwh) if total_consumption_mwh > 0 else 0.0

    return pd.DataFrame([
        {
            "country": country,
            "technology": technology,
            "ppa_to_electrolyser_ratio": ratio,
            "electrolyser_mw": electrolyser_mw,
            "ppa_capacity_mw": ppa_capacity_mw,
            "temporal_correlation": mode,
            "overall_rfnbo_fraction": overall_rfnbo_fraction,
            "overall_rfnbo_pct": overall_rfnbo_fraction * 100,
        }
    ])


def _build_ratio_plot(df: pd.DataFrame, country: str, mode: str, output_path: Path) -> None:
    if df.empty:
        return

    fig = go.Figure()
    palette = {
        "Solar": "#D7A441",
        "Wind Onshore": "#8E8B68",
        "Wind Offshore": "#C91F1F",
        "Solar + Wind Offshore": "#8BB8E8",
        "Solar + Wind Onshore": "#2E6B2E",
    }

    for technology in TECHNOLOGY_SWEEP:
        tech_df = df[df["technology"] == technology].sort_values("ppa_to_electrolyser_ratio")
        if tech_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=tech_df["ppa_to_electrolyser_ratio"],
                y=tech_df["overall_rfnbo_pct"],
                mode="lines",
                name=TECHNOLOGY_LABELS.get(technology, technology),
                line=dict(color=palette.get(technology, "#1f77b4"), width=3),
                hovertemplate=(
                    f"Ratio=%{{x:.3f}}<br>RFNBO=%{{y:.1f}}%<extra>{TECHNOLOGY_LABELS.get(technology, technology)}</extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=f"RFNBO % vs Production to consumption ratio - {country} ({mode})",
        xaxis_title="Production to consumption ratio",
        yaxis_title="% RFNBO H2",
        xaxis=dict(range=[0, 2]),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=70, r=30, t=70, b=100),
    )
    fig.update_xaxes(tickmode="linear", dtick=0.5)
    fig.update_yaxes(dtick=20)
    fig.write_html(output_path)


def _sweep_technology(
    prices_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    country: str,
    electrolyser_mw: float,
    ratio: float,
    technology: str,
    mode: str,
    renewable_share: float,
) -> pd.DataFrame:
    ppa_capacity_mw = electrolyser_mw * ratio
    solar_fraction, wind_fraction = _resolve_combined_split(technology)

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
        solar_fraction=solar_fraction,
        wind_fraction=wind_fraction,
    )

    if compliance_df.empty:
        return compliance_df

    if mode == "monthly":
        country_emission_factor = get_grid_emission_factor(country)
        compliance_df = aggregate_to_monthly(compliance_df, country_emission_factor)

    summary_df = _build_interval_summary(compliance_df, country, mode)
    summary_df["technology"] = technology
    summary_df["ppa_to_electrolyser_ratio"] = ratio
    summary_df["ppa_capacity_mw"] = ppa_capacity_mw
    summary_df["electrolyser_mw"] = electrolyser_mw
    summary_df["overall_rfnbo_fraction"] = summary_df["rfnbo_fraction"]
    overall_summary_df = _build_overall_summary(
        compliance_df,
        country=country,
        technology=technology,
        ratio=ratio,
        electrolyser_mw=electrolyser_mw,
        ppa_capacity_mw=ppa_capacity_mw,
        mode=mode,
    )
    return summary_df, overall_summary_df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "country",
        "technology",
        "ppa_to_electrolyser_ratio",
        "electrolyser_mw",
        "ppa_capacity_mw",
        "temporal_correlation",
        "interval_start",
        "interval_label",
        "overall_rfnbo_fraction",
        "overall_rfnbo_pct",
    ]
    existing = [column for column in columns if column in df.columns]
    return df[existing].copy()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    prices_df, generation_df, capacity_df = _load_input_data(args)
    renewable_share = calculate_renewable_share(generation_df)

    total_ratios = len(args.ratios)
    progress_every = max(0, args.progress_every)

    sweep_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for technology in args.technologies:
        for ratio_index, ratio in enumerate(args.ratios, start=1):
            if progress_every and (ratio_index == 1 or ratio_index % progress_every == 0 or ratio_index == total_ratios):
                print(f"[{technology}] ratio {ratio_index}/{total_ratios}: {ratio:.6f}")

            sweep_frame, curve_frame = _sweep_technology(
                prices_df=prices_df,
                generation_df=generation_df,
                capacity_df=capacity_df,
                country=args.country,
                electrolyser_mw=args.electrolyser_mw,
                ratio=ratio,
                technology=technology,
                mode=args.temporal_correlation,
                renewable_share=renewable_share,
            )
            if not sweep_frame.empty:
                sweep_frames.append(sweep_frame)
            if not curve_frame.empty:
                curve_frames.append(curve_frame)

    if not sweep_frames:
        raise ValueError("RFNBO ratio sweep produced no rows.")

    result_df = pd.concat(sweep_frames, ignore_index=True)
    result_df = _select_output_columns(result_df)
    result_df = result_df.sort_values(["technology", "ppa_to_electrolyser_ratio", "interval_start"]).reset_index(drop=True)

    curve_df = pd.concat(curve_frames, ignore_index=True)
    curve_df = curve_df.sort_values(["technology", "ppa_to_electrolyser_ratio"]).reset_index(drop=True)

    output_path = (
        Path(args.output_file)
        if args.output_file
        else Path("outputs") / f"{args.country}_{args.temporal_correlation}_ratio_sweep.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    plot_path = (
        Path(args.plot_file)
        if args.plot_file
        else Path("outputs") / f"{args.country}_{args.temporal_correlation}_ratio_sweep.html"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _build_ratio_plot(curve_df, args.country, args.temporal_correlation, plot_path)

    print(output_path)
    print(plot_path)


if __name__ == "__main__":
    main()