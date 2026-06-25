"""RFNBO Compliance Orchestrator.

This script evaluates the Renewable Fuels of Non-Biological Origin (RFNBO) compliance 
for a hydrogen electrolyser backed by a Power Purchase Agreement (PPA). It loads local 
ENTSO-E market data, runs the compliance engine in either hourly or monthly temporal 
correlation modes, and generates a detailed CSV dataset alongside four interactive HTML visualizations.

### Core Logic & Compliance Tracking
The script calculates the total greenhouse gas (GHG) emissions and the percentage of 
hydrogen produced that qualifies as RFNBO. It accounts for:
* Direct renewable energy sourced from the PPA (capped at electrolyser capacity).
* Grid energy consumed when the day-ahead market (DAM) price drops below a specific threshold (exempt from GHG penalties).
* The average renewable share of the local grid.
* The overall emission factor of the produced hydrogen compared against the strict 28.2 g CO2eq/MJ threshold.

### Expected Inputs
* **Market Data:** Local ENTSO-E CSV files for Day-Ahead Prices, Actual Generation, and Installed Capacity.
* **System Sizing:** Installed capacity of the electrolyser (`--electrolyser-mw`) and the PPA (`--ppa-capacity-mw`).
* **Technology Profile:** The specific PPA technology (e.g., "Solar", "Wind Onshore", or combined profiles).
* **Optional Overrides:** Grid renewable share, date ranges, and solar/wind split fractions for hybrid PPAs.

### Generated Outputs
All outputs are saved to the specified `--output-dir` (default: `outputs/`):
1.  **Results Data (`*_rfnbo_results.csv`):** A granular, interval-by-interval breakdown of energy consumption, emissions, and RFNBO fractions.
2.  **GHG Emissions Plot (`*_ghg_emissions.html`):** A line chart tracking absolute greenhouse gas emissions in tonnes of CO2eq over time.
3.  **RFNBO Percentage Plot (`*_rfnbo_pct.html`):** A line chart showing the interval RFNBO compliance rate, overlaid with a dashed line representing the overall average.
4.  **Emission Factor Plot (`*_emission_factor.html`):** A line chart tracking the g CO2eq/MJ metric, featuring a critical threshold line at 28.2.
5.  **Renewable Availability Plot (`*_renewable_availability.html`):** A stacked area and line chart detailing the physical energy composition (Raw PPA production vs. Electrolyser consumption, low-price grid exemptions, and baseline grid renewable energy).

### Example CLI Usage

**1. Standard Monthly Compliance Run (Defaults to Belgium):**
Evaluates a 50 MW electrolyser paired with an 80 MW Solar PPA using monthly temporal correlation.
> python rfnbo_orchestrator.py --country Belgium --electrolyser-mw 50 --ppa-capacity-mw 80 --ppa-technology "Solar" --temporal-correlation monthly

**2. Strict Hourly Compliance Run:**
Evaluates the same system using strict hourly matching and automatically opens the HTML plots in the browser upon completion.
> python rfnbo_orchestrator.py --country Belgium --electrolyser-mw 50 --ppa-capacity-mw 80 --ppa-technology "Solar" --temporal-correlation hourly --open-html

**3. Hybrid PPA with Specific Date Range:**
Evaluates a combined Solar and Offshore Wind PPA for a specific year, explicitly defining the solar fraction of the PPA capacity to 40%.
> python rfnbo_orchestrator.py \
    --country Germany \
    --electrolyser-mw 100 \
    --ppa-capacity-mw 150 \
    --ppa-technology "Solar + Wind Offshore" \
    --solar-fraction 0.4 \
    --start 2023-01-01 --end 2023-12-31 \
    --temporal-correlation hourly
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from rfnbo_calculations import calculate_renewable_share, calculate_rfnbo_compliance


def _find_latest_file(country_dir: Path, pattern: str) -> Path:
    files = sorted(country_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found for pattern '{pattern}' in '{country_dir}'."
        )
    return files[-1]


def _load_prices(file_path: Path) -> pd.DataFrame:
    prices_df = pd.read_csv(file_path)
    if "datetime" not in prices_df.columns:
        raise ValueError(f"Prices file is missing required column 'datetime': {file_path}")
    prices_df["datetime"] = pd.to_datetime(prices_df["datetime"], utc=True)
    return prices_df


def _load_generation(file_path: Path) -> pd.DataFrame:
    generation_df = pd.read_csv(file_path)
    required = {"timestamp", "generation_mw", "psr_type"}
    missing = required.difference(generation_df.columns)
    if missing:
        raise ValueError(
            f"Generation file is missing required columns {sorted(missing)}: {file_path}"
        )
    generation_df["timestamp"] = pd.to_datetime(generation_df["timestamp"], utc=True)
    if "resolution_minutes" not in generation_df.columns:
        generation_df["resolution_minutes"] = 60
    return generation_df


def _load_installed_capacity(file_path: Path) -> pd.DataFrame:
    capacity_df = pd.read_csv(file_path)
    if "timestamp" in capacity_df.columns:
        capacity_df["timestamp"] = pd.to_datetime(capacity_df["timestamp"], utc=True)
    return capacity_df


def _filter_date_range(
    prices_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not start and not end:
        return prices_df, generation_df

    start_ts = pd.to_datetime(start, utc=True) if start else prices_df["datetime"].min()
    end_ts = pd.to_datetime(end, utc=True) if end else prices_df["datetime"].max()

    prices_mask = (prices_df["datetime"] >= start_ts) & (prices_df["datetime"] <= end_ts)
    generation_mask = (generation_df["timestamp"] >= start_ts) & (generation_df["timestamp"] <= end_ts)

    return prices_df.loc[prices_mask].copy(), generation_df.loc[generation_mask].copy()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RFNBO compliance in hourly or monthly temporal correlation mode "
            "and produce GHG, RFNBO, and emission-factor plots."
        )
    )
    parser.add_argument("--country", default="Belgium", help="Country folder name under entsoe_data")
    parser.add_argument("--data-dir", default="entsoe_data", help="Base directory containing country subfolders")
    parser.add_argument(
        "--temporal-correlation",
        choices=["hourly", "monthly"],
        default="monthly",
        help="Compliance temporal correlation mode",
    )

    parser.add_argument("--prices-file", help="Optional explicit path to prices CSV")
    parser.add_argument("--generation-file", help="Optional explicit path to generation CSV")
    parser.add_argument("--capacity-file", help="Optional explicit path to installed capacity CSV")

    parser.add_argument("--start", help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", help="Optional end date filter (YYYY-MM-DD)")

    parser.add_argument("--electrolyser-mw", type=float, required=True, help="Electrolyser capacity in MW")
    parser.add_argument("--ppa-capacity-mw", type=float, required=True, help="PPA installed capacity in MW")
    parser.add_argument(
        "--ppa-technology",
        default="Solar",
        help="PPA technology (e.g., Solar, Wind Onshore, Wind Offshore, Solar + Wind Offshore)",
    )

    parser.add_argument(
        "--renewable-share",
        type=float,
        help="Optional grid renewable share [0..1]. If omitted, computed from generation data.",
    )
    parser.add_argument(
        "--solar-fraction",
        type=float,
        help="Optional solar fraction [0..1] for combined PPA technologies (solar share of combined PPA).",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store CSV and HTML plot outputs",
    )
    parser.add_argument(
        "--open-html",
        action="store_true",
        help="Open generated HTML plots in the default browser",
    )
    return parser


def _month_label(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["datetime"], utc=True).dt.strftime("%Y-%m")


def _add_threshold_line(fig: go.Figure, threshold: float, y_label: str) -> None:
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Threshold: {threshold:.1f} {y_label}",
        annotation_position="top left",
    )


def _write_and_open_html(fig: go.Figure, file_path: Path, open_html: bool) -> None:
    fig.write_html(file_path)
    if open_html:
        webbrowser.open(file_path.resolve().as_uri())


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    country_dir = Path(args.data_dir) / args.country
    if not country_dir.exists():
        raise FileNotFoundError(f"Country folder not found: {country_dir}")

    prices_file = Path(args.prices_file) if args.prices_file else _find_latest_file(country_dir, f"{args.country}_prices_*.csv")
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

    renewable_share = args.renewable_share
    if renewable_share is None:
        renewable_share = calculate_renewable_share(generation_df)

    # Compute solar / wind fractions from optional CLI input
    solar_fraction = None
    wind_fraction = None
    if getattr(args, "solar_fraction", None) is not None:
        if args.solar_fraction < 0.0 or args.solar_fraction > 1.0:
            raise ValueError("`--solar-fraction` must be between 0 and 1")
        solar_fraction = float(args.solar_fraction)
        wind_fraction = 1.0 - solar_fraction

    result_df = calculate_rfnbo_compliance(
        electrolyser_mw=args.electrolyser_mw,
        ppa_capacity_mw=args.ppa_capacity_mw,
        prices_df=prices_df,
        renewable_share=renewable_share,
        zone_name=args.country,
        temporal_correlation=args.temporal_correlation,
        use_price_threshold=True,
        ppa_technology=args.ppa_technology,
        generation_df=generation_df,
        installed_capacity_df=capacity_df,
        solar_fraction=solar_fraction,
        wind_fraction=wind_fraction,
    )

    if result_df.empty:
        raise ValueError("RFNBO calculation returned an empty dataframe.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = result_df.copy()
    result_df["datetime"] = pd.to_datetime(result_df["datetime"], utc=True)
    result_df["period_label"] = result_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    if args.temporal_correlation == "monthly":
        result_df["period_label"] = _month_label(result_df)

    result_df["ghg_emissions_t_co2eq"] = result_df["total_emissions_g_co2eq"] / 1_000_000
    result_df["rfnbo_pct"] = result_df["rfnbo_fraction"] * 100
    total_consumption_mwh = result_df["electrolyser_consumption_mwh"].sum()
    total_rfnbo_mwh = result_df["rfnbo_energy_mwh"].sum()
    overall_rfnbo_pct = (total_rfnbo_mwh / total_consumption_mwh * 100) if total_consumption_mwh > 0 else 0.0

    csv_out = output_dir / f"{args.country}_{args.temporal_correlation}_rfnbo_results.csv"
    result_df.to_csv(csv_out, index=False)

    ghg_title = f"{args.temporal_correlation.capitalize()} GHG Emissions - {args.country}"
    rfnbo_title = f"{args.temporal_correlation.capitalize()} RFNBO Percentage - {args.country}"
    ef_title = f"{args.temporal_correlation.capitalize()} Emission Factor - {args.country}"
    res_title = f"{args.temporal_correlation.capitalize()} Renewable Availability - {args.country}"

    ghg_fig = px.line(
        result_df,
        x="period_label",
        y="ghg_emissions_t_co2eq",
        markers=True,
        title=ghg_title,
        labels={"period_label": "Period", "ghg_emissions_t_co2eq": "GHG emissions [tCO2eq]"},
    )

    rfnbo_fig = px.line(
        result_df,
        x="period_label",
        y="rfnbo_pct",
        markers=True,
        title=rfnbo_title,
        labels={"period_label": "Period", "rfnbo_pct": "RFNBO [%]"},
    )
    rfnbo_fig.add_hline(
        y=overall_rfnbo_pct,
        line_dash="dash",
        line_color="darkgreen",
        annotation_text=f"Overall RFNBO: {overall_rfnbo_pct:.1f}%",
        annotation_position="top left",
    )

    ef_fig = px.line(
        result_df,
        x="period_label",
        y="emission_factor_mj",
        markers=True,
        title=ef_title,
        labels={"period_label": "Period", "emission_factor_mj": "Emission factor [g CO2eq/MJ]"},
    )
    _add_threshold_line(ef_fig, 28.2, "g CO2eq/MJ")

    res_fig = go.Figure()

    res_fig.add_trace(
        go.Scatter(
            x=result_df["period_label"],
            y=result_df["ppa_energy_mwh_raw"],
            mode="lines",
            name="Raw PPA energy",
            line=dict(color="gray", dash="dot", width=2),
        )
    )
    res_fig.add_trace(
        go.Scatter(
            x=result_df["period_label"],
            y=result_df["electrolyser_consumption_mwh"],
            mode="lines",
            name="Electrolyser consumption",
            line=dict(color="red", dash="dash", width=0.5),
            marker=dict(size=6),
        )
    )

    component_traces = [
        ("solar_energy_mwh", "Solar", "#FFA500"),
        ("wind_onshore_energy_mwh", "Wind Onshore", "#4169E1"),
        ("wind_offshore_energy_mwh", "Wind Offshore", "#1F77B4"),
    ]
    component_added = False

    for column_name, label, color in component_traces:
        if column_name in result_df.columns:
            component_added = True
            res_fig.add_trace(
                go.Scatter(
                    x=result_df["period_label"],
                    y=result_df[column_name],
                    mode="lines",
                    name=label,
                    stackgroup="res",
                    line=dict(width=0.5, color=color),
                )
            )

    if not component_added:
        res_fig.add_trace(
            go.Scatter(
                x=result_df["period_label"],
                y=result_df["ppa_energy_mwh"],
                mode="lines",
                name="Capped PPA energy",
                stackgroup="res",
                line=dict(width=0.5, color="steelblue"),
            )
        )

    res_fig.add_trace(
        go.Scatter(
            x=result_df["period_label"],
            y=result_df["grid_energy_low_price_mwh"],
            mode="lines",
            name="Price exemption energy no GHG",
            stackgroup="res",
            line=dict(width=0.5, color="darkorange"),
        )
    )
    res_fig.add_trace(
        go.Scatter(
            x=result_df["period_label"],
            y=result_df["e_grid_res_mwh"] - result_df["grid_energy_low_price_mwh"],
            mode="lines",
            name="Grid renewable share energy with avg GHG grid",
            stackgroup="res",
            line=dict(width=0.5, color="seagreen"),
        )
    )
    # res_fig.add_trace(
    #     go.Scatter(
    #         x=result_df["period_label"],
    #         y=result_df["rfnbo_energy_mwh"],
    #         mode="lines+markers",
    #         name="Total RFNBO energy",
    #         line=dict(color="black", width=2, dash="dash"),
    #     )
    #)
    res_fig.add_trace(
        go.Scatter(
            x=result_df["period_label"],
            y=result_df["e_total_res_mwh"],
            mode="lines+markers",
            name="Total available RES no GHG",
            line=dict(color="dimgray", width=2),
        )
    )
    res_fig.update_layout(
        title=res_title,
        xaxis_title="Period",
        yaxis_title="Energy [MWh]",
        legend_title="Component",
        hovermode="x unified",
    )

    mode = args.temporal_correlation
    ghg_html = output_dir / f"{args.country}_{mode}_ghg_emissions.html"
    rfnbo_html = output_dir / f"{args.country}_{mode}_rfnbo_pct.html"
    ef_html = output_dir / f"{args.country}_{mode}_emission_factor.html"
    res_html = output_dir / f"{args.country}_{mode}_renewable_availability.html"

    _write_and_open_html(ghg_fig, ghg_html, args.open_html)
    _write_and_open_html(rfnbo_fig, rfnbo_html, args.open_html)
    _write_and_open_html(ef_fig, ef_html, args.open_html)
    _write_and_open_html(res_fig, res_html, args.open_html)

    print("RFNBO orchestration completed.")
    print(f"Country: {args.country}")
    print(f"Temporal correlation: {args.temporal_correlation}")
    print(f"Renewable share used: {renewable_share:.4f}")
    print(f"Results CSV: {csv_out}")
    print(f"GHG plot: {ghg_html}")
    print(f"RFNBO plot: {rfnbo_html}")
    print(f"Emission factor plot: {ef_html}")
    print(f"Renewable availability plot: {res_html}")


if __name__ == "__main__":
    main()
