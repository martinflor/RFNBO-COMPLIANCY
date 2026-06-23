"""Standalone PPA sourcing cost sweep with capacity factors.

This script estimates the sourcing cost of a day-ahead market (DAM) referenced PPA.
Unlike standard MW-based sizing, this script sizes PPA capacities using technology-specific 
capacity factors. A ratio of 1.0 means the PPA is sized to match the electrolyser's 
annual *energy* demand (MWh), rather than its installed capacity (MW).

### Cost Calculation Logic (Financial PPA Settlement)
The script models the PPA purely financially. All electrolyser consumption is assumed 
to be purchased from the day-ahead market, and the PPA premium is settled separately:

    Baseline Grid Cost = Total Electrolyser Consumption * DAM price
    Incremental PPA Cost = Total Raw PPA Generation * Margin
    Total Cost = Baseline Grid Cost + Incremental PPA Cost

Note on "Residual Grid Energy": 
While the script calculates a `grid_energy_mwh` metric, this is strictly used for volume 
reporting in the summary CSV (to show how much physical energy was not matched by the PPA). 
It is NOT used in the cost calculations.

### Expected Inputs
* **Market Data:** ENTSO-E generation, prices, and installed capacity CSVs.
* **Electrolyser Capacity:** Required baseline load defined via `--electrolyser-mw`.
* **Capacity Factors:** Used to size the physical PPA. Can be statically defined via CLI 
  (e.g., `--cf-solar`) or calculated dynamically from the ENTSO-E data (`--use-entsoe-cfs`).

### Expected Outputs
1.  **Detailed CSV (`*_ppa_sourcing_cost_cf_detail.csv`):** Row-by-row temporal calculations.
2.  **Summary CSV (`*_ppa_sourcing_cost_cf_summary.csv`):** Compact aggregations showing costs, effective premiums, and overall RFNBO compliance.
3.  **Interactive Plots (`*_premium_plot.html` / `*_absolute_plot.html`):** HTML scatter plots generated for *each* swept margin.

### Example CLI Usage

**1. Basic Run (Using Default Fixed Capacity Factors):**
Runs the sweep for a 100 MW electrolyser using the hardcoded default capacity factors (Solar: 11%, Onshore: 22%, Offshore: 38%). Opens the effective premium plots automatically.
> python ppa_sweep_cf.py --country Belgium --electrolyser-mw 100 --open-html

**2. Dynamic Capacity Factors & Absolute Cost Plots:**
Calculates the capacity factors dynamically based on historical ENTSO-E generation data, and generates the optional absolute cost plots in addition to the premium plots.
> python ppa_sweep_cf.py \
    --country Germany \
    --electrolyser-mw 50 \
    --use-entsoe-cfs \
    --include-cost-plots \
    --open-html

**3. Targeted Scenario with Custom Static Capacity Factors:**
Sweeps specific energy ratios and margins for Solar and Offshore wind, overriding the default capacity factors manually.
> python ppa_sweep_cf.py \
    --country France \
    --electrolyser-mw 200 \
    --technologies "Solar" "Wind Offshore" \
    --ratios 0.5 1.0 1.5 \
    --margins 2.0 4.0 \
    --cf-solar 0.15 \
    --cf-offshore 0.42
"""

from __future__ import annotations

import argparse
import logging
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from RFNBO_orchestrator import (
    _filter_date_range,
    _find_latest_file,
    _load_generation,
    _load_installed_capacity,
    _load_prices,
)
from rfnbo_calculations import (
    calculate_ppa_production_from_generation_data,
    calculate_renewable_share,
    calculate_rfnbo_compliance,
    integrate_power_to_energy,
)


logger = logging.getLogger(__name__)

DEFAULT_RATIOS = [round(index * 0.25, 2) for index in range(0, 9)]
DEFAULT_MARGINS = [1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_TECHNOLOGIES = [
    "Solar",
    "Wind Offshore",
    "Wind Onshore",
    "Solar + Wind Offshore",
    "Solar + Wind Onshore",
]

TECHNOLOGY_LABELS = {
    "Solar": "Solar",
    "Wind Onshore": "Wind Onshore",
    "Wind Offshore": "Wind Offshore",
    "Solar + Wind Offshore": "Solar + Wind Offshore",
    "Solar + Wind Onshore": "Solar + Wind Onshore",
}


def _tech_slug(technology: str) -> str:
    return technology.lower().replace(" + ", "_plus_").replace(" ", "_")


def _psr_types_for_tech(tech: str) -> list[str]:
    t = tech.lower()
    if "solar" in t:
        return ["B16"]
    if "offshore" in t:
        return ["B18"]
    if "onshore" in t:
        return ["B19"]
    if "wind" in t:
        return ["B18", "B19"]
    return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate PPA sourcing cost using day-ahead market pricing, but size the PPA "
            "with capacity factors so the ratio represents energy rather than installed capacity."
        )
    )
    parser.add_argument("--country", default="Belgium", help="Country folder name under entsoe_data")
    parser.add_argument("--data-dir", default="entsoe_data", help="Base directory containing country subfolders")
    parser.add_argument(
        "--temporal-correlation",
        choices=["hourly", "monthly"],
        default="hourly",
        help="Aggregation mode for the sweep output",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=DEFAULT_RATIOS,
        help="Energy-to-consumption ratios to sweep (default: 0.0 to 2.0 in 0.25 steps)",
    )
    parser.add_argument(
        "--margins",
        nargs="+",
        type=float,
        default=DEFAULT_MARGINS,
        help="Extra DAM premiums in EUR/MWh to sweep (default: 1 to 5 EUR/MWh)",
    )
    parser.add_argument(
        "--technologies",
        nargs="+",
        default=DEFAULT_TECHNOLOGIES,
        choices=list(TECHNOLOGY_LABELS),
        help="PPA technologies to model (default: Solar, wind, and 50/50 combined cases)",
    )
    parser.add_argument(
        "--solar-fraction",
        type=float,
        help="Optional solar share for combined PPA technologies (solar fraction of combined PPA)",
    )
    # --- CAPACITY FACTOR CLI ARGUMENTS ---
    parser.add_argument("--cf-solar", type=float, default=0.11, help="Capacity factor for Solar (default: 0.11)")
    parser.add_argument("--cf-onshore", type=float, default=0.22, help="Capacity factor for Onshore Wind (default: 0.22)")
    parser.add_argument("--cf-offshore", type=float, default=0.38, help="Capacity factor for Offshore Wind (default: 0.38)")
    parser.add_argument(
        "--use-entsoe-cfs",
        action="store_true",
        help="Calculate CFs dynamically from ENTSO-E data instead of using the fixed CLI CFs.",
    )
    # --- NEW PLOT MANAGEMENT CLI ARGUMENTS ---
    parser.add_argument(
        "--include-cost-plots",
        action="store_true",
        help="Generate and open absolute cost plots in addition to the premium plots.",
    )
    # -----------------------------------------
    parser.add_argument("--prices-file", help="Optional explicit path to prices CSV")
    parser.add_argument("--generation-file", help="Optional explicit path to generation CSV")
    parser.add_argument("--capacity-file", help="Optional explicit path to installed capacity CSV")
    parser.add_argument("--start", help="Optional start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", help="Optional end date filter (YYYY-MM-DD)")
    parser.add_argument("--electrolyser-mw", type=float, required=True, help="Electrolyser capacity in MW")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store CSV and HTML outputs",
    )
    parser.add_argument(
        "--output-file",
        help="Optional detailed CSV path. Defaults to outputs/<country>_<mode>_ppa_sourcing_cost_cf_detail.csv",
    )
    parser.add_argument(
        "--summary-file",
        help="Optional summary CSV path. Defaults to outputs/<country>_<mode>_ppa_sourcing_cost_cf_summary.csv",
    )
    parser.add_argument(
        "--open-html",
        action="store_true",
        help="Open the generated HTML plots in the default browser",
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

    if args.start is not None or args.end is not None:
        start_ts = pd.to_datetime(args.start)
        if args.start and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        end_ts = pd.to_datetime(args.end)
        if args.end and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        if start_ts is not None:
            capacity_df = capacity_df[capacity_df["timestamp"] >= start_ts]
        if end_ts is not None:
            capacity_df = capacity_df[capacity_df["timestamp"] <= end_ts]

    if prices_df.empty:
        raise ValueError("No prices data left after applying date filters.")
    if generation_df.empty:
        raise ValueError("No generation data left after applying date filters.")

    return prices_df, generation_df, capacity_df


def _capacity_factor_for_techs(
    generation_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    tech_keys: list[str],
    default_cfs: dict[str, float],
) -> dict[str, float]:
    cfs = default_cfs.copy()

    for tech in tech_keys:
        psr_types = _psr_types_for_tech(tech)
        if not psr_types:
            continue

        gen_f = generation_df[generation_df["psr_type"].isin(psr_types)].copy()
        cap_f = capacity_df[capacity_df["psr_type"].isin(psr_types)].copy()
        if gen_f.empty:
            continue

        if "resolution_minutes" in gen_f.columns:
            total_gen_mwh = (gen_f["generation_mw"] * (gen_f["resolution_minutes"] / 60.0)).sum()
            period_hours = gen_f["resolution_minutes"].sum() / 60.0
        else:
            total_gen_mwh = gen_f["generation_mw"].sum()
            period_hours = len(gen_f)

        avg_inst_mw = cap_f["installed_capacity_mw"].mean() if not cap_f.empty else 0.0
        if avg_inst_mw > 0 and period_hours > 0:
            if "solar" in tech.lower():
                cfs["solar"] = float(total_gen_mwh) / (float(avg_inst_mw) * float(period_hours))
            elif "offshore" in tech.lower():
                cfs["offshore"] = float(total_gen_mwh) / (float(avg_inst_mw) * float(period_hours))
            elif "onshore" in tech.lower():
                cfs["onshore"] = float(total_gen_mwh) / (float(avg_inst_mw) * float(period_hours))

    return cfs


def _build_effective_premium_plot(df: pd.DataFrame, country: str, mode: str, margin: float, output_path: Path) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_df["rfnbo_marker_size"] = plot_df["overall_rfnbo_pct"].fillna(4.0).clip(lower=4.0)

    fig = px.scatter(
        plot_df,
        x="ppa_to_electrolyser_ratio",
        y="effective_rfnbo_premium_eur_mwh",
        color="overall_rfnbo_pct",
        symbol="technology",
        category_orders={"technology": list(TECHNOLOGY_LABELS)},
        symbol_sequence=["circle", "square", "diamond", "x", "cross"],
        size="rfnbo_marker_size",
        size_max=22,
        color_continuous_scale="Viridis",
        hover_data={
            "technology": True,
            "extra_margin_eur_mwh": ":.1f",
            "effective_rfnbo_premium_eur_mwh": ":.2f",
            "incremental_cost_eur": ":.2f",
            "total_raw_ppa_generation_mwh": ":.2f",
            "total_rfnbo_mwh": ":.2f",
            "total_ppa_energy_mwh": ":.2f",
            "overall_rfnbo_pct": ":.1f",
            "ppa_capacity_mw": ":.1f",
        },
        title=f"Effective RFNBO Premium vs Energy Ratio - {country} ({mode}) - Margin: {margin:g} €/MWh",
        labels={
            "ppa_to_electrolyser_ratio": "Production to consumption ratio",
            "effective_rfnbo_premium_eur_mwh": "Effective RFNBO Premium [€ / MWh of RFNBO]",
            "extra_margin_eur_mwh": "Extra margin [€/MWh]",
            "overall_rfnbo_pct": "RFNBO [%]",
        },
    )

    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.35)")))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=170, t=70, b=100),
        legend=dict(
            title_text="Technology",
            x=1.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        coloraxis_colorbar=dict(
            title="RFNBO [%]",
            x=1.18,
            thickness=18,
        ),
    )
    fig.update_xaxes(tickmode="linear", dtick=0.25)
    fig.update_yaxes(ticksuffix=" €")
    fig.write_html(output_path)


def _build_absolute_cost_plot(df: pd.DataFrame, country: str, mode: str, margin: float, output_path: Path) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_df["rfnbo_marker_size"] = plot_df["overall_rfnbo_pct"].fillna(4.0).clip(lower=4.0)

    fig = px.scatter(
        plot_df,
        x="ppa_to_electrolyser_ratio",
        y="absolute_cost_eur_mwh",
        color="overall_rfnbo_pct",
        symbol="technology",
        category_orders={"technology": list(TECHNOLOGY_LABELS)},
        symbol_sequence=["circle", "square", "diamond", "x", "cross"],
        size="rfnbo_marker_size",
        size_max=22,
        color_continuous_scale="Viridis",
        hover_data={
            "technology": True,
            "extra_margin_eur_mwh": ":.1f",
            "total_raw_ppa_generation_mwh": ":.2f",
            "total_ppa_energy_mwh": ":.2f",
            "realized_ppa_ratio": ":.3f",
            "incremental_cost_eur": ":.2f",
            "total_cost_eur": ":.2f",
            "overall_rfnbo_pct": ":.1f",
            "ppa_share_pct": ":.1f",
            "baseline_cost_eur": ":.2f",
            "incremental_cost_eur_mwh": ":.2f",
            "absolute_cost_eur_mwh": ":.2f",
            "ppa_capacity_mw": ":.1f",
            "electrolyser_mw": ":.1f",
        },
        title=f"PPA Absolute Aggregated Cost vs Energy Ratio - {country} ({mode}) - Margin: {margin:g} €/MWh",
        labels={
            "ppa_to_electrolyser_ratio": "Production to consumption ratio",
            "absolute_cost_eur_mwh": "Aggregated cost [€/MWh]",
            "extra_margin_eur_mwh": "Extra margin [€/MWh]",
            "overall_rfnbo_pct": "RFNBO [%]",
        },
    )

    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.35)")))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=170, t=70, b=100),
        legend=dict(
            title_text="Technology",
            x=1.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        coloraxis_colorbar=dict(
            title="RFNBO [%]",
            x=1.18,
            thickness=18,
        ),
    )
    fig.update_xaxes(tickmode="linear", dtick=0.25)
    fig.update_yaxes(ticksuffix=" €/MWh")
    fig.write_html(output_path)


def _scenario_capacity(
    electrolyser_mw: float,
    ratio: float,
    technology: str,
    cfs: dict[str, float],
    solar_fraction: float | None,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    technology_lower = technology.lower()
    if solar_fraction is None and "+" in technology:
        solar_fraction = 0.5

    if technology_lower == "solar":
        capacity_factor = cfs.get("solar", 0.11)
        ppa_capacity_mw = electrolyser_mw * ratio / capacity_factor
        return ppa_capacity_mw, None, None, capacity_factor, None

    if technology_lower == "wind offshore":
        capacity_factor = cfs.get("offshore", 0.38)
        ppa_capacity_mw = electrolyser_mw * ratio / capacity_factor
        return ppa_capacity_mw, None, None, capacity_factor, None

    if technology_lower == "wind onshore":
        capacity_factor = cfs.get("onshore", 0.22)
        ppa_capacity_mw = electrolyser_mw * ratio / capacity_factor
        return ppa_capacity_mw, None, None, capacity_factor, None

    if "+" in technology_lower:
        wind_key = "offshore" if "offshore" in technology_lower else "onshore"
        solar_share = 0.5 if solar_fraction is None else solar_fraction
        wind_share = 1.0 - solar_share
        solar_cf = cfs.get("solar", 0.11)
        wind_cf = cfs.get(wind_key, 0.38 if wind_key == "offshore" else 0.22)

        solar_capacity_mw = electrolyser_mw * ratio * solar_share / solar_cf
        wind_capacity_mw = electrolyser_mw * ratio * wind_share / wind_cf
        ppa_capacity_mw = solar_capacity_mw + wind_capacity_mw
        if ppa_capacity_mw <= 0:
            return 0.0, solar_share, wind_share, solar_cf, wind_cf

        solar_fraction_capacity = solar_capacity_mw / ppa_capacity_mw
        wind_fraction_capacity = wind_capacity_mw / ppa_capacity_mw
        return ppa_capacity_mw, solar_fraction_capacity, wind_fraction_capacity, solar_cf, wind_cf

    raise ValueError(f"Unsupported technology: {technology}")


def _scenario_rows(
    prices_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    country: str,
    electrolyser_mw: float,
    ratio: float,
    technology: str,
    mode: str,
    solar_fraction: float | None,
    cfs: dict[str, float],
    renewable_share: float,
) -> pd.DataFrame:
    ppa_capacity_mw, solar_fraction_capacity, wind_fraction_capacity, solar_cf, wind_cf = _scenario_capacity(
        electrolyser_mw=electrolyser_mw,
        ratio=ratio,
        technology=technology,
        cfs=cfs,
        solar_fraction=solar_fraction,
    )

    if ppa_capacity_mw <= 0:
        return pd.DataFrame()

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
        solar_fraction=solar_fraction_capacity,
        wind_fraction=wind_fraction_capacity,
    )

    if compliance_df.empty:
        return pd.DataFrame()

    compliance_df["period_label"] = pd.to_datetime(compliance_df["datetime"], utc=True).dt.strftime(
        "%Y-%m" if mode == "monthly" else "%Y-%m-%d %H:%M"
    )

    compliance_df["country"] = country
    compliance_df["technology"] = technology
    compliance_df["ppa_to_electrolyser_ratio"] = ratio
    compliance_df["ppa_capacity_mw"] = ppa_capacity_mw
    compliance_df["electrolyser_mw"] = electrolyser_mw
    compliance_df["temporal_correlation"] = mode
    compliance_df["solar_capacity_factor"] = solar_cf if solar_cf is not None else pd.NA
    compliance_df["wind_capacity_factor"] = wind_cf if wind_cf is not None else pd.NA
    compliance_df["realized_ppa_ratio"] = compliance_df["ppa_energy_mwh"] / compliance_df["electrolyser_consumption_mwh"].where(
        compliance_df["electrolyser_consumption_mwh"] > 0,
        pd.NA,
    )
    return compliance_df


def _apply_cost_columns(detail_df: pd.DataFrame, margin_eur_mwh: float) -> pd.DataFrame:
    df = detail_df.copy()
    df["extra_margin_eur_mwh"] = margin_eur_mwh
    
    # 1. Baseline: Check if already calculated upstream (Monthly mode compatibility)
    if "baseline_cost_eur" not in df.columns:
        df["baseline_cost_eur"] = df["electrolyser_consumption_mwh"] * df["price_eur_mwh"]
    
    # 2. Incremental: TOTAL PPA volume at the Margin price (Pay-as-produced)
    df["incremental_cost_eur"] = df["ppa_energy_mwh_raw"] * margin_eur_mwh 
    
    # 3. Total Cost
    df["total_cost_eur"] = df["baseline_cost_eur"] + df["incremental_cost_eur"]
    
    # Metrics
    df["absolute_cost_eur_mwh"] = df["total_cost_eur"] / df["electrolyser_consumption_mwh"].where(
        df["electrolyser_consumption_mwh"] > 0,
        pd.NA,
    )
    df["incremental_cost_eur_mwh"] = df["incremental_cost_eur"] / df["electrolyser_consumption_mwh"].where(
        df["electrolyser_consumption_mwh"] > 0,
        pd.NA,
    )
    df["ppa_share_pct"] = (
        df["ppa_energy_mwh"] / df["electrolyser_consumption_mwh"].where(df["electrolyser_consumption_mwh"] > 0, pd.NA)
    ) * 100
    
    return df


def _summarise_scenario(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df

    summary_rows: list[dict[str, float | str]] = []
    group_columns = [
        "country",
        "technology",
        "ppa_to_electrolyser_ratio",
        "extra_margin_eur_mwh",
        "ppa_capacity_mw",
        "electrolyser_mw",
        "temporal_correlation",
    ]

    for group_values, group_df in detail_df.groupby(group_columns, dropna=False):
        total_consumption_mwh = group_df["electrolyser_consumption_mwh"].sum()
        total_ppa_mwh = group_df["ppa_energy_mwh"].sum()
        total_unclipped_ppa_mwh = group_df["ppa_energy_mwh_raw"].sum()
        total_grid_mwh = group_df["grid_energy_mwh"].sum()
        
        total_cost_eur = group_df["total_cost_eur"].sum()
        baseline_cost_eur = group_df["baseline_cost_eur"].sum()
        incremental_cost_eur = group_df["incremental_cost_eur"].sum()
        
        absolute_cost_eur_mwh = total_cost_eur / total_consumption_mwh if total_consumption_mwh > 0 else 0.0
        incremental_cost_eur_mwh = incremental_cost_eur / total_consumption_mwh if total_consumption_mwh > 0 else 0.0
        ppa_share_pct = (total_ppa_mwh / total_consumption_mwh * 100) if total_consumption_mwh > 0 else 0.0
        
        total_rfnbo_mwh = group_df["rfnbo_energy_mwh"].sum(min_count=1)
        overall_rfnbo_fraction = (
            total_rfnbo_mwh / total_consumption_mwh
            if total_consumption_mwh > 0 and pd.notna(total_rfnbo_mwh)
            else pd.NA
        )

        effective_rfnbo_premium = (
            incremental_cost_eur / total_rfnbo_mwh 
            if pd.notna(total_rfnbo_mwh) and total_rfnbo_mwh > 0 
            else pd.NA
        )

        summary_rows.append(
            {
                "country": group_values[0],
                "technology": group_values[1],
                "ppa_to_electrolyser_ratio": group_values[2],
                "extra_margin_eur_mwh": group_values[3],
                "ppa_capacity_mw": group_values[4],
                "electrolyser_mw": group_values[5],
                "temporal_correlation": group_values[6],
                "total_consumption_mwh": total_consumption_mwh,
                "total_raw_ppa_generation_mwh": total_unclipped_ppa_mwh,
                "total_ppa_energy_mwh": total_ppa_mwh,
                "realized_ppa_ratio": total_ppa_mwh / total_consumption_mwh if total_consumption_mwh > 0 else 0.0,
                "total_grid_energy_mwh": total_grid_mwh,
                "baseline_cost_eur": baseline_cost_eur,
                "incremental_cost_eur": incremental_cost_eur,
                "total_cost_eur": total_cost_eur,
                "absolute_cost_eur_mwh": absolute_cost_eur_mwh,
                "incremental_cost_eur_mwh": incremental_cost_eur_mwh,
                "effective_rfnbo_premium_eur_mwh": effective_rfnbo_premium,
                "ppa_share_pct": ppa_share_pct,
                "total_rfnbo_mwh": total_rfnbo_mwh,
                "overall_rfnbo_fraction": overall_rfnbo_fraction,
                "overall_rfnbo_pct": overall_rfnbo_fraction * 100 if pd.notna(overall_rfnbo_fraction) else pd.NA,
                "solar_capacity_factor": group_df["solar_capacity_factor"].iloc[0],
                "wind_capacity_factor": group_df["wind_capacity_factor"].iloc[0],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values(["extra_margin_eur_mwh", "ppa_to_electrolyser_ratio"])


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.solar_fraction is not None and not (0.0 <= args.solar_fraction <= 1.0):
        raise ValueError("--solar-fraction must be between 0 and 1")

    prices_df, generation_df, capacity_df = _load_input_data(args)
    renewable_share = calculate_renewable_share(generation_df)

    if args.use_entsoe_cfs:
        cfs = _capacity_factor_for_techs(
            generation_df=generation_df,
            capacity_df=capacity_df,
            tech_keys=["solar", "offshore", "onshore"],
            default_cfs={"solar": args.cf_solar, "offshore": args.cf_offshore, "onshore": args.cf_onshore}
        )
        print("Capacity factors calculated dynamically from ENTSO-E data:")
    else:
        cfs = {
            "solar": args.cf_solar,
            "offshore": args.cf_offshore,
            "onshore": args.cf_onshore,
        }
        print("Capacity factors used (CLI/Defaults):")

    print(f"  - solar: {cfs['solar']:.4f}")
    print(f"  - offshore: {cfs['offshore']:.4f}")
    print(f"  - onshore: {cfs['onshore']:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for technology in args.technologies:
        solar_fraction = args.solar_fraction
        if solar_fraction is None and "+" in technology:
            solar_fraction = 0.5

        for ratio in args.ratios:
            base_detail_df = _scenario_rows(
                prices_df=prices_df,
                generation_df=generation_df,
                capacity_df=capacity_df,
                country=args.country,
                electrolyser_mw=args.electrolyser_mw,
                ratio=ratio,
                technology=technology,
                mode=args.temporal_correlation,
                solar_fraction=solar_fraction,
                cfs=cfs,
                renewable_share=renewable_share,
            )

            if base_detail_df.empty:
                continue

            for margin in args.margins:
                print(f"Running {technology} ratio {ratio:g} margin {margin:g} €/MWh...")
                detail_df = _apply_cost_columns(base_detail_df, margin)
                detail_frames.append(detail_df)
                summary_frames.append(_summarise_scenario(detail_df))

    if not detail_frames:
        raise ValueError("PPA sourcing cost sweep produced no rows.")

    detail_out = Path(args.output_file) if args.output_file else output_dir / f"{args.country}_{args.temporal_correlation}_ppa_sourcing_cost_cf_detail.csv"
    summary_out = Path(args.summary_file) if args.summary_file else output_dir / f"{args.country}_{args.temporal_correlation}_ppa_sourcing_cost_cf_summary.csv"

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    detail_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(detail_out)
    print(summary_out)

    generated_plots = []
    for margin in summary_df["extra_margin_eur_mwh"].unique():
        margin_df = summary_df[summary_df["extra_margin_eur_mwh"] == margin]
        
        # 1. ALWAYS Generate and track Effective Premium Plot
        premium_plot_out = output_dir / f"{args.country}_{args.temporal_correlation}_ppa_sourcing_cost_cf_summary_margin_{margin:g}_premium_plot.html"
        _build_effective_premium_plot(margin_df, args.country, args.temporal_correlation, margin, premium_plot_out)
        generated_plots.append(premium_plot_out)
        print(premium_plot_out)

        # 2. OPTIONALLY Generate and track Absolute Cost Plot based on CLI flag
        if args.include_cost_plots:
            absolute_plot_out = output_dir / f"{args.country}_{args.temporal_correlation}_ppa_sourcing_cost_cf_summary_margin_{margin:g}_absolute_plot.html"
            _build_absolute_cost_plot(margin_df, args.country, args.temporal_correlation, margin, absolute_plot_out)
            generated_plots.append(absolute_plot_out)
            print(absolute_plot_out)

    if args.open_html:
        for plot_path in generated_plots:
            webbrowser.open(plot_path.resolve().as_uri())


if __name__ == "__main__":
    main()