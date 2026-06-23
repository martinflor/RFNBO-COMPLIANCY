"""Standalone RFNBO Hybrid Energy-Ratio Sweep & Temporal Simulation CLI.

This script evaluates how the internal energy mix of a hybrid Power Purchase 
Agreement (PPA) impacts overall Renewable Fuels of Non-Biological Origin (RFNBO) 
compliance. It generates detailed datasets and interactive plots for Solar + Offshore 
and Solar + Onshore combinations.

It also includes a standalone temporal simulation mode to export scaled PPA generation 
timeseries without running the full compliance engine.

### Sizing Logic (Energy Split via Capacity Factors)
Unlike MW-based models, this script divides the **Annual Energy Volume (MWh)** between 
solar and wind. It relies on Capacity Factors (CFs) to determine how many physical MW 
need to be installed to hit the requested energy fraction.
* Solar MW = (Electrolyser MW * Base Ratio * Solar Fraction) / Solar CF
* Wind MW = (Electrolyser MW * Base Ratio * Wind Fraction) / Wind CF
* *Note:* Because solar has a lower CF than wind, achieving a 50% solar energy fraction 
  requires installing significantly more Solar MW than Wind MW. CFs can be calculated 
  dynamically from ENTSO-E data using `--use-data`, otherwise defaults are used.

### Expected Inputs
* **Market Data:** ENTSO-E generation, prices, and installed capacity CSVs.
* **Electrolyser Capacity:** Required baseline load defined via `--electrolyser-mw`.
* **Run Mode:** * `--sweep`: Runs the multi-ratio optimization sweep.
  * `--temporal`: Runs a single sizing calculation and exports the MW generation timeseries.

### Expected Outputs
1.  **Sweep Mode (`--sweep`):**
    * **CSV:** Granular dataset containing RFNBO percentages for every tested solar energy split.
    * **Plot (`*.html`):** Interactive line chart displaying the Ratio of Solar Production 
        (X-axis) against the overall RFNBO % (Y-axis) for different overall PPA sizes.
2.  **Temporal Mode (`--temporal`):**
    * **CSV:** A timestep-by-timestep output of the physical generation (MW) of the sized assets.

### Example CLI Usage

**1. Energy-Based Sweep (Using Dynamic Capacity Factors):**
Runs the 100-point solar energy sweep (from 0% to 100% solar energy) for base PPA-to-electrolyser 
ratios of 0.75, 1.0, and 1.25. Dynamically calculates CFs from the historical data.
> python rfnbo_hybrid_energy_sweep.py \
    --electrolyser-mw 50 \
    --sweep \
    --use-data \
    --open-html

**2. Single Run Temporal Export (No Sweep):**
Sizing a PPA to perfectly match the electrolyser's energy demand (ratio 1.0), with an energy 
mix of 60% Onshore Wind and 40% Solar. Exports the hourly generation curve.
> python rfnbo_hybrid_energy_sweep.py \
    --electrolyser-mw 100 \
    --base-ppa-ratio 1.0 \
    --tech-mix "onshore:0.6,solar:0.4" \
    --temporal \
    --use-data \
    --out-csv outputs/scaled_ppa_generation.csv

**3. Targeted Date Range Sweep with Defaults:**
Runs the sweep for a specific historical window using the hardcoded default capacity factors 
(since `--use-data` is omitted).
> python rfnbo_hybrid_energy_sweep.py \
    --electrolyser-mw 20 \
    --sweep \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --open-html
"""


import argparse
import json
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from rfnbo_calculations import (
    PPA_TECHNOLOGY_PSR_TYPES,
    aggregate_to_monthly,
    calculate_rfnbo_compliance,
    calculate_renewable_share,
    get_grid_emission_factor,
)

DEFAULT_CFS = {
    "onshore": 0.25,
    "offshore": 0.30,
    "solar": 0.15,
}

# Default paths / settings (put more in-code so CLI is shorter)
DEFAULT_GEN_CSV = Path("entsoe_data/Belgium/Belgium_generation_20200101_20251220.csv")
DEFAULT_INST_CSV = Path("entsoe_data/Belgium/Belgium_installed_capacity_20200101_20251220.csv")
DEFAULT_PRICES_CSV = Path("entsoe_data/Belgium/Belgium_prices_20200101_20251220.csv")
DEFAULT_BASE_PPA_RATIOS = "0.75,1,1.25"
DEFAULT_SWEEP_TYPES = "solar-offshore,solar-onshore"
DEFAULT_SOLAR_RATIO_STEPS = 100

TECHNOLOGY_SCENARIOS = [
    ("Solar + Wind Offshore", "offshore"),
    ("Solar + Wind Onshore", "onshore"),
]


def parse_tech_mix(s: str):
    if not s:
        return {"offshore": 1.0}
    try:
        d = json.loads(s)
        return {k.lower(): float(v) for k, v in d.items()}
    except Exception:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        d = {}
        for p in parts:
            if ":" in p or "=" in p:
                k, v = p.replace("=",":").split(":", 1)
                d[k.lower().strip()] = float(v)
            else:
                d[p.lower()] = 1.0
        ssum = sum(d.values())
        if ssum == 0:
            return d
        return {k: v / ssum for k, v in d.items()}


def compute_required_capacities(electrolyser_mw, base_ratio, tech_mix, cfs):
    capacities = {}
    for tech, frac in tech_mix.items():
        cf = cfs.get(tech, None)
        if cf is None or cf <= 0:
            raise ValueError(f"No capacity factor for tech '{tech}'")
        capacities[tech] = electrolyser_mw * base_ratio * frac / cf
    return capacities


def _selected_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["hourly", "monthly"]
    return [mode]


def _solar_ratio_values(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    return [round(index / (count - 1), 6) for index in range(count)]


def _plot_color(base_ratio: float) -> str:
    palette = {
        0.75: "#9DC3E6",
        1.0: "#5B9BD5",
        1.25: "#2F5597",
    }
    return palette.get(float(base_ratio), "#1f77b4")


def _build_capacity_factor_plot(df: pd.DataFrame, country: str, mode: str, technology: str, output_path: Path) -> None:
    if df.empty:
        return

    fig = go.Figure()
    for base_ratio in sorted(df["base_ratio"].unique()):
        ratio_df = df[df["base_ratio"] == base_ratio].sort_values("solar_ratio")
        if ratio_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=ratio_df["solar_ratio"],
                y=ratio_df["overall_rfnbo_pct"],
                mode="lines",
                name=f"Prod. to cons. ratio = {base_ratio:g}",
                line=dict(color=_plot_color(float(base_ratio)), width=3),
                hovertemplate=(
                    "Solar ratio=%{x:.3f}<br>RFNBO=%{y:.1f}%"
                    f"<extra>Prod. to cons. ratio = {base_ratio:g}</extra>"
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


def _build_output_basename(country: str, mode: str, technology_slug: str) -> str:
    return f"{country}_{mode}_{technology_slug}_capacity_factor_sweep"


def _build_compliance_point(
    prices_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    country: str,
    electrolyser_mw: float,
    base_ratio: float,
    solar_ratio: float,
    technology: str,
    technology_slug: str,
    mode: str,
    renewable_share: float,
    cfs: dict,
) -> pd.DataFrame:
    wind_ratio = 1.0 - solar_ratio
    solar_key = "solar"
    wind_key = technology_slug

    solar_capacity_mw = electrolyser_mw * base_ratio * solar_ratio / cfs[solar_key]
    wind_capacity_mw = electrolyser_mw * base_ratio * wind_ratio / cfs[wind_key]
    ppa_capacity_mw = solar_capacity_mw + wind_capacity_mw

    if ppa_capacity_mw <= 0:
        return pd.DataFrame()

    solar_fraction = solar_capacity_mw / ppa_capacity_mw
    wind_fraction = wind_capacity_mw / ppa_capacity_mw

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
        return pd.DataFrame()

    if mode == "monthly":
        compliance_df = aggregate_to_monthly(compliance_df, get_grid_emission_factor(country))

    total_consumption_mwh = compliance_df["electrolyser_consumption_mwh"].sum()
    total_rfnbo_mwh = compliance_df["rfnbo_energy_mwh"].sum()
    overall_rfnbo_fraction = (total_rfnbo_mwh / total_consumption_mwh) if total_consumption_mwh > 0 else 0.0

    return pd.DataFrame([
        {
            "country": country,
            "technology": technology,
            "technology_slug": technology_slug,
            "temporal_correlation": mode,
            "base_ratio": base_ratio,
            "solar_ratio": solar_ratio,
            "wind_ratio": wind_ratio,
            "solar_capacity_mw": solar_capacity_mw,
            "wind_capacity_mw": wind_capacity_mw,
            "ppa_capacity_mw": ppa_capacity_mw,
            "electrolyser_mw": electrolyser_mw,
            "overall_rfnbo_fraction": overall_rfnbo_fraction,
            "overall_rfnbo_pct": overall_rfnbo_fraction * 100,
            "total_consumption_mwh": total_consumption_mwh,
            "total_rfnbo_mwh": total_rfnbo_mwh,
            "interval_count": len(compliance_df),
        }
    ])


def find_columns_by_keyword(columns, keywords):
    cols = [c for c in columns if any(k in c.lower() for k in keywords)]
    return cols


def _psr_types_for_tech(tech: str):
    t = tech.lower()
    if "solar" in t:
        return ["B16"]
    if "offshore" in t:
        return ["B18"]
    if "onshore" in t:
        return ["B19"]
    if "wind" in t:
        return ["B18", "B19"]
    # fallback: try to lookup in PPA_TECHNOLOGY_PSR_TYPES keys
    for k, v in PPA_TECHNOLOGY_PSR_TYPES.items():
        if tech.lower() in k.lower():
            return v if isinstance(v, list) else [v]
    return []


def compute_cfs_from_data(gen_csv: Path, inst_csv: Path, tech_keys, start_date=None, end_date=None):
    gen = pd.read_csv(gen_csv, parse_dates=["timestamp"], low_memory=False)
    inst = pd.read_csv(inst_csv, parse_dates=["timestamp"], low_memory=False)
    def _to_ts(s):
        if s is None:
            return None
        ts = pd.to_datetime(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        return ts

    s_ts = _to_ts(start_date)
    e_ts = _to_ts(end_date)
    if s_ts is not None:
        gen = gen[gen['timestamp'] >= s_ts]
        inst = inst[inst['timestamp'] >= s_ts]
    if e_ts is not None:
        gen = gen[gen['timestamp'] <= e_ts]
        inst = inst[inst['timestamp'] <= e_ts]
    cfs = {}
    for tech in tech_keys:
        psr_types = _psr_types_for_tech(tech)
        if not psr_types:
            cfs[tech] = DEFAULT_CFS.get(tech, None)
            continue

        gen_f = gen[gen['psr_type'].isin(psr_types)].copy()
        inst_f = inst[inst['psr_type'].isin(psr_types)].copy()

        if gen_f.empty:
            cfs[tech] = DEFAULT_CFS.get(tech, None)
            continue

        # compute total energy in MWh using resolution_minutes
        if 'resolution_minutes' in gen_f.columns:
            gen_f['energy_mwh'] = gen_f['generation_mw'] * (gen_f['resolution_minutes'] / 60.0)
            total_gen_mwh = gen_f['energy_mwh'].sum()
            period_hours = (gen_f['resolution_minutes'].sum() / 60.0)
        else:
            # assume hourly
            total_gen_mwh = gen_f['generation_mw'].sum()
            period_hours = len(gen_f)

        avg_inst_mw = inst_f['installed_capacity_mw'].mean() if not inst_f.empty else 0.0

        if avg_inst_mw > 0 and period_hours > 0:
            cfs[tech] = float(total_gen_mwh) / (float(avg_inst_mw) * float(period_hours))
        else:
            cfs[tech] = DEFAULT_CFS.get(tech, None)

    return cfs


def temporal_simulation(
    gen_csv: Path,
    inst_csv: Path,
    desired_caps,
    out_csv: Path,
    start_date=None,
    end_date=None,
    temporal_correlation='hourly',
):
    gen = pd.read_csv(gen_csv, parse_dates=["timestamp"], low_memory=False)
    inst = pd.read_csv(inst_csv, parse_dates=["timestamp"], low_memory=False)
    def _to_ts(s):
        if s is None:
            return None
        ts = pd.to_datetime(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        return ts

    s_ts = _to_ts(start_date)
    e_ts = _to_ts(end_date)
    if s_ts is not None:
        gen = gen[gen['timestamp'] >= s_ts]
        inst = inst[inst['timestamp'] >= s_ts]
    if e_ts is not None:
        gen = gen[gen['timestamp'] <= e_ts]
        inst = inst[inst['timestamp'] <= e_ts]

    # ensure generation grouped by timestamp & psr_type
    timestamps = sorted(gen['timestamp'].unique())
    result = pd.DataFrame({'timestamp': timestamps})

    for tech, cap in desired_caps.items():
        psr_types = _psr_types_for_tech(tech)
        if not psr_types:
            continue

        gen_f = gen[gen['psr_type'].isin(psr_types)].copy()
        if gen_f.empty:
            continue

        # sum generation per timestamp
        gen_by_time = gen_f.groupby('timestamp')['generation_mw'].sum().reset_index()

        # get average installed capacity for these PSR types
        inst_f = inst[inst['psr_type'].isin(psr_types)].copy()
        avg_inst = inst_f['installed_capacity_mw'].mean() if not inst_f.empty else 0.0
        if avg_inst <= 0:
            continue

        scaling = cap / avg_inst
        gen_by_time['ppa_mw'] = gen_by_time['generation_mw'] * scaling

        # merge onto result
        result = result.merge(gen_by_time[['timestamp', 'ppa_mw']], on='timestamp', how='left')
        result['ppa_mw'] = result['ppa_mw'].fillna(0)
        result.rename(columns={'ppa_mw': tech}, inplace=True)

    # monthly aggregation option
    if temporal_correlation == 'monthly':
        df = result.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        monthly = df.resample('ME').sum().reset_index()
        if out_csv:
            monthly.to_csv(out_csv, index=False)
        return monthly

    if out_csv:
        result.to_csv(out_csv, index=False)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--electrolyser-mw", type=float, required=True)
    p.add_argument("--base-ppa-ratio", type=float, default=1.0)
    p.add_argument("--base-ppa-ratios", type=str, default=DEFAULT_BASE_PPA_RATIOS, help="Comma-separated base PPA ratios for sweep")
    p.add_argument("--tech-mix", type=str, default="offshore:1", help="JSON or comma list e.g. 'onshore:0.5,solar:0.5' or '{\"onshore\":0.5,\"solar\":0.5}'")
    p.add_argument("--sweep", dest='do_sweep', action='store_true', help='Run the solar/wind sweep')
    p.add_argument("--sweep-count", type=int, default=DEFAULT_SOLAR_RATIO_STEPS, help='Number of solar-ratio points between 0 and 1')
    p.add_argument("--sweep-types", type=str, default=DEFAULT_SWEEP_TYPES, help='Comma list of sweep combos (default set in code)')
    p.add_argument("--start-date", type=str, help='Start timestamp (inclusive) e.g. 2020-01-01')
    p.add_argument("--end-date", type=str, help='End timestamp (inclusive) e.g. 2020-12-31')
    p.add_argument("--temporal-correlation", type=str, choices=['hourly','monthly','all'], default='all', help='Temporal correlation handling')
    p.add_argument("--hours-per-year", type=float, default=8760)
    p.add_argument("--use-data", action="store_true")
    p.add_argument("--prices-file", type=str, default=str(DEFAULT_PRICES_CSV), help="Path to prices CSV (default set in code)")
    p.add_argument("--generation-csv", type=str, default=str(DEFAULT_GEN_CSV), help="Path to generation CSV for temporal scaling (default set in code)")
    p.add_argument("--installed-csv", type=str, default=str(DEFAULT_INST_CSV), help="Path to installed capacity CSV used to infer CFs (default set in code)")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--out-csv", type=str, help="Output CSV for temporal PPA series")
    p.add_argument("--open-html", action="store_true", help="Open generated HTML plots in the browser")
    args = p.parse_args()

    tech_mix = parse_tech_mix(args.tech_mix)
    tech_keys = list(tech_mix.keys())

    def _to_ts(value: str | None):
        if value is None:
            return None
        ts = pd.to_datetime(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts

    prices_df = None
    generation_df = None
    capacity_df = None
    renewable_share = 0.0

    if args.use_data or args.do_sweep or args.temporal:
        prices_path = Path(args.prices_file)
        generation_path = Path(args.generation_csv)
        installed_path = Path(args.installed_csv)

        if not prices_path.exists():
            raise FileNotFoundError(f"Prices CSV not found: {prices_path}")
        if not generation_path.exists():
            raise FileNotFoundError(f"Generation CSV not found: {generation_path}")
        if not installed_path.exists():
            raise FileNotFoundError(f"Installed-capacity CSV not found: {installed_path}")

        prices_df = pd.read_csv(prices_path, parse_dates=["datetime"], low_memory=False)
        generation_df = pd.read_csv(generation_path, parse_dates=["timestamp"], low_memory=False)
        capacity_df = pd.read_csv(installed_path, parse_dates=["timestamp"], low_memory=False)

        start_ts = _to_ts(args.start_date)
        end_ts = _to_ts(args.end_date)
        if start_ts is not None:
            prices_df = prices_df[prices_df["datetime"] >= start_ts]
            generation_df = generation_df[generation_df["timestamp"] >= start_ts]
            capacity_df = capacity_df[capacity_df["timestamp"] >= start_ts]
        if end_ts is not None:
            prices_df = prices_df[prices_df["datetime"] <= end_ts]
            generation_df = generation_df[generation_df["timestamp"] <= end_ts]
            capacity_df = capacity_df[capacity_df["timestamp"] <= end_ts]

        if prices_df.empty:
            raise ValueError("No prices data left after applying date filters.")
        if generation_df.empty:
            raise ValueError("No generation data left after applying date filters.")

        renewable_share = calculate_renewable_share(generation_df)

    if args.use_data:
        if not args.generation_csv or not args.installed_csv:
            raise SystemExit("--use-data requires --generation-csv and --installed-csv")
        cfs = compute_cfs_from_data(Path(args.generation_csv), Path(args.installed_csv), tech_keys, start_date=args.start_date, end_date=args.end_date)
        print(f"\nCapacity factors (calculated from data for {args.start_date} to {args.end_date}):")
        for tech in tech_keys:
            print(f"  - {tech}: {cfs[tech]:.4f}")
    else:
        cfs = DEFAULT_CFS.copy()
        print("\nCapacity factors (using default values):")
        for tech in tech_keys:
            print(f"  - {tech}: {cfs[tech]:.4f}")

    capacities = compute_required_capacities(args.electrolyser_mw, args.base_ppa_ratio, tech_mix, cfs)

    hours = args.hours_per_year
    energies = {t: capacities[t] * cfs[t] * hours for t in capacities}

    print("Sizing results:")
    for t in capacities:
        print(f"- {t}: capacity {capacities[t]:.3f} MW -> annual energy {energies[t]:.0f} MWh (cf={cfs[t]:.3f})")

    if args.do_sweep:
        if prices_df is None or generation_df is None or capacity_df is None:
            raise SystemExit("--sweep requires --use-data and the input CSVs")

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        base_ratios = [float(value) for value in args.base_ppa_ratios.split(",") if value.strip()]
        sweep_types = {value.strip() for value in args.sweep_types.split(",") if value.strip()}
        solar_ratios = _solar_ratio_values(args.sweep_count)
        modes = _selected_modes(args.temporal_correlation)

        generated_html_paths: list[Path] = []
        for mode in modes:
            for technology, technology_slug in TECHNOLOGY_SCENARIOS:
                scenario_key = f"solar-{technology_slug}"
                if scenario_key not in sweep_types:
                    continue

                scenario_frames: list[pd.DataFrame] = []
                cfs_local = compute_cfs_from_data(
                    Path(args.generation_csv),
                    Path(args.installed_csv),
                    ["solar", technology_slug],
                    start_date=args.start_date,
                    end_date=args.end_date,
                )

                for base_ratio in base_ratios:
                    for solar_ratio in solar_ratios:
                        frame = _build_compliance_point(
                            prices_df=prices_df,
                            generation_df=generation_df,
                            capacity_df=capacity_df,
                            country="Belgium",
                            electrolyser_mw=args.electrolyser_mw,
                            base_ratio=base_ratio,
                            solar_ratio=solar_ratio,
                            technology=technology,
                            technology_slug=technology_slug,
                            mode=mode,
                            renewable_share=renewable_share,
                            cfs=cfs_local,
                        )
                        if not frame.empty:
                            scenario_frames.append(frame)

                if not scenario_frames:
                    continue

                scenario_df = pd.concat(scenario_frames, ignore_index=True)
                scenario_df = scenario_df.sort_values(["base_ratio", "solar_ratio"]).reset_index(drop=True)

                basename = _build_output_basename("Belgium", mode, technology_slug)
                csv_path = out_dir / f"{basename}.csv"
                html_path = out_dir / f"{basename}.html"
                scenario_df.to_csv(csv_path, index=False)
                _build_capacity_factor_plot(scenario_df, "Belgium", mode, technology, html_path)
                generated_html_paths.append(html_path)
                print(csv_path)
                print(html_path)

        if args.open_html:
            for html_path in generated_html_paths:
                webbrowser.open(html_path.resolve().as_uri())

        print("Sweep finished.")
        return

    # Non-sweep / single run temporal
    if args.temporal:
        if prices_df is None or generation_df is None or capacity_df is None:
            raise SystemExit("--temporal requires the input CSVs")
        if args.temporal_correlation == "all":
            print("`--temporal-correlation all` is for sweep mode; using hourly for the single-run temporal output.")
        single_mode = args.temporal_correlation if args.temporal_correlation in {"hourly", "monthly"} else "hourly"
        out = Path(args.out_csv) if args.out_csv else None
        series = temporal_simulation(
            Path(args.generation_csv),
            Path(args.installed_csv),
            capacities,
            out,
            start_date=args.start_date,
            end_date=args.end_date,
            temporal_correlation=single_mode,
        )
        print(f"Temporal simulation produced {len(series)} timesteps and {len(series.columns)} tech columns")


if __name__ == "__main__":
    main()
