# RFNBO Compliancy Calculator

A Streamlit application for testing RFNBO (Renewable Fuel of Non-Biological Origin) compliance of electrolyser operations based on ENTSOE electricity market data.

## Overview

This application helps determine whether hydrogen produced through electrolysis meets RFNBO compliance requirements by:
- Fetching real-time data from ENTSOE (day-ahead prices, generation mix)
- Calculating the renewable fraction of consumed electricity
- Applying temporal correlation rules (hourly or monthly)
- Implementing the 20% price threshold rule
- Verifying 70% GHG savings requirement

## Features

### 📊 Key Capabilities

1. **Data Fetching from ENTSOE**
   - Day-ahead electricity prices
   - Generation mix by source type
   - Automatic data aggregation for selected month

2. **Real PPA Generation Data** ✨ NEW
   - Use actual solar/wind generation from ENTSOE
   - Hour-by-hour capacity factor variation
   - More accurate than constant capacity factor
   - Supports Solar, Wind Onshore, Wind Offshore
   - Automatic fallback to constant CF if data unavailable

3. **ENTSOE Data Explorer**
   - View raw ENTSOE data in dedicated tab
   - Interactive charts showing price distribution and generation mix
   - **Automatic highlighting of low-price periods** (prices < 20€/MWh shown in orange)
   - Detailed statistics and metrics
   - Separate views for prices and generation data
   - Download raw data as CSV

4. **RFNBO Compliance Calculation**
   - PPA (Power Purchase Agreement) energy tracking (100% RFNBO)
   - Grid energy with renewable share calculation
   - 20% price threshold rule (grid considered 100% renewable during low-price periods)
   - GHG savings calculation vs 94 g CO₂eq/MJ fossil comparator

5. **Temporal Correlation**
   - Hourly correlation: Hour-by-hour matching of consumption with renewable production
   - Monthly correlation: Monthly averaged matching

6. **Flexible Configuration**
   - Configurable electrolyser capacity
   - Configurable PPA capacity and capacity factor
   - Manual or automatic renewable share calculation
   - Enable/disable price threshold rule

7. **Interactive Visualizations**
   - RFNBO fraction over time
   - Energy sources breakdown (PPA vs Grid)
   - Price correlation with compliance
   - Monthly summary statistics
   - Price distribution histograms
   - Generation mix by source type

## Installation

### Prerequisites

- Python 3.8 or higher
- ENTSOE API key (set as environment variable `ENTSOE_API_KEY`)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your ENTSOE API key:
```bash
# Windows
set ENTSOE_API_KEY=your_api_key_here

# Linux/Mac
export ENTSOE_API_KEY=your_api_key_here
```

## Usage

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Standalone Ratio Sweep

Run the non-Streamlit ratio sweep to compare the five approved technologies across PPA-to-electrolyser ratios and export a CSV-only interval summary:

```bash
python rfnbo_ratio_sweep_cli.py --country Belgium --electrolyser-mw 50 --ratios 0.75 1.0 1.25
```

The output CSV contains one row per interval, technology, and ratio with the overall RFNBO percentage.

### Solar Ratio Sweep

Run the solar-share sweep to see how the solar fraction inside a combined PPA changes RFNBO compliance for both offshore and onshore wind combinations. This command generates four graphs by default: hourly and monthly results for offshore wind and for onshore wind.

```bash
python rfnbo_solar_ratio_sweep_cli.py --country Belgium --electrolyser-mw 50 --start 2023-01-01 --end 2023-01-31
```

You can override the default PPA ratios with `--ppa-ratios` and the solar-share resolution with `--solar-ratios`.

### PPA Sourcing Cost Sweep

Run the PPA sourcing cost sweep to estimate a contract price that follows the day-ahead market plus an extra margin. The script sweeps both the production-to-consumption ratio and the extra margin, then reports total cost and RFNBO percentage. The margin is only charged on PPA energy that is actually produced; residual grid energy stays at DAM.

```bash
python PPA_sourcing_cost.py --country Belgium --electrolyser-mw 50 --ratios 0.5 0.75 1.0 1.25 --margins 1 2 3 4 5
```

The default run compares Solar, Wind Offshore, Wind Onshore, and the two 50/50 combined cases. The script writes one HTML plot per technology so the curves stay readable: color = extra margin, bubble size = RFNBO percentage. If you want only a subset, pass `--technologies` with one or more values.

For annual numbers, pass an explicit `--start` and `--end` window such as `--start 2025-01-01 --end 2025-12-31`. Otherwise the script uses the full loaded dataset.

### Capacity-Factor-Aware PPA Sourcing Cost Sweep

Use the CF-aware variant when you want the ratio to represent energy rather than installed capacity. It sizes each PPA from the relevant capacity factor before calculating the DAM + margin cost and produces the same aggregated all-technologies plot.

```bash
python PPA_sourcing_cost_capacity_factor.py --country Belgium --electrolyser-mw 50 --ratios 0.5 0.75 1.0 1.25 --margins 1 2 3 4 5
```

The default run uses data-derived capacity factors from the loaded ENTSOE generation and installed-capacity series, with a fallback to the built-in defaults if needed.

### Configuration Steps

1. **Select Country**: Choose the electricity bidding zone
2. **Set Time Period**: Select year and month for analysis
3. **Configure Electrolyser**: Set electrolyser capacity in MW
4. **Configure PPA**: 
   - Set PPA renewable capacity in MW
   - Set expected capacity factor (0-1)
5. **Choose Temporal Correlation**: Hourly or monthly
6. **Set Renewable Mix**:
   - Calculate from ENTSOE data (annual or hourly)
   - Or manually input a renewable share percentage
7. **Enable/Disable Price Rule**: Toggle the 20% price threshold rule
8. **Click "Fetch Data & Calculate"**: Start the analysis

### Application Tabs

The application has two main tabs:

#### 1. 🎯 RFNBO Analysis Tab
- Shows calculated RFNBO compliance results
- Interactive charts and visualizations
- Compliance status and key metrics
- Downloadable results

#### 2. 📊 ENTSOE Data Explorer Tab
- View raw ENTSOE data before processing
- **Low-price highlighting**: Rows with prices < 20€/MWh highlighted in orange
- Price distribution charts
- Generation mix by source type
- Statistics and metrics for fetched data
- Download raw data as CSV

### Understanding Results

#### Compliance Status
- ✅ **COMPLIANT**: Overall RFNBO fraction ≥ 70%
- ❌ **NOT COMPLIANT**: Overall RFNBO fraction < 70%

#### Key Metrics
- **Total Consumption**: Total energy consumed by electrolyser (MWh)
- **RFNBO Fraction**: Percentage of consumption that qualifies as RFNBO
- **Avg GHG Savings**: Average greenhouse gas savings vs fossil comparator
- **Compliant Hours**: Percentage of hours meeting compliance threshold

#### Visualizations

1. **RFNBO Fraction Over Time**: Shows temporal variation in RFNBO compliance
2. **Energy Sources**: Stacked area chart showing PPA vs Grid consumption
3. **Price vs Compliance**: Scatter plot showing relationship between electricity prices and compliance
4. **Monthly Summary**: Key statistics for the analyzed period

## RFNBO Compliance Rules

### 1. GHG Savings Requirement
- Minimum 70% GHG savings required
- Fossil comparator: 94 g CO₂eq/MJ
- Renewable electricity: 0 g CO₂eq/MJ

### 2. Energy Sources
- **PPA Energy**: 100% RFNBO (direct renewable contract)
- **Grid Energy**: RFNBO fraction = renewable share in national mix
  - Exception: 100% RFNBO when prices < 20th percentile

### 3. Temporal Correlation
- **Hourly**: Consumption must match renewable production hour-by-hour
- **Monthly**: Consumption can be averaged over the month

## File Structure

```
RFNBO-COMPLIANCY/
├── streamlit_app.py          # Main Streamlit application (UI layer)
├── rfnbo_calculations.py     # Core RFNBO calculation functions
├── fetch_entsoe.py            # ENTSOE API data fetching functions
├── example_usage.py           # Programmatic usage examples
├── test_setup.py              # Setup verification script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PROJECT_STRUCTURE.md       # Detailed architecture documentation
├── QUICKSTART.md              # Quick start guide
├── CHANGELOG.md               # Version history
└── DATA_EXPLORER_GUIDE.md     # Data explorer feature guide
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed architecture documentation.**

## Technical Details

### ENTSOE PSR Types (Generation Sources)

**Renewable Sources (RFNBO-eligible):**
- B09: Geothermal
- B11: Hydro Run-of-river
- B12: Hydro Water Reservoir
- B13: Marine
- B15: Other renewable
- B16: Solar
- B18: Wind Offshore
- B19: Wind Onshore

**Non-Renewable Sources:**
- B01: Biomass
- B02-B08: Fossil fuels (coal, gas, oil, etc.)
- B10: Hydro Pumped Storage
- B14: Nuclear
- B17: Waste
- B20: Other

### Calculation Methodology

1. **Fetch Market Data**: Download day-ahead prices and generation mix from ENTSOE
2. **Calculate Renewable Share**: 
   - Sum renewable generation / Total generation
   - Can be annual average or hourly varying
3. **Apply PPA**: Direct renewable consumption from PPA (100% RFNBO)
4. **Grid Consumption**: 
   - When consumption > PPA, use grid
   - Grid RFNBO = Grid consumption × renewable share
   - If prices < 20th percentile: Grid RFNBO = Grid consumption × 1.0
5. **Calculate Total RFNBO**: PPA RFNBO + Grid RFNBO
6. **Verify Compliance**: Check if GHG savings ≥ 70%

### GHG Savings Formula

```
renewable_fraction = RFNBO_energy / Total_energy
actual_emission = (1 - renewable_fraction) × 94 g CO₂eq/MJ
ghg_savings = (94 - actual_emission) / 94
compliance = ghg_savings ≥ 0.70
```

## Troubleshooting

### Common Issues

1. **No data available**: 
   - Check ENTSOE API key is set correctly
   - Verify the selected country and date have available data
   - Some countries may not report all data types

2. **Slow data fetching**:
   - ENTSOE API can be slow during peak hours
   - Consider shorter time periods for faster results

3. **Missing generation data**:
   - Not all countries report generation mix to ENTSOE
   - Use manual renewable share input as fallback

## API Key

To get an ENTSOE API key:
1. Register at https://transparency.entsoe.eu/
2. Create an account
3. Request API access through your account settings
4. Set the key as environment variable `ENTSOE_API_KEY`

## License

This project is for internal use at H2 Growth Unit.

## Support

For questions or issues, contact the H2 Growth Unit team.

