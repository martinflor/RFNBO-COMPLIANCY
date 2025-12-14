import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

# Import ENTSOE data fetching functions
from fetch_entsoe import (
    fetch_day_ahead_prices, 
    fetch_generation_mix,
    fetch_all_generation_types,
    fetch_installed_capacity_all_types,
    get_backend_country_name,
    BIDDING_ZONES,
    COUNTRY_TIMEZONES
)

# Import RFNBO calculation functions
from rfnbo_calculations import (
    calculate_renewable_share,
    calculate_rfnbo_compliance,
    aggregate_to_monthly,
    is_rfnbo_compliant,
    calculate_statistics,
    calculate_generation_statistics,
    get_grid_emission_factor,
    calculate_ppa_production_from_generation_data,
    PSR_TYPE_MAPPING,
    RENEWABLE_PSR_TYPES,
    PPA_TECHNOLOGY_PSR_TYPES,
    MAX_EMISSION_FACTOR_MJ,
    PRICE_THRESHOLD_EUR_MWH,
    FOSSIL_COMPARATOR_MJ
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RFNBO Compliancy Calculator",
    page_icon="⚡",
    layout="wide"
)

def fetch_month_data(country: str, year: int, month: int, fetch_capacity: bool = False):
    """
    Fetch all required data for a given month from ENTSOE.
    
    Args:
        country: Country name
        year: Year
        month: Month (1-12)
        fetch_capacity: Whether to fetch installed capacity data
    
    Returns:
        dict with 'prices', 'generation', and optionally 'installed_capacity' DataFrames
    """
    # Generate date range for the month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    all_prices = []
    all_generation = []
    all_capacity = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, date in enumerate(date_range):
        date_str = date.strftime('%Y-%m-%d')
        status_text.text(f"Fetching data for {date_str}...")
        
        try:
            # Fetch day-ahead prices
            prices_df = fetch_day_ahead_prices(date_str, country)
            if not prices_df.empty:
                all_prices.append(prices_df)
            
            # Fetch generation mix
            gen_df = fetch_all_generation_types(date_str, country)
            if not gen_df.empty:
                all_generation.append(gen_df)
            
            # Fetch installed capacity if requested (only first day needed as it's constant)
            if fetch_capacity and idx == 0:
                capacity_df = fetch_installed_capacity_all_types(date_str, country)
                if not capacity_df.empty:
                    all_capacity.append(capacity_df)
        
        except Exception as e:
            logger.warning(f"Failed to fetch data for {date_str}: {e}")
            st.warning(f"⚠️ Could not fetch complete data for {date_str}")
        
        progress_bar.progress((idx + 1) / len(date_range))
    
    progress_bar.empty()
    status_text.empty()
    
    result = {}
    
    if all_prices:
        result['prices'] = pd.concat(all_prices, ignore_index=True)
        st.success(f"✅ Fetched {len(result['prices'])} price records")
    else:
        result['prices'] = pd.DataFrame()
        st.error("❌ No price data available for this period")
    
    if all_generation:
        result['generation'] = pd.concat(all_generation, ignore_index=True)
        st.success(f"✅ Fetched {len(result['generation'])} generation records")
    else:
        result['generation'] = pd.DataFrame()
        st.warning("⚠️ No generation data available for this period")
    
    if fetch_capacity:
        if all_capacity:
            result['installed_capacity'] = pd.concat(all_capacity, ignore_index=True)
            st.success(f"✅ Fetched {len(result['installed_capacity'])} installed capacity records")
        else:
            result['installed_capacity'] = pd.DataFrame()
            st.warning("⚠️ No installed capacity data available (will use fallback method)")
    else:
        result['installed_capacity'] = pd.DataFrame()
    
    return result

def create_visualizations(results_df: pd.DataFrame, monthly_summary: pd.DataFrame):
    """
    Create visualizations for RFNBO analysis.
    """
    if results_df.empty:
        st.error("No data to visualize")
        return
    
    # 0. PPA Capacity Factor and Production (if using real generation data)
    # COMMENTED OUT FOR NOW - capacity factor temporarily disabled
    if False and 'capacity_factor' in results_df.columns and results_df['capacity_factor'].std() > 0.01:
        st.subheader("☀️ PPA Capacity Factor and Production Profile")
        st.caption("Actual capacity factor and adjusted PPA production from ENTSOE generation data")
        
        # Add explanation
        with st.expander("ℹ️ How is PPA production calculated?"):
            st.markdown("""
            **Methodology** (using real ENTSOE data):
            
            1. **Fetch installed capacity** from ENTSOE (documentType A71)
               - Example: Belgium Solar = 5,000 MW installed nationally
            
            2. **Fetch actual generation** timeseries (documentType A75)
               - Example: At 12:00 → 4,000 MW generated
            
            3. **Calculate capacity factor**: `CF = Generation / Installed_Capacity`
               - Example: CF = 4,000 / 5,000 = **0.80 (80%)**
            
            4. **Scale to your PPA capacity**: `PPA_production = Generation × (PPA_capacity / National_capacity)`
               - Example: If PPA = 10 MW → Production = 4,000 × (10/5,000) = **8 MW**
               - Or equivalently: CF × PPA_capacity = 0.80 × 10 = **8 MW**
            
            **Benefits**:
            - ✅ Accounts for real weather patterns (solar peaks at midday, zero at night)
            - ✅ Wind production varies with actual wind conditions
            - ✅ Accurate temporal matching for RFNBO compliance
            - ✅ Better than constant capacity factor assumption
            """)
        
        # Create subplot with 2 y-axes
        fig0 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Capacity Factor (%)', 'PPA Production (MW)'),
            vertical_spacing=0.12,
            shared_xaxes=True
        )
        
        # Top plot: Capacity Factor
        fig0.add_trace(
            go.Scatter(
                x=results_df['datetime'],
                y=results_df['capacity_factor'] * 100,
                mode='lines',
                name='Capacity Factor',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)',
                hovertemplate='<b>%{x}</b><br>CF: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add mean CF line
        fig0.add_hline(
            y=results_df['capacity_factor'].mean() * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results_df['capacity_factor'].mean()*100:.1f}%",
            row=1, col=1
        )
        
        # Bottom plot: PPA Production
        fig0.add_trace(
            go.Scatter(
                x=results_df['datetime'],
                y=results_df['ppa_production_mw'],
                mode='lines',
                name='PPA Production',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.2)',
                hovertemplate='<b>%{x}</b><br>Production: %{y:.2f} MW<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add electrolyser consumption line for reference
        fig0.add_trace(
            go.Scatter(
                x=results_df['datetime'],
                y=results_df['electrolyser_consumption_mw'],
                mode='lines',
                name='Electrolyser Consumption',
                line=dict(color='blue', width=1, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Consumption: %{y:.2f} MW<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add mean production line
        fig0.add_hline(
            y=results_df['ppa_production_mw'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results_df['ppa_production_mw'].mean():.2f} MW",
            row=2, col=1
        )
        
        # Update axes labels
        fig0.update_xaxes(title_text="Date & Time", row=2, col=1)
        fig0.update_yaxes(title_text="Capacity Factor (%)", row=1, col=1)
        fig0.update_yaxes(title_text="Power (MW)", row=2, col=1)
        
        # Update layout
        fig0.update_layout(
            height=700,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig0, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean CF", f"{results_df['capacity_factor'].mean()*100:.1f}%")
        with col2:
            st.metric("Max CF", f"{results_df['capacity_factor'].max()*100:.1f}%")
        with col3:
            st.metric("Mean PPA Production", f"{results_df['ppa_production_mw'].mean():.2f} MW")
        with col4:
            st.metric("Max PPA Production", f"{results_df['ppa_production_mw'].max():.2f} MW")
    
    # 1. RFNBO Fraction Over Time
    st.subheader("📊 RFNBO Fraction Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=results_df['datetime'],
        y=results_df['rfnbo_fraction'] * 100,
        mode='lines',
        name='RFNBO Fraction',
        line=dict(color='green')
    ))
    fig1.add_hline(y=100, line_dash="dash", line_color="red", 
                   annotation_text="100% RFNBO Target")
    fig1.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="RFNBO Fraction (%)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Energy Sources Breakdown
    st.subheader("⚡ Energy Sources")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results_df['datetime'],
        y=results_df['ppa_energy_mwh'],
        mode='lines',
        name='PPA (100% RFNBO)',
        stackgroup='one',
        line=dict(color='green')
    ))
    fig2.add_trace(go.Scatter(
        x=results_df['datetime'],
        y=results_df['rfnbo_from_grid_mwh'],
        mode='lines',
        name='Grid (RFNBO)',
        stackgroup='one',
        line=dict(color='lightgreen')
    ))
    fig2.add_trace(go.Scatter(
        x=results_df['datetime'],
        y=results_df['grid_energy_mwh'] - results_df['rfnbo_from_grid_mwh'],
        mode='lines',
        name='Grid (Non-RFNBO)',
        stackgroup='one',
        line=dict(color='orange')
    ))
    fig2.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Energy (MWh)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Energy & Emissions Breakdown
    st.subheader("📋 Energy & Emissions Breakdown")
    
    st.markdown("""
    This table shows how your electrolyser's energy consumption is sourced and the associated emissions.
    - **For emission calculations**: Low-price grid (<20€/MWh) is treated as 0 emissions
    - **For RFNBO matching**: Uses actual renewable share from energy mix (low-price rule does NOT apply)
    """)
    
    # Create breakdown table
    breakdown_data = {
        'Energy Source': [],
        'Energy (MWh)': [],
        'Percentage (%)': [],
        'Emission Factor (g CO₂eq/kWh)': [],
        'Total Emissions (kg CO₂eq)': [],
        'RFNBO Status': []
    }
    
    total_consumption = results_df['electrolyser_consumption_mwh'].sum()
    ppa_energy = results_df['ppa_energy_mwh'].sum()
    grid_total = results_df['grid_energy_mwh'].sum()
    grid_low_price = results_df['grid_energy_low_price_mwh'].sum()
    grid_normal_price = results_df['grid_energy_normal_price_mwh'].sum()
    
    # Calculate renewable and non-renewable parts of grid (for RFNBO matching)
    if 'grid_renewable_share_mix' in results_df.columns:
        grid_renewable_part = (results_df['grid_energy_mwh'] * results_df['grid_renewable_share_mix']).sum()
        grid_non_renewable_part = (results_df['grid_energy_mwh'] * (1 - results_df['grid_renewable_share_mix'])).sum()
    else:
        grid_renewable_part = 0
        grid_non_renewable_part = grid_total
    
    # Get country emission factor
    country = st.session_state.get('country', 'Belgium')
    backend_country = get_backend_country_name(country)
    grid_ef = get_grid_emission_factor(backend_country)
    
    # PPA Energy (0 emissions, 100% RFNBO)
    breakdown_data['Energy Source'].append('1. PPA (Renewable Contract)')
    breakdown_data['Energy (MWh)'].append(ppa_energy)
    breakdown_data['Percentage (%)'].append(ppa_energy / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(0)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(0)
    breakdown_data['RFNBO Status'].append('✅ 100% RFNBO')
    
    # Grid - Low Price (< 20€/MWh)
    # When prices < 20€/MWh: Considered 100% renewable (no subdivision)
    # - 0 emissions for emission calculation
    # - 100% RFNBO for RFNBO matching
    breakdown_data['Energy Source'].append('2. Grid - Low Price (<20€/MWh)')
    breakdown_data['Energy (MWh)'].append(grid_low_price)
    breakdown_data['Percentage (%)'].append(grid_low_price / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(0)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(0)
    breakdown_data['RFNBO Status'].append('✅ 100% RFNBO (low price rule)')
    
    # Grid - Normal Price (≥ 20€/MWh)
    # Split normal-price grid into renewable and non-renewable parts based on energy mix
    if 'grid_renewable_share_mix' in results_df.columns:
        grid_normal_price_renewable = (results_df['grid_energy_normal_price_mwh'] * results_df['grid_renewable_share_mix']).sum()
        grid_normal_price_non_renewable = (results_df['grid_energy_normal_price_mwh'] * (1 - results_df['grid_renewable_share_mix'])).sum()
    else:
        grid_normal_price_renewable = 0
        grid_normal_price_non_renewable = grid_normal_price
    
    # Grid - Normal Price - Renewable Part (from energy mix)
    grid_normal_price_renewable_emissions = (grid_normal_price_renewable * grid_ef * 1000) / 1000  # kg
    breakdown_data['Energy Source'].append('3a. Grid - Normal Price - Renewable Part (≥20€/MWh)')
    breakdown_data['Energy (MWh)'].append(grid_normal_price_renewable)
    breakdown_data['Percentage (%)'].append(grid_normal_price_renewable / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(grid_ef)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(grid_normal_price_renewable_emissions)
    breakdown_data['RFNBO Status'].append('✅ Counts as RFNBO (from energy mix), has emissions')
    
    # Grid - Normal Price - Non-Renewable Part (fossil)
    grid_normal_price_non_renewable_emissions = (grid_normal_price_non_renewable * grid_ef * 1000) / 1000  # kg
    breakdown_data['Energy Source'].append('3b. Grid - Normal Price - Non-Renewable Part (≥20€/MWh)')
    breakdown_data['Energy (MWh)'].append(grid_normal_price_non_renewable)
    breakdown_data['Percentage (%)'].append(grid_normal_price_non_renewable / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(grid_ef)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(grid_normal_price_non_renewable_emissions)
    breakdown_data['RFNBO Status'].append('❌ Not RFNBO, causes emissions')
    
    # Separator
    breakdown_data['Energy Source'].append('─────────────')
    breakdown_data['Energy (MWh)'].append(None)
    breakdown_data['Percentage (%)'].append(None)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(None)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(None)
    breakdown_data['RFNBO Status'].append('')
    
    # Total
    total_emissions = results_df['emissions_g_co2_eq'].sum() / 1000
    breakdown_data['Energy Source'].append('TOTAL CONSUMPTION')
    breakdown_data['Energy (MWh)'].append(total_consumption)
    breakdown_data['Percentage (%)'].append(100)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append('-')
    breakdown_data['Total Emissions (kg CO₂eq)'].append(total_emissions)
    breakdown_data['RFNBO Status'].append('')
    
    # RFNBO breakdown (separate view)
    breakdown_data['Energy Source'].append('─────────────')
    breakdown_data['Energy (MWh)'].append(None)
    breakdown_data['Percentage (%)'].append(None)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(None)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(None)
    breakdown_data['RFNBO Status'].append('')
    
    breakdown_data['Energy Source'].append('RFNBO Analysis:')
    breakdown_data['Energy (MWh)'].append(None)
    breakdown_data['Percentage (%)'].append(None)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append(None)
    breakdown_data['Total Emissions (kg CO₂eq)'].append(None)
    breakdown_data['RFNBO Status'].append('')
    
    # Calculate Total RFNBO Energy
    # RFNBO = PPA + Low-price grid (100%) + Normal-price grid renewable part
    rfnbo_total = ppa_energy + grid_low_price + grid_normal_price_renewable
    
    breakdown_data['Energy Source'].append('→ Total RFNBO Energy')
    breakdown_data['Energy (MWh)'].append(rfnbo_total)
    breakdown_data['Percentage (%)'].append(rfnbo_total / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append('-')
    breakdown_data['Total Emissions (kg CO₂eq)'].append('-')
    breakdown_data['RFNBO Status'].append('✅ = 1 + 2 + 3a')
    
    # Calculate Non-RFNBO Energy
    non_rfnbo_total = grid_normal_price_non_renewable
    
    breakdown_data['Energy Source'].append('→ Non-RFNBO Energy')
    breakdown_data['Energy (MWh)'].append(non_rfnbo_total)
    breakdown_data['Percentage (%)'].append(non_rfnbo_total / total_consumption * 100 if total_consumption > 0 else 0)
    breakdown_data['Emission Factor (g CO₂eq/kWh)'].append('-')
    breakdown_data['Total Emissions (kg CO₂eq)'].append('-')
    breakdown_data['RFNBO Status'].append('❌ = 3b only')
    
    breakdown_df = pd.DataFrame(breakdown_data)
    
    # Display table with formatting
    def format_breakdown(val):
        """Format values, handling None/NaN"""
        if pd.isna(val) or val is None:
            return ''
        if isinstance(val, (int, float)):
            if abs(val) < 0.01 and val != 0:
                return f'{val:.3f}'
            return f'{val:.2f}'
        return str(val)
    
    st.dataframe(
        breakdown_df.style.format(format_breakdown, subset=['Energy (MWh)', 'Percentage (%)', 'Total Emissions (kg CO₂eq)']),
        use_container_width=True,
        hide_index=True
    )
    
    # Emission Factor Comparison
    st.subheader("🔬 Emission Factor Comparison vs Fossil Benchmark")
    
    total_emissions_g = results_df['emissions_g_co2_eq'].sum()
    total_energy_mj = results_df['total_consumption_mj'].sum()
    avg_ef_mj = total_emissions_g / total_energy_mj if total_energy_mj > 0 else 0
    avg_ef_kwh = avg_ef_mj * 3.6
    
    fossil_comparator = FOSSIL_COMPARATOR_MJ
    emission_limit = MAX_EMISSION_FACTOR_MJ
    savings = ((fossil_comparator - avg_ef_mj) / fossil_comparator * 100) if fossil_comparator > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Fossil Comparator",
            f"{fossil_comparator:.1f} g CO₂eq/MJ",
            help="Benchmark for fossil-based hydrogen production"
        )
    
    with col2:
        st.metric(
            "RFNBO Limit (30%)",
            f"{emission_limit:.1f} g CO₂eq/MJ",
            help="Maximum allowed emission factor for RFNBO compliance"
        )
    
    with col3:
        st.metric(
            "Your Emission Factor",
            f"{avg_ef_mj:.1f} g CO₂eq/MJ",
            delta=f"{savings:.1f}% savings",
            delta_color="normal" if savings >= 70 else "inverse",
            help="Weighted average emission factor of your electrolyser"
        )
    
    # Detailed comparison table
    comparison_data = {
        'Metric': [
            'Fossil Comparator (Benchmark)',
            'RFNBO Emission Limit (30% of fossil)',
            'Your Electrolyser Emission Factor',
            'Difference vs Limit',
            'GHG Savings vs Fossil',
            'Emission Compliance Status'
        ],
        'Value (g CO₂eq/MJ)': [
            f'{fossil_comparator:.2f}',
            f'{emission_limit:.2f}',
            f'{avg_ef_mj:.2f}',
            f'{avg_ef_mj - emission_limit:+.2f}' + (' ✅' if avg_ef_mj <= emission_limit else ' ❌'),
            f'{savings:.1f}%' + (' ✅ ≥70%' if savings >= 70 else ' ⚠️ <70%'),
            '✅ COMPLIANT' if avg_ef_mj <= emission_limit else '❌ NON-COMPLIANT'
        ],
        'Value (g CO₂eq/kWh)': [
            f'{fossil_comparator * 3.6:.2f}',
            f'{emission_limit * 3.6:.2f}',
            f'{avg_ef_kwh:.2f}',
            f'{(avg_ef_mj - emission_limit) * 3.6:+.2f}',
            '-',
            '-'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.caption("""
    **Key Insights:**
    - Emission factor must be **< 28.2 g CO₂eq/MJ** for compliance
    - This represents **70% GHG savings** vs fossil-based hydrogen
    - Lower emission factor = More renewable energy used
    """)
    
    # 4. Monthly Summary Statistics
    if not monthly_summary.empty:
        st.subheader("📈 Monthly Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Consumption",
                f"{monthly_summary['total_consumption_mwh'].values[0]:.0f} MWh"
            )
        
        with col2:
            emission_factor = monthly_summary['avg_emission_factor_mj'].values[0]
            st.metric(
                "Avg Emission Factor",
                f"{emission_factor:.1f} g CO₂eq/MJ",
                delta=f"{emission_factor - MAX_EMISSION_FACTOR_MJ:.1f} vs {MAX_EMISSION_FACTOR_MJ:.1f} limit",
                delta_color="inverse"
            )
        
        with col3:
            rfnbo_pct = monthly_summary['rfnbo_fraction'].values[0] * 100
            st.metric(
                "RFNBO Fraction",
                f"{rfnbo_pct:.1f}%",
                delta=f"{rfnbo_pct - 100:.1f}% vs 100% target"
            )
        
        with col4:
            compliance_pct = monthly_summary['overall_compliance_rate'].values[0] * 100
            st.metric(
                "Compliant Hours",
                f"{compliance_pct:.1f}%",
                delta=f"{monthly_summary['compliant_hours'].values[0]:.0f}/{monthly_summary['total_hours'].values[0]:.0f} hours"
            )

def highlight_low_prices(row):
    """
    Highlight rows where price is below 20€/MWh.
    Returns CSS styles for the row.
    """
    if row['price_eur_mwh'] < 20:
        return ['background-color: #FFA500; color: white'] * len(row)  # Orange
    return [''] * len(row)

def display_data_explorer_tab():
    """
    Display the ENTSOE Data Explorer tab.
    """
    st.header("📊 ENTSOE Data Explorer")
    st.markdown("""
    View raw data fetched from ENTSOE. Rows with prices **below 20€/MWh** are highlighted in **orange**.
    """)
    
    # Check if data exists in session state
    if 'fetched_data' not in st.session_state:
        st.info("👈 Please fetch data using the sidebar configuration first")
        return
    
    data = st.session_state['fetched_data']
    
    # Create sub-tabs for prices and generation
    data_tab1, data_tab2 = st.tabs(["💰 Day-Ahead Prices", "⚡ Generation Mix"])
    
    with data_tab1:
        st.subheader("Day-Ahead Electricity Prices")
        
        if data['prices'].empty:
            st.warning("No price data available")
        else:
            prices_df = data['prices'].copy()
            
            # Add additional information
            prices_df['below_20'] = prices_df['price_eur_mwh'] < 20
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(prices_df))
            with col2:
                st.metric("Avg Price", f"{prices_df['price_eur_mwh'].mean():.2f} €/MWh")
            with col3:
                st.metric("Min Price", f"{prices_df['price_eur_mwh'].min():.2f} €/MWh")
            with col4:
                below_20_count = prices_df['below_20'].sum()
                below_20_pct = (below_20_count / len(prices_df)) * 100
                st.metric("Below 20€/MWh", f"{below_20_count} ({below_20_pct:.1f}%)")
            
            st.markdown("---")
            
            # Price distribution chart
            st.subheader("📈 Price Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=prices_df['price_eur_mwh'],
                nbinsx=50,
                name='Price Distribution',
                marker=dict(color='blue', opacity=0.7)
            ))
            fig.add_vline(x=20, line_dash="dash", line_color="orange", 
                         annotation_text="20€/MWh threshold")
            fig.update_layout(
                xaxis_title="Price (€/MWh)",
                yaxis_title="Frequency",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price over time chart
            st.subheader("📉 Prices Over Time")
            fig2 = go.Figure()
            
            # Split data into below and above 20€/MWh
            below_20 = prices_df[prices_df['below_20']]
            above_20 = prices_df[~prices_df['below_20']]
            
            fig2.add_trace(go.Scatter(
                x=above_20['datetime'],
                y=above_20['price_eur_mwh'],
                mode='markers',
                name='Above 20€/MWh',
                marker=dict(color='blue', size=4)
            ))
            fig2.add_trace(go.Scatter(
                x=below_20['datetime'],
                y=below_20['price_eur_mwh'],
                mode='markers',
                name='Below 20€/MWh (Low price)',
                marker=dict(color='orange', size=6)
            ))
            fig2.add_hline(y=20, line_dash="dash", line_color="orange",
                          annotation_text="20€/MWh threshold")
            fig2.update_layout(
                xaxis_title="Date & Time",
                yaxis_title="Price (€/MWh)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Data table with highlighting
            st.subheader("📋 Detailed Price Data")
            st.caption("Rows highlighted in orange have prices below 20€/MWh")
            
            # Display options
            col1, col2 = st.columns([1, 3])
            with col1:
                show_all = st.checkbox("Show all data", value=False)
            with col2:
                if not show_all:
                    st.info("Showing first 100 rows. Check 'Show all data' to see everything.")
            
            # Prepare display dataframe
            display_df = prices_df[['datetime', 'zone', 'price_eur_mwh', 'resolution_minutes']].copy()
            display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            
            if not show_all:
                display_df = display_df.head(100)
            
            # Apply styling
            styled_df = display_df.style.apply(
                lambda row: ['background-color: #FFA500; color: white' if prices_df.loc[row.name, 'below_20'] else '' for _ in row],
                axis=1
            ).format({
                'price_eur_mwh': '{:.2f}',
                'resolution_minutes': '{:.0f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download button
            csv = prices_df[['datetime', 'zone', 'price_eur_mwh', 'resolution_minutes']].to_csv(index=False)
            st.download_button(
                label="📥 Download Price Data as CSV",
                data=csv,
                file_name=f"entsoe_prices_{st.session_state.get('country', 'data')}.csv",
                mime="text/csv"
            )
    
    with data_tab2:
        st.subheader("Generation Mix by Source")
        
        if data['generation'].empty:
            st.warning("No generation data available")
        else:
            gen_df = data['generation'].copy()
            
            # Add readable names
            gen_df['source_type'] = gen_df['psr_type'].map(PSR_TYPE_MAPPING)
            gen_df['is_renewable'] = gen_df['psr_type'].isin(RENEWABLE_PSR_TYPES)
            
            # Calculate energy using trapezoidal integration
            # Note: Data is in MW (power), need to integrate to get MWh (energy)
            stats = calculate_generation_statistics(gen_df)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", stats.get('total_records', 0))
            with col2:
                st.metric("Total Generation", f"{stats.get('total_generation', 0):,.0f} MWh")
            with col3:
                st.metric("Renewable Share", f"{stats.get('renewable_share', 0):.1f}%")
            with col4:
                st.metric("Source Types", stats.get('unique_sources', 0))
            
            st.markdown("---")
            
            # Generation by source type (using power data directly for visualization)
            st.subheader("📊 Average Generation Power by Source Type")
            st.caption("Showing average power output (MW) per source. Note: Data is instantaneous power, not energy.")
            source_summary = gen_df.groupby(['source_type', 'is_renewable'])['generation_mw'].mean().reset_index()
            source_summary = source_summary.sort_values('generation_mw', ascending=False)
            
            fig3 = px.bar(
                source_summary,
                x='source_type',
                y='generation_mw',
                color='is_renewable',
                color_discrete_map={True: 'green', False: 'gray'},
                labels={'generation_mw': 'Average Power (MW)', 'source_type': 'Source Type', 'is_renewable': 'Renewable'},
                title="Average Generation Power by Source Type"
            )
            fig3.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Installed Capacity by Source Type
            installed_capacity_df = None
            if 'fetched_data' in st.session_state:
                if 'installed_capacity' in st.session_state['fetched_data']:
                    installed_capacity_df = st.session_state['fetched_data']['installed_capacity']
            
            if installed_capacity_df is not None and not installed_capacity_df.empty:
                # Filter out rows with None psr_type (shouldn't happen after our fix, but just in case)
                capacity_df = installed_capacity_df[installed_capacity_df['psr_type'].notna()].copy()
                
                if not capacity_df.empty:
                    st.subheader("🏭 Installed Capacity by Source Type")
                    st.caption("Showing installed generation capacity (MW) per source type")
                    
                    # Add readable names and renewable flag
                    capacity_df['source_type'] = capacity_df['psr_type'].map(PSR_TYPE_MAPPING)
                    capacity_df['is_renewable'] = capacity_df['psr_type'].isin(RENEWABLE_PSR_TYPES)
                    
                    # Filter out rows where source_type mapping failed (None)
                    capacity_df = capacity_df[capacity_df['source_type'].notna()]
                    
                    if not capacity_df.empty:
                        # Group by source type and calculate average installed capacity
                        capacity_summary = capacity_df.groupby(['source_type', 'is_renewable'])['installed_capacity_mw'].mean().reset_index()
                        capacity_summary = capacity_summary.sort_values('installed_capacity_mw', ascending=False)
                        
                        fig_capacity = px.bar(
                            capacity_summary,
                            x='source_type',
                            y='installed_capacity_mw',
                            color='is_renewable',
                            color_discrete_map={True: 'green', False: 'gray'},
                            labels={'installed_capacity_mw': 'Installed Capacity (MW)', 'source_type': 'Source Type', 'is_renewable': 'Renewable'},
                            title="Installed Capacity by Source Type"
                        )
                        fig_capacity.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_capacity, use_container_width=True)
                    else:
                        st.warning("⚠️ Installed capacity data found but no valid PSR types. Please check the data.")
                else:
                    st.info("ℹ️ Installed capacity data is empty or contains no valid PSR types.")
            else:
                st.info("ℹ️ No installed capacity data available. Please fetch data to see this plot.")
            
            # Generation over time (aggregated)
            st.subheader("📈 Generation Power Over Time")
            st.caption("Showing instantaneous power (MW) over time")
            hourly_gen = gen_df.groupby(['timestamp', 'is_renewable'])['generation_mw'].sum().reset_index()
            
            fig4 = px.area(
                hourly_gen,
                x='timestamp',
                y='generation_mw',
                color='is_renewable',
                color_discrete_map={True: 'green', False: 'gray'},
                labels={'generation_mw': 'Power (MW)', 'timestamp': 'Date & Time', 'is_renewable': 'Renewable'}
            )
            fig4.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig4, use_container_width=True)
            
            # Scaled Production Profile
            st.subheader("📊 Scaled Production Profile")
            st.caption("Production profile scaled by PPA Capacity / Total Installed Capacity. Based on sidebar configuration.")
            
            # Get PPA configuration from session state (sidebar selections)
            ppa_capacity_mw = st.session_state.get('ppa_capacity_mw', 1.5)
            ppa_technology = st.session_state.get('ppa_technology', None)
            solar_fraction = st.session_state.get('solar_fraction', None)
            wind_fraction = st.session_state.get('wind_fraction', None)
            
            if ppa_technology is None:
                st.warning("⚠️ No PPA technology selected. Please configure PPA settings in the sidebar and fetch data.")
            else:
                # Get installed capacity data
                installed_capacity_df = None
                if 'fetched_data' in st.session_state and 'installed_capacity' in st.session_state['fetched_data']:
                    installed_capacity_df = st.session_state['fetched_data']['installed_capacity']
                
                if installed_capacity_df is None or installed_capacity_df.empty:
                    st.warning("⚠️ No installed capacity data available. Please fetch data with capacity information.")
                else:
                    # Check if combined technology
                    is_combined = '+' in ppa_technology
                    
                    if is_combined:
                        # Combined technology - show both components
                        tech_components = ppa_technology.split(' + ')
                        tech_components = [t.strip() for t in tech_components]
                        
                        # Get fractions (default 50/50)
                        if solar_fraction is None:
                            fractions = [0.5, 0.5]
                        else:
                            fractions = [solar_fraction, wind_fraction]
                        
                        # Calculate production for each component
                        fig5 = go.Figure()
                        
                        colors = ['#FFA500', '#4169E1']  # Orange for solar, blue for wind
                        total_production_by_time = None
                        
                        for idx, (tech, fraction) in enumerate(zip(tech_components, fractions)):
                            tech_capacity = ppa_capacity_mw * fraction
                            
                            # Get PSR types
                            psr_mapping = {
                                'Solar': ['B16'],
                                'Wind Onshore': ['B19'],
                                'Wind Offshore': ['B18']
                            }
                            psr_types = psr_mapping.get(tech, ['B16'])
                            
                            # Filter and group generation data
                            tech_gen = gen_df[gen_df['psr_type'].isin(psr_types)].copy()
                            
                            if not tech_gen.empty:
                                tech_by_time = tech_gen.groupby('timestamp')['generation_mw'].sum().reset_index()
                                
                                # Get installed capacity
                                tech_cap = installed_capacity_df[
                                    (installed_capacity_df['psr_type'].notna()) & 
                                    (installed_capacity_df['psr_type'].isin(psr_types))
                                ].copy()
                                
                                if not tech_cap.empty:
                                    installed_cap = tech_cap['installed_capacity_mw'].mean()
                                    
                                    if installed_cap > 0:
                                        scaling_factor = tech_capacity / installed_cap
                                        tech_by_time['scaled_production_mw'] = tech_by_time['generation_mw'] * scaling_factor
                                        
                                        # Add to plot
                                        fig5.add_trace(go.Scatter(
                                            x=tech_by_time['timestamp'],
                                            y=tech_by_time['scaled_production_mw'],
                                            mode='lines',
                                            name=f'{tech} ({fraction*100:.0f}%)',
                                            line=dict(color=colors[idx], width=2),
                                            stackgroup='ppa',
                                            hovertemplate=f'<b>%{{x}}</b><br>{tech}: %{{y:.2f}} MW<extra></extra>'
                                        ))
                                        
                                        # Track total production
                                        if total_production_by_time is None:
                                            total_production_by_time = tech_by_time[['timestamp']].copy()
                                            total_production_by_time['total_mw'] = tech_by_time['scaled_production_mw']
                                        else:
                                            total_production_by_time = total_production_by_time.merge(
                                                tech_by_time[['timestamp', 'scaled_production_mw']],
                                                on='timestamp',
                                                how='outer',
                                                suffixes=('', '_new')
                                            )
                                            total_production_by_time['total_mw'] = total_production_by_time['total_mw'].fillna(0) + total_production_by_time['scaled_production_mw'].fillna(0)
                                            total_production_by_time = total_production_by_time[['timestamp', 'total_mw']]
                        
                        if total_production_by_time is not None:
                            # Add mean line for total production
                            mean_total = total_production_by_time['total_mw'].mean()
                            fig5.add_hline(
                                y=mean_total,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Mean Total: {mean_total:.2f} MW"
                            )
                            
                            fig5.update_layout(
                                xaxis_title="Date & Time",
                                yaxis_title="Power (MW)",
                                hovermode='x unified',
                                height=500,
                                showlegend=True,
                                title=f"{ppa_technology} Production Profile (Total: {ppa_capacity_mw:.2f} MW PPA)"
                            )
                            
                            st.plotly_chart(fig5, use_container_width=True)
                            
                            # Statistics
                            st.markdown("**Portfolio Composition:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{tech_components[0]} Capacity", f"{ppa_capacity_mw * fractions[0]:.2f} MW ({fractions[0]*100:.0f}%)")
                            with col2:
                                st.metric(f"{tech_components[1]} Capacity", f"{ppa_capacity_mw * fractions[1]:.2f} MW ({fractions[1]*100:.0f}%)")
                            with col3:
                                st.metric("Mean Total Production", f"{mean_total:.2f} MW")
                        else:
                            st.warning("⚠️ Could not calculate production for any technology component")
                    
                    else:
                        # Single technology
                        psr_types = PPA_TECHNOLOGY_PSR_TYPES.get(ppa_technology, ['B16'])
                        
                        # Filter generation data for the selected technology
                        tech_generation = gen_df[gen_df['psr_type'].isin(psr_types)].copy()
                        
                        if tech_generation.empty:
                            st.warning(f"⚠️ No generation data available for {ppa_technology} in this country/period")
                        else:
                            # Group by timestamp to get total generation for the technology
                            tech_by_time = tech_generation.groupby('timestamp').agg({
                                'generation_mw': 'sum'
                            }).reset_index()
                            
                            # Filter installed capacity for this technology
                            tech_capacity = installed_capacity_df[
                                (installed_capacity_df['psr_type'].notna()) & 
                                (installed_capacity_df['psr_type'].isin(psr_types))
                            ].copy()
                            
                            if not tech_capacity.empty:
                                total_installed_capacity_mw = tech_capacity['installed_capacity_mw'].mean()
                                
                                if total_installed_capacity_mw > 0:
                                    # Calculate scaling factor
                                    scaling_factor = ppa_capacity_mw / total_installed_capacity_mw
                                    
                                    # Scale the production profile
                                    tech_by_time['scaled_production_mw'] = tech_by_time['generation_mw'] * scaling_factor
                                    
                                    # Display info
                                    st.info(f"""
                                    **Scaling Information:**
                                    - Technology: {ppa_technology}
                                    - Total Installed Capacity ({ppa_technology}): {total_installed_capacity_mw:.2f} MW
                                    - PPA Capacity (from sidebar): {ppa_capacity_mw:.2f} MW
                                    - Scaling Factor: {scaling_factor:.4f}
                                    """)
                                    
                                    # Create plot
                                    fig5 = go.Figure()
                                    
                                    # Original generation
                                    fig5.add_trace(go.Scatter(
                                        x=tech_by_time['timestamp'],
                                        y=tech_by_time['generation_mw'],
                                        mode='lines',
                                        name=f'Original {ppa_technology} Generation',
                                        line=dict(color='lightblue', width=1),
                                        opacity=0.5,
                                        hovertemplate='<b>%{x}</b><br>Original: %{y:.2f} MW<extra></extra>'
                                    ))
                                    
                                    # Scaled production
                                    fig5.add_trace(go.Scatter(
                                        x=tech_by_time['timestamp'],
                                        y=tech_by_time['scaled_production_mw'],
                                        mode='lines',
                                        name='Scaled PPA Production',
                                        line=dict(color='green', width=2),
                                        fill='tozeroy',
                                        fillcolor='rgba(0, 255, 0, 0.2)',
                                        hovertemplate='<b>%{x}</b><br>Scaled: %{y:.2f} MW<extra></extra>'
                                    ))
                                    
                                    # Add mean line for scaled production
                                    mean_scaled = tech_by_time['scaled_production_mw'].mean()
                                    fig5.add_hline(
                                        y=mean_scaled,
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Mean: {mean_scaled:.2f} MW"
                                    )
                                    
                                    fig5.update_layout(
                                        xaxis_title="Date & Time",
                                        yaxis_title="Power (MW)",
                                        hovermode='x unified',
                                        height=400,
                                        showlegend=True,
                                        title=f"{ppa_technology} Production Profile (Scaled to {ppa_capacity_mw:.2f} MW PPA)"
                                    )
                                    
                                    st.plotly_chart(fig5, use_container_width=True)
                                    
                                    # Statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Mean Scaled Production", f"{mean_scaled:.2f} MW")
                                    with col2:
                                        st.metric("Max Scaled Production", f"{tech_by_time['scaled_production_mw'].max():.2f} MW")
                                    with col3:
                                        st.metric("Min Scaled Production", f"{tech_by_time['scaled_production_mw'].min():.2f} MW")
                                    with col4:
                                        avg_cf = (tech_by_time['generation_mw'] / total_installed_capacity_mw).mean()
                                        st.metric("Avg Capacity Factor", f"{avg_cf*100:.1f}%")
                                else:
                                    st.warning("⚠️ Installed capacity is zero, cannot calculate scaling factor")
                            else:
                                st.warning(f"⚠️ No installed capacity data for {ppa_technology}")
            
            # Detailed data table
            st.subheader("📋 Detailed Generation Data")
            
            # Display options
            col1, col2 = st.columns([1, 3])
            with col1:
                show_all_gen = st.checkbox("Show all generation data", value=False)
            with col2:
                if not show_all_gen:
                    st.info("Showing first 100 rows. Check 'Show all data' to see everything.")
            
            # Prepare display dataframe
            display_gen_df = gen_df[['timestamp', 'zone', 'source_type', 'generation_mw', 'is_renewable']].copy()
            display_gen_df['timestamp'] = display_gen_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_gen_df = display_gen_df.rename(columns={'generation_mw': 'power_mw'})
            
            if not show_all_gen:
                display_gen_df = display_gen_df.head(100)
            
            st.caption("Note: Values shown are instantaneous power (MW), not energy (MWh)")
            st.dataframe(
                display_gen_df.style.format({
                    'power_mw': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_gen = gen_df[['timestamp', 'zone', 'psr_type', 'source_type', 'generation_mw', 'resolution_minutes', 'is_renewable']].to_csv(index=False)
            st.download_button(
                label="📥 Download Generation Data as CSV",
                data=csv_gen,
                file_name=f"entsoe_generation_{st.session_state.get('country', 'data')}.csv",
                mime="text/csv"
            )
            st.caption("CSV contains: power (MW), resolution (minutes), and renewable flag. Use trapezoidal integration to calculate energy.")

def run_sensitivity_analysis(data, country, temporal_correlation, ratios):
    """
    Run sensitivity analysis for different PPA capacity ratios and technologies.
    
    Args:
        data: Dictionary with prices, generation, and installed_capacity DataFrames
        country: Country name
        temporal_correlation: 'hourly' or 'monthly'
        ratios: List of PPA capacity to electrolyser capacity ratios
    
    Returns:
        Dictionary with results for each technology
    """
    electrolyser_mw = 1.0  # Fixed at 1 MW
    backend_country = get_backend_country_name(country)
    
    # Calculate renewable share once
    if data['generation'].empty:
        renewable_share = 0.30
    else:
        renewable_share = calculate_renewable_share(data['generation'], 'annual')
    
    # Technologies to analyze
    technologies = {
        'Solar': ['Solar'],
        'Wind Onshore': ['Wind Onshore'],
        'Wind Offshore': ['Wind Offshore'],
        'Solar + Wind Offshore': ['Solar', 'Wind Offshore'],
        'Solar + Wind Onshore': ['Solar', 'Wind Onshore']
    }
    
    results = {}
    
    for tech_name, tech_list in technologies.items():
        results[tech_name] = []
        
        for ratio in ratios:
            ppa_capacity_mw = electrolyser_mw * ratio
            
            try:
                # For combined technologies, split capacity 50/50
                if len(tech_list) == 2:
                    # Calculate production for each technology separately
                    combined_production = None
                    
                    for tech in tech_list:
                        tech_capacity = ppa_capacity_mw * 0.5  # 50% each
                        
                        ppa_prod_df = calculate_ppa_production_from_generation_data(
                            generation_df=data['generation'],
                            ppa_technology=tech,
                            ppa_capacity_mw=tech_capacity,
                            prices_df=data['prices'],
                            installed_capacity_df=data.get('installed_capacity', pd.DataFrame())
                        )
                        
                        if combined_production is None:
                            combined_production = ppa_prod_df.copy()
                        else:
                            # Add the production from the second technology
                            combined_production['ppa_production_mw'] += ppa_prod_df['ppa_production_mw']
                    
                    # Create a temporary DataFrame for calculation
                    calc_df = data['prices'].copy()
                    calc_df = calc_df.merge(combined_production, on='datetime', how='left')
                    calc_df['ppa_production_mw'] = calc_df['ppa_production_mw'].fillna(0)
                    
                    # Prepare calculation parameters
                    calc_params = {
                        'electrolyser_mw': electrolyser_mw,
                        'ppa_capacity_mw': ppa_capacity_mw,
                        'prices_df': calc_df,
                        'renewable_share': renewable_share,
                        'zone_name': backend_country,
                        'temporal_correlation': temporal_correlation,
                        'use_price_threshold': True,
                        'ppa_technology': None,  # We already calculated production
                        'generation_df': pd.DataFrame()  # Empty to use our pre-calculated production
                    }
                    
                    # Manually set PPA production in the dataframe
                    result = calculate_rfnbo_compliance(**calc_params)
                    # Override with our combined production
                    timestep_hours = result['resolution_minutes'] / 60 if 'resolution_minutes' in result.columns else 1.0
                    result['ppa_production_mw'] = calc_df['ppa_production_mw']
                    result['ppa_energy_mwh'] = result['ppa_production_mw'] * timestep_hours
                    result['grid_consumption_mw'] = (result['electrolyser_consumption_mw'] - result['ppa_production_mw']).clip(lower=0)
                    result['grid_energy_mwh'] = result['grid_consumption_mw'] * timestep_hours
                    
                    # Recalculate RFNBO metrics
                    result['rfnbo_from_ppa_mwh'] = result['ppa_energy_mwh']
                    if isinstance(renewable_share, pd.DataFrame):
                        result = result.merge(renewable_share, left_on='datetime', right_on='timestamp', how='left', suffixes=('', '_renewable'))
                        result['grid_renewable_share_mix'] = result['renewable_share'].fillna(renewable_share['renewable_share'].mean())
                    else:
                        result['grid_renewable_share_mix'] = renewable_share
                    result['rfnbo_from_grid_mwh'] = result['grid_energy_mwh'] * result['grid_renewable_share_mix']
                    result['rfnbo_energy_mwh'] = result['rfnbo_from_ppa_mwh'] + result['rfnbo_from_grid_mwh']
                    result['rfnbo_fraction'] = result['rfnbo_energy_mwh'] / result['electrolyser_consumption_mwh']
                    result['rfnbo_fraction'] = result['rfnbo_fraction'].clip(upper=1.0)
                    
                else:
                    # Single technology
                    calc_params = {
                        'electrolyser_mw': electrolyser_mw,
                        'ppa_capacity_mw': ppa_capacity_mw,
                        'prices_df': data['prices'],
                        'renewable_share': renewable_share,
                        'zone_name': backend_country,
                        'temporal_correlation': temporal_correlation,
                        'use_price_threshold': True,
                        'ppa_technology': tech_list[0],
                        'generation_df': data['generation'],
                        'installed_capacity_df': data.get('installed_capacity', pd.DataFrame())
                    }
                    
                    result = calculate_rfnbo_compliance(**calc_params)
                
                # Aggregate monthly
                monthly = aggregate_to_monthly(result)
                rfnbo_fraction = monthly['rfnbo_fraction'].values[0]
                
                results[tech_name].append({
                    'ratio': ratio,
                    'rfnbo_fraction': rfnbo_fraction
                })
                
            except Exception as e:
                logger.error(f"Error calculating for {tech_name} at ratio {ratio}: {str(e)}")
                results[tech_name].append({
                    'ratio': ratio,
                    'rfnbo_fraction': 0
                })
    
    return results

def display_sensitivity_analysis_ppa_sizing():
    """Display the PPA sizing sensitivity analysis tab."""
    st.header("📈 Sensitivity Analysis: PPA Sizing Impact on RFNBO Compliance")
    st.markdown("""
    This analysis shows how different PPA capacity ratios affect RFNBO compliance for various technologies.
    - **X-axis**: Production to consumption ratio (PPA capacity / Electrolyser capacity)
    - **Y-axis**: % of RFNBO H₂ (renewable hydrogen fraction)
    - **Electrolyser**: Fixed at 1 MW
    - **Combined technologies**: 50%/50% split between the two sources
    """)
    
    # Check if data exists
    if 'fetched_data' not in st.session_state:
        st.info("👈 Please fetch data using the sidebar first (🚀 Fetch Data & Calculate button)")
        return
    
    data = st.session_state['fetched_data']
    
    if data['prices'].empty or data['generation'].empty:
        st.warning("⚠️ Complete price and generation data is required for sensitivity analysis")
        return
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_ratio = st.number_input("Min Ratio", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="ppa_sizing_min_ratio")
        max_ratio = st.number_input("Max Ratio", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="ppa_sizing_max_ratio")
    
    with col2:
        num_points = st.slider("Number of Points", min_value=10, max_value=100, value=50, key="ppa_sizing_num_points")
        temporal_correlation = st.selectbox("Temporal Correlation", ['hourly', 'monthly'], 
                                           index=0 if st.session_state.get('temporal_correlation') == 'hourly' else 1,
                                           key="ppa_sizing_temporal_correlation")
    
    if st.button("🚀 Run Sensitivity Analysis", type="primary", key="ppa_sizing_run_button"):
        # Generate ratios
        ratios = np.linspace(min_ratio, max_ratio, num_points)
        
        with st.spinner("Running sensitivity analysis... This may take a few minutes."):
            country = st.session_state.get('country', 'Belgium')
            
            results = run_sensitivity_analysis(data, country, temporal_correlation, ratios)
            
            # Store in session state
            st.session_state['sensitivity_results'] = results
            st.session_state['sensitivity_ratios'] = ratios
    
    # Display results if available
    if 'sensitivity_results' in st.session_state:
        st.subheader("📊 Results")
        
        results = st.session_state['sensitivity_results']
        
        # Create the plot
        fig = go.Figure()
        
        colors = {
            'Solar': '#FFA500',
            'Wind Onshore': '#8B4513',
            'Wind Offshore': '#DC143C',
            'Solar + Wind Offshore': '#4169E1',
            'Solar + Wind Onshore': '#228B22'
        }
        
        for tech_name, data_points in results.items():
            if data_points:
                x_vals = [d['ratio'] for d in data_points]
                y_vals = [d['rfnbo_fraction'] * 100 for d in data_points]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=tech_name,
                    line=dict(color=colors.get(tech_name, '#000000'), width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>Ratio: %{x:.2f}<br>RFNBO: %{y:.1f}%<extra></extra>'
                ))
        
        # Add 100% line
        fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                     annotation_text="100% RFNBO Target",
                     annotation_position="right")
        
        fig.update_layout(
            xaxis_title="Production to Consumption Ratio (PPA Capacity / Electrolyser Capacity)",
            yaxis_title="% RFNBO H₂",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(range=[0, 105])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("📋 Summary: Ratio to Reach 100% RFNBO")
        
        summary_data = []
        for tech_name, data_points in results.items():
            if data_points:
                # Find the ratio where RFNBO >= 100%
                reaching_100 = [d for d in data_points if d['rfnbo_fraction'] >= 1.0]
                if reaching_100:
                    min_ratio_100 = min(d['ratio'] for d in reaching_100)
                    summary_data.append({
                        'Technology': tech_name,
                        'Min Ratio for 100% RFNBO': f"{min_ratio_100:.2f}",
                        'PPA Capacity Needed (MW)': f"{min_ratio_100 * 1.0:.2f}"  # Electrolyser is 1 MW
                    })
                else:
                    max_rfnbo = max(d['rfnbo_fraction'] for d in data_points) * 100
                    summary_data.append({
                        'Technology': tech_name,
                        'Min Ratio for 100% RFNBO': f">{max_ratio:.2f} (max: {max_rfnbo:.1f}%)",
                        'PPA Capacity Needed (MW)': f">{max_ratio:.2f}"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Download button
        if results:
            # Create CSV data
            csv_data = []
            for tech_name, data_points in results.items():
                for d in data_points:
                    csv_data.append({
                        'Technology': tech_name,
                        'Ratio': d['ratio'],
                        'RFNBO_Fraction': d['rfnbo_fraction'],
                        'RFNBO_Percentage': d['rfnbo_fraction'] * 100
                    })
            
            csv_df = pd.DataFrame(csv_data)
            csv = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sensitivity Analysis Results",
                data=csv,
                file_name=f"sensitivity_analysis_ppa_sizing_{st.session_state.get('country', 'data')}_{st.session_state.get('year', '')}_{st.session_state.get('month', ''):02d}.csv",
                mime="text/csv"
            )

def display_sensitivity_analysis_solar_wind_split():
    """Display the Solar/Wind split sensitivity analysis tab."""
    st.header("🌤️ Sensitivity Analysis: Solar/Wind Portfolio Optimization")
    st.markdown("""
    This analysis shows how different Solar/Wind splits affect RFNBO compliance for combined portfolios.
    - **X-axis**: Solar fraction (0 = 100% Wind, 1 = 100% Solar)
    - **Y-axis**: % of RFNBO H₂ (renewable hydrogen fraction)
    - **Electrolyser**: Fixed at 1 MW
    - **Three scenarios**: PPA to Electrolyser ratios of 0.75, 1.0, and 1.25
    """)
    
    # Check if data exists
    if 'fetched_data' not in st.session_state:
        st.info("👈 Please fetch data using the sidebar first (🚀 Fetch Data & Calculate button)")
        return
    
    data = st.session_state['fetched_data']
    
    if data['prices'].empty or data['generation'].empty:
        st.warning("⚠️ Complete price and generation data is required for sensitivity analysis")
        return
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        combined_technology = st.selectbox(
            "Combined Technology Portfolio",
            ['Solar + Wind Offshore', 'Solar + Wind Onshore'],
            help="Select which combined technology to analyze",
            key="solar_wind_split_technology"
        )
    
    with col2:
        num_points = st.slider("Number of Split Points", min_value=10, max_value=50, value=21,
                              help="Number of points between 0% and 100% solar",
                              key="solar_wind_split_num_points")
        temporal_correlation = st.selectbox("Temporal Correlation", ['hourly', 'monthly'], 
                                           index=0 if st.session_state.get('temporal_correlation') == 'hourly' else 1,
                                           key="solar_wind_split_temporal_correlation")
    
    # Fixed ratios to analyze
    ppa_ratios = [0.75, 1.0, 1.25]
    
    st.info(f"📊 Will analyze {num_points} different Solar/Wind splits for each of the 3 PPA ratios: {ppa_ratios}")
    
    if st.button("🚀 Run Solar/Wind Split Analysis", type="primary", key="solar_wind_split_run_button"):
        # Generate solar fractions from 0 to 1
        solar_fractions = np.linspace(0, 1, num_points)
        
        with st.spinner("Running Solar/Wind split analysis... This may take several minutes."):
            country = st.session_state.get('country', 'Belgium')
            
            results = run_solar_wind_split_analysis(
                data, country, temporal_correlation, 
                combined_technology, solar_fractions, ppa_ratios
            )
            
            # Store in session state
            st.session_state['solar_wind_split_results'] = results
            st.session_state['solar_wind_split_fractions'] = solar_fractions
            st.session_state['solar_wind_split_ratios'] = ppa_ratios
            st.session_state['solar_wind_split_tech'] = combined_technology
    
    # Display results if available
    if 'solar_wind_split_results' in st.session_state:
        st.subheader("📊 Results")
        
        results = st.session_state['solar_wind_split_results']
        solar_fractions = st.session_state['solar_wind_split_fractions']
        ppa_ratios = st.session_state['solar_wind_split_ratios']
        tech_name = st.session_state['solar_wind_split_tech']
        
        # Create the plot
        fig = go.Figure()
        
        colors = {
            0.75: '#FF6B6B',  # Red
            1.0: '#4ECDC4',   # Teal
            1.25: '#45B7D1'   # Blue
        }
        
        for ratio in ppa_ratios:
            if ratio in results:
                data_points = results[ratio]
                x_vals = [d['solar_fraction'] * 100 for d in data_points]  # Convert to percentage
                y_vals = [d['rfnbo_fraction'] * 100 for d in data_points]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=f'PPA/Electrolyser = {ratio}',
                    line=dict(color=colors.get(ratio, '#000000'), width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Ratio: %{fullData.name}</b><br>Solar: %{x:.0f}%<br>RFNBO: %{y:.1f}%<extra></extra>'
                ))
        
        # Add 100% RFNBO line
        fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                     annotation_text="100% RFNBO Target",
                     annotation_position="right")
        
        fig.update_layout(
            xaxis_title="Solar Fraction (%)",
            yaxis_title="% RFNBO H₂",
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(range=[0, 105])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis: Show which solar fraction is optimal for each ratio
        st.subheader("🎯 Optimal Solar Fractions")
        
        optimal_data = []
        for ratio in ppa_ratios:
            if ratio in results:
                data_points = results[ratio]
                # Find the solar fraction that gets closest to 100% RFNBO without going under
                compliant_points = [d for d in data_points if d['rfnbo_fraction'] >= 1.0]
                
                if compliant_points:
                    # Find the point with minimum solar fraction that still achieves 100%
                    optimal_point = min(compliant_points, key=lambda x: x['solar_fraction'])
                    optimal_data.append({
                        'PPA/Electrolyser Ratio': f"{ratio}",
                        'Optimal Solar Fraction': f"{optimal_point['solar_fraction']*100:.1f}%",
                        'RFNBO Achieved': f"{optimal_point['rfnbo_fraction']*100:.1f}%",
                        'Status': '✅ 100% RFNBO Achievable'
                    })
                else:
                    # Find the maximum RFNBO achieved
                    max_point = max(data_points, key=lambda x: x['rfnbo_fraction'])
                    optimal_data.append({
                        'PPA/Electrolyser Ratio': f"{ratio}",
                        'Optimal Solar Fraction': f"{max_point['solar_fraction']*100:.1f}%",
                        'RFNBO Achieved': f"{max_point['rfnbo_fraction']*100:.1f}%",
                        'Status': f"❌ Max {max_point['rfnbo_fraction']*100:.1f}%"
                    })
        
        if optimal_data:
            optimal_df = pd.DataFrame(optimal_data)
            st.dataframe(optimal_df, use_container_width=True, hide_index=True)
        
        # Summary insights
        st.subheader("💡 Key Insights")
        
        with st.expander("View Detailed Analysis", expanded=True):
            st.markdown(f"""
            **Portfolio Analysis for {tech_name}:**
            
            This analysis helps you understand the trade-offs between solar and wind in your renewable portfolio:
            
            - **100% Wind (0% Solar)**: Typically more stable output, but may have lower capacity factors
            - **100% Solar (100% Solar)**: Higher daytime peaks, but zero production at night
            - **Mixed Portfolio**: Combines complementary generation patterns for more consistent supply
            
            **Observations from your results:**
            - Compare the curves to see which ratio (0.75, 1.0, or 1.25) reaches 100% RFNBO
            - The optimal solar/wind split may differ depending on your PPA sizing
            - A larger PPA capacity (ratio 1.25) gives more flexibility in portfolio composition
            """)
        
        # Download button
        if results:
            # Create CSV data
            csv_data = []
            for ratio in ppa_ratios:
                if ratio in results:
                    for d in results[ratio]:
                        csv_data.append({
                            'Technology': tech_name,
                            'PPA_to_Electrolyser_Ratio': ratio,
                            'Solar_Fraction': d['solar_fraction'],
                            'Solar_Percentage': d['solar_fraction'] * 100,
                            'Wind_Fraction': 1 - d['solar_fraction'],
                            'Wind_Percentage': (1 - d['solar_fraction']) * 100,
                            'RFNBO_Fraction': d['rfnbo_fraction'],
                            'RFNBO_Percentage': d['rfnbo_fraction'] * 100
                        })
            
            csv_df = pd.DataFrame(csv_data)
            csv = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Solar/Wind Split Analysis Results",
                data=csv,
                file_name=f"sensitivity_analysis_solar_wind_{st.session_state.get('country', 'data')}_{st.session_state.get('year', '')}_{st.session_state.get('month', ''):02d}.csv",
                mime="text/csv"
            )

def run_solar_wind_split_analysis(data, country, temporal_correlation, combined_technology, solar_fractions, ppa_ratios):
    """
    Run sensitivity analysis for different Solar/Wind splits at multiple PPA capacity ratios.
    
    Args:
        data: Dictionary with prices, generation, and installed_capacity DataFrames
        country: Country name
        temporal_correlation: 'hourly' or 'monthly'
        combined_technology: 'Solar + Wind Offshore' or 'Solar + Wind Onshore'
        solar_fractions: Array of solar fractions to test (0 to 1)
        ppa_ratios: List of PPA to electrolyser capacity ratios (e.g., [0.75, 1.0, 1.25])
    
    Returns:
        Dictionary with results for each ratio
    """
    electrolyser_mw = 1.0  # Fixed at 1 MW
    backend_country = get_backend_country_name(country)
    
    # Calculate renewable share once
    if data['generation'].empty:
        renewable_share = 0.30
    else:
        renewable_share = calculate_renewable_share(data['generation'], 'annual')
    
    results = {}
    
    for ratio in ppa_ratios:
        results[ratio] = []
        ppa_capacity_mw = electrolyser_mw * ratio
        
        for solar_fraction in solar_fractions:
            wind_fraction = 1 - solar_fraction
            
            try:
                calc_params = {
                    'electrolyser_mw': electrolyser_mw,
                    'ppa_capacity_mw': ppa_capacity_mw,
                    'prices_df': data['prices'],
                    'renewable_share': renewable_share,
                    'zone_name': backend_country,
                    'temporal_correlation': temporal_correlation,
                    'use_price_threshold': True,
                    'ppa_technology': combined_technology,
                    'generation_df': data['generation'],
                    'installed_capacity_df': data.get('installed_capacity', pd.DataFrame()),
                    'solar_fraction': solar_fraction,
                    'wind_fraction': wind_fraction
                }
                
                result = calculate_rfnbo_compliance(**calc_params)
                
                # Aggregate monthly
                monthly = aggregate_to_monthly(result)
                rfnbo_fraction = monthly['rfnbo_fraction'].values[0]
                
                results[ratio].append({
                    'solar_fraction': solar_fraction,
                    'wind_fraction': wind_fraction,
                    'rfnbo_fraction': rfnbo_fraction
                })
                
            except Exception as e:
                logger.error(f"Error calculating for ratio {ratio}, solar fraction {solar_fraction}: {str(e)}")
                results[ratio].append({
                    'solar_fraction': solar_fraction,
                    'wind_fraction': wind_fraction,
                    'rfnbo_fraction': 0
                })
    
    return results

def display_sensitivity_analysis_tab():
    """Display the main sensitivity analysis tab with sub-tabs."""
    st.header("📈 Sensitivity Analysis")
    
    # Create sub-tabs for different sensitivity analyses
    sens_tab1, sens_tab2 = st.tabs([
        "📊 PPA Sizing Analysis", 
        "🌤️ Solar/Wind Split Analysis"
    ])
    
    with sens_tab1:
        display_sensitivity_analysis_ppa_sizing()
    
    with sens_tab2:
        display_sensitivity_analysis_solar_wind_split()

def main():
    st.title("⚡ RFNBO Compliancy Calculator for Electrolysers")
    st.markdown("""
    This application calculates the RFNBO (Renewable Fuel of Non-Biological Origin) compliance 
    of an electrolyser based on its energy consumption profile and renewable energy sources.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Country selection
    country_options = list(set([k for k in BIDDING_ZONES.keys() if not any(x in k for x in ['Germany 50', 'Germany Amprion', 'Germany TenneT', 'Germany TransnetBW'])]))
    country_options.sort()
    country = st.sidebar.selectbox("Country", country_options, index=country_options.index('Belgium') if 'Belgium' in country_options else 0)
    
    # Time period
    st.sidebar.subheader("📅 Time Period")
    year = st.sidebar.number_input("Year", min_value=2015, max_value=2024, value=2023)
    month = st.sidebar.selectbox("Month", range(1, 13), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    
    # Electrolyser configuration
    st.sidebar.subheader("🔋 Electrolyser Configuration")
    electrolyser_mw = st.sidebar.number_input("Electrolyser Capacity (MW)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
    
    # PPA configuration
    st.sidebar.subheader("🌱 PPA Configuration")
    ppa_capacity_mw = st.sidebar.number_input("PPA Capacity (MW)", min_value=0.0, max_value=10000.0, value=1.5, step=0.1)
    
    ppa_technology = st.sidebar.selectbox(
        "PPA Technology",
        ['Solar', 'Wind Onshore', 'Wind Offshore', 'Solar + Wind Offshore', 'Solar + Wind Onshore'],
        index=0
    )
    
    # Portfolio allocation for combined technologies
    if '+' in ppa_technology:
        st.sidebar.caption("Combined PPA portfolio configuration:")
        portfolio_split = st.sidebar.slider(
            "Solar / Wind Split (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Percentage of PPA capacity allocated to Solar (remainder goes to Wind)"
        )
        solar_fraction = portfolio_split / 100
        wind_fraction = 1 - solar_fraction
        st.sidebar.caption(f"Solar: {solar_fraction*100:.0f}% | Wind: {wind_fraction*100:.0f}%")
    else:
        solar_fraction = None
        wind_fraction = None
    
    st.sidebar.caption("Will use actual generation data from ENTSOE for this technology")
    
    # Temporal correlation
    st.sidebar.subheader("⏱️ Temporal Correlation")
    temporal_correlation = st.sidebar.radio("Correlation Type", ['hourly', 'monthly'])
    
    # Fetch data button
    if st.sidebar.button("🚀 Fetch Data & Calculate", type="primary"):
        with st.spinner("Fetching data from ENTSOE..."):
            # Always fetch installed capacity (useful for plots, and only fetched once on first day)
            fetch_capacity = True  # Always fetch for visualization purposes
            data = fetch_month_data(country, year, month, fetch_capacity=fetch_capacity)
        
        # Store data in session state for access from both tabs
        st.session_state['fetched_data'] = data
        st.session_state['country'] = country
        st.session_state['year'] = year
        st.session_state['month'] = month
        st.session_state['ppa_capacity_mw'] = ppa_capacity_mw
        st.session_state['ppa_technology'] = ppa_technology
        st.session_state['solar_fraction'] = solar_fraction if '+' in ppa_technology else None
        st.session_state['wind_fraction'] = wind_fraction if '+' in ppa_technology else None
        
        if data['prices'].empty:
            st.error("❌ Unable to proceed without price data")
            return
        
        # Calculate renewable share from ENTSOE (always use 'annual' mode for monthly average)
        if data['generation'].empty:
            st.warning("⚠️ No generation data available, using default 30% renewable share")
            renewable_share = 0.30
        else:
            renewable_share = calculate_renewable_share(data['generation'], 'annual')
            if isinstance(renewable_share, float):
                st.info(f"ℹ️ Calculated monthly renewable share: {renewable_share * 100:.1f}%")
            else:
                avg_renewable = renewable_share['renewable_share'].mean()
                st.info(f"ℹ️ Calculated average renewable share: {avg_renewable * 100:.1f}%")
        
        # Calculate RFNBO compliance
        with st.spinner("Calculating RFNBO compliance..."):
            # Get backend country name for emission factor lookup
            backend_country = get_backend_country_name(country)
            
            # Prepare parameters for calculation
            calc_params = {
                'electrolyser_mw': electrolyser_mw,
                'ppa_capacity_mw': ppa_capacity_mw,
                'prices_df': data['prices'],
                'renewable_share': renewable_share,
                'zone_name': backend_country,
                'temporal_correlation': temporal_correlation,
                'use_price_threshold': True,  # Always use price threshold (20€/MWh rule)
                'ppa_technology': ppa_technology,
                'generation_df': data['generation'],
                'solar_fraction': solar_fraction,
                'wind_fraction': wind_fraction
            }
            
            # Add installed capacity data if available
            if 'installed_capacity' in data and not data['installed_capacity'].empty:
                calc_params['installed_capacity_df'] = data['installed_capacity']
                st.info(f"ℹ️ Using real {ppa_technology} generation data with actual installed capacity for PPA production")
            else:
                st.info(f"ℹ️ Using real {ppa_technology} generation data (estimated capacity) for PPA production")
            
            if data['generation'].empty:
                st.warning("⚠️ No generation data available, PPA production will be estimated")
            
            results = calculate_rfnbo_compliance(**calc_params)
        
        if results.empty:
            st.error("❌ Unable to calculate RFNBO compliance")
            return
        
        # Aggregate to monthly if needed
        if temporal_correlation == 'monthly':
            monthly_summary = aggregate_to_monthly(results)
        else:
            monthly_summary = aggregate_to_monthly(results)
        
        # Store results in session state
        st.session_state['results'] = results
        st.session_state['monthly_summary'] = monthly_summary
        st.session_state['renewable_share'] = renewable_share
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 RFNBO Analysis", "📊 ENTSOE Data Explorer", "📈 Sensitivity Analysis"])
    
    with tab1:
        # RFNBO Analysis Tab
        if 'results' in st.session_state and 'monthly_summary' in st.session_state:
            results = st.session_state['results']
            monthly_summary = st.session_state['monthly_summary']
            
            # Display results
            st.header("📊 RFNBO Compliance Results")
            
            # Get compliance status
            compliance = is_rfnbo_compliant(monthly_summary)
            
            # Overall compliance status
            if compliance['overall_compliant']:
                st.success("✅ RFNBO COMPLIANT - All requirements met!")
            else:
                st.error("❌ NOT RFNBO COMPLIANT")
            
            # Show individual checks
            col1, col2 = st.columns(2)
            with col1:
                if compliance['emission_compliant']:
                    st.success(f"✅ Emission Check: {compliance['emission_factor_mj']:.2f} < {MAX_EMISSION_FACTOR_MJ:.2f} g CO₂eq/MJ")
                else:
                    st.error(f"❌ Emission Check: {compliance['emission_factor_mj']:.2f} ≥ {MAX_EMISSION_FACTOR_MJ:.2f} g CO₂eq/MJ")
            
            with col2:
                rfnbo_pct = compliance['rfnbo_fraction'] * 100
                if compliance['rfnbo_compliant']:
                    st.success(f"✅ RFNBO Matching: {rfnbo_pct:.1f}% ≥ 100%")
                else:
                    st.warning(f"⚠️ RFNBO Matching: {rfnbo_pct:.1f}% < 100%")
            
            # Visualizations
            create_visualizations(results, monthly_summary)
            
            # Detailed hourly breakdown
            st.subheader("📊 Hourly Energy & Emissions Breakdown")
            
            with st.expander("📋 View Detailed Hourly Breakdown", expanded=False):
                st.caption("Detailed breakdown of energy sources and emissions for each hour")
                
                # Create a detailed breakdown dataframe
                hourly_breakdown = results[['datetime', 'price_eur_mwh', 
                                           'electrolyser_consumption_mwh', 'ppa_energy_mwh', 
                                           'grid_energy_mwh', 'grid_energy_low_price_mwh',
                                           'grid_energy_normal_price_mwh']].copy()
                
                # Calculate grid renewable and non-renewable parts
                if 'grid_renewable_share_mix' in results.columns:
                    hourly_breakdown['grid_renewable_energy_mwh'] = results['grid_energy_mwh'] * results['grid_renewable_share_mix']
                    hourly_breakdown['grid_non_renewable_energy_mwh'] = results['grid_energy_mwh'] * (1 - results['grid_renewable_share_mix'])
                else:
                    hourly_breakdown['grid_renewable_energy_mwh'] = 0
                    hourly_breakdown['grid_non_renewable_energy_mwh'] = results['grid_energy_mwh']
                
                # Add emissions and compliance
                hourly_breakdown['emissions_g_co2_eq'] = results['emissions_g_co2_eq']
                hourly_breakdown['emission_factor_mj'] = results['emission_factor_mj']
                hourly_breakdown['emission_compliant'] = results['is_emission_compliant']
                hourly_breakdown['rfnbo_fraction'] = results['rfnbo_fraction']
                hourly_breakdown['rfnbo_compliant'] = results['is_rfnbo_100pct']
                hourly_breakdown['overall_compliant'] = results['is_compliant']
                
                # Rename columns for clarity
                hourly_breakdown = hourly_breakdown.rename(columns={
                    'datetime': 'Date & Time',
                    'price_eur_mwh': 'Price (€/MWh)',
                    'electrolyser_consumption_mwh': 'Total Consumption (MWh)',
                    'ppa_energy_mwh': 'PPA Energy (MWh)',
                    'grid_energy_mwh': 'Grid Total (MWh)',
                    'grid_energy_low_price_mwh': 'Grid Low-Price <20€ (MWh)',
                    'grid_energy_normal_price_mwh': 'Grid Normal-Price ≥20€ (MWh)',
                    'grid_renewable_energy_mwh': 'Grid Renewable Part (MWh)',
                    'grid_non_renewable_energy_mwh': 'Grid Non-Renewable (MWh)',
                    'emissions_g_co2_eq': 'Emissions (g CO₂eq)',
                    'emission_factor_mj': 'EF (g CO₂eq/MJ)',
                    'emission_compliant': 'Emission ✓',
                    'rfnbo_fraction': 'RFNBO %',
                    'rfnbo_compliant': 'RFNBO ✓',
                    'overall_compliant': 'Overall ✓'
                })
                
                # Add option to show all or sample
                show_all_hourly = st.checkbox("Show all hours", value=False, key="show_all_hourly")
                
                if not show_all_hourly:
                    st.info("Showing first 100 hours. Check 'Show all hours' to see complete data.")
                    display_hourly = hourly_breakdown.head(100)
                else:
                    display_hourly = hourly_breakdown
                
                # Display with formatting
                st.dataframe(
                    display_hourly.style.format({
                        'Price (€/MWh)': '{:.2f}',
                        'Total Consumption (MWh)': '{:.3f}',
                        'PPA Energy (MWh)': '{:.3f}',
                        'Grid Total (MWh)': '{:.3f}',
                        'Grid Low-Price <20€ (MWh)': '{:.3f}',
                        'Grid Normal-Price ≥20€ (MWh)': '{:.3f}',
                        'Grid Renewable Part (MWh)': '{:.3f}',
                        'Grid Non-Renewable (MWh)': '{:.3f}',
                        'Emissions (g CO₂eq)': '{:.1f}',
                        'EF (g CO₂eq/MJ)': '{:.2f}',
                        'RFNBO %': '{:.1%}'
                    }),
                    use_container_width=True,
                    height=500
                )
                
                st.caption("""
                **Column Explanations:**
                - **PPA Energy**: Direct renewable energy from PPA (0 emissions)
                - **Grid Low-Price <20€**: Grid energy when prices < 20€/MWh (0 emissions for emission calc)
                - **Grid Normal-Price ≥20€**: Grid energy when prices ≥ 20€/MWh (country emission factor applies)
                - **Grid Renewable Part**: Renewable portion of grid based on energy mix (for RFNBO matching)
                - **Grid Non-Renewable**: Fossil portion of grid (causes emissions)
                - **EF**: Emission factor in g CO₂eq/MJ (limit: 28.2)
                - **RFNBO %**: Renewable fraction (target: 100%)
                """)
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download RFNBO Results as CSV",
                data=csv,
                file_name=f"rfnbo_results_{st.session_state.get('country', 'data')}_{st.session_state.get('year', '')}_{st.session_state.get('month', ''):02d}.csv",
                mime="text/csv"
            )
        else:
            st.info("👈 Configure parameters in the sidebar and click 'Fetch Data & Calculate' to see RFNBO analysis")
    
    with tab2:
        # Data Explorer Tab
        display_data_explorer_tab()
    
    with tab3:
        # Sensitivity Analysis Tab
        display_sensitivity_analysis_tab()
    
    # Information section
    with st.expander("ℹ️ About RFNBO Compliance"):
        st.markdown(f"""
        ### What is RFNBO?
        
        RFNBO (Renewable Fuel of Non-Biological Origin) refers to hydrogen produced through 
        electrolysis using renewable electricity.
        
        ### Compliance Requirements (TWO checks required)
        
        #### 1. **Emission Factor Check**
        
        The weighted emission factor must be < **{MAX_EMISSION_FACTOR_MJ:.1f} g CO₂eq/MJ** (30% of fossil comparator {FOSSIL_COMPARATOR_MJ} g CO₂eq/MJ)
        
        **Emission calculations:**
        - PPA energy: **0 g CO₂eq/kWh** (renewable)
        - Grid energy when price < **{PRICE_THRESHOLD_EUR_MWH:.0f}€/MWh**: **0 g CO₂eq/kWh** (considered renewable)
        - Grid energy when price ≥ {PRICE_THRESHOLD_EUR_MWH:.0f}€/MWh: **Country emission factor** (e.g., 162 g CO₂eq/kWh for Belgium)
        
        #### 2. **RFNBO Matching Check**
        
        RFNBO energy must match or exceed consumption (≥100%)
        
        **RFNBO energy calculation:**
        - RFNBO = PPA energy + (Grid energy × renewable share from energy mix)
        - Renewable share is calculated from ENTSOE generation mix data (monthly average)
        - Temporal correlation: hourly or monthly matching
        
        ### Grid Energy Rules (Applied Automatically)
        
        1. **Low-price rule** (emission factor):
           - When price < {PRICE_THRESHOLD_EUR_MWH:.0f}€/MWh → emission factor = 0
           - This rule is always applied
           
        2. **Renewable mix** (RFNBO matching):
           - Based on actual renewable share in national energy mix from ENTSOE data
           - Calculated as monthly average from actual generation data
        
        ### Data Sources
        
        - Day-ahead electricity prices from ENTSOE
        - Generation mix data (renewable vs non-renewable) from ENTSOE
        - Installed capacity per production type from ENTSOE
        - Country-specific grid emission factors
        """)

if __name__ == "__main__":
    main()

