"""
RFNBO Calculation Logic

This module contains all the core calculation functions for RFNBO compliance analysis.
Separated from the Streamlit UI for better maintainability and testing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Union

# Set up logging
logger = logging.getLogger(__name__)

# Constants
FOSSIL_COMPARATOR_MJ = 94  # 94 g CO₂eq/MJ (fossil fuel baseline)
MAX_EMISSION_FACTOR_MJ = 0.30 * FOSSIL_COMPARATOR_MJ  # 28.2 g CO₂eq/MJ (30% of fossil)
RENEWABLE_EMISSION_MJ = 0  # g CO₂eq/MJ (PPA and low-price grid)
PRICE_THRESHOLD_EUR_MWH = 20.0  # €/MWh (absolute threshold, not percentile)

# Grid emission factors by country [g CO₂eq/kWh]
# Source: Need to be provided per country
GRID_EMISSION_FACTORS = {
    'Belgium': 162,  # g CO₂eq/kWh
    'France': 71,
    'Germany (DE/LU)': 357,
    'Netherlands': 400,  # Example values - need real data
    'Spain': 250,
    'Italy Centro Nord': 300,
    'Italy Centro Sud': 300,
    # Add more countries as needed
}

# Default emission factor if country not found
DEFAULT_GRID_EMISSION_FACTOR = 200  # g CO₂eq/kWh

# PSR Type mapping (ENTSOE codes to readable names)
PSR_TYPE_MAPPING = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B03": "Fossil Coal-derived gas",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B07": "Fossil Oil shale",
    "B08": "Fossil Peat",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and poundage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other"
}

RENEWABLE_PSR_TYPES = ["B09", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]

# PPA Technology to PSR Type mapping
PPA_TECHNOLOGY_PSR_TYPES = {
    'Solar': ['B16'],  # Solar
    'Wind Onshore': ['B19'],  # Wind Onshore
    'Wind Offshore': ['B18'],  # Wind Offshore
    'Wind': ['B18', 'B19'],  # Both wind types combined
    'Solar + Wind Offshore': [('B16', 0.5), ('B18', 0.5)],  # Solar 50%, Wind Offshore 50%
    'Solar + Wind Onshore': [('B16', 0.5), ('B19', 0.5)]  # Solar 50%, Wind Onshore 50%
}


def calculate_emission_factor(
    ppa_energy_mwh: float,
    grid_energy_low_price_mwh: float,
    grid_energy_normal_price_mwh: float,
    country_emission_factor: float,
    electrolyser_consumption_mwh: float
) -> float:
    """
    Calculate weighted emission factor for hydrogen production.
    
    Emission sources:
    - PPA energy: 0 g CO₂eq/kWh (renewable)
    - Grid energy at low prices (<20€/MWh): 0 g CO₂eq/kWh (considered renewable)
    - Grid energy at normal prices: country emission factor g CO₂eq/kWh
    
    Args:
        ppa_energy_mwh: Energy from PPA in MWh
        grid_energy_low_price_mwh: Grid energy when price < 20€/MWh in MWh
        grid_energy_normal_price_mwh: Grid energy when price ≥ 20€/MWh in MWh
        country_emission_factor: Grid emission factor in g CO₂eq/kWh
        electrolyser_consumption_mwh: Total electrolyser energy consumption in MWh
    
    Returns:
        Weighted emission factor in g CO₂eq/kWh
    
    Example:
        >>> calculate_emission_factor(50, 10, 40, 162, 100)  # MWh, Belgium
        64.8  # g CO₂eq/kWh
    """
    if electrolyser_consumption_mwh == 0:
        return 0.0
    
    # PPA and low-price grid have 0 emissions
    # Only normal-price grid contributes to emissions
    total_emissions_g = grid_energy_normal_price_mwh * 1000 * country_emission_factor  # Convert MWh to kWh
    
    # Weighted emission factor in g CO₂eq/kWh
    # Divide by electrolyser consumption (not total energy sources)
    emission_factor_kwh = total_emissions_g / (electrolyser_consumption_mwh * 1000)
    
    return emission_factor_kwh


def convert_emission_factor_kwh_to_mj(emission_factor_kwh: float) -> float:
    """
    Convert emission factor from g CO₂eq/kWh to g CO₂eq/MJ.
    
    1 kWh = 3.6 MJ
    
    Args:
        emission_factor_kwh: Emission factor in g CO₂eq/kWh
    
    Returns:
        Emission factor in g CO₂eq/MJ
    """
    return emission_factor_kwh / 3.6


def is_emission_compliant(emission_factor_kwh: float) -> bool:
    """
    Check if emission factor meets RFNBO requirement.
    
    Requirement: emission factor < 30% of fossil comparator
    
    Args:
        emission_factor_kwh: Emission factor in g CO₂eq/kWh
    
    Returns:
        True if compliant, False otherwise
    """
    emission_factor_mj = convert_emission_factor_kwh_to_mj(emission_factor_kwh)
    return emission_factor_mj < MAX_EMISSION_FACTOR_MJ


def calculate_renewable_share(generation_df: pd.DataFrame, temporal_mode: str = 'constant') -> Union[float, pd.DataFrame]:
    """
    Calculate renewable share from generation data using proper energy integration.
    
    This function analyzes the generation mix data from ENTSOE to determine
    what fraction of electricity generation comes from renewable sources.
    Uses trapezoidal integration to convert power (MW) to energy (MWh).
    
    Args:
        generation_df: DataFrame with generation data from ENTSOE (power in MW)
        temporal_mode: 'constant' for single value, 'varying' for time-varying
    
    Returns:
        If constant: single float value (0-1)
        If varying: DataFrame with timestamp and renewable_share columns
    
    Example:
        >>> gen_df = fetch_all_generation_types("2023-06-15", "Belgium")
        >>> renewable_share = calculate_renewable_share(gen_df, 'constant')
        >>> print(f"Renewable share: {renewable_share * 100:.1f}%")
    """
    if generation_df.empty:
        return 0.0 if temporal_mode == 'constant' else pd.DataFrame()
    
    # Add readable names and identify renewable sources
    generation_df = generation_df.copy()
    generation_df['source_type'] = generation_df['psr_type'].map(PSR_TYPE_MAPPING)
    generation_df['is_renewable'] = generation_df['psr_type'].isin(RENEWABLE_PSR_TYPES)
    
    # Convert power (MW) to energy (MWh) using trapezoidal integration
    # Energy = 0.5 * (power[k] + power[k+1]) * timestep_hours
    def integrate_power_to_energy(df_group):
        """Apply trapezoidal integration to convert MW to MWh."""
        df_sorted = df_group.sort_values('timestamp').copy()
        
        if len(df_sorted) <= 1:
            # If only one point, use rectangular integration
            timestep_hours = df_sorted['resolution_minutes'].iloc[0] / 60
            df_sorted['generation_mwh'] = df_sorted['generation_mw'] * timestep_hours
            return df_sorted
        
        # Trapezoidal rule: energy = 0.5 * (power[k] + power[k+1]) * timestep_hours
        df_sorted['generation_mwh'] = 0.0
        
        for i in range(len(df_sorted) - 1):
            timestep_hours = df_sorted['resolution_minutes'].iloc[i] / 60
            power_avg = 0.5 * (df_sorted['generation_mw'].iloc[i] + df_sorted['generation_mw'].iloc[i+1])
            df_sorted.loc[df_sorted.index[i], 'generation_mwh'] = power_avg * timestep_hours
        
        # For the last point, use rectangular integration
        timestep_hours = df_sorted['resolution_minutes'].iloc[-1] / 60
        df_sorted.loc[df_sorted.index[-1], 'generation_mwh'] = df_sorted['generation_mw'].iloc[-1] * timestep_hours
        
        return df_sorted
    
    # Apply integration per PSR type to maintain data integrity
    generation_df = generation_df.groupby('psr_type', group_keys=False).apply(integrate_power_to_energy)
    
    if temporal_mode == 'constant':
        # Calculate constant average using integrated energy
        total_energy = generation_df['generation_mwh'].sum()
        renewable_energy = generation_df[generation_df['is_renewable']]['generation_mwh'].sum()
        return renewable_energy / total_energy if total_energy > 0 else 0.0
    
    else:  # varying
        # Group by timestamp to get varying renewable share
        hourly = generation_df.groupby('timestamp').agg({
            'generation_mwh': 'sum'
        }).reset_index()
        
        renewable_hourly = generation_df[generation_df['is_renewable']].groupby('timestamp').agg({
            'generation_mwh': 'sum'
        }).reset_index()
        
        hourly = hourly.merge(
            renewable_hourly, 
            on='timestamp', 
            how='left', 
            suffixes=('_total', '_renewable')
        )
        hourly['generation_mwh_renewable'] = hourly['generation_mwh_renewable'].fillna(0)
        hourly['renewable_share'] = hourly['generation_mwh_renewable'] / hourly['generation_mwh_total']
        hourly['renewable_share'] = hourly['renewable_share'].fillna(0)
        
        return hourly[['timestamp', 'renewable_share']]


def get_grid_emission_factor(zone_name: str) -> float:
    """
    Get grid emission factor for a given zone/country.
    
    Args:
        zone_name: Country or zone name
    
    Returns:
        Emission factor in g CO₂eq/kWh
    """
    return GRID_EMISSION_FACTORS.get(zone_name, DEFAULT_GRID_EMISSION_FACTOR)


def calculate_ppa_production_from_generation_data(
    generation_df: pd.DataFrame,
    ppa_technology: str,
    ppa_capacity_mw: float,
    prices_df: pd.DataFrame,
    installed_capacity_df: pd.DataFrame = None,
    solar_fraction: float = None,
    wind_fraction: float = None
) -> pd.DataFrame:
    """
    Calculate PPA production based on actual generation data from ENTSOE.
    
    Supports single and combined technologies (e.g., Solar + Wind Offshore).
    For combined technologies, each component is scaled independently then summed.
    
    **Methodology**:
    1. Get actual installed capacity from ENTSOE (documentType A68)
    2. Scale to PPA: PPA_production[t] = generation[t] × (PPA_capacity / installed_capacity)
    3. For combined: Scale each technology independently, then sum
    
    Args:
        generation_df: DataFrame with generation data from ENTSOE (in MW)
        ppa_technology: Technology type ('Solar', 'Wind Onshore', 'Wind Offshore', 
                                         'Solar + Wind Offshore', 'Solar + Wind Onshore')
        ppa_capacity_mw: Total PPA installed capacity in MW
        prices_df: DataFrame with prices (for timestamp alignment)
        installed_capacity_df: DataFrame with installed capacity from ENTSOE (optional)
        solar_fraction: Fraction of PPA capacity for solar (for combined tech, default 0.5)
        wind_fraction: Fraction of PPA capacity for wind (for combined tech, default 0.5)
    
    Returns:
        DataFrame with columns:
        - datetime: Time
        - ppa_production_mw: Actual PPA production in MW
        - (for combined) solar_production_mw: Solar component
        - (for combined) wind_production_mw: Wind component
    
    Example:
        >>> gen_df = fetch_all_generation_types("2023-06-15", "Belgium")
        >>> capacity_df = fetch_installed_capacity_all_types("2023-06-15", "Belgium")
        >>> prices_df = fetch_day_ahead_prices("2023-06-15", "Belgium")
        >>> # Single technology
        >>> ppa_prod = calculate_ppa_production_from_generation_data(
        ...     gen_df, 'Solar', 10.0, prices_df, capacity_df
        ... )
        >>> # Combined technology
        >>> ppa_prod = calculate_ppa_production_from_generation_data(
        ...     gen_df, 'Solar + Wind Offshore', 10.0, prices_df, capacity_df,
        ...     solar_fraction=0.5, wind_fraction=0.5
        ... )
    """
    if generation_df.empty:
        # No generation data - return zeros
        logger.warning("No generation data available, PPA production set to 0")
        result_df = prices_df[['datetime']].copy()
        result_df['ppa_production_mw'] = 0
        return result_df
    
    # Check if this is a combined technology
    is_combined = '+' in ppa_technology
    
    if is_combined:
        # Combined technology: process each component separately
        # Default to 50/50 split if not specified
        if solar_fraction is None:
            solar_fraction = 0.5
        if wind_fraction is None:
            wind_fraction = 0.5
        
        # Parse technology components
        tech_components = ppa_technology.split(' + ')
        if len(tech_components) != 2:
            logger.error(f"Invalid combined technology format: {ppa_technology}")
            result_df = prices_df[['datetime']].copy()
            result_df['ppa_production_mw'] = 0
            return result_df
        
        # Process each technology component
        combined_production = prices_df[['datetime']].copy()
        combined_production['ppa_production_mw'] = 0
        
        tech_fractions = [solar_fraction, wind_fraction]
        component_productions = {}
        
        for tech, fraction in zip(tech_components, tech_fractions):
            tech = tech.strip()
            tech_capacity = ppa_capacity_mw * fraction
            
            # Get PSR types for this component
            psr_types_mapping = {
                'Solar': ['B16'],
                'Wind Onshore': ['B19'],
                'Wind Offshore': ['B18']
            }
            psr_types = psr_types_mapping.get(tech, ['B16'])
            
            # Process this technology component
            tech_production = _process_single_technology(
                generation_df, psr_types, tech, tech_capacity,
                installed_capacity_df, prices_df
            )
            
            component_productions[tech.lower().replace(' ', '_')] = tech_production['ppa_production_mw']
            combined_production['ppa_production_mw'] += tech_production['ppa_production_mw']
        
        # Add component columns for combined technologies
        for comp_name, comp_prod in component_productions.items():
            combined_production[f'{comp_name}_production_mw'] = comp_prod
        
        return combined_production
    
    else:
        # Single technology
        # Get PSR types for the selected technology
        psr_types = PPA_TECHNOLOGY_PSR_TYPES.get(ppa_technology, ['B16'])
        
        return _process_single_technology(
            generation_df, psr_types, ppa_technology, ppa_capacity_mw,
            installed_capacity_df, prices_df
        )


def _process_single_technology(
    generation_df: pd.DataFrame,
    psr_types: list,
    technology_name: str,
    capacity_mw: float,
    installed_capacity_df: pd.DataFrame,
    prices_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Helper function to process a single technology and return scaled production.
    """
    # Filter generation data for the selected technology
    tech_generation = generation_df[generation_df['psr_type'].isin(psr_types)].copy()
    
    if tech_generation.empty:
        # Technology not available in this country
        logger.warning(f"No generation data for technology {technology_name}, PPA production set to 0")
        result_df = prices_df[['datetime']].copy()
        result_df['ppa_production_mw'] = 0
        return result_df
    
    # Group by timestamp to get total generation for the technology
    tech_by_time = tech_generation.groupby('timestamp').agg({
        'generation_mw': 'sum'
    }).reset_index()
    
    # Get installed capacity
    if installed_capacity_df is not None and not installed_capacity_df.empty:
        # Use actual installed capacity from ENTSOE (preferred method)
        tech_capacity = installed_capacity_df[installed_capacity_df['psr_type'].isin(psr_types)].copy()
        
        if not tech_capacity.empty:
            # Get installed capacity (typically constant, to be verified)
            total_installed_capacity_mw = tech_capacity['installed_capacity_mw'].mean()
            
            if total_installed_capacity_mw > 0:
                
                # Scale to PPA capacity: PPA_production = generation × (PPA_capacity / installed_capacity)
                scaling_factor = capacity_mw / total_installed_capacity_mw
                tech_by_time['ppa_production_mw'] = tech_by_time['generation_mw'] * scaling_factor
            else:
                # Installed capacity is zero - fallback
                logger.warning("Installed capacity is zero, using fallback method")
                installed_capacity_df = None  # Trigger fallback below
        else:
            # No capacity data for this technology
            logger.warning(f"No installed capacity data for {technology_name}, using fallback method")
            installed_capacity_df = None  # Trigger fallback
    
    if installed_capacity_df is None or installed_capacity_df.empty:
        # Fallback: Estimate from 99th percentile of generation (old method)
        max_generation_mw = tech_by_time['generation_mw'].quantile(0.99)
        
        if max_generation_mw == 0:
            # No meaningful generation data
            logger.warning(f"No meaningful generation data for {technology_name}, setting PPA production to 0")
            result_df = prices_df[['datetime']].copy()
            result_df['ppa_production_mw'] = 0
            return result_df
        
        # Scale to PPA capacity based on estimated max generation
        scaling_factor = capacity_mw / max_generation_mw
        tech_by_time['ppa_production_mw'] = tech_by_time['generation_mw'] * scaling_factor
        
        logger.info(f"Using estimated capacity (99th percentile): {max_generation_mw:.2f} MW for {technology_name}")
    
    # Merge with prices_df to align timestamps
    result_df = prices_df[['datetime']].copy()
    
    result_df = result_df.merge(
        tech_by_time[['timestamp', 'ppa_production_mw']],
        left_on='datetime',
        right_on='timestamp',
        how='left'
    )
    
    result_df['ppa_production_mw'] = result_df['ppa_production_mw'].fillna(0)
    result_df = result_df[['datetime', 'ppa_production_mw']].copy()
    
    return result_df


def calculate_rfnbo_compliance(
    electrolyser_mw: float,
    ppa_capacity_mw: float,
    prices_df: pd.DataFrame,
    renewable_share: Union[float, pd.DataFrame],
    zone_name: str,
    temporal_correlation: str = 'hourly',
    use_price_threshold: bool = True,
    ppa_technology: str = None,
    generation_df: pd.DataFrame = None,
    installed_capacity_df: pd.DataFrame = None,
    solar_fraction: float = None,
    wind_fraction: float = None
) -> pd.DataFrame:
    """
    Calculate RFNBO compliance for electrolyser operation.
    
    This implements the correct RFNBO methodology:
    
    1. **Emission Factor Calculation**:
       - PPA energy: 0 g CO₂eq/kWh (renewable)
       - Grid energy when price < 20€/MWh: 0 g CO₂eq/kWh (considered renewable)
       - Grid energy when price ≥ 20€/MWh: country emission factor
       - Calculate weighted emission factor
       - Check: emission factor < 28.2 g CO₂eq/MJ (30% of fossil comparator)
    
    2. **RFNBO Percentage Calculation**:
       - RFNBO energy = PPA energy + (grid energy × renewable share from energy mix)
       - Calculate temporal matching (hourly or monthly)
       - RFNBO % = (RFNBO energy / electrolyser consumption) × 100%
    
    Args:
        electrolyser_mw: Electrolyser capacity in MW
        ppa_capacity_mw: PPA renewable capacity in MW
        prices_df: DataFrame with day-ahead prices from ENTSOE
        renewable_share: Either float (annual) or DataFrame (hourly) with renewable share
        zone_name: Country/zone name for emission factor lookup
        temporal_correlation: 'hourly' or 'monthly'
        use_price_threshold: Whether to apply 20€/MWh price rule
        ppa_technology: PPA technology type ('Solar', 'Wind', etc.) - uses real generation data
        generation_df: DataFrame with generation data - calculates actual PPA production
        installed_capacity_df: DataFrame with installed capacity data - uses for accurate scaling
    
    Returns:
        DataFrame with hourly results including:
        - datetime: Timestamp
        - electrolyser_consumption_mw: Electrolyser power consumption
        - electrolyser_consumption_mwh: Electrolyser energy consumption
        - ppa_energy_mwh: Energy from PPA
        - grid_energy_low_price_mwh: Grid energy when price < 20€/MWh
        - grid_energy_normal_price_mwh: Grid energy when price ≥ 20€/MWh
        - emission_factor_kwh: Emission factor in g CO₂eq/kWh
        - emission_factor_mj: Emission factor in g CO₂eq/MJ
        - is_emission_compliant: Boolean - emission check
        - rfnbo_energy_mwh: Total RFNBO-qualifying energy
        - rfnbo_fraction: RFNBO fraction (0-1)
        - is_rfnbo_100pct: Boolean - 100% RFNBO check
    
    Example:
        >>> prices_df = fetch_day_ahead_prices("2023-06-15", "Belgium")
        >>> gen_df = fetch_all_generation_types("2023-06-15", "Belgium")
        >>> capacity_df = fetch_installed_capacity_all_types("2023-06-15", "Belgium")
        >>> renewable_share = calculate_renewable_share(gen_df, 'annual')
        >>> results = calculate_rfnbo_compliance(
        ...     electrolyser_mw=1.0,
        ...     ppa_capacity_mw=1.5,
        ...     prices_df=prices_df,
        ...     renewable_share=renewable_share,
        ...     zone_name='Belgium',
        ...     ppa_technology='Solar',
        ...     generation_df=gen_df,
        ...     installed_capacity_df=capacity_df
        ... )
        >>> print(f"Average RFNBO fraction: {results['rfnbo_fraction'].mean() * 100:.1f}%")
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    # Get grid emission factor for this country
    country_emission_factor = get_grid_emission_factor(zone_name)
    
    # Create base dataframe
    df = prices_df.copy()
    df['electrolyser_consumption_mw'] = electrolyser_mw
    
    # Add resolution column if not present
    if 'resolution_minutes' not in df.columns:
        df['resolution_minutes'] = 60  # default to hourly
    
    # Convert power to energy (MWh)
    timestep_hours = df['resolution_minutes'] / 60
    df['electrolyser_consumption_mwh'] = df['electrolyser_consumption_mw'] * timestep_hours
    
    # Calculate PPA production from actual generation data
    if ppa_technology is not None and generation_df is not None and not generation_df.empty:
        # Use actual generation data for the selected technology
        ppa_production_df = calculate_ppa_production_from_generation_data(
            generation_df=generation_df,
            ppa_technology=ppa_technology,
            ppa_capacity_mw=ppa_capacity_mw,
            prices_df=df,
            installed_capacity_df=installed_capacity_df,
            solar_fraction=solar_fraction,
            wind_fraction=wind_fraction
        )
        df = df.merge(ppa_production_df, on='datetime', how='left')
        # Fill any missing values with 0 (no PPA production)
        df['ppa_production_mw'] = df['ppa_production_mw'].fillna(0)
    else:
        # No generation data available - set PPA production to 0
        logger.warning("No generation data available for PPA production calculation, setting to 0")
        df['ppa_production_mw'] = 0
    
    df['ppa_energy_mwh'] = df['ppa_production_mw'] * timestep_hours
    
    # Calculate grid consumption
    df['grid_consumption_mw'] = (df['electrolyser_consumption_mw'] - df['ppa_production_mw']).clip(lower=0)
    df['grid_energy_mwh'] = df['grid_consumption_mw'] * timestep_hours
    
    # PPA can't exceed electrolyser consumption
    #df['ppa_energy_mwh'] = np.minimum(df['ppa_energy_mwh'], df['electrolyser_consumption_mwh'])
    
    # Identify low-price periods (< 20€/MWh)
    df['is_low_price'] = df['price_eur_mwh'] < PRICE_THRESHOLD_EUR_MWH if use_price_threshold else False
    
    # Split grid energy by price level
    df['grid_energy_low_price_mwh'] = df['grid_energy_mwh'] * df['is_low_price'].astype(float)
    df['grid_energy_normal_price_mwh'] = df['grid_energy_mwh'] * (~df['is_low_price']).astype(float)
    
    # ============================================
    # PART 1: EMISSION FACTOR CALCULATION
    # ============================================
    
    # Calculate total emissions for each hour (only from normal-price grid)
    df['emissions_g_co2_eq'] = df['grid_energy_normal_price_mwh'] * 1000 * country_emission_factor
    
    # Calculate total consumption in MJ for emission factor calculation
    df['total_consumption_mj'] = df['electrolyser_consumption_mwh'] * 3600  # 1 MWh = 3600 MJ
    
    # Calculate emission factor for each hour
    df['emission_factor_kwh'] = df.apply(
        lambda row: calculate_emission_factor(
            row['ppa_energy_mwh'],
            row['grid_energy_low_price_mwh'],
            row['grid_energy_normal_price_mwh'],
            country_emission_factor,
            row['electrolyser_consumption_mwh']
        ),
        axis=1
    )
    
    # Convert to g CO₂eq/MJ
    df['emission_factor_mj'] = df['emission_factor_kwh'].apply(convert_emission_factor_kwh_to_mj)
    
    # Check emission compliance
    df['is_emission_compliant'] = df['emission_factor_kwh'].apply(is_emission_compliant)
    
    # ============================================
    # PART 2: RFNBO PERCENTAGE CALCULATION
    # ============================================
    
    # Determine renewable share for grid (from energy mix, not price rule)
    if isinstance(renewable_share, float):
        # constant for all hours
        df['grid_renewable_share_mix'] = renewable_share
    else:
        # time-varying
        df = df.merge(renewable_share, left_on='datetime', right_on='timestamp', how='left', suffixes=('', '_renewable'))
        df['grid_renewable_share_mix'] = df['renewable_share'].fillna(
            renewable_share['renewable_share'].mean() if not renewable_share.empty else 0
        )
    
    # Calculate RFNBO-qualifying energy
    # RFNBO = PPA energy + (grid energy × renewable share from energy mix)
    df['rfnbo_from_ppa_mwh'] = df['ppa_energy_mwh']  # PPA is 100% RFNBO
    df['rfnbo_from_grid_mwh'] = df['grid_energy_mwh'] * df['grid_renewable_share_mix']
    df['rfnbo_energy_mwh'] = df['rfnbo_from_ppa_mwh'] + df['rfnbo_from_grid_mwh']
    
    # Calculate RFNBO fraction
    df['rfnbo_fraction'] = df['rfnbo_energy_mwh'] / df['electrolyser_consumption_mwh']
    df['rfnbo_fraction'] = df['rfnbo_fraction'].clip(upper=1.0)  # Cap at 100%
    
    # Check if 100% RFNBO (RFNBO energy ≥ consumption)
    df['is_rfnbo_100pct'] = df['rfnbo_energy_mwh'] >= df['electrolyser_consumption_mwh']
    
    # Overall compliance: both emission AND RFNBO checks must pass
    df['is_compliant'] = df['is_emission_compliant'] & (df['rfnbo_fraction'] >= 1.0)
    
    return df


def aggregate_to_monthly(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly RFNBO results to monthly summary.
    
    This function takes the detailed hourly calculation results and produces
    a monthly summary with key metrics for the entire period.
    
    Args:
        hourly_df: DataFrame from calculate_rfnbo_compliance()
    
    Returns:
        DataFrame with one row containing monthly summary:
        - total_consumption_mwh: Total energy consumed
        - ppa_energy_mwh: Total PPA energy
        - grid_energy_mwh: Total grid energy
        - grid_energy_low_price_mwh: Grid energy during low prices
        - grid_energy_normal_price_mwh: Grid energy during normal prices
        - avg_emission_factor_kwh: Average emission factor (g CO₂eq/kWh)
        - avg_emission_factor_mj: Average emission factor (g CO₂eq/MJ)
        - emission_compliant_hours: Hours meeting emission requirement
        - rfnbo_energy_mwh: Total RFNBO energy
        - non_rfnbo_energy_mwh: Total non-RFNBO energy
        - rfnbo_fraction: Overall RFNBO fraction (0-1)
        - rfnbo_100pct_hours: Hours with 100% RFNBO
        - compliant_hours: Hours meeting both checks
        - total_hours: Total number of hours
    
    Example:
        >>> results = calculate_rfnbo_compliance(...)
        >>> summary = aggregate_to_monthly(results)
        >>> print(f"Monthly RFNBO fraction: {summary['rfnbo_fraction'].values[0] * 100:.1f}%")
        >>> print(f"Emission factor: {summary['avg_emission_factor_mj'].values[0]:.2f} g CO₂eq/MJ")
    """
    if hourly_df.empty:
        return pd.DataFrame()
    
    hourly_df = hourly_df.copy()
    
    # Calculate monthly summary statistics
    total_consumption = hourly_df['electrolyser_consumption_mwh'].sum()
    
    monthly_summary = {
        'total_consumption_mwh': total_consumption,
        'ppa_energy_mwh': hourly_df['ppa_energy_mwh'].sum(),
        'grid_energy_mwh': hourly_df['grid_energy_mwh'].sum(),
        'grid_energy_low_price_mwh': hourly_df['grid_energy_low_price_mwh'].sum(),
        'grid_energy_normal_price_mwh': hourly_df['grid_energy_normal_price_mwh'].sum(),
        'avg_emission_factor_kwh': hourly_df['emission_factor_kwh'].mean(),
        'avg_emission_factor_mj': hourly_df['emission_factor_mj'].mean(),
        'emission_compliant_hours': hourly_df['is_emission_compliant'].sum(),
        'rfnbo_energy_mwh': hourly_df['rfnbo_energy_mwh'].sum(),
        'non_rfnbo_energy_mwh': total_consumption - hourly_df['rfnbo_energy_mwh'].sum(),
        'rfnbo_fraction': hourly_df['rfnbo_energy_mwh'].sum() / total_consumption if total_consumption > 0 else 0,
        'rfnbo_100pct_hours': hourly_df['is_rfnbo_100pct'].sum(),
        'compliant_hours': hourly_df['is_compliant'].sum(),
        'total_hours': len(hourly_df),
        'emission_compliance_rate': hourly_df['is_emission_compliant'].sum() / len(hourly_df),
        'rfnbo_compliance_rate': hourly_df['is_rfnbo_100pct'].sum() / len(hourly_df),
        'overall_compliance_rate': hourly_df['is_compliant'].sum() / len(hourly_df)
    }
    
    return pd.DataFrame([monthly_summary])


def is_rfnbo_compliant(monthly_summary: pd.DataFrame) -> dict:
    """
    Determine if overall RFNBO compliance is met.
    
    Checks two requirements:
    1. Emission factor < 28.2 g CO₂eq/MJ (30% of fossil comparator)
    2. RFNBO fraction ≥ 100% (renewable energy matches or exceeds consumption)
    
    Args:
        monthly_summary: DataFrame from aggregate_to_monthly()
    
    Returns:
        Dictionary with compliance status:
        - emission_compliant: Boolean - emission check passed
        - rfnbo_compliant: Boolean - RFNBO matching check passed
        - overall_compliant: Boolean - both checks passed
        - emission_factor_mj: Average emission factor
        - rfnbo_fraction: RFNBO fraction
    
    Example:
        >>> summary = aggregate_to_monthly(results)
        >>> compliance = is_rfnbo_compliant(summary)
        >>> if compliance['overall_compliant']:
        ...     print("✅ RFNBO COMPLIANT")
        ... else:
        ...     print("❌ NOT COMPLIANT")
    """
    if monthly_summary.empty:
        return {
            'emission_compliant': False,
            'rfnbo_compliant': False,
            'overall_compliant': False,
            'emission_factor_mj': float('inf'),
            'rfnbo_fraction': 0.0
        }
    
    emission_factor = monthly_summary['avg_emission_factor_mj'].values[0]
    rfnbo_fraction = monthly_summary['rfnbo_fraction'].values[0]
    
    emission_compliant = emission_factor < MAX_EMISSION_FACTOR_MJ
    rfnbo_compliant = rfnbo_fraction >= 1.0  # 100% RFNBO
    
    return {
        'emission_compliant': emission_compliant,
        'rfnbo_compliant': rfnbo_compliant,
        'overall_compliant': emission_compliant and rfnbo_compliant,
        'emission_factor_mj': emission_factor,
        'rfnbo_fraction': rfnbo_fraction
    }


def calculate_statistics(prices_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for price data.
    
    Useful for the data explorer and understanding price patterns.
    
    Args:
        prices_df: DataFrame with price_eur_mwh column
    
    Returns:
        Dictionary with statistics:
        - total_records: Number of records
        - avg_price: Average price in €/MWh
        - min_price: Minimum price
        - max_price: Maximum price
        - below_20_count: Count of hours below 20€/MWh
        - below_20_pct: Percentage below 20€/MWh
    """
    if prices_df.empty:
        return {}
    
    below_20 = (prices_df['price_eur_mwh'] < 20).sum()
    
    return {
        'total_records': len(prices_df),
        'avg_price': prices_df['price_eur_mwh'].mean(),
        'min_price': prices_df['price_eur_mwh'].min(),
        'max_price': prices_df['price_eur_mwh'].max(),
        'below_20_count': below_20,
        'below_20_pct': (below_20 / len(prices_df)) * 100 if len(prices_df) > 0 else 0
    }


def calculate_generation_statistics(generation_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for generation mix data.
    
    NOTE: This function converts power (MW) to energy (MWh) using trapezoidal integration.
    
    Args:
        generation_df: DataFrame with generation data from ENTSOE (power in MW)
    
    Returns:
        Dictionary with statistics:
        - total_records: Number of records
        - total_generation: Total generation in MWh (integrated)
        - renewable_generation: Renewable generation in MWh (integrated)
        - renewable_share: Renewable share as percentage
        - unique_sources: Number of different generation sources
    """
    if generation_df.empty:
        return {}
    
    # Add readable names and renewable flag
    gen_df = generation_df.copy()
    gen_df['source_type'] = gen_df['psr_type'].map(PSR_TYPE_MAPPING)
    gen_df['is_renewable'] = gen_df['psr_type'].isin(RENEWABLE_PSR_TYPES)
    
    # Integrate power to energy using trapezoidal rule
    def integrate_power_to_energy(df_group):
        """Apply trapezoidal integration to convert MW to MWh."""
        df_sorted = df_group.sort_values('timestamp').copy()
        
        if len(df_sorted) <= 1:
            timestep_hours = df_sorted['resolution_minutes'].iloc[0] / 60
            df_sorted['generation_mwh'] = df_sorted['generation_mw'] * timestep_hours
            return df_sorted
        
        df_sorted['generation_mwh'] = 0.0
        
        for i in range(len(df_sorted) - 1):
            timestep_hours = df_sorted['resolution_minutes'].iloc[i] / 60
            power_avg = 0.5 * (df_sorted['generation_mw'].iloc[i] + df_sorted['generation_mw'].iloc[i+1])
            df_sorted.loc[df_sorted.index[i], 'generation_mwh'] = power_avg * timestep_hours
        
        timestep_hours = df_sorted['resolution_minutes'].iloc[-1] / 60
        df_sorted.loc[df_sorted.index[-1], 'generation_mwh'] = df_sorted['generation_mw'].iloc[-1] * timestep_hours
        
        return df_sorted
    
    gen_df = gen_df.groupby('psr_type', group_keys=False).apply(integrate_power_to_energy)
    
    total_gen = gen_df['generation_mwh'].sum()
    renewable_gen = gen_df[gen_df['is_renewable']]['generation_mwh'].sum()
    
    return {
        'total_records': len(gen_df),
        'total_generation': total_gen,
        'renewable_generation': renewable_gen,
        'renewable_share': (renewable_gen / total_gen * 100) if total_gen > 0 else 0,
        'unique_sources': gen_df['psr_type'].nunique()
    }

