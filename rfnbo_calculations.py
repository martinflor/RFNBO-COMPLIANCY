"""
RFNBO Calculation Logic

This module contains all the core calculation functions for RFNBO compliance analysis.
Separated from the Streamlit UI for better maintainability and testing.
"""

import pandas as pd
import numpy as np
import logging

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
    e_nres_mwh: float,
    country_emission_factor: float,
    electrolyser_consumption_mwh: float
) -> float:
    """
    Calculate GHG emission factor for hydrogen production.

    Formula: EF_H2 = E_NRES * EF_grid / E_H2

    Where:
    - E_NRES = non-renewable energy = max(E_H2 - E_totalRES, 0)
    - E_totalRES = E_PPA + E_DAPrices  (PPA + grid when DA price < 20€/MWh)
    - EF_grid = country grid emission factor [g CO₂eq/kWh]
    - E_H2 = total electrolyser consumption

    Only non-renewable energy (normal-price grid above the PPA supply) contributes
    to GHG emissions; PPA and low-price grid energy are treated as zero-emission.

    Args:
        e_nres_mwh: Non-renewable energy consumed by electrolyser [MWh]
        country_emission_factor: Grid emission factor [g CO₂eq/kWh]
        electrolyser_consumption_mwh: Total electrolyser energy consumption [MWh]

    Returns:
        Emission factor in g CO₂eq/kWh

    Example:
        >>> calculate_emission_factor(40, 162, 100)  # 40 MWh non-renewable, Belgium
        64.8  # g CO₂eq/kWh
    """
    if electrolyser_consumption_mwh == 0:
        return 0.0

    # EF_H2 = E_NRES * EF_grid / E_H2
    # Units: MWh * (g CO₂eq/kWh) / MWh  →  g CO₂eq/kWh  (MWh cancels)
    return e_nres_mwh * country_emission_factor / electrolyser_consumption_mwh


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


def integrate_power_to_energy(
    df: pd.DataFrame,
    power_column: str,
    energy_column: str,
    resolution_column: str = 'resolution_minutes',
    timestamp_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Convert power (MW) to energy (MWh) using trapezoidal integration.
    
    This function applies trapezoidal integration to convert instantaneous power values
    to energy over time. More accurate than simple rectangular integration, especially
    when power values change significantly between timesteps.
    
    Args:
        df: DataFrame with power data
        power_column: Name of the power column (e.g., 'generation_mw', 'ppa_production_mw')
        energy_column: Name of the output energy column (e.g., 'generation_mwh', 'ppa_energy_mwh')
        resolution_column: Name of the resolution column (default: 'resolution_minutes')
        timestamp_column: Name of the timestamp column (default: 'timestamp')
    
    Returns:
        DataFrame with added energy column
    
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1)],
        ...     'power_mw': [100, 150],
        ...     'resolution_minutes': [60, 60]
        ... })
        >>> result = integrate_power_to_energy(df, 'power_mw', 'energy_mwh')
        >>> print(result['energy_mwh'].sum())  # Should be 125 MWh (average of 100 and 150)
    """
    if df.empty:
        return df
    
    # Validate required columns exist
    if power_column not in df.columns:
        raise ValueError(f"Power column '{power_column}' not found in DataFrame")
    if resolution_column not in df.columns:
        raise ValueError(f"Resolution column '{resolution_column}' not found in DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
    
    df = df.copy()
    
    # Fill NaN values in power column with 0 (no power = no energy)
    if df[power_column].isna().any():
        df[power_column] = df[power_column].fillna(0)
    
    # Sort by timestamp for proper integration (time-series data should be sorted)
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    
    if len(df) <= 1:
        # If only one point, use rectangular integration
        timestep_hours = df[resolution_column].iloc[0] / 60
        df[energy_column] = df[power_column] * timestep_hours
        return df
    
    # Initialize energy column
    df[energy_column] = 0.0
    
    # Trapezoidal rule: energy = 0.5 * (power[k] + power[k+1]) * timestep_hours
    for i in range(len(df) - 1):
        timestep_hours = df[resolution_column].iloc[i] / 60
        # Handle NaN values by treating them as 0
        power_i = df[power_column].iloc[i] if pd.notna(df[power_column].iloc[i]) else 0
        power_i1 = df[power_column].iloc[i+1] if pd.notna(df[power_column].iloc[i+1]) else 0
        power_avg = 0.5 * (power_i + power_i1)
        df.loc[i, energy_column] = power_avg * timestep_hours
    
    # For the last point, use rectangular integration
    timestep_hours = df[resolution_column].iloc[-1] / 60
    power_last = df[power_column].iloc[-1] if pd.notna(df[power_column].iloc[-1]) else 0
    df.loc[len(df) - 1, energy_column] = power_last * timestep_hours
    
    return df


def calculate_renewable_share(generation_df: pd.DataFrame) -> float:
    """
    Calculate renewable share from generation data using proper energy integration.

    Computes the fraction of total electricity generation (over the full period)
    that comes from renewable sources. A single constant value is returned and
    applied uniformly as α_gridRES in the RFNBO calculation — in line with the
    regulation, which does not allow time-varying grid renewable-share values.

    Uses trapezoidal integration to convert power (MW) to energy (MWh) before
    computing the share, so varying time-step resolutions are handled correctly.

    Args:
        generation_df: DataFrame with generation data from ENTSOE (power in MW).
                       Must contain columns: psr_type, generation_mw,
                       resolution_minutes, timestamp.

    Returns:
        Renewable share as a float in [0, 1].

    Example:
        >>> gen_df = fetch_all_generation_types("2023-06-15", "Belgium")
        >>> alpha = calculate_renewable_share(gen_df)
        >>> print(f"Renewable share: {alpha * 100:.1f}%")
    """
    if generation_df.empty:
        return 0.0

    # Add readable names and identify renewable sources
    generation_df = generation_df.copy()
    generation_df['source_type'] = generation_df['psr_type'].map(PSR_TYPE_MAPPING)
    generation_df['is_renewable'] = generation_df['psr_type'].isin(RENEWABLE_PSR_TYPES)

    # Convert power (MW) to energy (MWh) using trapezoidal integration
    # Applied per PSR type to maintain data integrity across different resolutions
    def integrate_group(df_group):
        """Wrapper to apply integration to each PSR type group."""
        return integrate_power_to_energy(
            df_group,
            power_column='generation_mw',
            energy_column='generation_mwh',
            resolution_column='resolution_minutes',
            timestamp_column='timestamp'
        )

    generation_df = generation_df.groupby('psr_type', group_keys=False).apply(integrate_group)

    total_energy = generation_df['generation_mwh'].sum()
    renewable_energy = generation_df[generation_df['is_renewable']]['generation_mwh'].sum()
    return renewable_energy / total_energy if total_energy > 0 else 0.0


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
    renewable_share: float,
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

    All calculations are performed at hourly granularity. Monthly aggregation
    is handled separately in aggregate_to_monthly().

    Methodology
    -----------
    **Step 1 – Hourly energy flows**

    E_H2   = electrolyser consumption [MWh] (trapezoidal integration of MW)
    E_PPA  = PPA renewable generation, capped at E_H2 [MWh]
    E_grid = max(E_H2 − E_PPA, 0)  →  electricity imported from the grid [MWh]

    Applying the DA-price rule (when use_price_threshold=True):
        E_DAPrices = E_grid   if DA price < 20 €/MWh
        E_DAPrices = 0        otherwise

    E_totalRES = E_PPA + E_DAPrices       (total renewable energy for GHG check)
    E_NRES     = max(E_H2 − E_totalRES, 0)  (non-renewable energy)

    **Step 2 – GHG Emission Factor** (hourly, then aggregated for monthly check)

        EF_H2 = Σ(E_NRES) × EF_grid / Σ(E_H2)

    Hourly: EF_H2_h = E_NRES_h × EF_grid / E_H2_h
    Monthly: EF_H2_m = Σ_month(E_NRES_h) × EF_grid / Σ_month(E_H2_h)

    Compliant if EF_H2 < 28.2 g CO₂eq/MJ (= 30 % of 94 g CO₂eq/MJ fossil ref.)

    **Step 3 – RFNBO Share** (temporal-correlation window: hourly or monthly)

        E_gridRES = E_grid × α_gridRES   (grid renewable mix share)
        E_RFNBO   = E_H2               if DA price < 20 €/MWh
        E_RFNBO   = E_PPA + E_gridRES  otherwise

        %H2_RFNBO = 100 × Σ(E_RFNBO) / Σ(E_H2)
        (Σ window = 1 h for hourly correlation, full period for monthly)

    Args:
        electrolyser_mw: Electrolyser capacity [MW]
        ppa_capacity_mw: PPA installed capacity [MW]
        prices_df: DataFrame with day-ahead prices (must contain 'price_eur_mwh')
        renewable_share: Constant float with the grid renewable mix share (α_gridRES)
                         for the studied country (per regulation, a single value
                         applies uniformly — no time-varying share is allowed)
        zone_name: Country/zone name – used for EF_grid lookup
        temporal_correlation: 'hourly' or 'monthly' (affects aggregation in
                              aggregate_to_monthly, not the hourly output here)
        use_price_threshold: Apply the 20 €/MWh DA-price rule (default True)
        ppa_technology: PPA technology ('Solar', 'Wind Onshore', etc.)
        generation_df: ENTSOE generation data for PPA production scaling
        installed_capacity_df: ENTSOE installed-capacity data for accurate scaling
        solar_fraction: Solar share of combined PPA capacity (default 0.5)
        wind_fraction: Wind share of combined PPA capacity (default 0.5)

    Returns:
        DataFrame with one row per hour containing:
        - datetime                      : timestamp
        - electrolyser_consumption_mw   : electrolyser power [MW]
        - electrolyser_consumption_mwh  : E_H2 [MWh]
        - ppa_energy_mwh                : E_PPA [MWh]
        - grid_energy_mwh               : E_grid [MWh]
        - grid_energy_low_price_mwh     : E_DAPrices [MWh]
        - grid_energy_normal_price_mwh  : E_grid at normal price [MWh]
        - e_total_res_mwh               : E_totalRES [MWh]
        - e_nres_mwh                    : E_NRES [MWh]
        - total_emissions_g_co2eq       : E_NRES × EF_grid [g CO₂eq]
        - emission_factor_kwh           : hourly EF_H2 [g CO₂eq/kWh]
        - emission_factor_mj            : hourly EF_H2 [g CO₂eq/MJ]
        - is_emission_compliant         : EF_H2 < 28.2 g CO₂eq/MJ
        - e_grid_res_mwh                : E_gridRES [MWh]
        - rfnbo_energy_mwh              : E_RFNBO [MWh]
        - rfnbo_fraction                : E_RFNBO / E_H2 (hourly, 0–1)
        - is_rfnbo_100pct               : rfnbo_fraction >= 1.0
        - is_compliant                  : emission_compliant AND rfnbo_100pct

    Example:
        >>> prices_df = fetch_day_ahead_prices("2023-06-15", "Belgium")
        >>> gen_df = fetch_all_generation_types("2023-06-15", "Belgium")
        >>> capacity_df = fetch_installed_capacity_all_types("2023-06-15", "Belgium")
        >>> renewable_share = calculate_renewable_share(gen_df)
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

    # Grid emission factor for this country [g CO₂eq/kWh]
    country_emission_factor = get_grid_emission_factor(zone_name)

    # -----------------------------------------------------------------------
    # Base dataframe setup
    # -----------------------------------------------------------------------
    df = prices_df.copy()
    df['electrolyser_consumption_mw'] = electrolyser_mw

    if 'resolution_minutes' not in df.columns:
        df['resolution_minutes'] = 60  # default to hourly

    # Ensure a 'datetime' column exists for integration
    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = df['timestamp']
    elif 'datetime' not in df.columns:
        if df.index.dtype.name.startswith('datetime'):
            df['datetime'] = df.index
        else:
            df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')

    # -----------------------------------------------------------------------
    # E_H2 – electrolyser consumption [MWh]  (trapezoidal integration)
    # -----------------------------------------------------------------------
    df = integrate_power_to_energy(
        df,
        power_column='electrolyser_consumption_mw',
        energy_column='electrolyser_consumption_mwh',
        resolution_column='resolution_minutes',
        timestamp_column='datetime'
    )

    # -----------------------------------------------------------------------
    # E_PPA – PPA renewable generation [MWh]
    # -----------------------------------------------------------------------
    if ppa_technology is not None and generation_df is not None and not generation_df.empty:
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
        df['ppa_production_mw'] = df['ppa_production_mw'].fillna(0)
    else:
        logger.warning("No generation data available for PPA production calculation, setting to 0")
        df['ppa_production_mw'] = 0

    # Convert PPA power → energy [MWh]
    df = integrate_power_to_energy(
        df,
        power_column='ppa_production_mw',
        energy_column='ppa_energy_mwh_raw',
        resolution_column='resolution_minutes',
        timestamp_column='datetime'
    )

    # E_PPA capped at E_H2 (cannot use more PPA than the electrolyser consumes)
    df['ppa_energy_mwh'] = np.minimum(df['ppa_energy_mwh_raw'], df['electrolyser_consumption_mwh'])
    df.drop(columns=['ppa_energy_mwh_raw'], inplace=True)

    # -----------------------------------------------------------------------
    # E_grid = max(E_H2 − E_PPA, 0)  →  grid imports [MWh]
    # -----------------------------------------------------------------------
    df['grid_energy_mwh'] = (df['electrolyser_consumption_mwh'] - df['ppa_energy_mwh']).clip(lower=0)

    # -----------------------------------------------------------------------
    # DA-price threshold rule
    # -----------------------------------------------------------------------
    df['is_low_price'] = (
        df['price_eur_mwh'] < PRICE_THRESHOLD_EUR_MWH if use_price_threshold else False
    )

    # E_DAPrices = E_grid when price < 20 €/MWh, else 0
    df['grid_energy_low_price_mwh'] = df['grid_energy_mwh'] * df['is_low_price'].astype(float)
    # Grid energy at normal prices (≥ 20 €/MWh)
    df['grid_energy_normal_price_mwh'] = df['grid_energy_mwh'] * (~df['is_low_price']).astype(float)

    # -----------------------------------------------------------------------
    # E_totalRES = E_PPA + E_DAPrices
    # E_NRES     = max(E_H2 − E_totalRES, 0)
    # -----------------------------------------------------------------------
    df['e_total_res_mwh'] = df['ppa_energy_mwh'] + df['grid_energy_low_price_mwh']
    df['e_nres_mwh'] = (df['electrolyser_consumption_mwh'] - df['e_total_res_mwh']).clip(lower=0)

    # ============================================
    # PART 1: GHG EMISSION FACTOR  (hourly)
    # EF_H2_h = E_NRES_h × EF_grid / E_H2_h
    # ============================================

    # Total CO₂eq emissions per hour [g CO₂eq]
    # E_NRES [MWh] × 1 000 [kWh/MWh] × EF_grid [g/kWh] = g CO₂eq
    df['total_emissions_g_co2eq'] = df['e_nres_mwh'] * 1000 * country_emission_factor

    # Hourly emission factor [g CO₂eq/kWh]
    # = E_NRES [MWh] × EF_grid [g/kWh] / E_H2 [MWh]   (MWh cancels → g/kWh)
    df['emission_factor_kwh'] = (
        df['e_nres_mwh'] * country_emission_factor
        / df['electrolyser_consumption_mwh'].replace(0, np.nan)
    ).fillna(0)

    # Convert to g CO₂eq/MJ  (1 kWh = 3.6 MJ)
    df['emission_factor_mj'] = df['emission_factor_kwh'].apply(convert_emission_factor_kwh_to_mj)

    # Hourly emission compliance check
    df['is_emission_compliant'] = df['emission_factor_kwh'].apply(is_emission_compliant)

    # ============================================
    # PART 2: RFNBO SHARE  (hourly building block)
    # ============================================

    # α_gridRES – constant grid renewable mix share (regulation does not allow time-varying values)
    df['grid_renewable_share_mix'] = float(renewable_share)

    # E_gridRES = E_grid × α_gridRES
    df['e_grid_res_mwh'] = df['grid_energy_mwh'] * df['grid_renewable_share_mix']

    # E_RFNBO = E_H2              if DA price < 20 €/MWh   (all consumption qualifies)
    #         = E_PPA + E_gridRES  otherwise
    df['rfnbo_energy_mwh'] = np.where(
        df['is_low_price'],
        df['electrolyser_consumption_mwh'],
        df['ppa_energy_mwh'] + df['e_grid_res_mwh']
    )
    # Cap at E_H2 (can't have more RFNBO energy than consumed)
    df['rfnbo_energy_mwh'] = np.minimum(df['rfnbo_energy_mwh'], df['electrolyser_consumption_mwh'])

    # Hourly RFNBO fraction = E_RFNBO / E_H2
    df['rfnbo_fraction'] = (
        df['rfnbo_energy_mwh']
        / df['electrolyser_consumption_mwh'].replace(0, np.nan)
    ).fillna(0).clip(upper=1.0)

    # -----------------------------------------------------------------------
    # Regulatory link: if GHG criterion is NOT met within the temporal
    # correlation window, the hydrogen produced does not qualify as RFNBO.
    #
    # • Hourly correlation  → check is per-hour; zero RFNBO immediately here.
    # • Monthly correlation → zeroing happens after monthly aggregation in
    #   aggregate_to_monthly(); keep raw hourly values intact here.
    # -----------------------------------------------------------------------
    if temporal_correlation == 'hourly':
        df['rfnbo_energy_mwh'] = np.where(df['is_emission_compliant'], df['rfnbo_energy_mwh'], 0.0)
        df['rfnbo_fraction']   = np.where(df['is_emission_compliant'], df['rfnbo_fraction'],   0.0)

    # Hourly 100 % RFNBO flag
    df['is_rfnbo_100pct'] = df['rfnbo_energy_mwh'] >= df['electrolyser_consumption_mwh']

    # Overall hourly compliance: emission AND RFNBO checks
    df['is_compliant'] = df['is_emission_compliant'] & df['is_rfnbo_100pct']

    # Breakdown columns for visualization
    if temporal_correlation == 'hourly':
        # Zero breakdown when GHG not met
        df['rfnbo_from_ppa_mwh'] = np.where(df['is_emission_compliant'], df['ppa_energy_mwh'], 0.0)
        df['rfnbo_from_grid_low_price_mwh'] = np.where(
            df['is_emission_compliant'],
            df['grid_energy_low_price_mwh'],
            0.0
        )
        df['rfnbo_from_grid_normal_price_mwh'] = np.where(
            df['is_emission_compliant'] & ~df['is_low_price'],
            df['e_grid_res_mwh'],
            0.0
        )
    else:
        # Monthly: keep raw hourly values; zeroing happens in aggregate_to_monthly()
        df['rfnbo_from_ppa_mwh'] = df['ppa_energy_mwh']
        df['rfnbo_from_grid_low_price_mwh'] = df['grid_energy_low_price_mwh']
        df['rfnbo_from_grid_normal_price_mwh'] = np.where(
            df['is_low_price'], 0.0, df['e_grid_res_mwh']
        )

    return df


def aggregate_to_monthly(hourly_df: pd.DataFrame, temporal_correlation: str = 'hourly') -> pd.DataFrame:
    """
    Aggregate hourly RFNBO results to a period summary.

    Implements the correct sum-based formulas for GHG emission factor and
    RFNBO share over the full period (monthly or any window):

        EF_H2      = Σ(E_NRES) × EF_grid / Σ(E_H2)
                   = Σ(total_emissions_g_co2eq) / (Σ(E_H2) × 1 000)
        %H2_RFNBO  = 100 × Σ(E_RFNBO) / Σ(E_H2)

    For **monthly** correlation the GHG check is applied at the period level:
    if the period EF_H2 ≥ 28.2 g CO₂eq/MJ, the entire period does not qualify
    as RFNBO and all RFNBO fields are set to zero.

    For **hourly** correlation the per-hour GHG zeroing was already applied in
    calculate_rfnbo_compliance(); this function just sums those zeroed values.

    Args:
        hourly_df: DataFrame produced by calculate_rfnbo_compliance()
        temporal_correlation: 'hourly' or 'monthly' (default 'hourly')

    Returns:
        Single-row DataFrame with period summary:
        - total_consumption_mwh         : Σ E_H2 [MWh]
        - ppa_energy_mwh                : Σ E_PPA [MWh]
        - grid_energy_mwh               : Σ E_grid [MWh]
        - grid_energy_low_price_mwh     : Σ E_DAPrices [MWh]
        - grid_energy_normal_price_mwh  : Σ normal-price grid [MWh]
        - e_total_res_mwh               : Σ E_totalRES [MWh]
        - e_nres_mwh                    : Σ E_NRES [MWh]
        - total_emissions_g_co2eq       : Σ CO₂eq emissions [g]
        - emission_factor_kwh           : period EF_H2 [g CO₂eq/kWh]  (sum-based)
        - emission_factor_mj            : period EF_H2 [g CO₂eq/MJ]   (sum-based)
        - emission_compliant_hours      : hours with hourly EF < threshold
        - rfnbo_energy_mwh              : Σ E_RFNBO [MWh]  (0 if monthly GHG fails)
        - non_rfnbo_energy_mwh          : Σ(E_H2) − Σ(E_RFNBO) [MWh]
        - rfnbo_fraction                : Σ(E_RFNBO) / Σ(E_H2)  (0 if monthly GHG fails)
        - rfnbo_pct                     : %H2_RFNBO (0 if monthly GHG fails)
        - rfnbo_100pct_hours            : hours with hourly rfnbo_fraction ≥ 1
        - compliant_hours               : hours passing both checks
        - total_hours                   : total number of hours
        - emission_compliance_rate      : fraction of hours with hourly EF compliant
        - rfnbo_compliance_rate         : fraction of hours with 100 % RFNBO
        - overall_compliance_rate       : fraction of hours fully compliant
        - period_emission_compliant     : period EF < 28.2 g CO₂eq/MJ
        - period_rfnbo_compliant        : period RFNBO fraction ≥ 1.0

    Example:
        >>> results = calculate_rfnbo_compliance(...)
        >>> summary = aggregate_to_monthly(results, temporal_correlation='monthly')
        >>> print(f"Period RFNBO: {summary['rfnbo_pct'].values[0]:.1f}%")
        >>> print(f"Emission factor: {summary['emission_factor_mj'].values[0]:.2f} g CO₂eq/MJ")
    """
    if hourly_df.empty:
        return pd.DataFrame()

    hourly_df = hourly_df.copy()
    total_consumption = hourly_df['electrolyser_consumption_mwh'].sum()
    total_rfnbo = hourly_df['rfnbo_energy_mwh'].sum()
    total_hours = len(hourly_df)

    # -----------------------------------------------------------------------
    # Period GHG Emission Factor
    # EF_H2 = Σ(total_emissions_g_co2eq) / (Σ(E_H2) × 1 000)
    #       = Σ(E_NRES × EF_grid × 1000) / (Σ(E_H2) × 1000)
    #       = Σ(E_NRES) × EF_grid / Σ(E_H2)     [g CO₂eq/kWh]
    # -----------------------------------------------------------------------
    total_emissions_g = hourly_df['total_emissions_g_co2eq'].sum()
    period_ef_kwh = (
        total_emissions_g / (total_consumption * 1000)
        if total_consumption > 0 else 0.0
    )
    period_ef_mj = convert_emission_factor_kwh_to_mj(period_ef_kwh)
    period_ghg_compliant = period_ef_mj < MAX_EMISSION_FACTOR_MJ

    # -----------------------------------------------------------------------
    # Period RFNBO share
    # %H2_RFNBO = 100 × Σ(E_RFNBO) / Σ(E_H2)
    #
    # For monthly correlation: if the period GHG check fails, the entire
    # period does not qualify as RFNBO → force RFNBO to 0.
    # -----------------------------------------------------------------------
    if temporal_correlation == 'monthly' and not period_ghg_compliant:
        total_rfnbo = 0.0

    period_rfnbo_fraction = total_rfnbo / total_consumption if total_consumption > 0 else 0.0

    monthly_summary = {
        # Energy volumes
        'total_consumption_mwh': total_consumption,
        'ppa_energy_mwh': hourly_df['ppa_energy_mwh'].sum(),
        'grid_energy_mwh': hourly_df['grid_energy_mwh'].sum(),
        'grid_energy_low_price_mwh': hourly_df['grid_energy_low_price_mwh'].sum(),
        'grid_energy_normal_price_mwh': hourly_df['grid_energy_normal_price_mwh'].sum(),
        'e_total_res_mwh': hourly_df['e_total_res_mwh'].sum(),
        'e_nres_mwh': hourly_df['e_nres_mwh'].sum(),
        'total_emissions_g_co2eq': total_emissions_g,
        # Period GHG emission factor (sum-based, not simple hourly average)
        'emission_factor_kwh': period_ef_kwh,
        'emission_factor_mj': period_ef_mj,
        # Period RFNBO share (sum-based; 0 if monthly GHG fails)
        'rfnbo_energy_mwh': total_rfnbo,
        'non_rfnbo_energy_mwh': total_consumption - total_rfnbo,
        'rfnbo_fraction': period_rfnbo_fraction,
        'rfnbo_pct': period_rfnbo_fraction * 100,
        # Hourly compliance counts
        'emission_compliant_hours': int(hourly_df['is_emission_compliant'].sum()),
        'rfnbo_100pct_hours': int(hourly_df['is_rfnbo_100pct'].sum()),
        'compliant_hours': int(hourly_df['is_compliant'].sum()),
        'total_hours': total_hours,
        # Hourly compliance rates
        'emission_compliance_rate': hourly_df['is_emission_compliant'].sum() / total_hours,
        'rfnbo_compliance_rate': hourly_df['is_rfnbo_100pct'].sum() / total_hours,
        'overall_compliance_rate': hourly_df['is_compliant'].sum() / total_hours,
        # Period-level compliance flags
        'period_emission_compliant': period_ghg_compliant,
        'period_rfnbo_compliant': period_rfnbo_fraction >= 1.0,
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
    
    # Use sum-based period emission factor and RFNBO fraction from aggregate_to_monthly
    emission_factor = monthly_summary['emission_factor_mj'].values[0]
    rfnbo_fraction = monthly_summary['rfnbo_fraction'].values[0]

    emission_compliant = emission_factor < MAX_EMISSION_FACTOR_MJ
    rfnbo_compliant = rfnbo_fraction >= 1.0  # 100 % RFNBO

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
    def integrate_group(df_group):
        """Wrapper to apply integration to each PSR type group."""
        return integrate_power_to_energy(
            df_group,
            power_column='generation_mw',
            energy_column='generation_mwh',
            resolution_column='resolution_minutes',
            timestamp_column='timestamp'
        )
    
    gen_df = gen_df.groupby('psr_type', group_keys=False).apply(integrate_group)
    
    total_gen = gen_df['generation_mwh'].sum()
    renewable_gen = gen_df[gen_df['is_renewable']]['generation_mwh'].sum()
    
    return {
        'total_records': len(gen_df),
        'total_generation': total_gen,
        'renewable_generation': renewable_gen,
        'renewable_share': (renewable_gen / total_gen * 100) if total_gen > 0 else 0,
        'unique_sources': gen_df['psr_type'].nunique()
    }

