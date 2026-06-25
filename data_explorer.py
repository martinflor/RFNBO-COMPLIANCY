########################################################################################################################################"
#Has nothing to do with RFNBO but useful for data exploration and understanding price/generation patterns.
##########################################################################################################################################
import pandas as pd
from rfnbo_calculations import integrate_power_to_energy, PSR_TYPE_MAPPING, RENEWABLE_PSR_TYPES

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