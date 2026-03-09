"""
ENTSOE Data Collection Script

This script fetches historical data from ENTSOE and stores it in CSV files.
It collects:
- Day-ahead prices
- Generation mix by production type
- Installed capacity by production type

Usage:
    python collect_entsoe_data.py --country Belgium --start 2023-01-01 --end 2023-12-31 --output ./data
    
    python collect_entsoe_data.py --country "Germany (DE/LU)" --start 2022-01-01 --end 2024-12-31
    
    # Resume interrupted collection
    python collect_entsoe_data.py --country Belgium --start 2023-01-01 --end 2023-12-31 --resume
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time
import logging
from typing import Optional

# Import from fetch_entsoe
from fetch_entsoe import (
    fetch_day_ahead_prices,
    fetch_all_generation_types,
    fetch_installed_capacity_all_types,
    get_backend_country_name,
    BIDDING_ZONES
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entsoe_data_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_country(country: str) -> str:
    """Validate that the country is supported by ENTSOE."""
    backend_country = get_backend_country_name(country)
    if backend_country not in BIDDING_ZONES:
        available_countries = sorted(BIDDING_ZONES.keys())
        raise ValueError(
            f"Country '{country}' not found. Available countries:\n" + 
            "\n".join(f"  - {c}" for c in available_countries)
        )
    return backend_country


def validate_dates(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Validate and parse date strings."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    if start > end:
        raise ValueError("Start date must be before end date")
    
    if end > datetime.now():
        logger.warning(f"End date {end_date} is in the future. Setting to today.")
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Warn if dates are very recent
    if (datetime.now() - end).days < 3:
        logger.warning("Requesting very recent data. ENTSOE typically has 1-2 day delay.")
    
    return start, end


def get_date_range(start_date: datetime, end_date: datetime) -> list[str]:
    """Generate list of dates between start and end."""
    date_list = []
    current = start_date
    while current <= end_date:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return date_list


def setup_output_directory(output_dir: str, country: str, start_date: str, end_date: str) -> dict:
    """Create output directory structure and return file paths."""
    base_path = Path(output_dir)
    country_safe = country.replace('/', '_').replace(' ', '_')
    country_path = base_path / country_safe
    
    # Create directories
    country_path.mkdir(parents=True, exist_ok=True)
    
    # Format dates for filenames (remove dashes for cleaner look)
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    
    # Define file paths with date range
    paths = {
        'prices': country_path / f"{country_safe}_prices_{start_str}_{end_str}.csv",
        'generation': country_path / f"{country_safe}_generation_{start_str}_{end_str}.csv",
        'capacity': country_path / f"{country_safe}_installed_capacity_{start_str}_{end_str}.csv"
    }
    
    logger.info(f"Output directory: {country_path}")
    return paths


def load_existing_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load existing CSV data if it exists."""
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded existing data from {file_path.name}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to load existing data from {file_path}: {e}")
            return None
    return None


def get_missing_dates(existing_df: Optional[pd.DataFrame], all_dates: list[str], date_column: str = 'datetime') -> list[str]:
    """Determine which dates are missing from existing data."""
    if existing_df is None or existing_df.empty:
        return all_dates
    
    # Extract dates from existing data
    try:
        existing_df[date_column] = pd.to_datetime(existing_df[date_column])
        existing_dates = set(existing_df[date_column].dt.strftime("%Y-%m-%d").unique())
        missing = [d for d in all_dates if d not in existing_dates]
        logger.info(f"Found {len(existing_dates)} existing dates, {len(missing)} dates to fetch")
        return missing
    except Exception as e:
        logger.warning(f"Could not determine missing dates: {e}. Fetching all dates.")
        return all_dates


def fetch_prices_for_period(country: str, dates: list[str]) -> pd.DataFrame:
    """Fetch day-ahead prices for a list of dates."""
    all_prices = []
    failed_dates = []
    
    print(f"\n{'='*60}")
    print(f"📊 FETCHING PRICES FOR {country.upper()}")
    print(f"{'='*60}")
    print(f"Total dates to fetch: {len(dates)}")
    print(f"{'='*60}\n")
    
    logger.info(f"Fetching prices for {len(dates)} dates...")
    
    for idx, date in enumerate(dates, 1):
        try:
            # Progress indicator
            percent = (idx / len(dates)) * 100
            print(f"[{idx:3d}/{len(dates)}] ({percent:5.1f}%) 📅 {date} - Fetching prices...", end='', flush=True)
            
            logger.info(f"[{idx}/{len(dates)}] Fetching prices for {date}...")
            df = fetch_day_ahead_prices(date, country)
            
            if not df.empty:
                all_prices.append(df)
                print(f" ✓ {len(df)} records")
                logger.info(f"  ✓ Got {len(df)} price records")
            else:
                print(f" ⚠ No data")
                logger.warning(f"  ⚠ No price data for {date}")
                failed_dates.append(date)
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f" ✗ Failed: {e}")
            logger.error(f"  ✗ Failed to fetch prices for {date}: {e}")
            failed_dates.append(date)
            time.sleep(2)  # Extra delay after error
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 PRICE COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(dates) - len(failed_dates)}/{len(dates)} dates")
    if failed_dates:
        print(f"✗ Failed: {len(failed_dates)} dates")
        logger.warning(f"Failed to fetch prices for {len(failed_dates)} dates: {failed_dates[:10]}...")
    print(f"{'='*60}\n")
    
    if all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        logger.info(f"Successfully fetched {len(combined)} total price records")
        return combined
    else:
        logger.warning("No price data collected")
        return pd.DataFrame()


def fetch_generation_for_period(country: str, dates: list[str]) -> pd.DataFrame:
    """Fetch generation mix for a list of dates."""
    all_generation = []
    failed_dates = []
    
    print(f"\n{'='*60}")
    print(f"⚡ FETCHING GENERATION MIX FOR {country.upper()}")
    print(f"{'='*60}")
    print(f"Total dates to fetch: {len(dates)}")
    print(f"Note: Generation queries ~20 PSR types per date (slower)")
    print(f"{'='*60}\n")
    
    logger.info(f"Fetching generation data for {len(dates)} dates...")
    logger.info("Note: Generation fetching queries multiple PSR types per date (takes longer)")
    
    for idx, date in enumerate(dates, 1):
        try:
            # Progress indicator
            percent = (idx / len(dates)) * 100
            print(f"[{idx:3d}/{len(dates)}] ({percent:5.1f}%) 📅 {date} - Fetching generation...", end='', flush=True)
            
            logger.info(f"[{idx}/{len(dates)}] Fetching generation for {date}...")
            df = fetch_all_generation_types(date, country)
            
            if not df.empty:
                all_generation.append(df)
                unique_types = df['psr_type'].nunique()
                print(f" ✓ {len(df)} records ({unique_types} PSR types)")
                logger.info(f"  ✓ Got {len(df)} generation records ({unique_types} PSR types)")
            else:
                print(f" ⚠ No data")
                logger.warning(f"  ⚠ No generation data for {date}")
                failed_dates.append(date)
            
            # Longer delay for generation (multiple API calls per date)
            time.sleep(1)
            
        except Exception as e:
            print(f" ✗ Failed: {str(e)[:50]}")
            logger.error(f"  ✗ Failed to fetch generation for {date}: {e}")
            failed_dates.append(date)
            time.sleep(3)  # Extra delay after error
    
    # Summary
    print(f"\n{'='*60}")
    print(f"⚡ GENERATION COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(dates) - len(failed_dates)}/{len(dates)} dates")
    if failed_dates:
        print(f"✗ Failed: {len(failed_dates)} dates")
        logger.warning(f"Failed to fetch generation for {len(failed_dates)} dates: {failed_dates[:10]}...")
    print(f"{'='*60}\n")
    
    if all_generation:
        combined = pd.concat(all_generation, ignore_index=True)
        logger.info(f"Successfully fetched {len(combined)} total generation records")
        return combined
    else:
        logger.warning("No generation data collected")
        return pd.DataFrame()


def fetch_capacity_for_period(country: str, dates: list[str]) -> pd.DataFrame:
    """Fetch installed capacity for a list of dates."""
    # Installed capacity typically doesn't change often, so we can sample less frequently
    # Fetch once per month
    sampled_dates = []
    seen_months = set()
    
    for date in dates:
        year_month = date[:7]  # YYYY-MM
        if year_month not in seen_months:
            sampled_dates.append(date)
            seen_months.add(year_month)
    
    print(f"\n{'='*60}")
    print(f"🏭 FETCHING INSTALLED CAPACITY FOR {country.upper()}")
    print(f"{'='*60}")
    print(f"Total dates requested: {len(dates)}")
    print(f"Monthly samples to fetch: {len(sampled_dates)}")
    print(f"(Capacity data is sampled monthly as it changes infrequently)")
    print(f"{'='*60}\n")
    
    logger.info(f"Fetching installed capacity for {len(sampled_dates)} dates (monthly samples)...")
    
    all_capacity = []
    failed_dates = []
    
    for idx, date in enumerate(sampled_dates, 1):
        try:
            # Progress indicator
            percent = (idx / len(sampled_dates)) * 100
            print(f"[{idx:3d}/{len(sampled_dates)}] ({percent:5.1f}%) 📅 {date} - Fetching capacity...", end='', flush=True)
            
            logger.info(f"[{idx}/{len(sampled_dates)}] Fetching capacity for {date}...")
            df = fetch_installed_capacity_all_types(date, country)
            
            if not df.empty:
                all_capacity.append(df)
                unique_types = df['psr_type'].nunique()
                print(f" ✓ {len(df)} records ({unique_types} PSR types)")
                logger.info(f"  ✓ Got {len(df)} capacity records ({unique_types} PSR types)")
            else:
                print(f" ⚠ No data")
                logger.warning(f"  ⚠ No capacity data for {date}")
                failed_dates.append(date)
            
            time.sleep(1)
            
        except Exception as e:
            print(f" ✗ Failed: {str(e)[:50]}")
            logger.error(f"  ✗ Failed to fetch capacity for {date}: {e}")
            failed_dates.append(date)
            time.sleep(3)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏭 CAPACITY COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(sampled_dates) - len(failed_dates)}/{len(sampled_dates)} samples")
    if failed_dates:
        print(f"✗ Failed: {len(failed_dates)} samples")
        logger.warning(f"Failed to fetch capacity for {len(failed_dates)} dates: {failed_dates[:10]}...")
    print(f"{'='*60}\n")
    
    if all_capacity:
        combined = pd.concat(all_capacity, ignore_index=True)
        logger.info(f"Successfully fetched {len(combined)} total capacity records")
        return combined
    else:
        logger.warning("No capacity data collected")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, file_path: Path, existing_df: Optional[pd.DataFrame] = None):
    """Save data to CSV, merging with existing data if present."""
    if df.empty:
        logger.warning(f"No data to save to {file_path.name}")
        return
    
    # Merge with existing data if present
    if existing_df is not None and not existing_df.empty:
        logger.info(f"Merging with existing data ({len(existing_df)} records)...")
        combined = pd.concat([existing_df, df], ignore_index=True)
        
        # Remove duplicates based on all columns
        before_dedup = len(combined)
        combined = combined.drop_duplicates()
        after_dedup = len(combined)
        
        if before_dedup > after_dedup:
            logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        # Sort by timestamp/datetime
        if 'datetime' in combined.columns:
            combined = combined.sort_values('datetime')
        elif 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp')
        
        df = combined
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    logger.info(f"✓ Saved {len(df)} records to {file_path.name}")




def main():
    parser = argparse.ArgumentParser(
        description='Collect historical data from ENTSOE and save to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect one year of data for Belgium
  python collect_entsoe_data.py --country Belgium --start 2023-01-01 --end 2023-12-31
  
  # Collect data for Germany
  python collect_entsoe_data.py --country "Germany (DE/LU)" --start 2022-01-01 --end 2024-12-31
  
  # Resume interrupted collection
  python collect_entsoe_data.py --country Belgium --start 2023-01-01 --end 2023-12-31 --resume
  
  # Specify output directory
  python collect_entsoe_data.py --country France --start 2023-01-01 --end 2023-12-31 --output ./my_data
  
Available Countries:
  Run with --list-countries to see all available countries
        """
    )
    
    parser.add_argument('--country', type=str, help='Country name (e.g., "Belgium", "Germany (DE/LU)")')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='./entsoe_data', 
                        help='Output directory (default: ./entsoe_data)')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume interrupted collection (skip already collected dates)')
    parser.add_argument('--prices-only', action='store_true', 
                        help='Only collect price data')
    parser.add_argument('--generation-only', action='store_true', 
                        help='Only collect generation data')
    parser.add_argument('--capacity-only', action='store_true', 
                        help='Only collect installed capacity data')
    parser.add_argument('--list-countries', action='store_true', 
                        help='List all available countries and exit')
    
    args = parser.parse_args()
    
    # List countries and exit
    if args.list_countries:
        print("\nAvailable Countries:")
        print("=" * 50)
        for country in sorted(BIDDING_ZONES.keys()):
            print(f"  {country}")
        print("\nNote: Use quotes for countries with spaces or special characters")
        print('Example: --country "Germany (DE/LU)"')
        return
    
    # Validate required arguments
    if not all([args.country, args.start, args.end]):
        parser.print_help()
        print("\nError: --country, --start, and --end are required")
        print("Use --list-countries to see available countries")
        sys.exit(1)
    
    try:
        # Validate inputs
        logger.info("=" * 60)
        logger.info("ENTSOE Data Collection Script")
        logger.info("=" * 60)
        
        country = validate_country(args.country)
        logger.info(f"Country: {country}")
        
        start_date, end_date = validate_dates(args.start, args.end)
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        
        num_days = (end_date - start_date).days + 1
        logger.info(f"Total Days: {num_days}")
        
        # Setup output directory with date range
        paths = setup_output_directory(args.output, country, args.start, args.end)
        
        # Generate date range
        all_dates = get_date_range(start_date, end_date)
        
        # Determine what to collect
        collect_prices = not (args.generation_only or args.capacity_only)
        collect_generation = not (args.prices_only or args.capacity_only)
        collect_capacity = not (args.prices_only or args.generation_only)
        
        # Track results
        prices_df = pd.DataFrame()
        generation_df = pd.DataFrame()
        capacity_df = pd.DataFrame()
        
        # PRICES
        if collect_prices:
            logger.info("\n" + "=" * 60)
            logger.info("COLLECTING DAY-AHEAD PRICES")
            logger.info("=" * 60)
            
            existing_prices = load_existing_data(paths['prices']) if args.resume else None
            dates_to_fetch = get_missing_dates(existing_prices, all_dates, 'datetime') if args.resume else all_dates
            
            if dates_to_fetch:
                new_prices = fetch_prices_for_period(country, dates_to_fetch)
                save_data(new_prices, paths['prices'], existing_prices)
                # Reload to get the complete merged dataset
                loaded_prices = load_existing_data(paths['prices'])
                prices_df = loaded_prices if loaded_prices is not None else new_prices
            else:
                logger.info("All price data already collected (use without --resume to re-fetch)")
                prices_df = existing_prices if existing_prices is not None else pd.DataFrame()
        
        # GENERATION
        if collect_generation:
            logger.info("\n" + "=" * 60)
            logger.info("COLLECTING GENERATION MIX")
            logger.info("=" * 60)
            
            existing_generation = load_existing_data(paths['generation']) if args.resume else None
            dates_to_fetch = get_missing_dates(existing_generation, all_dates, 'timestamp') if args.resume else all_dates
            
            if dates_to_fetch:
                new_generation = fetch_generation_for_period(country, dates_to_fetch)
                save_data(new_generation, paths['generation'], existing_generation)
                # Reload to get the complete merged dataset
                loaded_generation = load_existing_data(paths['generation'])
                generation_df = loaded_generation if loaded_generation is not None else new_generation
            else:
                logger.info("All generation data already collected (use without --resume to re-fetch)")
                generation_df = existing_generation if existing_generation is not None else pd.DataFrame()
        
        # INSTALLED CAPACITY
        if collect_capacity:
            logger.info("\n" + "=" * 60)
            logger.info("COLLECTING INSTALLED CAPACITY")
            logger.info("=" * 60)
            
            existing_capacity = load_existing_data(paths['capacity']) if args.resume else None
            dates_to_fetch = get_missing_dates(existing_capacity, all_dates, 'timestamp') if args.resume else all_dates
            
            if dates_to_fetch:
                new_capacity = fetch_capacity_for_period(country, dates_to_fetch)
                save_data(new_capacity, paths['capacity'], existing_capacity)
                # Reload to get the complete merged dataset
                loaded_capacity = load_existing_data(paths['capacity'])
                capacity_df = loaded_capacity if loaded_capacity is not None else new_capacity
            else:
                logger.info("All capacity data already collected (use without --resume to re-fetch)")
                capacity_df = existing_capacity if existing_capacity is not None else pd.DataFrame()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        
        logger.info(f"\nData saved to: {paths['prices'].parent}")
        logger.info(f"  - Prices: {len(prices_df):,} records")
        logger.info(f"  - Generation: {len(generation_df):,} records")
        logger.info(f"  - Capacity: {len(capacity_df):,} records")
        
        print(f"\n{'='*60}")
        print(f"✅ COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Country: {country}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Output: {paths['prices'].parent}")
        print(f"\nFiles created:")
        print(f"  📊 Prices:     {paths['prices'].name} ({len(prices_df):,} records)")
        print(f"  ⚡ Generation: {paths['generation'].name} ({len(generation_df):,} records)")
        print(f"  🏭 Capacity:   {paths['capacity'].name} ({len(capacity_df):,} records)")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        logger.warning("\n\nCollection interrupted by user. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

