"""
Batch Collection Script for Multiple Countries

This script collects ENTSOE data for multiple countries in sequence.
Useful for building datasets across different regions.

Usage:
    python batch_collect.py --start 2023-01-01 --end 2023-12-31
    python batch_collect.py --start 2020-01-01 --end 2023-12-31 --countries Belgium France Netherlands
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime

# Default countries to collect
DEFAULT_COUNTRIES = [
    'Belgium',
    'France',
    'Netherlands',
    'Germany (DE/LU)',
    'Spain',
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_collection(country: str, start_date: str, end_date: str, 
                   output_dir: str, resume: bool, prices_only: bool) -> bool:
    """
    Run data collection for a single country.
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'collect_entsoe_data.py',
        '--country', country,
        '--start', start_date,
        '--end', end_date,
        '--output', output_dir
    ]
    
    if resume:
        cmd.append('--resume')
    
    if prices_only:
        cmd.append('--prices-only')
    
    logger.info(f"Starting collection for {country}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        logger.info(f"✓ Successfully completed collection for {country}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to collect data for {country}: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"Collection interrupted for {country}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error for {country}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch collect ENTSOE data for multiple countries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect for default countries (Belgium, France, Netherlands, Germany, Spain)
  python batch_collect.py --start 2023-01-01 --end 2023-12-31
  
  # Collect for specific countries
  python batch_collect.py --start 2023-01-01 --end 2023-12-31 --countries Belgium France
  
  # Resume interrupted batch collection
  python batch_collect.py --start 2023-01-01 --end 2023-12-31 --resume
  
  # Quick collection: prices only
  python batch_collect.py --start 2023-01-01 --end 2023-12-31 --prices-only
        """
    )
    
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--countries', nargs='+', default=DEFAULT_COUNTRIES,
                        help='List of countries to collect (default: Belgium, France, Netherlands, Germany, Spain)')
    parser.add_argument('--output', type=str, default='./entsoe_data',
                        help='Output directory (default: ./entsoe_data)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted collections')
    parser.add_argument('--prices-only', action='store_true',
                        help='Only collect price data (faster)')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop batch if any country fails (default: continue)')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        if start > end:
            logger.error("Start date must be before end date")
            sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Display collection plan
    logger.info("=" * 70)
    logger.info("BATCH ENTSOE DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Countries: {', '.join(args.countries)}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Resume Mode: {'Yes' if args.resume else 'No'}")
    logger.info(f"Prices Only: {'Yes' if args.prices_only else 'No'}")
    logger.info("=" * 70)
    
    # Track results
    total = len(args.countries)
    successful = []
    failed = []
    
    # Process each country
    for idx, country in enumerate(args.countries, 1):
        logger.info(f"\n[{idx}/{total}] Processing: {country}")
        logger.info("-" * 70)
        
        try:
            success = run_collection(
                country=country,
                start_date=args.start,
                end_date=args.end,
                output_dir=args.output,
                resume=args.resume,
                prices_only=args.prices_only
            )
            
            if success:
                successful.append(country)
            else:
                failed.append(country)
                if args.stop_on_error:
                    logger.error("Stopping batch collection due to error (--stop-on-error)")
                    break
                    
        except KeyboardInterrupt:
            logger.warning("\n\nBatch collection interrupted by user")
            logger.info("Use --resume to continue the batch collection")
            break
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Countries: {total}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info(f"\n✓ Successful ({len(successful)}):")
        for country in successful:
            logger.info(f"  - {country}")
    
    if failed:
        logger.warning(f"\n✗ Failed ({len(failed)}):")
        for country in failed:
            logger.warning(f"  - {country}")
        logger.info("\nTip: Use --resume to retry failed countries")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Data saved to: {args.output}")
    logger.info("=" * 70)
    
    # Exit code
    if failed and not successful:
        sys.exit(1)  # All failed
    elif failed:
        sys.exit(2)  # Some failed
    else:
        sys.exit(0)  # All successful


if __name__ == "__main__":
    main()

