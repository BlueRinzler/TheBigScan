import os
import logging
from typing import Dict
import pandas as pd
from dateutil.utils import today
from dotenv import load_dotenv

import api
import rankings
import compute
import scanner

# ----------------------------------------------------------------------
# Configuration – load once, use everywhere
# ----------------------------------------------------------------------
def load_config() -> Dict[str, str]:
    """
    Load .env once and return a dictionary of all necessary file paths.
    This is the single source of truth – no more scattered os.getenv calls.
    """
    load_dotenv()
    # All required keys from .env
    keys = [
        'SYMBOLS', 'OHLCV_DATA_RAW', 'COMPUTED_DATA_RAW', 'SECTOR_RANKINGS',
        'ONE_MONTH_GAINERS', 'THREE_MONTH_GAINERS', 'SIX_MONTH_GAINERS',
        'ONE_YEAR_GAINERS',
        'ONE_MONTH_MOMENTUM', 'THREE_MONTH_MOMENTUM', 'SIX_MONTH_MOMENTUM',
        'ONE_YEAR_MOMENTUM',
        'CONSOLIDATION_SYMBOLS', 'MOMENTUM_SYMBOLS'
    ]
    config = {}
    for key in keys:
        value = os.getenv(key)
        if value is None:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        config[key] = value
    return config

# ----------------------------------------------------------------------
# Pipelines – each major step is a pure function accepting config
# ----------------------------------------------------------------------
def run_api_step(config: Dict[str, str]) -> None:
    """Fetch latest OHLCV data from Schwab API."""
    start_date = (today() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    api.fetch_OHLCV(config['SYMBOLS'], config['OHLCV_DATA_RAW'], start_date)
    logging.info("API data fetched and saved.")

def compute_and_rank(config: Dict[str, str]) -> None:
    """Compute indicators and generate sector rankings."""
    compute.compute_data(config['OHLCV_DATA_RAW'], config['COMPUTED_DATA_RAW'])
    rankings.generate_sector_rankings(
        config['COMPUTED_DATA_RAW'],
        config['SYMBOLS'],
        config['SECTOR_RANKINGS']
    )
    logging.info("Computed data and sector rankings created.")

def run_scanners(config: Dict[str, str]) -> None:
    """Run consolidation and momentum scanners for multiple lookback periods."""
    periods = [
        (config['ONE_MONTH_GAINERS'],  21),
        (config['THREE_MONTH_GAINERS'], 63),
        (config['SIX_MONTH_GAINERS'],  126),
        (config['ONE_YEAR_GAINERS'],   249),
    ]
    for output_path, days in periods:
        scanner.filter_consolidation(config['COMPUTED_DATA_RAW'], output_path, time=days)
        logging.debug(f"Consolidation scan {days}d → {output_path}")

    momentum_periods = [
        (config['ONE_MONTH_MOMENTUM'],  21),
        (config['THREE_MONTH_MOMENTUM'], 63),
        (config['SIX_MONTH_MOMENTUM'],  126),
        (config['ONE_YEAR_MOMENTUM'],   249),
    ]
    for output_path, days in momentum_periods:
        scanner.filter_momentum(config['COMPUTED_DATA_RAW'], output_path, time=days)
        logging.debug(f"Momentum scan {days}d → {output_path}")

def combine_scanner_results(config: Dict[str, str]) -> None:
    """Combine scanner outputs with sector rankings into two final CSVs."""
    # Consolidation
    combine_csv(
        config['ONE_MONTH_GAINERS'],
        config['THREE_MONTH_GAINERS'],
        config['SIX_MONTH_GAINERS'],
        config['ONE_YEAR_GAINERS'],
        config['SECTOR_RANKINGS'],
        config['CONSOLIDATION_SYMBOLS']
    )
    # Momentum
    combine_csv(
        config['ONE_MONTH_MOMENTUM'],
        config['THREE_MONTH_MOMENTUM'],
        config['SIX_MONTH_MOMENTUM'],
        config['ONE_YEAR_MOMENTUM'],
        config['SECTOR_RANKINGS'],
        config['MOMENTUM_SYMBOLS']
    )
    logging.info("Final combined CSVs created.")

# ----------------------------------------------------------------------
# Helper – remains unchanged but moved to the module level
# ----------------------------------------------------------------------
def combine_csv(one_month, three_month, six_month, one_year, sector_rankings, output):
    """Merge multiple scanner files, drop duplicates, attach sector ranks."""
    df1 = pd.read_csv(one_month)
    df2 = pd.read_csv(three_month)
    df3 = pd.read_csv(six_month)
    df4 = pd.read_csv(one_year)
    df5 = pd.read_csv(sector_rankings)

    df6 = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df6 = df6.drop_duplicates(subset=['symbol', 'datetime'], keep='first')
    final_df = df6.merge(df5, on='symbol', how='left')
    final_df = final_df.sort_values(['symbol'], ascending=False)
    final_df.to_csv(output, index=False)

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config()

    response = input("Do you want to run the API? (y/n): ").strip().lower()
    if response in ("y", "yes"):
        logging.info("Running the API...")
        run_api_step(config)

    # Always run the rest of the pipeline
    compute_and_rank(config)
    run_scanners(config)
    combine_scanner_results(config)
    logging.info("All done!")

if __name__ == '__main__':
    main()