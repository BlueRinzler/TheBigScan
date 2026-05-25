import pandas as pd
import numpy as np
from typing import Optional


def average_latest_performance(
    ohlcv: pd.DataFrame,
    rolling_window: int = 2,
    perf_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    For each symbol, compute the rolling mean of `perf_cols` and return
    the latest averaged values (1 row per symbol).
    """
    if perf_cols is None:
        perf_cols = ['%Year', '%6Month', '%3Month', '%1Month']

    # Ensure datetime ordering (assumes a 'datetime' column exists)
    ohlcv = ohlcv.sort_values(['symbol', 'datetime'])

    def _latest_avg(group):
        rolled = group[perf_cols].rolling(window=rolling_window, min_periods=1).mean()
        return rolled.iloc[-1]   # last row of the group → most recent date

    averaged = (
        ohlcv.groupby('symbol')
              .apply(_latest_avg)
              .reset_index()
    )
    return averaged


def attach_industry(
    performance: pd.DataFrame,
    sector_info: pd.DataFrame,
    symbol_col: str = 'symbol',
    sector_symbol_col: str = 'Symbol',
    industry_col: str = 'Industry'
) -> pd.DataFrame:
    """
    Add an 'industry' column to `performance` by mapping symbol -> industry.
    Uses the first occurrence if a symbol appears multiple times in sector_info.
    """
    # Build a clean mapping; take first industry per symbol (case‑insensitive)
    mapping = (
        sector_info[[sector_symbol_col, industry_col]]
        .drop_duplicates(sector_symbol_col)
        .set_index(sector_symbol_col)[industry_col]
    )
    performance = performance.copy()
    performance['industry'] = performance[symbol_col].str.upper().map(mapping)
    return performance


def add_intra_industry_ranks(
    df: pd.DataFrame,
    perf_cols: Optional[list] = None,
    rank_suffix: str = '_rank'
) -> pd.DataFrame:
    """
    Within each industry, rank symbols by each performance column.
    Highest value gets rank 1. Missing values are ignored and remain NaN.
    """
    if perf_cols is None:
        perf_cols = ['%Year', '%6Month', '%3Month', '%1Month']

    df = df.copy()
    # Clean infinities
    df[perf_cols] = df[perf_cols].replace([np.inf, -np.inf], np.nan)

    rank_cols = [f'{c}{rank_suffix}' for c in perf_cols]
    for rc in rank_cols:
        df[rc] = pd.NA

    for industry, ind_df in df.groupby('industry'):
        for pc, rc in zip(perf_cols, rank_cols):
            # Sort descending: highest % → rank 1
            sorted_idx = ind_df[pc].sort_values(ascending=False).index
            # Assign ranks 1..len(ind_df)
            df.loc[sorted_idx, rc] = range(1, len(ind_df) + 1)

    return df


def add_industry_stock_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'industry_stock_count' = number of symbols per industry."""
    counts = df.groupby('industry').size().rename('industry_stock_count')
    return df.merge(counts, on='industry')


# ----------------------------------------------------------------------
# Pure pipeline – orchestrates the logic, returns a DataFrame
# ----------------------------------------------------------------------

def compute_sector_rankings(
    ohlcv: pd.DataFrame,
    sector_info: pd.DataFrame,
    rolling_window: int = 2
) -> pd.DataFrame:
    """
    Full ranking computation, all steps:
    1. Average latest performance per symbol
    2. Attach industry
    3. Intra‑industry ranks
    4. Industry stock count
    Returns a clean DataFrame with columns:
    ['symbol', 'industry', '%Year_rank', '%6Month_rank', '%3Month_rank', '%1Month_rank', 'industry_stock_count']
    """
    # Step 1
    perf = average_latest_performance(ohlcv, rolling_window)
    # Step 2
    with_industry = attach_industry(perf, sector_info)
    # Step 3
    with_ranks = add_intra_industry_ranks(with_industry)
    # Step 4
    final = add_industry_stock_count(with_ranks)

    # Clean output columns
    output_cols = ['symbol', 'industry'] + \
                  [f'{c}_rank' for c in ['%Year','%6Month','%3Month','%1Month']] + \
                  ['industry_stock_count']
    final = final[output_cols].sort_values('industry').reset_index(drop=True)
    return final


# ----------------------------------------------------------------------
# I/O wrapper – original signature preserved, but easily mockable
# ----------------------------------------------------------------------

def generate_sector_rankings(
    ohclv_path: str,
    sector_path: str,
    output_path: str,
    rolling_window: int = 2
) -> None:
    ohlcv = pd.read_csv(ohclv_path)
    sector = pd.read_csv(sector_path)
    result = compute_sector_rankings(ohlcv, sector, rolling_window)
    result.to_csv(output_path, index=False)
    print("Rankings completed...")