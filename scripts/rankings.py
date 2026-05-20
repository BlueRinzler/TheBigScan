import pandas as pd
import numpy as np


def generate_sector_rankings(ohclv_path, sector_path, output_path, rolling_window=2):
    """
    Full pipeline:
      1. 5‑day (or rolling_window) average of %Year, %6Month, %3Month, %1Month per symbol.
      2. Intra‑sector ranks for each period (1 = best).
      3. Composite score (average of the four ranks) and a final sector_rank.

    Parameters
    ----------
    csv_file_path : str
    rolling_window : int, default 2
        Number of days to average over.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, sector, industry, %Year_rank, %6Month_rank, %3Month_rank,
                 %1Month_rank, composite_score, sector_rank.
        Sorted by sector then sector_rank.
        :param rolling_window:
        :param output_path:
        :param sector_path:
        :param ohclv_path:
    """
    # ---------- Step 1: Read & average ----------
    df = pd.read_csv(ohclv_path)
    df2 = pd.read_csv(sector_path)
    df = df.sort_values(['symbol', 'datetime'])

    perf_cols = ['%Year', '%6Month', '%3Month', '%1Month']

    # Rolling average per symbol, keep the last value
    averaged = (
        df.groupby('symbol')
        .apply(lambda g: g[perf_cols].rolling(window=rolling_window, min_periods=1).mean().iloc[-1])
        .reset_index()
    )

    # Attach sector & industry (take first occurrence per symbol)
    sector_map = df2.set_index('Symbol')['Sector']
    industry_map = df2.set_index('Symbol')['Industry']
    df['sector'] = df['symbol'].str.upper().map(sector_map)
    df['industry'] = df['symbol'].str.upper().map(industry_map)
    symbol_info = df[['symbol', 'sector', 'industry']].drop_duplicates('symbol')
    averaged = averaged.merge(symbol_info, on='symbol')

    # ---------- Step 2: Rank within each sector ----------
    result = averaged.copy()
    # Clean infinities
    result[perf_cols] = result[perf_cols].replace([np.inf, -np.inf], np.nan)

    rank_cols = [f'{c}_rank' for c in perf_cols]
    for rc in rank_cols:
        result[rc] = pd.NA

    # IMPORTANT: rank within SECTOR, not industry
    for sector, sector_df in result.groupby('industry'):
        for pc, rc in zip(perf_cols, rank_cols):
            sorted_idx = sector_df[pc].sort_values(ascending=False).index
            result.loc[sorted_idx, rc] = range(1, len(sector_df) + 1)

    # ---------- Step 3: Composite score & final rank ----------
    result['composite_score'] = result[rank_cols].mean(axis=1)

    result['sector_rank'] = (
        result.groupby('sector')['composite_score']
        .rank(ascending=True, method='min')
        .astype('Int64')
    )
    # ---------- Finalise ----------
    output_cols = ['symbol', 'sector', 'industry'] + rank_cols + ['composite_score', 'sector_rank']
    result = result[output_cols].sort_values(['industry', 'sector_rank']).reset_index(drop=True)
    result.to_csv(output_path, index=False)
    return result
