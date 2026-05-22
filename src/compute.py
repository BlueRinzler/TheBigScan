import pandas as pd
import os


def load_ohlcv_from_cache(csv_path):
    """Load raw OHLCV data from CSV."""
    if not os.path.exists(csv_path):
        print(f"Cache file {csv_path} not found. Run fetch_and_save_ohlcv first.")
        return None
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    print(f"Loaded cached data: {df['symbol'].nunique()} symbols, {len(df)} rows")
    return df



def compute_data(ohlcv_raw, output_csv, sma_10=10, sma_20=20, sma_50=50, window=20):
    """
    Read OHLV.csv, compute 10-day and 20-day SMA per symbol, and save to OHLVSMA.csv.
    Returns the enriched DataFrame.
    """
    # Load raw OHLCV data
    df = load_ohlcv_from_cache(ohlcv_raw)
    if df is None or df.empty:
        print("Cannot compute SMA: missing input data.")
        return None
    # Sort by symbol and date (required for rolling)
    df = df.sort_values(['symbol', 'datetime'])

    # Compute SMAs per symbol
    df['SMA_10'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_10, min_periods=sma_10).mean().fillna(0).round(4)
    )
    df['SMA_20'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_20, min_periods=sma_20).mean().fillna(0).round(4)
    )
    df['SMA_50'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_50, min_periods=sma_50).mean().fillna(0).round(4)
    )
    # Correct lookback periods (approximate trading days)
    periods = {
        '%Year': 249,
        '%6Month': 126,
        '%3Month': 63,
        '%1Month': 21
    }
    # For each period, compute the ratio using per-symbol shift
    for col_name, n_days in periods.items():
        df[col_name] = df.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(n_days)
        )
    df['$Volume'] = df['volume'] * ((df['high']* df['low']) / 2).round(0)

    df['Normalized'] = abs(df['SMA_20'] - df['SMA_10']) / (abs(df['SMA_20'] + df['SMA_10']) / 2)

    # Calculate daily high/low ratio (always >= 1)
    df['daily_ratio'] = (df['high'] / df['low']).round(4)
    df['daily_dollar'] = (df['high'] - df['low']).round(4)

    # Compute rolling mean of the ratio per symbol
    df['adr_percent'] = (
        df.groupby('symbol')['daily_ratio']
        .transform(lambda x: x.rolling(window=window, min_periods=window).mean()).fillna(0)
    )
    # Compute rolling mean of the ratio per symbol
    df['adr_dollar'] = (
        df.groupby('symbol')['daily_dollar']
        .transform(lambda x: x.rolling(window=window, min_periods=window).mean()).fillna(0).round(4)
    )
    # Convert ratio to percentage: (avg_ratio - 1) * 100
    df['adr_percent'] = ((df['adr_percent'] - 1) * 100).round(2)
    df['adr_distance'] = abs(df['SMA_10'] - df['close']) / df['adr_dollar']
    df['Normalized'] = (df['Normalized'] * 100).round(4)
    df['adr_distance'] = df['adr_distance'].round(4)

    df.drop(columns=['daily_ratio', 'daily_dollar' ], inplace=True)
    # Save to new CS
    df.to_csv(output_csv, index=False)
    print(f"Added SMA columns and saved to {output_csv}")
    return None



