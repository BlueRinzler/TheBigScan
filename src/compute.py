import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------------------
# 1. Pure I/O helper – can be mocked or used with temp files in tests
# ----------------------------------------------------------------------
def load_ohlcv_data(csv_path: str) -> pd.DataFrame:
    """Read OHLCV CSV with datetime parsing. No existence check, no prints."""
    return pd.read_csv(csv_path, parse_dates=['datetime'])

# Original signature kept for backward compatibility.
# In tests you can directly use `load_ohlcv_data` or mock this function.
def load_ohlcv_from_cache(csv_path: str, verbose: bool = True) -> pd.DataFrame | None:
    """Wrapper with existence check and logging – useful in production scripts."""
    if not os.path.exists(csv_path):
        if verbose:
            print(f"Cache file {csv_path} not found. Run fetch_and_save_ohlcv first.")
        return None
    df = load_ohlcv_data(csv_path)
    if verbose:
        print(f"Loaded cached data: {df['symbol'].nunique()} symbols, {len(df)} rows")
    return df

# ----------------------------------------------------------------------
# 2. Pure computation – all indicators, no I/O
# ----------------------------------------------------------------------
def compute_indicators(
    df: pd.DataFrame,
    sma_10: int = 10,
    sma_20: int = 20,
    sma_50: int = 50,
    window: int = 20
) -> pd.DataFrame:
    """
    Add SMA, Bollinger Bands, Keltner Channels, percentage gains, ADR%, ATR to a DataFrame.

    Expects columns: ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume'].
    Returns a new DataFrame with added columns.
    """
    df = df.sort_values(['symbol', 'datetime']).copy()

    # ---------- Simple Moving Averages ----------
    df['SMA_10'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_10, min_periods=sma_10).mean().round(4)
    )
    df['SMA_20'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_20, min_periods=sma_20).mean().round(4)
    )
    df['SMA_50'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_50, min_periods=sma_50).mean().round(4)
    )

    # ---------- Percentage gains over different periods ----------
    periods = {
        '%Year': 249,
        '%6Month': 126,
        '%3Month': 63,
        '%1Month': 21
    }
    for col_name, n_days in periods.items():
        df[col_name] = (
            df.groupby("symbol")["close"]
            .transform(lambda x: (x / x.shift(n_days) - 1) * 100)
            .round(2)
        )

    # ---------- Dollar volume ----------
    df['$Volume'] = df['volume'] * ((df['high'] + df['low']) / 2).round(0)

    # ---------- ADR% and ATR ----------
    df['daily_ratio'] = (df['high'] / df['low']).round(4)
    df['daily_range'] = (df['high'] - df['low']).round(4)

    df['adr_percent'] = df.groupby('symbol')['daily_ratio'].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    df['atr'] = df.groupby('symbol')['daily_range'].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )

    # ---------- Bollinger Bands & Keltner Channels ----------
    df['str_dev'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=window, min_periods=window).std()
    )

    df['lower_band'] = df['SMA_20'] - (2 * df['str_dev'])
    df['upper_band'] = df['SMA_20'] + (2 * df['str_dev'])
    df['lower_channel'] = df['SMA_20'] - (1.5 * df['atr'])
    df['upper_channel'] = df['SMA_20'] + (1.5 * df['atr'])

    # Convert ADR ratio to percentage
    df['adr_percent'] = ((df['adr_percent'] - 1) * 100).round(2)

    # Drop intermediate columns
    df.drop(columns=['daily_ratio', 'daily_range'], inplace=True)

    return df

# ----------------------------------------------------------------------
# 3. Production wrapper – the original function, now a thin orchestrator
# ----------------------------------------------------------------------
def compute_data(
    ohlcv_raw: str,
    output_csv: str,
    sma_10: int = 10,
    sma_20: int = 20,
    sma_50: int = 50,
    window: int = 20
) -> None:
    """
    Reads OHLCV CSV, computes all indicators, writes result to CSV.
    """
    df = load_ohlcv_from_cache(ohlcv_raw)
    if df is None or df.empty:
        print("Cannot compute SMA: missing input data.")
        return None
    result = compute_indicators(df, sma_10, sma_20, sma_50, window)
    result.to_csv(output_csv, index=False)
    print(f"Added SMA columns and saved to {output_csv}")
    return None