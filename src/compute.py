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
    Reads a CSV of OHLCV data, calculate 10 SMA, 20 SMA, 50 SMA, Bollinger Bands, and Keltner channels.
    Calculates % year gain, 6 month, 3 month and 1 month gain on lastest close.

    Parameters
    ----------
    ohlcv_raw : str
        Path to input CSV.
    output_csv : str
        Path to output CSV.
    sma_10 : int
        lookback period for SMA calculation.
    sma_20 : int
        lookback period for SMA calculation.
    sma_50 : int
        lookback period for SMA calculation.
    window : int
        lookback period for Bollinger and Keltner calculations.
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
        lambda x: x.rolling(window=sma_10, min_periods=sma_10).mean().round(4)
    )
    df['SMA_20'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_20, min_periods=sma_20).mean().round(4)
    )
    df['SMA_50'] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=sma_50, min_periods=sma_50).mean().round(4)
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
        df[col_name] = (
            df.groupby("symbol")["close"]
            .transform(lambda x: (x / x.shift(n_days) - 1) * 100)
            .round(2)
        )
    # Dollar volume using average of high and low
    df['$Volume'] = df['volume'] * ((df['high'] + df['low']) / 2).round(0)

    # Calculate daily ratio and its average
    df['daily_ratio'] = (df['high'] / df['low']).round(4)
    # ADR% from daily_ratio – keep NaN instead of fillna(0)
    df["adr_percent"] = df.groupby("symbol")["daily_ratio"].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    # Daily range and its average 
    df["daily_range"] = (df["high"] - df["low"]).round(4)
    df["atr"] = df.groupby("symbol")["daily_range"].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )  # No fillna(0) – keep NaN

    # Compute rolling std of the ratio per symbol
    df["str_dev"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(window=window, min_periods=window).std())
    )
    # Compute  lower band of bollinger bands
    df['lower_band'] = df['SMA_20'] - (2 *df['str_dev'])
    # Computer upper band of bollinger bands
    df["upper_band"] = df["SMA_20"] + (2 * df["str_dev"])
    # Compute  lower band of Keltner channel
    df["lower_channel"] = df["SMA_20"] - (1.5 * df['atr'])
    # Computer upper band of Keltner channel
    df["upper_channel"] = df["SMA_20"] + (1.5 * df['atr'])

    # Convert ratio to percentage: (avg_ratio - 1) * 100
    df['adr_percent'] = ((df['adr_percent'] - 1) * 100).round(2)

    df.drop(columns=['daily_ratio', 'daily_range' ], inplace=True)
    # Save to new CS
    df.to_csv(output_csv, index=False)
    print(f"Added SMA columns and saved to {output_csv}")
    return None



