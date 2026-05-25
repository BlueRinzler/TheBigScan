import pandas as pd
from typing import Optional

def load_data(input_file: str) -> pd.DataFrame:
    """Read CSV, parse dates, sort by symbol and datetime."""
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values(['symbol', 'datetime'])

def growth_filter(group: pd.DataFrame, lookback: int, factor: float = 1.2) -> bool:
    """Return True if symbol has enough data and latest close >= factor * close 'lookback' days ago."""
    if len(group) < lookback:
        return False
    latest_close = group['close'].iloc[-1]
    close_ago = group['close'].iloc[-lookback]
    return latest_close >= factor * close_ago

def apply_growth_filter(df: pd.DataFrame, lookback: int, factor: float = 1.2) -> pd.DataFrame:
    """Keep only symbols that pass the growth condition."""
    return df.groupby('symbol', group_keys=False).filter(
        lambda g: growth_filter(g, lookback, factor)
    )

def passes_bands(group: pd.DataFrame, window: int = 10, min_pass: int = 8) -> bool:
    """
    Check if, over the most recent `window` days, the Bollinger Bands are
    consistently inside (or touching) the Keltner Channels.
    """
    if len(group) < window:
        return False
    recent = group[['upper_band', 'lower_band', 'upper_channel', 'lower_channel']].tail(window).dropna()
    if len(recent) < min_pass:
        return False
    passing = (
        (recent['upper_band'] <= recent['upper_channel']) &
        (recent['lower_band'] >= recent['lower_channel'])
    ).sum()
    return passing >= min_pass

def apply_bands_filter(df: pd.DataFrame, window: int = 10, min_pass: int = 8) -> pd.DataFrame:
    """Keep only symbols that pass the Bollinger/Keltner containment check."""
    # Get boolean mask per symbol (True = keep)
    good_symbols = df.groupby('symbol').filter(
        lambda g: passes_bands(g, window, min_pass)
    )['symbol'].unique()
    return df[df['symbol'].isin(good_symbols)]

def get_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing only the most recent row for each symbol."""
    latest_idx = df.groupby('symbol')['datetime'].idxmax()
    return df.loc[latest_idx].copy()

def apply_latest_numeric_filters(
    latest_df: pd.DataFrame,
    adr_min: float = 4.0,
    dollar_volume_min: float = 1_000_000,
    close_min: float = 3.0,
    volume_min: float = 200_000,
    sma_check: bool = False
) -> pd.DataFrame:
    """
    Filter the latest‑row DataFrame using basic liquidity/price criteria.
    If `sma_check` is True, also require close > SMA_10, SMA_20, SMA_50.
    """
    mask = (
        (latest_df['adr_percent'] > adr_min) &
        (latest_df['$Volume'] > dollar_volume_min) &
        (latest_df['close'] > close_min) &
        (latest_df['volume'] > volume_min)
    )
    if sma_check:
        mask = mask & (
            (latest_df['SMA_10'] < latest_df['close']) &
            (latest_df['SMA_20'] < latest_df['close']) &
            (latest_df['SMA_50'] < latest_df['close'])
        )
    return latest_df[mask]

def save_data(df: pd.DataFrame, output_file: str, label: str = "output") -> None:
    """Save DataFrame to CSV and print summary."""
    df.to_csv(output_file, index=False)
    print(f"{label}: {len(df)} rows. Saved to {output_file}")

def filter_consolidation(
    input_file: str,
    output_file: str,
    time: int,
    min_pass_days: int = 8,
    growth_factor: float = 1.2,
    window: int = 10,
) -> None:
    """
    Screen for stocks in a consolidation after a growth surge.

    Parameters
    ----------
    time : int
        Lookback days for growth check (latest close >= growth_factor * close `time` days ago).
    min_pass_days : int, optional
        Min days (out of `window`) with Bollinger inside Keltner. Default 8.
    growth_factor : float, optional
        Growth multiplier, default 1.2 (+20%).
    window : int, optional
        Lookback window for the bands squeeze check. Default 10.
    """
    df = load_data(input_file)

    # 1. Growth filter
    df = apply_growth_filter(df, lookback=time, factor=growth_factor)

    # 2. Bands‑in‑channel (consolidation) filter
    df = apply_bands_filter(df, window=window, min_pass=min_pass_days)

    # 3. Latest row per symbol
    latest = get_latest_rows(df)

    # 4. Numeric filters (no SMA check here)
    final = apply_latest_numeric_filters(latest, sma_check=False)

    save_data(final, output_file, label="Consolidation final output")


def filter_momentum(
    input_file: str,
    output_file: str,
    time: int,
    growth_factor: float = 1.2,
) -> None:
    """
    Screen for stocks with strong upward momentum.

    Parameters
    ----------
    time : int
        Lookback days for growth check.
    growth_factor : float, optional
        Growth multiplier, default 1.2.
    """
    df = load_data(input_file)

    # 1. Growth filter
    df = apply_growth_filter(df, lookback=time, factor=growth_factor)

    # 2. Latest row per symbol
    latest = get_latest_rows(df)

    # 3. Numeric filters + SMA proximity (close above all three SMAs)
    final = apply_latest_numeric_filters(latest, sma_check=True)

    save_data(final, output_file, label="Momentum final output")