import pandas as pd


def filter_consolidation(
    input_file: str,
    output_file: str,
    time: int,
    min_pass_days: int = 8,
    norm_thresh: float = 5.0,      # Max Normalized (%) - SMA spread
    adr_dist_thresh: float = 0.7   # Max adr_distance (units)
) -> None:
    """
    Reads a CSV of market data, applies a growth filter,
    then a 10‑day SMA proximity filter that requires at least
    `min_pass_days` out of the last 10 days to have closes
    within 5% of their SMAs. Finally, keeps the latest row
    per symbol and applies remaining numeric filters including
    Normalized and adr_distance thresholds.

    Parameters
    ----------
    input_file : str
        Path to input CSV.
    output_file : str
        Path to output CSV.
    time : int
        Lookback days for the growth filter (latest close >= 1.2 * close that many days ago).
    min_pass_days : int, optional
        Minimum number of days (out of the last 10) that must pass
        the daily SMA proximity check. Default = 8.
    norm_thresh : float, optional
        Maximum allowed Normalized value (SMA20‑SMA10 spread %). Default = 5.0.
    adr_dist_thresh : float, optional
        Maximum allowed adr_distance (|price - SMA10| / adr_dollar). Default = 0.7.
    """

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime'])

    # ---- 1. Growth filter (unchanged) ----
    def passes_growth_check(group: pd.DataFrame) -> bool:
        if len(group) < time:
            return False
        latest_close = group['close'].iloc[-1]
        close_ago = group['close'].iloc[-time]
        return latest_close >= 1.2 * close_ago

    df = df.groupby('symbol', group_keys=False).filter(passes_growth_check)

    # ---- 2. 10‑day SMA proximity filter with minimum pass days ----
    def passes_sma_threshold(group: pd.DataFrame, window: int = 10, min_pass: int = 8) -> bool:
        """
        Return True if at least `min_pass` of the last `window` days
        have close within ±5% of all three SMAs on that day.
        """
        if len(group) < window:
            return False

        recent = group.tail(window)
        passing_days = 0

        for _, row in recent.iterrows():
            close = row['close']
            # Check all three SMAs for this row
            sma10_ok = 0.95 * close <= row['SMA_10'] <= 1.05 * close
            sma20_ok = 0.95 * close <= row['SMA_20'] <= 1.05 * close
            if sma10_ok and sma20_ok:
                passing_days += 1

        return passing_days >= min_pass

    symbols_ok = df.groupby('symbol').filter(
        lambda grp: passes_sma_threshold(grp, window=10, min_pass=min_pass_days)
    )['symbol'].unique()
    df = df[df['symbol'].isin(symbols_ok)]

    # ---- 3. Latest row per remaining symbol ----
    latest_idx = df.groupby('symbol')['datetime'].idxmax()
    latest_df = df.loc[latest_idx].copy()

    # ---- 4. Numeric filters (SMA part already handled) ----
    mask = (
        (latest_df['adr_percent'] > 4) &
        (latest_df['$Volume'] > 1_000_000) &
        (latest_df['close'] > 3) &
        (latest_df['volume'] > 200_000)
    )
    filtered = latest_df[mask]

    # ---- 5. Save ----
    filtered.to_csv(output_file, index=False)
    print(f"Final output: {len(filtered)} rows. Saved to {output_file}")



def filter_momentum(input_file: str, output_file: str, time: int) -> None:
    """
    Reads a CSV of market data, applies a 22-day growth filter first,
    then the original four filters, keeping only the latest row per symbol.
    """
    # Load the data
    df = pd.read_csv(input_file)

    # Ensure datetime is parsed correctly and sort by symbol and date
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime'])

    # ---- NEW: Filter symbols where latest close >= 1.2 * close 22 trading days ago ----
    def passes_growth_check(group: pd.DataFrame) -> bool:
        """Return True if symbol has >=23 rows and the 22‑day growth condition is met."""
        if len(group) < time:          # need at least latest + 22 prior trading days
            return False
        latest_close = group['close'].iloc[-1]
        close_ago = group['close'].iloc[-time]   # 22 trading days before the latest
        return latest_close >= 1.2 * close_ago

    # Create a mask of symbols that satisfy the growth condition
    symbols_ok = df.groupby('symbol', group_keys=False).filter(passes_growth_check)['symbol'].unique()
    # Keep only those symbols in the full dataset
    df = df[df['symbol'].isin(symbols_ok)]

    # ---- Step 1: Get the latest date for each (remaining) symbol ----
    latest_idx = df.groupby('symbol')['datetime'].idxmax()
    latest_df = df.loc[latest_idx].copy()

    # ---- Step 2: Apply all five post‑growth filters ----
    mask = (
        (latest_df['adr_percent'] > 4) &
        (latest_df['$Volume'] > 1_000_000) &
        (latest_df['close'] > 3) &
        (latest_df['volume'] > 200_000) &
        # New SMA proximity filter:
        (latest_df['SMA_10'] < latest_df['close']) &
        (latest_df['SMA_20'] < latest_df['close']) &
        (latest_df['SMA_50'] < latest_df['close'])
    )
    filtered = latest_df[mask]

    # ---- Step 3: Write the result ----
    filtered.to_csv(output_file, index=False)

    print(f"Growth filter kept {len(symbols_ok)} symbols. "
          f"Final output: {len(filtered)} rows. Saved to {output_file}")


