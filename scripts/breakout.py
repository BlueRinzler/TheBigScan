import pandas as pd
import os
from dotenv import load_dotenv

def analyze_breakouts(csv_file_path):
    """
    Load OHLCV data from a CSV, detect confirmed breakouts
    (close > 50‑period high, volume > 1.5x 20‑period average,
    and the next 5 closes are each ≥ 20% above the breakout close),
    print the results, and return the list of (symbol, date) pairs.
    """
    # Load environment variables (e.g., API keys) if needed
    load_dotenv()

    # Read and prepare the data
    df = pd.read_csv(csv_file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime'])

    # Inner function that processes one symbol's data
    def get_confirmed_breakouts(group):
        group = group.reset_index(drop=True)
        if len(group) < 50:
            return []

        group['high_50'] = group['close'].rolling(window=50, min_periods=50).max()
        group['avg_vol_20'] = group['volume'].rolling(window=20, min_periods=20).mean()

        potential = (group['close'] > group['high_50'].shift(1)) & \
                    (group['volume'] > 1.5 * group['avg_vol_20'].shift(1))

        breakouts = []
        symbol = group['symbol'].iloc[0]

        for idx in group.index[potential]:
            breakout_close = group.loc[idx, 'close']
            if idx + 5 >= len(group):
                continue
            next_5_closes = group.loc[idx + 1:idx + 5, 'close'].values
            if all(c >= 1.2 * breakout_close for c in next_5_closes):
                breakouts.append((symbol, group.loc[idx, 'datetime'].date()))
        return breakouts

    # Collect breakouts for every symbol
    all_breakouts = []
    for symbol, data in df.groupby('symbol'):
        all_breakouts.extend(get_confirmed_breakouts(data))

    # Print results (original behaviour)
    print("Confirmed breakouts (20% higher for next 5 closes):")
    for sym, date in all_breakouts:
        print(f"{sym} broke out on {date}")

    return all_breakouts

if __name__ == "__main__":
    load_dotenv()
    # Example usage – adjust the path as needed
    analyze_breakouts(os.getenv('OHLCV_DATA_RAW'))