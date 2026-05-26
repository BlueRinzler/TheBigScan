import os
import schwabdev
import pandas as pd
from typing import Optional, List

# ----------------------------------------------------------------------
# Pure functions – testable without files or network
# ----------------------------------------------------------------------

def filter_percent_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the 'Name' column contains a '%' character.
    Expects columns ['Symbol', 'Name'].
    Returns a cleaned DataFrame.
    """
    return df[~df['Name'].astype(str).str.contains('%', na=False)]


def load_tickers_from_csv(csv_path: str, symbol_col: Optional[str] = None) -> List[str]:
    """
    Read a CSV and return a list of cleaned, uppercase symbol strings.
    If symbol_col is not provided, it will detect the first column named 'symbol' (case-insensitive).
    """
    df = pd.read_csv(csv_path)
    if symbol_col is None:
        symbol_col = next((col for col in df.columns if col.lower() == 'symbol'), None)
        if symbol_col is None:
            raise ValueError("No column named 'symbol' found in the tickers CSV.")
    symbols = df[symbol_col].dropna().astype(str).str.upper().tolist()
    return symbols


def combine_ohlcv(original_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate original and new OHLCV DataFrames, then drop duplicates.
    Duplicates are identified by (symbol, datetime), keeping the first occurrence.
    Returns the combined DataFrame.
    """
    merged = pd.concat([original_df, new_df], ignore_index=True)
    merged.drop_duplicates(subset=['symbol', 'datetime'], keep='first', inplace=True)
    return merged


# ----------------------------------------------------------------------
# Dependency‑injectable function – the single point of external interaction
# ----------------------------------------------------------------------

def fetch_price_history_for_symbol(
    client,
    symbol: str,
    start_date: str,
    period_type: str = 'year',
    period: str = '1',
    frequency_type: str = 'daily'
) -> pd.DataFrame:
    """
    Use a Schwab client to fetch price history for one symbol.
    Returns a DataFrame with columns: datetime, open, high, low, close, volume, symbol.
    Raises an exception on failure; the caller can handle it.
    """
    resp = client.price_history(
        symbol,
        period_type,
        period,
        frequencyType=frequency_type,
        startDate=start_date
    ).json()
    # Extract candles
    df = pd.DataFrame(resp['candles'])
    # Ensure the symbol is present (API response contains 'symbol')
    df['symbol'] = resp.get('symbol', symbol)
    return df


# ----------------------------------------------------------------------
# Client creator – can be mocked or overridden in tests
# ----------------------------------------------------------------------

def create_schwab_client() -> schwabdev.Client:
    """
    Load environment variables and return a new Schwab client.
    This is the only place that reads .env and calls schwabdev.Client.
    """
    from dotenv import load_dotenv
    load_dotenv()
    app_key = os.getenv('APP_KEY')
    app_secret = os.getenv('APP_SECRET')
    if not app_key or not app_secret:
        raise EnvironmentError("APP_KEY and APP_SECRET must be set in environment.")
    return schwabdev.Client(app_key, app_secret)


# ----------------------------------------------------------------------
# I/O wrappers – preserve original signatures but delegate to pure logic
# ----------------------------------------------------------------------

def file_parser(raw_symbol_path: str, final_symbol_path: str) -> None:
    """
    Reads a CSV of stock symbols, removes those with '%' in the Name, and saves the result.
    (Original signature kept; thin I/O wrapper)
    """
    df = pd.read_csv(raw_symbol_path, usecols=['Symbol', 'Name'])
    cleaned = filter_percent_symbols(df)
    cleaned.to_csv(final_symbol_path, index=False)


def fetch_OHLCV(
    tickers: str,
    ohlcv_symb: str,
    start_date: str
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV data for all symbols in the tickers CSV,
    merge with any existing data in ohlcv_symb, and save back.

    Parameters
    ----------
    tickers : str
        Path to CSV containing symbols.
    ohlcv_symb : str
        Path to the CSV where historical OHLCV data is stored (read/write).
    start_date : str
        Start date for price history (format expected by Schwab API).

    Returns
    -------
    pd.DataFrame or None
        The combined DataFrame, or None if no data was fetched.
    """
    # Load the existing data (can be empty)
    try:
        existing_data = pd.read_csv(ohlcv_symb)
    except FileNotFoundError:
        existing_data = pd.DataFrame()

    # Read the list of symbols
    symbols = load_tickers_from_csv(tickers)

    # Create the API client (could also be injected)
    client = create_schwab_client()

    # Fetch data for all symbols
    all_data = []
    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        print(f"Fetching {symbol} ({i}/{total})...")
        try:
            df = fetch_price_history_for_symbol(client, symbol, start_date)
            all_data.append(df)
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")

    if not all_data:
        print("No data fetched.")
        return None

    # Combine fetched data
    new_data = pd.concat(all_data, ignore_index=True)
    # Convert datetime from Unix ms to string (YYYY-MM-DD)
    new_data['datetime'] = pd.to_datetime(new_data['datetime'], unit='ms').dt.strftime('%Y-%m-%d')

    # Merge with existing data and deduplicate
    combined = combine_ohlcv(existing_data, new_data)

    # Save back to the same file
    combined.to_csv(ohlcv_symb, index=False)
    print(f"\nSaved OHLCV data")
    return combined