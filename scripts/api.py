import os
import schwabdev
import pandas as pd
from dotenv import load_dotenv


def file_parser(raw_symbol_path, final_symbol_path):
    """
    Reads a CSV of stock symbols data, removes those with a % symbol, then saves the output.

    Parameters
    ----------
    raw_symbol_path : str
    final_symbol_path : str
    Returns
    -------
    None
    """
    df = pd.read_csv(raw_symbol_path, usecols=['Symbol', 'Name'])
    df_cleaned = df[~df['Name'].astype(str).str.contains('%', na=False)]
    df_cleaned.to_csv(final_symbol_path, index=False)
    return


def fetch_OHLCV(tickers,ohlcv_symb,start_date):
    """
    This fetches daily stock data from a Charles Schwab developer API. Uses the schwabdev library.

    Parameters
    ----------
    tickers : str
    ohlcv_symb : str
    start_date : datetime
    Returns
    -------
    pd.DataFrame
        Columns: open,high,low,close,volume,datetime,symbol
    """
    load_dotenv()
    original_stocks = pd.read_csv(ohlcv_symb)
    stocks = pd.read_csv(tickers)
    symbol_col = next((col for col in stocks.columns if col.lower() == 'symbol'), None)
    symbols = stocks[symbol_col].dropna().astype(str).str.upper().tolist()
    client = schwabdev.Client(os.getenv('APP_KEY'), os.getenv('APP_SECRET')) # create a client
    all_data = []
    total = len(stocks)
    for i, symbol in enumerate(symbols, 1):
        print(f"Fetching {symbol} ({i}/{total})...")
        try:
            ticker = symbol
            data = client.price_history(ticker,'year', '1',frequencyType='daily', startDate=start_date).json()
            df = pd.DataFrame(data['candles'])
            df['symbol'] = data['symbol']
            all_data.append(df)
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
    if not all_data:
        print("No data fetched.")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['datetime'], unit='ms').dt.strftime('%Y-%m-%d')
    merged_df = pd.concat([original_stocks, combined], ignore_index=True)
    merged_df.drop_duplicates(subset=['symbol', 'datetime'], keep='first')
    merged_df.to_csv(ohlcv_symb, index=False)
    print(f"\nSaved OHLCV data")
    return merged_df


# noinspection PyArgumentList
def combine_csv(one_month, three_month, six_month, one_year, rankings, output):
    df1 = pd.read_csv(one_month)
    df2 = pd.read_csv(three_month)
    df3 = pd.read_csv(six_month)
    df4 = pd.read_csv(one_year)
    df5 = pd.read_csv(rankings)

    df6 = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df6.drop_duplicates(subset=['symbol', 'datetime'], keep='first')
    final_df = df6.merge(df5, on='symbol', how='left')
    final_df.sort_values(['symbol'], ascending=False)
    final_df.to_csv(output, index=False)

