import os
import pandas as pd

from dateutil.utils import today
import api
import rankings
import compute
import scanner
from dotenv import load_dotenv

def run_API():
    load_dotenv()
    start_date = today() - pd.Timedelta(days=2)
    api.fetch_OHLCV(os.getenv('SYMBOLS'), os.getenv('OHLCV_DATA_RAW'), start_date)

# noinspection PyArgumentList
def combine_csv(one_month, three_month, six_month, one_year, sector_rankings, output):
    df1 = pd.read_csv(one_month)
    df2 = pd.read_csv(three_month)
    df3 = pd.read_csv(six_month)
    df4 = pd.read_csv(one_year)
    df5 = pd.read_csv(sector_rankings)

    df6 = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df6 = df6.drop_duplicates(subset=['symbol', 'datetime'], keep='first')
    final_df = df6.merge(df5, on='symbol', how='left')
    final_df = final_df.sort_values(['symbol'], ascending=False)
    final_df.to_csv(output, index=False)


def gen_data():
    load_dotenv()
    rankings.generate_sector_rankings(os.getenv('COMPUTED_DATA_RAW'), os.getenv('SYMBOLS'), os.getenv('SECTOR_RANKINGS'))
    compute.compute_data(os.getenv('OHLCV_DATA_RAW'), os.getenv('COMPUTED_DATA_RAW'))

    scanner.filter_consolidation(os.getenv('COMPUTED_DATA_RAW'), os.getenv('ONE_MONTH_GAINERS'), time=21)
    scanner.filter_consolidation(os.getenv('COMPUTED_DATA_RAW'), os.getenv('THREE_MONTH_GAINERS'),  time=63)
    scanner.filter_consolidation(os.getenv('COMPUTED_DATA_RAW'), os.getenv('SIX_MONTH_GAINERS'), time=126)
    scanner.filter_consolidation(os.getenv('COMPUTED_DATA_RAW'), os.getenv('ONE_YEAR_GAINERS'),  time=249)

    scanner.filter_momentum(os.getenv('COMPUTED_DATA_RAW'), os.getenv('ONE_MONTH_MOMENTUM'), time=21)
    scanner.filter_momentum(os.getenv('COMPUTED_DATA_RAW'), os.getenv('THREE_MONTH_MOMENTUM'), time=63)
    scanner.filter_momentum(os.getenv('COMPUTED_DATA_RAW'), os.getenv('SIX_MONTH_MOMENTUM'),time=126)
    scanner.filter_momentum(os.getenv('COMPUTED_DATA_RAW'), os.getenv('ONE_YEAR_MOMENTUM'), time=249)

    combine_csv(os.getenv('ONE_MONTH_GAINERS'),
                os.getenv('THREE_MONTH_GAINERS'),
                os.getenv('SIX_MONTH_GAINERS'),
                os.getenv('ONE_YEAR_GAINERS'),
                os.getenv('SECTOR_RANKINGS'),
                os.getenv('CONSOLIDATION_SYMBOLS'))

    combine_csv(os.getenv('ONE_MONTH_MOMENTUM'),
                os.getenv('THREE_MONTH_MOMENTUM'),
                os.getenv('SIX_MONTH_MOMENTUM'),
                os.getenv('ONE_YEAR_MOMENTUM'),
                os.getenv('SECTOR_RANKINGS'),
                os.getenv('MOMENTUM_SYMBOLS'))

if __name__ == '__main__':

    response = input("Do you want to run the API? (y/n): ").strip().lower()
    if response in ("y", "yes"):
        print("Running the API...")
        run_API()
    gen_data()
    print("Done!")


