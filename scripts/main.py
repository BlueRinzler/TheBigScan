import os
import pandas as pd

from dateutil.utils import today

from scripts.api import fetch_OHLCV
from scripts.compute import compute_data
from dotenv import load_dotenv

from scripts.rankings import generate_sector_rankings

if __name__ == '__main__':

    load_dotenv()
    start_date = today() - pd.Timedelta(days=2)
    #fetch_OHLCV(os.getenv('SYMBOLS'), os.getenv('OHLCV_DATA_RAW'),start_date)

    #compute_data(os.getenv('OHLCV_DATA_RAW'), os.getenv('COMPUTED_DATA_RAW'))
    #generate_sector_rankings(os.getenv('COMPUTED_DATA_RAW'),os.getenv('SYMBOLS'), os.getenv('SECTOR_RANKINGS'))


