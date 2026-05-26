import pandas as pd
from api import (
    filter_percent_symbols,
    combine_ohlcv,
)

def test_filter_percent_symbols():
    df = pd.DataFrame({
        'Symbol': ['A', 'B', 'C'],
        'Name': ['Apple', '50% Corp', 'Google']
    })
    result = filter_percent_symbols(df)
    assert len(result) == 2
    assert '50% Corp' not in result['Name'].values

def test_combine_ohlcv_drops_duplicates():
    old = pd.DataFrame({
        'symbol': ['A'],
        'datetime': ['2023-01-01'],
        'close': [100]
    })
    new = pd.DataFrame({
        'symbol': ['A', 'B'],
        'datetime': ['2023-01-01', '2023-01-02'],
        'close': [100, 200]
    })
    combined = combine_ohlcv(old, new)
    # Should keep first occurrence → old's row for A/2023-01-01
    assert len(combined) == 2
    assert combined[combined['symbol'] == 'A']['close'].values[0] == 100