import pandas as pd

from scanner import growth_filter, passes_bands, apply_latest_numeric_filters

def test_growth_filter_basic():
    df = pd.DataFrame({
        'close': [10, 11, 12, 13, 14]  # 14 vs 10 -> 40% growth
    })
    assert growth_filter(df, lookback=4, factor=1.2)

def test_growth_filter_short_data():
    df = pd.DataFrame({'close': [10, 12]})
    assert not growth_filter(df, lookback=5, factor=1.2)

def test_passes_bands_all_ok():
    group = pd.DataFrame({
        'upper_band':  [105, 106, 107],
        'lower_band':  [95,  96,  97],
        'upper_channel': [110, 111, 112],
        'lower_channel': [90,  91,  92],
    })
    assert passes_bands(group, window=3, min_pass=3)

def test_passes_bands_not_enough_data():
    group = pd.DataFrame({
        'upper_band':  [105],
        'lower_band':  [95],
        'upper_channel': [110],
        'lower_channel': [90],
    })
    assert not passes_bands(group, window=3, min_pass=2)  # only 1 row

def test_apply_latest_numeric_filters():
    df = pd.DataFrame({
        'symbol': ['A', 'B'],
        'adr_percent': [5, 2],
        '$Volume': [2e6, 500e3],
        'close': [10, 5],
        'volume': [300e3, 100e3],
        'SMA_10': [9, 4],
        'SMA_20': [8, 6],
        'SMA_50': [7, 3],
    })
    res = apply_latest_numeric_filters(df, sma_check=True)
    assert len(res) == 1
    assert res.iloc[0]['symbol'] == 'A'

