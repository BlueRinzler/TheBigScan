import pandas as pd
import numpy as np
import pytest
from compute import compute_indicators  # adjust import

# ----------------------------------------------------------------------
# Helper: Reusable DataFrame for single-symbol tests
# ----------------------------------------------------------------------
@pytest.fixture
def single_symbol_df():
    """6 days of clean data for one symbol, as in your existing test."""
    data = {
        'symbol': ['AAPL'] * 6,
        'datetime': pd.date_range('2026-01-01', periods=6),
        'open':   [10, 11, 12, 20, 21, 22],
        'high':   [11, 12, 13, 21, 22, 23],
        'low':    [9,  10, 11, 19, 20, 21],
        'close':  [10, 11, 12, 20, 21, 22],
        'volume': [100, 100, 100, 200, 200, 200],
    }
    return pd.DataFrame(data)

# ----------------------------------------------------------------------
# 1. Test all Simple Moving Averages thoroughly
# ----------------------------------------------------------------------
def test_sma_full_sequence(single_symbol_df):
    """Verify SMA_10, SMA_20, SMA_50 at every row with explicit values."""
    # Use same parameters as your original test
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)

    expected_sma10 = [np.nan, 10.5, 11.5, 16.0, 20.5, 21.5]
    expected_sma20 = [np.nan, np.nan, np.nan, 13.25, 16.0, 18.75]
    expected_sma50 = [np.nan]*5 + [16.0]

    for i, (s10, s20, s50) in enumerate(zip(expected_sma10, expected_sma20, expected_sma50)):
        if np.isnan(s10):
            assert np.isnan(result.loc[i, 'SMA_10'])
        else:
            assert result.loc[i, 'SMA_10'] == pytest.approx(s10, 0.0001)
        # SMA_20
        if np.isnan(s20):
            assert np.isnan(result.loc[i, 'SMA_20'])
        else:
            assert result.loc[i, 'SMA_20'] == pytest.approx(s20, 0.0001)
        # SMA_50
        if np.isnan(s50):
            assert np.isnan(result.loc[i, 'SMA_50'])
        else:
            assert result.loc[i, 'SMA_50'] == pytest.approx(s50, 0.0001)

# ----------------------------------------------------------------------
# 2. Dollar Volume calculation
# ----------------------------------------------------------------------
def test_dollar_volume(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    # high+low /2 rounded(0): [10, 11, 12, 20, 21, 22]  (since high+low even)
    # volume: [100,100,100,200,200,200]
    expected = [1000, 1100, 1200, 4000, 4200, 4400]
    for i, exp in enumerate(expected):
        assert result.loc[i, '$Volume'] == pytest.approx(exp, 1e-9)

# ----------------------------------------------------------------------
# 3. ATR calculation (rolling mean of daily_range)
# ----------------------------------------------------------------------
def test_atr_calculation(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    # daily_range = high-low: [2,2,2,2,2,2]
    # rolling mean with min_periods=2: index0 NaN, then all 2.0
    assert np.isnan(result.loc[0, 'atr'])
    for i in range(1, 6):
        assert result.loc[i, 'atr'] == 2.0

# ----------------------------------------------------------------------
# 4. ADR% conversion from daily_ratio
# ----------------------------------------------------------------------
def test_adr_percent_values(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    # daily_ratio = high/low rounded(4): [1.2222, 1.2, 1.1818, 1.1053, 1.1, 1.0952]
    # rolling mean window=2:
    # idx1: (1.2222+1.2)/2 = 1.2111 -> adr% = (1.2111-1)*100=21.11
    # idx2: (1.2+1.1818)/2=1.1909 -> 19.09
    # idx3: (1.1818+1.1053)/2=1.14355 -> 14.355 -> 14.36 (rounded to 2)
    # idx4: (1.1053+1.1)/2=1.10265 -> 10.265 -> 10.27
    # idx5: (1.1+1.0952)/2=1.0976 -> 9.76
    expected = [np.nan, 21.11, 19.09, 14.36, 10.27, 9.76]
    for i, exp in enumerate(expected):
        if np.isnan(exp):
            assert np.isnan(result.loc[i, 'adr_percent'])
        else:
            assert result.loc[i, 'adr_percent'] == pytest.approx(exp, 0.01)

# ----------------------------------------------------------------------
# 5. Bollinger Bands (lower_band, upper_band)
# ----------------------------------------------------------------------
def test_bollinger_bands(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    # SMA_20: [NaN, NaN, NaN, 13.25, 16.0, 18.75]
    # str_dev (std of close, window=2): [NaN, 0.7071, 0.7071, 5.6569, 0.7071, 0.7071]
    # lower = SMA_20 - 2*str_dev
    # upper = SMA_20 + 2*str_dev
    # idx3: lower = 13.25 - 11.3137 = 1.9363, upper = 13.25 + 11.3137 = 24.5637
    # idx4: lower = 16.0 - 1.4142 = 14.5858, upper = 17.4142
    # idx5: lower = 18.75 - 1.4142 = 17.3358, upper = 20.1642
    expected_lower = [np.nan, np.nan, np.nan, 1.9363, 14.5858, 17.3358]
    expected_upper = [np.nan, np.nan, np.nan, 24.5637, 17.4142, 20.1642]
    for i in range(6):
        if np.isnan(expected_lower[i]):
            assert np.isnan(result.loc[i, 'lower_band'])
        else:
            assert result.loc[i, 'lower_band'] == pytest.approx(expected_lower[i], 1e-3)
            assert result.loc[i, 'upper_band'] == pytest.approx(expected_upper[i], 1e-3)

# ----------------------------------------------------------------------
# 6. Keltner Channels
# ----------------------------------------------------------------------
def test_keltner_channels(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    # SMA_20 as before, ATR = 2.0 from index1 onward
    # lower_channel = SMA_20 - 1.5*ATR
    # upper_channel = SMA_20 + 1.5*ATR
    # idx3: lower = 13.25 - 3 = 10.25, upper = 16.25
    # idx4: lower = 16.0 - 3 = 13.0, upper = 19.0
    # idx5: lower = 18.75 - 3 = 15.75, upper = 21.75
    expected_lower = [np.nan, np.nan, np.nan, 10.25, 13.0, 15.75]
    expected_upper = [np.nan, np.nan, np.nan, 16.25, 19.0, 21.75]
    for i in range(6):
        if np.isnan(expected_lower[i]):
            assert np.isnan(result.loc[i, 'lower_channel'])
        else:
            assert result.loc[i, 'lower_channel'] == pytest.approx(expected_lower[i], 1e-9)
            assert result.loc[i, 'upper_channel'] == pytest.approx(expected_upper[i], 1e-9)

# ----------------------------------------------------------------------
# 7. Percentage gains – insufficient data yields NaN
# ----------------------------------------------------------------------
def test_percentage_gains_nan_for_short_history(single_symbol_df):
    result = compute_indicators(single_symbol_df, sma_10=2, sma_20=4, sma_50=6, window=2)
    for col in ['%Year', '%6Month', '%3Month', '%1Month']:
        assert result[col].isna().all(), f"{col} should be all NaN with only 6 days of data"

# ----------------------------------------------------------------------
# 8. Multi‑symbol isolation
# ----------------------------------------------------------------------
def test_multi_symbol_isolation():
    """Ensure calculations are performed per symbol and don't leak."""
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
        'datetime': pd.date_range('2026-01-01', periods=6),
        'open':   [10, 11, 12, 20, 21, 22],
        'high':   [11, 12, 13, 21, 22, 23],
        'low':    [9, 10, 11, 19, 20, 21],
        'close':  [10, 11, 12, 20, 21, 22],
        'volume': [100, 100, 100, 200, 200, 200],
    }
    df = pd.DataFrame(data)
    result = compute_indicators(df, sma_10=2, sma_20=3, sma_50=3, window=2)

    # AAPL rows (0‑2) should not be influenced by MSFT close values
    aapl = result[result['symbol'] == 'AAPL'].reset_index(drop=True)
    msft = result[result['symbol'] == 'MSFT'].reset_index(drop=True)

    # SMA_10 with window=2, min_periods=2
    # AAPL closes: 10,11,12 -> SMA_10: NaN, 10.5, 11.5
    # MSFT closes: 20,21,22 -> SMA_10: NaN, 20.5, 21.5
    assert np.isnan(aapl.loc[0, 'SMA_10'])
    assert aapl.loc[1, 'SMA_10'] == 10.5
    assert aapl.loc[2, 'SMA_10'] == 11.5

    assert np.isnan(msft.loc[0, 'SMA_10'])
    assert msft.loc[1, 'SMA_10'] == 20.5
    assert msft.loc[2, 'SMA_10'] == 21.5

    # Industry (if any) – not added by compute_indicators, just checking
    # ATR: both have daily_range 2, so ATR should be 2 for both after index0
    assert aapl.loc[1, 'atr'] == 2.0
    assert msft.loc[1, 'atr'] == 2.0

# ----------------------------------------------------------------------
# 9. Edge cases
# ----------------------------------------------------------------------
def test_empty_dataframe():
    df = pd.DataFrame(columns=['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume'])
    result = compute_indicators(df)
    assert result.empty
    # Expected columns should exist
    expected_cols = {'SMA_10', 'SMA_20', 'SMA_50', 'lower_band', 'upper_band',
                     'lower_channel', 'upper_channel', 'adr_percent', 'atr', '$Volume',
                     '%Year', '%6Month', '%3Month', '%1Month'}
    assert expected_cols.issubset(result.columns)

