import pandas as pd
import numpy as np
import pytest
from rankings import (
    average_latest_performance,
    attach_industry,
    add_intra_industry_ranks,
    add_industry_stock_count,
    compute_sector_rankings,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Two symbols, two days each, with performance data."""
    data = {
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'datetime': pd.date_range('2026-01-01', periods=4),
        '%Year':    [10.0, 20.0, 5.0, 15.0],
        '%6Month':  [ 8.0, 18.0, 3.0, 12.0],
        '%3Month':  [ 6.0, 12.0, 2.0, 9.0],
        '%1Month':  [ 4.0, 10.0, 1.0, 7.0],
        # extra columns that should be ignored
        'close': [150, 155, 250, 260],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_sector():
    """Mapping of symbol → industry."""
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Industry': ['Technology', 'Technology', 'Communication']
    })

# =============================================================================
# 1. average_latest_performance
# =============================================================================

def test_avg_latest_perf_single_symbol_single_day():
    """min_periods=1 → even 1 row returns that value."""
    df = pd.DataFrame({
        'symbol': ['A'],
        'datetime': pd.Timestamp('2026-01-01'),
        '%Year': [5.0],
        '%6Month': [3.0],
        '%3Month': [2.0],
        '%1Month': [1.0],
    })
    result = average_latest_performance(df, rolling_window=5)
    assert len(result) == 1
    assert result.loc[0, '%Year'] == 5.0
    assert result.loc[0, '%1Month'] == 1.0

def test_avg_latest_perf_window_1_gives_last_value(sample_ohlcv):
    """rolling_window=1 with two dates per symbol should pick the last row."""
    result = average_latest_performance(sample_ohlcv, rolling_window=1)
    # Should have 1 row per symbol = the most recent date
    # AAPL last row (day2): %Year=20, %6Month=18, %3Month=12, %1Month=10
    aapl = result[result['symbol'] == 'AAPL']
    assert len(aapl) == 1
    assert aapl['%Year'].values[0] == 20.0
    assert aapl['%1Month'].values[0] == 10.0

    msft = result[result['symbol'] == 'MSFT']
    assert msft['%Year'].values[0] == 15.0

def test_avg_latest_perf_rolling_average(sample_ohlcv):
    """rolling_window=2 averages both dates, then takes the average of the last window."""
    # For rolling_window=2 with only 2 rows per symbol, the last window is the average of both rows.
    # AAPL: %Year avg (10+20)/2=15, %6Month (8+18)/2=13, etc.
    result = average_latest_performance(sample_ohlcv, rolling_window=2)
    aapl = result[result['symbol'] == 'AAPL'].iloc[0]
    assert aapl['%Year'] == pytest.approx(15.0)
    assert aapl['%6Month'] == pytest.approx(13.0)
    assert aapl['%3Month'] == pytest.approx(9.0)
    assert aapl['%1Month'] == pytest.approx(7.0)

def test_avg_latest_perf_non_default_columns():
    """Custom perf_cols argument works."""
    df = pd.DataFrame({
        'symbol': ['X', 'X'],
        'datetime': pd.date_range('2026-01-01', periods=2),
        'custom_a': [1, 3],
        'custom_b': [10, 30],
    })
    result = average_latest_performance(df, rolling_window=2, perf_cols=['custom_a', 'custom_b'])
    # last rolling average (both rows) => (1+3)/2=2, (10+30)/2=20
    assert result.loc[0, 'custom_a'] == 2.0
    assert result.loc[0, 'custom_b'] == 20.0

def test_avg_latest_perf_ignores_irrelevant_columns(sample_ohlcv):
    """Output contains only symbol and perf_cols, not extra columns like close."""
    result = average_latest_performance(sample_ohlcv)
    assert 'close' not in result.columns
    assert set(result.columns) == {'symbol', '%Year', '%6Month', '%3Month', '%1Month'}

def test_avg_latest_perf_handles_missing_perf_column():
    """If a perf_col is missing, should raise KeyError."""
    df = pd.DataFrame({'symbol': ['A'], 'datetime': [pd.Timestamp.now()], '%Year': [1]})
    with pytest.raises(KeyError):
        average_latest_performance(df)  # expects all four default columns

# =============================================================================
# 2. attach_industry
# =============================================================================

def test_attach_industry_basic(sample_sector):
    perf = pd.DataFrame({'symbol': ['AAPL', 'MSFT', 'GOOGL']})
    result = attach_industry(perf, sample_sector)
    assert list(result['industry']) == ['Technology', 'Technology', 'Communication']

def test_attach_industry_case_insensitive(sample_sector):
    """Mapping is done via str.upper() → case insensitive."""
    perf = pd.DataFrame({'symbol': ['aapl', 'Msft', 'googl']})
    result = attach_industry(perf, sample_sector)
    assert list(result['industry']) == ['Technology', 'Technology', 'Communication']

def test_attach_industry_missing_symbol(sample_sector):
    """Symbol not in sector_info → NaN industry."""
    perf = pd.DataFrame({'symbol': ['AAPL', 'UNKNOWN']})
    result = attach_industry(perf, sample_sector)
    assert result.loc[0, 'industry'] == 'Technology'
    assert pd.isna(result.loc[1, 'industry'])

def test_attach_industry_duplicates_in_sector_info():
    """drop_duplicates keeps first occurrence, so mapping is consistent."""
    sector = pd.DataFrame({
        'Symbol': ['A', 'A'],
        'Industry': ['Tech', 'Finance']
    })
    perf = pd.DataFrame({'symbol': ['A']})
    result = attach_industry(perf, sector)
    assert result.loc[0, 'industry'] == 'Tech'  # first row kept

def test_attach_industry_does_not_mutate_original(sample_sector):
    perf = pd.DataFrame({'symbol': ['AAPL']})
    original = perf.copy()
    _ = attach_industry(perf, sample_sector)
    assert 'industry' not in perf.columns
    assert perf.equals(original)

# =============================================================================
# 3. add_intra_industry_ranks
# =============================================================================

@pytest.fixture
def perf_with_industry():
    """DataFrame ready for ranking, one industry with 3 stocks."""
    return pd.DataFrame({
        'symbol': ['A', 'B', 'C'],
        'industry': ['Tech', 'Tech', 'Tech'],
        '%Year':    [10, 20, 5],
        '%6Month':  [1, 2, 3],
        '%3Month':  [np.nan, 15, 10],
        '%1Month':  [0, 0, 0],   # all same
    })

def test_ranking_highest_gets_1(perf_with_industry):
    result = add_intra_industry_ranks(perf_with_industry)
    # %Year: values 10,20,5 → rank order: B=1, A=2, C=3
    assert result.loc[result['symbol']=='A', '%Year_rank'].values[0] == 2
    assert result.loc[result['symbol']=='B', '%Year_rank'].values[0] == 1
    assert result.loc[result['symbol']=='C', '%Year_rank'].values[0] == 3

def test_ranking_ignores_nan(perf_with_industry):
    """NaN values should remain NaN in rank column and not affect others."""
    result = add_intra_industry_ranks(perf_with_industry)
    # %3Month: B=15, C=10, A=NaN → B rank 1, C rank 2, A rank NaN
    assert pd.isna(result.loc[result['symbol'] == 'A', '%3Month_rank']).all()
    assert result.loc[result['symbol'] == 'B', '%3Month_rank'].values[0] == 1
    assert result.loc[result['symbol'] == 'C', '%3Month_rank'].values[0] == 2


def test_ranking_handles_infinite_values():
    df = pd.DataFrame({
        'symbol': ['A', 'B'],
        'industry': ['Fin', 'Fin'],
        '%Year': [np.inf, 10],
        '%6Month':  [8.0,2.0],
        '%3Month':  [6.0,2.0],
        '%1Month':  [3.0,3.0],
    })
    result = add_intra_industry_ranks(df)
    # inf replaced by NaN → only B gets rank 1, A remains NaN
    assert pd.isna(result.loc[0, '%Year_rank'])
    assert result.loc[1, '%Year_rank'] == 1


def test_ranking_ties_assign_different_ranks(perf_with_industry):
    """With identical values, sort order is stable but distinct ranks are assigned.
       This may be acceptable; we just verify no error and ranks are assigned."""
    result = add_intra_industry_ranks(perf_with_industry)
    # %1Month all 0 → each gets a rank 1,2,3 (order depends on index)
    ranks = result['%1Month_rank'].tolist()
    assert sorted(ranks) == [1, 2, 3]  # distinct ranks

def test_ranking_multiple_industries():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "industry": ["Tech", "Tech", "Fin", "Fin"],
            "%Year": [10.0, 20.0, 5.0, 15.0],
            "%6Month": [8.0, 18.0, 3.0, 12.0],
            "%3Month": [6.0, 12.0, 2.0, 9.0],
            "%1Month": [4.0, 10.0, 1.0, 7.0],
        }
    )
    result = add_intra_industry_ranks(df)
    # Tech: B(20) rank1, A(10) rank2
    assert result.loc[0, '%Year_rank'] == 2  # A
    assert result.loc[1, '%Year_rank'] == 1  # B
    # Fin: C(5) rank1, D(15) rank2
    assert result.loc[2, '%Year_rank'] == 2  # C
    assert result.loc[3, '%Year_rank'] == 1  # D

def test_ranking_empty_industry():
    df = pd.DataFrame({
        'symbol': ['AAPL'],
        'industry': [None],  # NaN group
        '%Year':    [10.0],
        '%6Month':  [8.0],
        '%3Month':  [6.0],
        '%1Month':  [3.0],
    })
    # Should still work, NaN group treated separately
    result = add_intra_industry_ranks(df)
    assert result.loc[0, '%Year_rank'] == 1

# =============================================================================
# 4. add_industry_stock_count
# =============================================================================

def test_stock_count():
    df = pd.DataFrame({
        'symbol': ['A', 'B', 'C'],
        'industry': ['Tech', 'Tech', 'Fin'],
    })
    result = add_industry_stock_count(df)
    assert list(result['industry_stock_count']) == [2, 2, 1]

# =============================================================================
# 5. compute_sector_rankings (integration)
# =============================================================================

def test_full_pipeline(sample_ohlcv, sample_sector):
    """End‑to‑end test with realistic data."""
    result = compute_sector_rankings(sample_ohlcv, sample_sector, rolling_window=1)

    # Check columns
    expected_cols = ['symbol', 'industry', '%Year_rank', '%6Month_rank',
                     '%3Month_rank', '%1Month_rank', 'industry_stock_count']
    assert list(result.columns) == expected_cols
    # Two symbols, both Technology → industry count 2
    assert result['industry_stock_count'].unique()[0] == 2
    # Rolling window 1 → last day values: AAPL %Year=20, MSFT %Year=15
    aapl = result[result['symbol'] == 'AAPL']
    msft = result[result['symbol'] == 'MSFT']
    # Ranks: AAPL %Year higher → rank1, MSFT rank2
    assert aapl['%Year_rank'].values[0] == 1
    assert msft['%Year_rank'].values[0] == 2
    # Verify the order is sorted by industry
    assert result['industry'].tolist() == ['Technology', 'Technology']

def test_full_pipeline_with_missing_industry(sample_ohlcv):
    """Symbol not in sector_info gets NaN industry, still ranked separately."""
    sector = pd.DataFrame({'Symbol': ['AAPL',], 'Industry': ['Tech']})
    result = compute_sector_rankings(sample_ohlcv, sector, rolling_window=1)

    # Make sure both symbols exist
    assert set(result['symbol']) == {'AAPL', 'MSFT'}

    aapl = result[result['symbol'] == 'AAPL'].iloc[0]
    msft = result[result['symbol'] == 'MSFT'].iloc[0]

    assert aapl['industry'] == 'Tech'
    assert pd.isna(msft['industry'])                    # NaN industry
    # Each symbol is alone in its industry group → rank 1 everywhere
    assert aapl['%Year_rank'] == 1
    assert msft['%Year_rank'] == 1
    # Industry stock count: AAPL’s industry (Tech) has 1, MSFT’s NaN group also 1
    assert aapl['industry_stock_count'] == 1
    assert msft['industry_stock_count'] == 1

def test_full_pipeline_empty_ohlcv():
    df = pd.DataFrame(columns=['symbol', 'datetime', '%Year', '%6Month', '%3Month', '%1Month'])
    sector = pd.DataFrame({'Symbol': ['X'], 'Industry': ['Tech']})
    result = compute_sector_rankings(df, sector)
    assert result.empty
    assert list(result.columns) == ['symbol', 'industry', '%Year_rank', '%6Month_rank',
                                    '%3Month_rank', '%1Month_rank', 'industry_stock_count']