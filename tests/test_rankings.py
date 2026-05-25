import pandas as pd

from rankings import compute_sector_rankings

def test_ranking_highest_gets_rank():
    ohlcv = pd.DataFrame({
        'symbol': ['A', 'A', 'B', 'B'],
        'datetime': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        '%Year': [10, 20, 5, 5],
        '%6Month': [1, 2, 3, 3],
        '%3Month': [0, 0, 0, 0],
        '%1Month': [0, 0, 0, 0],
    })
    sector = pd.DataFrame({
        'Symbol': ['A', 'B'],
        'Industry': ['Tech', 'Tech']
    })
    result = compute_sector_rankings(ohlcv, sector, rolling_window=2)

    # For %Year (latest values: A=20, B=5) → A rank 1, B rank 2
    assert result.loc[result['symbol'] == 'A', '%Year_rank'].iloc[0] == 1
    assert result.loc[result['symbol'] == 'B', '%Year_rank'].iloc[0] == 2
    # Industry stock count should be 2
    assert result['industry_stock_count'].iloc[0] == 2

if __name__ == '__main__':
    test_ranking_highest_gets_rank()

