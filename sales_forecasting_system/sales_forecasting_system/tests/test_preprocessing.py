import pandas as pd
from src.preprocessing import prepare_series
from src.config import ForecastConfig

def test_daily_aggregation_and_gap_fill():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-03"], "sales": [10, 30]})
    result = prepare_series(df, "date", "sales", ForecastConfig(min_periods_to_forecast=1), frequency="D")
    assert result.series.tolist() == [10.0, 0.0, 30.0]
    assert result.quality.gaps == 1

def test_negative_sales_retained():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "sales": [10, -2, 4]})
    result = prepare_series(df, "date", "sales", ForecastConfig(min_periods_to_forecast=1), frequency="D")
    assert result.series.sum() == 12
    assert result.quality.negative_sales == 1
