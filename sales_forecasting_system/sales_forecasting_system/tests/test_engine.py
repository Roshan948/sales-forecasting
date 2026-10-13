import pandas as pd
from src.engine import run_forecast
from src.config import ForecastConfig

def test_engine_end_to_end(tmp_path):
    p = tmp_path / "sales.csv"
    dates = pd.date_range("2023-01-01", periods=70, freq="D")
    pd.DataFrame({"Order Date": dates, "Sales": [100 + i * .5 for i in range(70)]}).to_csv(p, index=False)
    result = run_forecast(p, ForecastConfig(horizon=5, min_periods_to_forecast=8, min_periods_arima=20))
    assert result.status == "success"
    assert len(result.forecast) == 5
    assert result.selected_model in {"naive", "ets", "arima"}
