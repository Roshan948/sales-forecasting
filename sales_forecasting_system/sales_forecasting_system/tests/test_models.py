import numpy as np
import pandas as pd
from src.models import fit_model
from src.config import ForecastConfig

def test_naive_forecast_length_and_intervals():
    y = pd.Series(np.arange(1, 20, dtype=float), index=pd.date_range("2024-01-01", periods=19, freq="D"))
    model, result = fit_model("naive", y, "D", ForecastConfig())
    forecast = model.forecast(5)
    assert result.eligible
    assert len(forecast) == 5
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
