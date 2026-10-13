"""Naive, ARIMA, and exponential-smoothing forecasting models."""
from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .config import ForecastConfig
from .contracts import ModelResult, WarningMessage

class Forecaster(ABC):
    name = "base"
    def __init__(self, series: pd.Series, frequency: str, config: ForecastConfig):
        self.series = series.astype(float).dropna()
        self.frequency = frequency
        self.config = config
        self.fit_result = None
        self.residual_std = float(self.series.diff().dropna().std()) if len(self.series) > 1 else 0.0
    @abstractmethod
    def fit(self) -> ModelResult: ...
    @abstractmethod
    def forecast(self, horizon: int) -> pd.DataFrame: ...
    def describe(self) -> dict:
        return {"model": self.name, "frequency": self.frequency, "observations": len(self.series)}

class NaiveForecaster(Forecaster):
    name = "naive"
    def fit(self) -> ModelResult:
        if len(self.series) < 1:
            return ModelResult(self.name, False, "No usable observations.")
        self.last = float(self.series.iloc[-1])
        self.residual_std = float((self.series - self.series.shift(1)).dropna().std()) if len(self.series) > 1 else 0.0
        self.residual_std = 0.0 if not np.isfinite(self.residual_std) else self.residual_std
        return ModelResult(self.name, True, fitted_values=self.series.shift(1), residual_std=self.residual_std, params={"last_value": self.last})
    def forecast(self, horizon: int) -> pd.DataFrame:
        idx = _future_index(self.series.index, horizon, self.frequency)
        mean = np.repeat(self.last, horizon)
        spread = 1.96 * (self.residual_std or 0.0) * np.sqrt(np.arange(1, horizon + 1))
        return pd.DataFrame({"forecast": mean, "lower": mean-spread, "upper": mean+spread}, index=idx)

class ArimaForecaster(Forecaster):
    name = "arima"
    def fit(self) -> ModelResult:
        if len(self.series) < self.config.min_periods_arima:
            return ModelResult(self.name, False, f"Needs at least {self.config.min_periods_arima} observations.")
        if self.series.nunique() <= 1:
            return ModelResult(self.name, False, "Constant series; ARIMA is unnecessary.")
        best, best_aic, best_order = None, float("inf"), None
        candidates = list(itertools.product(range(4), range(2), range(4)))[:self.config.max_arima_candidates]
        for order in candidates:
            try:
                fit = ARIMA(self.series, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best, best_aic, best_order = fit, fit.aic, order
            except Exception:
                continue
        if best is None:
            return ModelResult(self.name, False, "All bounded ARIMA fits failed.")
        self.fit_result = best
        resid = np.asarray(best.resid, dtype=float)
        self.residual_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.0
        return ModelResult(self.name, True, fitted_values=pd.Series(best.fittedvalues, index=self.series.index[-len(best.fittedvalues):]),
                           residual_std=self.residual_std, params={"order": best_order, "aic": float(best_aic)})
    def forecast(self, horizon: int) -> pd.DataFrame:
        pred = self.fit_result.get_forecast(steps=horizon)
        ci = np.asarray(pred.conf_int())
        return pd.DataFrame({"forecast": np.asarray(pred.predicted_mean), "lower": ci[:,0], "upper": ci[:,1]},
                            index=_future_index(self.series.index, horizon, self.frequency))

class ETSForecaster(Forecaster):
    name = "ets"
    def fit(self) -> ModelResult:
        n = len(self.series)
        if n < 4:
            return ModelResult(self.name, False, "ETS needs at least four observations.")
        if self.series.nunique() <= 1:
            return ModelResult(self.name, False, "Constant series; use the naive baseline.")
        seasonal_period = self.config.seasonal_period or {"D": 7, "W-SUN": 52, "MS": 12, "QS": 4}.get(self.frequency, 7)
        candidates = [(None, False)]
        candidates.append(("add", False))
        if n >= 2 * seasonal_period:
            candidates.extend([("add", True)])
            if (self.series > 0).all():
                candidates.append(("mul", True))
        best, best_aic, best_spec = None, float("inf"), None
        for seasonal, use_trend in candidates:
            try:
                kwargs = {"trend": "add" if use_trend else None, "initialization_method": "estimated"}
                if seasonal:
                    kwargs.update(seasonal=seasonal, seasonal_periods=seasonal_period)
                fit = ExponentialSmoothing(self.series, **kwargs).fit(optimized=True)
                aic = float(getattr(fit, "aic", np.inf))
                if np.isfinite(aic) and aic < best_aic:
                    best, best_aic, best_spec = fit, aic, {"seasonal": seasonal, "trend": use_trend, "seasonal_period": seasonal_period}
            except Exception:
                continue
        if best is None:
            return ModelResult(self.name, False, "All ETS variants failed to fit.")
        self.fit_result = best
        resid = np.asarray(best.resid, dtype=float)
        self.residual_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.0
        return ModelResult(self.name, True, fitted_values=pd.Series(best.fittedvalues, index=self.series.index),
                           residual_std=self.residual_std, params=best_spec or {}, diagnostics={"aic": best_aic})
    def forecast(self, horizon: int) -> pd.DataFrame:
        mean = np.asarray(self.fit_result.forecast(horizon))
        spread = 1.96 * (self.residual_std or 0.0) * np.sqrt(np.arange(1, horizon+1))
        return pd.DataFrame({"forecast": mean, "lower": mean-spread, "upper": mean+spread},
                            index=_future_index(self.series.index, horizon, self.frequency))

def _future_index(index: pd.Index, horizon: int, frequency: str) -> pd.DatetimeIndex:
    last = pd.Timestamp(index[-1])
    freq = frequency
    if freq == "D": start = last + pd.Timedelta(days=1)
    elif freq == "W-SUN": start = last + pd.Timedelta(days=7)
    elif freq == "MS": start = last + pd.offsets.MonthBegin(1)
    elif freq == "QS": start = last + pd.offsets.QuarterBegin(startingMonth=1)
    else: start = last + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq=freq)

def build_forecaster(name: str, series: pd.Series, frequency: str, config: ForecastConfig) -> Forecaster:
    classes = {"naive": NaiveForecaster, "arima": ArimaForecaster, "ets": ETSForecaster}
    if name not in classes:
        raise ValueError(f"Unknown model {name!r}. Choose naive, arima, or ets.")
    return classes[name](series, frequency, config)

def fit_model(name: str, series: pd.Series, frequency: str, config: ForecastConfig) -> tuple[Forecaster, ModelResult]:
    model = build_forecaster(name, series, frequency, config)
    result = model.fit()
    return model, result
