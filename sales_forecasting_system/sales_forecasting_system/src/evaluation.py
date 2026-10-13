"""Leakage-safe chronological evaluation and model selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import ForecastConfig
from .contracts import EvaluationResult, ModelMetrics, WarningMessage
from .models import fit_model

MODEL_ORDER = {"naive": 0, "ets": 1, "arima": 2}

def _metrics(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> dict[str, float | None]:
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.abs(actual) + np.abs(predicted)
    smape = float(np.mean(np.divide(200 * np.abs(err), denom, out=np.zeros_like(err, dtype=float), where=denom != 0)))
    scale = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else 0.0
    mase = float(mae / scale) if scale > 0 else None
    return {"mae": mae, "rmse": rmse, "mase": mase, "smape": smape}

def _folds(n: int, horizon: int, config: ForecastConfig) -> list[tuple[int, int]]:
    test_size = max(1, min(horizon, max(1, int(n * config.holdout_fraction)), 20))
    min_train = max(4, n - test_size * min(config.max_rolling_folds, 3))
    folds = []
    start = min_train
    while start < n and len(folds) < config.max_rolling_folds:
        end = min(n, start + test_size)
        if end > start:
            folds.append((start, end))
        start = end
    if not folds:
        split = max(1, n - test_size)
        folds = [(split, n)] if split > 0 else []
    return folds

def evaluate_models(series: pd.Series, frequency: str, config: ForecastConfig) -> EvaluationResult:
    y = series.dropna().astype(float)
    if len(y) < 4:
        # No meaningful holdout; preserve the requested baseline behavior.
        return EvaluationResult([ModelMetrics("naive", True, None)], "naive",
            "History is too short for reliable model comparison; selected the naive baseline.", "short-history fallback",
            [WarningMessage("SHORT_VALIDATION_HISTORY", "warning", "Insufficient history for robust validation.", "evaluation")])
    horizon = max(1, min(config.horizon, 20))
    folds = _folds(len(y), horizon, config)
    records = {name: [] for name in ("naive", "ets", "arima")}
    reasons = {}
    for start, end in folds:
        train = y.iloc[:start]
        actual = y.iloc[start:end].to_numpy(dtype=float)
        assert train.index.max() < y.index[start], "Leakage guard: test indices must be later than training indices."
        for name in records:
            try:
                model, fit_result = fit_model(name, train, frequency, config)
                if not fit_result.eligible:
                    reasons[name] = fit_result.reason
                    continue
                prediction = model.forecast(end-start)["forecast"].to_numpy(dtype=float)
                records[name].append(_metrics(actual, prediction, train.to_numpy(dtype=float)))
            except Exception as exc:
                reasons[name] = str(exc)
    metrics = []
    for name, values in records.items():
        if not values:
            metrics.append(ModelMetrics(name, False, reasons.get(name, "Model did not produce a valid validation forecast.")))
            continue
        maes = [v["mae"] for v in values]
        metrics.append(ModelMetrics(name, True, None, float(np.mean(maes)),
            float(np.mean([v["rmse"] for v in values])),
            _mean_optional([v["mase"] for v in values]),
            _mean_optional([v["smape"] for v in values]), [float(x) for x in maes]))
    eligible = [m for m in metrics if m.eligible and m.mae is not None]
    warnings = []
    if not eligible:
        raise ValueError("All forecasting models failed validation.")
    by_mae = sorted(eligible, key=lambda m: (m.mae, MODEL_ORDER[m.model]))
    naive = next((m for m in eligible if m.model == "naive"), None)
    winner = by_mae[0]
    if config.model != "auto":
        chosen = next((m for m in eligible if m.model == config.model), None)
        if chosen is None:
            warnings.append(WarningMessage("REQUESTED_MODEL_INELIGIBLE", "warning",
                f"Requested model {config.model!r} was not eligible; selected the best eligible model.", "evaluation"))
        else:
            winner = chosen
    elif naive and winner.model != "naive" and winner.mae >= naive.mae * (1 - config.baseline_improvement_margin):
        winner = naive
        warnings.append(WarningMessage("BASELINE_NOT_BEATEN", "info",
            "No alternative beat the naive baseline by the configured 5% MAE margin; selected naive.", "evaluation"))
    rationale = (f"Selected {winner.model} using chronological validation and lowest mean MAE "
                 f"(MAE={winner.mae:.3f}). Tie-breaking favors naive, then ETS, then ARIMA. "
                 "Forecast intervals are approximate.")
    return EvaluationResult(metrics, winner.model, rationale,
                            "expanding chronological folds" if len(folds) > 1 else "single chronological holdout", warnings)

def _mean_optional(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(valid)) if valid else None
