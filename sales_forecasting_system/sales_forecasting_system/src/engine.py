"""Single entry point orchestrating ingestion through final forecast."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import logging
import pandas as pd
from .config import ForecastConfig
from .contracts import ForecastResult, WarningMessage, ForecastingError, ColumnDetectionError
from .ingestion import load_file
from .column_detection import detect_columns, apply_user_selection
from .preprocessing import prepare_series
from .evaluation import evaluate_models
from .models import fit_model

logger = logging.getLogger(__name__)

def run_forecast(source, config: ForecastConfig | None = None, overrides: dict | None = None,
                 progress_callback=None) -> ForecastResult:
    """Run complete workflow. If column detection is ambiguous, return needs_confirmation."""
    config = config or ForecastConfig()
    overrides = overrides or {}
    warnings: list[WarningMessage] = []
    def progress(stage: str, value: float):
        if progress_callback:
            progress_callback(stage, value)
    try:
        progress("Loading data", .05)
        ingestion = load_file(source, config)
        warnings.extend(ingestion.warnings)
        progress("Detecting columns", .15)
        detection = detect_columns(ingestion.frame, config)
        date_col = overrides.get("date_col", detection.selected_date)
        sales_col = overrides.get("sales_col", detection.selected_sales)
        if date_col and sales_col:
            if overrides.get("date_col") or overrides.get("sales_col"):
                detection = apply_user_selection(detection, ingestion.frame, date_col, sales_col)
            elif detection.needs_confirmation:
                return ForecastResult("needs_confirmation", pd.DataFrame(), pd.DataFrame(), None, None, None,
                    "Please confirm the detected date and sales columns before forecasting.",
                    {"file": asdict(ingestion.file_info)}, warnings + detection.warnings, detection)
        else:
            return ForecastResult("needs_confirmation", pd.DataFrame(), pd.DataFrame(), None, None, None,
                "Could not confidently detect both date and sales columns. Select them in the interface.",
                {"file": asdict(ingestion.file_info)}, warnings + detection.warnings, detection)
        warnings.extend(detection.warnings)
        progress("Preparing time series", .30)
        prepared = prepare_series(ingestion.frame, date_col, sales_col, config,
                                  frequency=overrides.get("frequency"),
                                  dayfirst=overrides.get("dayfirst"))
        warnings.extend(prepared.warnings)
        series = prepared.series.dropna()
        if len(series) < 1:
            raise ValueError("No usable time-series values remain after preprocessing.")
        if len(series) < config.min_periods_to_forecast:
            warnings.append(WarningMessage("BASELINE_ONLY_SHORT_HISTORY", "warning",
                f"Fewer than {config.min_periods_to_forecast} periods: model selection may be unreliable.", "engine"))
        progress("Evaluating models", .45)
        evaluation = evaluate_models(series, prepared.frequency, config)
        warnings.extend(evaluation.warnings)
        progress("Refitting selected model", .75)
        model, fitted = fit_model(evaluation.selected_model, series, prepared.frequency, config)
        if not fitted.eligible:
            model, fitted = fit_model("naive", series, prepared.frequency, config)
            evaluation.selected_model = "naive"
            warnings.append(WarningMessage("REFIT_FALLBACK", "warning",
                "Selected model failed on full history; fell back to naive.", "engine"))
        forecast = model.forecast(config.horizon).copy()
        forecast.index.name = "date"
        forecast = forecast.reset_index()
        historical = series.rename("sales").rename_axis("date").reset_index()
        metadata = {"file": asdict(ingestion.file_info), "frequency": prepared.frequency,
                    "frequency_confidence": prepared.frequency_confidence,
                    "frequency_alternatives": prepared.alternatives, "date_column": date_col,
                    "sales_column": sales_col, "horizon": config.horizon,
                    "config": asdict(config), "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python_packages": {name: _version(name) for name in ("pandas", "numpy", "statsmodels", "scikit-learn")}}
        progress("Complete", 1.0)
        return ForecastResult("success", historical, forecast, evaluation.selected_model, evaluation,
                              prepared.quality, evaluation.rationale, metadata, warnings, detection)
    except ForecastingError:
        raise
    except Exception as exc:
        logger.exception("Forecast pipeline failed")
        raise RuntimeError(f"Forecast failed during processing: {exc}") from exc

def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
