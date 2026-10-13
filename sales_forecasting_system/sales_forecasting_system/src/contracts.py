"""Typed contracts shared by every pipeline stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

@dataclass
class WarningMessage:
    code: str
    severity: str
    message: str
    stage: str

@dataclass
class FileInfo:
    name: str
    format: str
    encoding: str | None
    sheet: str | None
    rows: int
    columns: list[str]
    size_bytes: int | None = None

@dataclass
class IngestionResult:
    frame: pd.DataFrame
    file_info: FileInfo
    warnings: list[WarningMessage] = field(default_factory=list)

@dataclass
class ColumnCandidate:
    column: str
    confidence: float
    reasons: list[str] = field(default_factory=list)

@dataclass
class ColumnDetectionResult:
    date_candidates: list[ColumnCandidate]
    sales_candidates: list[ColumnCandidate]
    selected_date: str | None
    selected_sales: str | None
    needs_confirmation: bool
    warnings: list[WarningMessage] = field(default_factory=list)

@dataclass
class QualityReport:
    rows_before: int
    rows_after: int
    dropped_unparseable_dates: int
    dropped_invalid_sales: int
    missing_sales: int
    duplicate_rows: int
    negative_sales: int
    date_start: str | None
    date_end: str | None
    periods: int
    gaps: int
    zero_period_share: float
    outlier_periods: list[str] = field(default_factory=list)

@dataclass
class PreparedSeries:
    transactions: pd.DataFrame
    series: pd.Series
    frequency: str
    frequency_confidence: float
    alternatives: list[str]
    quality: QualityReport
    warnings: list[WarningMessage] = field(default_factory=list)

@dataclass
class ModelResult:
    name: str
    eligible: bool
    reason: str | None = None
    fitted_values: pd.Series | None = None
    residual_std: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    model: str
    eligible: bool
    reason: str | None
    mae: float | None = None
    rmse: float | None = None
    mase: float | None = None
    smape: float | None = None
    fold_mae: list[float] = field(default_factory=list)

@dataclass
class EvaluationResult:
    metrics: list[ModelMetrics]
    selected_model: str
    rationale: str
    validation_method: str
    warnings: list[WarningMessage] = field(default_factory=list)

@dataclass
class ForecastResult:
    status: str
    historical: pd.DataFrame
    forecast: pd.DataFrame
    selected_model: str | None
    evaluation: EvaluationResult | None
    quality: QualityReport | None
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[WarningMessage] = field(default_factory=list)
    column_detection: ColumnDetectionResult | None = None
    error: str | None = None

class ForecastingError(Exception):
    """Base class for errors that can be shown safely to an end user."""
    stage = "unknown"

class FileFormatError(ForecastingError):
    stage = "ingestion"

class ColumnDetectionError(ForecastingError):
    stage = "column_detection"

class InsufficientDataError(ForecastingError):
    stage = "preprocessing"
