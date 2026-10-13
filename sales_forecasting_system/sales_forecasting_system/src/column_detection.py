"""Heuristic column detection with explicit confidence and user overrides."""
from __future__ import annotations
import re
import pandas as pd
from .config import ForecastConfig
from .contracts import ColumnCandidate, ColumnDetectionResult, WarningMessage, ColumnDetectionError

DATE_WORDS = ("date", "day", "period", "month", "year", "time", "order date", "ship date")
SALES_WORDS = ("sales", "revenue", "amount", "total", "turnover", "net sales", "qty", "quantity", "profit")
ID_WORDS = ("id", "postal", "zip", "code", "number", "no.")

def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(r"[$€£¥₹,]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\\1", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def detect_columns(frame: pd.DataFrame, config: ForecastConfig | None = None) -> ColumnDetectionResult:
    config = config or ForecastConfig()
    date_candidates, sales_candidates = [], []
    for col in frame.columns:
        s = frame[col]
        name = str(col).strip().lower()
        reasons = []
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        parse_share = float(parsed.notna().mean()) if len(s) else 0
        keyword = any(w in name for w in DATE_WORDS)
        dtype_signal = 1.0 if pd.api.types.is_datetime64_any_dtype(s) else 0.0
        range_signal = 1.0 if parsed.notna().any() and parsed.dropna().dt.year.between(1990, 2100).mean() > .8 else 0.0
        monotonic_signal = 1.0 if parsed.notna().sum() > 2 and parsed.dropna().is_monotonic_increasing else 0.0
        score = min(1.0, .40 * parse_share + .25 * float(keyword) + .15 * dtype_signal + .10 * range_signal + .10 * monotonic_signal)
        if keyword: reasons.append("date-like header")
        if parse_share: reasons.append(f"{parse_share:.0%} values parse as dates")
        if monotonic_signal: reasons.append("mostly/increasing date order")
        if score > 0:
            date_candidates.append(ColumnCandidate(str(col), round(score, 3), reasons))

        nums = _numeric(s)
        numeric_share = float(nums.notna().mean()) if len(s) else 0
        nonnegative_share = float((nums.dropna() >= 0).mean()) if nums.notna().any() else 0
        variance_signal = 1.0 if nums.nunique(dropna=True) > 1 else 0.0
        id_like = any(w in name for w in ID_WORDS) or (nums.notna().sum() > 0 and nums.nunique(dropna=True) / max(1, nums.notna().sum()) > .98 and nums.dropna().diff().dropna().eq(1).mean() > .7)
        sales_keyword = any(w in name for w in SALES_WORDS)
        score_s = .45 * numeric_share + .30 * float(sales_keyword) + .10 * nonnegative_share + .15 * variance_signal
        if id_like: score_s -= .45
        score_s = max(0.0, min(1.0, score_s))
        reasons_s = []
        if sales_keyword: reasons_s.append("sales-like header")
        reasons_s.append(f"{numeric_share:.0%} values numeric")
        if id_like: reasons_s.append("ID-like column penalized")
        if score_s > 0:
            sales_candidates.append(ColumnCandidate(str(col), round(score_s, 3), reasons_s))

    date_candidates.sort(key=lambda x: x.confidence, reverse=True)
    sales_candidates.sort(key=lambda x: x.confidence, reverse=True)
    selected_date = date_candidates[0].column if date_candidates else None
    selected_sales = sales_candidates[0].column if sales_candidates else None
    date_gap = date_candidates[0].confidence - date_candidates[1].confidence if len(date_candidates) > 1 else 1
    sales_gap = sales_candidates[0].confidence - sales_candidates[1].confidence if len(sales_candidates) > 1 else 1
    needs = (selected_date is None or selected_sales is None or
             date_candidates[0].confidence < config.date_confidence_threshold if date_candidates else True)
    needs = needs or (sales_candidates[0].confidence < config.sales_confidence_threshold if sales_candidates else True)
    needs = needs or date_gap < config.confidence_gap or sales_gap < config.confidence_gap
    warnings = []
    if needs:
        warnings.append(WarningMessage("COLUMN_CONFIRMATION_REQUIRED", "warning",
            "Column detection is uncertain. Review or select the date and sales columns before forecasting.", "column_detection"))
    return ColumnDetectionResult(date_candidates, sales_candidates, selected_date, selected_sales, bool(needs), warnings)

def apply_user_selection(result: ColumnDetectionResult, frame: pd.DataFrame, date_col: str, sales_col: str) -> ColumnDetectionResult:
    if date_col not in frame.columns:
        raise ColumnDetectionError(f"Selected date column {date_col!r} does not exist.")
    if sales_col not in frame.columns:
        raise ColumnDetectionError(f"Selected sales column {sales_col!r} does not exist.")
    if date_col == sales_col:
        raise ColumnDetectionError("Date and sales columns must be different.")
    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    if dates.notna().mean() < .5:
        raise ColumnDetectionError(f"Selected date column {date_col!r} has fewer than 50% parseable dates.")
    nums = _numeric(frame[sales_col])
    if nums.notna().mean() < .5:
        raise ColumnDetectionError(f"Selected sales column {sales_col!r} has fewer than 50% numeric values.")
    result.selected_date = date_col
    result.selected_sales = sales_col
    result.needs_confirmation = False
    result.warnings = [w for w in result.warnings if w.code != "COLUMN_CONFIRMATION_REQUIRED"]
    return result
