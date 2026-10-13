"""Cleaning, aggregation, gap handling, and data-quality reporting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .analysis import detect_frequency, frequency_to_pandas
from .config import ForecastConfig
from .contracts import PreparedSeries, QualityReport, WarningMessage, InsufficientDataError

def parse_sales(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"[$€£¥₹]", "", regex=True).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\\1", regex=True)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "—": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")

def prepare_series(frame: pd.DataFrame, date_col: str, sales_col: str,
                   config: ForecastConfig | None = None, frequency: str | None = None,
                   dayfirst: bool | None = None) -> PreparedSeries:
    config = config or ForecastConfig()
    if date_col not in frame or sales_col not in frame:
        raise ValueError("Selected date or sales column is not present in the data.")
    work = frame.copy()
    rows_before = len(work)
    duplicate_rows = int(work.duplicated().sum())
    warnings: list[WarningMessage] = []
    if duplicate_rows:
        warnings.append(WarningMessage("DUPLICATE_ROWS", "warning",
            f"Found {duplicate_rows} exact duplicate rows; kept them in v1.", "preprocessing"))
    parsed_dates = pd.to_datetime(work[date_col], errors="coerce", format="mixed", dayfirst=bool(dayfirst) if dayfirst is not None else False)
    if dayfirst is None:
        # Surface ambiguous dates rather than silently claiming the chosen interpretation is certain.
        strings = work[date_col].dropna().astype(str)
        ambiguous = strings.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$").any()
        if ambiguous:
            warnings.append(WarningMessage("DATE_FORMAT_AMBIGUOUS", "warning",
                "Some dates may be day-first or month-first. Review the parsed date range and rerun with dayfirst=True if needed.", "preprocessing"))
    parsed_sales = parse_sales(work[sales_col])
    invalid_dates = int(parsed_dates.isna().sum())
    missing_sales = int(work[sales_col].isna().sum())
    invalid_sales = int((parsed_sales.isna() & work[sales_col].notna()).sum())
    negative_sales = int((parsed_sales < 0).sum())
    if invalid_dates:
        warnings.append(WarningMessage("INVALID_DATES_DROPPED", "warning", f"Dropped {invalid_dates} rows with unparseable dates.", "preprocessing"))
    if invalid_sales or missing_sales:
        warnings.append(WarningMessage("INVALID_SALES_EXCLUDED", "warning",
            f"{missing_sales} missing sales and {invalid_sales} non-numeric sales values were excluded from aggregation.", "preprocessing"))
    if negative_sales:
        warnings.append(WarningMessage("NEGATIVE_SALES", "warning", f"Found {negative_sales} negative sales values, likely returns; retained them.", "preprocessing"))
    work["_parsed_date"] = parsed_dates
    work["_parsed_sales"] = parsed_sales
    valid = work.loc[work["_parsed_date"].notna() & work["_parsed_sales"].notna()].copy()
    if valid.empty:
        raise InsufficientDataError("No rows contain both a valid date and numeric sales value.")
    inferred, confidence, alternatives = detect_frequency(valid["_parsed_date"])
    chosen = frequency or config.frequency
    if chosen == "auto":
        chosen = inferred
    chosen = frequency_to_pandas(chosen)
    if chosen != inferred:
        warnings.append(WarningMessage("FREQUENCY_OVERRIDE", "info",
            f"Using requested frequency {chosen}; automatic suggestion was {inferred}.", "preprocessing"))
    if chosen == "D":
        valid["_period"] = valid["_parsed_date"].dt.floor("D")
        grouped = valid.groupby("_period")["_parsed_sales"].sum().sort_index()
        full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    elif chosen == "W-SUN":
        valid["_period"] = valid["_parsed_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
        grouped = valid.groupby("_period")["_parsed_sales"].sum().sort_index()
        full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="W-SUN")
    elif chosen == "MS":
        valid["_period"] = valid["_parsed_date"].dt.to_period("M").dt.to_timestamp()
        grouped = valid.groupby("_period")["_parsed_sales"].sum().sort_index()
        full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="MS")
    else:
        valid["_period"] = valid["_parsed_date"].dt.to_period("Q").dt.start_time
        grouped = valid.groupby("_period")["_parsed_sales"].sum().sort_index()
        full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="QS")
    gaps = int(len(full_idx.difference(grouped.index)))
    series = grouped.reindex(full_idx)
    if gaps:
        warnings.append(WarningMessage("MISSING_PERIODS", "warning",
            f"Found {gaps} missing periods; applied gap policy {config.gap_policy!r}.", "preprocessing"))
        if config.gap_policy == "zero":
            series = series.fillna(0)
        elif config.gap_policy == "interpolate":
            series = series.interpolate(limit_direction="both")
        elif config.gap_policy != "missing":
            raise ValueError("gap_policy must be 'zero', 'interpolate', or 'missing'.")
    # Outliers are flagged, not removed.
    numeric = series.dropna()
    outliers = []
    if len(numeric) >= 4:
        q1, q3 = numeric.quantile([.25, .75])
        spread = q3 - q1
        if spread > 0:
            outliers = [str(idx.date()) for idx, value in numeric.items() if value < q1 - 1.5 * spread or value > q3 + 1.5 * spread]
    quality = QualityReport(
        rows_before=rows_before, rows_after=len(valid), dropped_unparseable_dates=invalid_dates,
        dropped_invalid_sales=invalid_sales, missing_sales=missing_sales, duplicate_rows=duplicate_rows,
        negative_sales=negative_sales, date_start=str(valid["_parsed_date"].min()),
        date_end=str(valid["_parsed_date"].max()), periods=len(series), gaps=gaps,
        zero_period_share=float((series.dropna() == 0).mean()) if series.notna().any() else 0.0,
        outlier_periods=outliers)
    transactions = valid.drop(columns=["_period"], errors="ignore").rename(columns={"_parsed_date": "date", "_parsed_sales": "sales"})
    series.name = "sales"
    if len(series.dropna()) < config.min_periods_to_forecast:
        warnings.append(WarningMessage("SHORT_HISTORY", "warning",
            f"Only {len(series.dropna())} usable periods; only baseline forecasting may be reliable.", "preprocessing"))
    return PreparedSeries(transactions, series.astype(float), chosen, confidence, alternatives, quality, warnings)
