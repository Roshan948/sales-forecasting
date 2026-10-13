# Data contracts and handling rules

The pipeline exchanges typed dataclasses defined in `src/contracts.py`. The core engine never imports Streamlit.

## Contracts
- `IngestionResult`: dataframe, file metadata, warnings.
- `ColumnDetectionResult`: ranked date/sales candidates, confidence, confirmation state.
- `PreparedSeries`: transaction-level rows, aggregated regular series, inferred frequency, quality report, warnings.
- `ModelResult`: eligibility, fitted values, residual estimate, parameters and diagnostics.
- `EvaluationResult`: per-model metrics, chosen model, validation method, rationale.
- `ForecastResult`: status, historical data, forecast intervals, selected model, quality, metadata and warnings.

## Input rules
- Supported: CSV, XLSX, XLS; 100 MB default limit.
- CSV encodings attempted in order: UTF-8, UTF-8 with BOM, CP1252, Latin-1.
- CSV delimiter sniffing supports comma, semicolon, tab and pipe.
- Currency symbols and thousands separators are stripped from sales values.
- Parentheses in sales values indicate negatives, e.g. `(125.00)` becomes `-125.00`.
- Negative sales are retained and flagged as likely returns.
- Missing or non-numeric sales are reported and excluded from sums; they are not silently converted to zero.
- Unparseable dates are reported and excluded.
- Exact duplicate rows are reported and retained in v1.
- Missing time periods default to zero with a warning; alternatives are interpolation or missing values.
- Ambiguous day-first/month-first dates are surfaced with a warning; the UI lets the user choose day-first.
- Product and region columns may be present but are not used for v1 forecasts.

## Defaults
See `src/config.py`. They are starting proposals from the supplied development plan and should be tuned after inspecting the test dataset.
