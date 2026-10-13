"""CSV and Excel ingestion with diagnostics and defensive validation."""
from __future__ import annotations
import csv
import io
import os
from pathlib import Path
from typing import Any
import pandas as pd
from .config import ForecastConfig
from .contracts import FileInfo, IngestionResult, WarningMessage, FileFormatError

SUPPORTED = {".csv", ".xlsx", ".xls"}

def _read_csv(raw: bytes, name: str) -> tuple[pd.DataFrame, str, list[WarningMessage]]:
    warnings: list[WarningMessage] = []
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            sample = raw[:65536].decode(encoding, errors="strict")
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ","
            frame = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=sep, on_bad_lines="warn")
            if encoding not in ("utf-8", "utf-8-sig"):
                warnings.append(WarningMessage("CSV_ENCODING_FALLBACK", "info",
                    f"CSV was read using {encoding} encoding.", "ingestion"))
            if sep != ",":
                warnings.append(WarningMessage("CSV_DELIMITER_DETECTED", "info",
                    f"Detected delimiter {sep!r}.", "ingestion"))
            return frame, encoding, warnings
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise FileFormatError(f"Could not parse CSV file {name!r}: {last_error}")

def load_file(path_or_buffer: Any, config: ForecastConfig | None = None) -> IngestionResult:
    """Load CSV/XLS/XLSX from a path or uploaded file-like object."""
    config = config or ForecastConfig()
    name = getattr(path_or_buffer, "name", None) or (str(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else "uploaded_file")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise FileFormatError(f"Unsupported file type {suffix or '(no extension)'}. Upload CSV, XLSX, or XLS.")
    try:
        if isinstance(path_or_buffer, (str, Path)):
            size = os.path.getsize(path_or_buffer)
            if size > config.max_file_mb * 1024 * 1024:
                raise FileFormatError(f"File is larger than the {config.max_file_mb} MB limit.")
            raw = Path(path_or_buffer).read_bytes()
        else:
            raw = path_or_buffer.getvalue() if hasattr(path_or_buffer, "getvalue") else path_or_buffer.read()
            size = len(raw)
            if size > config.max_file_mb * 1024 * 1024:
                raise FileFormatError(f"File is larger than the {config.max_file_mb} MB limit.")
    except ForecastingError:
        raise
    except Exception as exc:
        raise FileFormatError(f"Unable to read file: {exc}") from exc

    warnings: list[WarningMessage] = []
    encoding = None
    sheet = None
    try:
        if suffix == ".csv":
            frame, encoding, warnings = _read_csv(raw, name)
        else:
            book = pd.ExcelFile(io.BytesIO(raw))
            chosen = None
            for candidate in book.sheet_names:
                preview = pd.read_excel(book, sheet_name=candidate, header=None, nrows=8)
                if not preview.dropna(how="all").empty:
                    chosen = candidate
                    break
            if chosen is None:
                raise FileFormatError("Excel workbook contains no non-empty worksheet.")
            sheet = chosen
            # Read without a header first so leading blank/title rows can be detected.
            raw_frame = pd.read_excel(book, sheet_name=sheet, header=None)
            nonempty_rows = raw_frame.dropna(how="all")
            if nonempty_rows.empty:
                raise FileFormatError("Selected worksheet is empty.")
            first_data_index = nonempty_rows.index[0]
            header_row = int(first_data_index)
            # If the first row is a single title cell, prefer the first row with multiple populated cells.
            for idx, row in raw_frame.head(10).iterrows():
                if row.notna().sum() >= 2:
                    header_row = int(idx)
                    break
            frame = pd.read_excel(book, sheet_name=sheet, header=header_row)
            encoding = None
    except FileFormatError:
        raise
    except Exception as exc:
        raise FileFormatError(f"Could not read {suffix} file {name!r}: {exc}") from exc

    if frame.empty:
        raise FileFormatError("The file has headers but no data rows.")
    if frame.shape[1] < 2:
        raise FileFormatError("The file has only one column; at least a date and a sales column are expected.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        warnings.append(WarningMessage("DUPLICATE_HEADERS", "warning",
            f"Duplicate column headers found: {duplicates}. They will be renamed uniquely.", "ingestion"))
        frame.columns = _unique_headers(frame.columns)
    null_columns = frame.columns[frame.isna().all()].astype(str).tolist()
    if null_columns:
        warnings.append(WarningMessage("ALL_NULL_COLUMNS", "warning",
            f"All-null columns detected: {null_columns}.", "ingestion"))
    frame.columns = [str(c).strip() if str(c).strip() else f"unnamed_{i+1}" for i, c in enumerate(frame.columns)]
    info = FileInfo(name=Path(name).name, format=suffix[1:].upper(), encoding=encoding, sheet=sheet,
                    rows=len(frame), columns=list(frame.columns), size_bytes=size)
    return IngestionResult(frame=frame, file_info=info, warnings=warnings)

def _unique_headers(columns) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for i, col in enumerate(columns):
        base = str(col).strip() or f"unnamed_{i+1}"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return out
