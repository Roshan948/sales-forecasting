"""JSON-safe conversion helpers for engine results."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import math
import numpy as np
import pandas as pd

def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return [{"date": _safe(k), **{str(c): _safe(v) for c, v in row.items()}} for k, row in value.set_index(value.columns[0]).iterrows()] if len(value.columns) else []
    if isinstance(value, pd.Series):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return _safe(value)

def _safe(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value
