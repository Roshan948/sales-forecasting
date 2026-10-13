"""Time-series frequency inference."""
from __future__ import annotations
import numpy as np
import pandas as pd

def detect_frequency(dates: pd.Series | pd.DatetimeIndex) -> tuple[str, float, list[str]]:
    values = pd.Series(pd.to_datetime(dates, errors="coerce")).dropna().sort_values().drop_duplicates()
    if len(values) < 2:
        return "D", 0.0, ["D", "W-SUN", "MS", "QS"]
    gaps = values.diff().dropna().dt.total_seconds() / 86400
    med = float(gaps.median())
    if med <= 2.0:
        freq = "D"
    elif med <= 10:
        freq = "W-SUN"
    elif med <= 45:
        freq = "MS"
    elif med <= 120:
        freq = "QS"
    else:
        freq = "MS"
    common = {"D": ["D", "W-SUN", "MS", "QS"], "W-SUN": ["W-SUN", "D", "MS", "QS"],
              "MS": ["MS", "W-SUN", "QS", "D"], "QS": ["QS", "MS", "W-SUN", "D"]}
    # Confidence combines agreement between median and modal gaps with regularity.
    mode_gap = float(gaps.round().mode().iloc[0]) if not gaps.round().mode().empty else med
    regularity = 1 - min(1, float(np.median(np.abs(gaps - med))) / max(med, 1))
    agreement = 1.0 if abs(mode_gap - med) <= max(1, med * .25) else .5
    return freq, round(max(0, min(1, .6 * regularity + .4 * agreement)), 3), common[freq]

def frequency_to_pandas(freq: str) -> str:
    mapping = {"D": "D", "W": "W-SUN", "W-SUN": "W-SUN", "M": "MS", "MS": "MS",
               "Q": "QS", "QS": "QS"}
    if freq not in mapping:
        raise ValueError(f"Unsupported frequency {freq!r}. Use D, W-SUN, MS, or QS.")
    return mapping[freq]
