"""Central configuration and reproducible defaults."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 12
    frequency: str = "auto"              # auto, D, W-SUN, MS, QS
    model: str = "auto"                  # auto, naive, arima, ets
    date_confidence_threshold: float = 0.80
    sales_confidence_threshold: float = 0.80
    confidence_gap: float = 0.20
    min_periods_to_forecast: int = 8
    min_periods_arima: int = 20
    max_rolling_folds: int = 5
    holdout_fraction: float = 0.20
    baseline_improvement_margin: float = 0.05
    gap_policy: str = "zero"             # zero, interpolate, missing
    max_file_mb: int = 100
    random_seed: int = 42
    max_arima_candidates: int = 20
    seasonal_period: int | None = None
