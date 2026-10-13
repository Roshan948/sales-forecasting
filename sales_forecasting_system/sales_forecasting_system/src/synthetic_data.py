"""Deterministic synthetic sales data generator for demos and regression tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def generate_sales_data(path: str | Path, periods: int = 180, seed: int = 42,
                        frequency: str = "D", messy: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=periods, freq=frequency)
    t = np.arange(periods)
    weekly = 12 * np.sin(2 * np.pi * t / 7) if frequency == "D" else 0
    yearly = 20 * np.sin(2 * np.pi * t / 365.25)
    sales = np.maximum(0, 100 + .25 * t + weekly + yearly + rng.normal(0, 8, periods))
    frame = pd.DataFrame({"Order Date": dates, "Sales": sales.round(2), "Region": rng.choice(["East", "West", "Central"], periods)})
    if messy:
        frame = frame.drop(index=frame.index[::23]).reset_index(drop=True)
        frame["Order Date"] = frame["Order Date"].dt.strftime("%d/%m/%Y")
        frame = frame.rename(columns={"Order Date": "order_day", "Sales": "Net Revenue"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

if __name__ == "__main__":
    generate_sales_data("data/synthetic/sales_clean.csv")
    generate_sales_data("data/synthetic/sales_messy.csv", messy=True)
    print("Created synthetic datasets in data/synthetic/")
