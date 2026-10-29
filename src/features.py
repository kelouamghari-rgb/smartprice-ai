"""Feature engineering shared by training, the optimizer and the app."""
import numpy as np
import pandas as pd

from .config import PEAK_HOURS

CATEGORICAL = ["product"]
NUMERIC = ["hour", "day_of_week", "price_index", "promotion", "is_weekend",
           "is_peak_hour", "log_price", "log_price_x_peak"]
FEATURE_COLS = CATEGORICAL + NUMERIC

# Economic common sense baked into the model: a higher price never raises demand,
# a promotion never lowers it.
MONOTONIC = {"price_index": -1, "log_price": -1, "log_price_x_peak": -1, "promotion": 1}

REQUIRED_COLS = ["product", "hour", "day_of_week", "price_mad", "promotion"]


def build_features(df: pd.DataFrame, reference_prices: dict) -> pd.DataFrame:
    """Turn raw rows into model features.

    Unit cost is deliberately NOT a feature: customers react to price, not to what
    the product costs the shop. `price_index` (price / menu price) lets all products
    share what they learn about price sensitivity.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    unknown = set(df["product"]) - set(reference_prices)
    if unknown:
        raise ValueError(f"Unknown products: {sorted(unknown)}")

    out = pd.DataFrame(index=df.index)
    out["product"] = df["product"]
    out["hour"] = df["hour"]
    out["day_of_week"] = df["day_of_week"]
    out["price_index"] = df["price_mad"] / df["product"].map(reference_prices)
    out["promotion"] = df["promotion"].astype(int)
    out["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    out["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    # Log-price terms let the model learn constant-elasticity behaviour, and how it
    # differs between peak and off-peak hours.
    out["log_price"] = np.log(out["price_index"])
    out["log_price_x_peak"] = out["log_price"] * out["is_peak_hour"]
    return out[FEATURE_COLS]