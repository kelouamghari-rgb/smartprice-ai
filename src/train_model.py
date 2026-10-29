"""Train the demand model.  Run: python -m src.train_model"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_poisson_deviance,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATALOG, DATA_PATH, METRICS_PATH, MODEL_PATH
from .features import CATEGORICAL, FEATURE_COLS, MONOTONIC, NUMERIC, build_features

REFERENCE_PRICES = {name: p.base_price for name, p in CATALOG.items()}


def make_pipeline() -> Pipeline:
    """Poisson gradient boosting (units sold are counts) with monotonic constraints."""
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[list(CATALOG)], handle_unknown="ignore",
                              sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    # One constraint per transformed column: product one-hots first, then numerics.
    constraints = [0] * len(CATALOG) + [MONOTONIC.get(c, 0) for c in NUMERIC]
    regressor = HistGradientBoostingRegressor(
        loss="poisson",
        monotonic_cst=constraints,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=8,
        min_samples_leaf=30,
        early_stopping=False,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, baseline: np.ndarray) -> dict:
    pred = np.clip(pipe.predict(X), 1e-6, None)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "poisson_deviance": float(mean_poisson_deviance(y, pred)),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
    }


def train() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run: python -m src.generate_data")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    X = build_features(df, REFERENCE_PRICES)
    y = df["units_sold"]

    # Honest evaluation: train on the past, test on the future (no random shuffling).
    cutoff = df["date"].sort_values().unique()[int(df["date"].nunique() * 0.8)]
    is_train = (df["date"] < cutoff).to_numpy()
    X_tr, X_te, y_tr, y_te = X[is_train], X[~is_train], y[is_train], y[~is_train]

    # Naive baseline: average sales per product & hour. The model must beat this.
    means = y_tr.groupby([X_tr["product"], X_tr["hour"]]).mean()
    baseline = pd.Series(list(zip(X_te["product"], X_te["hour"]))).map(means).to_numpy()

    pipe = make_pipeline().fit(X_tr, y_tr)
    metrics = evaluate(pipe, X_te, y_te, baseline)
    print("--- Held-out (future) metrics ---")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:.4f}")
    if metrics["mae"] >= metrics["baseline_mae"]:
        print("WARNING: model does not beat the naive baseline.")

    # Production model uses all the data.
    final = make_pipeline().fit(X, y)

    support = X.groupby("product")["price_index"].agg(["min", "max"])
    bundle = {
        "pipeline": final,
        "reference_prices": REFERENCE_PRICES,
        "price_support": {p: (float(r["min"]), float(r["max"])) for p, r in support.iterrows()},
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "sklearn_version": sklearn.__version__,
        "features": FEATURE_COLS,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({k: v for k, v in bundle.items() if k != "pipeline"}, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    train()