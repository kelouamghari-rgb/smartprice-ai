from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cafe_sales.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer business-specific domain features."""
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 12, 13, 19]).astype(int)
    df["margin_ratio"] = (df["price_mad"] - df["unit_cost_mad"]) / df["price_mad"]
    return df

def train():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = [
        "hour", "day_of_week", "product", "price_mad", 
        "unit_cost_mad", "promotion", "is_weekend", "is_peak_hour", "margin_ratio"
    ]
    
    X = df[feature_cols]
    y = df["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    categorical_features = ["product"]
    numerical_features = [c for c in feature_cols if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(random_state=42, n_estimators=150, learning_rate=0.08, max_depth=6)),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"--- Model Metrics ---")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"R² Score:  {r2:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved optimized XGBoost model to {MODEL_PATH}")

if __name__ == "__main__":
    train()