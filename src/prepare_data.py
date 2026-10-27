from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cafe_sales.csv"

def prepare_data():
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Run 'python -m src.generate_data' first to generate synthetic data.")
        return None

    df = pd.read_csv(DATA_PATH)
    
    print("--- Dataset Information ---")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Data Sample (First 5 Rows) ---")
    print(df.head())

    print("\n--- Summary Statistics per Product ---")
    summary = df.groupby("product").agg(
        avg_price=("price_mad", "mean"),
        avg_cost=("unit_cost_mad", "mean"),
        avg_units_sold=("units_sold", "mean"),
        total_units_sold=("units_sold", "sum")
    ).reset_index()
    print(summary)

    return df

if __name__ == "__main__":
    prepare_data()