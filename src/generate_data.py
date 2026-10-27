
from pathlib import Path

import numpy as np
import pandas as pd

# Save the dataset in the project's data folder.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "cafe_sales.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

products = {
    "Espresso": (15.0, 4.0, 8.0),
    "Cappuccino": (25.0, 8.0, 7.0),
    "Latte": (28.0, 9.0, 6.0),
    "Croissant": (12.0, 5.0, 5.0),
    "Sandwich": (35.0, 16.0, 4.0),
}

hour_factors = {
    8: 0.8, 9: 1.1, 10: 1.25, 11: 1.15,
    12: 1.35, 13: 1.3, 14: 0.85, 15: 0.55,
    16: 0.5, 17: 0.65, 18: 0.8, 19: 1.0,
    20: 0.95, 21: 0.6,
}

rows = []

for date in pd.date_range("2025-01-01", periods=180):
    day = date.dayofweek
    weekend_factor = 1.18 if day >= 5 else 1.0

    for hour, hour_factor in hour_factors.items():
        for product, (base_price, cost, demand) in products.items():
            promotion = int(
                hour in (15, 16) and rng.random() < 0.28
            )

            price = round(
                base_price * rng.choice([0.90, 0.95, 1.0, 1.0, 1.05]),
                2,
            )

            promo_factor = 1.22 if promotion else 1.0
            price_factor = (price / base_price) ** -1.6

            expected_units = (
                demand
                * hour_factor
                * weekend_factor
                * promo_factor
                * price_factor
            )

            units_sold = int(rng.poisson(expected_units))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "hour": hour,
                "day_of_week": day,
                "product": product,
                "price_mad": price,
                "unit_cost_mad": cost,
                "promotion": promotion,
                "units_sold": units_sold,
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print(f"Dataset created: {OUTPUT}")
print(f"Total records: {len(df):,}")
print("\nFirst five rows:")
print(df.head().to_string(index=False))
