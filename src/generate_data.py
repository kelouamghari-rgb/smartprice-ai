"""Generate the synthetic cafe sales dataset.  Run: python -m src.generate_data"""
import numpy as np
import pandas as pd

from .config import CATALOG, DATA_PATH, ELASTICITY, PEAK_HOURS, PRICE_GRID, PROMO_HOURS

HOUR_FACTORS = {
    8: 0.8, 9: 1.1, 10: 1.25, 11: 1.15, 12: 1.35, 13: 1.3, 14: 0.85,
    15: 0.55, 16: 0.5, 17: 0.65, 18: 0.8, 19: 1.0, 20: 0.95, 21: 0.6,
}


def generate(seed: int = 42, start: str = "2025-01-01", days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range(start, periods=days):
        day = date.dayofweek
        weekend_factor = 1.18 if day >= 5 else 1.0
        for hour, hour_factor in HOUR_FACTORS.items():
            elasticity = ELASTICITY["peak"] if hour in PEAK_HOURS else ELASTICITY["offpeak"]
            for name, p in CATALOG.items():
                promotion = int(hour in PROMO_HOURS and rng.random() < 0.28)
                price = round(p.base_price * rng.choice(PRICE_GRID), 2)
                expected = (
                    p.base_demand * hour_factor * weekend_factor
                    * (1.22 if promotion else 1.0)
                    * (price / p.base_price) ** -elasticity
                )
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "day_of_week": day,
                    "product": name,
                    "price_mad": price,
                    "unit_cost_mad": p.unit_cost,
                    "promotion": promotion,
                    "units_sold": int(rng.poisson(expected)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset created: {DATA_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()