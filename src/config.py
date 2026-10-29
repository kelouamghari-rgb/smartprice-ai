"""Single source of truth for paths, the product catalog and business constants."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cafe_sales.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


@dataclass(frozen=True)
class Product:
    base_price: float   # current menu price (MAD) - also the reference for price_index
    unit_cost: float    # cost of goods (MAD)
    base_demand: float  # average units per hour in the synthetic data


CATALOG = {
    "Espresso":   Product(15.0, 4.0, 8.0),
    "Cappuccino": Product(25.0, 8.0, 7.0),
    "Latte":      Product(28.0, 9.0, 6.0),
    "Croissant":  Product(12.0, 5.0, 5.0),
    "Sandwich":   Product(35.0, 16.0, 4.0),
}

HOURS = tuple(range(8, 22))
PEAK_HOURS = (8, 9, 12, 13, 19)
PROMO_HOURS = (15, 16)  # promotions only ever ran at these hours in the data

# Prices tested in the historical data, as a multiple of the menu price.
PRICE_GRID = tuple(round(0.60 + 0.05 * i, 2) for i in range(19))  # 0.60 .. 1.50

# Price sensitivity used by the synthetic generator (peak hours = less sensitive).
ELASTICITY = {"peak": 1.4, "offpeak": 1.9}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]