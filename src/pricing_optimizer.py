"""Vectorised pricing engine: demand curves, optimal prices, hourly schedules."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import CATALOG, HOURS, MODEL_PATH
from .features import build_features


@dataclass(frozen=True)
class Recommendation:
    price: float
    demand: float
    revenue: float
    profit: float
    low: float             # lowest price considered
    high: float            # highest price considered
    at_boundary: bool      # optimum sits on the edge of the searched range


class PricingEngine:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.reference_prices = bundle["reference_prices"]
        self.price_support = bundle["price_support"]
        self.metrics = bundle.get("metrics", {})
        self.meta = {k: v for k, v in bundle.items() if k != "pipeline"}

    @classmethod
    def load(cls, path=MODEL_PATH) -> "PricingEngine":
        return cls(joblib.load(path))

    # ---- core prediction -------------------------------------------------
    def predict_demand(self, product, prices, hour, day_of_week, promotion=0) -> np.ndarray:
        prices = np.atleast_1d(np.asarray(prices, dtype=float))
        raw = pd.DataFrame({
            "product": product,
            "hour": hour,
            "day_of_week": day_of_week,
            "price_mad": prices,
            "promotion": int(promotion),
        })
        X = build_features(raw, self.reference_prices)
        return np.clip(self.pipeline.predict(X), 0, None)

    # ---- price bounds ----------------------------------------------------
    def price_bounds(self, product, unit_cost, current_price=None,
                     min_margin_pct=10.0, max_change_pct=None):
        """Range we can trust: the prices seen in training, narrowed by business rules."""
        ref = self.reference_prices[product]
        lo_idx, hi_idx = self.price_support[product]
        low, high = lo_idx * ref, hi_idx * ref
        low = max(low, unit_cost * (1 + min_margin_pct / 100))
        if max_change_pct is not None and current_price:
            low = max(low, current_price * (1 - max_change_pct / 100))
            high = min(high, current_price * (1 + max_change_pct / 100))
        if low >= high:  # rules too strict for the data: fall back to a single point
            low = high = max(low, unit_cost * (1 + min_margin_pct / 100))
        return float(low), float(high)

    # ---- curves & optimisation ------------------------------------------
    def curve(self, product, unit_cost, hour, day_of_week, promotion=0,
              bounds=None, n=80) -> pd.DataFrame:
        low, high = bounds or self.price_bounds(product, unit_cost)
        prices = np.linspace(low, high, n) if high > low else np.array([low])
        demand = self.predict_demand(product, prices, hour, day_of_week, promotion)
        return pd.DataFrame({
            "Price": prices,
            "Demand": demand,
            "Revenue": prices * demand,
            "Profit": (prices - unit_cost) * demand,
        })

    def optimize(self, product, unit_cost, hour, day_of_week, promotion=0,
                 bounds=None, n=80) -> Recommendation:
        bounds = bounds or self.price_bounds(product, unit_cost)
        c = self.curve(product, unit_cost, hour, day_of_week, promotion, bounds, n)
        best = c.loc[c["Profit"].idxmax()]
        span = bounds[1] - bounds[0]
        at_edge = span > 0 and (
            abs(best["Price"] - bounds[0]) < 0.02 * span
            or abs(best["Price"] - bounds[1]) < 0.02 * span
        )
        return Recommendation(float(best["Price"]), float(best["Demand"]),
                              float(best["Revenue"]), float(best["Profit"]),
                              bounds[0], bounds[1], bool(at_edge))

    def evaluate_price(self, product, price, unit_cost, hour, day_of_week, promotion=0):
        d = float(self.predict_demand(product, price, hour, day_of_week, promotion)[0])
        return {"demand": d, "revenue": price * d, "profit": (price - unit_cost) * d}

    def elasticity(self, product, price, hour, day_of_week, promotion=0, step=0.10) -> float:
        """Approximate % change in demand per 1% change in price around `price`."""
        lo, hi = price * (1 - step), price * (1 + step)
        q = self.predict_demand(product, [lo, hi], hour, day_of_week, promotion)
        if q.min() <= 0:
            return float("nan")
        return float(np.log(q[1] / q[0]) / np.log(hi / lo))

    # ---- higher-level views ----------------------------------------------
    def hourly_schedule(self, product, unit_cost, current_price, day_of_week,
                        min_margin_pct=10.0, max_change_pct=None) -> pd.DataFrame:
        rows = []
        bounds = self.price_bounds(product, unit_cost, current_price,
                                   min_margin_pct, max_change_pct)
        for h in HOURS:
            promo = 0
            rec = self.optimize(product, unit_cost, h, day_of_week, promo, bounds, n=60)
            base = self.evaluate_price(product, current_price, unit_cost, h, day_of_week, promo)
            rows.append({
                "Hour": h,
                "Current Price": current_price,
                "Optimal Price": rec.price,
                "Current Profit": base["profit"],
                "Optimal Profit": rec.profit,
                "Uplift %": _pct(rec.profit, base["profit"]),
            })
        return pd.DataFrame(rows)

    def menu(self, hour, day_of_week, promotion=0, min_margin_pct=10.0,
             costs: dict | None = None) -> pd.DataFrame:
        rows = []
        for name, p in CATALOG.items():
            cost = (costs or {}).get(name, p.unit_cost)
            bounds = self.price_bounds(name, cost, min_margin_pct=min_margin_pct)
            rec = self.optimize(name, cost, hour, day_of_week, promotion, bounds, n=50)
            base = self.evaluate_price(name, p.base_price, cost, hour, day_of_week, promotion)
            rows.append({
                "Product": name,
                "Unit Cost": cost,
                "Current Price": p.base_price,
                "Recommended Price": rec.price,
                "Change %": _pct(rec.price, p.base_price),
                "Est. Units / h": rec.demand,
                "Est. Profit / h": rec.profit,
                "Profit Uplift %": _pct(rec.profit, base["profit"]),
            })
        return pd.DataFrame(rows)


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100 if old and old > 0 else 0.0


@lru_cache(maxsize=1)
def get_engine() -> PricingEngine:
    """Load the model once per process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m src.train_model")
    return PricingEngine.load()


def optimize_price(product_name, unit_cost, price_range, hour=12, day_of_week=2, promotion=0):
    """Backward-compatible helper: best price from an explicit list of candidates."""
    engine = get_engine()
    prices = np.asarray(list(price_range), dtype=float)
    demand = engine.predict_demand(product_name, prices, hour, day_of_week, promotion)
    profit = (prices - unit_cost) * demand
    i = int(np.argmax(profit))
    return float(prices[i]), float(demand[i]), float(profit[i])


if __name__ == "__main__":
    e = get_engine()
    r = e.optimize("Espresso", 4.0, hour=12, day_of_week=2)
    print(f"Optimal: {r.price:.2f} MAD | demand {r.demand:.1f} | profit {r.profit:.2f} MAD")