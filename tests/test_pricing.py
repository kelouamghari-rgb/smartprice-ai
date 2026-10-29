import unittest

import numpy as np
import pandas as pd

from src.config import CATALOG
from src.features import FEATURE_COLS, build_features
from src.pricing_optimizer import get_engine, optimize_price

REFS = {n: p.base_price for n, p in CATALOG.items()}


class TestFeatures(unittest.TestCase):
    def test_columns_and_cost_not_used(self):
        df = pd.DataFrame({"product": ["Latte"], "hour": [12], "day_of_week": [6],
                           "price_mad": [28.0], "promotion": [0]})
        X = build_features(df, REFS)
        self.assertEqual(list(X.columns), FEATURE_COLS)
        self.assertEqual(X.loc[0, "price_index"], 1.0)
        self.assertEqual(X.loc[0, "is_weekend"], 1)
        self.assertNotIn("unit_cost_mad", X.columns)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame({"product": ["Tea"], "hour": [9], "day_of_week": [1],
                                         "price_mad": [5.0], "promotion": [0]}), REFS)
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame({"product": ["Latte"]}), REFS)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()

    def test_legacy_helper(self):
        prices = [10.0, 15.0, 20.0, 25.0, 30.0]
        best, demand, profit = optimize_price("Espresso", 4.0, prices)
        self.assertIn(best, prices)
        self.assertGreaterEqual(demand, 0)

    def test_demand_decreases_with_price(self):
        for product, p in CATALOG.items():
            prices = np.linspace(0.7, 1.4, 12) * p.base_price
            q = self.engine.predict_demand(product, prices, 12, 2)
            self.assertTrue(np.all(np.diff(q) <= 1e-9), product)

    def test_promotion_never_lowers_demand(self):
        a = self.engine.predict_demand("Latte", 28.0, 15, 2, 0)[0]
        b = self.engine.predict_demand("Latte", 28.0, 15, 2, 1)[0]
        self.assertGreaterEqual(b, a)

    def test_optimum_respects_bounds_and_margin(self):
        for product, p in CATALOG.items():
            b = self.engine.price_bounds(product, p.unit_cost, min_margin_pct=20)
            r = self.engine.optimize(product, p.unit_cost, 12, 2, bounds=b)
            self.assertGreaterEqual(r.price, b[0] - 1e-9)
            self.assertLessEqual(r.price, b[1] + 1e-9)
            self.assertGreaterEqual(r.price, p.unit_cost * 1.2 - 1e-9)

    def test_max_change_guardrail(self):
        b = self.engine.price_bounds("Espresso", 4.0, current_price=15.0, max_change_pct=10)
        self.assertGreaterEqual(b[0], 13.5 - 1e-9)
        self.assertLessEqual(b[1], 16.5 + 1e-9)

    def test_optimum_beats_or_matches_current(self):
        for product, p in CATALOG.items():
            r = self.engine.optimize(product, p.unit_cost, 12, 2)
            base = self.engine.evaluate_price(product, p.base_price, p.unit_cost, 12, 2)
            self.assertGreaterEqual(r.profit, base["profit"] - 1e-6)

    def test_elasticity_is_negative(self):
        self.assertLess(self.engine.elasticity("Cappuccino", 25.0, 12, 2), 0)

    def test_schedule_and_menu_shapes(self):
        s = self.engine.hourly_schedule("Espresso", 4.0, 15.0, 2)
        self.assertEqual(len(s), 14)
        m = self.engine.menu(12, 2)
        self.assertEqual(set(m["Product"]), set(CATALOG))


if __name__ == "__main__":
    unittest.main()