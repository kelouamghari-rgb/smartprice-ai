import unittest
from src.pricing_optimizer import optimize_price

class TestPricingOptimizer(unittest.TestCase):
    def test_optimization_output(self):
        prices_to_test = [10.0, 15.0, 20.0, 25.0, 30.0]
        best_price, demand, profit = optimize_price(
            product_name="Espresso",
            unit_cost=4.0,
            price_range=prices_to_test
        )
        
        # Vérifications de base
        self.assertIsNotNone(best_price)
        self.assertIn(best_price, prices_to_test)
        self.assertGreaterEqual(demand, 0)

if __name__ == "__main__":
    unittest.main()