from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "demand_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def optimize_price(product_name, unit_cost, price_range, hour=12, day_of_week=2, promotion=0):
    model = load_model()
    best_price = None
    max_profit = -float("inf")
    best_demand = 0

    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if hour in [8, 9, 12, 13, 19] else 0

    results = []
    for price in price_range:
        margin_ratio = (price - unit_cost) / price if price > 0 else 0
        features = pd.DataFrame([{
            "hour": hour,
            "day_of_week": day_of_week,
            "product": product_name,
            "price_mad": price,
            "unit_cost_mad": unit_cost,
            "promotion": promotion,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "margin_ratio": margin_ratio
        }])
        
        predicted_demand = max(0, float(model.predict(features)[0]))
        profit = (price - unit_cost) * predicted_demand
        
        if profit > max_profit:
            max_profit = profit
            best_price = price
            best_demand = predicted_demand

    return best_price, best_demand, max_profit

if __name__ == "__main__":
    prices_to_test = [i * 0.5 for i in range(20, 100)]
    best_p, demand, profit = optimize_price("Espresso", unit_cost=4.0, price_range=prices_to_test)
    print(f"Optimal Price: {best_p} MAD | Predicted Sales: {demand:.1f} | Expected Profit: {profit:.2f} MAD")