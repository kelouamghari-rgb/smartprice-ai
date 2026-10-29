# ⚡ SmartPrice AI

Dynamic pricing for a cafe: predicts demand at any price and recommends the price that maximises profit, hour by hour.

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data     # create data/cafe_sales.csv
python -m src.train_model       # train and save the model
python -m unittest discover -s tests -t .   # run tests
streamlit run app.py            # open the dashboard
```
The app also builds the data and model automatically on first launch if they are missing.

## How it works
| Step | File | What it does |
|---|---|---|
| Config | `src/config.py` | Paths, product catalog, business constants |
| Data | `src/generate_data.py` | Synthetic sales with realistic price sensitivity (peak hours less elastic) |
| Features | `src/features.py` | One shared feature pipeline for training and prediction |
| Model | `src/train_model.py` | Poisson gradient boosting, monotonic in price and promotion, time-based evaluation |
| Engine | `src/pricing_optimizer.py` | Vectorised demand curves, optimiser, guardrails, hourly schedule, elasticity |
| UI | `app.py` | Optimizer, daily schedule, menu matrix, model health |

Design choices
- **Counts use Poisson loss.** Units sold are counts, so the model uses Poisson loss.
- **Common-sense constraints.** Higher price never increases predicted demand; a promotion never decreases it.
- **No look-ahead.** Evaluation trains on the past and tests on the most recent 20% of days, and is compared to a naive baseline.
- **Stay inside the data.** Tree models can't extrapolate, so prices are only searched within the range seen in training, then narrowed by margin and max-change guardrails.
- **Cost is not a feature.** Customers react to price, not to your cost.

## Using your own data
Replace `data/cafe_sales.csv` (columns: `date, hour, day_of_week, product, price_mad, unit_cost_mad, promotion, units_sold`), update `CATALOG` in `src/config.py`, then retrain. Real prices need enough variation per product, otherwise the optimiser cannot learn price sensitivity.

## Known limits
- The bundled data is synthetic; results show the method, not real market behaviour.
- Promotions only occur at 15:00–17:00 in the data, so promotion effects elsewhere are extrapolated (the app warns you).
- Profit curves from tree models are slightly stepped; treat the recommended price as a good region, not an exact point.