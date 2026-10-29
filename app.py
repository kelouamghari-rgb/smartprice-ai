import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CATALOG, DAY_NAMES, DATA_PATH, HOURS, MODEL_PATH, PROMO_HOURS
from src.pricing_optimizer import PricingEngine

st.set_page_config(page_title="SmartPrice AI", page_icon="⚡", layout="wide")

SKY, GREEN, AMBER, RED, SLATE = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#94A3B8"

st.markdown("""
<style>
.hero { background: linear-gradient(135deg,#1E293B,#0F172A); border:1px solid #334155;
        padding:1.4rem 1.8rem; border-radius:16px; margin-bottom:1.2rem; }
.hero h1 { color:#38BDF8; font-size:1.9rem; font-weight:800; margin:0 0 .2rem 0; padding:0; }
.hero p { color:#94A3B8; margin:0; }
.kpi { background:#1E293B; border:1px solid #334155; border-radius:12px; padding:1rem; text-align:center; }
.kpi .t { color:#94A3B8; font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.kpi .v { font-size:1.6rem; font-weight:800; margin-top:.15rem; }
.kpi .s { color:#94A3B8; font-size:.8rem; margin-top:.1rem; }
</style>
""", unsafe_allow_html=True)


def show(obj, **kw):
    """Render a chart/table full-width across Streamlit versions."""
    fn = st.plotly_chart if isinstance(obj, go.Figure) else st.dataframe
    try:
        fn(obj, width="stretch", **kw)
    except Exception:  # older Streamlit without width="stretch"
        fn(obj, use_container_width=True, **kw)


def kpi(col, title, value, sub="", color="#F8FAFC"):
    col.markdown(
        f'<div class="kpi"><div class="t">{title}</div>'
        f'<div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the AI model (first run only)…")
def load_engine() -> PricingEngine:
    if not DATA_PATH.exists():
        from src.generate_data import main as make_data
        make_data()
    if not MODEL_PATH.exists():
        from src.train_model import train
        train()
    return PricingEngine.load()


def style_fig(fig, height=440):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                      height=height, margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


@st.cache_data(show_spinner=False)
def cached_menu(hour, dow, promo, margin):
    return load_engine().menu(hour, dow, promo, margin)


@st.cache_data(show_spinner=False)
def cached_schedule(product, cost, price, dow, margin, max_change):
    return load_engine().hourly_schedule(product, cost, price, dow, margin, max_change)


engine = load_engine()

st.markdown('<div class="hero"><h1>⚡ SmartPrice AI</h1>'
            '<p>Demand forecasting and profit-maximising prices for your cafe.</p></div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎯 Scenario")
    product = st.selectbox("Product", list(CATALOG))
    meta = CATALOG[product]
    unit_cost = st.number_input("Unit cost (MAD)", min_value=0.5, value=meta.unit_cost, step=0.5)
    current_price = st.number_input("Current price (MAD)", min_value=1.0, value=meta.base_price, step=0.5)
    hour = st.slider("Hour of day", min(HOURS), max(HOURS), 12)
    day = st.selectbox("Day", range(7), index=2, format_func=lambda i: DAY_NAMES[i])
    promotion = int(st.toggle("Promotion running", value=False))

    st.markdown("### 🛡️ Guardrails")
    min_margin = st.slider("Minimum margin over cost (%)", 0, 100, 30, step=5)
    limit_change = st.toggle("Limit change vs current price", value=False)
    max_change = st.slider("Max change (%)", 5, 50, 15, step=5) if limit_change else None

bounds = engine.price_bounds(product, unit_cost, current_price, min_margin, max_change)
rec = engine.optimize(product, unit_cost, hour, day, promotion, bounds)
base = engine.evaluate_price(product, current_price, unit_cost, hour, day, promotion)
uplift = (rec.profit - base["profit"]) / base["profit"] * 100 if base["profit"] > 0 else 0.0
elasticity = engine.elasticity(product, rec.price, hour, day, promotion)

# ---------------------------------------------------------------- warnings
if promotion and hour not in PROMO_HOURS:
    st.warning(f"Promotions only ran between {PROMO_HOURS[0]}:00 and {PROMO_HOURS[-1] + 1}:00 in the "
               "historical data, so the promotion effect at this hour is an extrapolation.")
if not (bounds[0] <= current_price <= bounds[1]):
    st.info("Your current price is outside the range the model was trained on or your guardrails; "
            "its profit estimate is less reliable.")
if rec.at_boundary:
    st.info("The best price sits at the edge of the allowed range. Loosen the guardrails or collect "
            "data at more price points to see if a better price exists.")

# ---------------------------------------------------------------- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
change = (rec.price - current_price) / current_price * 100
kpi(c1, "Recommended price", f"{rec.price:.2f} MAD", f"{change:+.1f}% vs current", SKY)
kpi(c2, "Projected demand", f"{rec.demand:.1f}", "units this hour")
kpi(c3, "Expected profit", f"{rec.profit:.0f} MAD", f"vs {base['profit']:.0f} today", GREEN)
kpi(c4, "Profit uplift", f"{uplift:+.1f}%", "vs current price", GREEN if uplift >= 0 else RED)
kpi(c5, "Price elasticity", f"{elasticity:.2f}" if np.isfinite(elasticity) else "n/a",
    "elastic" if np.isfinite(elasticity) and elasticity < -1 else "inelastic", AMBER)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Optimizer", "🕒 Daily schedule", "📋 Menu matrix", "🩺 Model health"])

# ---------------------------------------------------------------- tab 1
with tab1:
    curve = engine.curve(product, unit_cost, hour, day, promotion, bounds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["Price"], y=curve["Demand"], name="Demand (units)",
                             line=dict(color=SKY, width=3)))
    fig.add_trace(go.Scatter(x=curve["Price"], y=curve["Profit"], name="Profit (MAD)",
                             line=dict(color=GREEN, width=3), yaxis="y2"))
    fig.add_trace(go.Scatter(x=[rec.price], y=[rec.profit], mode="markers", name="Recommended",
                             marker=dict(size=15, color=RED, symbol="star"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=[current_price], y=[base["profit"]], mode="markers", name="Current",
                             marker=dict(size=12, color=AMBER), yaxis="y2"))
    style_fig(fig)
    fig.update_layout(xaxis_title="Price (MAD)", yaxis=dict(title="Demand", gridcolor="#334155"),
                      yaxis2=dict(title="Profit (MAD)", overlaying="y", side="right"))
    show(fig)
    st.caption(f"Searched {bounds[0]:.2f}–{bounds[1]:.2f} MAD: the price range seen in historical data, "
               "narrowed by your guardrails. The model is not trusted outside it.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown(f"#### Hour-by-hour pricing for **{product}** on {DAY_NAMES[day]}")
    sched = cached_schedule(product, unit_cost, current_price, day, min_margin, max_change)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sched["Hour"], y=sched["Optimal Profit"], name="Optimal profit",
                         marker_color=GREEN, opacity=.55, yaxis="y2"))
    fig.add_trace(go.Scatter(x=sched["Hour"], y=sched["Optimal Price"], name="Optimal price",
                             mode="lines+markers", line=dict(color=SKY, width=3)))
    fig.add_trace(go.Scatter(x=sched["Hour"], y=sched["Current Price"], name="Current price",
                             line=dict(color=AMBER, dash="dash")))
    style_fig(fig, 400)
    fig.update_layout(xaxis_title="Hour", yaxis=dict(title="Price (MAD)", gridcolor="#334155"),
                      yaxis2=dict(title="Profit (MAD)", overlaying="y", side="right", showgrid=False))
    show(fig)
    total_c, total_o = sched["Current Profit"].sum(), sched["Optimal Profit"].sum()
    st.metric("Estimated daily profit uplift", f"{total_o - total_c:+.0f} MAD",
              f"{(total_o - total_c) / total_c * 100:+.1f}%" if total_c > 0 else None)
    show(sched.round(2).set_index("Hour"))

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown(f"#### Whole-menu recommendations — {DAY_NAMES[day]}, {hour}:00")
    menu = cached_menu(hour, day, promotion, min_margin)
    show(menu.round(2).set_index("Product"))
    st.download_button("⬇️ Download CSV", menu.round(2).to_csv(index=False),
                       file_name="smartprice_menu.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 4
with tab4:
    m = engine.metrics
    a, b, c, d = st.columns(4)
    a.metric("R² (future data)", f"{m.get('r2', 0):.2f}")
    b.metric("MAE (units)", f"{m.get('mae', 0):.2f}", f"naive baseline {m.get('baseline_mae', 0):.2f}",
             delta_color="off")
    c.metric("Training rows", f"{engine.meta.get('rows', 0):,}")
    d.metric("Trained", str(engine.meta.get("trained_at", "?"))[:10])
    st.caption("Metrics come from a time-based split: the model is trained on the first 80% of days "
               "and tested on the most recent 20%.")
    st.markdown("#### Price coverage in training data")
    cov = pd.DataFrame([{"Product": p, "Lowest price": lo * engine.reference_prices[p],
                         "Highest price": hi * engine.reference_prices[p]}
                        for p, (lo, hi) in engine.price_support.items()])
    show(cov.round(2).set_index("Product"))