import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.pricing_optimizer import load_model

st.set_page_config(
    page_title="SmartPrice AI Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System
st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .css-1d38152 { background-color: #1E293B !important; }
    
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .hero-card h1 { color: #38BDF8 !important; font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
    .hero-card p { color: #94A3B8; font-size: 1.05rem; }

    .kpi-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-title { color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; margin-top: 0.2rem; }
    .text-sky { color: #38BDF8; }
    .text-emerald { color: #34D399; }
    .text-amber { color: #FBBF24; }
    </style>
""", unsafe_allow_html=True)

# Top Banner
st.markdown("""
    <div class="hero-card">
        <h1>⚡ SmartPrice AI — Enterprise Dynamic Pricing</h1>
        <p>AI-powered revenue optimization, demand forecasting, and profit maximization suite.</p>
    </div>
""", unsafe_allow_html=True)

model = load_model()

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Live Context Settings")
    
    products_catalog = {
        "Espresso": {"base_price": 15.0, "cost": 4.0},
        "Cappuccino": {"base_price": 25.0, "cost": 8.0},
        "Latte": {"base_price": 28.0, "cost": 9.0},
        "Croissant": {"base_price": 12.0, "cost": 5.0},
        "Sandwich": {"base_price": 35.0, "cost": 16.0},
    }

    product = st.selectbox("Product Focus", list(products_catalog.keys()))
    unit_cost = st.number_input("Unit Cost (MAD)", min_value=1.0, value=products_catalog[product]["cost"], step=0.5)
    baseline_price = st.number_input("Current Fixed Price (MAD)", min_value=1.0, value=products_catalog[product]["base_price"], step=0.5)

    st.markdown("---")
    hour = st.slider("Hour of Day", 8, 21, 12)
    day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    day_of_week = st.selectbox("Day of Week", options=list(day_mapping.keys()), format_func=lambda x: day_mapping[x])
    promotion = st.radio("Active Marketing Campaign?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Scenario Computations
price_range = np.linspace(unit_cost * 1.1, unit_cost * 4.0, 100)
is_weekend = 1 if day_of_week >= 5 else 0
is_peak_hour = 1 if hour in [8, 9, 12, 13, 19] else 0

results = []
for p in price_range:
    margin_ratio = (p - unit_cost) / p if p > 0 else 0
    features = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "product": product,
        "price_mad": p,
        "unit_cost_mad": unit_cost,
        "promotion": promotion,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "margin_ratio": margin_ratio
    }])
    pred_demand = max(0, float(model.predict(features)[0]))
    profit = (p - unit_cost) * pred_demand
    revenue = p * pred_demand
    results.append({"Price": p, "Demand": pred_demand, "Revenue": revenue, "Profit": profit})

df_results = pd.DataFrame(results)
best_row = df_results.loc[df_results["Profit"].idxmax()]

# Baseline prediction
baseline_margin = (baseline_price - unit_cost) / baseline_price if baseline_price > 0 else 0
b_features = pd.DataFrame([{
    "hour": hour, "day_of_week": day_of_week, "product": product,
    "price_mad": baseline_price, "unit_cost_mad": unit_cost,
    "promotion": promotion, "is_weekend": is_weekend,
    "is_peak_hour": is_peak_hour, "margin_ratio": baseline_margin
}])
b_demand = max(0, float(model.predict(b_features)[0]))
b_profit = (baseline_price - unit_cost) * b_demand

profit_uplift_pct = ((best_row["Profit"] - b_profit) / b_profit * 100) if b_profit > 0 else 0.0

# KPI Grid
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Optimal Price</div>
            <div class="kpi-val text-sky">{best_row['Price']:.2f} MAD</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Projected Demand</div>
            <div class="kpi-val">{best_row['Demand']:.1f} units</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Expected Profit</div>
            <div class="kpi-val text-emerald">{best_row['Profit']:.2f} MAD</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Dynamic Profit Uplift</div>
            <div class="kpi-val text-amber">+{profit_uplift_pct:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs
tab1, tab2 = st.tabs(["📈 Elasticity & Optimization Curves", "📋 Catalog Pricing Matrix"])

with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_results["Price"], y=df_results["Demand"],
        name="Demand Curve (Units)", mode='lines',
        line=dict(color='#38BDF8', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_results["Price"], y=df_results["Profit"],
        name="Profit Curve (MAD)", mode='lines',
        line=dict(color='#34D399', width=3),
        yaxis="y2"
    ))

    # Baseline vs Optimal markers
    fig.add_trace(go.Scatter(
        x=[best_row["Price"]], y=[best_row["Profit"]],
        mode='markers+text', name='Optimal Price',
        marker=dict(size=14, color='#F87171', symbol='star'),
        text=[f"Optimal: {best_row['Price']:.2f} MAD"],
        textposition="top center", yaxis="y2"
    ))

    fig.add_trace(go.Scatter(
        x=[baseline_price], y=[b_profit],
        mode='markers+text', name='Current Fixed Price',
        marker=dict(size=12, color='#FBBF24', symbol='circle'),
        text=[f"Current: {baseline_price:.2f} MAD"],
        textposition="bottom center", yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        height=480,
        xaxis=dict(title=dict(text="Tested Price (MAD)"), gridcolor='#334155'),
        yaxis=dict(title=dict(text="Predicted Demand", font=dict(color="#38BDF8")), gridcolor='#334155'),
        yaxis2=dict(title=dict(text="Expected Profit (MAD)", font=dict(color="#34D399")), overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Storewide Menu Recommendations")
    
    catalog_summary = []
    for item_name, meta in products_catalog.items():
        c_p_range = np.linspace(meta["cost"] * 1.1, meta["cost"] * 4.0, 50)
        best_p, best_d, max_prof = 0, 0, -float("inf")
        
        for p in c_p_range:
            m_ratio = (p - meta["cost"]) / p if p > 0 else 0
            f = pd.DataFrame([{
                "hour": hour, "day_of_week": day_of_week, "product": item_name,
                "price_mad": p, "unit_cost_mad": meta["cost"],
                "promotion": promotion, "is_weekend": is_weekend,
                "is_peak_hour": is_peak_hour, "margin_ratio": m_ratio
            }])
            d = max(0, float(model.predict(f)[0]))
            prof = (p - meta["cost"]) * d
            if prof > max_prof:
                max_prof, best_p, best_d = prof, p, d
                
        catalog_summary.append({
            "Product": item_name,
            "Unit Cost": f"{meta['cost']:.2f} MAD",
            "Current Base Price": f"{meta['base_price']:.2f} MAD",
            "Optimal Recommended Price": f"{best_p:.2f} MAD",
            "Est. Daily Sales": f"{best_d:.1f}",
            "Est. Net Profit": f"{max_prof:.2f} MAD"
        })
        
    st.dataframe(pd.DataFrame(catalog_summary), use_container_width=True)