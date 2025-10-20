import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Power Consumption Forecast",
    page_icon="⚡",
    layout="wide"
)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("daily_state_usage.csv", parse_dates=["Dates"])
    return df

daily_state_usage = load_data()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Adjust the options below to customize your forecast:")

state = st.sidebar.selectbox("Select a State", sorted(daily_state_usage["States"].unique()))
forecast_years = st.sidebar.slider("Select Forecast Duration (Years)", 1, 10, 4)
show_trend = st.sidebar.checkbox("Show Trend Line", True)
show_metrics = st.sidebar.checkbox("Show Summary Metrics", True)

st.sidebar.markdown("---")
st.sidebar.markdown("📅 **Filter Historical Range**")
min_date = daily_state_usage["Dates"].min()
max_date = daily_state_usage["Dates"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# ==============================
# MAIN CONTENT
# ==============================
st.title("⚡ Power Consumption Forecast Dashboard")
st.markdown("### Explore state-wise power consumption trends and forecast for the next few years")

state_df = daily_state_usage[daily_state_usage["States"] == state]
state_df = state_df[(state_df["Dates"] >= pd.to_datetime(date_range[0])) &
                    (state_df["Dates"] <= pd.to_datetime(date_range[1]))]

state_df = state_df.set_index("Dates").resample("M")["Usage"].mean().fillna(method="ffill")

# ==============================
# FORECAST MODEL
# ==============================
model = ExponentialSmoothing(state_df, trend="add", seasonal=None)
fit = model.fit()
future_dates = pd.date_range(state_df.index[-1] + pd.offsets.MonthEnd(1), periods=forecast_years * 12, freq="M")
forecast = pd.Series(fit.forecast(len(future_dates)), index=future_dates)

# ==============================
# PLOTS
# ==============================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(state_df.index, state_df.values, label="Historical", color="tab:blue", linewidth=2)
ax.plot(forecast.index, forecast.values, label="Forecast", color="tab:orange", linestyle="--", linewidth=2)

if show_trend:
    trend = state_df.rolling(6).mean()
    ax.plot(trend.index, trend.values, color="tab:green", linestyle=":", label="Trend (6-month avg)")

ax.set_title(f"📈 Consumption Trend for {state}", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Usage (GWh)")
ax.legend()
st.pyplot(fig)

# ==============================
# SUMMARY METRICS
# ==============================
if show_metrics:
    avg_usage = state_df.mean()
    growth_rate = ((forecast[-1] - state_df.iloc[-1]) / state_df.iloc[-1]) * 100

    st.markdown("### 📊 Summary Metrics")
    st.markdown(f"- **Average Usage:** {avg_usage:.2f} GWh")
    st.markdown(f"- **Forecasted Growth (next {forecast_years} years):** ~{growth_rate:.2f}%")

# ==============================
# DOWNLOAD FORECAST BUTTON
# ==============================
forecast_df = pd.DataFrame({
    "Date": forecast.index,
    "Forecasted Usage (GWh)": forecast.values
})

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forecast Data as CSV",
    data=csv,
    file_name=f"{state}_forecast_{forecast_years}yrs.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Developed as part of the Power Consumption Data Science Project 🔬")
