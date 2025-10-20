import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power Consumption Forecast", layout="wide")

st.title("⚡ Power Consumption Forecast Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("daily_state_usage.csv")
    df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Dates"])
    return df

df = load_data()

# --- User Inputs ---
states = sorted(df["States"].unique())
state_choice = st.selectbox("Select a State", states)
forecast_years = st.slider("Select Forecast Duration (Years)", 1, 5, 2)

# --- Filter and Display ---
state_df = df[df["States"] == state_choice]

st.subheader(f"📈 Consumption Trend for {state_choice}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(state_df["Dates"], state_df["Usage"], label="Historical", color="tab:blue")

# Mock simple forecast (extend last value slightly upward)
import numpy as np
future_dates = pd.date_range(state_df["Dates"].max(), periods=forecast_years * 12, freq="ME")
forecast = state_df["Usage"].iloc[-1] * (1.01 ** (np.arange(len(future_dates)) / 12))
ax.plot(future_dates, forecast, label="Forecast", linestyle="--", color="tab:orange")

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Usage (GWh)")
st.pyplot(fig)

# --- Metrics Display ---
st.subheader("📊 Summary Metrics")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
fit = model.fit()
future_index = pd.date_range(state_df["Dates"].max(), periods=forecast_years*12, freq="M")
forecast = fit.forecast(len(future_index))
future_dates = pd.date_range(
    state_df["Dates"].max() + pd.offsets.MonthEnd(1),
    periods=forecast_years * 12,
    freq="M"
)
col1, col2 = st.columns(2)
col1.metric("Average Usage (GWh)", f"{state_df['Usage'].mean():.2f}")
growth = (forecast.iloc[-1] / state_df['Usage'].iloc[-1] - 1) * 100
col2.metric(f"Forecasted Growth ({forecast_years} yrs)", f"{growth:.2f}%")

st.markdown("---")
with st.expander("📋 Show Raw Data"):
    st.dataframe(state_df)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(state_df["Usage"], trend="add", seasonal=None)
fit = model.fit()
forecast = fit.forecast(forecast_years * 12)
future_dates = pd.date_range(
    state_df["Dates"].max() + pd.offsets.MonthEnd(1),
    periods=len(forecast),
    freq="M"
)
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Usage": forecast})
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Forecast Data", csv, "forecast.csv", "text/csv")
