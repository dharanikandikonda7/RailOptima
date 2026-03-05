import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.cluster import KMeans
from utils import (
    calculate_occupancy,
    classify_risk,
    detect_platform_conflicts,
    recommend_actions,
    calculate_efficiency_score
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="RailOptima Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom Enterprise Styling
# --------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            font-size: 28px;
            font-weight: 600;
        }
        h2 {
            font-size: 20px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("RailOptima Smart Demand and Resource Planning Dashboard")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/synthetic_railway_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/demand_model.pkl")

model = load_model()

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("Filters")

route_selected = st.sidebar.selectbox("Select Route", df["Route"].unique())

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

scenario_boost = st.sidebar.slider(
    "Scenario Demand Increase (%)",
    0, 50, 0
)

filtered_df = df[
    (df["Route"] == route_selected) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# --------------------------------------------------
# KPI Section
# --------------------------------------------------
st.subheader("Operational Metrics")

col1, col2, col3, col4 = st.columns(4)

total_passengers = int(filtered_df["Passenger_Count"].sum())
avg_occupancy = round(filtered_df["Occupancy_Rate"].mean(), 2)
high_risk_count = (filtered_df["Occupancy_Rate"] > 90).sum()
platform_conflicts = detect_platform_conflicts(filtered_df)

col1.metric("Total Passengers", total_passengers)
col2.metric("Average Occupancy (%)", avg_occupancy)
col3.metric("High Risk Days", high_risk_count)
col4.metric("Platform Conflicts", platform_conflicts)

# --------------------------------------------------
# Demand Trend Visualization
# --------------------------------------------------
st.subheader("Passenger Demand Trend")

trend_fig = px.line(
    filtered_df,
    x="Date",
    y="Passenger_Count",
    title="Passenger Demand Over Time"
)

st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------------------------
# Advanced Feature: Route Clustering
# --------------------------------------------------
st.subheader("Route Performance Clustering")

cluster_df = df.groupby("Route").agg({
    "Passenger_Count": "mean",
    "Occupancy_Rate": "mean",
    "Delay_Minutes": "mean"
}).reset_index()

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df["Cluster"] = kmeans.fit_predict(
    cluster_df[["Passenger_Count", "Occupancy_Rate", "Delay_Minutes"]]
)

cluster_fig = px.scatter(
    cluster_df,
    x="Passenger_Count",
    y="Occupancy_Rate",
    color=cluster_df["Cluster"].astype(str),
    hover_name="Route",
    title="Route Segmentation Based on Demand and Occupancy"
)

st.plotly_chart(cluster_fig, use_container_width=True)

st.write("Cluster 0, 1, and 2 represent low, medium, and high demand route segments based on operational characteristics.")

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("Next 7-Day Demand Forecast")

latest = filtered_df.iloc[-1]

future_data = pd.DataFrame({
    "Weekend": [latest["Weekend"]] * 7,
    "Holiday": [0] * 7,
    "Peak_Hour": [1] * 7,
    "Delay_Minutes": [latest["Delay_Minutes"]] * 7,
    "Number_of_Coaches": [latest["Number_of_Coaches"]] * 7
})

predictions = model.predict(future_data)
predictions = predictions * (1 + scenario_boost / 100)

forecast_df = pd.DataFrame({
    "Day": [f"Day {i+1}" for i in range(7)],
    "Predicted_Passengers": predictions
})

st.dataframe(forecast_df, use_container_width=True)

# --------------------------------------------------
# Risk Analysis and Recommendations
# --------------------------------------------------
st.subheader("Operational Recommendations")

latest_capacity = latest["Seat_Capacity"]
predicted_avg = predictions.mean()

predicted_occupancy = calculate_occupancy(predicted_avg, latest_capacity)
risk_level = classify_risk(predicted_occupancy)

conflict_flag = platform_conflicts > 0
recommendations = recommend_actions(predicted_occupancy, conflict_flag)

st.write(f"Predicted Occupancy: {round(predicted_occupancy, 2)} percent")
st.write(f"Risk Level: {risk_level}")

for rec in recommendations:
    st.write(rec)

# --------------------------------------------------
# Efficiency Score
# --------------------------------------------------
st.subheader("Resource Efficiency Score")

overcrowding_penalty = max(0, predicted_occupancy - 85)
conflict_penalty = platform_conflicts * 5

efficiency_score = calculate_efficiency_score(
    predicted_occupancy,
    overcrowding_penalty,
    conflict_penalty
)

if efficiency_score >= 80:
    st.success(f"Efficiency Score: {round(efficiency_score,2)} (Excellent)")
elif efficiency_score >= 60:
    st.warning(f"Efficiency Score: {round(efficiency_score,2)} (Moderate)")
else:
    st.error(f"Efficiency Score: {round(efficiency_score,2)} (Poor)")