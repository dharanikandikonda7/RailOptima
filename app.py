import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
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
# Custom CSS Styling
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

h1{
font-size:36px;
font-weight:700;
text-align:center;
color:#1e293b;
}

.card{
background:white;
padding:20px;
border-radius:14px;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
transition:0.2s;
}

.card:hover{
transform:translateY(-4px);
box-shadow:0 8px 18px rgba(0,0,0,0.12);
}

.metric-card{
background:linear-gradient(135deg,#6366f1,#4f46e5);
color:white;
padding:18px;
border-radius:12px;
text-align:center;
font-size:18px;
font-weight:600;
}

.small-card{
background:white;
padding:15px;
border-radius:12px;
box-shadow:0 3px 8px rgba(0,0,0,0.07);
}

</style>
""", unsafe_allow_html=True)

st.title("🚆 RailOptima Smart Demand & Resource Planning")

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
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Filters")

route_selected = st.sidebar.selectbox(
    "Select Route",
    df["Route"].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

scenario_boost = st.sidebar.slider(
    "Demand Increase Scenario (%)",
    0,50,0
)

filtered_df = df[
(df["Route"]==route_selected) &
(df["Date"]>=pd.to_datetime(date_range[0])) &
(df["Date"]<=pd.to_datetime(date_range[1]))
]

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
st.subheader("📊 Operational Overview")

col1,col2,col3,col4 = st.columns(4)

total_passengers=int(filtered_df["Passenger_Count"].sum())
avg_occupancy=round(filtered_df["Occupancy_Rate"].mean(),2)
high_risk_count=(filtered_df["Occupancy_Rate"]>90).sum()
platform_conflicts=detect_platform_conflicts(filtered_df)

with col1:
    st.markdown(f"""
    <div class="metric-card">
    👥 Total Passengers<br><br>
    {total_passengers}
    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
    📈 Avg Occupancy<br><br>
    {avg_occupancy} %
    </div>
    """,unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
    ⚠️ High Risk Days<br><br>
    {high_risk_count}
    </div>
    """,unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
    🚉 Platform Conflicts<br><br>
    {platform_conflicts}
    </div>
    """,unsafe_allow_html=True)

# --------------------------------------------------
# DEMAND TREND CARD
# --------------------------------------------------
st.subheader("📈 Passenger Demand Trend")

trend_fig=px.line(
filtered_df,
x="Date",
y="Passenger_Count",
markers=True,
template="plotly_white"
)

trend_fig.update_layout(height=320)

st.plotly_chart(trend_fig,use_container_width=True)

# --------------------------------------------------
# SIDE BY SIDE CARDS
# --------------------------------------------------
st.subheader("📊 Route Insights")

col1,col2=st.columns(2)

# Bar Chart
with col1:

    route_summary=df.groupby("Route")["Passenger_Count"].mean().reset_index()

    route_fig=px.bar(
    route_summary,
    x="Route",
    y="Passenger_Count",
    color="Route",
    template="plotly_white",
    title="Average Demand by Route"
    )

    route_fig.update_layout(height=300)

    st.plotly_chart(route_fig,use_container_width=True)

# Pie Chart
with col2:

    occ_counts=filtered_df["Occupancy_Rate"].apply(
    lambda x:"High (>90%)" if x>90 else "Normal"
    ).value_counts().reset_index()

    occ_counts.columns=["Category","Count"]

    pie_fig=px.pie(
    occ_counts,
    names="Category",
    values="Count",
    hole=0.45,
    title="Occupancy Distribution"
    )

    pie_fig.update_layout(height=300)

    st.plotly_chart(pie_fig,use_container_width=True)

# --------------------------------------------------
# FORECAST
# --------------------------------------------------
st.subheader("🔮 7-Day Passenger Demand Forecast")

latest=filtered_df.iloc[-1]

future_data=pd.DataFrame({
"Weekend":[latest["Weekend"]]*7,
"Holiday":[0]*7,
"Peak_Hour":[1]*7,
"Delay_Minutes":[latest["Delay_Minutes"]]*7,
"Number_of_Coaches":[latest["Number_of_Coaches"]]*7
})

predictions=model.predict(future_data)
predictions=predictions*(1+scenario_boost/100)

forecast_df=pd.DataFrame({
"Day":[f"Day {i+1}" for i in range(7)],
"Predicted_Passengers":predictions
})

with st.expander("📅 View Forecast Table"):
    st.dataframe(forecast_df,use_container_width=True)

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------
st.subheader("🧠 Operational Recommendations")

latest_capacity=latest["Seat_Capacity"]
predicted_avg=predictions.mean()

predicted_occupancy=calculate_occupancy(predicted_avg,latest_capacity)
risk_level=classify_risk(predicted_occupancy)

conflict_flag=platform_conflicts>0
recommendations=recommend_actions(predicted_occupancy,conflict_flag)

st.info(f"Predicted Occupancy: {round(predicted_occupancy,2)} %")
st.warning(f"Risk Level: {risk_level}")

for rec in recommendations:
    st.markdown(f"✔️ {rec}")

# --------------------------------------------------
# EFFICIENCY SCORE
# --------------------------------------------------
st.subheader("⚡ Resource Efficiency Score")

overcrowding_penalty=max(0,predicted_occupancy-85)
conflict_penalty=platform_conflicts*5

efficiency_score=calculate_efficiency_score(
predicted_occupancy,
overcrowding_penalty,
conflict_penalty
)

st.progress(int(efficiency_score))

if efficiency_score>=80:
    st.success(f"Efficiency Score: {round(efficiency_score,2)} (Excellent)")
elif efficiency_score>=60:
    st.warning(f"Efficiency Score: {round(efficiency_score,2)} (Moderate)")
else:
    st.error(f"Efficiency Score: {round(efficiency_score,2)} (Poor)")