# RailOptima – Smart Demand and Resource Planning Dashboard

## Project Overview

RailOptima is a data-driven railway planning assistant designed to improve operational efficiency using predictive analytics and intelligent recommendations.

The system predicts passenger demand, detects congestion risks, identifies platform conflicts, recommends coach adjustments, and calculates a resource efficiency score through an interactive Streamlit dashboard.

This project demonstrates the use of machine learning, data processing, and decision logic to support real-world railway planning operations.

---

## Problem Statement

Railway systems frequently face challenges such as overcrowding, inefficient coach allocation, platform scheduling conflicts, and unpredictable peak demand.

Traditional planning approaches rely heavily on static schedules and historical averages, which often fail to adapt to dynamic demand patterns.

RailOptima addresses this problem by combining predictive modeling with rule-based recommendations to support smarter operational decisions.

---

## Objectives

- Predict passenger demand using machine learning
- Detect high congestion risk routes
- Recommend coach allocation adjustments
- Identify platform scheduling conflicts
- Provide peak-period alerts
- Simulate demand scenarios using what-if analysis
- Calculate a resource efficiency score

---

## System Architecture

### Data Layer

- Synthetic dataset covering 12 months
- Ten major railway routes
- Daily train records
- Realistic demand variation logic

### Processing Layer

Feature engineering includes:

- Weekend indicator
- Holiday indicator
- Peak hour indicator
- Delay influence
- Route popularity multiplier

### Machine Learning Layer

Model used: RandomForestRegressor

The model predicts passenger demand based on operational and temporal features.

**Features used for training:**

- Weekend  
- Holiday  
- Peak_Hour  
- Delay_Minutes  
- Number_of_Coaches  

### Recommendation Engine

The system uses a hybrid logic approach:

- If predicted occupancy > 85% → Recommend adding 1 coach  
- If predicted occupancy > 100% → Recommend adding special train service  
- If platform conflict detected within 20 minutes → Suggest platform reassignment  
- If occupancy < 50% → Recommend reducing one coach  

### Efficiency Score Formula

Efficiency Score = Seat Utilization Percentage  
minus Overcrowding Penalty  
minus Platform Conflict Penalty  

Score is scaled between 0 and 100.

---

## Dataset Logic

The synthetic dataset simulates realistic railway demand patterns using the following assumptions:

- Weekends increase demand by 20 percent  
- Holidays increase demand by 35 percent  
- Peak hours increase demand by 25 percent  
- Delays above 40 minutes reduce demand by 10 percent  
- Each route has a popularity multiplier  

Total records generated: Approximately 3650

---

## Project Structure
RailOptima/
│
├── app.py
├── train_model.py
├── utils.py
│
├── data/
│ ├── generate_dataset.py
│ └── synthetic_railway_data.csv
│
├── models/
│ └── demand_model.pkl
│
├── requirements.txt
└── README.md

---

## Installation and Setup

Step 1: Clone the repository
git clone <repository_url>
cd RailOptima

Step 2: Install dependencies
- pip install -r requirements.txt

Step 3: Generate dataset
- python data/generate_dataset.py


Step 4: Train model
- python train_model.py


Step 5: Run dashboard
- streamlit run app.py


---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Plotly  
- Joblib  

---

## Key Features

- Machine learning-based passenger demand prediction  
- Congestion risk classification  
- Platform conflict detection  
- Smart operational recommendations  
- Scenario simulation mode  
- Resource efficiency scoring  
- Interactive visualization dashboard  

---

## Future Improvements

- Route clustering using K-Means  
- Advanced time-series forecasting  
- Delay prediction model  
- Automatic PDF report generation  
- Deployment on Streamlit Cloud  
- Integration with real railway datasets  

---

## Conclusion

RailOptima demonstrates how machine learning and intelligent rule-based systems can enhance railway demand planning, reduce overcrowding, and improve overall operational efficiency.

The project combines predictive modeling, operational logic, and interactive visualization to deliver a practical decision-support system for railway management.