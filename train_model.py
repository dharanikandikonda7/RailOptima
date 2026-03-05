import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data/synthetic_railway_data.csv")

# ---------------------------
# Feature Selection
# ---------------------------
features = [
    "Weekend",
    "Holiday",
    "Peak_Hour",
    "Delay_Minutes",
    "Number_of_Coaches"
]

X = df[features]
y = df["Passenger_Count"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# ---------------------------
# Save Model
# ---------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/demand_model.pkl")

print("✅ Model saved to models/demand_model.pkl")