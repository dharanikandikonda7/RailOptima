import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# -------------------------
# Configuration
# -------------------------
np.random.seed(42)

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start_date, end_date)

routes = [
    ("R1", "Hyderabad", "Bangalore", 1.3),
    ("R2", "Mumbai", "Delhi", 1.5),
    ("R3", "Chennai", "Kolkata", 1.2),
    ("R4", "Delhi", "Jaipur", 1.1),
    ("R5", "Pune", "Hyderabad", 1.25),
    ("R6", "Ahmedabad", "Mumbai", 1.15),
    ("R7", "Lucknow", "Delhi", 1.2),
    ("R8", "Bangalore", "Chennai", 1.35),
    ("R9", "Patna", "Kolkata", 1.1),
    ("R10", "Nagpur", "Mumbai", 1.05)
]

platforms = [1, 2, 3, 4, 5]

data = []

# -------------------------
# Generate Data
# -------------------------
for date in date_range:
    for route_id, source, destination, popularity in routes:

        base_demand = np.random.randint(400, 800)

        weekend = 1 if date.weekday() >= 5 else 0
        holiday = 1 if random.random() < 0.05 else 0

        peak_hour = 1 if random.random() < 0.4 else 0

        delay = np.random.randint(0, 60)

        demand = base_demand

        # Demand Logic
        if weekend:
            demand *= 1.2

        if holiday:
            demand *= 1.35

        if peak_hour:
            demand *= 1.25

        if delay > 40:
            demand *= 0.9

        demand *= popularity

        number_of_coaches = np.random.randint(10, 18)
        seat_capacity = number_of_coaches * 72

        occupancy_rate = (demand / seat_capacity) * 100

        data.append([
            f"TR{random.randint(100,999)}",
            route_id,
            source,
            destination,
            date.strftime("%Y-%m-%d"),
            f"{random.randint(5,23)}:{random.choice(['00','30'])}",
            int(demand),
            seat_capacity,
            round(occupancy_rate, 2),
            number_of_coaches,
            random.choice(platforms),
            delay,
            weekend,
            holiday,
            peak_hour
        ])

# -------------------------
# Create DataFrame
# -------------------------
columns = [
    "Train_ID",
    "Route",
    "Source",
    "Destination",
    "Date",
    "Departure_Time",
    "Passenger_Count",
    "Seat_Capacity",
    "Occupancy_Rate",
    "Number_of_Coaches",
    "Platform_Number",
    "Delay_Minutes",
    "Weekend",
    "Holiday",
    "Peak_Hour"
]

df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv("synthetic_railway_data.csv", index=False)

print("✅ Synthetic dataset generated successfully!")