import pandas as pd

# ----------------------------
# Calculate Occupancy
# ----------------------------
def calculate_occupancy(passengers, capacity):
    if capacity == 0:
        return 0
    return (passengers / capacity) * 100


# ----------------------------
# Risk Classification
# ----------------------------
def classify_risk(occupancy):
    if occupancy < 70:
        return "Low"
    elif occupancy < 90:
        return "Medium"
    else:
        return "High"


# ----------------------------
# Platform Conflict Detection
# Rule: Same platform within 20 minutes
# ----------------------------
def detect_platform_conflicts(df):

    if df.empty:
        return 0

    df = df.copy()

    # Convert time to datetime
    df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], format="%H:%M")

    conflicts = 0

    for platform in df["Platform_Number"].unique():

        platform_df = df[df["Platform_Number"] == platform]
        platform_df = platform_df.sort_values("Departure_Time")

        times = platform_df["Departure_Time"].tolist()

        for i in range(len(times) - 1):
            diff = (times[i + 1] - times[i]).seconds / 60
            if diff <= 20:
                conflicts += 1

    return conflicts


# ----------------------------
# Recommendation Engine
# ----------------------------
def recommend_actions(predicted_occupancy, platform_conflict):

    recommendations = []

    # Overcrowding rules
    if predicted_occupancy > 100:
        recommendations.append("Add special additional train service.")
        recommendations.append("Increase number of coaches by 2.")
    elif predicted_occupancy > 85:
        recommendations.append("Add 1 extra coach.")
    elif predicted_occupancy < 50:
        recommendations.append("Reduce 1 coach to optimize resource usage.")

    # Platform conflict rule
    if platform_conflict:
        recommendations.append("Reassign one train to a different platform.")

    if not recommendations:
        recommendations.append("No major operational changes required.")

    return recommendations


# ----------------------------
# Efficiency Score Calculation
# ----------------------------
def calculate_efficiency_score(utilization, overcrowding_penalty, conflict_penalty):

    score = utilization - overcrowding_penalty - conflict_penalty

    # Clamp between 0–100
    score = max(0, min(score, 100))

    return score