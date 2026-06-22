import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("train.csv")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False, errors="coerce")

# Remove invalid dates
df = df.dropna(subset=["Order Date"])

# Sort by date
df = df.sort_values("Order Date")

# Create numeric day column
df["Day_Number"] = (df["Order Date"] - df["Order Date"].min()).dt.days

# Features and target
X = df[["Day_Number"]]
y = df["Sales"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
last_day = df["Day_Number"].max()

future_days = pd.DataFrame({
    "Day_Number": range(last_day + 1, last_day + 31)
})

future_days["Predicted Sales"] = model.predict(future_days)

# Save predictions
future_days.to_csv("future_predictions.csv", index=False)

print("Prediction completed successfully!")
print(future_days.head())

print("\nFile saved as future_predictions.csv")