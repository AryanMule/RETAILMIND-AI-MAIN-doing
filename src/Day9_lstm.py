# ============================================================
# Day 9 - RNN / LSTM Demand Forecasting
# RetailMind AI
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------

# Dataset should be inside the data folder
data_path = "data/train.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(df.head())
print("\nDataset shape:", df.shape)


# ------------------------------------------------------------
# 2. BASIC DATA CLEANING
# ------------------------------------------------------------

# Convert date column from string to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort dataset by date
df = df.sort_values("date")

# Remove rows with missing values
df = df.dropna()

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. SELECT ONE STORE AND ONE ITEM
# ------------------------------------------------------------
# The dataset contains many stores and many items.
# To keep the Day 9 LSTM simple, we forecast demand
# for one store-product combination.

store_id = 1
item_id = 1

data = df[
    (df["store"] == store_id) &
    (df["item"] == item_id)
].copy()

print("\nSelected Store:", store_id)
print("Selected Item:", item_id)
print("Number of records:", len(data))


# ------------------------------------------------------------
# 4. CREATE CLEAN DAILY TIME SERIES
# ------------------------------------------------------------

# Keep only date and sales
data = data[["date", "sales"]]

# Make date the index
data = data.set_index("date")

# Resample into daily sales.
# sum() makes sure there is one sales value for each day.
data = data.resample("D").sum()

# Fill any missing day with 0 sales
data["sales"] = data["sales"].fillna(0)

print("\nTime Series Data:")
print(data.head())


# ------------------------------------------------------------
# 5. HOLD OUT THE MOST RECENT WEEKS FOR TESTING
# ------------------------------------------------------------
# Task asks us to test using the most recent weeks.
# We use the final 4 weeks = 28 days.

test_days = 28

train_data = data.iloc[:-test_days]
test_data = data.iloc[-test_days:]

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))


# ------------------------------------------------------------
# 6. SCALE THE SALES VALUES
# ------------------------------------------------------------
# LSTM works better when values are on a smaller scale.
# MinMaxScaler converts values into the range 0 to 1.

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data[["sales"]])

# We transform the full data using the scaler learned
# only from training data.
all_scaled = scaler.transform(data[["sales"]])


# ------------------------------------------------------------
# 7. CREATE LOOK-BACK WINDOWS
# ------------------------------------------------------------
# Example with look_back = 30:
#
# Previous 30 days sales -> predict next day's sales

look_back = 30

X = []
y = []

for i in range(look_back, len(all_scaled)):

    # Previous 30 days
    X.append(all_scaled[i - look_back:i])

    # Next day
    y.append(all_scaled[i])


X = np.array(X)
y = np.array(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 8. SPLIT WINDOWS INTO TRAIN AND TEST
# ------------------------------------------------------------
# The last 28 predictions are kept for testing.

X_train = X[:-test_days]
y_train = y[:-test_days]

X_test = X[-test_days:]
y_test = y[-test_days:]

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# ------------------------------------------------------------
# 9. BUILD THE LSTM MODEL
# ------------------------------------------------------------

model = Sequential()

# LSTM learns patterns from previous days
model.add(
    LSTM(
        50,
        activation="relu",
        input_shape=(look_back, 1)
    )
)

# Final output = next day's predicted demand
model.add(Dense(1))


# ------------------------------------------------------------
# 10. COMPILE THE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------------
# 11. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ------------------------------------------------------------
# 12. MAKE DEMAND PREDICTIONS
# ------------------------------------------------------------

predictions_scaled = model.predict(X_test)

# Convert predictions back to original sales values
predictions = scaler.inverse_transform(predictions_scaled)

# Convert actual test values back to original sales values
actual = scaler.inverse_transform(y_test)


# ------------------------------------------------------------
# 13. EVALUATE MODEL
# ------------------------------------------------------------
# MAE = average absolute prediction error
# RMSE = gives more penalty to large prediction errors

mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ------------------------------------------------------------
# 14. CREATE RESULTS DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame({
    "Date": test_data.index,
    "Actual_Sales": actual.flatten(),
    "Predicted_Sales": predictions.flatten()
})

print("\nPredicted vs Actual Demand:")
print(results.head(10))


# ------------------------------------------------------------
# 15. PLOT ACTUAL VS PREDICTED SALES
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    results["Date"],
    results["Actual_Sales"],
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted_Sales"],
    label="Predicted Sales"
)

plt.title("LSTM Demand Forecast - Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

# Save plot so it can be shown as Day 9 output/artefact
plt.savefig("day9_demand_forecast.png")

plt.show()


# ------------------------------------------------------------
# 16. SAVE PREDICTIONS
# ------------------------------------------------------------
# This output can later be used by the inventory module.

results.to_csv(
    "day9_demand_predictions.csv",
    index=False
)

print("\nPredictions saved as:")
print("day9_demand_predictions.csv")


# ------------------------------------------------------------
# 17. SAVE THE DEMAND FORECASTER
# ------------------------------------------------------------
# Saving the trained model allows the inventory planning
# module to use this model later.

model.save("demand_forecaster.keras")

print("\nModel saved as:")
print("demand_forecaster.keras")


# ------------------------------------------------------------
# 18. BUSINESS VALUE
# ------------------------------------------------------------

print("\n==============================")
print("BUSINESS VALUE")
print("==============================")

print(
    "The LSTM forecasts future product demand using historical "
    "sales. These demand predictions can help inventory planning "
    "reduce stockouts and avoid excess inventory."
)

print("\nDay 9 completed successfully!")
