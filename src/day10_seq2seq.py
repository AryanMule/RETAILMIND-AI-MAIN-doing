# ============================================================
# Day 10 - Seq2Seq + Attention Demand Forecasting
# RetailMind AI
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    Attention,
    Concatenate
)
from tensorflow.keras.models import Model


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

data_path = "data/train.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(df.head())


# ------------------------------------------------------------
# 2. CLEAN DATA
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df = df.dropna()


# ------------------------------------------------------------
# 3. SELECT ONE STORE AND ONE ITEM
# ------------------------------------------------------------
# Same idea as Day 9.
# We forecast one clean store-item time series.

store_id = 1
item_id = 1

data = df[
    (df["store"] == store_id) &
    (df["item"] == item_id)
].copy()

print("\nSelected Store:", store_id)
print("Selected Item:", item_id)


# ------------------------------------------------------------
# 4. CREATE DAILY TIME SERIES
# ------------------------------------------------------------

data = data[["date", "sales"]]

data = data.set_index("date")

data = data.resample("D").sum()

data["sales"] = data["sales"].fillna(0)

print("\nTime series:")
print(data.head())


# ------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ------------------------------------------------------------
# Keep the latest 28 days for testing.
# This matches the idea used in Day 9.

test_days = 28

train_data = data.iloc[:-test_days]
test_data = data.iloc[-test_days:]


# ------------------------------------------------------------
# 6. SCALE DATA
# ------------------------------------------------------------

scaler = MinMaxScaler()

# Fit scaler ONLY using training data
scaler.fit(train_data[["sales"]])

scaled_sales = scaler.transform(data[["sales"]])


# ------------------------------------------------------------
# 7. CREATE SEQ2SEQ WINDOWS
# ------------------------------------------------------------
#
# Day 9:
# Previous 30 days -> next 1 day
#
# Day 10:
# Previous 30 days -> next 14 days
#

look_back = 30
forecast_days = 14

X = []
y = []

for i in range(look_back, len(scaled_sales) - forecast_days + 1):

    # Encoder input
    # Previous 30 days
    X.append(
        scaled_sales[i - look_back:i]
    )

    # Decoder target
    # Next 14 days
    y.append(
        scaled_sales[i:i + forecast_days]
    )


X = np.array(X)
y = np.array(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 8. TRAIN / TEST SEQUENCES
# ------------------------------------------------------------
# Keep the most recent available sequence for final testing.

X_train = X[:-1]
y_train = y[:-1]

X_test = X[-1:]
y_test = y[-1:]

print("\nTraining sequences:", X_train.shape)
print("Testing sequences:", X_test.shape)


# ------------------------------------------------------------
# 9. BUILD SEQ2SEQ MODEL
# ------------------------------------------------------------

# Encoder input
encoder_input = Input(
    shape=(look_back, 1),
    name="encoder_input"
)


# ------------------------------------------------------------
# ENCODER
# ------------------------------------------------------------
# return_sequences=True is important because Attention
# needs information from every historical timestep.

encoder_output, state_h, state_c = LSTM(
    64,
    return_sequences=True,
    return_state=True,
    name="encoder_lstm"
)(encoder_input)


# ------------------------------------------------------------
# DECODER
# ------------------------------------------------------------
# Repeat the encoder's final hidden state for each future day.

decoder_input = RepeatVector(
    forecast_days,
    name="repeat_vector"
)(state_h)


decoder_output = LSTM(
    64,
    return_sequences=True,
    name="decoder_lstm"
)(
    decoder_input,
    initial_state=[state_h, state_c]
)


# ------------------------------------------------------------
# 10. ADD ATTENTION
# ------------------------------------------------------------
# Attention compares the decoder states with the encoder
# history and focuses on useful historical days.

attention_layer = Attention(name="attention")

attention_output = attention_layer(
    [decoder_output, encoder_output]
)


# Combine decoder information with attention information
combined = Concatenate(name="combine_attention")(
    [decoder_output, attention_output]
)


# ------------------------------------------------------------
# 11. OUTPUT LAYER
# ------------------------------------------------------------
# Dense(1) predicts sales for every decoder timestep.
#
# Output:
# next 14 days of demand

output = Dense(
    1,
    name="forecast"
)(combined)


# Create model
model = Model(
    inputs=encoder_input,
    outputs=output
)


# ------------------------------------------------------------
# 12. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------------
# 13. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Seq2Seq + Attention model...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ------------------------------------------------------------
# 14. PREDICT NEXT 14 DAYS
# ------------------------------------------------------------

prediction_scaled = model.predict(X_test)

# Shape is (1, 14, 1)
# Convert it to (14, 1)

prediction_scaled = prediction_scaled[0]

actual_scaled = y_test[0]


# Convert back into real sales values
predictions = scaler.inverse_transform(
    prediction_scaled
)

actual = scaler.inverse_transform(
    actual_scaled
)


# ------------------------------------------------------------
# 15. EVALUATE MULTI-STEP FORECAST
# ------------------------------------------------------------

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print("\n================================")
print("DAY 10 MODEL EVALUATION")
print("================================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ------------------------------------------------------------
# 16. GET FORECAST DATES
# ------------------------------------------------------------

forecast_dates = data.index[-forecast_days:]


# ------------------------------------------------------------
# 17. CREATE RESULTS DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame({

    "Date": forecast_dates,

    "Actual_Sales": actual.flatten(),

    "Predicted_Sales": predictions.flatten()

})

print("\nActual vs Predicted:")
print(results)


# ------------------------------------------------------------
# 18. PLOT ACTUAL VS PREDICTED
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    results["Date"],
    results["Actual_Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted_Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title(
    "Day 10 - Seq2Seq Attention Demand Forecast"
)

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "day10_forecast.png"
)

plt.show()


# ------------------------------------------------------------
# 19. VISUALISE ATTENTION WEIGHTS
# ------------------------------------------------------------
#
# We create another attention layer configured to return
# the attention scores.
#
# These scores show how strongly each forecast step
# looks at the previous 30 days.
# ------------------------------------------------------------

attention_visualizer = tf.keras.layers.Attention()

attention_values, attention_scores = attention_visualizer(
    [decoder_output, encoder_output],
    return_attention_scores=True
)

attention_model = Model(
    inputs=encoder_input,
    outputs=attention_scores
)

scores = attention_model.predict(X_test)

# Remove batch dimension
scores = scores[0]


plt.figure(figsize=(12, 6))

plt.imshow(
    scores,
    aspect="auto",
    cmap="viridis"
)

plt.colorbar(
    label="Attention Weight"
)

plt.title(
    "Attention Weights"
)

plt.xlabel(
    "Previous 30 Days"
)

plt.ylabel(
    "Forecast Day"
)

plt.tight_layout()

plt.savefig(
    "day10_attention_weights.png"
)

plt.show()


# ------------------------------------------------------------
# 20. SAVE FORECAST FOR PROCUREMENT
# ------------------------------------------------------------
# This CSV represents the multi-period demand forecast
# that can later feed procurement / inventory planning.

results.to_csv(
    "day10_multiweek_forecast.csv",
    index=False
)


# ------------------------------------------------------------
# 21. SAVE MODEL
# ------------------------------------------------------------

model.save(
    "day10_seq2seq_attention.keras"
)


# ------------------------------------------------------------
# 22. SIMPLE DAY 9 VS DAY 10 EXPLANATION
# ------------------------------------------------------------

print("\n================================")
print("DAY 9 VS DAY 10")
print("================================")

print(
    "Day 9 LSTM predicts one next period at a time."
)

print(
    "Day 10 Seq2Seq predicts multiple future periods together."
)

print(
    "Attention helps the model focus on important historical days."
)


# ------------------------------------------------------------
# 23. BUSINESS VALUE
# ------------------------------------------------------------

print("\n================================")
print("BUSINESS VALUE")
print("================================")

print(
    "Multi-step demand forecasts help procurement teams "
    "plan inventory for several future days instead of "
    "only predicting the next day."
)


print("\nFiles created:")
print("1. day10_forecast.png")
print("2. day10_attention_weights.png")
print("3. day10_multiweek_forecast.csv")
print("4. day10_seq2seq_attention.keras")

print("\nDay 10 completed successfully!")