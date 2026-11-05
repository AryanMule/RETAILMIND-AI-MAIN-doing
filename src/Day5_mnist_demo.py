import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 2. Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# 3. Build ANN model
model = keras.Sequential([
    layers.Input(shape=(28, 28)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(64, activation="relu"),

    layers.Dense(10, activation="softmax")
])


# 4. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 5. Display model architecture
model.summary()


# 6. Train model
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# 7. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nMNIST Test Loss:", test_loss)
print("MNIST Test Accuracy:", test_accuracy)


# 8. Predict one test image
predictions = model.predict(X_test[:5])

for i, prediction in enumerate(predictions):
    predicted_digit = prediction.argmax()

    print(
        f"Image {i + 1}: "
        f"Actual = {y_test[i]}, "
        f"Predicted = {predicted_digit}"
    )


# 9. Save model
model.save("models/mnist_demo.keras")

print("\nMNIST model saved successfully!")