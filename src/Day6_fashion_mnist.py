
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# 1. Set random seed
tf.random.set_seed(42)
np.random.seed(42)


# 2. Project directory
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# 3. Class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# 4. Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    keras.datasets.fashion_mnist.load_data()
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)


# 5. Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# 6. Add channel dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("CNN input shape:", X_train.shape)


# 7. Build CNN architecture
model = keras.Sequential([

    layers.Input(shape=(28, 28, 1)),

    # First convolution block
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classification layers
    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])


# 8. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 9. Display architecture
model.summary()


# 10. Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# 11. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# 12. Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFashion-MNIST Results")
print("---------------------")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# 13. Predict sample images
predictions = model.predict(X_test[:10], verbose=0)

print("\nSample Predictions")
print("------------------")

for i, prediction in enumerate(predictions):

    predicted_class = np.argmax(prediction)

    print(
        f"Image {i + 1}: "
        f"Actual = {class_names[y_test[i]]}, "
        f"Predicted = {class_names[predicted_class]}"
    )


# 14. Save CNN model
model.save(MODELS_DIR / "fashion_mnist_cnn.keras")

print("\nFashion-MNIST CNN saved successfully!")
