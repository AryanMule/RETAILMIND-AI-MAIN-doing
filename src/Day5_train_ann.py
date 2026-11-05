from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)

from Day4_feature_engineering import CustomerFeatureEngineer


# 1. Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


# 2. Set project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# 3. Load dataset
X, y = load_dataset()

print("Dataset loaded:", X.shape)


# 4. Split dataset
X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 5. Feature engineering
feature_engineer = CustomerFeatureEngineer()

X_train_engineered = feature_engineer.fit_transform(X_train)
X_test_engineered = feature_engineer.transform(X_test)


# 6. Preprocessing
preprocessor = build_preprocessor(X_train_engineered)

X_train_processed = preprocessor.fit_transform(X_train_engineered)
X_test_processed = preprocessor.transform(X_test_engineered)


# Convert sparse matrices into dense arrays if required
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()


X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

y_train_array = np.asarray(y_train, dtype=np.float32)
y_test_array = np.asarray(y_test, dtype=np.float32)

print("Processed training shape:", X_train_processed.shape)


# 7. Calculate class weights
classes = np.unique(y_train_array)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_array
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("Class weights:", class_weights)


# 8. Build ANN model
model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(32, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])


# 9. Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)


# 10. Display architecture
model.summary()


# 11. Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# 12. Train ANN
history = model.fit(
    X_train_processed,
    y_train_array,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)


# 13. Predict churn probabilities
y_prob = model.predict(X_test_processed).ravel()

y_pred = (y_prob >= 0.5).astype(int)


# 14. Evaluate ANN
accuracy = accuracy_score(y_test_array, y_pred)
precision = precision_score(y_test_array, y_pred, zero_division=0)
recall = recall_score(y_test_array, y_pred, zero_division=0)
f1 = f1_score(y_test_array, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test_array, y_prob)

print("\n========== ANN MODEL RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# 15. Save trained ANN
model.save(MODELS_DIR / "churn_v5_ann.keras")


# 16. Save preprocessing objects
joblib.dump(
    {
        "feature_engineer": feature_engineer,
        "preprocessor": preprocessor
    },
    MODELS_DIR / "day5_preprocessing.joblib"
)


# 17. Save training history
np.savez(
    MODELS_DIR / "day5_training_history.npz",
    loss=history.history["loss"],
    val_loss=history.history["val_loss"],
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"]
)


print("\nANN model saved successfully!")
print("Preprocessing objects saved successfully!")