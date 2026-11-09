
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.Day6_image_pipeline import (
    load_image_metadata,
    split_image_data,
    create_tf_dataset
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "fashion_cnn.keras"
CLASS_NAMES_PATH = MODELS_DIR / "fashion_class_names.json"


# --------------------------------------------------
# 2. Load class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Categories:", class_names)


# --------------------------------------------------
# 3. Load test dataset
# --------------------------------------------------

df, _ = load_image_metadata()

_, test_df = split_image_data(df)

test_dataset = create_tf_dataset(
    test_df,
    training=False
)


# --------------------------------------------------
# 4. Load trained CNN
# --------------------------------------------------

model = keras.models.load_model(MODEL_PATH)

print("\nCNN model loaded successfully!")

model.summary()


# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n========== CNN TEST RESULTS ==========")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

y_prob = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)


# Collect actual labels in the same dataset order
y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
])

accuracy = accuracy_score(y_true, y_pred)

print("\nCalculated Accuracy:", round(accuracy, 4))


# --------------------------------------------------
# 7. Classification report
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\n========== CLASSIFICATION REPORT ==========")

print(report_df.round(4))


# Save classification report
report_df.to_csv(
    MODELS_DIR / "day6_classification_report.csv"
)


# --------------------------------------------------
# 8. Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    MODELS_DIR / "day6_confusion_matrix.csv"
)


# --------------------------------------------------
# 9. Plot confusion matrix
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

plt.title("Fashion Product CNN - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODELS_DIR / "day6_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix image saved!")


# --------------------------------------------------
# 10. Find frequently confused categories
# --------------------------------------------------

confused_pairs = []

for i in range(len(class_names)):

    for j in range(len(class_names)):

        if i != j and cm[i, j] > 0:

            confused_pairs.append({
                "Actual Category": class_names[i],
                "Predicted Category": class_names[j],
                "Incorrect Predictions": int(cm[i, j])
            })


confused_df = pd.DataFrame(confused_pairs)

if not confused_df.empty:

    confused_df = confused_df.sort_values(
        by="Incorrect Predictions",
        ascending=False
    )

    print("\n========== MOST CONFUSED CATEGORIES ==========")

    print(confused_df.head(10).to_string(index=False))

    confused_df.to_csv(
        MODELS_DIR / "day6_confused_categories.csv",
        index=False
    )

else:
    print("\nNo classification errors found!")


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n========== DAY 6 EVALUATION COMPLETED ==========")

print(f"Test Accuracy: {accuracy:.4f}")

print("\nGenerated files:")
print("- day6_classification_report.csv")
print("- day6_confusion_matrix.csv")
print("- day6_confusion_matrix.png")
print("- day6_confused_categories.csv")

print("\nCNN evaluation completed successfully!")
