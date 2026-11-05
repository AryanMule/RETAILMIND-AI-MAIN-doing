from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import load_dataset, split_dataset


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


# 2. Load dataset and reproduce the same split
X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

y_test = np.asarray(y_test)


# 3. Load ANN and preprocessing objects
ann_model = keras.models.load_model(
    MODELS_DIR / "churn_v5_ann.keras"
)

preprocessing_objects = joblib.load(
    MODELS_DIR / "day5_preprocessing.joblib"
)

feature_engineer = preprocessing_objects["feature_engineer"]
preprocessor = preprocessing_objects["preprocessor"]


# 4. Prepare test data for ANN
X_test_engineered = feature_engineer.transform(X_test)

X_test_processed = preprocessor.transform(X_test_engineered)

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)


# 5. ANN predictions
ann_prob = ann_model.predict(
    X_test_processed,
    verbose=0
).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)


# 6. Load Day 4 XGBoost champion
xgb_model = joblib.load(
    MODELS_DIR / "churn_v4.joblib"
)

xgb_pred = xgb_model.predict(X_test)

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]


# 7. Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


# 8. Compare models
results = [
    evaluate_model(
        "Day 4 XGBoost",
        y_test,
        xgb_pred,
        xgb_prob
    ),

    evaluate_model(
        "Day 5 ANN",
        y_test,
        ann_pred,
        ann_prob
    )
]


comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))


# 9. Save comparison results
comparison.to_csv(
    MODELS_DIR / "day5_model_comparison.csv",
    index=False
)


# 10. Identify the model with higher recall
best_model = comparison.loc[
    comparison["Recall"].idxmax()
]

print("\n========== BEST MODEL BY RECALL ==========")
print("Model:", best_model["Model"])
print("Recall:", round(best_model["Recall"], 4))

print("\nComparison saved successfully!")