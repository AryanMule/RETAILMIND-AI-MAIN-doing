
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Import reusable functions
from preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

X, y = load_dataset()

print("Dataset loaded:", X.shape)


# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 3. BUILD PREPROCESSOR
# ==========================================

preprocessor = build_preprocessor(X_train)


# ==========================================
# 4. BUILD ML PIPELINE
# ==========================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model_pipeline.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. PREDICTIONS
# ==========================================

y_pred = model_pipeline.predict(X_test)

y_prob = model_pipeline.predict_proba(X_test)[:, 1]


# ==========================================
# 7. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)


print("\n========== DAY 1 BASELINE ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. SAVE BASELINE MODEL
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

joblib.dump(model_pipeline, MODEL_PATH)

print("\nBaseline model saved successfully!")
print("Location:", MODEL_PATH)
