
from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.Day3_preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Dataset loaded successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 2. DEFINE MODELS
# ==========================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

}


# ==========================================
# 3. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}

for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    # Create a fresh preprocessor for each model
    preprocessor = build_preprocessor(X_train)

    # Complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print(f"{model_name} training completed!")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================
# 4. COMPARE ALL MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

# Day 1 baseline
baseline = pd.DataFrame([{
    "Model": "Logistic Regression (Day 1)",
    "Accuracy": 0.8055358410220014,
    "Precision": 0.6572327044025157,
    "Recall": 0.5588235294117647,
    "ROC-AUC": 0.8418610659019866
}])

comparison_df = pd.concat(
    [baseline, results_df],
    ignore_index=True
)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


# ==========================================
# 5. SELECT CHAMPION MODEL
# ==========================================

# Prioritise churn recall.
# Use ROC-AUC and precision as tie-breakers.

champion_name = max(
    results,
    key=lambda row: (
        row["Recall"],
        row["ROC-AUC"],
        row["Precision"]
    )
)["Model"]

champion_model = trained_models[champion_name]

print("\n========== CHAMPION MODEL ==========")

print("Selected model:", champion_name)

champion_metrics = results_df[
    results_df["Model"] == champion_name
].iloc[0]

print("Accuracy:", round(champion_metrics["Accuracy"], 4))
print("Precision:", round(champion_metrics["Precision"], 4))
print("Recall:", round(champion_metrics["Recall"], 4))
print("ROC-AUC:", round(champion_metrics["ROC-AUC"], 4))


# ==========================================
# 6. SAVE CHAMPION MODEL
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_comparison_champion.joblib"

joblib.dump(champion_model, MODEL_PATH)

print("\nChampion model saved successfully!")
print("Model location:", MODEL_PATH)


# ==========================================
# 7. SAVE COMPARISON RESULTS
# ==========================================

RESULTS_PATH = ROOT_DIR / "models" / "day2_model_comparison.csv"

comparison_df.to_csv(RESULTS_PATH, index=False)

print("\nComparison results saved!")
print("Results location:", RESULTS_PATH)

print("\nDAY 2 MODEL COMPARISON COMPLETED!")
