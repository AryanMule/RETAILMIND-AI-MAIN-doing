
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from Day3_preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)

from Day4_feature_engineering import CustomerFeatureEngineer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Dataset loaded successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. Class imbalance calculation
# --------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass distribution:")
print("Non-churn customers:", negative_count)
print("Churn customers:", positive_count)

print("XGBoost scale_pos_weight:", round(scale_pos_weight, 2))


# --------------------------------------------------
# 4. Create preprocessor
# --------------------------------------------------

X_engineered = CustomerFeatureEngineer().fit_transform(X_train)

preprocessor = build_preprocessor(X_engineered)


# --------------------------------------------------
# 5. Define ensemble models
# --------------------------------------------------

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = []
trained_models = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("feature_engineering", CustomerFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline


# --------------------------------------------------
# 7. Compare ensemble models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nENSEMBLE MODEL COMPARISON")
print("=" * 80)

print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

results_df.to_csv(
    MODEL_DIR / "day4_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 9. Select Day 4 champion
# --------------------------------------------------

champion_row = results_df.sort_values(
    by=["Recall", "ROC-AUC"],
    ascending=False
).iloc[0]

champion_name = champion_row["Model"]

champion_model = trained_models[champion_name]

joblib.dump(
    champion_model,
    MODEL_DIR / "churn_v4.joblib"
)

print("\nDay 4 Champion:", champion_name)

print("Saved model:", MODEL_DIR / "churn_v4.joblib")

print("\nEnsemble training completed successfully!")
