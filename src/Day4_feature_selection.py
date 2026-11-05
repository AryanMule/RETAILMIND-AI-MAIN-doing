
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from Day3_preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)

from Day4_feature_engineering import CustomerFeatureEngineer


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Dataset loaded successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 2. Prepare preprocessor
# --------------------------------------------------

# Generate engineered columns to identify the
# numerical and categorical features.

feature_engineer = CustomerFeatureEngineer()

X_train_engineered = feature_engineer.fit_transform(X_train)

preprocessor = build_preprocessor(X_train_engineered)


# --------------------------------------------------
# 3. Create pipeline WITHOUT feature selection
# --------------------------------------------------

pipeline_without_selection = Pipeline([
    ("feature_engineering", CustomerFeatureEngineer()),

    ("preprocessor", preprocessor),

    ("classifier", LogisticRegression(max_iter=2000))
])


# --------------------------------------------------
# 4. Create pipeline WITH feature selection
# --------------------------------------------------

pipeline_with_selection = Pipeline([
    ("feature_engineering", CustomerFeatureEngineer()),

    ("preprocessor", preprocessor),

    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=20
    )),

    ("classifier", LogisticRegression(max_iter=2000))
])


# --------------------------------------------------
# 5. Train and evaluate both pipelines
# --------------------------------------------------

models = {
    "Without Feature Selection": pipeline_without_selection,
    "With Feature Selection": pipeline_with_selection
}

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


# --------------------------------------------------
# 6. Display comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nFEATURE SELECTION COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 7. Display selected feature count
# --------------------------------------------------

selector = pipeline_with_selection.named_steps[
    "feature_selection"
]

print("\nTotal transformed features:",
      len(selector.get_support()))

print("Selected features:",
      selector.get_support().sum())

print("\nFeature selection analysis completed!")
