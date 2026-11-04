from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Keep this path unchanged for the existing prediction helper.
MODEL_PATH = ROOT / "models" / "churn_day1.joblib"

DAY2_MODEL_PATH = ROOT / "models" / "churn_day2.joblib"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def prepare_features(data):
    features = data.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    ).copy()

    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    return features


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Place the Telco Churn CSV here: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Day 1 model is required but was not found: {MODEL_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    X = prepare_features(data)
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Churn must contain only 'Yes' and 'No', "
            "with no missing values."
        )

    # Same split settings as Day 1.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    # Reuse the preprocessing configuration from Day 1.
    # Only load the trusted model created by your own project.
    day1_pipeline = joblib.load(MODEL_PATH)
    preprocessing = day1_pipeline.named_steps["preprocessing"]

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
        ),
        "SVM": SVC(
            random_state=42,
        ),
    }

    results = []
    champion_pipeline = None
    champion_name = None
    best_score = None

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", clone(preprocessing)),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            churn_scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            churn_scores = pipeline.decision_function(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "ROC-AUC": roc_auc_score(y_test, churn_scores),
        }

        results.append(metrics)

        # Recall is primary; ROC-AUC breaks recall ties.
        selection_score = (
            metrics["Recall"],
            metrics["ROC-AUC"],
        )

        if best_score is None or selection_score > best_score:
            best_score = selection_score
            champion_name = name
            champion_pipeline = pipeline

    comparison = pd.DataFrame(results).sort_values(
        by=["Recall", "ROC-AUC"],
        ascending=False,
        kind="stable",
    )

    print("\nDay 2 model comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    DAY2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, DAY2_MODEL_PATH)

    # Verify that the saved pipeline reproduces predictions.
    loaded_pipeline = joblib.load(DAY2_MODEL_PATH)

    original_predictions = champion_pipeline.predict(X_test)
    loaded_predictions = loaded_pipeline.predict(X_test)

    assert (original_predictions == loaded_predictions).all()
    print("\nSaved-model reload check: PASSED")

    sample_predictions = loaded_pipeline.predict(X_test.head(3))

    print("\nSaved champion predictions for three customers:")
    print(
        [
            "Yes" if prediction == 1 else "No"
            for prediction in sample_predictions
        ]
    )

    reason = (
        f"{champion_name} was selected by highest churn-class "
        "recall, using ROC-AUC to break recall ties. "
        "If both metrics tie exactly, the first candidate "
        "in the listed order is retained."
    )

    print(f"\nDay 2 champion: {champion_name}")
    print(reason)
    print(f"Model saved to: {DAY2_MODEL_PATH}")

    table_lines = [
        "| Model | Accuracy | Precision | Recall | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in comparison.to_dict(orient="records"):
        table_lines.append(
            f"| {row['Model']} "
            f"| {row['Accuracy']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['ROC-AUC']:.4f} |"
        )

    report = (
        "\n\n## Day 2: Model comparison\n\n"
        "Reused Day 1 preprocessing configuration. "
        "All three models used the same 80/20 stratified "
        "split with random_state=42.\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        + reason
        + "\n\nSaved pipeline: `models/churn_day2.joblib`\n\n"
        "This champion is the best of the three Day 2 "
        "candidates under the stated selection rule on "
        "this hold-out set. It is not necessarily better "
        "than the Day 1 baseline.\n"
    )

    with (ROOT / "README.md").open(
        "a",
        encoding="utf-8",
    ) as readme:
        readme.write(report)

    print("\nDay 2 results appended to README.md")


if __name__ == "__main__":
    main()