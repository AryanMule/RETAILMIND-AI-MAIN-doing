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

from train import prepare_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DAY1_MODEL_PATH = ROOT / "models" / "churn_day1.joblib"
DAY2_MODEL_PATH = ROOT / "models" / "churn_day2.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not DAY1_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Day 1 model not found: {DAY1_MODEL_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    X = prepare_features(data)
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Churn must contain only 'Yes' and 'No', "
            "with no missing values."
        )

    # Reproduce Day 1's split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load only the trusted model created by your own project.
    day1_pipeline = joblib.load(DAY1_MODEL_PATH)

    # Reuse Day 1's preprocessing configuration.
    preprocessing = day1_pipeline.named_steps["preprocessing"]

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
    }

    results = []
    champion_pipeline = None
    champion_name = None
    best_score = None

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

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

        if name == "SVM":
            scores = pipeline.decision_function(X_test)
        else:
            scores = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
            "ROC-AUC": roc_auc_score(y_test, scores),
        }
        results.append(metrics)

        # Highest recall wins; ROC-AUC breaks recall ties.
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

    print("\nDay 2 comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    DAY2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, DAY2_MODEL_PATH)

    # Check that saving and reloading preserve predictions.
    loaded_pipeline = joblib.load(DAY2_MODEL_PATH)

    assert (
        champion_pipeline.predict(X_test)
        == loaded_pipeline.predict(X_test)
    ).all()

    print("\nSaved-model reload check: PASSED")

    sample_predictions = loaded_pipeline.predict(X_test.head(3))
    print(
        "Sample churn predictions:",
        [
            "Yes" if value == 1 else "No"
            for value in sample_predictions
        ],
    )

    reason = (
        f"{champion_name} was selected by highest churn-class "
        f"recall ({best_score[0]:.4f}), with ROC-AUC "
        f"({best_score[1]:.4f}) used to break recall ties. "
        "If both metrics tie exactly, the first candidate "
        "in the listed order is retained."
    )

    print(f"\nChampion: {champion_name}")
    print(reason)
    print(f"Saved to: {DAY2_MODEL_PATH}")

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
        "All models used the same 80/20 stratified split "
        "with random_state=42.\n\n"
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

    print("\nComparison appended to README.md")


if __name__ == "__main__":
    main()