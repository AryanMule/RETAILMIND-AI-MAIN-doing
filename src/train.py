from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_day1.joblib"

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

    data = pd.read_csv(DATA_PATH)

    print("Dataset shape:", data.shape)
    print("\nSchema:")
    data.info()

    print("\nTarget counts:")
    print(data["Churn"].value_counts(dropna=False))

    print("\nTarget proportions:")
    print(data["Churn"].value_counts(normalize=True))

    X = prepare_features(data)
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Churn must contain only 'Yes' and 'No', with no missing values."
        )

    print("\nMissing feature values after numeric conversion:")
    print(X.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in NUMERIC_COLUMNS
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    print("\nDay 1 test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Confirm that the saved pipeline reproduces the predictions.
    loaded_pipeline = joblib.load(MODEL_PATH)
    loaded_predictions = loaded_pipeline.predict(X_test)

    assert (predictions == loaded_predictions).all()
    print("\nSaved-model reload check: PASSED")
    print(f"Model saved to: {MODEL_PATH}")

    metric_rows = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in metrics.items()
    )

    report = (
        "\n\n### Day 1 baseline results\n\n"
        "Model: Logistic Regression\n\n"
        "Split: 80/20, stratified, random_state=42\n\n"
        "| Metric | Test score |\n"
        "| --- | ---: |\n"
        f"{metric_rows}\n"
    )

    with (ROOT / "README.md").open(
        "a", encoding="utf-8"
    ) as readme:
        readme.write(report)

    print("Baseline metrics appended to README.md")


if __name__ == "__main__":
    main()