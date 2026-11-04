from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train import prepare_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DAY1_MODEL_PATH = ROOT / "models" / "churn_day1.joblib"
MODEL_PATH = ROOT / "models" / "churn_v3.joblib"


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

    # Preserve the same split used on previous days.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load only the trusted artifact created by your own project.
    day1_pipeline = joblib.load(DAY1_MODEL_PATH)
    preprocessing = day1_pipeline.named_steps["preprocessing"]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    candidates = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classifier__max_depth": [3, 6, None],
                "classifier__min_samples_leaf": [1, 10],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
            {
                "classifier__max_depth": [6, None],
                "classifier__min_samples_leaf": [1, 5],
            },
        ),
        "SVM": (
            SVC(
                kernel="rbf",
                random_state=42,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.01],
            },
        ),
    }

    results = []
    parameter_lines = []
    champion = None
    champion_name = None
    champion_params = None
    best_cv_recall = float("-inf")

    print(f"Training rows: {len(X_train)}")
    print(f"Hold-out rows: {len(X_test)}")
    print("Selection metric: mean CV churn recall")

    for name, (classifier, parameter_grid) in candidates.items():
        print(f"\nTuning {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", clone(preprocessing)),
                ("classifier", classifier),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring={
                "recall": "recall",
                "accuracy": "accuracy",
            },
            refit="recall",
            cv=cv,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        index = search.best_index_
        scores = search.cv_results_

        train_recall = scores["mean_train_recall"][index]
        validation_recall = scores["mean_test_recall"][index]

        results.append(
            {
                "Model": name,
                "Train recall": train_recall,
                "CV recall": validation_recall,
                "CV recall std": scores["std_test_recall"][index],
                "Recall gap": train_recall - validation_recall,
                "Train accuracy": scores["mean_train_accuracy"][index],
                "CV accuracy": scores["mean_test_accuracy"][index],
            }
        )

        parameter_lines.append(
            f"- {name}: `{search.best_params_}`"
        )

        print("Best parameters:", search.best_params_)
        print(f"Mean CV recall: {validation_recall:.4f}")

        # Select using CV results only, not hold-out results.
        # An exact tie retains the first candidate.
        if validation_recall > best_cv_recall:
            best_cv_recall = validation_recall
            champion = search.best_estimator_
            champion_name = name
            champion_params = search.best_params_

    comparison = pd.DataFrame(results).sort_values(
        by="CV recall",
        ascending=False,
        kind="stable",
    )

    print("\nTuned model comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nChampion: {champion_name}")
    print("Chosen parameters:", champion_params)

    # Calculate a learning curve for the selected configuration.
    # No hold-out data is used here.
    print("\nCalculating champion learning curve...")

    sizes, train_scores, validation_scores = learning_curve(
        estimator=clone(champion),
        X=X_train,
        y=y_train,
        train_sizes=[0.3, 0.6, 1.0],
        cv=cv,
        scoring="recall",
        shuffle=True,
        random_state=42,
        n_jobs=1,
        error_score="raise",
    )

    curve = pd.DataFrame(
        {
            "Training rows per fold": sizes,
            "Train recall": train_scores.mean(axis=1),
            "Validation recall": validation_scores.mean(axis=1),
            "Validation recall std": validation_scores.std(axis=1),
        }
    )

    print("\nChampion learning curve:")
    print(
        curve.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # GridSearchCV has already refitted the champion on all X_train.
    # Evaluate it once on the hold-out after model selection.
    predictions = champion.predict(X_test)

    if hasattr(champion.named_steps["classifier"], "predict_proba"):
        churn_scores = champion.predict_proba(X_test)[:, 1]
    else:
        churn_scores = champion.decision_function(X_test)

    holdout_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, churn_scores),
    }

    print("\nChampion hold-out metrics:")
    for name, value in holdout_metrics.items():
        print(f"{name}: {value:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODEL_PATH)

    loaded_model = joblib.load(MODEL_PATH)
    assert (
        predictions == loaded_model.predict(X_test)
    ).all()

    print("\nSaved-model reload check: PASSED")
    print(f"Saved to: {MODEL_PATH}")

    # Build Markdown tables without adding a table-formatting library.
    comparison_lines = [
        "| Model | Train recall | CV recall | CV std | "
        "Recall gap | Train accuracy | CV accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in comparison.to_dict(orient="records"):
        comparison_lines.append(
            f"| {row['Model']} "
            f"| {row['Train recall']:.4f} "
            f"| {row['CV recall']:.4f} "
            f"| {row['CV recall std']:.4f} "
            f"| {row['Recall gap']:.4f} "
            f"| {row['Train accuracy']:.4f} "
            f"| {row['CV accuracy']:.4f} |"
        )

    curve_lines = [
        "| Training rows per fold | Train recall | "
        "Validation recall | Validation std |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for row in curve.to_dict(orient="records"):
        curve_lines.append(
            f"| {int(row['Training rows per fold'])} "
            f"| {row['Train recall']:.4f} "
            f"| {row['Validation recall']:.4f} "
            f"| {row['Validation recall std']:.4f} |"
        )

    holdout_lines = [
        "| Metric | Hold-out score |",
        "| --- | ---: |",
    ]

    for name, value in holdout_metrics.items():
        holdout_lines.append(f"| {name} | {value:.4f} |")

    report = (
        "\n\n## Day 3: Tuning and bias-variance inspection\n\n"
        "Selection: highest mean churn recall from three-fold "
        "stratified CV on the training partition. "
        "Exact ties retain the first candidate.\n\n"
        + "\n".join(comparison_lines)
        + "\n\n### Best parameters per model\n\n"
        + "\n".join(parameter_lines)
        + f"\n\n### Locked champion: {champion_name}\n\n"
        + f"Parameters: `{champion_params}`\n\n"
        + f"Mean CV recall: {best_cv_recall:.4f}\n\n"
        + "Saved pipeline: `models/churn_v3.joblib`\n\n"
        + "### Champion learning curve\n\n"
        + "\n".join(curve_lines)
        + "\n\n### Hold-out evaluation\n\n"
        + "\n".join(holdout_lines)
        + "\n\n### Interpretation notes\n\n"
        "A large train-versus-validation gap suggests variance. "
        "Low, converged scores may suggest bias, but recall alone "
        "does not establish underfitting. Check accuracy and "
        "precision as well.\n\n"
        "Rising validation recall with more training data suggests "
        "that more data may help. Flat, low scores with a small "
        "gap suggest reviewing model capacity or feature signal. "
        "A persistent large gap suggests considering stronger "
        "regularisation. These are diagnostic indications, "
        "not automatic conclusions.\n\n"
        "The hold-out was already inspected on previous days, "
        "so it is not a fresh independent final test set. "
        "Learning curves also reuse the tuning data and are "
        "diagnostic, not an unbiased final performance estimate.\n"
    )

    with (ROOT / "README.md").open(
        "a", encoding="utf-8"
    ) as readme:
        readme.write(report)

    print("\nDay 3 results appended to README.md")


if __name__ == "__main__":
    main()