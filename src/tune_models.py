
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load and split dataset
# --------------------------------------------------

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

print("Dataset loaded successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. Cross-validation strategy
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 4. Define candidate models and parameter grids
# --------------------------------------------------

model_configs = {

    "Decision Tree": {
        "classifier": DecisionTreeClassifier(
            random_state=42
        ),

        "params": {
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_split": [2, 10, 20],
            "classifier__min_samples_leaf": [1, 5, 10]
        }
    },

    "Random Forest": {
        "classifier": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 5],
            "classifier__max_features": ["sqrt"]
        }
    },

    "SVM": {
    "classifier": CalibratedClassifierCV(
        estimator=SVC(random_state=42),
        method="sigmoid",
        cv=3
    ),

    "params": {
        "classifier__estimator__C": [0.1, 1, 10],
        "classifier__estimator__kernel": ["linear", "rbf"],
        "classifier__estimator__gamma": ["scale", "auto"]
    }
}
}


# --------------------------------------------------
# 5. GridSearchCV for every model
# --------------------------------------------------

results = []
best_models = {}

for model_name, config in model_configs.items():

    print("\n" + "=" * 60)
    print("TUNING:", model_name)
    print("=" * 60)

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", config["classifier"])
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    best_index = grid_search.best_index_

    train_recall = grid_search.cv_results_[
        "mean_train_score"
    ][best_index]

    validation_recall = grid_search.best_score_

    results.append({
        "Model": model_name,
        "Best CV Recall": validation_recall,
        "Training CV Recall": train_recall,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test ROC-AUC": roc_auc,
        "Best Parameters": grid_search.best_params_
    })

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation Recall:")
    print(round(validation_recall, 4))

    print("\nTest Recall:")
    print(round(recall, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 6. Compare tuned models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nFINAL TUNING COMPARISON")
print("=" * 80)

print(
    results_df[
        [
            "Model",
            "Best CV Recall",
            "Training CV Recall",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test ROC-AUC"
        ]
    ].round(4).to_string(index=False)
)


# --------------------------------------------------
# 7. Select best tuned model
# --------------------------------------------------

champion_row = results_df.sort_values(
    by=["Best CV Recall", "Test ROC-AUC"],
    ascending=False
).iloc[0]

champion_name = champion_row["Model"]
champion_model = best_models[champion_name]

champion_path = MODEL_DIR / "churn_v3.joblib"

joblib.dump(champion_model, champion_path)

print("\nTUNED CHAMPION:", champion_name)
print("Saved model:", champion_path)


# --------------------------------------------------
# 8. Save comparison results
# --------------------------------------------------

results_df.to_csv(
    MODEL_DIR / "day3_tuning_results.csv",
    index=False
)

print("\nDay 3 tuning results saved successfully!")
