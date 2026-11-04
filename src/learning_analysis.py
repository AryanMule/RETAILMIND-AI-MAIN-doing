
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold

from preprocessing import load_dataset, split_dataset


# 1. Project paths

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "churn_v3.joblib"
OUTPUT_PATH = ROOT_DIR / "models" / "learning_curve.png"


# 2. Load dataset

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)


# 3. Load tuned champion

model = joblib.load(MODEL_PATH)

print("Champion model loaded successfully!")


# 4. Cross-validation setup

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 5. Generate learning curve

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    cv=cv,
    scoring="recall",
    n_jobs=-1
)


# 6. Calculate mean scores

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

print("\nLearning Curve Results")

for size, train_score, val_score in zip(
    train_sizes,
    train_mean,
    validation_mean
):
    print(
        f"Training Size: {size:4d} | "
        f"Training Recall: {train_score:.4f} | "
        f"Validation Recall: {val_score:.4f}"
    )


# 7. Plot learning curve

plt.figure(figsize=(9, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training Recall"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation Recall"
)

plt.xlabel("Number of Training Samples")
plt.ylabel("Recall")
plt.title("Learning Curve - Tuned Random Forest")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(OUTPUT_PATH)

plt.show()

print("\nLearning curve saved:", OUTPUT_PATH)
