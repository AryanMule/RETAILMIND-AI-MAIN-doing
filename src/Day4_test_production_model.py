
from pathlib import Path
import joblib

from Day3_preprocessing import load_dataset, split_dataset

# 1. Load saved production model

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "churn_v4.joblib"

model = joblib.load(MODEL_PATH)

print("Production model loaded successfully!")


# 2. Load test data

X, y = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)


# 3. Predict using saved model

predictions = model.predict(X_test.head(5))

probabilities = model.predict_proba(X_test.head(5))[:, 1]


# 4. Display predictions

print("\nCUSTOMER CHURN PREDICTIONS")
print("=" * 50)

for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities)
):

    churn_status = "Yes" if prediction == 1 else "No"

    print(f"\nCustomer {i + 1}")
    print("Predicted Churn:", churn_status)
    print("Churn Probability:", round(probability, 4))


print("\nProduction model verification completed!")
