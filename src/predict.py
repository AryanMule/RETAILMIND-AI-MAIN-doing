import joblib
import pandas as pd

from train import DATA_PATH, MODEL_PATH, prepare_features


def predict(customers):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python src/train.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    features = prepare_features(customers)

    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    return pd.DataFrame(
        {
            "prediction": predictions,
            "churn_label": [
                "Yes" if value == 1 else "No"
                for value in predictions
            ],
            "churn_probability": probabilities,
        },
        index=customers.index,
    )


def main():
    customers = pd.read_csv(DATA_PATH).head(3)

    # The target is not supplied to the prediction helper.
    customers = customers.drop(columns=["Churn"])

    print("Predictions for three sample customers:")
    print(predict(customers).to_string(index=False))


if __name__ == "__main__":
    main()