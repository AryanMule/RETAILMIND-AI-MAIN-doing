
from pathlib import Path
import joblib
import pandas as pd


# 1. Locate saved model

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"


# 2. Load trained pipeline

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# 3. Create reusable prediction function

def predict(customer_data):

    # Convert dictionary into DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Make prediction
    prediction = model.predict(customer_df)[0]

    # Get churn probability
    probability = model.predict_proba(customer_df)[0][1]

    # Convert numerical prediction into label
    churn_label = "Yes" if prediction == 1 else "No"

    return {
        "prediction": churn_label,
        "churn_probability": round(float(probability), 4)
    }


# 4. Test with a new customer

if __name__ == "__main__":

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.50,
        "TotalCharges": 141.00
    }

    result = predict(customer)

    print("\n========== CUSTOMER PREDICTION ==========")
    print(result)
