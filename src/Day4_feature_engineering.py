
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # No information needs to be learned during fit.
        return self

    def transform(self, X):

        # Work on a copy to avoid modifying the original dataset.
        X = X.copy()

        # ---------------------------------------
        # 1. Tenure Group
        # ---------------------------------------

        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 48, float("inf")],
            labels=["New", "Medium", "Long"]
        )

        # ---------------------------------------
        # 2. Average Monthly Charge
        # ---------------------------------------

        X["avg_monthly_charge"] = (
            X["TotalCharges"] /
            X["tenure"].replace(0, float("nan"))
        )

        # ---------------------------------------
        # 3. Number of Services
        # ---------------------------------------

        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        X["num_services"] = (
            X[service_columns] == "Yes"
        ).sum(axis=1)

        # ---------------------------------------
        # 4. Month-to-Month Contract Flag
        # ---------------------------------------

        X["is_month_to_month"] = (
            X["Contract"] == "Month-to-month"
        ).astype(int)

        # ---------------------------------------
        # 5. Support Service Flag
        # ---------------------------------------

        X["has_support_service"] = (
            (X["TechSupport"] == "Yes") |
            (X["OnlineSecurity"] == "Yes")
        ).astype(int)

        # ---------------------------------------
        # 6. Internet Service Flag
        # ---------------------------------------

        X["has_online_service"] = (
            X["InternetService"].notna() &
            (X["InternetService"] != "No")
        ).astype(int)

        return X


# ---------------------------------------
# Test the feature engineer
# ---------------------------------------

if __name__ == "__main__":

    from Day3_preprocessing import load_dataset

    X, y = load_dataset()

    feature_engineer = CustomerFeatureEngineer()

    X_engineered = feature_engineer.fit_transform(X)

    print("Original dataset shape:", X.shape)
    print("Engineered dataset shape:", X_engineered.shape)

    print("\nNewly created features:")

    new_features = [
        "tenure_group",
        "avg_monthly_charge",
        "num_services",
        "is_month_to_month",
        "has_support_service",
        "has_online_service"
    ]

    print(X_engineered[new_features].head())

    print("\nFeature engineering completed successfully!")
