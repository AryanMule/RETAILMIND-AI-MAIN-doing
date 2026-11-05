
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges into numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Separate features and target
    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ==========================================
# 4. BUILD PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor(X):

    # Identify numerical columns
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ==========================================
# 5. TEST MODULE
# ==========================================

if __name__ == "__main__":

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = build_preprocessor(X_train)

    print("Dataset loaded successfully!")

    print("Original features:", X.shape)

    print("Training records:", len(X_train))

    print("Testing records:", len(X_test))

    print("Training target distribution:")
    print(y_train.value_counts())

    print("\nPreprocessor created successfully!")

    print("\nNumerical features:")
    print(X_train.select_dtypes(include=["number"]).columns.tolist())

    print("\nCategorical features:")
    print(X_train.select_dtypes(exclude=["number"]).columns.tolist())
