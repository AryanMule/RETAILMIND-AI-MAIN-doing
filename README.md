# RetailMind AI - Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using customer demographics, account information, subscribed services, and billing details.

The project follows a structured machine learning workflow covering exploratory data analysis, preprocessing, baseline training, model comparison, hyperparameter tuning, cross-validation, learning-curve analysis, feature engineering, feature selection, ensemble learning, and production model verification.

---

## Project Objectives

- Analyze customer data to identify churn patterns.
- Clean and preprocess raw customer information.
- Build reusable Scikit-learn preprocessing pipelines.
- Train and compare multiple classification models.
- Optimize models using GridSearchCV and stratified cross-validation.
- Analyze bias and variance using learning curves.
- Create domain-based customer features.
- Evaluate feature selection using SelectKBest.
- Handle class imbalance during ensemble training.
- Save and reload complete trained pipelines for future predictions.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training, selection, tuning, and evaluation |
| XGBoost | Gradient-boosted ensemble model |
| Matplotlib | Learning-curve visualization |
| Joblib | Model serialization |
| Git and GitHub | Version control |

---

## Dataset Information

**Dataset:** Telco Customer Churn  
**Source:** Kaggle - Blastchar Telco Customer Churn Dataset

The dataset contains customer information from a telecommunications company.

| Property | Value |
|---|---:|
| Total records | 7,043 |
| Total columns in the raw dataset | 21 |
| Model input features after removing `customerID` and `Churn` | 19 |
| Numerical input features | 4 |
| Categorical input features | 15 |
| Target variable | Churn |
| Missing `TotalCharges` values after conversion | 11 |
| Duplicate records | 0 |

### Target Distribution

The target variable is `Churn`:

- `No` -> 0
- `Yes` -> 1

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is imbalanced, with fewer customers leaving than staying. Recall is therefore an important evaluation metric because identifying customers who may churn is a primary objective.

---

## Project Structure

```text
RETAILMIND-AI-MAIN-doing/
|
|-- data/
|   `-- raw/
|       `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|
|-- notebooks/
|   `-- churn_analysis.py
|
|-- src/
|   |-- __init__.py
|   |-- Day1_train.py
|   |-- Day1_predict.py
|   |-- Day2_compare_models.py
|   |-- Day2_learning_analysis.py
|   |-- Day3_preprocessing.py
|   |-- Day3_tune_models.py
|   |-- Day4_feature_engineering.py
|   |-- Day4_feature_selection.py
|   |-- Day4_train_ensembles.py
|   `-- Day4_test_production_model.py
|
|-- models/
|   |-- churn_pipeline.joblib
|   |-- churn_comparison_champion.joblib
|   |-- churn_v3.joblib
|   |-- churn_v4.joblib
|   |-- day2_model_comparison.csv
|   |-- day3_tuning_results.csv
|   `-- learning_curve.png
|
|-- README.md
|-- requirements.txt
`-- .gitignore
```

> The model files, result files, and learning-curve image are generated after their corresponding scripts run successfully.

---

# Day 1 - Exploratory Data Analysis and Baseline Model

## 1. Exploratory Data Analysis

The initial analysis examined:

- Dataset dimensions and structure.
- Numerical and categorical columns.
- Missing values.
- Duplicate records.
- Customer churn distribution.
- Data types and feature distributions.

### Data Cleaning

The `TotalCharges` column was initially stored as a string and was converted to a numerical column:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

This conversion identified 11 missing values. The `customerID` column was excluded from training because it is an identifier rather than a meaningful predictive feature.

## 2. Data Preprocessing

A reusable preprocessing pipeline was developed in `src/Day3_preprocessing.py` and reused by the training scripts.

### Numerical Features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Operations:

1. Missing-value imputation using the median.
2. Feature scaling using `StandardScaler`.

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

Operations:

1. Missing-value imputation using the most frequent value.
2. One-hot encoding using `OneHotEncoder`.

### Train-Test Split

| Dataset | Records |
|---|---:|
| Training | 5,634 |
| Testing | 1,409 |
| Total | 7,043 |

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification preserves approximately the same target-class distribution in both sets.

## 3. Baseline Model - Logistic Regression

A Logistic Regression classifier was trained using the shared preprocessing pipeline.

### Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.19% |

### Confusion Matrix

```text
[[926 109]
 [165 209]]
```

The baseline model was saved as:

```text
models/churn_pipeline.joblib
```

## 4. Prediction Pipeline

A reusable prediction function was implemented in `src/Day1_predict.py`.

The function:

- Loads the saved baseline pipeline.
- Accepts customer information.
- Predicts whether the customer is likely to churn.
- Returns the predicted class and churn probability.

Example output:

```text
Model loaded successfully!

========== CUSTOMER PREDICTION ==========
{'prediction': 'Yes', 'churn_probability': 0.6801}
```

The predicted probability is a model estimate, not a guarantee of actual customer behaviour.

---

# Day 2 - Model Comparison

## 1. Objective

Day 2 evaluated multiple classification algorithms using the same training and testing datasets and the same reusable preprocessing strategy.

Models compared:

- Logistic Regression baseline from Day 1
- Decision Tree
- Random Forest
- Support Vector Machine

## 2. Model Evaluation

The models were evaluated using accuracy, precision, recall, ROC-AUC, and confusion matrices.

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 65.73% |
| Random Forest | 77.79% | 60.34% | 47.59% | 81.62% |
| SVM | 79.13% | 64.18% | 48.40% | 79.05% |

## 3. Day 2 Champion Selection

Among the three newly evaluated Day 2 algorithms, Decision Tree achieved the highest churn recall at 50.53% and was saved as the Day 2 comparison champion.

Logistic Regression still remained stronger overall than the new Day 2 candidates, with 55.88% recall and 84.19% ROC-AUC.

Saved artifacts:

```text
models/churn_comparison_champion.joblib
models/day2_model_comparison.csv
```

---

# Day 3 - Hyperparameter Tuning and Model Optimization

## 1. Objective

Day 3 focused on improving model performance through hyperparameter tuning and cross-validation.

Main activities:

- Defined hyperparameter grids.
- Applied `GridSearchCV`.
- Performed five-fold stratified cross-validation.
- Compared training and validation recall.
- Evaluated tuned models on the holdout test set.
- Selected and saved a tuned champion model.
- Analyzed bias and variance using a learning curve.

## 2. GridSearchCV

```python
GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
```

### Cross-Validation Strategy

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Recall was selected as the main cross-validation score because detecting customers who may churn is the primary project objective.

## 3. Hyperparameter Grids

### Decision Tree

```python
{
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__min_samples_leaf": [1, 5, 10]
}
```

### Random Forest

```python
{
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_leaf": [1, 5],
    "classifier__max_features": ["sqrt"]
}
```

### Support Vector Machine

SVM used `CalibratedClassifierCV` with an underlying SVC estimator.

```python
{
    "classifier__estimator__C": [0.1, 1, 10],
    "classifier__estimator__kernel": ["linear", "rbf"],
    "classifier__estimator__gamma": ["scale", "auto"]
}
```

## 4. Best Hyperparameters

### Decision Tree

```python
{
    "classifier__max_depth": 5,
    "classifier__min_samples_leaf": 10,
    "classifier__min_samples_split": 2
}
```

### Random Forest

```python
{
    "classifier__max_depth": 10,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 5,
    "classifier__n_estimators": 100
}
```

### SVM

```python
{
    "classifier__estimator__C": 0.1,
    "classifier__estimator__gamma": "scale",
    "classifier__estimator__kernel": "linear"
}
```

## 5. Tuned Model Comparison

| Model | Best CV Recall | Training CV Recall | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.5144 | 0.5530 | 0.7984 | 0.6347 | **0.5668** | 0.8282 |
| Random Forest | **0.5157** | 0.6077 | **0.8084** | **0.6793** | 0.5267 | **0.8418** |
| SVM | 0.5144 | 0.5189 | 0.7970 | 0.6467 | 0.5187 | 0.8287 |

## 6. Training vs Validation Performance

| Model | Training Recall | Validation Recall | Difference |
|---|---:|---:|---:|
| Decision Tree | 0.5530 | 0.5144 | 0.0386 |
| Random Forest | 0.6077 | 0.5157 | 0.0920 |
| SVM | 0.5189 | 0.5144 | 0.0045 |

### Observations

- Decision Tree showed a relatively small training-validation gap.
- Random Forest showed the largest gap, suggesting possible overfitting.
- SVM had almost identical training and validation recall, indicating consistent but modest performance.

## 7. Tuned Champion Model

Random Forest was selected as the Day 3 tuned champion because it achieved the highest cross-validation recall. It also achieved the highest test ROC-AUC among the tuned candidates.

### Champion Results

- Best CV Recall: 51.57%
- Test Accuracy: 80.84%
- Test Precision: 67.93%
- Test Recall: 52.67%
- Test ROC-AUC: 84.18%

Saved artifacts:

```text
models/churn_v3.joblib
models/day3_tuning_results.csv
```

> Decision Tree achieved the highest holdout test recall at 56.68%, while Random Forest achieved the highest cross-validation recall and test ROC-AUC. Champion selection was based on cross-validation performance rather than one holdout metric.

## 8. Learning Curve and Bias-Variance Analysis

The learning curve compared training recall and validation recall as the training-set size increased.

The analysis is implemented in:

```text
src/Day2_learning_analysis.py
```

### Learning-Curve Results

| Training Size | Training Recall | Validation Recall | Gap |
|---:|---:|---:|---:|
| 450 | 0.6918 | 0.4963 | 0.1955 |
| 1,352 | 0.6410 | 0.4809 | 0.1601 |
| 2,253 | 0.6388 | 0.5124 | 0.1264 |
| 3,154 | 0.6454 | 0.5351 | 0.1103 |
| 4,507 | 0.6080 | 0.5151 | 0.0929 |

The train-validation gap generally became smaller as more training data was used. Validation recall improved from 0.4963 at 450 samples to 0.5151 at 4,507 samples, although the scores fluctuated across training sizes. The final remaining gap suggests some variance is still present.

Generated visualization:

```text
models/learning_curve.png
```

---

# Day 4 - Feature Engineering and Ensemble Learning

## 1. Feature Engineering

A reusable Scikit-learn transformer named `CustomerFeatureEngineer` was implemented in `src/Day4_feature_engineering.py`.

Six new features were created:

| Feature | Description |
|---|---|
| `tenure_group` | Groups customers by tenure |
| `avg_monthly_charge` | Calculates the average historical monthly charge |
| `num_services` | Counts subscribed services |
| `is_month_to_month` | Identifies month-to-month contracts |
| `has_support_service` | Identifies support or security subscriptions |
| `has_online_service` | Identifies internet subscribers |

The model input dataset increased from 19 features to 25 features after feature engineering.

## 2. Feature Selection

`SelectKBest` with `f_classif` was used to test whether reducing the transformed feature set improved model performance.

| Property | Result |
|---|---:|
| Original transformed features | 53 |
| Selected features | 20 |
| Feature reduction | 62.26% |

### Performance Comparison

| Metric | Without Selection | With Selection |
|---|---:|---:|
| Accuracy | 79.91% | 80.06% |
| Precision | 65.12% | 65.98% |
| Recall | 52.41% | 51.34% |
| ROC-AUC | 84.16% | 83.69% |

Feature selection slightly improved accuracy and precision but reduced recall and ROC-AUC. Because churn detection prioritizes recall, all 53 transformed features were retained for ensemble training.

## 3. Ensemble Learning

Two ensemble algorithms were trained and compared:

- Random Forest using bagging.
- XGBoost using gradient boosting.

Both models used the same train-test split, engineered features, and preprocessing pipeline.

Class imbalance was addressed with:

- `class_weight="balanced"` for Random Forest.
- `scale_pos_weight` for XGBoost.

Training class distribution:

| Class | Records |
|---|---:|
| Non-churn | 4,139 |
| Churn | 1,495 |

The calculated XGBoost positive-class weight was approximately 2.77.

### Ensemble Model Comparison

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | **75.94%** | 75.73% |
| Precision | **53.18%** | 52.87% |
| Recall | 78.34% | **78.88%** |
| F1 Score | **63.35%** | 63.30% |
| ROC-AUC | 84.14% | **84.39%** |

## 4. Day 4 Champion Model

**Selected model: XGBoost**

XGBoost was selected because churn recall is the project's primary metric, and XGBoost achieved the higher recall and ROC-AUC among the two Day 4 ensemble candidates.

### Final Metrics

- Accuracy: 75.73%
- Precision: 52.87%
- Recall: 78.88%
- F1 Score: 63.30%
- ROC-AUC: 84.39%

The complete trained pipeline was saved as:

```text
models/churn_v4.joblib
```

## 5. Production Model Verification

The saved Day 4 pipeline was loaded independently and used to generate predictions for five customer records.

| Customer | Prediction | Churn Probability |
|---|---|---:|
| 1 | No | 2.34% |
| 2 | Yes | 93.30% |
| 3 | No | 21.59% |
| 4 | Yes | 54.68% |
| 5 | No | 1.21% |

The production verification completed successfully, confirming that the saved pipeline can reload and generate both churn predictions and probabilities.

## 6. Day 4 Conclusion

Day 4 introduced domain-based feature engineering, feature-selection experiments, class-imbalance handling, and ensemble learning.

XGBoost achieved 78.88% churn recall, a substantial improvement over the earlier Day 1 to Day 3 models. The final Day 4 pipeline was saved and successfully reloaded for production-style predictions.

---

# Installation and Execution

Run all commands from the project root directory:

```text
RETAILMIND-AI-MAIN-doing>
```

## 1. Clone the Repository

```bash
git clone <repository-url>
cd RETAILMIND-AI-MAIN-doing
```

Replace `<repository-url>` with the actual GitHub repository URL.

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Dataset Setup

Place the Kaggle dataset at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 5. Run Exploratory Data Analysis

```bash
python notebooks/churn_analysis.py
```

## 6. Run Day 1 Baseline Training

```bash
python -m src.Day1_train
```

## 7. Run Day 1 Prediction

```bash
python -m src.Day1_predict
```

## 8. Run Day 2 Model Comparison

```bash
python -m src.Day2_compare_models
```

## 9. Run Day 3 Hyperparameter Tuning

```bash
python -m src.Day3_tune_models
```

## 10. Run Learning-Curve Analysis

Run this after Day 3 tuning because the script loads `models/churn_v3.joblib`:

```bash
python -m src.Day2_learning_analysis
```

## 11. Run Day 4 Feature Engineering Test

```bash
python -m src.Day4_feature_engineering
```

## 12. Run Day 4 Feature Selection

```bash
python -m src.Day4_feature_selection
```

## 13. Train Day 4 Ensemble Models

```bash
python -m src.Day4_train_ensembles
```

## 14. Verify the Day 4 Production Model

Run this only after `models/churn_v4.joblib` has been created:

```bash
python -m src.Day4_test_production_model
```

> Use package-style execution with `python -m src.<module_name>` because the source files import modules through the `src` package.

---

# Project Progress

| Day | Activities | Status |
|---|---|---|
| Day 1 | EDA, preprocessing, Logistic Regression baseline, and prediction pipeline | Completed |
| Day 2 | Decision Tree, Random Forest, and SVM comparison | Completed |
| Day 3 | GridSearchCV, cross-validation, tuned champion selection, and learning-curve analysis | Completed |
| Day 4 | Feature engineering, feature-selection experiment, ensemble training, and production verification | Completed |

---

## Future Enhancements

- Analyze feature importance using XGBoost gain scores and SHAP values.
- Optimize the classification threshold to balance recall and precision.
- Perform additional XGBoost hyperparameter tuning.
- Add automated unit tests for preprocessing and feature engineering.
- Build a REST API using FastAPI.
- Develop a customer churn prediction dashboard.
- Monitor model performance and data drift after deployment.

---

## Author

**Aryan Mule**

GitHub: `AryanMule`

---

*RetailMind AI - Turning customer data into actionable churn predictions.*
