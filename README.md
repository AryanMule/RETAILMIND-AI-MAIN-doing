# RetailMind AI – Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using customer demographics, account information, subscribed services, and billing details.

The project follows a structured machine learning workflow, including Exploratory Data Analysis (EDA), data preprocessing, model training, model comparison, hyperparameter tuning, cross-validation, and model evaluation.

---

## Project Objectives

- Analyze customer data to identify churn patterns.
- Clean and preprocess raw customer information.
- Build reusable preprocessing pipelines.
- Train multiple machine learning classification models.
- Compare models using appropriate evaluation metrics.
- Optimize model performance using GridSearchCV.
- Analyze bias and variance using learning curves.
- Save trained models for future predictions.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib | Data visualization and learning curves |
| Joblib | Model serialization |
| Git & GitHub | Version control |

---

## Dataset Information

**Dataset:** Telco Customer Churn

**Source:** Kaggle – Blastchar Telco Customer Churn Dataset

The dataset contains customer information from a telecommunications company.

| Property | Value |
|---|---:|
| Total records | 7,043 |
| Total columns | 21 |
| Numerical features | 4 |
| Categorical features | 15 |
| Target variable | Churn |
| Missing TotalCharges values after conversion | 11 |
| Duplicate records | 0 |

### Target Distribution

The target variable is `Churn`.

- `No` → 0
- `Yes` → 1

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is imbalanced, with fewer customers leaving the company than staying.

Therefore, Recall is an important evaluation metric because identifying customers who may churn is a primary objective.

---

## Project Structure

```text
RetailMind_AI/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_analysis.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── compare_models.py
│   ├── tune_models.py
│   └── learning_analysis.py
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── churn_comparison_champion.joblib
│   ├── churn_v3.joblib
│   ├── day2_model_comparison.csv
│   ├── day3_tuning_results.csv
│   └── learning_curve.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

*Note: The learning curve image is generated when the learning analysis script runs successfully.*

---

# Day 1 – Exploratory Data Analysis and Baseline Model

## 1. Exploratory Data Analysis

Performed initial dataset analysis to understand:

- Dataset dimensions and structure.
- Numerical and categorical columns.
- Missing values.
- Duplicate records.
- Customer churn distribution.
- Data types and feature distributions.

### Data Cleaning

The `TotalCharges` column was initially stored as a string.

It was converted into a numerical column using:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

This conversion identified 11 missing values.

The `customerID` column was excluded from model training because it is an identifier rather than a meaningful predictive feature.

## 2. Data Preprocessing

A reusable preprocessing pipeline was developed.

### Numerical Features

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

Operations:

1. Missing value imputation using median.
2. Feature scaling using StandardScaler.

### Categorical Features

- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

Operations:

1. Missing value imputation using most frequent value.
2. One-hot encoding using OneHotEncoder.

### Train-Test Split

| Dataset | Records |
|---|---:|
| Training | 5,634 |
| Testing | 1,409 |
| Total | 7,043 |

Configuration:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification preserves the approximate target-class distribution in both sets.

## 3. Baseline Model – Logistic Regression

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

A reusable prediction function was implemented in `src/predict.py`.

The function:

- Loads the saved model.
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

# Day 2 – Model Comparison

## 1. Objective

The objective of Day 2 was to evaluate different classification algorithms using identical training and testing datasets.

The following models were trained:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

All models used the same reusable preprocessing strategy.

## 2. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 65.73% |
| Random Forest | 77.79% | 60.34% | 47.59% | 81.62% |
| SVM | 79.13% | 64.18% | 48.40% | 79.05% |

## 3. Day 2 Champion Selection

Among the three newly evaluated algorithms, Decision Tree achieved the highest churn recall of 50.53%.

However, Logistic Regression remained the strongest overall model compared with the Day 2 candidates, achieving 55.88% recall and 84.19% ROC-AUC.

The Decision Tree comparison champion was saved as:

```text
models/churn_comparison_champion.joblib
```

The comparison results were saved as:

```text
models/day2_model_comparison.csv
```

---

# Day 3 – Hyperparameter Tuning and Model Optimization

## 1. Objective

Day 3 focused on improving model performance through hyperparameter tuning and cross-validation.

The main activities included:

- Defining hyperparameter grids.
- Applying GridSearchCV.
- Performing 5-fold stratified cross-validation.
- Comparing training and validation scores.
- Evaluating tuned models on the holdout test set.
- Selecting and saving a tuned champion model.
- Analyzing bias and variance using learning curves.

## 2. GridSearchCV

GridSearchCV was used to find suitable hyperparameter combinations for each candidate model.

Configuration:

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

StratifiedKFold was configured with:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Each model was trained and evaluated across five folds.

Recall was selected as the primary cross-validation scoring metric because detecting customers who may churn is the main project objective.

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

SVM used CalibratedClassifierCV with an underlying SVC estimator.

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

Training and validation recall were compared to understand possible overfitting.

| Model | Training Recall | Validation Recall | Difference |
|---|---:|---:|---:|
| Decision Tree | 0.5530 | 0.5144 | 0.0386 |
| Random Forest | 0.6077 | 0.5157 | 0.0920 |
| SVM | 0.5189 | 0.5144 | 0.0045 |

### Observations

- Decision Tree shows a relatively small training-validation gap.
- Random Forest shows a larger gap, suggesting possible overfitting.
- SVM has very similar training and validation recall, indicating more consistent but comparatively modest performance.

These differences are useful indicators, but they should be interpreted together with learning curves and other evaluation metrics.

## 7. Tuned Champion Model

Random Forest was selected as the Day 3 tuned champion based on its highest cross-validation recall.

It also achieved the highest ROC-AUC among the three tuned models.

### Important Results

- Best CV Recall: 51.57%
- Test Accuracy: 80.84%
- Test Precision: 67.93%
- Test Recall: 52.67%
- Test ROC-AUC: 84.18%

The tuned champion was saved as:

```text
models/churn_v3.joblib
```

The tuning results were exported to:

```text
models/day3_tuning_results.csv
```

**Metric-specific observation:** Decision Tree achieved the highest holdout test recall (56.68%), while Random Forest achieved the highest cross-validation recall and test ROC-AUC. The selected champion is based on cross-validation performance, not on having the highest score for every metric.

---

## 8. Learning Curve and Bias-Variance Analysis

The learning curve is used to understand how model performance changes as the training dataset size increases.

It compares:

- Training Recall
- Validation Recall

The analysis is implemented in:

```text
src/learning_analysis.py
```

The script uses Scikit-learn's `learning_curve()` function and generates a visualization.

### Learning Curve Interpretation

| Observation | Possible Interpretation |
|---|---|
| High training score, low validation score | Overfitting / high variance |
| Low training score, low validation score | Underfitting / high bias |
| Both scores improve as data increases | More data may help |
| Both scores stabilize close together | Model performance may have converged |

**Learning curve results:** To be added after executing `python src/learning_analysis.py` and reviewing the generated plot.

Expected visualization:

```text
models/learning_curve.png
```

---
## Day 4 – Feature Engineering and Ensemble Learning

## 1. Feature Engineering

Implemented a reusable Scikit-learn transformer named `CustomerFeatureEngineer`.

Six new features were created:

| Feature | Description |
|---|---|
| tenure_group | Groups customers by tenure |
| avg_monthly_charge | Average historical monthly charge |
| num_services | Number of subscribed services |
| is_month_to_month | Identifies monthly contracts |
| has_support_service | Identifies support/security subscriptions |
| has_online_service | Identifies internet subscribers |

The original dataset contained 19 features. After feature engineering, it contained 25 features.

## 2. Feature Selection

Used Scikit-learn's `SelectKBest` with `f_classif` to select informative features.

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

### Observation

Feature selection slightly improved accuracy and precision but reduced recall and ROC-AUC.

Since churn detection is the primary objective, all 53 transformed features were retained for ensemble training.

## 3. Ensemble Learning

Implemented and compared two ensemble algorithms:

- Random Forest – Bagging
- XGBoost – Gradient Boosting

Both models used the same training and testing split, along with the engineered features and preprocessing pipeline.

Class imbalance was addressed using:
- `class_weight="balanced"` in Random Forest.
- `scale_pos_weight` in XGBoost.

The calculated XGBoost positive-class weight was approximately 2.77.

### Ensemble Model Comparison

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 75.94% | 75.87% |
| Precision | 53.18% | 53.06% |
| Recall | 78.34% | 78.88% |
| F1 Score | 63.35% | 63.44% |
| ROC-AUC | 84.14% | 84.42% |

# 4. Day 4 Champion Model

**Selected model: XGBoost**

XGBoost was selected because it achieved the highest recall, F1-score, and ROC-AUC among the two Day 4 ensemble candidates.

### Final Metrics

- Accuracy: 75.87%
- Precision: 53.06%
- Recall: 78.88%
- F1 Score: 63.44%
- ROC-AUC: 84.42%

The complete trained pipeline was saved as:

`models/churn_v4.joblib`

# 5. Production Model Verification

Loaded the saved model and generated predictions for five customer records.

| Customer | Prediction | Churn Probability |
|---|---|---:|
| 1 | No | 2.09% |
| 2 | Yes | 94.56% |
| 3 | No | 25.12% |
| 4 | Yes | 56.77% |
| 5 | No | 1.63% |

The saved model successfully loaded and generated predictions using the complete preprocessing and feature-engineering pipeline.

## 6. Day 4 Conclusion

Day 4 introduced domain-based feature engineering, feature selection experiments, and ensemble learning.

XGBoost achieved the highest churn recall among the Day 4 ensemble candidates, reaching 78.88%.

The model is saved as a reusable pipeline for subsequent prediction and deployment work.

## Day 5 - Neural Networks and Deep Learning

### 1. MNIST Neural Network Demo
- Loaded the MNIST handwritten digit dataset.
- Normalized image pixel values.
- Built a neural network using Keras Sequential API.
- Used Dense layers with ReLU and Softmax activation.
- Trained and evaluated the model.

### 2. Customer Churn ANN
- Reused Day 4 feature engineering.
- Applied numerical scaling and categorical encoding.
- Built an ANN using TensorFlow/Keras.
- Used ReLU hidden layers and Sigmoid output.
- Used Binary Crossentropy loss and Adam optimizer.
- Applied Dropout and EarlyStopping.
- Used class weights to handle class imbalance.

### 3. Model Comparison
Compared the Day 5 ANN with the Day 4 XGBoost champion using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 4. Saved Artifacts
- churn_v5_ann.keras
- day5_preprocessing.joblib
- day5_training_history.npz
- day5_model_comparison.csv

# Installation and Execution

## 1. Clone the Repository

```bash
git clone https://github.com/Shubh823/RetailMind_AI.git

cd RetailMind_AI
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Dataset Setup

Place the Kaggle dataset inside:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 5. Run Exploratory Data Analysis

```bash
python notebooks/churn_analysis.py
```

## 6. Run Baseline Model

```bash
python src/train.py
```

## 7. Run Prediction

```bash
python src/predict.py
```

## 8. Compare Models

```bash
python src/compare_models.py
```

## 9. Run Hyperparameter Tuning

```bash
python src/tune_models.py
```

## 10. Run Learning Curve Analysis

```bash
python src/learning_analysis.py
```

---

# Project Progress

| Day | Activities | Status |
|---|---|---|
| Day 1 | EDA, preprocessing, Logistic Regression baseline, prediction pipeline | Completed |
| Day 2 | Decision Tree, Random Forest, SVM model comparison | Completed |
| Day 3 | GridSearchCV, hyperparameter tuning, champion selection | Completed |
| Day 3 | Learning curve and bias-variance analysis | Pending final visualization review |

---

## Future Enhancements

- Experiment with additional classification algorithms.
- Improve churn recall through class weighting and threshold optimization.
- Perform feature importance analysis.
- Experiment with ensemble learning.
- Build a REST API using FastAPI.
- Develop a customer churn prediction dashboard.
- Monitor model performance and data drift.

---

## Author

**Shubham Patil**

GitHub: [Shubh823](https://github.com/Shubh823)

---

*RetailMind AI – Turning customer data into actionable churn predictions.*