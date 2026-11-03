RetailMind AI
RetailMind AI is an incremental retail AI platform built over 35 days.

Day 1
Build a customer-churn baseline using pandas, scikit-learn, and joblib.

Dataset
Place the original Telco Customer Churn CSV at:

data/WA_Fn-UseC_-Telco-Customer-Churn.csv

The dataset is kept locally and is not committed.

Train
Run from the project root:

python src/train.py

Predict
Run from the project root:

python src/predict.py

Model
Algorithm: Logistic Regression
Test split: 20%, stratified by Churn
Random seed: 42
Target: 0 = No churn, 1 = Churn
Saved pipeline: models/churn_day1.joblib
Training appends the measured baseline metrics below.

### Day 1 baseline results

Model: Logistic Regression

Split: 80/20, stratified, random_state=42

| Metric | Test score |
| --- | ---: |
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| ROC-AUC | 0.8419 |
