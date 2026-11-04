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


## Day 2: Model comparison

Reused Day 1 preprocessing configuration. All models used the same 80/20 stratified split with random_state=42.

| Model | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7289 | 0.4896 | 0.5053 | 0.6573 |
| SVM | 0.7913 | 0.6418 | 0.4840 | 0.7905 |
| Random Forest | 0.7779 | 0.6034 | 0.4759 | 0.8162 |

Decision Tree was selected by highest churn-class recall (0.5053), with ROC-AUC (0.6573) used to break recall ties. If both metrics tie exactly, the first candidate in the listed order is retained.

Saved pipeline: `models/churn_day2.joblib`

This champion is the best of the three Day 2 candidates under the stated selection rule on this hold-out set. It is not necessarily better than the Day 1 baseline.
