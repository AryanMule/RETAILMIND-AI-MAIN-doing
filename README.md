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


## Day 3: Tuning and bias-variance inspection

Selection: highest mean churn recall from three-fold stratified CV on the training partition. Exact ties retain the first candidate.

| Model | Train recall | CV recall | CV std | Recall gap | Train accuracy | CV accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5558 | 0.5525 | 0.0121 | 0.0033 | 0.8073 | 0.8058 |
| SVM | 0.7120 | 0.5291 | 0.0233 | 0.1829 | 0.8734 | 0.7863 |
| Decision Tree | 0.5565 | 0.5010 | 0.0429 | 0.0555 | 0.8142 | 0.7898 |
| Random Forest | 0.6241 | 0.4923 | 0.0225 | 0.1318 | 0.8611 | 0.8023 |

### Best parameters per model

- Logistic Regression: `{'classifier__C': 10.0}`
- Decision Tree: `{'classifier__max_depth': 6, 'classifier__min_samples_leaf': 1}`
- Random Forest: `{'classifier__max_depth': None, 'classifier__min_samples_leaf': 5}`
- SVM: `{'classifier__C': 10.0, 'classifier__gamma': 'scale'}`

### Locked champion: Logistic Regression

Parameters: `{'classifier__C': 10.0}`

Mean CV recall: 0.5525

Saved pipeline: `models/churn_v3.joblib`

### Champion learning curve

| Training rows per fold | Train recall | Validation recall | Validation std |
| ---: | ---: | ---: | ---: |
| 1126 | 0.5852 | 0.5672 | 0.0120 |
| 2253 | 0.5712 | 0.5498 | 0.0307 |
| 3756 | 0.5558 | 0.5525 | 0.0121 |

### Hold-out evaluation

| Metric | Hold-out score |
| --- | ---: |
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| ROC-AUC | 0.8412 |

### Interpretation notes

A large train-versus-validation gap suggests variance. Low, converged scores may suggest bias, but recall alone does not establish underfitting. Check accuracy and precision as well.

Rising validation recall with more training data suggests that more data may help. Flat, low scores with a small gap suggest reviewing model capacity or feature signal. A persistent large gap suggests considering stronger regularisation. These are diagnostic indications, not automatic conclusions.

The hold-out was already inspected on previous days, so it is not a fresh independent final test set. Learning curves also reuse the tuning data and are diagnostic, not an unbiased final performance estimate.

### Day 3 diagnosis

Logistic Regression with C=10.0 was selected because it achieved the highest mean three-fold CV churn recall: 0.5525.

Its training recall was 0.5558, giving a small train-versus-CV recall gap of 0.0033. Training accuracy of 0.8073 and CV accuracy of 0.8058 were also close. These results suggest little overfitting under this evaluation, but do not establish underfitting.

SVM and Random Forest showed larger recall gaps of 0.1829 and 0.1318 respectively, suggesting overfitting.

The champion's learning curve showed validation recall of 0.5672, 0.5498, and 0.5525 as training size increased. There was no consistent upward trend, and training and validation recall converged. This does not provide clear evidence that more data alone would substantially improve recall. Reviewing feature signal or model capacity is a reasonable later investigation. No additional changes were implemented today.

The locked champion achieved hold-out accuracy of 0.8055, precision of 0.6572, recall of 0.5588, and ROC-AUC of 0.8412. The hold-out set was inspected on previous days, so these results are not a fresh independent final evaluation.

Saved pipeline: models/churn_v3.joblib
