# RetailMind AI - Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using customer demographics, account information, subscribed services, and billing details.

The project follows a structured workflow covering exploratory data analysis, preprocessing, baseline training, model comparison, hyperparameter tuning, cross-validation, learning-curve analysis, feature engineering, feature selection, ensemble learning, production model verification, neural-network experimentation, CNN image classification, transfer learning with VGG16, Grad-CAM explainability, weakly supervised product segmentation, background removal, catalog-image generation, LSTM-based product demand forecasting.

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
- Build and evaluate artificial neural networks with TensorFlow and Keras.
- Build a CNN-based fashion product image classifier.
- Upgrade the image classifier using VGG16 transfer learning and fine-tuning.
- Compare the Day 6 CNN with the Day 7 VGG16 classifier.
- Explain image-classification predictions using Grad-CAM.
- Estimate product foreground masks using Grad-CAM and OpenCV GrabCut.
- Generate transparent-background and white-background catalog images.
- Localize the main product using a mask-derived bounding box.
- Forecast future product demand from historical sales data.
- Build time-series look-back windows for next-period demand prediction.
- Train an LSTM model for store-product demand forecasting.
- Evaluate demand forecasts using MAE and RMSE.
- Generate demand predictions that can later support inventory planning.
- Extend demand forecasting from next-period prediction to a 14-day multi-step Seq2Seq forecast.
- Use Attention to connect future forecast steps with relevant historical sales timesteps.
- Generate multi-period demand forecasts for procurement and inventory planning.
- Save and reload trained pipelines, preprocessing objects, and neural-network models.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training, selection, tuning, scaling, and evaluation |
| XGBoost | Gradient-boosted ensemble model |
| TensorFlow and Keras | ANN, CNN, transfer-learning, and LSTM development |
| LSTM | Sequential demand forecasting from historical sales |
| Seq2Seq | Encoder-decoder multi-step demand forecasting |
| Attention | Historical-timestep weighting for multi-step forecasts |
| VGG16 | ImageNet-pretrained transfer-learning backbone |
| OpenCV | Image loading, mask refinement, background removal, and output generation |
| Grad-CAM | Visual explanation of VGG16 classification decisions |
| Pillow | Image loading and validation |
| Matplotlib | Learning-curve and demand-forecast visualization |
| Joblib | Model and preprocessing-object serialization |
| Git and GitHub | Version control |

---

## Dataset Information

### Customer Churn Dataset

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

#### Target Distribution

- `No` -> 0
- `Yes` -> 1

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is imbalanced, with fewer customers leaving than staying. Recall is therefore an important evaluation metric because identifying customers who may churn is a primary objective.

### Computer-Vision Dataset

Days 6, 7, and 8 require the Kaggle `paramaggarwal/fashion-product-images-small` dataset.

```text
data/fashion-product-images-small/
|-- styles.csv
`-- images/
    |-- <product-id>.jpg
    `-- ...
```

### Demand Forecasting Dataset

Days 9 and 10 use the historical store-item sales dataset for demand forecasting. Day 9 performs next-period forecasting, while Day 10 reuses the same dataset for multi-step Seq2Seq forecasting.

Expected input file:

```text
data/train.csv
```

Expected columns:

| Column | Purpose |
|---|---|
| `date` | Date of the sales observation |
| `store` | Store identifier |
| `item` | Product/item identifier |
| `sales` | Historical product demand |

Current Day 9 configuration:

```text
Store: 1
Item: 1
Look-back window: 30 days
Test period: Most recent 28 days
```

---

## Project Structure

```text
RETAILMIND-AI-MAIN-doing/
|
|-- data/
|   |-- raw/
|   |   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|   |-- fashion-product-images-small/        # Required for Days 6-8
|   |   |-- styles.csv
|   |   `-- images/
|   `-- train.csv                            # Day 9 demand dataset
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
|   |-- Day4_test_production_model.py
|   |-- Day5_mnist_demo.py
|   |-- Day5_train_ann.py
|   |-- Day5_compare_models.py
|   |-- Day6_fashion_mnist.py
|   |-- Day6_image_pipeline.py
|   |-- Day6_train_cnn.py
|   |-- Day6_evaluate_cnn.py
|   |-- Day7_transfer_learning.py
|   |-- Day8_segmentation_gradcam.py
|   |-- Day9_demand_forecasting.py
|   `-- Day10_seq2seq_attention.py
|
|-- outputs/
|   |-- day8/                                # Day 8 images and prediction JSON
|   |-- day9/                                # Day 9 predictions and forecast plot
|   `-- day10/                               # Day 10 forecast and attention outputs
|
|-- models/
|   |-- churn_pipeline.joblib
|   |-- churn_comparison_champion.joblib
|   |-- churn_v3.joblib
|   |-- churn_v4.joblib
|   |-- churn_v5_ann.keras
|   |-- day2_model_comparison.csv
|   |-- day3_tuning_results.csv
|   |-- day4_model_comparison.csv
|   |-- day5_preprocessing.joblib
|   |-- day5_training_history.npz
|   |-- day5_model_comparison.csv
|   |-- mnist_demo.keras
|   |-- fashion_mnist_cnn.keras
|   |-- fashion_cnn.keras                   # Expected after Day 6 completion
|   |-- fashion_vgg16.keras                 # Expected after Day 7 completion
|   |-- demand_forecaster.keras             # Expected after Day 9 execution
|   |-- day9_sales_scaler.joblib            # Recommended Day 9 scaler artifact
|   |-- day10_seq2seq_attention.keras       # Expected after Day 10 execution
|   |-- day10_sales_scaler.joblib           # Recommended Day 10 scaler artifact
|   `-- learning_curve.png
|
|-- README.md
|-- requirements.txt
`-- .gitignore
```

> Model files, result files, and visualizations are generated only after their corresponding scripts run successfully.

---

# Day 1 - Exploratory Data Analysis and Baseline Model

## 1. Exploratory Data Analysis

The initial analysis examined dataset dimensions, column types, missing values, duplicate records, feature distributions, and customer churn distribution.

`TotalCharges` was converted from text to a numeric column:

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
```

The conversion identified 11 missing values. The `customerID` column was excluded because it is an identifier rather than a predictive feature.

## 2. Data Preprocessing

Numerical features:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numerical operations:

1. Median missing-value imputation.
2. StandardScaler feature scaling.

Categorical features were processed with most-frequent imputation and one-hot encoding. The stratified split produced 5,634 training records and 1,409 testing records.

## 3. Logistic Regression Baseline

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.19% |

Confusion matrix:

```text
[[926 109]
 [165 209]]
```

Saved model:

```text
models/churn_pipeline.joblib
```

`src/Day1_predict.py` reloads the baseline pipeline and returns the predicted churn class and probability.

---

# Day 2 - Model Comparison

Day 2 compared Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine using the same preprocessing strategy and holdout dataset.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 84.19% |
| Decision Tree | 72.89% | 48.96% | 50.53% | 65.73% |
| Random Forest | 77.79% | 60.34% | 47.59% | 81.62% |
| SVM | 79.13% | 64.18% | 48.40% | 79.05% |

Among the three newly evaluated Day 2 algorithms, Decision Tree achieved the highest churn recall and was saved as the Day 2 comparison champion. Logistic Regression remained stronger overall.

```text
models/churn_comparison_champion.joblib
models/day2_model_comparison.csv
```

---

# Day 3 - Hyperparameter Tuning and Model Optimization

Day 3 used GridSearchCV, five-fold StratifiedKFold cross-validation, recall-based scoring, training-versus-validation comparisons, holdout evaluation, and learning-curve analysis.

| Model | Best CV Recall | Training CV Recall | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.5144 | 0.5530 | 0.7984 | 0.6347 | **0.5668** | 0.8282 |
| Random Forest | **0.5157** | 0.6077 | **0.8084** | **0.6793** | 0.5267 | **0.8418** |
| SVM | 0.5144 | 0.5189 | 0.7970 | 0.6467 | 0.5187 | 0.8287 |

Random Forest was selected because it achieved the highest cross-validation recall and highest test ROC-AUC among the tuned candidates.

```text
models/churn_v3.joblib
models/day3_tuning_results.csv
models/learning_curve.png
```

The final learning-curve training-validation recall gap was 0.0929, indicating that some variance remained.

---

# Day 4 - Feature Engineering and Ensemble Learning

## Feature Engineering

`CustomerFeatureEngineer` introduced:

- `tenure_group`
- `avg_monthly_charge`
- `num_services`
- `is_month_to_month`
- `has_support_service`
- `has_online_service`

The input increased from 19 to 25 features. After preprocessing, 53 transformed features were available.

## Feature Selection

SelectKBest with `f_classif` reduced the transformed features from 53 to 20. Because selection reduced recall and ROC-AUC, all 53 transformed features were retained.

## Ensemble Comparison

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | **75.94%** | 75.73% |
| Precision | **53.18%** | 52.87% |
| Recall | 78.34% | **78.88%** |
| F1 Score | **63.35%** | 63.30% |
| ROC-AUC | 84.14% | **84.39%** |

XGBoost was selected because churn recall was the primary metric and it achieved higher recall and ROC-AUC.

```text
models/churn_v4.joblib
```

The saved model was independently reloaded and verified on five customer records.

---

# Day 5 - Neural Networks and Deep Learning

## MNIST ANN Demo

The MNIST ANN contained 109,386 trainable parameters and achieved 97.76% test accuracy after early stopping.

## Customer Churn ANN

The churn ANN used 53 processed inputs, dense ReLU layers, dropout, sigmoid output, binary cross-entropy, Adam, early stopping, and class weights.

| Metric | Day 5 ANN |
|---|---:|
| Accuracy | 73.17% |
| Precision | 49.66% |
| Recall | 78.88% |
| F1 Score | 60.95% |
| ROC-AUC | 83.64% |

## XGBoost vs ANN

| Metric | Day 4 XGBoost | Day 5 ANN |
|---|---:|---:|
| Accuracy | **75.73%** | 73.17% |
| Precision | **52.87%** | 49.66% |
| Recall | **78.88%** | **78.88%** |
| F1 Score | **63.30%** | 60.95% |
| ROC-AUC | **84.39%** | 83.64% |

XGBoost remained the overall churn champion.

```text
models/churn_v5_ann.keras
models/day5_preprocessing.joblib
models/day5_training_history.npz
models/day5_model_comparison.csv
models/mnist_demo.keras
```

---

# Day 6 - CNN Image Classification

## Objective

Day 6 introduces a computer-vision module that classifies fashion product images. The workflow starts with a Fashion-MNIST CNN prototype and then moves to the real Fashion Product Images dataset with augmentation, CNN training, confusion-matrix evaluation, and model persistence.

## Implemented Files

| File | Purpose | Current status |
|---|---|---|
| `src/Day6_fashion_mnist.py` | Fashion-MNIST CNN prototype | Model artifact exists; final metric not documented |
| `src/Day6_image_pipeline.py` | Metadata loading, image matching, splitting, TensorFlow datasets, and augmentation | Implemented but blocked by missing dataset |
| `src/Day6_train_cnn.py` | Train the real fashion-product CNN | Pending dataset and pipeline success |
| `src/Day6_evaluate_cnn.py` | Load and evaluate the trained CNN | Pending trained real-product model |

Required data:

```text
data/fashion-product-images-small/styles.csv
data/fashion-product-images-small/images/
```

Expected artifacts:

```text
models/fashion_cnn.keras
models/day6_training_history.npz
models/day6_class_names.json
models/day6_confusion_matrix.png
models/day6_evaluation.csv
```

**Overall status: Pending because the Fashion Product Images dataset is not installed locally.**

---

# Day 7 - Transfer Learning and Object Detection

## Objective

Day 7 upgrades the Day 6 computer-vision module using transfer learning. An ImageNet-pretrained VGG16 network is adapted to classify Fashion Product Images categories, fine-tuned, evaluated, and compared with the Day 6 CNN.

The roadmap also includes YOLO-based product detection. The current implementation covers VGG16 transfer learning only.

## Files and Status

| File | Purpose | Current status |
|---|---|---|
| `src/Day7_transfer_learning.py` | Train and fine-tune VGG16 and compare it with Day 6 CNN | Implemented; execution pending |
| `src/Day7_yolo_detection.py` | YOLO product detection | Not implemented |

The script loads and validates images, encodes `masterCategory`, creates stratified splits, applies VGG16 preprocessing, trains a custom classification head, fine-tunes the top four VGG16 layers using a learning rate of `0.00001`, evaluates the model, compares it with `models/fashion_cnn.keras`, and saves `models/fashion_vgg16.keras`.

```text
Input image: 128 x 128 x 3
Backbone: VGG16 with ImageNet weights
Pooling: GlobalAveragePooling2D
Dense layer: 128 units with ReLU
Dropout: 0.4
Output: Softmax
Head-training epochs: 5 maximum
Fine-tuning epochs: 3 maximum
Early-stopping patience: 2
```

Expected artifacts:

```text
models/fashion_vgg16.keras
models/day7_class_names.json
models/day7_training_history.npz
models/day7_model_comparison.csv
```

**Overall status: In progress. Transfer-learning code is prepared, but execution depends on Day 6 and the real image dataset. YOLO remains pending.**

---

# Day 8 - Segmentation and End-to-End Computer Vision

## Objective

Day 8 adds product segmentation, background removal, catalog-image generation, Grad-CAM explainability, and product localization to the Day 7 VGG16 workflow.

## Implementation

```text
src/Day8_segmentation_gradcam.py
```

The pipeline reuses:

```text
models/fashion_vgg16.keras
```

It maintains a `128 x 128` input, VGG16 `preprocess_input`, and `masterCategory` target mapping.

## Grad-CAM and Segmentation Workflow

```text
Input Image
    |
    v
VGG16 Classification
    |
    v
Grad-CAM Heatmap using block5_conv3
    |
    v
Initial Binary Mask
    |
    v
OpenCV GrabCut Refinement
    |
    v
Final Product Mask
```

This is weakly supervised segmentation. VGG16 performs classification, Grad-CAM estimates the important region, and GrabCut refines the foreground mask. It is not a separately trained semantic-segmentation model.

The final mask creates a transparent-background PNG and a white-background catalog image. A bounding rectangle is derived from the segmentation mask and is not a YOLO-predicted box.

## Generated Outputs

```text
outputs/day8/<image-name>/
|-- 01_original.jpg
|-- 02_gradcam_heatmap.jpg
|-- 03_gradcam_overlay.jpg
|-- 04_initial_mask.png
|-- 05_final_mask.png
|-- 06_transparent_product.png
|-- 07_clean_catalog.jpg
|-- 08_detected_product.jpg
`-- prediction.json
```

Execution:

```bash
python -m src.Day8_segmentation_gradcam
python -m src.Day8_segmentation_gradcam data/test_product.jpg
```

**Overall status: In progress. The code is implemented, but successful execution depends on the verified Day 7 model and Fashion Product Images dataset.**

---

# Day 9 - RNN / LSTM Demand Forecasting

## Objective

Day 9 introduces time-series forecasting to RetailMind AI. The objective is to forecast future product demand from historical store-item sales using an LSTM neural network. These forecasts can later support inventory-planning decisions by helping estimate future stock requirements.

## Dataset

```text
data/train.csv
```

Expected columns:

```text
date
store
item
sales
```

The current experiment focuses on:

```text
Store ID: 1
Item ID: 1
```

## Implementation

```text
src/Day9_demand_forecasting.py
```

```text
Historical Sales
    |
    v
Select Store and Item
    |
    v
Create Daily Time Series
    |
    v
Scale Sales Values
    |
    v
Create 30-Day Look-Back Windows
    |
    v
Train LSTM
    |
    v
Predict Next-Day Demand
    |
    v
Compare Predicted vs Actual Sales
    |
    v
MAE / RMSE Evaluation
```

## Time-Series Preparation

The pipeline:

1. Loads `data/train.csv`.
2. Converts `date` to datetime format.
3. Sorts records chronologically.
4. Removes missing values.
5. Selects store 1 and item 1.
6. Keeps `date` and `sales`.
7. Resamples the series to daily frequency.
8. Fills missing daily sales values with zero.

## Train-Test Strategy

The most recent 28 days are held out for testing. Earlier observations are used for training, preserving chronological order.

## Scaling and Look-Back Windows

`MinMaxScaler` scales the sales series. The scaler is fitted only on training data. The model uses the previous 30 daily sales values to predict the following day's demand.

```text
Previous 30 Days of Sales
            |
            v
          LSTM
            |
            v
Next-Day Sales Prediction
```

## LSTM Architecture

```text
Input: 30 time steps x 1 sales feature
LSTM: 50 units
Dense: 1 output
Optimizer: Adam
Loss: Mean Squared Error
Metric: Mean Absolute Error
```

The current evaluation is a one-step-ahead rolling forecast. Each test input uses the latest available actual sales observations.

## Evaluation

| Metric | Result |
|---|---|
| MAE | Pending execution |
| RMSE | Pending execution |

The actual values must be added only after successful script execution.

## Forecast Visualization

The pipeline plots actual sales against predicted sales for the 28-day test period.

Expected output:

```text
outputs/day9/day9_demand_forecast.png
```

## Expected Day 9 Artifacts

```text
models/demand_forecaster.keras
models/day9_sales_scaler.joblib
outputs/day9/day9_demand_predictions.csv
outputs/day9/day9_demand_forecast.png
```

The forecaster can later feed RetailMind's inventory-planning module.

## Business Value

Demand forecasting helps estimate future inventory requirements and can reduce:

- Stockouts caused by insufficient inventory.
- Overstock caused by ordering more products than expected demand.
- Unnecessary inventory holding costs.

## Day 9 Status

```text
Historical sales loading             Implemented
Store-item filtering                 Implemented
Daily time-series preparation        Implemented
30-day look-back windows             Implemented
LSTM demand model                    Implemented
Recent 28-day test holdout           Implemented
MAE evaluation                       Implemented
RMSE evaluation                      Implemented
Actual-vs-predicted visualization    Implemented
Forecast persistence                 Implemented
Successful end-to-end execution      Not yet verified
Actual MAE and RMSE                   Not yet recorded
```

**Overall status: In progress. The Day 9 LSTM demand-forecasting workflow is implemented, but successful execution and actual MAE/RMSE results still need verification.**


---

# Day 10 – Seq2Seq & Attention Forecasting

## Objective

Extend the Day 9 LSTM demand forecasting module to multi-step forecasting using a Sequence-to-Sequence (Seq2Seq) model with Attention.

The objective is to predict multiple future demand values together and provide a longer forecasting horizon for inventory and procurement planning.

Day 9 predicts the next demand value, while Day 10 predicts several future demand values from the same historical sales sequence.

---

## Dataset

**Dataset:** Store Item Demand Forecasting Challenge

**Source:** Kaggle – Store Item Demand Forecasting Challenge

Day 10 reuses the same sales dataset used in Day 9.

The dataset is stored at:

```text
data/train.csv
```


## Implementation

The Day 10 implementation is maintained in:

```text
src/Day10_seq2seq_attention.py
```

The forecasting workflow extends Day 9 from a single next-period output to a 14-day output sequence:

```text
Previous 30 Days of Sales
           |
           v
      LSTM Encoder
           |
           +-------------------+
           |                   |
           v                   v
Encoder Sequence Outputs   Final Hidden/Cell States
           |                   |
           |              LSTM Decoder
           |                   |
           +---- Attention ----+
                    |
                    v
             Next 14 Days
```

## Seq2Seq Architecture

```text
Encoder input: 30 historical days x 1 sales feature
Encoder LSTM: 64 units
Decoder horizon: 14 future days
Decoder LSTM: 64 units
Attention: Decoder states attend over encoder sequence outputs
Output: Dense(1) for each of the 14 future timesteps
Optimizer: Adam
Loss: Mean Squared Error
Metric: Mean Absolute Error
```

## Evaluation and Visualization

Day 10 evaluates the 14-day forecast using MAE and RMSE and compares predicted demand with actual sales.

| Metric | Result |
|---|---|
| MAE | Pending execution |
| RMSE | Pending execution |

The implementation also creates an Attention heatmap. The intended attention-score layout is:

```text
Rows: 14 forecast days
Columns: 30 historical days
```

Expected outputs:

```text
outputs/day10/day10_forecast.png
outputs/day10/day10_attention_weights.png
outputs/day10/day10_multiweek_forecast.csv
```

## Expected Day 10 Artifacts

```text
models/day10_seq2seq_attention.keras
models/day10_sales_scaler.joblib
outputs/day10/day10_forecast.png
outputs/day10/day10_attention_weights.png
outputs/day10/day10_multiweek_forecast.csv
```

## Day 9 vs Day 10

```text
Day 9:  30 historical days -> LSTM -> next 1 day
Day 10: 30 historical days -> Seq2Seq + Attention -> next 14 days
```

The longer forecasting horizon is intended to support short-term procurement and inventory planning.

## Day 10 Status

```text
Seq2Seq encoder-decoder code         Implemented
Attention mechanism                  Implemented
14-day multi-step output             Implemented
MAE/RMSE evaluation                  Implemented
Forecast visualization               Implemented
Attention heatmap                    Implemented
Procurement forecast export          Implemented
Leakage-safe train/test split         Needs correction before final verification
Successful end-to-end execution      Not yet verified
Actual MAE and RMSE                   Not yet recorded
```

**Overall status: In progress. The Day 10 model structure is implemented, but the sequence split should be corrected so held-out target values cannot enter training before final execution and verification.**

# Installation and Execution

Run all commands from the project root:

```text
RETAILMIND-AI-MAIN-doing>
```

## 1. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Dataset Setup

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
data/fashion-product-images-small/styles.csv
data/fashion-product-images-small/images/
data/train.csv
```

## 4. Run Project Modules

```bash
python notebooks/churn_analysis.py
python -m src.Day1_train
python -m src.Day1_predict
python -m src.Day2_compare_models
python -m src.Day3_tune_models
python -m src.Day2_learning_analysis
python -m src.Day4_feature_engineering
python -m src.Day4_feature_selection
python -m src.Day4_train_ensembles
python -m src.Day4_test_production_model
python -m src.Day5_mnist_demo
python -m src.Day5_train_ann
python -m src.Day5_compare_models
python -m src.Day6_fashion_mnist
python -m src.Day6_image_pipeline
python -m src.Day6_train_cnn
python -m src.Day6_evaluate_cnn
python -m src.Day7_transfer_learning
python -m src.Day8_segmentation_gradcam
python -m src.Day9_demand_forecasting
python -m src.Day10_seq2seq_attention
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
| Day 5 | MNIST ANN demo, churn ANN training, saved neural-network artifacts, and model comparison | Completed |
| Day 6 | Fashion-MNIST CNN, real-image pipeline, CNN training, confusion-matrix evaluation, and catalog classifier | Pending, dataset required |
| Day 7 | VGG16 transfer learning, fine-tuning, Day 6 comparison, and YOLO product detection | In progress; transfer-learning code added, execution and YOLO pending |
| Day 8 | Grad-CAM, weak segmentation, background removal, catalog-image generation, and localization | In progress; code added, execution pending Day 7 model |
| Day 9 | LSTM store-item demand forecasting, MAE/RMSE evaluation, and demand visualization | In progress; code added, execution and metrics pending |
| Day 10 | Seq2Seq 14-day demand forecasting with Attention and attention visualization | In progress; code added, sequence-split correction and execution pending |

---

## Upcoming Roadmap

- Complete Day 6 after installing the Fashion Product Images Small dataset.
- Execute and verify the Day 7 VGG16 transfer-learning classifier.
- Implement the Day 7 YOLO object-detection module.
- Execute and verify the Day 8 segmentation and Grad-CAM pipeline.
- Inspect generated masks, transparent products, catalog images, and bounding boxes.
- Execute and verify the Day 9 LSTM demand-forecasting pipeline.
- Record the actual Day 9 MAE and RMSE.
- Verify the predicted-versus-actual demand visualization.
- Verify the saved demand forecaster and scaler.
- Correct and verify the Day 10 train/test sequence split.
- Execute the Day 10 Seq2Seq and Attention forecasting pipeline.
- Record Day 10 MAE and RMSE and verify the attention heatmap.
- Verify the Day 10 multi-period procurement forecast and saved model artifacts.
- Continue with later roadmap items only after the current day is complete.
- Add automated unit tests during the MLOps stage.
- Build the FastAPI inference service during the API stage.
- Add model monitoring and drift detection during the monitoring stage.

---

## Author

**Aryan Mule**

GitHub: `AryanMule`

---

*RetailMind AI - Turning customer data into actionable churn predictions.*
