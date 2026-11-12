# RetailMind AI - Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using customer demographics, account information, subscribed services, and billing details.

The project follows a structured workflow covering exploratory data analysis, preprocessing, baseline training, model comparison, hyperparameter tuning, cross-validation, learning-curve analysis, feature engineering, feature selection, ensemble learning, production model verification, neural-network experimentation, CNN image classification, transfer learning with VGG16, Grad-CAM explainability, weakly supervised product segmentation, background removal, and catalog-image generation.

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
- Save and reload trained pipelines, preprocessing objects, and neural-network models.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training, selection, tuning, and evaluation |
| XGBoost | Gradient-boosted ensemble model |
| TensorFlow and Keras | ANN, CNN, and transfer-learning development |
| VGG16 | ImageNet-pretrained transfer-learning backbone |
| OpenCV | Image loading, mask refinement, background removal, and output generation |
| Grad-CAM | Visual explanation of VGG16 classification decisions |
| Pillow | Image loading and validation |
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

- `No` -> 0
- `Yes` -> 1

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is imbalanced, with fewer customers leaving than staying. Recall is therefore an important evaluation metric because identifying customers who may churn is a primary objective.

### Computer-Vision Dataset

Days 6, 7, and 8 require the Kaggle `paramaggarwal/fashion-product-images-small` dataset.

Expected local structure:

```text
data/fashion-product-images-small/
|-- styles.csv
`-- images/
    |-- <product-id>.jpg
    `-- ...
```

---

## Project Structure

```text
RETAILMIND-AI-MAIN-doing/
|
|-- data/
|   |-- raw/
|   |   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|   `-- fashion-product-images-small/        # Required for Days 6 and 7
|       |-- styles.csv
|       `-- images/
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
|   `-- Day8_segmentation_gradcam.py
|
|-- outputs/
|   `-- day8/                          # Generated Day 8 images and prediction JSON
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
|   |-- fashion_cnn.keras                 # Expected after Day 6 completion
|   |-- fashion_vgg16.keras               # Expected after Day 7 transfer learning
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

Categorical features were processed with most-frequent imputation and one-hot encoding.

The stratified split produced 5,634 training records and 1,409 testing records.

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

Saved artifacts:

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

Random Forest was selected because it achieved the highest cross-validation recall and the highest test ROC-AUC among the tuned candidates.

Saved artifacts:

```text
models/churn_v3.joblib
models/day3_tuning_results.csv
models/learning_curve.png
```

The final learning-curve training-validation recall gap was 0.0929, indicating that some variance remained.

---

# Day 4 - Feature Engineering and Ensemble Learning

## Feature Engineering

`CustomerFeatureEngineer` introduced six features:

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

XGBoost was selected because churn recall was the primary metric and it achieved the higher recall and ROC-AUC.

Saved model:

```text
models/churn_v4.joblib
```

The saved model was independently reloaded and verified on five customer records.

---

# Day 5 - Neural Networks and Deep Learning

Day 5 introduced TensorFlow and Keras neural networks.

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

Saved artifacts:

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

## Current Blocker

The real-image workflow requires:

```text
data/fashion-product-images-small/styles.csv
data/fashion-product-images-small/images/
```

Until these are installed, no real-product CNN metrics or final catalog-classifier artifact should be reported.

## Expected Artifacts

```text
models/fashion_cnn.keras
models/day6_training_history.npz
models/day6_class_names.json
models/day6_confusion_matrix.png
models/day6_evaluation.csv
```

## Day 6 Status

```text
Fashion-MNIST CNN prototype          Model artifact exists; metric not recorded
Real-image loading pipeline          Blocked by missing styles.csv and images/
Real-product CNN training            Not verified
CNN evaluation and confusion matrix  Not verified
Saved catalog classifier             Not verified
```

**Overall status: Pending because the Fashion Product Images dataset is not installed locally.**

---

# Day 7 - Transfer Learning and Object Detection

## 1. Objective

Day 7 upgrades the Day 6 computer-vision module using transfer learning. An ImageNet-pretrained VGG16 network is adapted to classify the Fashion Product Images categories, fine-tuned on the project dataset, evaluated on the same test split, and compared with the Day 6 custom CNN.

The complete Day 7 roadmap also includes YOLO-based product detection. The current implementation covers the VGG16 transfer-learning classifier only; object detection remains to be implemented separately.

## 2. Implemented Day 7 File

| File | Purpose | Current status |
|---|---|---|
| `src/Day7_transfer_learning.py` | Train and fine-tune a VGG16-based product classifier and compare it with the Day 6 CNN | Implemented; execution pending dataset and Day 6 model |
| `src/Day7_yolo_detection.py` | Detect fashion products using YOLO bounding boxes and confidence scores | Not implemented |

## 3. Transfer-Learning Workflow

The Day 7 transfer-learning script performs the following steps:

1. Loads `styles.csv` and matches product IDs with image files.
2. Uses `masterCategory` as the classification target.
3. Encodes category labels with `LabelEncoder`.
4. Creates stratified training, validation, and test splits.
5. Resizes images to `128 x 128` pixels.
6. Applies VGG16 `preprocess_input` preprocessing.
7. Loads VGG16 with ImageNet weights and excludes its original top classifier.
8. Freezes the pretrained convolutional base.
9. Adds global average pooling, a 128-unit ReLU layer, dropout, and a softmax output layer.
10. Trains the new classification head.
11. Unfreezes the top VGG16 layers and fine-tunes them using a lower learning rate.
12. Evaluates the VGG16 model on the test dataset.
13. Loads and evaluates the Day 6 CNN when `models/fashion_cnn.keras` exists.
14. Compares Day 6 CNN accuracy with Day 7 VGG16 accuracy.
15. Saves the upgraded model as `models/fashion_vgg16.keras`.

## 4. VGG16 Architecture and Training

```text
Input image: 128 x 128 x 3
Pretrained backbone: VGG16 with ImageNet weights
Original VGG16 top: Removed
Pooling: GlobalAveragePooling2D
Dense layer: 128 units with ReLU
Regularization: Dropout 0.4
Output: Softmax with one unit per product category
```

Initial classification-head training uses:

```text
Optimizer: Adam
Loss: sparse_categorical_crossentropy
Epochs: 5 maximum
Early stopping patience: 2
```

Fine-tuning uses:

```text
Unfrozen area: Top four VGG16 layers
Optimizer: Adam
Learning rate: 0.00001
Epochs: 3 maximum
Early stopping patience: 2
```

## 5. Day 6 Comparison Dependency

The comparison requires the real-product Day 6 CNN:

```text
models/fashion_cnn.keras
```

The Fashion-MNIST prototype model below is not a substitute because it was trained on a different dataset:

```text
models/fashion_mnist_cnn.keras
```

Both models must use the same test images, target categories, and class-to-index mapping for the comparison to be valid.

## 6. Current Blockers and Verification Requirements

Day 7 cannot be marked complete until the following are verified:

- The Fashion Product Images dataset is installed.
- The Day 6 image pipeline runs successfully.
- The Day 6 real-product CNN is trained and saved.
- Day 6 and Day 7 use the same category mapping.
- The VGG16 script runs successfully.
- Actual VGG16 test accuracy is recorded.
- Actual Day 6-versus-Day 7 comparison results are recorded.
- The saved VGG16 model reloads successfully.
- The YOLO object-detection portion is implemented and verified.

## 7. Expected Day 7 Artifacts

```text
models/fashion_vgg16.keras
models/day7_class_names.json
models/day7_training_history.npz
models/day7_model_comparison.csv
```

The following object-detection outputs may be added after YOLO is implemented:

```text
models/day7_yolo_predictions/
models/day7_detection_results.csv
```

These artifacts must not be reported as completed until their creation has been verified.

## 8. Day 7 Status

```text
VGG16 transfer-learning code         Implemented
Classification-head training         Not yet run successfully
VGG16 fine-tuning                     Not yet run successfully
Day 6 CNN comparison                  Pending Day 6 real-product model
Saved VGG16 classifier                Not yet verified
YOLO object detection                 Not implemented
```

**Overall status: In progress. The transfer-learning code is prepared, but execution depends on completing Day 6 and installing the real image dataset. YOLO is still pending.**

---

# Day 8 - Segmentation and End-to-End Computer Vision

## Objective

Extend the RetailMind AI computer-vision pipeline by adding product segmentation, background removal, clean catalog-image generation, Grad-CAM explainability, and product localization.

Day 8 builds on the VGG16 transfer-learning model developed on Day 7 and creates a more complete and explainable computer-vision workflow.

---

## Dataset

**Dataset:** Fashion Product Images Small

**Source:** Kaggle - Fashion Product Images Small

The dataset contains:

- Product images stored inside the `images/` directory.
- `styles.csv` containing product metadata.
- Product categories available through the `masterCategory` column.

The dataset does not contain pixel-level segmentation masks or ground-truth bounding-box annotations.

---

## Technologies and Libraries

- Python
- TensorFlow / Keras
- VGG16
- OpenCV
- NumPy
- Grad-CAM

---

## Implementation

The complete Day 8 implementation is available in:

```text
src/Day8_segmentation_gradcam.py
```

The pipeline integrates the Day 7 VGG16 classification model with segmentation, background removal, explainability, and product localization.

---

## 1. Day 7 Model Integration

Day 8 reuses the transfer-learning model trained on Day 7:

```text
models/fashion_vgg16.keras
```

The same input configuration used during Day 7 is maintained:

- Image size: `128 x 128`
- VGG16 preprocessing using `preprocess_input`
- Target variable: `masterCategory`

Using the same preprocessing during training and prediction ensures consistency between Day 7 and Day 8.

---

## 2. Product Classification

The saved VGG16 model is used to classify the input product image.

The pipeline:

1. Loads the product image using OpenCV.
2. Converts the image from BGR to RGB.
3. Resizes the image to `128 x 128`.
4. Applies VGG16 `preprocess_input`.
5. Passes the image through the saved Day 7 model.
6. Selects the class having the highest predicted probability.

The pipeline returns:

- Predicted class index
- Predicted `masterCategory`
- Prediction confidence

---

## 3. Grad-CAM Explainability

Grad-CAM was added to make the VGG16 classification result more explainable.

Grad-CAM uses the final convolutional layer of VGG16:

```text
block5_conv3
```

The process calculates the gradients of the predicted class with respect to the convolutional feature maps. These gradients identify the image regions that contributed most strongly to the model prediction.

Typical interpretation:

- Red and yellow regions indicate stronger influence on the prediction.
- Blue regions indicate lower influence on the prediction.

A Grad-CAM overlay is generated by placing the heatmap over the original product image.

---

## 4. Product Segmentation

The Grad-CAM heatmap is used to estimate the important foreground region of the product image.

```text
Input Image
    |
    v
VGG16 Classification
    |
    v
Grad-CAM Heatmap
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

This is a weakly supervised segmentation approach. VGG16 performs image classification, while Grad-CAM estimates the important region and OpenCV GrabCut refines it into a foreground mask. It is not a separately trained semantic-segmentation model.

---

## 5. Background Removal and Catalog Image

The final product mask is used to create:

- A transparent-background PNG where the mask becomes the alpha channel.
- A clean catalog image with the product placed on a uniform white background.

The pipeline also keeps the largest connected foreground component to reduce small noisy regions.

---

## 6. Product Localization

A bounding rectangle is calculated from the final segmentation mask. The output image displays:

- Product bounding box
- Predicted category
- Prediction confidence

The bounding box is derived from the segmentation mask. It is not a YOLO-predicted detection box.

---

## 7. Generated Outputs

For every processed image, Day 8 creates a separate folder under:

```text
outputs/day8/<image-name>/
```

Expected generated files:

```text
01_original.jpg
02_gradcam_heatmap.jpg
03_gradcam_overlay.jpg
04_initial_mask.png
05_final_mask.png
06_transparent_product.png
07_clean_catalog.jpg
08_detected_product.jpg
prediction.json
```

The JSON file stores the predicted class, confidence, bounding-box coordinates, model path, Grad-CAM layer, and segmentation method.

---

## 8. Day 8 Execution

Run with the default dataset image:

```bash
python -m src.Day8_segmentation_gradcam
```

Run with a custom image:

```bash
python -m src.Day8_segmentation_gradcam data/test_product.jpg
```

Day 8 requires:

```text
data/fashion-product-images-small/styles.csv
data/fashion-product-images-small/images/
models/fashion_vgg16.keras
```

---

## 9. Day 8 Status

```text
Day 8 pipeline code                 Implemented
VGG16 model integration             Implemented
Grad-CAM explainability             Implemented
Weakly supervised segmentation      Implemented
Background removal                  Implemented
White catalog-image generation      Implemented
Mask-derived bounding box           Implemented
Successful end-to-end execution     Not yet verified
Output quality review               Not yet verified
```

**Overall status: In progress. The Day 8 code is implemented, but successful execution depends on the verified Day 7 VGG16 model and the Fashion Product Images dataset.**

---

# Installation and Execution

Run all commands from the project root directory:

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
```

Run Day 7 only after the Fashion Product Images dataset is installed. A valid Day 6 comparison additionally requires `models/fashion_cnn.keras`.

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

---

## Upcoming Roadmap

- Complete Day 6 after installing the Fashion Product Images Small dataset.
- Execute and verify the Day 7 VGG16 transfer-learning classifier.
- Implement the Day 7 YOLO object-detection module.
- Execute and verify the Day 8 segmentation and Grad-CAM pipeline.
- Inspect generated masks, transparent products, catalog images, and bounding boxes.
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
