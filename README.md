# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It explores transaction patterns, builds binary and multiclass classification models, addresses class imbalance using SMOTE, and analyses changes in feature distributions over time using Population Stability Index (PSI).

The objective is to evaluate different machine learning approaches for identifying fraudulent transactions and understanding potential data distribution changes.

## Project Objectives

- Understand the dataset through exploratory data analysis (EDA).
- Clean and preprocess transaction data.
- Perform feature engineering and extract temporal features.
- Build binary classification models to distinguish fraudulent and legitimate transactions.
- Address class imbalance using SMOTE.
- Build multiclass models to identify different fraud categories.
- Evaluate model performance using appropriate classification metrics.
- Analyse feature and target distribution changes using PSI.

## Dataset Overview

| Description | Details |
|---|---|
| Dataset | `credit_card_fraud_drift_2026.csv` |
| Total records | 120,000 |
| Original columns | 73 |
| Transaction period | January–June 2026 |
| Modelling-eligible records | 118,107 |
| Excluded records | 1,893 Friendly-Fraud records |

**Note:** Friendly-Fraud records were excluded from transaction-time modelling because they represent post-transaction information.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

## Project Workflow

The project was completed through five main stages.

### 1. Data Understanding and Exploratory Data Analysis

- Inspected dataset dimensions, data types and missing values.
- Examined fraudulent and legitimate transaction distributions.
- Analysed fraud categories, transaction amounts and monthly trends.
- Investigated outliers and potential data leakage.

### 2. Feature Engineering and Preprocessing

- Extracted and validated temporal features.
- Selected numerical and categorical predictors.
- Applied numerical transformations and one-hot encoding.
- Created chronological training, validation and test datasets.

**Chronological Data Split**

| Dataset | Period | Records |
|---|---|---:|
| Training | January–February 2026 | 38,476 |
| Validation | March 2026 | 20,141 |
| Testing | April–June 2026 | 59,490 |
| **Total** | **January–June 2026** | **118,107** |

### 3. Binary Classification and SMOTE

- Trained Logistic Regression, Decision Tree and Random Forest models.
- Compared baseline and expanded feature sets.
- Applied SMOTE only to the training data.
- Evaluated models using accuracy, precision, recall, F1-score and confusion matrices.
- Compared model performance before and after SMOTE.

### 4. Multiclass Classification

- Classified transactions into legitimate and different fraud categories.
- Compared Logistic Regression, Decision Tree and Random Forest.
- Evaluated weighted and macro-averaged performance.
- Compared results before and after SMOTE.

### 5. Data Drift Analysis

- Used Population Stability Index (PSI) to examine feature distribution changes.
- Analysed numerical, binary and categorical features.
- Compared training, validation and test distributions.
- Examined changes in target distributions.

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── Data/
│   ├── Raw/
│   └── processed/
│
├── models/
│   ├── baseline_preprocessor.joblib
│   ├── expanded_preprocessor.joblib
│   └── (18 additional trained model files)
│
├── notebooks/
│   ├── 01_Data_Understanding_and_EDA.ipynb
│   ├── 02_Feature_Engineering_and_Preprocessing.ipynb
│   ├── 03_Binary_Classification_and_SMOTE.ipynb
│   ├── 04_Multiclass_Classification.ipynb
│   └── 05_Data_Drift_Analysis.ipynb
│
├── reports/
├── visualizations/
├── README.md
├── requirements.txt
└── Credit_Card_Fraud_Detection_Presentation.pdf
```

The `models/` directory contains all 20 trained model and preprocessing artifacts.

## Model Performance and Key Results

### 1. Binary Classification

Three machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing configuration was **Random Forest with the expanded feature set and SMOTE**.

**Best Model: Random Forest (Expanded Features + SMOTE)**

| Metric | Test Score |
|---|---:|
| Accuracy | 99.97% |
| Precision | 99.98% |
| Recall | 99.65% |
| F1-score | 99.81% |

**Confusion Matrix**

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 54,656 | 1 |
| Fraud | 17 | 4,816 |

SMOTE was applied exclusively to the training dataset. Validation and test datasets retained their original class distributions.

SMOTE did not improve every model. Its impact varied depending on the algorithm and feature set.

### 2. Multiclass Classification

The multiclass task considered five classes:

- Legitimate
- Account-Takeover
- Card-Not-Present
- Card-Present
- Synthetic-Identity

The **Random Forest model before SMOTE** achieved the strongest overall performance.

| Metric | Weighted Score | Macro Score |
|---|---:|---:|
| Precision | 98.88% | 90.59% |
| Recall | 98.81% | 88.17% |
| F1-score | 98.76% | 88.51% |

The Synthetic-Identity class had comparatively lower recall, indicating that identifying certain minority fraud categories remains challenging.

### 3. Data Drift Analysis

Population Stability Index (PSI) was used to compare feature and target distributions across training, validation and test periods.

**Overall Feature Drift Summary**

| Drift Category | Number of Features |
|---|---:|
| Low drift | 96 |
| Moderate drift | 0 |
| High drift | 3 |
| **Total** | **99** |

The three high-drift features were:

| Feature | Train vs Test PSI |
|---|---:|
| `temporal__month` | 26.938967 |
| `temporal__week_of_year` | 25.446169 |
| `continuous_log__account_age_days` | 0.336742 |

Target distributions remained relatively stable across the periods analysed.

**Important:** High PSI in month and week-of-year is associated with the chronological separation of the datasets. Stable target prevalence does not necessarily guarantee stable model performance.

## Recommended Trained Models

The following three artifacts are highlighted for convenient reference based on the model evaluation results:

| Model | File |
|---|---|
| Best Binary Classification Model | `random_forest_expanded_smote.joblib` |
| Best Multiclass Classification Model | `random_forest_multiclass_before_smote.joblib` |
| Expanded Feature Preprocessor | `expanded_preprocessor.joblib` |

All 20 trained model and preprocessing files, including the remaining experimental models, are available in the `models/` directory.

### Model Loading Note

The saved models are intended to be used with compatible preprocessing, feature ordering and package versions. The corresponding notebooks document the training and preprocessing workflow.

## Limitations

- Source and timing provenance verification was incomplete for 34 retained features.
- Standard SMOTE applied after one-hot encoding may create unrealistic fractional categorical indicators.
- Friendly-Fraud records were excluded because they represent post-transaction information.
- The very high binary classification scores should be interpreted cautiously until feature provenance and potential leakage risks are fully verified.
- Chronological distribution changes may affect model generalisation over time.

## Future Improvements

- Verify the source and availability timing of all retained features.
- Investigate alternative resampling methods suitable for categorical data.
- Evaluate model performance on additional future transaction periods.
- Improve detection of minority fraud categories.
- Explore ongoing model and data drift monitoring.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for credit card fraud analysis, covering exploratory data analysis, feature engineering, binary and multiclass classification, class imbalance handling and data drift monitoring.

The experiments highlight the importance of evaluating multiple models, understanding class imbalance, monitoring changes in data distributions and carefully examining potential data leakage before drawing conclusions about model performance.

## Disclaimer

This project is an experimental machine learning study. The reported results are based on the available dataset and should not be interpreted as evidence of production-level fraud detection performance. Further feature provenance verification and evaluation on independent future data are necessary.