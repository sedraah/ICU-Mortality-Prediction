# ICU Mortality Prediction

A supervised machine learning pipeline for predicting in-hospital mortality of ICU patients using physiological measurements recorded during the first 48 hours of admission. Built on the [PhysioNet Computing in Cardiology Challenge dataset](https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet) via Kaggle.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Summary](#pipeline-summary)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Classification Models](#classification-models)
- [Tools & Libraries](#tools--libraries)

---

## Overview

Predicting ICU mortality early can help clinicians prioritise resources and escalate care in time. This project applies a complete ML pipeline from raw time-series clinical data to trained and evaluated classifiers to predict whether a patient will die in hospital based on their first 48 hours of ICU measurements.

Three classification models are compared: Decision Tree, K-Nearest Neighbours, and Support Vector Machine.

---

## Dataset

**Source:** [Kaggle — Predict Mortality of ICU Patients (PhysioNet)](https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet)

The dataset has two components that are merged as the first pipeline step:

| Component | Description |
|---|---|
| `Outcomes-a.txt` | One row per patient: RecordID, SAPS-I, SOFA, Length of Stay, Survival, and the binary target `In-hospital_death` |
| `set-a.zip` | ~4,000 individual `.txt` files, each containing time-stamped physiological measurements over 48 hours |

**Merging approach:** Each patient file is parsed, and all time-series readings are aggregated into a single row by computing the mean value per parameter over the 48-hour window. The resulting feature matrix is joined to the outcomes file on `RecordID`.

**Final merged shape:** ~4,000 patients × 37 features
**Target variable:** `In-hospital_death` (binary: 0 = survived, 1 = died in hospital)

---

## Pipeline Summary

```
Raw patient files (set-a.zip)
        ↓  Parse + mean-aggregate per parameter
Feature matrix (4000 × 37)
        ↓  Merge with Outcomes-a.txt on RecordID
Merged dataset
        ↓  Drop columns >50% missing → drop RecordID, Survival, MechVent
        ↓  Replace sentinel values (-1, 0) with NaN
        ↓  Physiological clipping (domain-valid ranges)
        ↓  Feature selection (top 10 by correlation with target)
        ↓  Stratified train/test split (70/30)
        ↓  Pipeline: Median imputation → log1p transform → StandardScaler
        ↓  SMOTE (training set only)
        ↓  GridSearchCV with 5-fold CV (F1 scoring)
Trained models: Decision Tree | KNN | SVM
        ↓  Evaluation on imbalanced test set
Performance metrics: Accuracy, Precision, Recall, F1
```

---

## Exploratory Data Analysis

### Class Distribution
The target variable is heavily imbalanced: **3,446 patients (86.1%) survived** and only **554 (13.9%) died in hospital**, an approximately 6:1 ratio. This is clinically expected for general ICU populations but poses a challenge for classifiers that default to predicting the majority class.

### Feature Correlation with Mortality
Pearson correlation was computed between each feature and `In-hospital_death`. Key findings:

| Feature | Direction | Clinical Interpretation |
|---|---|---|
| GCS (Glasgow Coma Scale) | Negative (strongest) | Higher consciousness → lower mortality |
| Age | Positive | Older patients have reduced physiological reserve |
| BUN (Blood Urea Nitrogen) | Positive | Elevated BUN indicates renal failure |
| Creatinine | Positive | Renal dysfunction marker |
| Lactate | Positive | Tissue hypoxia and septic shock indicator |

### Feature Correlation Heatmap
The full pairwise heatmap revealed notable redundancies:
- Blood pressure variants (SysABP, DiasABP, MAP, NISysABP, NIDiasABP, NIMAP) are strongly inter-correlated, i.e. they measure the same haemodynamic state via different methods
- BUN and Creatinine are strongly correlated (both reflect kidney filtration)
- SAPS-I and SOFA are moderately correlated with each other and with many physiological variables, both were dropped as they are composite scores derived from the same raw measurements, which would introduce multicollinearity

---

## Classification Models

All models were trained using GridSearchCV with 5-fold cross-validation, optimising for F1-score to prioritize minority class performance.

### Decision Tree (DT)
A rule-based model that recursively partitions the feature space through binary splits. Highly interpretable & clinically valuable when model reasoning must be explained to medical staff. 

### K-Nearest Neighbours (KNN)
A non-parametric instance-based classifier: a patient's outcome is predicted by majority vote among the k most similar patients in the training set. 

### Support Vector Machine (SVM)
A margin-based classifier that finds the optimal hyperplane separating classes with maximum margin. Uses the RBF kernel for non-linear decision boundaries. Generally more robust to overfitting than DTs on small-to-medium datasets. Well-suited to the high-dimensional scaled feature space.

---


## Tools & Libraries

- **Python 3**
- `pandas`, `numpy` for data loading, merging, preprocessing
- `scikit-learn:` pipelines, imputation, scaling, RFECV, GridSearchCV, DT, KNN, SVM, metrics
- `imbalanced-learn:`  SMOTE oversampling
- `matplotlib`, `seaborn:` visualizations
- `scipy:` statistical tests

