# Heart Disease Prediction using Machine Learning

## Project Overview
This project aims to predict the presence of cardiovascular disease using machine learning techniques. It uses patient health data such as age, blood pressure, BMI, cholesterol, etc., to classify whether a person has heart disease or not.

## Objective
- Predict heart disease using machine learning
- Analyze important health factors
- Build an accurate and explainable model

# Dataset
- Source: Kaggle (Cardiovascular Disease Dataset)
- Records: ~70,000 patients
- Features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, etc.
- Target: `cardio` (0 = No disease, 1 = Disease)

## Data Preprocessing
- Removed incorrect values (e.g., negative blood pressure)
- Converted age from days to years
- Created new feature: BMI
- Handled categorical features
- Created age groups

## Models Used
- Logistic Regression
- Random Forest
- XGBoost (Best performing model)

## Model Performance

| Metric      | Value |
|------------|------|
| Accuracy   | 73% |
| ROC-AUC    | 0.79 |

## 📈 Feature Importance
- Age
- Blood Pressure (ap_hi)
- BMI
- Weight

These features have the highest impact on prediction.

## SHAP Analysis
SHAP is used to explain model predictions.

- Shows how each feature affects prediction
- Helps in understanding model decisions
- Important features: age, ap_hi, cholestero


