# Explainable Credit Risk Predictor

An end-to-end machine learning project that predicts loan default risk and explains each prediction in plain language using SHAP and an LLM — built for both technical and non-technical stakeholders in a lending workflow.

## Overview





This project trains a credit risk model on applicant financial data (32,581 records) and deploys it as an interactive Streamlit web interface. Beyond just predicting default probability, the app pairs SHAP feature attribution with an LLM-powered explanation layer, translating raw model outputs into a clear, human-readable summary of why an applicant was scored as risky or low-risk.

## Features

- **Preprocessing pipeline** — median/mode imputation, standard scaling, and one-hot encoding for numerical and categorical features.
- **Model comparison** — evaluated Logistic Regression, Decision Tree, Random Forest, and XGBoost using stratified cross-validation.
- **Class imbalance handling** — SMOTE applied within the training pipeline to address the skew toward non-default cases.
- **Hyperparameter tuning** — Optuna used to tune the final XGBoost model.
- **Threshold tuning** — decision threshold optimized (0.25) to prioritize recall on defaulters over raw accuracy.
- **Explainability** — SHAP waterfall plots show per-applicant feature impact; an LLM (via Groq) converts these into a plain-language risk explanation.
- **Interactive UI** — Streamlit app for entering applicant details and viewing predictions in real time.

## Results

Final tuned XGBoost model (test set, threshold = 0.25):

| Metric | Score |
|---|---|
| ROC-AUC | 94.7% |
| Recall (Default) | 80.7% |
| Precision (Default) | 78.5% |
| F1 (Default) | 79.6% |
| Accuracy | 91.1% |

XGBoost outperformed Logistic Regression, Decision Tree, and Random Forest across all cross-validated metrics.

## Tech Stack

- **Modeling:** Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE), Optuna
- **Explainability:** SHAP
- **LLM Integration:** Groq API (Llama 3.3 70B)
- **App:** Streamlit
- **Data Handling:** pandas, NumPy

## How It Works

1. Applicant details are entered via the Streamlit sidebar.
2. Inputs are transformed using the saved preprocessing pipeline.
3. The XGBoost model predicts a default probability.
4. SHAP computes per-feature contributions for that specific prediction.
5. The top contributing factors are passed to an LLM, which generates a plain-language explanation of the risk level — written for a non-technical audience.
