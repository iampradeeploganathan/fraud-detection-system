# Real-Time Fraud Detection System

An end-to-end machine learning system for detecting fraudulent financial transactions using XGBoost, custom feature engineering, threshold optimization, SHAP explainability, FastAPI, Streamlit, and Docker.

---

## Problem Statement

Financial fraud causes significant losses for payment platforms and banks.

The objective of this project is to identify fraudulent transactions while maximizing fraud recall and minimizing false negatives.

---

## Dataset

Source:
Kaggle - Synthetic Financial Transaction Dataset

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Transactions | 6,362,620 |
| Fraudulent Transactions | 8,213 |
| Fraud Rate | 0.129% |
| Features Before Engineering | 11 |
| Features After Engineering | 11 |

Target Variable:

```text
isFraud

0 = Legitimate Transaction
1 = Fraudulent Transaction
