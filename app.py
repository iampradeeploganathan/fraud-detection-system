from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# ==================================
# Load Trained Artifact
# ==================================
print("APP FILE LOADED")

artifact = joblib.load("best_fraud_detection_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
scaler = artifact["scaler"]

# ==================================
# FastAPI App
# ==================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0",
    description="Production Grade Fraud Detection System"
)

# ==================================
# Input Schema
# ==================================

class Transaction(BaseModel):
    type: int
    amount: float
    log_amount: float
    is_high_amount: int
    hour: int
    is_night: int
    orig_balance_diff: float
    dest_balance_diff: float
    orig_error: float
    dest_error: float
    amount_to_balance_ratio: float


# ==================================
# Home Endpoint
# ==================================

@app.get("/")
def home():
    return {
        "status": "running",
        "model": artifact["model_name"],
        "threshold": float(threshold)
    }


# ==================================
# Prediction Endpoint
# ==================================

@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.dict()])

    # Ensure exact feature order
    data = data[features]

    # Scale
    data_scaled = scaler.transform(data)

    probability = model.predict_proba(data_scaled)[0][1]

    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(float(probability), 6),
        "threshold": round(float(threshold), 6),
        "prediction": prediction,
        "label": "Fraud" if prediction else "Legitimate"
    }
print(app.routes)