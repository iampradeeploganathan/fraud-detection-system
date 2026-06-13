import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    artifact = joblib.load("best_fraud_detection_model.pkl")
    return artifact

artifact = load_model()

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
scaler = artifact["scaler"]

# =========================
# Sidebar
# =========================

st.sidebar.title("🛡️ Fraud Detection")

st.sidebar.success(f"Model: {artifact['model_name']}")
st.sidebar.info(f"Threshold: {threshold:.4f}")

st.sidebar.markdown("---")

st.sidebar.write("### About")
st.sidebar.write(
    """
    Real-Time Fraud Detection System

    - XGBoost Classifier
    - Threshold Optimization
    - Feature Engineering
    - SHAP Explainability
    """
)

# =========================
# Main Title
# =========================

st.title("💳 Real-Time Fraud Detection System")

st.markdown(
    "Enter transaction details below and click **Predict**."
)

# =========================
# Input Form
# =========================

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        options=[
            ("Payment", 0),
            ("Transfer", 1),
            ("Cash Out", 2),
            ("Debit", 3),
            ("Cash In", 4)
        ],
        format_func=lambda x: x[0]
    )[1]

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

with col2:

    oldbalanceOrg = st.number_input(
        "Old Balance Origin",
        min_value=0.0,
        value=10000.0
    )

    newbalanceOrig = st.number_input(
        "New Balance Origin",
        min_value=0.0,
        value=9000.0
    )

    oldbalanceDest = st.number_input(
        "Old Balance Destination",
        min_value=0.0,
        value=5000.0
    )

    newbalanceDest = st.number_input(
        "New Balance Destination",
        min_value=0.0,
        value=6000.0
    )

# =========================
# Prediction
# =========================

if st.button("🚀 Predict Fraud", use_container_width=True):

    # Feature Engineering

    log_amount = np.log1p(amount)

    is_high_amount = int(amount > 200000)

    is_night = int(hour >= 22 or hour <= 5)

    orig_balance_diff = (
        oldbalanceOrg - newbalanceOrig
    )

    dest_balance_diff = (
        oldbalanceDest - newbalanceDest
    )

    orig_error = (
        oldbalanceOrg
        - amount
        - newbalanceOrig
    )

    dest_error = (
        newbalanceDest
        - oldbalanceDest
        - amount
    )

    amount_to_balance_ratio = (
        amount / (oldbalanceOrg + 1)
    )

    data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "hour": hour,
        "is_night": is_night,
        "orig_balance_diff": orig_balance_diff,
        "dest_balance_diff": dest_balance_diff,
        "orig_error": orig_error,
        "dest_error": dest_error,
        "amount_to_balance_ratio": amount_to_balance_ratio
    }])

    data = data[features]

    scaled_data = scaler.transform(data)

    probability = model.predict_proba(
        scaled_data
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"🚨 FRAUD DETECTED\n\n"
            f"Probability: {probability:.4%}"
        )

    else:

        st.success(
            f"✅ LEGITIMATE TRANSACTION\n\n"
            f"Probability: {probability:.4%}"
        )

    st.metric(
        "Fraud Probability",
        f"{probability:.4%}"
    )

    st.metric(
        "Decision Threshold",
        f"{threshold:.4%}"
    )

    # Feature Preview

    st.subheader("Generated Features")

    st.dataframe(data)

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "Built with Streamlit, XGBoost, SHAP and Scikit-Learn"
)