# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
try:
    model = joblib.load('models/best_xgb_model.joblib')  # Adjust path if needed
    scaler = joblib.load('models/scaler.joblib')  # Adjust path if needed
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure they are in the 'models/' directory.")
    st.stop()

def preprocess_input(data):
    """Preprocesses input data for prediction."""
    df = pd.DataFrame([data])
    # One-hot encode 'type' (match training)
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    # Ensure all columns are present (handle rare 'type' values)
    expected_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                       'isFlaggedFraud', 'type_CASH_IN', 'type_CASH_OUT', 'type_PAYMENT', 'type_TRANSFER',
                       'hour', 'amount_ratio_old_origin', 'amount_ratio_new_origin', 'amount_ratio_old_dest',
                       'amount_ratio_new_dest', 'balance_diff_origin', 'balance_diff_dest', 'is_movement_consistent',
                       'is_large_transaction']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df

st.title('Fraud Detection System')

# Create input fields for features
step = st.number_input('Step', value=1)
type = st.selectbox('Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount = st.number_input('Amount', value=0.0)
oldbalanceOrg = st.number_input('Old Balance Origin', value=0.0)
newbalanceOrig = st.number_input('New Balance Origin', value=0.0)
oldbalanceDest = st.number_input('Old Balance Destination', value=0.0)
newbalanceDest = st.number_input('New Balance Destination', value=0.0)
isFlaggedFraud = st.number_input('Is Flagged Fraud', value=0)

if st.button('Predict'):
    input_data = {'step': step, 'type': type, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
                  'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest,
                  'newbalanceDest': newbalanceDest, 'isFlaggedFraud': isFlaggedFraud}
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    st.write('Prediction:', 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent')