# src/feature_engineering.py

import pandas as pd

def feature_engineering(df):
    """
    Performs feature engineering on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """

    df['hour'] = df['step'] % 24  # Hour of the day
    df['amount_ratio_old_origin'] = df['amount'] / (df['oldbalanceOrg'] + 1e-9)  # Avoid division by zero
    df['amount_ratio_new_origin'] = df['amount'] / (df['newbalanceOrig'] + 1e-9)
    df['amount_ratio_old_dest'] = df['amount'] / (df['oldbalanceDest'] + 1e-9)
    df['amount_ratio_new_dest'] = df['amount'] / (df['newbalanceDest'] + 1e-9)
    df['balance_diff_origin'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    df['is_movement_consistent'] = ((df['balance_diff_origin'] < 0) & (df['balance_diff_dest'] > 0)) | ((df['balance_diff_origin'] > 0) & (df['balance_diff_dest'] < 0))
    df['is_movement_consistent'] = df['is_movement_consistent'].astype(int)

    df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    return df