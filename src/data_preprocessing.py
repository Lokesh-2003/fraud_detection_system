# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def encode_categorical(df, columns):
    """Encodes categorical columns using one-hot encoding."""
    return pd.get_dummies(df, columns=columns, drop_first=True)

def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def apply_smote(X_train, y_train, random_state=42):
    """Applies SMOTE to the training data to handle class imbalance."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def scale_features(X_train, X_test, numerical_cols):
    """Scales numerical features using StandardScaler."""

    # Avoid scaling non-numerical columns
    cols_to_scale = [col for col in numerical_cols if col in X_train.columns and col in X_test.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    return X_train, X_test