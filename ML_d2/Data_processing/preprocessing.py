"""
File: preprocessing.py
Purpose: Feature preprocessing utilities for ML pipelines
Author: Khyati Sharma

Covers:
- Train-test split
- Scaling
- Encoding
- Outlier handling
- Class imbalance handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


# ---------------- SPLIT ----------------
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into train & test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Data split completed")
    return X_train, X_test, y_train, y_test


# ---------------- SCALING ----------------
def scale_features(X_train, X_test, method="standard"):
    """
    Scale numerical features
    method: standard / minmax
    """
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling done")
    return X_train_scaled, X_test_scaled, scaler


# ---------------- ENCODING ----------------
def encode_target(y_train, y_test):
    """
    Encode target labels
    """
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    print("Target encoding done")
    return y_train_enc, y_test_enc, encoder


# ---------------- OUTLIER HANDLING ----------------
def remove_outliers(df, col):
    """
    Remove outliers using IQR
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean_df = df[(df[col] >= lower) & (df[col] <= upper)]

    print("Outliers removed")
    return clean_df


# ---------------- IMBALANCE ----------------
def balance_classes(X, y):
    """
    Handle class imbalance using SMOTE
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("Class imbalance handled using SMOTE")
    return X_res, y_res


# ---------------- FEATURE SELECTION ----------------
def select_features(df, target_col, corr_threshold=0.3):
    """
    Select features based on correlation
    """
    corr = df.corr()[target_col].abs()
    selected = corr[corr > corr_threshold].index.tolist()
    selected.remove(target_col)

    print("Selected features:", selected)
    return df[selected], df[target_col]


# ---------------- MAIN (TEST) ----------------
def main():
    df = pd.read_csv("datasets/cleaned_data.csv")

    X, y = select_features(df, "target")

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test
    )

    y_train_enc, y_test_enc, enc = encode_target(y_train, y_test)

    X_bal, y_bal = balance_classes(X_train_scaled, y_train_enc)


def encode_extracurricular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 'Extracurricular Activities' column (Yes/No → 1/0)
    """
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    return df


def scale_features(X: pd.DataFrame):
    """
    Scale numerical features using StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled



if __name__ == "__main__":
    main()
