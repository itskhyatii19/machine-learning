"""
Feature Engineering: Encoding Utilities
Author: Khyati Sharma
Purpose: Encode categorical features for machine learning models

Includes:
- Binary Encoding
- Ordinal Encoding
- Label Encoding
- One-Hot Encoding
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder # type: ignore

def label_encode(df, column):
    df = df.copy()
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df, encoder


def binary_encode(df, column, mapping=None):
    """
    Binary encode a categorical column (e.g., Yes/No).

    Example:
        Yes → 1
        No  → 0
    """
    df = df.copy()

    if mapping is None:
        mapping = {"Yes": 1, "No": 0}

    df[column] = df[column].map(mapping)
    return df


def ordinal_encode(df, column, order):
    """
    Ordinal encode a categorical column with meaningful order.

    Example:
        order = ["Low", "Medium", "High"]
    """
    df = df.copy()
    ordinal_map = {value: idx for idx, value in enumerate(order)}
    df[column] = df[column].map(ordinal_map)
    return df


def label_encode(df, column):
    """
    Label encode a categorical column (no inherent order).

    Suitable for:
    - Tree-based models
    - Target variables
    """
    df = df.copy()
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df, encoder


def one_hot_encode(df, columns, drop_first=True):
    """
    One-hot encode categorical columns.

    Suitable for:
    - Linear models
    - Logistic Regression
    - Distance-based models
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    return df
