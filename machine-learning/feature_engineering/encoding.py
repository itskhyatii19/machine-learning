"""
Feature Engineering: Encoding Utilities
Author: Khyati Sharma
Purpose: Encode categorical features for ML models
"""

import pandas as pd


def binary_encode(df, column, mapping=None):
    """
    Encode binary categorical column (e.g., Yes/No).

    Args:
        df (DataFrame)
        column (str)
        mapping (dict): Optional custom mapping

    Returns:
        DataFrame
    """
    df = df.copy()

    if mapping is None:
        mapping = {"Yes": 1, "No": 0}

    df[column] = df[column].map(mapping)
    return df
