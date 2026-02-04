"""
Feature Engineering: Scaling Utilities
Author: Khyati Sharma
Purpose: Scale numerical features for machine learning models

Includes:
- Standardization (StandardScaler)
- Normalization (MinMaxScaler)
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore


def standard_scale(X):
    """
    Apply standardization (Z-score scaling) to features.

    Transforms data to have:
    - Mean = 0
    - Standard deviation = 1

    Formula:
        X_scaled = (X - mean) / standard_deviation

    When to use:
    - Models that assume normally distributed data
    - Distance-based or gradient-based models

    Common use cases:
    - Logistic Regression
    - Linear Regression
    - Support Vector Machines (SVM)
    - PCA
    - Neural Networks

    Args:
        X (DataFrame or array-like): Numerical features

    Returns:
        Scaled array
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def minmax_scale(X):
    """
    Apply normalization (Min-Max scaling) to features.

    Transforms data to a fixed range, usually [0, 1].

    Formula:
        X_scaled = (X - min) / (max - min)

    When to use:
    - Features have known fixed bounds
    - You want to preserve relative distances
    - Data is not normally distributed

    Common use cases:
    - K-Nearest Neighbors (KNN)
    - Neural Networks
    - Image / pixel-based data
    - Distance-based models sensitive to scale

    Args:
        X (DataFrame or array-like): Numerical features

    Returns:
        Scaled array
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)
