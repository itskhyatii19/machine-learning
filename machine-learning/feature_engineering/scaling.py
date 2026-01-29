"""
Feature Engineering: Scaling Utilities
Author: Khyati Sharma
Purpose: Scale numerical features for ML models
"""

from sklearn.preprocessing import StandardScaler


def standard_scale(X):
    """
    Apply standard scaling to features.

    Args:
        X (DataFrame or array)

    Returns:
        Scaled array
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)
