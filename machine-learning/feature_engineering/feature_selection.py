"""
Feature Engineering: Feature Selection
Author: Khyati Sharma
Purpose: Select important features using statistical methods
"""

from sklearn.feature_selection import SelectKBest, f_regression


def select_top_k_features(X, y, k=5):
    """
    Select top k features based on correlation with target.

    Args:
        X (array-like)
        y (array-like)
        k (int)

    Returns:
        Reduced feature matrix
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    return selector.fit_transform(X, y)
