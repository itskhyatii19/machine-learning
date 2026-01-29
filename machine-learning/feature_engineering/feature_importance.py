"""
Feature Engineering: Feature Importance
Author: Khyati Sharma
Purpose: Extract and analyze feature importance from models
"""

import pandas as pd


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models.

    Args:
        model: trained model with feature_importances_
        feature_names (list)

    Returns:
        DataFrame sorted by importance
    """
    importance = model.feature_importances_

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
