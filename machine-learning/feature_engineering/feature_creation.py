"""
Feature Creation Module
Author: Khyati Sharma

Purpose:
Create new features based on insights from EDA
to improve model learning.
"""

import pandas as pd


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create FamilySize feature from SibSp and Parch
    """
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IsAlone feature
    """
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into categories based on distribution
    """
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 20, 40, 60, 100],
        labels=["Child", "Teen", "Adult", "MidAge", "Senior"]
    )
    return df
