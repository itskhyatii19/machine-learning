"""
EDA: Titanic Survival Dataset
Author: Khyati Sharma

Purpose:
This script demonstrates the essential questions that should be asked
when first exploring any dataset, along with practical answers using code.

Dataset: Titanic (train.csv)
Target Variable: Survived
"""

import os
import pandas as pd

# ========== LOAD DATA ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "train.csv")

df = pd.read_csv(DATA_PATH)

# ============================================================
# QUESTION 1: How big is the dataset?
# Why?
# - Determines computational cost
# - Helps decide train-test strategy
# ============================================================
print("\nQ1. How big is the dataset?")
print("Shape (rows, columns):", df.shape)

# ============================================================
# QUESTION 2: What does the data look like?
# Why?
# - Helps understand feature meaning
# - Detects obvious data quality issues
# ============================================================
print("\nQ2. How does the data look?")
print(df.sample(5))

# ============================================================
# QUESTION 3: What are the data types of each column?
# Why?
# - Identifies numerical vs categorical features
# - Guides preprocessing and encoding decisions
# ============================================================
print("\nQ3. What are the data types?")
df.info()

# ============================================================
# QUESTION 4: Are there missing values?
# Why?
# - Missing data impacts model performance
# - Determines imputation or feature removal strategies
# ============================================================
print("\nQ4. Are there missing values?")
print(df.isnull().sum())

# ============================================================
# QUESTION 5: How does the data look statistically?
# Why?
# - Detects skewness and outliers
# - Understands feature distributions
# ============================================================
print("\nQ5. Statistical summary of numerical features")
print(df.describe())

# ============================================================
# QUESTION 6: Are there duplicate records?
# Why?
# - Duplicate rows can bias training
# - Indicates data collection issues
# ============================================================
print("\nQ6. Are there duplicate rows?")
print("Number of duplicate rows:", df.duplicated().sum())

# ============================================================
# QUESTION 7: How are features related to the target?
# Why?
# - Helps identify important predictors
# - Guides feature selection and modeling approach
# ============================================================
print("\nQ7. Correlation of numerical features with Survived")
print(
    df.corr(numeric_only=True)["Survived"]
    .sort_values(ascending=False)
)
