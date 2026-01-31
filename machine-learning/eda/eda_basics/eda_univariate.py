"""
EDA: Titanic Dataset – Univariate Analysis
Author: Khyati Sharma

Purpose:
Demonstrate how to perform univariate analysis by asking
the right questions for categorical and numerical features.

Dataset: Titanic (train.csv)
Target Variable: Survived
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== LOAD DATA (PORTABLE PATH) ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "train.csv")

df = pd.read_csv(DATA_PATH)

# ============================================================
# UNIVARIATE ANALYSIS – CATEGORICAL VARIABLES
# ============================================================

# QUESTION 1: What is the distribution of categorical features?
# Why?
# - Identifies class imbalance
# - Helps choose encoding strategies
# - Important for classification problems

print("\nUnivariate Analysis: Categorical Variables")

# Embarked Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Embarked", data=df)
plt.title("Passenger Distribution by Embarkation Port")
plt.show()

# Sex Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
df["Sex"].value_counts().plot(
    kind="pie",
    autopct="%.2f",
    startangle=90
)
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# ============================================================
# UNIVARIATE ANALYSIS – NUMERICAL VARIABLES
# ============================================================

# QUESTION 2: How are numerical values distributed?
# Why?
# - Detects skewness
# - Identifies outliers
# - Guides scaling and transformation

print("\nUnivariate Analysis: Numerical Variables")

# Age Histogram
plt.figure(figsize=(6, 4))
plt.hist(df["Age"].dropna(), bins=5)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution (Histogram)")
plt.show()

# Age Distribution Plot
plt.figure(figsize=(6, 4))
sns.histplot(df["Age"].dropna(), kde=True)
plt.title("Age Distribution with Density Curve")
plt.show()

# Age Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot (Outlier Detection)")
plt.show()

# ============================================================
# QUESTION 3: What are the key statistical properties?
# Why?
# - Summarizes central tendency
# - Helps detect skewness
# ============================================================

print("\nAge Statistics")
print("Minimum Age:", df["Age"].min())
print("Maximum Age:", df["Age"].max())
print("Mean Age:", df["Age"].mean())
print("Skewness:", df["Age"].skew())
