"""
EDA: Bivariate & Multivariate Analysis
Author: Khyati Sharma

Purpose:
Demonstrate how to analyze relationships between:
- Numerical vs Numerical
- Numerical vs Categorical
- Categorical vs Categorical
- Multi-feature interactions

Datasets Used:
- Titanic (train.csv)
- Tips (seaborn)
- Iris (seaborn)
- Flights (seaborn)
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# ============================================================
# LOAD DATASETS
# ============================================================

# Titanic (local dataset)
BASE_DIR = os.path.dirname(__file__)
TITANIC_PATH = os.path.join(BASE_DIR, "..", "datasets", "train.csv")
titanic = pd.read_csv(TITANIC_PATH)

# Seaborn datasets
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
flights = sns.load_dataset("flights")

# ============================================================
# 1. NUMERICAL vs NUMERICAL
# QUESTION:
# How do two numerical variables relate to each other?
# Why?
# - Detects linear/non-linear relationships
# - Helps decide regression feasibility
# ============================================================

plt.figure(figsize=(6, 4))
sns.scatterplot(
    x="total_bill",
    y="tip",
    hue="sex",
    style="smoker",
    size="size",
    data=tips
)
plt.title("Total Bill vs Tip (with categorical context)")
plt.show()

# ============================================================
# 2. NUMERICAL vs CATEGORICAL (BAR PLOT)
# QUESTION:
# How does a numerical variable vary across categories?
# Why?
# - Compares group-wise averages
# - Highlights group disparities
# ============================================================

plt.figure(figsize=(6, 4))
sns.barplot(
    x="Pclass",
    y="Age",
    hue="Sex",
    data=titanic
)
plt.title("Average Age by Passenger Class and Gender")
plt.show()

# ============================================================
# 3. NUMERICAL vs CATEGORICAL (BOX PLOT)
# QUESTION:
# How is a numerical variable distributed across categories?
# Why?
# - Detects outliers per group
# - Compares medians and spread
# ============================================================

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="Sex",
    y="Age",
    hue="Survived",
    data=titanic
)
plt.title("Age Distribution by Gender and Survival")
plt.show()

# ============================================================
# 4. NUMERICAL vs CATEGORICAL (DISTRIBUTION)
# QUESTION:
# How does distribution differ between outcome classes?
# Why?
# - Useful for classification boundary intuition
# ============================================================

plt.figure(figsize=(6, 4))
sns.kdeplot(
    titanic[titanic["Survived"] == 0]["Age"],
    label="Did Not Survive"
)
sns.kdeplot(
    titanic[titanic["Survived"] == 1]["Age"],
    label="Survived"
)
plt.title("Age Distribution by Survival Status")
plt.legend()
plt.show()

# ============================================================
# 5. CATEGORICAL vs CATEGORICAL (HEATMAP)
# QUESTION:
# How do two categorical variables interact?
# Why?
# - Reveals joint distributions
# - Useful for contingency analysis
# ============================================================

plt.figure(figsize=(6, 4))
sns.heatmap(
    pd.crosstab(titanic["Pclass"], titanic["Survived"]),
    annot=True,
    cmap="coolwarm"
)
plt.title("Survival Count by Passenger Class")
plt.show()

# Survival rate by embarkation port
print("\nSurvival Rate by Embarkation Port (%)")
print(
    titanic.groupby("Embarked")["Survived"].mean() * 100
)

# ============================================================
# 6. CLUSTER MAP (CATEGORICAL vs CATEGORICAL)
# QUESTION:
# Are there natural groupings between categories?
# Why?
# - Identifies similarity patterns
# ============================================================

sns.clustermap(
    pd.crosstab(titanic["Parch"], titanic["Survived"]),
    cmap="viridis"
)
plt.title("Cluster Map: Parents/Children vs Survival")
plt.show()

# ============================================================
# 7. MULTIVARIATE RELATIONSHIPS (PAIRPLOT)
# QUESTION:
# How do multiple numerical features interact together?
# Why?
# - Visual feature separability
# - Model selection insight
# ============================================================

sns.pairplot(
    iris,
    hue="species"
)
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# ============================================================
# 8. TIME SERIES ANALYSIS (LINE PLOT)
# QUESTION:
# How does a numerical variable change over time?
# Why?
# - Detects trends and seasonality
# ============================================================

yearly_passengers = flights.groupby("year")["passengers"].sum().reset_index()

plt.figure(figsize=(6, 4))
sns.lineplot(
    x="year",
    y="passengers",
    data=yearly_passengers
)
plt.title("Yearly Air Passengers Trend")
plt.show()

# Monthly-seasonal clustering
sns.clustermap(
    flights.pivot_table(
        values="passengers",
        index="month",
        columns="year"
    ),
    cmap="coolwarm"
)
plt.title("Passenger Trends by Month and Year")
plt.show()
