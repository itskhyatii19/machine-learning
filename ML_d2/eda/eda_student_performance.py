"""
EDA: Student Performance Dataset
Author: Khyati Sharma
Purpose: Understand the dataset and save visual insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== PATH SETUP ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Datasets", "StudentPerformance.csv")
VISUALS_PATH = os.path.join(BASE_DIR, "visuals")

os.makedirs(VISUALS_PATH, exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())

TARGET = "Performance Index"

# ========== HELPER FUNCTION ==========
def save_plot(filename):
    plt.savefig(os.path.join(VISUALS_PATH, filename), bbox_inches="tight")
    plt.show()
    plt.close()

# ========== 1. TARGET DISTRIBUTION ==========
plt.figure()
sns.histplot(df[TARGET], kde=True)
plt.title("Performance Index Distribution")
save_plot("performance_index_distribution.png")

# ========== 2. STUDY HOURS VS PERFORMANCE ==========
plt.figure()
sns.scatterplot(
    x="Hours Studied",
    y=TARGET,
    data=df
)
plt.title("Hours Studied vs Performance Index")
save_plot("hours_studied_vs_performance.png")

# ========== 3. EXTRACURRICULAR ACTIVITIES ==========
plt.figure()
sns.boxplot(
    x="Extracurricular Activities",
    y=TARGET,
    data=df
)
plt.title("Extracurricular Activities vs Performance Index")
save_plot("extracurricular_vs_performance.png")

# ========== 4. HOURS STUDIED ==========
plt.figure()
sns.scatterplot(
    x="Hours Studied",
    y=TARGET,
    data=df
)
plt.title("Hours Studied vs Performance Index")
save_plot("hours_studied_vs_performance.png")

# ========== 5. SLEEP HOURS ==========
plt.figure()
sns.scatterplot(
    x="Sleep Hours",
    y=TARGET,
    data=df
)
plt.title("Sleep Hours vs Performance Index")
save_plot("sleep_hours_vs_performance.png")

# ========== 6. CORRELATION ==========
plt.figure(figsize=(10, 6))

numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap (Numeric Features)")
save_plot("correlation_heatmap.png")

print("EDA completed successfully. Visuals saved in eda/visuals/")