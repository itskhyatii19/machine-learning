"""
EDA: Automated Data Profiling with Pandas Profiling
Author: Khyati Sharma

Purpose:
Demonstrate how automated EDA tools can be used to quickly
generate a comprehensive data quality and exploratory report.

Dataset: Titanic (train.csv)

"""
"""
NOTE:
Automated profiling tools like ydata-profiling currently do not
support Python 3.13+.

This script is provided for conceptual completeness.
To run:
- Use Python 3.10 or 3.11
- OR create a virtual environment with a supported version
"""

import os
import pandas as pd
from ydata_profiling import ProfileReport


# ============================================================
# LOAD DATA (PORTABLE PATH)
# ============================================================

BASE_DIR = os.path.dirname(__file__)               # eda/eda_basics
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")

df = pd.read_csv(DATA_PATH)

# ============================================================
# QUESTION:
# Can we quickly generate a holistic overview of the dataset?
#
# Why use pandas profiling?
# - Rapid data quality checks
# - Missing value analysis
# - Feature distributions
# - Correlation insights
# - Early warnings (duplicates, skewness, imbalance)
# ============================================================

# ========== GENERATE PROFILE ==========
profile = ProfileReport(
    df,
    title="Titanic Dataset - Automated EDA Report",
    explorative=True
)

# ========== SAVE REPORT ==========
OUTPUT_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "titanic_pandas_profile.html"
)

profile.to_file(OUTPUT_PATH)

print(f"Profiling report saved to: {OUTPUT_PATH}")
