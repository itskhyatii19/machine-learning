"""
Regression: Linear Regression – Student Performance
Author: Khyati Sharma
Purpose: Predict student performance index using Linear Regression
"""

import sys
import os

# ========== PATH SETUP ==========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data_processing.preprocessing import (
    encode_extracurricular,
    scale_features
)

# ========== PATH ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Datasets", "StudentPerformance.csv")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

TARGET = "Performance Index"

# ========== FEATURES & TARGET ==========
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode categorical feature
X = encode_extracurricular(X)

# Scale features (required for Linear Regression)
X_scaled = scale_features(X)

# ========== TRAIN TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ========== MODEL ==========
model = LinearRegression()
model.fit(X_train, y_train)

# ========== EVALUATION ==========
y_pred = model.predict(X_test)

print("Linear Regression Evaluation")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
