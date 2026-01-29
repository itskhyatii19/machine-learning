"""
Regression: Polynomial Regression – Student Performance
Author: Khyati Sharma
Purpose: Capture non-linear relationships using Polynomial Regression
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data_processing.preprocessing import (
    encode_extracurricular,
    scale_features
)

# ========== PATH ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Datasets", "StudentPerformance.csv")

df = pd.read_csv(DATA_PATH)

TARGET = "Performance Index"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = encode_extracurricular(X)
X_scaled = scale_features(X)

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Polynomial Regression Evaluation")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
