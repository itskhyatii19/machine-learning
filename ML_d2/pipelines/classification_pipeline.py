"""
ML Pipeline: Classification – Student Performance
Author: Khyati Sharma
Purpose: Build a reusable classification pipeline
"""

import sys
import os

# ========== PATH SETUP ==========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ========== PATH ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Datasets", "StudentPerformance.csv")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

# ========== TARGET ENGINEERING ==========
def categorize_performance(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

df["Performance Category"] = df["Performance Index"].apply(categorize_performance)

X = df.drop(columns=["Performance Index", "Performance Category"])
y = df["Performance Category"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== COLUMN TYPES ==========
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = ["Extracurricular Activities"]

# ========== PREPROCESSOR ==========
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", "passthrough", categorical_features),
    ]
)

# Convert Yes/No to 1/0 before pipeline
X["Extracurricular Activities"] = X["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# ========== PIPELINE ==========
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=500))
    ]
)

# ========== TRAIN TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ========== TRAIN ==========
pipeline.fit(X_train, y_train)

# ========== EVALUATE ==========
y_pred = pipeline.predict(X_test)

print("Pipeline Classification Report")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
