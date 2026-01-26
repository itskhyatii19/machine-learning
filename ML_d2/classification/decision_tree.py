"""
Classification: Decision Tree – Student Performance
Author: Khyati Sharma
Purpose: Classify student performance into categories using Decision Tree
"""

import sys
import os

# ========== PATH SETUP (MUST COME FIRST) ==========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from Data_processing.preprocessing import encode_extracurricular

# ========== PATH ==========
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "Datasets", "StudentPerformance.csv")

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)

# ========== CREATE CLASS LABEL ==========
def categorize_performance(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

df["Performance Category"] = df["Performance Index"].apply(categorize_performance)

# ========== FEATURES & TARGET ==========
X = df.drop(columns=["Performance Index", "Performance Category"])
y = df["Performance Category"]

# Encode categorical feature (shared preprocessing)
X = encode_extracurricular(X)

# ========== TRAIN TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ========== MODEL ==========
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ========== EVALUATION ==========
y_pred = model.predict(X_test)

print("Decision Tree – Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
