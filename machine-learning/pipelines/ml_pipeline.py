"""
File: ml_pipeline.py
Purpose: End-to-end ML training & evaluation pipeline
Author: Khyati Sharma

Steps:
1. Load dataset
2. Preprocess data
3. Train model
4. Evaluate
5. Save model
"""

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ML_d2.Data_processing.preprocessing import (
    split_data,
    scale_features,
    encode_target,
    balance_classes
)



# ---------------- LOAD ----------------
def load_data(path):
    """
    Load dataset
    """
    df = pd.read_csv(path)
    print("Dataset loaded")
    return df


# ---------------- PREP ----------------
def prepare_data(df, target_col):
    """
    Split features & target
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


# ---------------- TRAIN ----------------
def train_model(X_train, y_train):
    """
    Train ML model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Model trained")
    return model


# ---------------- EVALUATE ----------------
def evaluate(model, X_test, y_test):
    """
    Evaluate model
    """
    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n",
          classification_report(y_test, preds))


# ---------------- SAVE ----------------
def save_model(model):
    """
    Save trained model
    """
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")


# ---------------- MAIN ----------------
def main():
    path = "datasets/cleaned_data.csv"
    target_col = "target"   # replace

    df = load_data(path)

    X, y = prepare_data(df, target_col)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Encode target
    y_train_enc, y_test_enc, encoder = encode_target(
        y_train, y_test
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test
    )

    # Handle imbalance
    X_bal, y_bal = balance_classes(
        X_train_scaled, y_train_enc
    )

    # Train
    model = train_model(X_bal, y_bal)

    # Evaluate
    evaluate(model, X_test_scaled, y_test_enc)

    # Save
    save_model(model)


if __name__ == "__main__":
    main()
