"""
File: linear_regression.py
Purpose: Production-style Linear Regression pipeline
Author: Khyati Sharma
 """

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from Regression.metrics import regression_metrics



# ---------------- LOAD ----------------
def load_data(path):
    """Load dataset"""
    df = pd.read_csv(path)
    return df


# ---------------- PREP ----------------
def prepare_data(df, target_col):
    """Split features & target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ---------------- SPLIT ----------------
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


# ---------------- TRAIN ----------------
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---------------- MAIN ----------------
def main():
    path = "../datasets/student.csv"   # change file name
    target = "marks"                   # change target

    df = load_data(path)

    X, y = prepare_data(df, target)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    preds = model.predict(X_test)

    regression_metrics(y_test, preds)

    joblib.dump(model, "linear_model.pkl")
    print("\nModel saved as linear_model.pkl")


if __name__ == "__main__":
    main()
