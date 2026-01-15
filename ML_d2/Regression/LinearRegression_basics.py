"""
File: LinearRegression_basics.py
Purpose: End-to-end Linear Regression workflow using scikit-learn
Author: Khyati Sharma

Steps:
1. Create dataset
2. Split features & target
3. Train-test split
4. Train model
5. Evaluate performance
6. Visualize results
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def create_dataset():
    """Create sample dataset"""
    data = {
        "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8],
        "marks": [35, 40, 50, 60, 70, 75, 80, 85]
    }
    return pd.DataFrame(data)


def split_features_target(df):
    """Separate features and target"""
    X = df[["hours_studied"]]
    y = df["marks"]
    return X, y


def train_test_data(X, y):
    """Perform train-test split"""
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, predictions):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R² Score:", r2)


def visualize_results(X, y, X_test, predictions):
    """Plot regression results"""
    plt.figure(figsize=(8, 5))

    plt.scatter(X, y, label="Actual Data")
    plt.plot(
        X_test,
        predictions,
        linewidth=2,
        label="Regression Line"
    )

    plt.xlabel("Hours Studied")
    plt.ylabel("Marks Obtained")
    plt.title("Linear Regression: Study Hours vs Marks")
    plt.legend()
    plt.show()


def main():
    df = create_dataset()
    print("\nDataset:\n", df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_data(X, y)

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nPredictions:", predictions)
    print("Actual:", y_test.values)

    evaluate_model(y_test, predictions)

    visualize_results(X, y, X_test, predictions)


if __name__ == "__main__":
    main()
