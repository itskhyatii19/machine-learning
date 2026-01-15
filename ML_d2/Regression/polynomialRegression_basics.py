"""
File: polynomialRegression_basics.py
Purpose: Polynomial Regression with evaluation & visualization
Author: Khyati Sharma

Workflow:
1. Create dataset
2. Split features & target
3. Train-test split
4. Generate polynomial features
5. Train model
6. Evaluate performance
7. Visualize curve
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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


def generate_polynomial_features(X_train, X_test, degree=3):
    """Generate polynomial features"""
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    return poly, X_train_poly, X_test_poly


def train_model(X_train_poly, y_train):
    """Train polynomial regression model"""
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model


def evaluate_model(y_test, predictions):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nPolynomial Regression Evaluation")
    print("MAE:", mae)
    print("R² Score:", r2)


def visualize_results(X, y, poly, model):
    """Plot polynomial curve"""
    X_full_poly = poly.transform(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual Data")
    plt.plot(
        X,
        model.predict(X_full_poly),
        linewidth=2,
        label="Polynomial Curve"
    )

    plt.xlabel("Hours Studied")
    plt.ylabel("Marks")
    plt.title("Polynomial Regression")
    plt.legend()
    plt.show()


def main():
    df = create_dataset()
    print("\nDataset:\n", df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_data(X, y)

    poly, X_train_poly, X_test_poly = generate_polynomial_features(
        X_train, X_test, degree=3
    )

    model = train_model(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    evaluate_model(y_test, predictions)

    visualize_results(X, y, poly, model)


if __name__ == "__main__":
    main()
