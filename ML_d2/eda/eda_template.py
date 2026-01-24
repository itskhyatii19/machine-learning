"""
EDA TEMPLATE
Author: Khyati Sharma
Purpose: Perform structured exploratory data analysis on any dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    return pd.read_csv(path)


def basic_overview(df):
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())


def missing_values(df):
    print("\nMissing Values:")
    print(df.isnull().sum())


def univariate_analysis(df):
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def correlation_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def run_eda(path):
    df = load_data(path)
    basic_overview(df)
    missing_values(df)
    univariate_analysis(df)
    correlation_analysis(df)


if __name__ == "__main__":
    run_eda("datasets/sample.csv")
