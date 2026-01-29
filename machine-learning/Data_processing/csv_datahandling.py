"""
File: csv_datahandling.py
Purpose: Advanced CSV handling for real ML projects
Author: Khyati Sharma
"""

import pandas as pd


# Load dataset
def load_data(path):
    """Load CSV file"""
    df = pd.read_csv(path)
    print("\nDataset loaded successfully")
    return df


# Basic exploration
def explore_data(df):
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nInfo:")
    print(df.info())
    print("\nSummary:")
    print(df.describe())


# Handle missing values
def handle_missing(df):
    """
    Fill or drop missing values
    """
    print("\nMissing values before:")
    print(df.isnull().sum())

    df = df.fillna("Unknown")   # simple strategy
    print("\nMissing values after:")
    print(df.isnull().sum())

    return df


# Remove duplicates
def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"\nDuplicates removed: {before-after}")
    return df


# Text cleaning (important for your project)
def clean_text_column(df, col):
    """
    Lowercase + remove extra spaces
    """
    df[col] = df[col].str.lower().str.strip()
    return df


# Feature engineering
def create_feature(df):
    """
    Combine multiple columns into one
    (for ML vectorization later)
    """
    if "skills" in df.columns and "role" in df.columns:
        df["combined_text"] = df["skills"] + " " + df["role"]
        print("\nNew feature: combined_text created")

    return df


# Export cleaned data
def save_clean_data(df):
    df.to_csv("cleaned_data.csv", index=False)
    print("\nCleaned dataset saved!")


def main():
    df = load_data("jobs.csv")   # replace with your real dataset

    explore_data(df)

    df = handle_missing(df)
    df = remove_duplicates(df)

    # Clean text columns
    if "skills" in df.columns:
        df = clean_text_column(df, "skills")

    df = create_feature(df)

    save_clean_data(df)


if __name__ == "__main__":
    main()