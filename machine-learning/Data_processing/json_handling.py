"""
File: json_handling.py
Purpose: Production-style JSON preprocessing pipeline
Author: Khyati Sharma
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


# ---------------- LOAD ----------------
def load_json(file_path):
    """Load JSON file safely"""
    try:
        df = pd.read_json(file_path)
        logging.info("JSON loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return None


# ---------------- EDA ----------------
def basic_eda(df):
    logging.info("Performing EDA")

    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSample:\n", df.head())


# ---------------- LIST HANDLING ----------------
def handle_list_columns(df):
    """Convert list columns to string"""
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        logging.info("List columns processed")
    return df


# ---------------- VALUE ANALYSIS ----------------
def value_analysis(df):
    for col in df.columns:
        print(f"\nColumn: {col}")
        print("Unique:", df[col].nunique())
        print(df[col].value_counts().head())


# ---------------- CORRELATION ----------------
def correlation_analysis(df):
    num_df = df.select_dtypes(include=["number"])
    print("\nCorrelation matrix:\n", num_df.corr())


# ---------------- MEMORY ----------------
def memory_profile(df):
    print("\nMemory usage:\n", df.memory_usage(deep=True))


# ---------------- CHUNK PROCESS ----------------
def chunk_processing(df, size=10000):
    logging.info("Processing in chunks")
    for i in range(0, len(df), size):
        chunk = df.iloc[i:i + size]
        print(f"Chunk {i//size + 1} -> {chunk.shape}")


# ---------------- COMPRESSION ----------------
def compress_json(df, output_file):
    df.to_json(output_file, compression="gzip")
    logging.info("Compressed JSON saved")

    reloaded = pd.read_json(output_file, compression="gzip")
    print("\nReloaded compressed data:\n", reloaded.head())


# ---------------- ENCODING ----------------
def encode_categorical(df):
    """Encode categorical columns"""
    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
        encoders[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    logging.info("Categorical encoding completed")
    return df, encoders


# ---------------- NORMALIZE ----------------
def normalize_nested_json(df, output_file):
    nested_df = pd.json_normalize(df.to_dict("records"))
    nested_df.to_json(output_file, orient="records", lines=True)
    logging.info("Nested JSON flattened & saved")
    return nested_df


# ---------------- DATETIME ----------------
def convert_datetime(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logging.info("Datetime conversion done")
    return df


# ---------------- MAIN ----------------
def main():
    file_path = "datasets/train.json"   # use relative path

    df = load_json(file_path)
    if df is None:
        return

    basic_eda(df)

    df = handle_list_columns(df)

    value_analysis(df)

    correlation_analysis(df)

    memory_profile(df)

    chunk_processing(df)

    compress_json(df, "datasets/train_compressed.json")

    df, encoders = encode_categorical(df)

    nested_df = normalize_nested_json(df, "datasets/train_nested.json")

    date_cols = ["created_at", "updated_at"]   # change if needed
    df = convert_datetime(df, date_cols)

    logging.info("JSON pipeline completed successfully")


if __name__ == "__main__":
    main()
