import pandas as pd

"""
File: json_handling.py
Purpose: Comprehensive JSON data handling & preprocessing pipeline
Author: Khyati Sharma
"""

def load_json(file_path):
    """Load JSON file into DataFrame"""
    return pd.read_json(file_path)


def basic_eda(df):
    """Perform basic exploratory data analysis"""
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe(include='all'))
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nFirst rows:\n", df.head())
    print("\nLast rows:\n", df.tail())


def handle_list_columns(df):
    """Convert list-type columns to strings"""
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
    return df


def value_analysis(df):
    """Print unique values and value counts"""
    for col in df.columns:
        print(f"\nUnique values in '{col}':\n", df[col].unique())
        print(f"\nValue counts in '{col}':\n", df[col].value_counts())


def correlation_analysis(df):
    """Display correlation matrix"""
    numeric_df = df.select_dtypes(include=["number"])
    print("\nCorrelation Matrix:\n", numeric_df.corr())


def memory_profile(df):
    """Show memory usage"""
    print("\nMemory Usage:\n", df.memory_usage())


def chunk_processing(df, chunk_size=10000):
    """Process DataFrame in chunks"""
    print("\nChunk Processing:")
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        print(f"Chunk {i // chunk_size + 1} shape:", chunk.shape)


def compress_json(df, output_file):
    """Save & reload compressed JSON"""
    df.to_json(output_file, compression='gzip')
    df_compressed = pd.read_json(output_file, compression='gzip')
    print("\nCompressed DataFrame Shape:", df_compressed.shape)
    print(df_compressed.head())


def encode_categorical(df):
    """Encode categorical columns"""
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    print("\nEncoded DataFrame:\n", df.head())
    return df


def normalize_nested_json(df, output_file):
    """Flatten nested JSON"""
    nested_df = pd.json_normalize(df.to_dict(orient='records'))
    print("\nNested DataFrame Shape:", nested_df.shape)
    print(nested_df.head())
    nested_df.to_json(output_file, orient='records', lines=True)


def convert_datetime(df, date_columns):
    """Convert columns to datetime"""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    print("\nDatetime Converted Data:\n", df.head())
    return df


def main():
    file_path = r"C:\Users\itskh\Desktop\ML2026\python-foundations\ML_d2\Datasets\train.json"

    df = load_json(file_path)

    basic_eda(df)

    df = handle_list_columns(df)

    value_analysis(df)

    correlation_analysis(df)

    memory_profile(df)

    chunk_processing(df)

    compress_json(df, "train_compressed.json")

    df = encode_categorical(df)

    normalize_nested_json(df, "train_nested.json")

    date_cols = ['date_column1', 'date_column2']  # replace if present
    df = convert_datetime(df, date_cols)


if __name__ == "__main__":
    main()
