"""
File: sql_handling.py
Purpose: MySQL database handling for ML & backend projects
Author: Khyati Sharma

This file demonstrates:
- Secure DB connection
- Reading SQL tables into pandas
- Data cleaning
- CRUD operations
- Bulk insert (fast)
"""

import mysql.connector
import pandas as pd
import logging


# ---------------- LOGGING SETUP ----------------
logging.basicConfig(level=logging.INFO)


# ---------------- CONNECT ----------------
def connect_db():
    """
    Connects to MySQL server and selects database
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="YourStrongPassword@123",   
            auth_plugin="mysql_native_password"
        )

        logging.info("Connected to MySQL successfully")

        # Force database selection
        cursor = conn.cursor()
        cursor.execute("USE abc")
        logging.info("Database 'abc' selected")

        return conn

    except Exception as e:
        logging.error(f"Connection failed: {e}")
        return None


# ---------------- LOAD TABLE ----------------
def load_table(conn, table_name):
    """
    Loads SQL table into pandas DataFrame
    """
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)

    logging.info(f"Table '{table_name}' loaded")
    return df


# ---------------- EDA ----------------
def explore_data(df):
    """
    Performs basic data exploration
    """
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nInfo:")
    print(df.info())
    print("\nSummary:")
    print(df.describe())


# ---------------- CLEANING ----------------
def handle_missing(df):
    """
    Handles missing values
    """
    print("\nMissing values before:")
    print(df.isnull().sum())

    df = df.fillna("Unknown")

    print("\nMissing values after:")
    print(df.isnull().sum())
    return df


def remove_duplicates(df):
    """
    Removes duplicate rows
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"\nDuplicates removed: {before-after}")
    return df


# ---------------- BULK INSERT ----------------
def bulk_insert(df, conn, table_name):
    """
    Inserts large data efficiently (production style)
    """
    cursor = conn.cursor()

    placeholders = ", ".join(["%s"] * len(df.columns))
    query = f"INSERT INTO {table_name} VALUES ({placeholders})"

    data = [tuple(row) for _, row in df.iterrows()]

    cursor.executemany(query, data)
    conn.commit()

    logging.info("Bulk insert completed")


# ---------------- CRUD ----------------
def fetch_filtered(conn, condition):
    """
    Fetch rows using SQL condition
    Example: salary > 50000
    """
    query = f"SELECT * FROM employee WHERE {condition}"
    df = pd.read_sql(query, conn)
    return df


def update_record(conn, column, old, new):
    """
    Updates records
    """
    cursor = conn.cursor()
    query = f"UPDATE employee SET {column}=%s WHERE {column}=%s"
    cursor.execute(query, (new, old))
    conn.commit()
    logging.info("Record updated")


def delete_record(conn, condition):
    """
    Deletes records
    """
    cursor = conn.cursor()
    query = f"DELETE FROM employee WHERE {condition}"
    cursor.execute(query)
    conn.commit()
    logging.info("Record deleted")


# ---------------- SAVE CLEAN DATA ----------------
def save_to_db(df, conn, table_name):
    """
    Creates new table & saves cleaned data
    """
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cols = ", ".join([f"{c} TEXT" for c in df.columns])
    cursor.execute(f"CREATE TABLE {table_name} ({cols})")

    bulk_insert(df, conn, table_name)
    logging.info(f"Cleaned data saved to '{table_name}'")


# ---------------- MAIN ----------------
def main():
    table_name = "employee"

    conn = connect_db()
    if conn is None:
        return

    # Load
    df = load_table(conn, table_name)

    # Explore
    explore_data(df)

    # Clean
    df = handle_missing(df)
    df = remove_duplicates(df)

    # Save
    save_to_db(df, conn, "cleaned_employee")

    # Advanced operations demo
    high_salary = fetch_filtered(conn, "salary > 50000")
    print("\nHigh salary employees:")
    print(high_salary.head())

    # update_record(conn, "department", "HR", "Human Resources")
    # delete_record(conn, "salary < 20000")

    conn.close()
    logging.info("Connection closed")


if __name__ == "__main__":
    main()
