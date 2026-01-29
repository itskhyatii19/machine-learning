"""
File: api_data_collection_pipeline.py
Author: Khyati Sharma

Purpose:
--------
Demonstrates a complete, ML-ready API data collection pipeline using
The Movie Database (TMDB) API.

Key Learning Outcomes:
----------------------
1. Understanding REST APIs and JSON responses
2. Handling API authentication via query parameters
3. Pagination handling (critical for large datasets)
4. Defensive programming for unreliable API responses
5. Creating clean, ML-ready datasets

IMPORTANT:
----------
Never hardcode API keys in public repositories.
Use environment variables or config files instead.
(API key here is for learning/demo purposes only.)
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

import time
import requests
import pandas as pd
import numpy as np


# =========================
# 2. GLOBAL CONFIGURATION
# =========================

BASE_URL = "https://api.themoviedb.org/3/movie/top_rated"

API_KEY = "YOUR_API_KEY_HERE"  # use env vars in real projects

PARAMS = {
    "api_key": API_KEY,
    "language": "en-US",
}

REQUEST_DELAY = 0.3  # polite API usage


# =========================
# 3. HELPER FUNCTION
# =========================

def fetch_page(page: int) -> dict:
    """
    Fetches a single page of results from the TMDB API.

    Parameters
    ----------
    page : int
        Page number to fetch

    Returns
    -------
    dict
        Parsed JSON response

    Raises
    ------
    HTTPError if request fails
    """

    params = PARAMS.copy()
    params["page"] = page

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


# =========================
# 4. DATA EXTRACTION LOGIC
# =========================

def extract_movie_fields(movie: dict) -> dict:
    """
    Extracts relevant ML features from a single movie record.

    Parameters
    ----------
    movie : dict
        Raw movie JSON object

    Returns
    -------
    dict
        Cleaned movie features
    """

    return {
        "id": movie.get("id", np.nan),
        "title": movie.get("title", np.nan),
        "overview": movie.get("overview", np.nan),
        "release_date": movie.get("release_date", np.nan),
        "popularity": movie.get("popularity", np.nan),
        "vote_average": movie.get("vote_average", np.nan),
        "vote_count": movie.get("vote_count", np.nan),
    }


# =========================
# 5. PAGINATION PIPELINE
# =========================

def collect_movies(start_page: int = 1, end_page: int = 5) -> pd.DataFrame:
    """
    Collects movie data across multiple API pages.

    Parameters
    ----------
    start_page : int
        First page to fetch
    end_page : int
        Last page to fetch (inclusive)

    Returns
    -------
    pd.DataFrame
        Consolidated movie dataset
    """

    records = []

    for page in range(start_page, end_page + 1):
        print(f"[INFO] Fetching page {page}")

        try:
            data = fetch_page(page)
            results = data.get("results", [])
        except Exception as e:
            print(f"[ERROR] Page {page} failed: {e}")
            continue

        for movie in results:
            records.append(extract_movie_fields(movie))

        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(records)


# =========================
# 6. MAIN EXECUTION
# =========================

if __name__ == "__main__":

    df = collect_movies(start_page=1, end_page=428)

    print("\n[INFO] Dataset preview:")
    print(df.head())

    print("\n[INFO] Dataset shape:", df.shape)

    # Save dataset for ML workflows
    df.to_csv("movies_top_rated.csv", index=False)
