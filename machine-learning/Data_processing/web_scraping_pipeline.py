"""
File: web_scraping_pipeline.py
Author: Khyati Sharma

--------
Demonstrates a complete, production-style web scraping pipeline using
requests + BeautifulSoup. This script is written for ML learners and
open-source contributors who need clean, structured datasets.

Key Learning Outcomes:
----------------------
1. How to safely download web pages using headers
2. How to inspect and navigate HTML structures
3. Container-first scraping (industry best practice)
4. Robust error handling for missing fields
5. Pagination handling
6. ML-ready DataFrame creation

IMPORTANT:
----------
Always check a website's robots.txt and terms of service before scraping.
This script is for educational purposes only.
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

import time
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


# =========================
# 2. GLOBAL CONFIGURATION
# =========================

BASE_URL = "https://www.ambitionbox.com/list-of-companies?page={}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 1  # seconds (ethical scraping)


# =========================
# 3. HELPER FUNCTION
# =========================

def get_soup(url: str) -> BeautifulSoup:
    """
    Downloads a webpage and returns a BeautifulSoup object.

    Parameters
    ----------
    url : str
        Target webpage URL

    Returns
    -------
    BeautifulSoup
        Parsed HTML document

    Raises
    ------
    HTTPError if request fails
    """
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


# =========================
# 4. DATA EXTRACTION LOGIC
# =========================

def extract_company_data(company_card) -> dict:
    """
    Extracts structured data from a single company container.

    Parameters
    ----------
    company_card : bs4.element.Tag
        HTML container holding company information

    Returns
    -------
    dict
        Extracted company attributes
    """

    def safe_text(selector_func):
        """Utility to safely extract text or return NaN."""
        try:
            return selector_func().text.strip()
        except Exception:
            return np.nan

    info_entities = company_card.find_all("p", class_="infoEntity")

    return {
        "name": safe_text(lambda: company_card.find("h2")),
        "rating": safe_text(lambda: company_card.find("p", class_="rating")),
        "reviews": safe_text(lambda: company_card.find("a", class_="review-count")),
        "company_type": info_entities[0].text.strip() if len(info_entities) > 0 else np.nan,
        "headquarters": info_entities[1].text.strip() if len(info_entities) > 1 else np.nan,
        "company_age": info_entities[2].text.strip() if len(info_entities) > 2 else np.nan,
        "employee_count": info_entities[3].text.strip() if len(info_entities) > 3 else np.nan,
    }


# =========================
# 5. PAGINATION SCRAPER
# =========================

def scrape_companies(start_page: int = 1, end_page: int = 5) -> pd.DataFrame:
    """
    Scrapes company data across multiple pages.

    Parameters
    ----------
    start_page : int
        Starting page number
    end_page : int
        Ending page number (inclusive)

    Returns
    -------
    pd.DataFrame
        Consolidated company dataset
    """

    all_records = []

    for page in range(start_page, end_page + 1):
        print(f"[INFO] Scraping page {page}")

        try:
            soup = get_soup(BASE_URL.format(page))
        except Exception as e:
            print(f"[ERROR] Failed to fetch page {page}: {e}")
            continue

        company_cards = soup.find_all("div", class_="company-content-wrapper")

        for card in company_cards:
            record = extract_company_data(card)
            all_records.append(record)

        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(all_records)


# =========================
# 6. MAIN EXECUTION
# =========================

if __name__ == "__main__":

    df = scrape_companies(start_page=1, end_page=10)

    print("\n[INFO] Dataset preview:")
    print(df.head())

    print("\n[INFO] Dataset shape:", df.shape)

    # Dataset is now ML-ready:
    # - Rows = samples
    # - Columns = features
    # - Missing values handled as NaN
