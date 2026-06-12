"""
Codealpha_WebScraper
====================
A multi-purpose web scraper that collects structured data from public websites.
Demonstrates use of BeautifulSoup for HTML parsing and requests for HTTP fetching.

Targets: books.toscrape.com — a scraping-friendly practice site.
Collected data: Book title, price, rating, availability
Output: books_dataset.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URL = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"
OUTPUT_FILE = "books_dataset.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a page and return a BeautifulSoup object, or None on failure."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; educational-scraper/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def parse_books(soup: BeautifulSoup) -> list[dict]:
    """Extract all books from a single catalogue page."""
    books = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"]
        price = article.select_one("p.price_color").get_text(strip=True)
        availability = article.select_one("p.availability").get_text(strip=True)
        rating_class = article.p["class"][1]          # e.g. "Three"
        rating = RATING_MAP.get(rating_class, 0)
        books.append({
            "title": title,
            "price": price,
            "rating": rating,
            "availability": availability
        })
    return books


def get_next_page(soup: BeautifulSoup) -> str | None:
    """Return the absolute URL of the next page, or None if last page."""
    next_btn = soup.select_one("li.next > a")
    if next_btn:
        return BASE_URL + next_btn["href"]
    return None


def save_to_csv(data: list[dict], filepath: str):
    """Save the dataset to a CSV file."""
    if not data:
        print("[WARN] No data to save.")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "price", "rating", "availability"])
        writer.writeheader()
        writer.writerows(data)
    print(f"\n[✓] Dataset saved → {filepath}  ({len(data)} records)")


def scrape(max_pages: int = 5) -> list[dict]:
    """
    Scrape books across multiple catalogue pages.

    Args:
        max_pages: Maximum number of pages to scrape (default 5).
    Returns:
        List of book dicts.
    """
    all_books = []
    url = START_URL
    page = 1

    print(f"Starting scrape (max {max_pages} pages)...\n")

    while url and page <= max_pages:
        print(f"  Scraping page {page}: {url}")
        soup = fetch_page(url)
        if not soup:
            break

        books = parse_books(soup)
        all_books.extend(books)
        print(f"    → {len(books)} books collected")

        url = get_next_page(soup)
        page += 1
        time.sleep(1)  # Polite crawl delay

    print(f"\nTotal books scraped: {len(all_books)}")
    return all_books


def display_sample(data: list[dict], n: int = 5):
    """Print a sample of the scraped data."""
    print(f"\n{'='*65}")
    print(f"  SAMPLE OUTPUT (first {min(n, len(data))} records)")
    print(f"{'='*65}")
    for i, row in enumerate(data[:n], 1):
        stars = "★" * row["rating"] + "☆" * (5 - row["rating"])
        print(f"\n[{i}] {row['title']}")
        print(f"    Price: {row['price']}  |  Rating: {stars}  |  {row['availability']}")
    print(f"\n{'='*65}")


if __name__ == "__main__":
    dataset = scrape(max_pages=5)
    display_sample(dataset)
    save_to_csv(dataset, OUTPUT_FILE)
