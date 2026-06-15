 Codealpha_WebScraper

A multi-page web scraper built with **BeautifulSoup** and **Requests** that collects structured book data from a public website and exports it as a clean CSV dataset.

 Features

- Scrapes book titles, prices, star ratings, and availability across multiple pages
- Handles pagination automatically
- Exports results to `books_dataset.csv`
- Includes polite crawl delays and error handling
- Sample preview printed to console after scraping

 Tech Stack

| Tool | Purpose |
|------|---------|
| `requests` | HTTP page fetching |
| `BeautifulSoup4` | HTML parsing & data extraction |
| `csv` (stdlib) | Dataset export |

 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the scraper
```bash
python scraper.py
```

### 3. Output
- Console: live progress + sample of scraped data  
- File: `books_dataset.csv` with all collected records

 Project Structure

```
Codealpha_WebScraper/
├── scraper.py          # Main scraper script
├── requirements.txt    # Dependencies
├── books_dataset.csv   # Output dataset (generated on run)
└── README.md
```

Sample Output (CSV)

| title | price | rating | availability |
|-------|-------|--------|--------------|
| A Light in the Attic | £51.77 | 3 | In stock |
| Tipping the Velvet | £53.74 | 1 | In stock |

 Configuration

In `scraper.py`, adjust `max_pages` in the `scrape()` call:
```python
dataset = scrape(max_pages=5)   # Change to scrape more/fewer pages (max 50)
```

 Ethical Scraping

- Always check a site's `robots.txt` before scraping
- This project uses [books.toscrape.com](https://books.toscrape.com) — a site built specifically for scraping practice
- A 1-second delay between requests is included to avoid server overload

---
*Built as Task 1/4 for the CodeAlpha Internship.*
