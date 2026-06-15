"""
Task 2 - Data Visualization
============================
Visualizes book data scraped in Task 1 using Matplotlib and Seaborn.
Generates 4 charts saved as a single PNG dashboard.

Dataset: books_dataset.csv (from Task 1 Web Scraper)
Output: books_dashboard.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Config ──────────────────────────────────────────────────────────────
DATA_FILE = "books_dataset.csv"
OUTPUT_FILE = "books_dashboard.png"
sns.set_theme(style="darkgrid", palette="muted")

# ── Load or generate sample data ─────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"[✓] Loaded {len(df)} records from {DATA_FILE}")
    else:
        print("[INFO] books_dataset.csv not found — using sample data for demo")
        df = pd.DataFrame({
            "title": [f"Book {i}" for i in range(1, 51)],
            "price": [round(10 + (i * 1.3) % 45, 2) for i in range(50)],
            "rating": [((i % 5) + 1) for i in range(50)],
            "availability": ["In stock"] * 42 + ["Out of stock"] * 8
        })
    # Clean price column (remove £ symbol)
    df["price"] = df["price"].astype(str).str.replace("£", "", regex=False).astype(float)
    return df


def plot_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Books Dataset — Visual Analysis", fontsize=18, fontweight="bold", y=0.98)

    colors = sns.color_palette("muted", 5)

    # ── Chart 1: Rating Distribution (Bar) ──────────────────────────────
    ax1 = axes[0, 0]
    rating_counts = df["rating"].value_counts().sort_index()
    bars = ax1.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_title("Rating Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Star Rating")
    ax1.set_ylabel("Number of Books")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels(["★", "★★", "★★★", "★★★★", "★★★★★"])
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(int(bar.get_height())), ha="center", fontsize=10)

    # ── Chart 2: Price Distribution (Histogram) ──────────────────────────
    ax2 = axes[0, 1]
    ax2.hist(df["price"], bins=15, color=colors[1], edgecolor="white", linewidth=0.8)
    ax2.set_title("Price Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Price (£)")
    ax2.set_ylabel("Number of Books")
    ax2.axvline(df["price"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: £{df['price'].mean():.2f}")
    ax2.legend()

    # ── Chart 3: Top 10 Most Expensive Books (Horizontal Bar) ────────────
    ax3 = axes[1, 0]
    top10 = df.nlargest(10, "price")[["title", "price"]].reset_index(drop=True)
    top10["short_title"] = top10["title"].str[:30] + "..."
    ax3.barh(top10["short_title"], top10["price"], color=colors[2], edgecolor="white")
    ax3.set_title("Top 10 Most Expensive Books", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Price (£)")
    ax3.invert_yaxis()

    # ── Chart 4: Availability (Pie) ───────────────────────────────────────
    ax4 = axes[1, 1]
    avail_counts = df["availability"].value_counts()
    ax4.pie(avail_counts.values, labels=avail_counts.index, autopct="%1.1f%%",
            colors=[colors[3], colors[4]], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax4.set_title("Stock Availability", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"[✓] Dashboard saved → {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print(f"\nDataset Summary:\n{df.describe()}\n")
    plot_dashboard(df)
