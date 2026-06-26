"""
Task 4 - Sentiment Analysis
============================
Analyzes IMDB movie reviews using NLP techniques to classify
sentiments as Positive, Negative, or Neutral.

Techniques used:
- Text preprocessing (cleaning, stopword removal)
- VADER Sentiment Analysis (lexicon-based NLP)
- Machine Learning (Logistic Regression)
- Visualizations: bar chart, pie chart, word clouds, confusion matrix

Dataset: IMDB Movie Reviews (built-in sample + sklearn dataset)
Output:  sentiment_dashboard.png, results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

# ── Sample IMDB-style reviews dataset ────────────────────────────────────
SAMPLE_REVIEWS = [
    ("This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat.", "positive"),
    ("One of the best films I've ever seen. Brilliant direction and outstanding performances.", "positive"),
    ("Incredible cinematography and a deeply moving storyline. A true masterpiece.", "positive"),
    ("Great movie! Loved every minute of it. The characters felt real and relatable.", "positive"),
    ("A wonderful experience from start to finish. Highly recommend to everyone.", "positive"),
    ("Amazing plot twists and excellent cast. Will definitely watch again.", "positive"),
    ("The film exceeded all my expectations. Emotionally powerful and visually stunning.", "positive"),
    ("Beautifully crafted with a touching story. One of the year's best.", "positive"),
    ("Loved the humor and heart in this film. A feel-good movie that delivers.", "positive"),
    ("Exceptional storytelling with great pacing. Could not take my eyes off the screen.", "positive"),
    ("Terrible movie. Waste of time and money. The plot made no sense at all.", "negative"),
    ("Awful acting and boring storyline. I almost fell asleep halfway through.", "negative"),
    ("One of the worst movies I have ever seen. Completely disappointing and predictable.", "negative"),
    ("Poor direction and weak script. The characters were completely unconvincing.", "negative"),
    ("Dreadful film with no redeeming qualities. Do not bother watching this.", "negative"),
    ("The worst cinema experience I have had in years. Totally unwatchable.", "negative"),
    ("Bad acting, terrible plot, and painfully slow pacing. A complete disaster.", "negative"),
    ("Deeply disappointing. The trailers were far better than the actual film.", "negative"),
    ("Horrible movie with no clear story. The ending made absolutely no sense.", "negative"),
    ("Nothing worked in this film. Poor writing, poor acting, poor everything.", "negative"),
    ("The movie was okay. Not great, not bad. Just an average film overall.", "neutral"),
    ("Decent watch for a lazy afternoon but nothing memorable or special.", "neutral"),
    ("It was fine. Some parts were enjoyable but others dragged on too long.", "neutral"),
    ("Average film with a few good moments but nothing to write home about.", "neutral"),
    ("Not bad but not great either. Pretty standard for this type of movie.", "neutral"),
    ("The film had its moments but overall felt a bit underwhelming.", "neutral"),
    ("Some good scenes here and there but the overall experience was mediocre.", "neutral"),
    ("A passable film. You won't regret watching it but won't remember it either.", "neutral"),
    ("Middle of the road movie. Had potential but didn't quite deliver.", "neutral"),
    ("Watchable but forgettable. Nothing stands out as particularly good or bad.", "neutral"),
    # Extra positive
    ("Absolutely loved it! Best movie of the year without a doubt.", "positive"),
    ("Stunning visuals and a gripping story. Completely blown away.", "positive"),
    ("Heartwarming and hilarious. The perfect family movie.", "positive"),
    # Extra negative
    ("I want my two hours back. Absolutely dreadful experience.", "negative"),
    ("Painfully bad. The script felt like it was written in an hour.", "negative"),
    ("Regret watching this. One of the most boring films in recent memory.", "negative"),
]

# ── Text Cleaning ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


# ── Load Data ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("[✓] Loading IMDB movie reviews dataset...")
    df = pd.DataFrame(SAMPLE_REVIEWS, columns=["review", "sentiment"])
    df["cleaned"] = df["review"].apply(clean_text)
    print(f"    Total reviews: {len(df)}")
    print(f"    Distribution:\n{df['sentiment'].value_counts().to_string()}\n")
    return df


# ── VADER Sentiment Scoring ───────────────────────────────────────────────
def vader_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("[✓] Running VADER sentiment analysis...")
    sia = SentimentIntensityAnalyzer()
    df["vader_score"] = df["review"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["vader_label"] = df["vader_score"].apply(
        lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
    )
    acc = (df["vader_label"] == df["sentiment"]).mean()
    print(f"    VADER Accuracy: {acc:.2%}\n")
    return df


# ── ML Model ─────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    print("[✓] Training Logistic Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tf, y_train)
    y_pred = model.predict(X_test_tf)

    acc = accuracy_score(y_test, y_pred)
    print(f"    Model Accuracy: {acc:.2%}")
    print(f"\n    Classification Report:\n{classification_report(y_test, y_pred)}")
    return model, tfidf, y_test, y_pred


# ── Dashboard ─────────────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, y_test, y_pred):
    print("[✓] Generating dashboard...")
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
    sentiment_counts = df["sentiment"].value_counts()

    # Title
    fig.text(0.5, 0.97, "IMDB Movie Reviews — Sentiment Analysis",
             ha="center", fontsize=18, fontweight="bold", color="white")
    fig.text(0.5, 0.935, "NLP-powered classification using VADER + Logistic Regression",
             ha="center", fontsize=11, color="#aaaaaa")

    # ── Chart 1: Sentiment Distribution Bar ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a2e")
    bars = ax1.bar(sentiment_counts.index,
                   sentiment_counts.values,
                   color=[COLORS[s] for s in sentiment_counts.index],
                   edgecolor="#0f0f1a", linewidth=1.5, width=0.5)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(int(bar.get_height())), ha="center", color="white", fontsize=11, fontweight="bold")
    ax1.set_title("Sentiment Distribution", color="white", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Reviews", color="#aaaaaa")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333355")
    ax1.set_facecolor("#1a1a2e")
    for spine in ax1.spines.values():
        spine.set_color("#333355")

    # ── Chart 2: Pie Chart ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a2e")
    wedges, texts, autotexts = ax2.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=[COLORS[s] for s in sentiment_counts.index],
        startangle=90,
        wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2}
    )
    for t in texts: t.set_color("white")
    for at in autotexts: at.set_color("white"); at.set_fontsize(10)
    ax2.set_title("Sentiment Share", color="white", fontsize=12, fontweight="bold", pad=10)

    # ── Chart 3: VADER Score Distribution ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1a1a2e")
    for sentiment, color in COLORS.items():
        subset = df[df["sentiment"] == sentiment]["vader_score"]
        ax3.hist(subset, bins=8, alpha=0.7, color=color, label=sentiment, edgecolor="#0f0f1a")
    ax3.axvline(0.05, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax3.axvline(-0.05, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_title("VADER Score Distribution", color="white", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Compound Score", color="#aaaaaa")
    ax3.set_ylabel("Count", color="#aaaaaa")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for spine in ax3.spines.values(): spine.set_color("#333355")

    # ── Chart 4: Positive Word Cloud ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#1a1a2e")
    pos_text = " ".join(df[df["sentiment"] == "positive"]["cleaned"])
    wc_pos = WordCloud(width=400, height=250, background_color="#1a1a2e",
                       colormap="Greens", max_words=50).generate(pos_text)
    ax4.imshow(wc_pos, interpolation="bilinear")
    ax4.axis("off")
    ax4.set_title("Positive Reviews — Key Words", color="#2ecc71", fontsize=12, fontweight="bold", pad=10)

    # ── Chart 5: Negative Word Cloud ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#1a1a2e")
    neg_text = " ".join(df[df["sentiment"] == "negative"]["cleaned"])
    wc_neg = WordCloud(width=400, height=250, background_color="#1a1a2e",
                       colormap="Reds", max_words=50).generate(neg_text)
    ax5.imshow(wc_neg, interpolation="bilinear")
    ax5.axis("off")
    ax5.set_title("Negative Reviews — Key Words", color="#e74c3c", fontsize=12, fontweight="bold", pad=10)

    # ── Chart 6: Confusion Matrix ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1a1a2e")
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax6, cbar=False,
                linewidths=0.5, linecolor="#0f0f1a")
    ax6.set_title("Confusion Matrix", color="white", fontsize=12, fontweight="bold", pad=10)
    ax6.set_xlabel("Predicted", color="#aaaaaa")
    ax6.set_ylabel("Actual", color="#aaaaaa")
    ax6.tick_params(colors="white")
    for spine in ax6.spines.values(): spine.set_color("#333355")

    plt.savefig("sentiment_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[✓] Dashboard saved → sentiment_dashboard.png")


# ── Predict new review ────────────────────────────────────────────────────
def predict_review(review: str, model, tfidf):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(review)["compound"]
    cleaned = clean_text(review)
    ml_pred = model.predict(tfidf.transform([cleaned]))[0]
    vader_pred = "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
    print(f"\n  Review   : {review}")
    print(f"  VADER    : {vader_pred} (score: {score:.3f})")
    print(f"  ML Model : {ml_pred}")


# ── Save Results ──────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame):
    df[["review", "sentiment", "vader_score", "vader_label"]].to_csv("results.csv", index=False)
    print("[✓] Results saved → results.csv")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  IMDB Sentiment Analysis — CodeAlpha Task 4")
    print("=" * 60 + "\n")

    df = load_data()
    df = vader_analysis(df)
    model, tfidf, y_test, y_pred = train_model(df)
    plot_dashboard(df, y_test, y_pred)
    save_results(df)

    print("\n── Live Prediction Demo ──────────────────────────────────")
    predict_review("This movie was absolutely brilliant! I loved every second.", model, tfidf)
    predict_review("Terrible film. Complete waste of time. Very disappointing.", model, tfidf)
    predict_review("It was an okay movie. Nothing special but not bad either.", model, tfidf)

    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
