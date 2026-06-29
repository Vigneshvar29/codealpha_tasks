# Task 3 - Sentiment Analysis

Analyzes IMDB movie reviews using NLP to classify them as Positive, Negative, or Neutral.
Combines a lexicon-based approach (VADER) with a Machine Learning model (Logistic Regression).

## What it does

- Cleans and preprocesses raw review text
- Runs VADER sentiment scoring on each review
- Trains a Logistic Regression classifier using TF-IDF features
- Generates a full visual dashboard with 6 charts
- Predicts sentiment on any new review you give it

## Charts in the dashboard

- Sentiment distribution (bar chart)
- Sentiment share (pie chart)
- VADER score distribution (histogram)
- Positive reviews word cloud
- Negative reviews word cloud
- Confusion matrix (model evaluation)

## Tech used

- Python
- NLTK (VADER lexicon)
- Scikit-learn (TF-IDF + Logistic Regression)
- Matplotlib & Seaborn
- WordCloud

## How to run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

## Output

- `sentiment_dashboard.png` — visual dashboard
- `results.csv` — all reviews with sentiment scores and labels

---
*Built as Task 4/4 for the CodeAlpha Internship.*
