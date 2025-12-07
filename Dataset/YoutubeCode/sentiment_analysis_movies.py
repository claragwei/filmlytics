#!/usr/bin/env python3
"""
sentiment_analysis_movies.py
----------------------------
This script analyzes sentiment for movie descriptions and critic reviews.

Requirements:
    pip install transformers torch pandas tqdm

Input:
    movies.csv  (must have columns: 'title', 'description', 'critic_reviews')

Output:
    movies_with_sentiment_scores.csv
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def main():
    # Load your dataset
    try:
        movies = pd.read_csv("movies.csv")
    except FileNotFoundError:
        print("❌ Error: 'movies.csv' not found in the current directory.")
        return

    # Verify required columns
    required_columns = {"description", "critic_reviews"}
    if not required_columns.issubset(movies.columns):
        print(f"❌ Error: Dataset must contain columns {required_columns}. Found: {movies.columns.tolist()}")
        return

    # Initialize Hugging Face sentiment pipeline
    print("🔍 Loading sentiment analysis model...")
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Define helper functions
    def get_sentiment_score(text):
        """Return sentiment polarity (-1 to +1) from text."""
        if not isinstance(text, str) or text.strip() == "":
            return 0.0
        result = sentiment_pipeline(text[:512])[0]
        score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        return score

    def get_sentiment_label(score):
        """Convert numeric score to POSITIVE / NEGATIVE / NEUTRAL label."""
        if score > 0:
            return "POSITIVE"
        elif score < 0:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    # Apply sentiment scoring
    tqdm.pandas(desc="Analyzing Description Sentiment")
    movies["sentiment_score_description"] = movies["description"].progress_apply(get_sentiment_score)

    tqdm.pandas(desc="Analyzing Critic Review Sentiment")
    movies["sentiment_score_critic"] = movies["critic_reviews"].progress_apply(get_sentiment_score)
    movies["critic_sentiment_label"] = movies["sentiment_score_critic"].apply(get_sentiment_label)

    # Save the new dataset
    output_path = "movies_with_sentiment_scores.csv"
    movies.to_csv(output_path, index=False)
    print(f"\n✅ Sentiment scores added and file saved as '{output_path}'")

    # Preview results
    print(movies[["title", "sentiment_score_description", "sentiment_score_critic", "critic_sentiment_label"]].head())

if __name__ == "__main__":
    main()
