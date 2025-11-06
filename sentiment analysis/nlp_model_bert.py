import json
import pandas as pd
from tqdm import tqdm
import os
from transformers import pipeline

# =========================
# Setup
# =========================
INPUT_JSON = "rt scraping/rt_reviews.json"
PROGRESS_JSON = "sentiment analysis/rottentomatoes_sentiment_progress.json"
OUTPUT_CSV = "sentiment analysis/rottentomatoes_sentiment_bert.csv"

# =========================
# Load Hugging Face Sentiment Model
# =========================
print("Loading DistilBERT sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # use CPU; change to 0 if running on GPU
)

# =========================
# Load Data
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    movies = json.load(f)

# Load progress if exists
if os.path.exists(PROGRESS_JSON):
    with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

# Convert to dict for faster lookups
processed_titles = {m["title"] for m in results}
print(f"Resuming from {len(processed_titles)} movies already processed...")

# =========================
# Helper: Compute BERT Sentiment
# =========================
def get_sentiment_bert(texts):
    """Compute average sentiment score (-1 to 1) for a list of texts"""
    preds = sentiment_pipeline(texts, truncation=True, max_length=512)
    # Convert labels to numeric sentiment
    scores = []
    for p in preds:
        if p["label"].upper() == "POSITIVE":
            scores.append(p["score"])
        elif p["label"].upper() == "NEGATIVE":
            scores.append(-p["score"])
        else:
            scores.append(0)
    return sum(scores) / len(scores) if scores else "N/A"

# =========================
# Process Movies
# =========================
for movie in tqdm(movies, desc="Analyzing sentiment (BERT)"):
    title = movie.get("title", "N/A")
    if title in processed_titles:
        continue

    reviews = movie.get("reviews", [])
    if not reviews:
        avg_sentiment = "N/A"
    else:
        # Process in smaller batches for efficiency
        batch_size = 8
        batch_scores = []
        for i in range(0, len(reviews), batch_size):
            batch = [r for r in reviews[i:i + batch_size] if isinstance(r, str)]
            if batch:
                batch_score = get_sentiment_bert(batch)
                batch_scores.append(batch_score)
        avg_sentiment = sum(batch_scores) / len(batch_scores) if batch_scores else "N/A"

    results.append({
        "title": title,
        "tomatometer": movie.get("tomatometer", "N/A"),
        "audience_score": movie.get("audience_score", "N/A"),
        "sentiment": avg_sentiment,
        "num_reviews": len(reviews),
        "url": movie.get("url", "")
    })

    # Save progress every 100 movies
    if len(results) % 100 == 0:
        with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved progress after {len(results)} movies")

# =========================
# Final Save
# =========================
with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save as CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, quoting=1)

print("\nDone!")
print(f"Processed {len(results)} movies total.")
print(f"Saved results to {OUTPUT_CSV}")
