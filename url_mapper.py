import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
from urllib.parse import quote_plus

INPUT_CSV = "tmdb_english_movies.csv"       # CSV with a 'title' column (and optional 'year')
OUTPUT_JSON = "rt_movie_urls.json"
DELAY = 1.0                    # Delay between requests (seconds)
MAX_RETRIES = 3

# Load existing cache if present
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# Function: search Rotten Tomatoes via HTML search page
def search_rt_html(title, year=None):
    search_url = f"https://www.rottentomatoes.com/search?search={quote_plus(title)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/141.0.0.0 Safari/537.36"
        )
    }

    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(search_url, headers=headers, timeout=10)
            if res.status_code != 200:
                time.sleep(DELAY)
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            # Select first movie result
            result = soup.select_one("search-page-media-row a")
            if not result:
                # fallback selector for older layout
                result = soup.select_one("search-page-result a")
            if result:
                href = result.get("href")
                if href and href.startswith("/m/") or href.startswith("/tv/"):
                    return f"https://www.rottentomatoes.com{href}"
            return None
        except Exception:
            time.sleep(DELAY)
    return None

# Load input CSV
df = pd.read_csv(INPUT_CSV)
if "title" not in df.columns:
    raise ValueError("CSV must include a 'title' column.")
if "year" not in df.columns:
    df["year"] = None

total = len(df)
for i, row in df.iterrows():
    title = str(row["title"]).strip()
    year = row.get("year")

    if title in cache:
        print(f"[{i+1}/{total}] Cached: {title}")
        continue

    print(f"[{i+1}/{total}] Searching: {title}")
    url = search_rt_html(title, year)
    cache[title] = url or "N/A"

    # Save progress every 50 entries
    if (i + 1) % 10 == 0 or i + 1 == total:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"✅ Saved progress ({i+1}/{total})")

    time.sleep(DELAY)

print("\n🎉 Finished building Rotten Tomatoes URL cache!")
print(f"Saved to {OUTPUT_JSON}")
