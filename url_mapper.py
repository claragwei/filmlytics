import requests
import pandas as pd
import json
import time
import os

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
INPUT_CSV = "movies.csv"       # Your CSV with a "title" column (and optional "year")
OUTPUT_JSON = "rt_movie_urls.json"
MAX_RETRIES = 3
DELAY = 1.0                    # seconds between API calls

# ------------------------------------------------------
# LOAD EXISTING CACHE
# ------------------------------------------------------
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# ------------------------------------------------------
# FUNCTION: Search Rotten Tomatoes for a movie
# ------------------------------------------------------
def search_rt_url(title, year=None):
    base_url = "https://www.rottentomatoes.com/napi/search/"
    params = {"query": title}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/141.0.0.0 Safari/537.36"
        )
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(DELAY)
                continue

            data = response.json()
            movies = data.get("movies", [])
            if not movies:
                return None

            # Try to pick the best match by title/year similarity
            for m in movies:
                name = m.get("name", "").lower()
                url = m.get("url", "")
                release_year = m.get("releaseYear", None)
                if name == title.lower() or (year and str(release_year) == str(year)):
                    return f"https://www.rottentomatoes.com{url}"

            # fallback: return the first movie match
            return f"https://www.rottentomatoes.com{movies[0]['url']}"
        except Exception:
            time.sleep(DELAY)
    return None

# ------------------------------------------------------
# LOAD INPUT CSV
# ------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
if "title" not in df.columns:
    raise ValueError("Your CSV must include a 'title' column.")
if "year" not in df.columns:
    df["year"] = None

# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
total = len(df)
for i, row in df.iterrows():
    title = str(row["title"]).strip()
    year = row.get("year")

    if title in cache:
        print(f"[{i+1}/{total}] Cached: {title}")
        continue

    print(f"[{i+1}/{total}] Searching: {title}")
    url = search_rt_url(title, year)
    cache[title] = url or "N/A"

    # Save progress every 50 entries
    if (i + 1) % 50 == 0 or i + 1 == total:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Saved progress ({i+1}/{total})")

    time.sleep(DELAY)

print("\nFinished building Rotten Tomatoes URL cache!")
print(f"Saved to {OUTPUT_JSON}")
