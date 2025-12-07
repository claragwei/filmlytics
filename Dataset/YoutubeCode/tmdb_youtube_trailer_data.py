import os
import requests
import pandas as pd
import csv
import time
import sys

# ==============================
# CONFIGURATION
# ==============================
TMDB_API_KEY = "API Key"
YOUTUBE_API_KEY = "API Key"

INPUT_CSV = os.path.expanduser("~/Desktop/STA 160/tmdb_cleaned.csv")
OUTPUT_CSV = os.path.expanduser("~/Desktop/STA 160/trailer_dataset.csv")

# Increase CSV field size limit (fixes large text fields)
csv.field_size_limit(sys.maxsize)


# ==============================
# FUNCTION: GET TRAILERS FROM TMDB
# ==============================
def get_tmdb_trailers(movie_id):
    """Fetch trailer information from TMDB."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ TMDB error for movie ID {movie_id}")
        return []

    videos = response.json().get("results", [])
    trailers = []

    for v in videos:
        if v.get("site", "").lower() == "youtube" and v.get("type", "").lower() == "trailer":
            trailers.append({
                "video_id": v.get("key"),
                "video_name": v.get("name"),
                "published_at": v.get("published_at", ""),
                "official": v.get("official", False)
            })

    return trailers


# ==============================
# FUNCTION: GET YOUTUBE METRICS
# ==============================
def get_youtube_metrics(video_id):
    """Fetch metadata and stats for a YouTube video."""
    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    )
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ YouTube error for video ID {video_id}")
        return {}

    data = response.json()
    if not data.get("items"):
        return {}

    item = data["items"][0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "tags": ", ".join(snippet.get("tags", [])) if snippet.get("tags") else "",
        "description": snippet.get("description", ""),
        "category_id": snippet.get("categoryId", ""),
        "favorite_count": stats.get("favoriteCount", ""),
        "view_count": stats.get("viewCount", ""),
        "like_count": stats.get("likeCount", ""),
        "comment_count": stats.get("commentCount", ""),
    }


# ==============================
# MAIN FUNCTION: BUILD DATASET
# ==============================
def build_dataset():
    """Read movie IDs and titles, fetch trailer + YouTube data, and save CSV."""
    print("📂 Loading input CSV...")
    try:
        df = pd.read_csv(
            INPUT_CSV,
            engine="python",          # more tolerant parser
            on_bad_lines="skip",      # skip malformed rows
            encoding="utf-8",         # ensure UTF-8 handling
        )
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    if "id" not in df.columns or "title" not in df.columns:
        print("❌ Input CSV must contain columns: 'id' and 'title'")
        return

    all_rows = []
    total = len(df)

    for idx, row in df.iterrows():
        movie_id = row["id"]
        movie_title = row["title"]

        print(f"\n🎬 [{idx+1}/{total}] Processing: {movie_title} (ID: {movie_id})")

        trailers = get_tmdb_trailers(movie_id)
        if not trailers:
            print(f"⚠️ No trailers found for {movie_title}")
            continue

        for t in trailers:
            yt_data = get_youtube_metrics(t["video_id"])
            trailer_url = f"https://www.youtube.com/watch?v={t['video_id']}"

            all_rows.append({
                "id": movie_id,
                "title": movie_title,
                "trailer_url": trailer_url,
                "video_name": t["video_name"],
                "published_at": t["published_at"],
                "official": t["official"],
                **yt_data
            })

        # optional short sleep to avoid API rate limits
        time.sleep(0.2)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Trailer dataset saved to {OUTPUT_CSV}")
    else:
        print("\n⚠️ No trailer data collected.")


# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
    build_dataset()
