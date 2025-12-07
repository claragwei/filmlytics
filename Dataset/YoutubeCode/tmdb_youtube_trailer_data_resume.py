import pandas as pd
import requests
import json
import os
import sys
import csv

# ===== CONFIG =====
TMDB_API_KEY = "API Key"
YOUTUBE_API_KEY = "Api Key"

INPUT_CSV = os.path.expanduser("~/Desktop/STA 160/tmdb_ids.csv")
OUTPUT_CSV = os.path.expanduser("~/Desktop/STA 160/trailer_dataset.csv")
PROGRESS_FILE = os.path.expanduser("~/Desktop/STA 160/trailer_progress.json")

# Allow reading large CSV safely
csv.field_size_limit(sys.maxsize)

# ===== LOAD/SAVE PROGRESS =====
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_ids": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

# ===== TMDB TRAILERS =====
def get_tmdb_trailers(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ TMDB error for movie ID {movie_id}")
        return []

    videos = response.json().get("results", [])
    trailers = []
    for v in videos:
        if v.get("site","").lower() == "youtube" and v.get("type","").lower() == "trailer":
            trailers.append({
                "video_id": v.get("key"),
                "video_name": v.get("name"),
                "published_at": v.get("published_at",""),
                "official": v.get("official", False)
            })
    return trailers

# ===== YOUTUBE METRICS =====
def get_youtube_metrics(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    
    # Quota limit handling
    if response.status_code in [403, 429]:
        print("🚫 YouTube quota reached. Pausing...")
        raise Exception("quota_exceeded")
    
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
        "description": snippet.get("description",""),
        "category_id": snippet.get("categoryId",""),
        "favorite_count": stats.get("favoriteCount",""),
        "view_count": stats.get("viewCount",""),
        "like_count": stats.get("likeCount",""),
        "comment_count": stats.get("commentCount",""),
    }

# ===== BUILD DATASET =====
def build_dataset():
    try:
        df = pd.read_csv(INPUT_CSV, usecols=["id"])
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    progress = load_progress()
    completed_ids = set(progress.get("completed_ids", []))
    all_rows = []

    for _, row in df.iterrows():
        movie_id = row["id"]

        if movie_id in completed_ids:
            continue

        print(f"Processing ID: {movie_id}")

        trailers = get_tmdb_trailers(movie_id)
        if not trailers:
            completed_ids.add(movie_id)
            progress["completed_ids"] = list(completed_ids)
            save_progress(progress)
            continue

        for t in trailers:
            try:
                yt_data = get_youtube_metrics(t["video_id"])
            except Exception as e:
                if "quota_exceeded" in str(e):
                    print("🛑 Quota limit reached. Saving progress and exiting...")
                    if all_rows:
                        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
                    progress["completed_ids"] = list(completed_ids)
                    save_progress(progress)
                    sys.exit(0)
                else:
                    print(f"Error fetching YouTube data: {e}")
                    continue

            trailer_url = f"https://www.youtube.com/watch?v={t['video_id']}"
            all_rows.append({
                "id": movie_id,
                "trailer_url": trailer_url,
                "video_name": t["video_name"],
                "published_at": t["published_at"],
                "official": t["official"],
                **yt_data
            })

        completed_ids.add(movie_id)
        progress["completed_ids"] = list(completed_ids)
        save_progress(progress)

        # Save progress every 200 trailers
        if len(all_rows) % 200 == 0:
            pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Progress saved ({len(all_rows)} trailers)")

    # Final save
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Trailer dataset saved to {OUTPUT_CSV}")
    else:
        print("⚠️ No trailer data collected.")

# ===== RUN =====
if __name__ == "__main__":
    build_dataset()
