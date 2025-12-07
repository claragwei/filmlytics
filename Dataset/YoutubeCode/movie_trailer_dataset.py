import os
import requests
import pandas as pd
from datetime import datetime

# ==== CONFIGURATION ====
TMDB_API_KEY = "API Key"
YOUTUBE_API_KEY = "API Key"

INPUT_CSV = "tmdb_filtered_2.csv"                 # Input movie list file
OUTPUT_CSV = "movie_trailer_dataset.csv"      # Output dataset file


# ==== FUNCTION: SEARCH MOVIE ON TMDB ====
def search_movie(movie_name):
    """Search TMDB for a movie and return basic info."""
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_name}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        movie = data["results"][0]
        return {
            "id": movie["id"],
            "title": movie["title"],
            "release_date": movie.get("release_date", "")
        }
    return None

# ==== FUNCTION: GET TRAILERS FROM TMDB ====
def get_trailers(movie_id):
    """Retrieve all YouTube trailers for a given movie."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    response = requests.get(url, params=params)
    videos = response.json().get("results", [])

    trailers = []
    for v in videos:
        if v.get("site", "").lower() == "youtube" and v.get("type", "").lower() == "trailer":
            trailers.append({
                "video_id": v["key"],
                "video_name": v.get("name", ""),
                "published_at": v.get("published_at", ""),
                "official": v.get("official", False),
                "type": v.get("type", "")
            })
    return trailers

# ==== FUNCTION: GET YOUTUBE METRICS ====
def get_youtube_metrics(video_id):
    """Fetch metrics and metadata from YouTube API."""
    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "items" not in data or not data["items"]:
        return {}

    item = data["items"][0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "youtube_title": snippet.get("title"),
        "youtube_description": snippet.get("description"),
        "tags": ", ".join(snippet.get("tags", [])) if snippet.get("tags") else "",
        "category_id": snippet.get("categoryId"),
        "favorite_count": stats.get("favoriteCount"),
        "view_count": stats.get("viewCount"),
        "like_count": stats.get("likeCount"),
        "comment_count": stats.get("commentCount"),
    }

# ==== FUNCTION: CALCULATE DAYS UNTIL RELEASE ====
def days_until_release(published_at, release_date):
    """Calculate days from trailer publish date to movie release."""
    try:
        pub_date = datetime.strptime(published_at[:10], "%Y-%m-%d")
        rel_date = datetime.strptime(release_date, "%Y-%m-%d")
        return (rel_date - pub_date).days
    except Exception:
        return None

# ==== FUNCTION: BUILD DATASET ====
def build_dataset(movie_list):
    all_rows = []

    for movie_name in movie_list:
        print(f"🎬 Processing: {movie_name}")
        movie_data = search_movie(movie_name)
        if not movie_data:
            print(f"❌ Movie not found: {movie_name}")
            continue

        trailers = get_trailers(movie_data["id"])
        if not trailers:
            print(f"⚠️ No trailers found for: {movie_name}")
            continue

        for trailer in trailers:
            yt_data = get_youtube_metrics(trailer["video_id"])
            if not yt_data:
                continue

            trailer_url = f"https://www.youtube.com/watch?v={trailer['video_id']}"
            days_diff = days_until_release(trailer["published_at"], movie_data["release_date"])

            all_rows.append({
                "movie_title": movie_data["title"],
                "release_date": movie_data["release_date"],
                "trailer_url": trailer_url,
                "video_name": trailer["video_name"],
                "published_at": trailer["published_at"],
                "official": trailer["official"],
                "type": trailer["type"],
                "days_until_release": days_diff,
                **yt_data
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Dataset saved to {OUTPUT_CSV}")
        print(df.head())
    else:
        print("\n⚠️ No data collected.")

# ==== MAIN ENTRY POINT ====
if __name__ == "__main__":
    df_movies = pd.read_csv(INPUT_CSV)
    movie_list = df_movies["title"].dropna().tolist()
    build_dataset(movie_list)
