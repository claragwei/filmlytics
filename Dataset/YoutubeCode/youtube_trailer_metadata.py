import pandas as pd
import requests
import time
from urllib.parse import urlparse, parse_qs

# ================================
# 🔑 INSERT YOUR YOUTUBE API KEY HERE
# ================================
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"

# ================================
# 📂 INPUT / OUTPUT FILE PATHS
# ================================
input_csv = "/Users/dylansidhu/Desktop/STA 160/movies_list.csv"
output_csv = "/Users/dylansidhu/Desktop/STA 160/movie_trailer_dataset.csv"

# ================================
# 🔍 HELPER FUNCTION — Extract Video ID from YouTube URL
# ================================
def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path[1:]
        else:
            return None
    except Exception:
        return None

# ================================
# 🎬 FUNCTION — Get YouTube video details
# ================================
def get_youtube_video_details(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()
    if not data.get("items"):
        return None

    item = data["items"][0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "video_id": video_id,
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "tags": ", ".join(snippet.get("tags", [])) if snippet.get("tags") else "",
        "category_id": snippet.get("categoryId"),
        "published_at": snippet.get("publishedAt"),
        "view_count": stats.get("viewCount"),
        "like_count": stats.get("likeCount"),
        "favorite_count": stats.get("favoriteCount"),
        "comment_count": stats.get("commentCount")
    }

# ================================
# 🚀 MAIN SCRIPT
# ================================
def main():
    # Load your movies list
    movies_df = pd.read_csv(input_csv)
    print(f"Loaded {len(movies_df)} movies from {input_csv}")

    results = []

    for i, row in movies_df.iterrows():
        title = row.get("title")
        url = row.get("trailer_url")
        video_id = extract_video_id(url)

        if not video_id:
            print(f"⚠️ Could not extract video ID for: {title}")
            continue

        details = get_youtube_video_details(video_id)
        if details:
            details["movie_title"] = title
            details["trailer_url"] = url
            results.append(details)
            print(f"✅ Processed {i+1}/{len(movies_df)} — {title}")
        else:
            print(f"❌ Failed to get data for: {title}")

        # Small delay to stay within API limits
        time.sleep(0.1)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n🎉 Done! Saved {len(results_df)} trailers to {output_csv}")

# ================================
# 🏁 RUN
# ================================
if __name__ == "__main__":
    main()
