"""
tmdb_trailer_coverage_check.py
------------------------------
Checks which movies from a list exist on TMDB and have YouTube trailers.

Input:  movies_list.csv  (must contain a column 'title')
Output: tmdb_trailer_coverage.csv
"""

import requests
import pandas as pd
from time import sleep

# ==== CONFIG ====
TMDB_API_KEY = "API Key"
INPUT_CSV = "tmdb_filtered_2.csv"                 # Input movie list file
OUTPUT_CSV = "movie_trailer_check.csv"

# ==== TMDB FUNCTIONS ====
def search_movie(movie_name):
    """Search for movie on TMDB and return ID if found."""
    url = "https://api.themoviedb.org/3/search/movie"
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

def has_trailers(movie_id):
    """Check if TMDB has any YouTube trailers for a movie."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    response = requests.get(url, params=params)
    videos = response.json().get("results", [])

    trailers = [v for v in videos if v.get("site", "").lower() == "youtube" and v.get("type", "").lower() == "trailer"]
    return len(trailers) > 0

# ==== MAIN CHECK ====
def check_tmdb_coverage(movie_list):
    results = []

    for movie_name in movie_list:
        print(f"🔎 Checking: {movie_name}")
        movie_data = search_movie(movie_name)
        sleep(0.25)  # gentle rate limit

        if not movie_data:
            results.append({"title": movie_name, "found_on_tmdb": False, "has_trailer": False})
            continue

        found_trailer = has_trailers(movie_data["id"])
        sleep(0.25)

        results.append({
            "title": movie_name,
            "found_on_tmdb": True,
            "tmdb_movie_id": movie_data["id"],
            "release_date": movie_data["release_date"],
            "has_trailer": found_trailer
        })

    # Create dataframe and summary
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    found = df["found_on_tmdb"].sum()
    with_trailer = df["has_trailer"].sum()
    total = len(df)

    print("\n✅ TMDB Coverage Check Complete")
    print(f"Total movies checked: {total}")
    print(f"Found on TMDB: {found} ({found/total:.1%})")
    print(f"With YouTube trailer(s): {with_trailer} ({with_trailer/total:.1%})")
    print(f"Results saved to: {OUTPUT_CSV}")

# ==== MAIN ENTRY POINT ====
if __name__ == "__main__":
    df_movies = pd.read_csv(INPUT_CSV)
    movie_list = df_movies["title"].dropna().tolist()
    check_tmdb_coverage(movie_list)
