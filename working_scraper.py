import requests
from bs4 import BeautifulSoup
import json
import time
import random
import pandas as pd
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/141.0.0.0 Safari/537.36"
}

start_row = 1  # Change this number to start from a different row
MOVIES = pd.read_csv('mapped.csv')['movieURL'].iloc[start_row:].tolist()


def fetch_rt_data(url, max_reviews=20):
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    script_tag = soup.select_one("media-scorecard-manager script")

    title = soup.title.text.strip() if soup.title else "N/A"
    tomatometer = "N/A"
    audience_score = "N/A"

    if script_tag:
        try:
            data_json = json.loads(script_tag.text.strip())
            tomatometer = (
                data_json.get("tomatometer", {}).get("score")
                or data_json.get("criticsScore", {}).get("scorePercent")
                or data_json.get("criticsScore", {}).get("value")
                or "N/A"
            )
            audience_score = (
                data_json.get("audienceScore", {}).get("score")
                or data_json.get("audienceScore", {}).get("scorePercent")
                or data_json.get("audienceScore", {}).get("value")
                or "N/A"
            )
        except json.JSONDecodeError:
            pass

    reviews = fetch_reviews(url, max_reviews)
    return {
        "title": title,
        "tomatometer": f"{tomatometer}",
        "audience_score": f"{audience_score}%",
        "reviews": reviews
    }


def fetch_reviews(movie_url, max_reviews):
    review_url = f"{movie_url}/reviews"
    print("Fetching critic reviews from", review_url)

    reviews = []
    page = 1
    while len(reviews) < max_reviews:
        res = requests.get(f"{review_url}?page={page}&type=top_critics", headers=HEADERS, timeout=10)
        if res.status_code != 200:
            break
        review_soup = BeautifulSoup(res.text, "html.parser")

        blocks = review_soup.select("review-speech-balloon, div.review-row, div.review-table-row")
        if not blocks:
            break

        for b in blocks:
            if len(reviews) >= max_reviews:
                break
            quote_tag = b.select_one(".the_review, p")
            quote = quote_tag.text.strip() if quote_tag else None
            if quote:
                reviews.append(quote)

        next_page = review_soup.select_one("button[aria-label='next page'], a[rel='next']")
        if not next_page:
            break
        page += 1
        time.sleep(1)

    return reviews[:max_reviews]


def main():
    save_file = "rottentomatoes_dataset.json"
    all_movies = []

    # Load progress if exists
    if os.path.exists(save_file):
        print("Previous progress found. Loading data...")
        with open(save_file, "r", encoding="utf-8") as f:
            all_movies = json.load(f)
        scraped_urls = {m.get("url") for m in all_movies if m.get("url")}
    else:
        scraped_urls = set()

    save_interval = 10
    counter = len(all_movies)

    for url in MOVIES:
        # Skip movies already scraped
        if url in scraped_urls:
            print(f"Skipping already scraped: {url}")
            continue

        print(f"\nFetching {url}")
        movie_data = fetch_rt_data(url, max_reviews=20)
        if movie_data:
            movie_data["url"] = url  # store URL to identify progress
            all_movies.append(movie_data)
            scraped_urls.add(url)
            counter += 1

        # Save every 10 movies
        if counter % save_interval == 0:
            print(f"Saving progress at {counter} movies...")
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(all_movies, f, indent=2, ensure_ascii=False)

        time.sleep(random.uniform(2, 4))

    # Final save
    print("\nFinal save...")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_movies, f, indent=2, ensure_ascii=False)

    print("\nDone. All data saved.")




if __name__ == "__main__":
    main()
