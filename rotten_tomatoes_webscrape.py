from bs4 import BeautifulSoup
import requests
import json
import time
from urllib.parse import urljoin

# === CONFIG ===
OMDB_KEY = "ba29c3f5" 
BASE_URL = "https://www.rottentomatoes.com"
YEARS = range(2015, 2026)
MOVIE_LIMIT_PER_YEAR = 20 
SLEEP_BETWEEN_REQUESTS = 2

def fetch_movie_links(list_url, limit=MOVIE_LIMIT_PER_YEAR):
    """Scrape movie titles and links from a Rotten Tomatoes listing page."""
    print(f"\nFetching movie list from {list_url}")
    resp = requests.get(list_url, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        print(f"Failed to load {list_url}: HTTP {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    movies = soup.select('a[data-qa^="discovery-media-list-item"]')

    results = []
    for a in movies[:limit]:
        href = a.get("href")
        title_tag = a.select_one('span[data-qa="discovery-media-list-item-title"]')
        title = title_tag.text.strip() if title_tag else None
        if href and title:
            results.append({
                "title": title,
                "url": urljoin(BASE_URL, href)
            })

    print(f"Found {len(results)} movies on {list_url}")
    return results

def get_movie_year(title):
    """Fetch release year via OMDb (since Rotten Tomatoes pages vary)."""
    url = f"https://www.omdbapi.com/?apikey={OMDB_KEY}&t={title}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("Response") == "True":
            year = data.get("Year")
            if year:
                return int(year.split("–")[0])
    except Exception as e:
        print(f"OMDb lookup failed for {title}: {e}")
    return None

def get_critic_reviews(movie_url, max_pages=2):
    """Scrape critic reviews and correctly detect fresh/rotten status."""
    print(f"Fetching reviews for {movie_url}")
    all_reviews = []
    page = 1

    while page <= max_pages:
        url = f"{movie_url}/reviews?type=top_critics&page={page}"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
        except requests.exceptions.Timeout:
            print(f"Timeout on {url}")
            break

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} on {url}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        review_blocks = soup.select("review-speech-balloon, div.review-row, div.review-table-row")

        if not review_blocks:
            print("No review blocks found.")
            break

        for block in review_blocks:
            quote_tag = block.select_one(".the_review") or block.select_one("p")
            quote = quote_tag.text.strip() if quote_tag else None
            if not quote:
                continue

            is_fresh = False
            is_rotten = False

            icon = block.select_one("rt-icon")
            if icon and icon.get("name") == "fresh":
                is_fresh = True
            elif icon and icon.get("name") == "rotten":
                is_rotten = True

            if not (is_fresh or is_rotten):
                classes = " ".join(icon.get("class", []) if icon else [])
                is_fresh = "fresh" in classes
                is_rotten = "rotten" in classes

            all_reviews.append({
                "critic_review": quote,
                "isFresh": is_fresh,
                "isRotten": is_rotten
            })

        page += 1
        time.sleep(1.5)

    print(f"Collected {len(all_reviews)} reviews for {movie_url}")
    return all_reviews

def crawl_all():
    all_data = []
    list_pages = [
        f"{BASE_URL}/browse/movies_in_theaters/sort:popular",
        f"{BASE_URL}/browse/movies_at_home/sort:popular"
    ]

    for list_url in list_pages:
        movies = fetch_movie_links(list_url)
        for i, m in enumerate(movies, 1):
            title, url = m["title"], m["url"]
            year = get_movie_year(title)
            if not year or year not in YEARS:
                print(f"Skipping {title} — year {year}")
                continue

            reviews = get_critic_reviews(url)
            for r in reviews:
                all_data.append({"movie": title, "year": year, **r})

            if i % 5 == 0:
                save_to_json(all_data, "rt_partial.json")
                print(f"Partial save ({len(all_data)} reviews so far)")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_data

def save_to_json(data, filename="rt_critic_reviews_2015_2025.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved {len(data)} total reviews to {filename}")

if __name__ == "__main__":
    data = crawl_all()
    save_to_json(data)
