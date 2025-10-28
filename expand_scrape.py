from crawlbase import CrawlingAPI
from bs4 import BeautifulSoup
import json
import time
import urllib.parse

API_TOKEN = 'FeQRCmjUd-JGi_I6G0ERyw'
OUTPUT_FILE = 'rottentomatoes_targeted.json'

crawling_api = CrawlingAPI({'token': API_TOKEN})

def fetch_html(url):
    response = crawling_api.get(url)
    if response['headers'].get('pc_status') == '200':
        return response['body'].decode('utf-8')
    else:
        print(f"Failed to fetch: {url} ({response['headers'].get('pc_status')})")
        return None

def search_movie(title):
    query = urllib.parse.quote(title)
    search_url = f"https://www.rottentomatoes.com/search?search={query}"
    html = fetch_html(search_url)
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    result = soup.select_one('search-page-media-row a')
    if result and result['href']:
        return "https://www.rottentomatoes.com" + result['href']
    return None

def parse_movie_page(url):
    html = fetch_html(url)
    if not html:
        return None
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select_one('h1[data-qa="score-panel-movie-title"]')
    critics = soup.select_one('score-board').get('tomatometerscore') if soup.select_one('score-board') else ''
    audience = soup.select_one('score-board').get('audiencescore') if soup.select_one('score-board') else ''
    year = ''
    year_tag = soup.select_one('a[href*="year/"]')
    if year_tag:
        year = year_tag.text.strip()
    return {
        'title': title.text.strip() if title else '',
        'year': year,
        'critics_score': critics,
        'audience_score': audience,
        'link': url
    }

def crawl_movies(movie_list):
    results = []
    for title in movie_list:
        print(f"Searching for: {title}")
        movie_url = search_movie(title)
        if movie_url:
            print(f"Found: {movie_url}")
            movie_data = parse_movie_page(movie_url)
            if movie_data:
                results.append(movie_data)
        else:
            print(f"No result found for {title}")
        time.sleep(3)
    return results

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} movies to {filename}")

if __name__ == "__main__":
    movie_list = [
        "Oppenheimer",
        "Barbie",
        "Everything Everywhere All at Once",
        "The Batman",
        "Parasite",
        "Get Out"
    ]
    data = crawl_movies(movie_list)
    save_to_json(data, OUTPUT_FILE)
