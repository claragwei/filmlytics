import requests
import pandas as pd
import time

TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiMDAzMDhjZWMzZWZiMWEwYzUzYzMxYWNmOWZkNzA1NyIsIm5iZiI6MTc2MTA2ODg3NS40NzksInN1YiI6IjY4ZjdjNzRiMjVlMzNjMTdiYzg3ZDViYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Q5JNWorEJL9w9rwop3QVePSSzqx90pGN6VB04YAwGAY"
BASE_URL = "https://api.themoviedb.org/3"

class TMDBBatchCollector:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.base_url = BASE_URL
        self.movies = []
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0 / 20.0
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
       
    def _respect_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1

    def get_discover_movies(self, page=1, date_from='2010-01-01', date_to='2015-12-31'):
        """Get movies for a specific date range"""
        url = f"{self.base_url}/discover/movie"

        params = {
            'region': 'US',
            'primary_release_date.gte': date_from,
            'primary_release_date.lte': date_to,
            'include_adult': False,
            'sort_by': 'release_date.desc',
            'page': page,
            'language': 'en-US',
            'vote_count.gte': 1,  # Exclude movies with 0 votes
        }

        self._respect_rate_limit()

        # Retry logic for network errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
                return response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"  Connection error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return {}

    def get_movie_details(self, movie_id):
        """Get detailed info for a single movie"""
        url = f"{self.base_url}/movie/{movie_id}"

        params = {
            'language': 'en-US',
            'append_to_response': 'credits,keywords,images,videos'
        }

        self._respect_rate_limit()

        # Retry logic for network errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
                return response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"  Connection error on movie {movie_id}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to get details for movie {movie_id} after {max_retries} attempts")
                    return {}

    def extract_features(self, movie_data):
        """Extract only the required features"""
        try:
            features = {}

            features['id'] = movie_data.get('id')
            features['title'] = movie_data.get('title')
            features['release_date'] = movie_data.get('release_date')
            features['budget'] = movie_data.get('budget', 0)
            features['runtime'] = movie_data.get('runtime', 0)
            features['vote_count'] = movie_data.get('vote_count', 0)
            features['vote_average'] = movie_data.get('vote_average', 0)
            features['popularity'] = movie_data.get('popularity', 0)
            features['tagline'] = movie_data.get('tagline', '')
            features['overview'] = movie_data.get('overview', '')

            genres = movie_data.get('genres', [])
            features['genres'] = ', '.join([g['name'] for g in genres]) if genres else ''

            prod_companies = movie_data.get('production_companies', [])
            features['production_companies'] = ', '.join([c['name'] for c in prod_companies]) if prod_companies else ''

            prod_countries = movie_data.get('production_countries', [])
            features['production_countries'] = ', '.join([c['name'] for c in prod_countries]) if prod_countries else ''

            keywords = movie_data.get('keywords', {})
            keyword_list = keywords.get('keywords', []) if isinstance(keywords, dict) else []
            features['keywords'] = ', '.join([k['name'] for k in keyword_list]) if keyword_list else ''

            credits = movie_data.get('credits', {})
            cast = credits.get('cast', [])
            features['cast'] = ', '.join([c['name'] for c in cast[:10]]) if cast else ''

            crew = credits.get('crew', [])
            directors = [c['name'] for c in crew if c['job'] == 'Director']
            producers = [c['name'] for c in crew if c['job'] == 'Producer']
            features['directors'] = ', '.join(directors) if directors else ''
            features['producers'] = ', '.join(producers) if producers else ''

            images = movie_data.get('images', {})
            posters = images.get('posters', [])
            if posters:
                poster_path = posters[0].get('file_path')
                features['poster_url'] = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            else:
                features['poster_url'] = None

            videos = movie_data.get('videos', {})
            video_results = videos.get('results', [])
            trailers = [v for v in video_results if v['type'] == 'Trailer']
            if trailers:
                trailer_key = trailers[0].get('key')
                features['trailer_url'] = f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else None
            else:
                features['trailer_url'] = None
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def collect_movies_for_period(self, date_from, date_to, period_name, max_pages=500):
        """Collect all movies for a specific date range"""
        print(f"Collecting {period_name}...")
        period_movies = []

        for page in range(1, max_pages + 1):
            discover_data = self.get_discover_movies(page=page, date_from=date_from, date_to=date_to)

            if not discover_data or 'results' not in discover_data:
                break

            movies = discover_data['results']
            if not movies:
                break

            for movie in movies:
                movie_id = movie['id']
                details = self.get_movie_details(movie_id)
                features = self.extract_features(details)

                if features:
                    period_movies.append(features)

            # Print progress every 10 pages
            if page % 10 == 0:
                print(f"  Page {page}: {len(period_movies)} total movies collected")

        print(f"  Total for {period_name}: {len(period_movies)} movies")
        return period_movies

    def collect_all_periods(self):
        """Collect movies by quarter for 2010-2017 (no quarters exceed 500 pages)"""
        all_movies = []

        # Years 2010-2017 (Part 1)
        all_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

        quarters = [
            ('Q1', '01-01', '03-31'),
            ('Q2', '04-01', '06-30'),
            ('Q3', '07-01', '09-30'),
            ('Q4', '10-01', '12-31')
        ]

        # Collect quarters for all years
        for year in all_years:
            for q_name, q_start, q_end in quarters:
                period_movies = self.collect_movies_for_period(
                    f'{year}-{q_start}', f'{year}-{q_end}', f'{year} {q_name}'
                )
                all_movies.extend(period_movies)

        self.movies = all_movies
        return all_movies

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        df = pd.DataFrame(self.movies)
        return df

def main():
    collector = TMDBBatchCollector(TMDB_BEARER_TOKEN)

    print("Collecting movies from 2010-2017")
    print("=" * 70)
    print()

    collector.collect_all_periods()

    df = collector.to_dataframe()

    print()
    print("=" * 70)
    print("Collection complete")
    print(f"Total movies collected: {len(df):,}")
    print(f"Total API requests made: {collector.request_count:,}")

    df.to_csv('tmdb_part1.csv', index=False, quoting=1)
    print(f"Saved to: tmdb_part1.csv")

    return df

if __name__ == "__main__":
    main()

