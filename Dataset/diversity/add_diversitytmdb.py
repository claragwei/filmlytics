"""
Fast Gender Data Collection - Uses Movie IDs from Existing CSV
Much faster than searching for each person by name!
Uses the same TMDB API structure as your original scraping scripts
"""

import pandas as pd
import requests
import time
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiMDAzMDhjZWMzZWZiMWEwYzUzYzMxYWNmOWZkNzA1NyIsIm5iZiI6MTc2MTA2ODg3NS40NzksInN1YiI6IjY4ZjdjNzRiMjVlMzNjMTdiYzg3ZDViYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Q5JNWorEJL9w9rwop3QVePSSzqx90pGN6VB04YAwGAY"

BASE_URL = "https://api.themoviedb.org/3"
INPUT_CSV = "/Users/nidhideshmukh/Documents/GitHub/filmlytics/Dataset/diversity/complete_data.csv"
OUTPUT_CSV = "complete_data_with_diversity.csv"

# Rate limiting (40 requests per 10 seconds)
MIN_REQUEST_INTERVAL = 1.0 / 35.0  # ~35 requests per 10 seconds to be safe

# ============================================================================
# TMDB API FUNCTIONS
# ============================================================================

class FastGenderCollector:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        self.last_request_time = 0
        self.request_count = 0
        
    def _respect_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_movie_credits(self, movie_id):
        """
        Get cast and crew with genders for a movie
        Same API structure as your original tmdb_part1.py!
        """
        url = f"{BASE_URL}/movie/{movie_id}"
        params = {
            'language': 'en-US',
            'append_to_response': 'credits'  # Only need credits, not images/videos
        }
        
        self._respect_rate_limit()
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit hit
                print(f"   ⚠️  Rate limit hit, waiting 10 seconds...")
                time.sleep(10)
                return self.get_movie_credits(movie_id)  # Retry
            else:
                return None
                
        except Exception as e:
            print(f"   ⚠️  Error fetching movie {movie_id}: {e}")
            return None
    
    def extract_gender_data(self, movie_data):
        """
        Extract gender data from movie credits
        Returns genders for cast and directors
        """
        if not movie_data:
            return [], []
        
        credits = movie_data.get('credits', {})
        
        # Get cast genders (top 10)
        cast = credits.get('cast', [])
        cast_genders = [c.get('gender', 0) for c in cast[:10]]
        
        # Get director genders
        crew = credits.get('crew', [])
        directors = [c for c in crew if c.get('job') == 'Director']
        director_genders = [d.get('gender', 0) for d in directors]
        
        return cast_genders, director_genders

def calculate_diversity_metrics(cast_genders):
    """Calculate diversity metrics from gender list"""
    if not cast_genders:
        return {
            'female_cast_count': 0,
            'male_cast_count': 0,
            'female_cast_percentage': 0.0,
            'gender_balance_score': 0.0
        }
    
    female_count = cast_genders.count(1)
    male_count = cast_genders.count(2)
    total = len(cast_genders)
    
    female_percentage = (female_count / total * 100) if total > 0 else 0
    gender_balance_score = max(0, 100 - abs(50 - female_percentage) * 2)
    
    return {
        'female_cast_count': female_count,
        'male_cast_count': male_count,
        'female_cast_percentage': round(female_percentage, 2),
        'gender_balance_score': round(gender_balance_score, 2)
    }

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def add_gender_data_fast():
    """
    Fast method: Use movie IDs from your CSV to fetch gender data
    Much faster than searching by person names!
    """
    
    print("\n" + "="*70)
    print("⚡ FAST GENDER DATA COLLECTION (Using Movie IDs)")
    print("="*70 + "\n")
    
    # Load your existing CSV
    print(f"📂 Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"✓ Loaded {len(df):,} movies\n")
    
    # Check for resume capability
    cache_file = "gender_progress.json"
    processed_ids = set()
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            processed_ids = set(cache_data.get('processed_ids', []))
            print(f"✓ Resuming: {len(processed_ids):,} movies already processed\n")
    except FileNotFoundError:
        print("ℹ️  Starting fresh (no previous progress found)\n")
    
    # Initialize collector
    collector = FastGenderCollector(TMDB_BEARER_TOKEN)
    
    # Initialize new columns
    if 'cast_genders' not in df.columns:
        df['cast_genders'] = None
        df['female_cast_count'] = 0
        df['male_cast_count'] = 0
        df['female_cast_percentage'] = 0.0
        df['gender_balance_score'] = 0.0
        df['director_genders'] = None
        df['director_gender'] = 0
        df['female_director'] = False
    
    # Process each movie
    print("🔍 Fetching gender data from TMDB...")
    print(f"⏱️  Estimated time: ~{(len(df) * 0.03):.1f} minutes (much faster!)\n")
    
    start_time = time.time()
    movies_processed = 0
    
    try:
        for idx, row in df.iterrows():
            movie_id = row['id']
            
            # Skip if already processed
            if movie_id in processed_ids:
                continue
            
            # Progress update
            if movies_processed % 100 == 0 and movies_processed > 0:
                elapsed = time.time() - start_time
                rate = movies_processed / elapsed
                remaining = len(df) - movies_processed
                eta_minutes = (remaining / rate) / 60 if rate > 0 else 0
                
                print(f"Progress: {movies_processed:,}/{len(df):,} ({movies_processed/len(df)*100:.1f}%) | "
                      f"ETA: {eta_minutes:.1f} min | "
                      f"API calls: {collector.request_count:,}")
            
            # Fetch gender data
            movie_data = collector.get_movie_credits(movie_id)
            cast_genders, director_genders = collector.extract_gender_data(movie_data)
            
            # Save to dataframe
            df.at[idx, 'cast_genders'] = str(cast_genders)
            df.at[idx, 'director_genders'] = str(director_genders)
            
            # Calculate metrics
            if cast_genders:
                metrics = calculate_diversity_metrics(cast_genders)
                df.at[idx, 'female_cast_count'] = metrics['female_cast_count']
                df.at[idx, 'male_cast_count'] = metrics['male_cast_count']
                df.at[idx, 'female_cast_percentage'] = metrics['female_cast_percentage']
                df.at[idx, 'gender_balance_score'] = metrics['gender_balance_score']
            
            # Director gender
            if director_genders:
                df.at[idx, 'director_gender'] = director_genders[0]
                df.at[idx, 'female_director'] = (director_genders[0] == 1)
            
            processed_ids.add(movie_id)
            movies_processed += 1
            
            # Save progress every 500 movies
            if movies_processed % 500 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                with open(cache_file, 'w') as f:
                    json.dump({'processed_ids': list(processed_ids)}, f)
                print(f"   💾 Progress saved")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving progress...")
        df.to_csv(OUTPUT_CSV, index=False)
        with open(cache_file, 'w') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)
        print(f"✓ Progress saved to {OUTPUT_CSV}")
        print("Run script again to resume!")
        return df
    
    # Final save
    print(f"\n✓ Processed {movies_processed:,} movies!")
    print(f"✓ Total API requests: {collector.request_count:,}")
    
    print(f"\n💾 Saving to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✓ Saved!\n")
    
    # Statistics
    print("="*70)
    print("📊 DIVERSITY STATISTICS")
    print("="*70 + "\n")
    
    movies_with_data = (df['female_cast_percentage'] > 0).sum()
    print(f"Movies with gender data: {movies_with_data:,} ({movies_with_data/len(df)*100:.1f}%)")
    
    female_directors = df['female_director'].sum()
    male_directors = (df['director_gender'] == 2).sum()
    print(f"\nFemale directors: {female_directors:,} ({female_directors/len(df)*100:.1f}%)")
    print(f"Male directors: {male_directors:,} ({male_directors/len(df)*100:.1f}%)")
    
    avg_female = df[df['female_cast_percentage'] > 0]['female_cast_percentage'].mean()
    print(f"\nAverage female cast: {avg_female:.1f}%")
    
    # Sample movies
    print("\n" + "="*70)
    print("🎬 SAMPLE MOVIES")
    print("="*70 + "\n")
    
    sample = df[df['female_cast_percentage'] > 0].head(5)
    for _, movie in sample.iterrows():
        print(f"📽️  {movie['title']}")
        print(f"   Female cast: {movie['female_cast_percentage']:.1f}%")
        print(f"   Gender balance: {movie['gender_balance_score']:.1f}/100")
        print(f"   Female director: {'Yes' if movie['female_director'] else 'No'}")
        print()
    
    print("="*70)
    print("✅ COMPLETE!")
    print("="*70 + "\n")
    
    return df

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n⚡ This script uses movie IDs (much faster than name search!)")
        print("Same API structure as your tmdb_part1.py and tmdb_part2.py\n")
        
        df = add_gender_data_fast()
        
        if df is not None:
            print("🎉 Success! Now run update_mongodb_with_diversity.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Progress saved. Run again to resume!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()