"""
Add Gender Diversity Data to Movie Dataset
Uses TMDB API to fetch actor and director genders
Updated to support both API Key and Bearer Token authentication
"""

import pandas as pd
import requests
import time
from typing import List, Dict
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# CHOOSE YOUR AUTHENTICATION METHOD (comment out the one you DON'T use):

# Option 1: API Key (v3 API)
# TMDB_API_KEY = "YOUR_API_KEY_HERE"
# USE_BEARER_TOKEN = False

# Option 2: Bearer Token (Recommended)
TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiMDAzMDhjZWMzZWZiMWEwYzUzYzMxYWNmOWZkNzA1NyIsIm5iZiI6MTc2MTA2ODg3NS40NzksInN1YiI6IjY4ZjdjNzRiMjVlMzNjMTdiYzg3ZDViYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Q5JNWorEJL9w9rwop3QVePSSzqx90pGN6VB04YAwGAY"  # Replace with your actual token
USE_BEARER_TOKEN = True

# File paths
INPUT_CSV = "/Users/nidhideshmukh/Documents/GitHub/filmlytics/Dataset/diversity/complete_data.csv"
OUTPUT_CSV = "complete_data_with_diversity.csv"

# Rate limiting (TMDB allows 40 requests per 10 seconds)
REQUESTS_PER_BATCH = 35  # Stay under 40 to be safe
SLEEP_TIME = 10  # seconds between batches

# ============================================================================
# TMDB API FUNCTIONS
# ============================================================================

def get_person_gender(person_name: str) -> int:
    """
    Get gender for a person from TMDB API
    Supports both API Key and Bearer Token authentication
    
    Returns:
        0 = Not specified / unknown
        1 = Female
        2 = Male
        3 = Non-binary (rarely used by TMDB)
    """
    try:
        url = "https://api.themoviedb.org/3/search/person"
        
        if USE_BEARER_TOKEN:
            # Bearer Token authentication
            headers = {
                "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
                "Content-Type": "application/json;charset=utf-8"
            }
            params = {
                "query": person_name,
                "include_adult": "false",
                "language": "en-US",
                "page": 1
            }
            response = requests.get(url, params=params, headers=headers, timeout=5)
        else:
            # API Key authentication
            params = {
                "api_key": TMDB_API_KEY,
                "query": person_name,
                "include_adult": "false",
                "language": "en-US",
                "page": 1
            }
            response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['results']:
                # Get the first (most popular) result
                gender = data['results'][0].get('gender', 0)
                return gender
        elif response.status_code == 429:
            # Rate limit hit - wait and retry once
            print(f"   ⚠️  Rate limit hit, waiting 10 seconds...")
            time.sleep(10)
            return get_person_gender(person_name)  # Retry once
        
        return 0  # Unknown if API call fails
        
    except Exception as e:
        print(f"   ⚠️  Error fetching gender for {person_name}: {e}")
        return 0

def parse_name_list(name_string):
    """Parse comma-separated names from string"""
    if pd.isna(name_string) or name_string == '':
        return []
    
    # Handle both string and list formats
    if isinstance(name_string, str):
        # Remove brackets and quotes if present
        name_string = name_string.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
        return [name.strip() for name in name_string.split(',') if name.strip()]
    
    return name_string if isinstance(name_string, list) else []

def get_cast_genders(cast_list: List[str], cache: Dict) -> List[int]:
    """
    Get genders for list of cast members
    Uses cache to avoid duplicate API calls
    Includes rate limiting
    """
    genders = []
    
    for actor in cast_list[:10]:  # Limit to top 10 actors to save API calls
        if actor in cache:
            # Use cached result
            genders.append(cache[actor])
        else:
            # Fetch from API
            gender = get_person_gender(actor)
            cache[actor] = gender
            genders.append(gender)
            time.sleep(0.25)  # Rate limiting: 0.25 seconds between requests
    
    return genders

def get_director_gender(directors_list: List[str], cache: Dict) -> int:
    """Get gender of first director (or primary director)"""
    if not directors_list:
        return 0
    
    director = directors_list[0]  # Get first director
    
    if director in cache:
        return cache[director]
    
    gender = get_person_gender(director)
    cache[director] = gender
    time.sleep(0.25)  # Rate limiting
    
    return gender

def calculate_diversity_metrics(cast_genders: List[int]) -> Dict:
    """
    Calculate diversity metrics from gender list
    
    Returns:
        Dictionary with diversity metrics
    """
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
    
    # Gender balance score: 100 = perfect balance (50/50), 0 = all one gender
    # Formula: 100 - |50 - female_percentage| * 2
    gender_balance_score = max(0, 100 - abs(50 - female_percentage) * 2)
    
    return {
        'female_cast_count': female_count,
        'male_cast_count': male_count,
        'female_cast_percentage': round(female_percentage, 2),
        'gender_balance_score': round(gender_balance_score, 2)
    }

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def add_diversity_to_dataset():
    """Main function to add diversity data to CSV"""
    
    print("\n" + "="*70)
    print("🎬 ADDING GENDER DIVERSITY DATA TO DATASET")
    print("="*70 + "\n")
    
    # Check authentication
    if USE_BEARER_TOKEN:
        if TMDB_BEARER_TOKEN == "YOUR_BEARER_TOKEN_HERE":
            print("❌ ERROR: Please set your TMDB Bearer Token in the script!")
            print("Get your token at: https://www.themoviedb.org/settings/api")
            print("Look for 'API Read Access Token'")
            return
        print("✓ Using Bearer Token authentication\n")
    else:
        if TMDB_API_KEY == "YOUR_API_KEY_HERE":
            print("❌ ERROR: Please set your TMDB API Key in the script!")
            print("Get your key at: https://www.themoviedb.org/settings/api")
            return
        print("✓ Using API Key authentication\n")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERROR: {INPUT_CSV} not found!")
        print("Make sure the file is in the same folder as this script.")
        return
    
    # Load dataset
    print(f"📂 Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"✓ Loaded {len(df):,} movies\n")
    
    # Initialize cache for API results (to avoid duplicate calls)
    gender_cache = {}
    
    # Try to load existing cache if available
    cache_file = "gender_cache.json"
    try:
        with open(cache_file, 'r') as f:
            gender_cache = json.load(f)
            print(f"✓ Loaded cache with {len(gender_cache):,} existing entries")
            print("   (This will speed up processing by reusing previous results)\n")
    except FileNotFoundError:
        print("ℹ️  No existing cache found, starting fresh\n")
    
    # Parse cast and directors
    print("📋 Parsing cast and director lists...")
    df['cast_list'] = df['cast'].apply(parse_name_list)
    df['directors_list'] = df['directors'].apply(parse_name_list)
    print("✓ Parsing complete\n")
    
    # Initialize new columns
    df['cast_genders'] = None
    df['female_cast_count'] = 0
    df['male_cast_count'] = 0
    df['female_cast_percentage'] = 0.0
    df['gender_balance_score'] = 0.0
    df['director_gender'] = 0
    df['female_director'] = False
    
    # Process each movie
    print("🔍 Fetching gender data from TMDB API...")
    print(f"⚠️  This will take a while due to rate limiting (40 requests per 10 seconds)")
    print(f"⏱️  Estimated time: ~{(len(df) * 11 * 0.25 / 3600):.1f} hours for full dataset")
    print(f"💡 TIP: You can stop anytime (Ctrl+C) and resume later!\n")
    
    total_movies = len(df)
    request_count = 0
    start_time = time.time()
    
    try:
        for idx, row in df.iterrows():
            # Progress update every 10 movies
            if idx % 10 == 0:
                progress = (idx / total_movies) * 100
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta_seconds = (total_movies - idx) / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"Progress: {idx:,}/{total_movies:,} ({progress:.1f}%) | "
                      f"Cache: {len(gender_cache):,} | "
                      f"ETA: {eta_hours:.1f}h")
            
            # Get cast genders
            cast_list = row['cast_list']
            if cast_list:
                cast_genders = get_cast_genders(cast_list, gender_cache)
                df.at[idx, 'cast_genders'] = str(cast_genders)
                
                # Calculate metrics
                metrics = calculate_diversity_metrics(cast_genders)
                df.at[idx, 'female_cast_count'] = metrics['female_cast_count']
                df.at[idx, 'male_cast_count'] = metrics['male_cast_count']
                df.at[idx, 'female_cast_percentage'] = metrics['female_cast_percentage']
                df.at[idx, 'gender_balance_score'] = metrics['gender_balance_score']
                
                request_count += len(cast_list[:10])
            
            # Get director gender
            directors_list = row['directors_list']
            if directors_list:
                director_gender = get_director_gender(directors_list, gender_cache)
                df.at[idx, 'director_gender'] = director_gender
                df.at[idx, 'female_director'] = (director_gender == 1)
                request_count += 1
            
            # Rate limiting: pause after batch
            if request_count >= REQUESTS_PER_BATCH:
                print(f"   ⏸️  Rate limit: Pausing {SLEEP_TIME}s (processed {REQUESTS_PER_BATCH} requests)")
                time.sleep(SLEEP_TIME)
                request_count = 0
                
                # Save cache every batch
                with open(cache_file, 'w') as f:
                    json.dump(gender_cache, f)
            
            # Save progress every 100 movies
            if idx > 0 and idx % 100 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"   💾 Progress saved to {OUTPUT_CSV}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user!")
        print("💾 Saving progress...")
        
        # Save current progress
        df.to_csv(OUTPUT_CSV, index=False)
        with open(cache_file, 'w') as f:
            json.dump(gender_cache, f)
        
        print(f"✓ Progress saved to {OUTPUT_CSV}")
        print(f"✓ Cache saved with {len(gender_cache):,} entries")
        print("\nYou can resume later by running this script again!")
        return df
    
    print(f"\n✓ Processed all {total_movies:,} movies!\n")
    
    # Save final cache
    print("💾 Saving gender cache...")
    with open(cache_file, 'w') as f:
        json.dump(gender_cache, f)
    print(f"✓ Saved cache with {len(gender_cache):,} entries\n")
    
    # Save updated dataset
    print(f"💾 Saving updated dataset to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✓ Dataset saved!\n")
    
    # Print statistics
    print("="*70)
    print("📊 DIVERSITY STATISTICS")
    print("="*70 + "\n")
    
    movies_with_cast_data = (df['female_cast_percentage'] > 0).sum()
    movies_with_director_data = (df['director_gender'] > 0).sum()
    
    print(f"Movies with cast gender data: {movies_with_cast_data:,} ({movies_with_cast_data/len(df)*100:.1f}%)")
    print(f"Movies with director gender data: {movies_with_director_data:,} ({movies_with_director_data/len(df)*100:.1f}%)")
    
    female_directors = df['female_director'].sum()
    male_directors = (df['director_gender'] == 2).sum()
    
    print(f"\nFemale directors: {female_directors:,} ({female_directors/len(df)*100:.1f}%)")
    print(f"Male directors: {male_directors:,} ({male_directors/len(df)*100:.1f}%)")
    
    avg_female_cast = df[df['female_cast_percentage'] > 0]['female_cast_percentage'].mean()
    avg_balance = df[df['gender_balance_score'] > 0]['gender_balance_score'].mean()
    
    print(f"\nAverage female cast percentage: {avg_female_cast:.1f}%")
    print(f"Average gender balance score: {avg_balance:.1f}/100")
    
    # Show examples
    print("\n" + "="*70)
    print("🎬 SAMPLE MOVIES WITH DIVERSITY DATA")
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
    print("="*70)
    print("\nNext steps:")
    print("1. ✅ Your diversity data is saved in:", OUTPUT_CSV)
    print("2. 📊 Run update_mongodb_with_diversity.py to add this data to MongoDB")
    print("3. 🤖 Update your XGBoost and GNN models to include diversity features")
    print("4. 🌐 Add diversity visualizations to your Streamlit website")
    print("\n")
    
    return df

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("🎭 CINEMANIACS DIVERSITY DATA COLLECTION")
        print("="*70)
        
        df_updated = add_diversity_to_dataset()
        
        if df_updated is not None:
            print("🎉 Success! Your dataset now includes gender diversity data!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Your progress has been saved.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If you see rate limit errors, the script will retry automatically.")
        print("If you see other errors, check your authentication token/key.")