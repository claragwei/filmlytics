from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm

# Configuration
MONGODB_URI = "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"
DATABASE_NAME = "cinemaniacs"
COLLECTION_NAME = "movies"
CSV_WITH_DIVERSITY = "/Users/nidhideshmukh/Documents/GitHub/filmlytics/complete_data_with_diversity.csv"

def update_mongodb_with_diversity():
    print("\n" + "="*70)
    print("📊 UPDATING MONGODB WITH DIVERSITY DATA")
    print("="*70 + "\n")
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    print(f"✓ Connected to database: {DATABASE_NAME}\n")
    
    # Load CSV
    print(f"📂 Loading CSV: {CSV_WITH_DIVERSITY}")
    df = pd.read_csv(CSV_WITH_DIVERSITY)
    print(f"✓ Loaded {len(df):,} movies with diversity data\n")
    
    # Update each movie
    print("📝 Updating MongoDB documents...")
    
    updated_count = 0
    not_found_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating movies"):
        try:
            tmdb_id = int(row['id']) if pd.notna(row['id']) else None
            
            if tmdb_id is None:
                not_found_count += 1
                continue
            
            # Prepare diversity data
            diversity_data = {
                "diversity": {
                    "female_cast_count": int(row['female_cast_count']) if pd.notna(row['female_cast_count']) else 0,
                    "male_cast_count": int(row['male_cast_count']) if pd.notna(row['male_cast_count']) else 0,
                    "female_cast_percentage": float(row['female_cast_percentage']) if pd.notna(row['female_cast_percentage']) else 0.0,
                    "gender_balance_score": float(row['gender_balance_score']) if pd.notna(row['gender_balance_score']) else 0.0,
                    "director_gender": int(row['director_gender']) if pd.notna(row['director_gender']) else 0,
                    "female_director": bool(row['female_director']) if pd.notna(row['female_director']) else False,
                    "cast_genders": row['cast_genders'] if pd.notna(row['cast_genders']) else "[]",
                    "director_genders": row['director_genders'] if pd.notna(row['director_genders']) else "[]"
                }
            }
            
            # Update document
            result = collection.update_one(
                {"tmdb_id": tmdb_id},
                {"$set": diversity_data}
            )
            
            if result.modified_count > 0:
                updated_count += 1
            elif result.matched_count == 0:
                not_found_count += 1
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n⚠️  Error updating movie {row.get('title', 'Unknown')}: {e}")
    
    print(f"\n✓ Update complete!\n")
    
    # Statistics
    print("="*70)
    print("📊 UPDATE STATISTICS")
    print("="*70)
    print(f"\nTotal movies in CSV: {len(df):,}")
    print(f"Successfully updated: {updated_count:,}")
    print(f"Not found in MongoDB: {not_found_count:,}")
    print(f"Errors: {error_count:,}")
    
    # Verification
    print("\n" + "="*70)
    print("🔍 VERIFICATION")
    print("="*70 + "\n")
    
    with_diversity = collection.count_documents({"diversity": {"$exists": True}})
    print(f"Documents with diversity data: {with_diversity:,}")
    
    # Sample movie
    sample = collection.find_one(
        {"diversity.female_cast_percentage": {"$gt": 0}},
        {"title": 1, "diversity": 1}
    )
    
    if sample:
        print(f"\n📽️  Sample Movie: {sample['title']}")
        print(f"   Female cast: {sample['diversity']['female_cast_percentage']:.1f}%")
        print(f"   Gender balance: {sample['diversity']['gender_balance_score']:.1f}/100")
        print(f"   Female director: {sample['diversity']['female_director']}")
    
    # Averages
    pipeline = [
        {"$match": {"diversity.female_cast_percentage": {"$gt": 0}}},
        {"$group": {
            "_id": None,
            "avg_female_percentage": {"$avg": "$diversity.female_cast_percentage"},
            "avg_balance_score": {"$avg": "$diversity.gender_balance_score"},
            "female_directors": {"$sum": {"$cond": ["$diversity.female_director", 1, 0]}},
            "total": {"$sum": 1}
        }}
    ]
    
    result = list(collection.aggregate(pipeline))
    
    if result:
        stats = result[0]
        print(f"\n📊 Database Averages:")
        print(f"   Average female cast: {stats['avg_female_percentage']:.1f}%")
        print(f"   Average gender balance: {stats['avg_balance_score']:.1f}/100")
        print(f"   Female directors: {stats['female_directors']:,} ({(stats['female_directors']/stats['total']*100):.1f}%)")
    
    print("\n" + "="*70)
    print("✅ MONGODB UPDATE COMPLETE!")
    print("="*70 + "\n")
    
    client.close()

if __name__ == "__main__":
    update_mongodb_with_diversity()