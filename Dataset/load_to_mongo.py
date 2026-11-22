"""Load complete_data_with_diversity.csv into MongoDB Atlas (cinemaniacs.movies).

Usage:  python Dataset/load_to_mongo.py [--dry-run]
URI comes from $MONGO_URI or Website/.streamlit/secrets.toml (never hardcoded).
Reuses the document layout from mongodb_setup.py and adds a `diversity` block.
"""
import os, sys, tomllib, pathlib
import pandas as pd, certifi, pymongo
from pymongo import MongoClient
from pymongo.server_api import ServerApi

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mongodb_setup import MovieDatabaseManager  # noqa: E402

CSV = ROOT / "Dataset/diversity/complete_data_with_diversity.csv"
DRY = "--dry-run" in sys.argv

def get_uri():
    if os.environ.get("MONGO_URI"):
        return os.environ["MONGO_URI"]
    return tomllib.load(open(ROOT / "Website/.streamlit/secrets.toml", "rb"))["MONGO_URI"]

def diversity(row):
    f = lambda k, d=0.0: float(row[k]) if pd.notna(row.get(k)) else d
    i = lambda k: int(row[k]) if pd.notna(row.get(k)) else 0
    return {
        "female_cast_count": i("female_cast_count"), "male_cast_count": i("male_cast_count"),
        "female_cast_percentage": f("female_cast_percentage"), "gender_balance_score": f("gender_balance_score"),
        "director_gender": i("director_gender"),
        "female_director": bool(row["female_director"]) if pd.notna(row.get("female_director")) else False,
        "cast_genders": row["cast_genders"] if pd.notna(row.get("cast_genders")) else "[]",
        "director_genders": row["director_genders"] if pd.notna(row.get("director_genders")) else "[]",
    }

df = pd.read_csv(CSV, low_memory=False)
print(f"CSV rows: {len(df):,} | unique ids: {df['id'].nunique():,} | null ids: {df['id'].isna().sum()}")
# One row per TMDB id: the CSV has ~1.2k ids that were matched to several Rotten Tomatoes pages
# (e.g. remakes sharing a title). Keep the row with a parseable audience score, else the first.
has_score = pd.to_numeric(df["audience_score"].astype(str).str.rstrip("%"), errors="coerce").notna()
df = df.assign(_has=has_score).sort_values("_has", ascending=False, kind="stable").drop_duplicates("id").drop(columns="_has")
print(f"After de-duplicating on id: {len(df):,} rows")
docs, bad = [], 0
for idx, row in df.iterrows():
    try:
        d = MovieDatabaseManager.prepare_movie_document(None, row)
        d["diversity"] = diversity(row)
        docs.append(d)
    except Exception as e:
        bad += 1
        if bad <= 5: print("row", idx, "error:", e)
print(f"Built {len(docs):,} documents, {bad} failed")
if DRY:
    sys.exit(0)

client = MongoClient(get_uri(), server_api=ServerApi("1"), tlsCAFile=certifi.where())
coll = client["cinemaniacs"].movies
print("Existing docs before load:", coll.count_documents({}))
coll.delete_many({})
for i in range(0, len(docs), 1000):
    coll.insert_many(docs[i:i + 1000])
print("Inserted. Count now:", coll.count_documents({}))
for field in ["tmdb_id", "production.genres", "tmdb_metrics.is_successful", "tmdb_metrics.vote_count",
              "rotten_tomatoes.has_rt_url", "people.directors", "people.cast"]:
    coll.create_index([(field, pymongo.ASCENDING)])
coll.create_index([("title", pymongo.ASCENDING)])
print("Indexes:", sorted(coll.index_information()))
s = client["cinemaniacs"].command("collStats", "movies")
print(f"Storage: {s['storageSize']/1e6:.1f} MB (limit 512 MB)")
