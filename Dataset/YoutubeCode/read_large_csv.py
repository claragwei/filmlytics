import pandas as pd
import sys, csv

csv.field_size_limit(sys.maxsize)

csv_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned.csv"

try:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", quoting=3)
    print(f"✅ Loaded successfully with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"❌ Error loading file: {e}")
