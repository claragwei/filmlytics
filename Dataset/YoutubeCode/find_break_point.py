import pandas as pd
import sys, csv

csv.field_size_limit(sys.maxsize)

csv_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned.csv"

chunk_size = 10000
total_rows = 0

try:
    for i, chunk in enumerate(pd.read_csv(csv_path, engine="python", chunksize=chunk_size, on_bad_lines="skip", quoting=3)):
        total_rows += len(chunk)
        print(f"✅ Chunk {i+1} read — Total rows so far: {total_rows}")
except Exception as e:
    print(f"❌ Error occurred after {total_rows} rows — {e}")

