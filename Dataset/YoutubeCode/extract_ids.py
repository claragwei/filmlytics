import pandas as pd

INPUT_CSV = "tmdb_cleaned.csv"
OUTPUT_CSV = "tmdb_ids.csv"
CHUNK_SIZE = 10000  # read 10k rows at a time

all_ids = []

for chunk in pd.read_csv(INPUT_CSV, usecols=['id'], chunksize=CHUNK_SIZE, engine='python', on_bad_lines='skip'):
    all_ids.append(chunk)

df_ids = pd.concat(all_ids, ignore_index=True)
df_ids.to_csv(OUTPUT_CSV, index=False)
print(f"✅ tmdb_ids.csv created with {len(df_ids)} rows")

