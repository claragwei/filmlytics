import pandas as pd

# Read the CSV
df_original = pd.read_csv('tmdb_vote5_quarterly.csv')
df = df_original.copy()

print("Data Filtering (Vote >= 5.0)")
print("=" * 50)
print(f"Before: {len(df)} movies")

key_columns = ['runtime', 'vote_average', 'genres', 'cast', 'directors', 'overview', 'production_companies', 'poster_url', 'trailer_url']

df = df.dropna(subset=key_columns)

# Remove empty strings in key columns
for col in key_columns:
    df = df[df[col] != '']

print(f"After filtering: {len(df)} movies")
print(f"Removed: {len(df_original) - len(df)} movies")

# Save filtered data
df.to_csv('tmdb_vote5_filtered.csv', index=False, quoting=1)
print(f"Saved to tmdb_vote5_filtered.csv")

