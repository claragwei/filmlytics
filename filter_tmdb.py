import pandas as pd

# Read CSV
df_original = pd.read_csv('tmdb_final_2.csv')
df = df_original.copy()

print(f"Before: {len(df)} movies")

# Remove unwanted columns
columns_to_drop = ['budget', 'popularity', 'tagline', 'keywords', 'producers']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Keep only key columns with good data
key_columns = ['title', 'release_date', 'genres', 'cast', 'directors', 'overview', 'production_companies', 'production_countries', 'poster_url', 'trailer_url']

df = df.dropna(subset=key_columns)

# Remove empty strings in key columns
for col in key_columns:
    df = df[df[col] != '']

print(f"After filtering: {len(df)} movies")
print(f"Removed: {len(df_original) - len(df)} movies")

# Save filtered data
df.to_csv('tmdb_filtered_2.csv', index=False, quoting=1)
print(f"Saved to tmdb_filtered_2.csv")