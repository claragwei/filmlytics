import pandas as pd

# =======================
# 1. Load datasets
# =======================
tmdb_file = 'tmdb data/tmdb_cleaned.csv'
rt_file = 'url_mapping/movie_info.csv'

tmdb_df = pd.read_csv(tmdb_file, engine="python", on_bad_lines="warn")
rt_df = pd.read_csv(rt_file)

# =======================
# 2. Clean titles for matching
# =======================
tmdb_df['title_clean'] = tmdb_df['title'].str.lower().str.strip()
rt_df['title_clean'] = rt_df['title'].str.lower().str.strip()

# =======================
# 3. Merge datasets on cleaned title
# =======================
merged_df = pd.merge(
    tmdb_df,
    rt_df[['title_clean', 'url']],
    on='title_clean',
    how='left'
)

# =======================
# 4. Fill missing URLs with N/A
# =======================
merged_df['url'] = merged_df['url'].fillna("N/A")

# =======================
# 5. Add has_url column (1 if URL exists, 0 if N/A)
# =======================
merged_df['has_url'] = (merged_df['url'] != "N/A").astype(int)

# =======================
# 6. Drop helper column
# =======================
merged_df = merged_df.drop(columns=['title_clean'])

# =======================
# 7. Save updated TMDB dataset
# =======================
output_file = 'tmdb_with_urls.csv'
merged_df.to_csv(output_file, index=False, quoting=1)

print("="*50)
print(f"Total movies in TMDB dataset: {len(tmdb_df)}")
print(f"Movies with RT URLs: {merged_df['url'].ne('N/A').sum()}")
print(f"Movies without RT URLs: {merged_df['url'].eq('N/A').sum()}")
print(f"Saved merged dataset to: {output_file}")
print("="*50)
