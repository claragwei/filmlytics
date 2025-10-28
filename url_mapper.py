import pandas as pd

# =======================
# 1. Load datasets
# =======================
tmdb_file = 'tmdb_vote5_filtered.csv'  # your full TMDB dataset
rt_file = 'movies.csv'                # RT URLs dataset with 'title' and 'rt_url'

tmdb_df = pd.read_csv(tmdb_file)
rt_df = pd.read_csv(rt_file)

# =======================
# 2. Clean titles for matching
# =======================
tmdb_df['title_clean'] = tmdb_df['title'].str.lower().str.strip()
rt_df['title_clean'] = rt_df['movieTitle'].str.lower().str.strip()

# =======================
# 3. Merge datasets
# =======================
merged_df = pd.merge(tmdb_df, rt_df[['title_clean', 'movieURL']], on='title_clean', how='left')

# Flag which movies have RT URLs
merged_df['has_rt_url'] = merged_df['movieURL'].notna()

# =======================
# 4. Separate matched and unmatched movies
# =======================
matched_df = merged_df[merged_df['has_rt_url']].copy()
unmatched_df = merged_df[~merged_df['has_rt_url']].copy()

# =======================
# 5. Remove duplicates in matched movies
# =======================
matched_df = matched_df.drop_duplicates(subset=['title_clean'])

# =======================
# 6. Drop helper column
# =======================
matched_df = matched_df.drop(columns=['title_clean', 'has_rt_url'])
unmatched_df = unmatched_df.drop(columns=['title_clean', 'has_rt_url'])

# =======================
# 7. Save to CSV
# =======================
matched_file = 'tmdb_with_rt_urls.csv'
unmatched_file = 'tmdb_without_rt_urls.csv'

matched_df.to_csv(matched_file, index=False, quoting=1)
unmatched_df.to_csv(unmatched_file, index=False, quoting=1)

# =======================
# 8. Reporting
# =======================
print("="*50)
print(f"Total movies in TMDB dataset: {len(tmdb_df)}")
print(f"Matched movies with RT URLs (no duplicates): {len(matched_df)}")
print(f"Unmatched movies without RT URLs: {len(unmatched_df)}")
print(f"Saved matched movies to: {matched_file}")
print(f"Saved unmatched movies to: {unmatched_file}")
print("="*50)

# Load the unmatched movies CSV if not already loaded
unmatched_df = pd.read_csv('tmdb_without_rt_urls.csv')

# Ensure release_date is in datetime format
unmatched_df['release_date'] = pd.to_datetime(unmatched_df['release_date'], errors='coerce')

# Filter for movies released in 2024 or later
unmatched_2024_plus = unmatched_df[unmatched_df['release_date'].dt.year >= 2024].copy()

# Save the filtered unmatched movies
unmatched_2024_plus.to_csv('tmdb_unmatched_2024_plus.csv', index=False, quoting=1)

print(f"Total unmatched movies from 2024 onward: {len(unmatched_2024_plus)}")

