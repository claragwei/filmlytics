import pandas as pd
import numpy as np

print("="*80)
print("TMDB DATA CLEANING")
print("="*80)

# 1. LOAD DATA
print("\nLoading data from tmdb_complete.csv...")
df = pd.read_csv('tmdb_complete.csv')
print(f"Initial dataset: {len(df):,} movies, {len(df.columns)} columns")

# 2. REMOVE ROWS WITH MISSING CRITICAL DATA
print("\nRemoving rows with missing critical data...")
initial_count = len(df)

df = df[df['title'].notna()]
print(f"  Removed {initial_count - len(df):,} rows with missing title")

initial_count = len(df)
df = df[df['vote_average'] > 0]
print(f"  Removed {initial_count - len(df):,} rows with vote_average = 0")

initial_count = len(df)
df = df[df['vote_count'] >= 5]
print(f"  Removed {initial_count - len(df):,} rows with vote_count < 5")

initial_count = len(df)
df = df[df['genres'].notna() & (df['genres'].str.strip() != '')]
print(f"  Removed {initial_count - len(df):,} rows with missing genres")

initial_count = len(df)
df = df[df['overview'].notna() & (df['overview'].str.strip() != '')]
print(f"  Removed {initial_count - len(df):,} rows with missing overview")

initial_count = len(df)
df = df[df['release_date'].notna()]
print(f"  Removed {initial_count - len(df):,} rows with missing release_date")

print(f"\nDataset after filtering: {len(df):,} movies")

# 3. DROP UNNECESSARY COLUMNS
print("\nDropping unnecessary columns...")
drop_cols = ['popularity', 'tagline', 'keywords', 'producers', 'imdb_id']
# Only drop columns that exist
drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)
print(f"  Dropped: {', '.join(drop_cols)}")

# Reset index after filtering
df = df.reset_index(drop=True)

# 4. TARGET VARIABLE
print("\nCreating target variable...")
df['is_successful'] = (df['vote_average'] >= 6.0).astype(int)
print(f"  Created: is_successful (vote_average >= 6.0)")
print(f"  Success rate: {df['is_successful'].mean()*100:.1f}%")
print(f"  Successful movies: {df['is_successful'].sum():,}")
print(f"  Unsuccessful movies: {(~df['is_successful'].astype(bool)).sum():,}")

# 5. SAVE CLEANED DATA
print("\nSaving cleaned data to tmdb_cleaned.csv...")
df.to_csv('tmdb_cleaned.csv', index=False)

# 6. SUMMARY
print("\n" + "="*80)
print("CLEANING COMPLETE")
print("="*80)
print(f"Final dataset size: {len(df):,} movies")
print(f"Number of columns: {len(df.columns)}")
print(f"Success rate: {df['is_successful'].mean()*100:.1f}%")
print(f"Vote count threshold: >= 5")
print(f"Success threshold: vote_average >= 6.0")
print(f"\nSaved to: tmdb_cleaned.csv")
print("="*80)

