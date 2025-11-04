import pandas as pd
import os

def merge_tmdb_parts():
    """Merge the two TMDB collection parts into a single CSV file"""
    
    print("Merging TMDB collection parts...")
    print("=" * 70)

    part1_file = 'tmdb_part1.csv'
    part2_file = 'tmdb_part2.csv'
    
    if not os.path.exists(part1_file):
        print(f"Error: {part1_file} not found")
        return None

    if not os.path.exists(part2_file):
        print(f"Error: {part2_file} not found")
        return None

    # Load both CSV files
    print(f"Loading {part1_file}...")
    df_part1 = pd.read_csv(part1_file)
    print(f"  Loaded {len(df_part1):,} movies from Part 1 (2010-2017)")

    print(f"Loading {part2_file}...")
    df_part2 = pd.read_csv(part2_file)
    print(f"  Loaded {len(df_part2):,} movies from Part 2 (2018-2025)")
    
    # Merge the dataframes
    print("\nMerging dataframes...")
    df_merged = pd.concat([df_part1, df_part2], ignore_index=True)

    # Remove duplicates based on movie ID
    initial_count = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['id'], keep='first')
    duplicates_removed = initial_count - len(df_merged)

    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate movies")

    # Sort by release date
    print("Sorting by release date...")
    df_merged = df_merged.sort_values('release_date', ascending=True)

    # Save merged file
    output_file = 'tmdb_complete.csv'
    print(f"\nSaving merged data to {output_file}...")
    df_merged.to_csv(output_file, index=False, quoting=1)

    print()
    print("=" * 70)
    print("Merge complete")
    print(f"Total movies in merged file: {len(df_merged):,}")
    print(f"Saved to: {output_file}")
    print()
    print("Summary:")
    print(f"  Part 1 (2010-2017): {len(df_part1):,} movies")
    print(f"  Part 2 (2018-2025): {len(df_part2):,} movies")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final total: {len(df_merged):,} movies")
    
    # Show year distribution
    print("\nYear distribution:")
    df_merged['year'] = pd.to_datetime(df_merged['release_date'], errors='coerce').dt.year
    year_counts = df_merged['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,} movies")
    
    return df_merged

if __name__ == "__main__":
    merge_tmdb_parts()

