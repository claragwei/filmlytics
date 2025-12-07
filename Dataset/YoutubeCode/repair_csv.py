import pandas as pd

input_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned.csv"
output_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned_repaired.csv"

try:
    print("🔧 Reading CSV safely using Python engine...")
    df = pd.read_csv(input_path, engine="python", on_bad_lines="skip", quoting=3)
    print(f"✅ Loaded successfully with {len(df)} rows.")
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned CSV saved to {output_path}")
except Exception as e:
    print(f"❌ Error during cleaning: {e}")
