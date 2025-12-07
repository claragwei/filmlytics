import csv

input_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned.csv"
good_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned_good.csv"
bad_path = "/Users/dylansidhu/Desktop/STA 160/tmdb_cleaned_bad.csv"

good_lines = []
bad_lines = []

with open(input_path, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    for i, line in enumerate(f):
        try:
            csv.reader([line]).__next__()
            good_lines.append(line)
        except Exception:
            bad_lines.append((i + 1, line.strip()))

print(f"✅ Found {len(good_lines)} good lines and {len(bad_lines)} bad lines.")
if bad_lines:
    print("⚠️ Example bad lines:")
    for b in bad_lines[:5]:
        print(f"Line {b[0]}: {b[1][:100]}...")

with open(bad_path, "w", encoding="utf-8") as fb:
    for _, l in bad_lines:
        fb.write(l + "\n")

print(f"💾 Bad lines saved to {bad_path}")
