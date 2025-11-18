from pathlib import Path
import pandas as pd
import argparse

import matplotlib.pyplot as plt


def make_year_bar(csv_path: str, out_path: str = "year_counts.png"):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found")

        df = pd.read_csv(csv_path)
        if "release_date" not in df.columns:
            raise ValueError("CSV must contain a 'release_date' column")

        df['year'] = pd.to_datetime(df['release_date']).dt.year
        years = df['year'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(years.index, years.values, color="#b51515", edgecolor="k")
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Movies by Release Year")
        plt.xticks(years.index, rotation=45, ha="right")  # Show all year ticks
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
        plt.show()

    # Add year plot to command line interface
if __name__ == "__main__":
    make_year_bar("tmdb_scraping/tmdb_filtered_2.csv", "year_counts.png")