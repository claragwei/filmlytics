from pathlib import Path
import argparse
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
visualization_maker.py

Creates a bar plot counting movies by vote_average from a CSV file,
using bins of a given size (default 0.5).
"""

import matplotlib.pyplot as plt


def make_vote_average_bar(csv_path: str, bin_size: float = 0.5, out_path: str = "vote_average_counts.png"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if "vote_average" not in df.columns:
        raise ValueError("CSV must contain a 'vote_average' column")

    votes = df["vote_average"].dropna().astype(float)
    if votes.empty:
        raise ValueError("No vote_average values found in the CSV")

    # create bin edges (cover full range, include last bin)
    min_edge = np.floor(votes.min() / bin_size) * bin_size
    max_edge = np.ceil(votes.max() / bin_size) * bin_size
    bins = np.arange(min_edge, max_edge + bin_size, bin_size)

    # bin the data and count per interval (left-inclusive, right-exclusive)
    binned = pd.cut(votes, bins=bins, right=False, include_lowest=True)
    counts = binned.value_counts(sort=False)

    # readable labels for x-axis
    labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in counts.index]

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, counts.values, color="#b51515", edgecolor="k")
    ax.set_xlabel("Vote Average")
    ax.set_ylabel("Number of Movies")
    ax.set_title(f"Movies by Average Vote")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bar plot of movie counts by vote_average.")
    parser.add_argument("--csv", default="tmdb_scraping/tmdb_filtered_2.csv", help="Path to CSV file (default: tmdb_scraping/tmdb_filtered_2.csv)")
    parser.add_argument("--bin-size", type=float, default=0.5, help="Bin size for vote_average (default: 0.5)")
    parser.add_argument("--out", default="vote_average_counts.png", help="Output image path")
    args = parser.parse_args()

    make_vote_average_bar(args.csv, args.bin_size, args.out)


    