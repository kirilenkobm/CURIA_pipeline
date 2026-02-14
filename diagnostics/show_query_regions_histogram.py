#!/usr/bin/env python3
"""Show histogram of query region lengths from clusters file."""
import json
import sys
from collections import Counter

def format_histogram(counts, bin_order, max_width=60):
    """Format counts as ASCII histogram in specified order."""
    if not counts:
        return

    max_count = max(counts.values())
    for length_bin in bin_order:
        if length_bin not in counts:
            continue
        count = counts[length_bin]
        bar_width = int((count / max_count) * max_width)
        bar = '█' * bar_width
        print(f"{length_bin:>10} | {bar} {count}")

if len(sys.argv) < 2:
    print("Usage: ./show_query_regions_histogram.py <query_regions_clusters.json>")
    sys.exit(1)

clusters_path = sys.argv[1]

with open(clusters_path) as f:
    clusters = json.load(f)

lengths = []
for cluster_data in clusters.values():
    merged = cluster_data["merged_region"]
    length = merged["end"] - merged["start"]
    lengths.append(length)

# Create bins
bins = {}
for length in lengths:
    if length < 100:
        bin_label = "<100"
    elif length < 500:
        bin_label = "100-500"
    elif length < 1000:
        bin_label = "500-1K"
    elif length < 5000:
        bin_label = "1K-5K"
    elif length < 10000:
        bin_label = "5K-10K"
    elif length < 50000:
        bin_label = "10K-50K"
    elif length < 100000:
        bin_label = "50K-100K"
    else:
        bin_label = ">100K"

    bins[bin_label] = bins.get(bin_label, 0) + 1

print(f"Total query regions: {len(lengths)}")
print(f"Mean length: {sum(lengths)/len(lengths):.0f} bp")
print(f"Median length: {sorted(lengths)[len(lengths)//2]} bp")
print(f"Min: {min(lengths)} bp, Max: {max(lengths)} bp")
print("\nLength distribution:")

# Define proper bin order
bin_order = ["<100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", ">100K"]
format_histogram(bins, bin_order)
