#!/usr/bin/env python3
"""Show histogram of reference lncRNA total_length distribution from preprocessed JSON."""
import json
import sys

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
    print("Usage: ./show_reference_lncrna_histogram.py <reference_islands.json>")
    sys.exit(1)

ref_json_path = sys.argv[1]

with open(ref_json_path) as f:
    ref_data = json.load(f)

# Extract lengths
total_lengths = []
sum_exons_lengths = []
transcripts_with_islands = []

for tid, data in ref_data.items():
    total_len = data.get("total_length", 0)
    sum_exons = data.get("sum_exons_length", 0)
    has_islands = len(data.get("islands", [])) > 0

    total_lengths.append(total_len)
    sum_exons_lengths.append(sum_exons)
    if has_islands:
        transcripts_with_islands.append(total_len)

# Create bins for total_length
bins_total = {}
bins_exons = {}
bins_with_islands = {}

for length in total_lengths:
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
    bins_total[bin_label] = bins_total.get(bin_label, 0) + 1

for length in sum_exons_lengths:
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
    bins_exons[bin_label] = bins_exons.get(bin_label, 0) + 1

for length in transcripts_with_islands:
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
    bins_with_islands[bin_label] = bins_with_islands.get(bin_label, 0) + 1

print(f"Total reference transcripts: {len(ref_data)}")
print(f"Transcripts with islands: {len(transcripts_with_islands)} ({100*len(transcripts_with_islands)/len(ref_data):.1f}%)")
print()

print("=== Total Length (genomic span) ===")
print(f"Mean: {sum(total_lengths)/len(total_lengths):.0f} bp")
print(f"Median: {sorted(total_lengths)[len(total_lengths)//2]} bp")
print(f"Min: {min(total_lengths)} bp, Max: {max(total_lengths)} bp")
print("\nDistribution:")
bin_order = ["<100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", ">100K"]
format_histogram(bins_total, bin_order)

print("\n=== Sum of Exons Length (spliced length) ===")
print(f"Mean: {sum(sum_exons_lengths)/len(sum_exons_lengths):.0f} bp")
print(f"Median: {sorted(sum_exons_lengths)[len(sum_exons_lengths)//2]} bp")
print(f"Min: {min(sum_exons_lengths)} bp, Max: {max(sum_exons_lengths)} bp")
print("\nDistribution:")
format_histogram(bins_exons, bin_order)

if transcripts_with_islands:
    print("\n=== Total Length (transcripts WITH islands only) ===")
    print(f"Mean: {sum(transcripts_with_islands)/len(transcripts_with_islands):.0f} bp")
    print(f"Median: {sorted(transcripts_with_islands)[len(transcripts_with_islands)//2]} bp")
    print(f"Min: {min(transcripts_with_islands)} bp, Max: {max(transcripts_with_islands)} bp")
    print("\nDistribution:")
    format_histogram(bins_with_islands, bin_order)
