#!/usr/bin/env python3
"""Show histogram of reference transcript lengths from joblist (before processing)."""
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
    print("Usage: ./show_reference_joblist_histogram.py <reference_islands_joblist.txt>")
    sys.exit(1)

joblist_path = sys.argv[1]

# Parse joblist
total_lengths = []
sum_exons_lengths = []
transcript_data = []  # Store (transcript_id, total_len, sum_exons)

with open(joblist_path) as f:
    header = f.readline()  # Skip header
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split("\t")

        # Extract: transcript_id, chrom, start, end, strand, exon_blocks
        transcript_id = parts[0]
        start = int(parts[2])
        end = int(parts[3])
        exon_blocks_str = parts[5]

        # Total length (genomic span)
        total_len = end - start
        total_lengths.append(total_len)

        # Sum of exons lengths
        sum_exons = 0
        for block_str in exon_blocks_str.split(";"):
            s, e = block_str.split(",")
            sum_exons += int(e) - int(s)
        sum_exons_lengths.append(sum_exons)

        transcript_data.append((transcript_id, total_len, sum_exons))

# Create bins
bins_total = {}
bins_exons = {}

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

print(f"Total reference transcripts to process: {len(total_lengths)}")
print()

print("=== Total Length (genomic span, start to end) ===")
print(f"Mean: {sum(total_lengths)/len(total_lengths):.0f} bp")
print(f"Median: {sorted(total_lengths)[len(total_lengths)//2]:,} bp")
print(f"Min: {min(total_lengths):,} bp, Max: {max(total_lengths):,} bp")
print("\nDistribution:")
bin_order = ["<100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", ">100K"]
format_histogram(bins_total, bin_order)

print("\n=== Sum of Exons Length (spliced/exonic sequence) ===")
print(f"Mean: {sum(sum_exons_lengths)/len(sum_exons_lengths):.0f} bp")
print(f"Median: {sorted(sum_exons_lengths)[len(sum_exons_lengths)//2]:,} bp")
print(f"Min: {min(sum_exons_lengths):,} bp, Max: {max(sum_exons_lengths):,} bp")
print("\nDistribution:")
format_histogram(bins_exons, bin_order)

# Show some size stats
print("\n=== Processing Complexity ===")
print(f"Transcripts > 100K total length: {sum(1 for l in total_lengths if l > 100000)}")
print(f"Transcripts > 50K exonic length: {sum(1 for l in sum_exons_lengths if l > 50000)}")
print(f"Transcripts > 10K exonic length: {sum(1 for l in sum_exons_lengths if l > 10000)}")

# Show outliers by exonic length (what actually gets processed)
print("\n=== Outliers (>50K exonic length) ===")
outliers_exons = [(tid, tl, se) for tid, tl, se in transcript_data if se > 50000]
outliers_exons.sort(key=lambda x: x[2], reverse=True)
for tid, total_len, sum_exons in outliers_exons:
    print(f"{tid:30s} | total: {total_len:>8,} bp | exons: {sum_exons:>8,} bp")
