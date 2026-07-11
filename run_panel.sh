#!/usr/bin/env bash
# Full-panel CURIA driver: hg38 reference vs a set of query species, ONE GPU lane.
#
# Runs species SEQUENTIALLY (one curia process at a time) so a single H100 is not
# oversubscribed. Reference-island scanning is species-independent, so every run
# shares ONE --ref-islands-db (hg38_ref_islands.db); after the first species the
# reference is served from cache instead of re-embedded.
#
# Usage:
#   ./run_panel.sh                 # run the default PANEL below
#   ./run_panel.sh rheMac10        # smoke test: single species
#   ./run_panel.sh sp1 sp2 sp3     # arbitrary subset
#
# Input naming convention (edit chain_for / twobit_for if yours differs):
#   reference 2bit : ../hg38.2bit
#   query 2bit     : ../<sp>.2bit
#   chain          : ../hg38.<sp>.allfilled.chain.gz
set -uo pipefail

# ---- paths ----------------------------------------------------------------
PY=./.venv/bin/python3
REF_BED=input_data/reference_annotation/hg38.primary_only.bed
REF_META=input_data/reference_annotation/hg38.primary_only.transcript_metadata.tsv
REF_2BIT=../hg38.2bit
OUT_ROOT=preprint_results
REF_ISLANDS_DB=hg38_ref_islands.db          # shared cache — one per lane

chain_for()  { echo "../hg38.$1.allfilled.chain.gz"; }
twobit_for() { echo "../$1.2bit"; }

# ---- H100 (80 GB) GPU tuning ----------------------------------------------
# gpu-max-tokens is the real memory guard (caps count*padded_len for long
# embed_once islands). 32 GB card ran 65536; 80 GB gives headroom for ~3x.
# If you OOM: drop MAX_TOKENS to 131072, then 98304, before touching MAX_BATCH.
GPU_MAX_BATCH=${GPU_MAX_BATCH:-12800}
GPU_MAX_TOKENS=${GPU_MAX_TOKENS:-196608}
GPU_MIN_BATCH=${GPU_MIN_BATCH:-32}
CPU_WORKERS=${CPU_WORKERS:-256}             # = host thread count (EPYC 7713P: 256)

# ---- panel source ----------------------------------------------------------
# Priority: CLI args > paper/species_panel.txt > built-in fallback.
# Marsupials use hybrid best-chain projection (classifier over-filters at 180 My).
PANEL_FILE=${PANEL_FILE:-paper/species_panel.txt}
PANEL_FALLBACK=(
  gorGor6 rheMac10 mm39 rn7 HLoryCun3 bosTau9 susScr11 equCab3 felCat9
  eriEur2 HLpteVam2 dasNov3 HLeleMax1 monDom5 HLnotEug3
)
MARSUPIALS=" monDom5 HLdidVir1 HLnotEug3 "

if [ "$#" -gt 0 ]; then
  PANEL=("$@")
elif [ -f "$PANEL_FILE" ]; then
  PANEL=()
  while read -r sp || [ -n "$sp" ]; do
    sp="${sp%%[[:space:]]}"; [ -n "$sp" ] && [ "${sp#\#}" = "$sp" ] && PANEL+=("$sp")
  done < "$PANEL_FILE"
  echo "# Panel from $PANEL_FILE"
else
  PANEL=("${PANEL_FALLBACK[@]}")
fi

# Chain<->2bit name mismatches (e.g. HLoryCun3: accession '.1' suffix) are
# auto-normalized at startup instead of failing 16 min in.
AUTO_FIX=${AUTO_FIX:---auto-fix-chrom-names}

echo "# Panel (${#PANEL[@]}): ${PANEL[*]}"
echo "# GPU: max-batch=$GPU_MAX_BATCH max-tokens=$GPU_MAX_TOKENS  CPU workers=$CPU_WORKERS"
echo "# Shared ref-islands cache: $REF_ISLANDS_DB"
echo

failed=()
for sp in "${PANEL[@]}"; do
  chain=$(chain_for "$sp"); qbit=$(twobit_for "$sp")
  out="$OUT_ROOT/hg38_vs_$sp"
  if [ ! -f "$chain" ] || [ ! -f "$qbit" ]; then
    echo "# [SKIP] $sp — missing input (chain=$chain qbit=$qbit)"; failed+=("$sp:missing-input"); continue
  fi
  proj=(--projection-mode orthologous)
  case "$MARSUPIALS" in *" $sp "*) proj=(--projection-mode best-chain --best-chain-topk 1);; esac

  echo "=================================================================="
  echo "# $sp -> $out  (${proj[*]})   $(date '+%H:%M:%S')"
  echo "=================================================================="
  $PY ./curia.py \
    --ref-bed12 "$REF_BED" --reference-metadata "$REF_META" \
    --chain "$chain" --ref-2bit "$REF_2BIT" --query-2bit "$qbit" \
    --output-dir "$out" \
    --ref-islands-db "$REF_ISLANDS_DB" \
    $AUTO_FIX \
    "${proj[@]}" \
    --gpu-max-batch "$GPU_MAX_BATCH" --gpu-min-batch "$GPU_MIN_BATCH" \
    --gpu-max-tokens "$GPU_MAX_TOKENS" --cpu-max-workers "$CPU_WORKERS" \
    --gpu-logger --skip-preflight
  rc=$?
  if [ $rc -ne 0 ]; then echo "# [FAIL] $sp exited $rc"; failed+=("$sp:rc$rc"); fi
done

echo
echo "=================================================================="
echo "# DONE. ${#PANEL[@]} attempted, ${#failed[@]} failed."
[ ${#failed[@]} -gt 0 ] && printf '#   %s\n' "${failed[@]}"
echo "=================================================================="
