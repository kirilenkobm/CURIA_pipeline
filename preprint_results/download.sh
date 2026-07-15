#!/usr/bin/env bash
set -euo pipefail

ZENODO_RECORD_ID="21383175"
ARCHIVE_NAME="CURIA_preprint_results_v1.tar.gz"
EXPECTED_SHA256="1c1068ab06fab6e8903179258afa4d73a1990d896e7dd204f9ee99f1d68ea0e3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${SCRIPT_DIR}/${ARCHIVE_NAME}"
URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/${ARCHIVE_NAME}?download=1"

if compgen -G "${SCRIPT_DIR}/hg38_vs_*" >/dev/null; then
  echo "Error: existing hg38_vs_* result directories were found."
  echo "Remove or move them before downloading the archived snapshot."
  exit 1
fi

echo "Downloading ${ARCHIVE_NAME}..."
curl --fail --location --progress-bar \
  "${URL}" \
  --output "${ARCHIVE_PATH}"

echo "Verifying SHA-256 checksum..."
echo "${EXPECTED_SHA256}  ${ARCHIVE_PATH}" | shasum -a 256 -c -

echo "Extracting results..."
tar -xzf "${ARCHIVE_PATH}" -C "${SCRIPT_DIR}"

rm "${ARCHIVE_PATH}"

if [[ ! -d "${SCRIPT_DIR}/hg38_vs_mm39" ]]; then
  echo "Error: extraction completed, but expected result directories were not found."
  exit 1
fi

echo "Done. Results were extracted into:"
echo "${SCRIPT_DIR}"