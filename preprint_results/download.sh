#!/usr/bin/env bash
set -euo pipefail

ZENODO_RECORD_ID="RECORD_ID"
ARCHIVE_NAME="CURIA_preprint_results_v1.tar.gz"
EXPECTED_SHA256="SHA256_HERE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${SCRIPT_DIR}/${ARCHIVE_NAME}"
URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/${ARCHIVE_NAME}?download=1"

echo "Downloading ${ARCHIVE_NAME}..."
curl --fail --location --progress-bar \
  "${URL}" \
  --output "${ARCHIVE_PATH}"

if [[ "${EXPECTED_SHA256}" != "SHA256_HERE" ]]; then
  echo "Verifying SHA-256 checksum..."
  echo "${EXPECTED_SHA256}  ${ARCHIVE_PATH}" | shasum -a 256 -c -
fi

echo "Extracting results..."
tar -xzf "${ARCHIVE_PATH}" -C "${SCRIPT_DIR}"

rm "${ARCHIVE_PATH}"

echo "Done. Results were extracted into:"
echo "${SCRIPT_DIR}"
