#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/bulk_download_bfsp.sh ukwin 2021-01-01 2026-02-26
#   ./scripts/bulk_download_bfsp.sh ukplace 2021-01-01 2026-02-26

MARKET="${1:-ukwin}"
START_DATE="${2:-2021-01-01}"
END_DATE="${3:-$(date +%F)}"

RAW_ROOT="/media/ben/STORAGE/Horse Racing"
LOG_DIR="${RAW_ROOT}/_logs_${MARKET}_download"

# Prefer the currently active venv python (so it works from PyCharm terminal)
VENV_PY="$(python -c 'import sys; print(sys.executable)')"

# Point at your existing downloader project for now
DOWNLOADER="/home/ben/projects/bfsp_downloader/bfsp_download_day.py"

mkdir -p "${LOG_DIR}"

echo "Market: ${MARKET}"
echo "Range : ${START_DATE} -> ${END_DATE}"
echo "Logs  : ${LOG_DIR}"
echo "Py    : ${VENV_PY}"
echo "DL    : ${DOWNLOADER}"
echo

d="${START_DATE}"
fails=0

while [[ "${d}" < "${END_DATE}" || "${d}" == "${END_DATE}" ]]; do
  y="$(date -d "${d}" +%Y)"
  mname="$(date -d "${d}" +'%B %Y')"
  ddmmyyyy="$(date -d "${d}" +%d%m%Y)"

  out_dir="${RAW_ROOT}/${y}/Results/${mname}"
  mkdir -p "${out_dir}"

  out_file="${out_dir}/dwbfprices${MARKET}${ddmmyyyy}.csv"
  log_file="${LOG_DIR}/${MARKET}_${d}.log"

  # Skip if already downloaded and non-empty
  if [[ -s "${out_file}" ]]; then
    echo "[SKIP] ${d} -> exists"
    d="$(date -I -d "${d} + 1 day")"
    continue
  fi

  echo "[GET ] ${d} -> ${out_file}"

  # Call your downloader; it must support market + date + output
  # Adjust args here if your downloader expects different flags.
  if "${VENV_PY}" "${DOWNLOADER}" --market "${MARKET}" --date "${d}" --out "${out_file}" > "${log_file}" 2>&1; then
    # Validate file looks like CSV with header (basic guard)
    head1="$(head -n 1 "${out_file}" || true)"
    if [[ "${#head1}" -lt 10 ]]; then
      echo "[BAD ] ${d} -> empty/invalid header (see log)"
      rm -f "${out_file}"
      echo "${d},bad_header" >> "${LOG_DIR}/failures.csv"
      fails=$((fails+1))
    else
      echo "[OK  ] ${d}"
    fi
  else
    echo "[FAIL] ${d} (see log)"
    rm -f "${out_file}"
    echo "${d},download_error" >> "${LOG_DIR}/failures.csv"
    fails=$((fails+1))
  fi

  # Polite sleep to avoid hammering endpoints
  sleep 1

  d="$(date -I -d "${d} + 1 day")"
done

echo
echo "Done. Failures: ${fails}"
if [[ -f "${LOG_DIR}/failures.csv" ]]; then
  echo "Failure log: ${LOG_DIR}/failures.csv"
fi