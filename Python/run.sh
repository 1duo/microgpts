#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install python
  fi
  command -v python3 >/dev/null 2>&1 || { echo "missing 'python3'" >&2; exit 1; }
}

ensure_python3

ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}

fetch_gist() {
  local gist_url
  gist_url="https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw"
  GIST_FILE="$(mktemp -t microgpt-gist-XXXXXX.py)"
  curl -fsSL "$gist_url" -o "$GIST_FILE" \
    || { echo "failed to download microgpt gist (need curl)" >&2; exit 1; }
}

cleanup() { rm -f "${GIST_FILE:-}"; }
trap cleanup EXIT

ensure_input
fetch_gist
python3 "$GIST_FILE"
