#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_node() {
  if command -v node >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install node
  fi
  command -v node >/dev/null 2>&1 || { echo "missing 'node'" >&2; exit 1; }
}

ensure_node
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
ensure_input
node microgpt.ts
