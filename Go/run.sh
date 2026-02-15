#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_go() {
  if command -v go >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install go
  fi
  command -v go >/dev/null 2>&1 || { echo "missing 'go'" >&2; exit 1; }
}

ensure_go
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  command -v curl >/dev/null 2>&1 || { echo "missing 'curl'" >&2; exit 1; }
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
ensure_input
go run microgpt.go
