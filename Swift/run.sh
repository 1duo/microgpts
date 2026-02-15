#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_swiftc() {
  if command -v swiftc >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]]; then
    xcode-select --install >/dev/null 2>&1 || true
    if command -v brew >/dev/null 2>&1; then
      brew install swift || true
    fi
  fi
  command -v swiftc >/dev/null 2>&1 || { echo "missing 'swiftc'" >&2; exit 1; }
}

ensure_swiftc
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
cleanup() { rm -f microgpt; }
trap cleanup EXIT
ensure_input
rm -f microgpt
swiftc -warnings-as-errors -O microgpt.swift -o microgpt
./microgpt
