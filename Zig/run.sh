#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_zig() {
  if command -v zig >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install zig
  fi
  command -v zig >/dev/null 2>&1 || { echo "missing 'zig'" >&2; exit 1; }
}

ensure_zig
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
cleanup() { rm -rf .zig-cache zig-out; }
trap cleanup EXIT
ensure_input
cleanup
zig run -O ReleaseFast microgpt.zig
