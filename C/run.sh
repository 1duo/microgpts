#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_clang() {
  if command -v clang >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]]; then
    xcode-select --install >/dev/null 2>&1 || true
    if command -v brew >/dev/null 2>&1; then
      brew install llvm
      export PATH="$(brew --prefix llvm)/bin:$PATH"
    fi
  fi
  command -v clang >/dev/null 2>&1 || { echo "missing 'clang'" >&2; exit 1; }
}

ensure_clang
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  command -v curl >/dev/null 2>&1 || { echo "missing 'curl'" >&2; exit 1; }
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
cleanup() { rm -f microgpt; }
trap cleanup EXIT
ensure_input
if clang -std=c23 -x c -fsyntax-only /dev/null >/dev/null 2>&1; then
  CSTD="-std=c23"
else
  CSTD="-std=c2x"
fi
clang -O3 "$CSTD" -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wnull-dereference -Werror -o microgpt microgpt.c -lm
./microgpt
