#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-}"

ensure_clangxx() {
  if [[ -n "$CXX" ]] && command -v "$CXX" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && [[ -x /usr/bin/clang++ ]]; then
    CXX=/usr/bin/clang++
    return
  fi
  if command -v clang++ >/dev/null 2>&1; then
    CXX="$(command -v clang++)"
    return
  fi
  if [[ "$(uname)" == "Darwin" ]]; then
    xcode-select --install >/dev/null 2>&1 || true
    if command -v brew >/dev/null 2>&1; then
      brew install llvm || true
      if [[ -x "$(brew --prefix llvm)/bin/clang++" ]]; then
        CXX="$(brew --prefix llvm)/bin/clang++"
        return
      fi
    fi
  fi
  command -v clang++ >/dev/null 2>&1 || { echo "missing 'clang++'" >&2; exit 1; }
  CXX="$(command -v clang++)"
}

ensure_input() {
  [[ -f input.txt ]] && return
  command -v curl >/dev/null 2>&1 || { echo "missing 'curl'" >&2; exit 1; }
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}

cleanup() { rm -f microgpt; }
trap cleanup EXIT

ensure_clangxx
ensure_input
"$CXX" -O3 -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wnull-dereference -Werror -o microgpt microgpt.cpp
./microgpt
