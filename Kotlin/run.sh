#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ensure_kotlin() {
  if command -v kotlinc >/dev/null 2>&1; then
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install kotlin
  fi
  command -v kotlinc >/dev/null 2>&1 || { echo "missing 'kotlinc'" >&2; exit 1; }
}

setup_java() {
  if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
    JAVA_BIN="$(command -v java)"
    return
  fi
  if [[ -x /opt/homebrew/opt/openjdk/bin/java ]]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk
    export PATH="$JAVA_HOME/bin:$PATH"
    JAVA_BIN="$JAVA_HOME/bin/java"
    return
  fi
  if [[ -x /usr/local/opt/openjdk/bin/java ]]; then
    export JAVA_HOME=/usr/local/opt/openjdk
    export PATH="$JAVA_HOME/bin:$PATH"
    JAVA_BIN="$JAVA_HOME/bin/java"
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install openjdk
    local openjdk_prefix
    openjdk_prefix="$(brew --prefix openjdk)"
    if [[ -x "$openjdk_prefix/bin/java" ]]; then
      export JAVA_HOME="$openjdk_prefix"
      export PATH="$JAVA_HOME/bin:$PATH"
      JAVA_BIN="$JAVA_HOME/bin/java"
      return
    fi
  fi
  echo "missing 'java'" >&2
  exit 1
}

ensure_kotlin
setup_java
ensure_input() {
  if [[ -f input.txt ]]; then
    return
  fi
  command -v curl >/dev/null 2>&1 || { echo "missing 'curl'" >&2; exit 1; }
  curl -fsSL "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt" -o input.txt \
    || { echo "failed to download input.txt (need curl)" >&2; exit 1; }
}
cleanup() { rm -f microgpt.jar; }
trap cleanup EXIT
ensure_input
kotlinc microgpt.kt -Werror -include-runtime -d microgpt.jar
"$JAVA_BIN" -jar microgpt.jar
