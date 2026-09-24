#!/usr/bin/env bash
# Run cargo natively if available, otherwise inside the rust:1.94 container.
# Usage: scripts/cargo.sh <cargo args...>
# Env:   NL_TARGET  — per-caller target subdir (default: "docker"), lets
#                     several builds run concurrently without clobbering.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if command -v cargo >/dev/null 2>&1; then
  exec cargo "$@"
fi
IMAGE="${NL_RUST_IMAGE:-rust:1.94-slim-bookworm}"
TARGET="${NL_TARGET:-docker}"
TTY=(); [ -t 1 ] && TTY=(-t)
exec docker run --rm -i ${TTY[@]+"${TTY[@]}"} \
  -v "$ROOT":/work -w /work \
  -v neurolithe-cargo-registry:/usr/local/cargo/registry \
  -v neurolithe-apt-cache:/var/cache/apt \
  -e CARGO_TARGET_DIR="/work/target/$TARGET" \
  -e CARGO_TERM_COLOR=always \
  "$IMAGE" bash -c '
    if [[ " $* " == *" kafka"* ]] && ! command -v cmake >/dev/null; then
      apt-get update -qq >/dev/null && apt-get install -y -qq cmake g++ make >/dev/null
    fi
    rustup component add clippy rustfmt >/dev/null 2>&1 || true
    cargo "$@"' _ "$@"
