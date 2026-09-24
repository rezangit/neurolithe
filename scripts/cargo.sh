#!/usr/bin/env bash
# Run cargo natively if available, otherwise inside a cached dev container.
# Usage: scripts/cargo.sh <cargo args...>
# Env:   NL_TARGET      per-caller target subdir (default: "docker"), so several
#                       builds can run concurrently without clobbering each other.
#        NL_RUST_IMAGE  base image (default: rust:1.94-slim-trixie, rustc 1.94.1 as
#                       pinned in rust-toolchain.toml). Trixie, not bookworm: the
#                       prebuilt onnxruntime that ort links needs GCC 14's
#                       libstdc++ (`__cxa_call_terminate`); bookworm has GCC 12.
#
# The container image is built once locally from the base, with the C/C++
# toolchain that every build needs: g++ (libstdc++ for the default
# `local-embeddings` feature via ort) and cmake/make (vendored librdkafka for
# `--features kafka`), plus clippy and rustfmt. Bump DEV_REV when this recipe
# changes, so a fresh image is built.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if command -v cargo >/dev/null 2>&1; then
  exec cargo "$@"
fi

BASE="${NL_RUST_IMAGE:-rust:1.94-slim-trixie}"
DEV_REV=1
IMAGE="neurolithe-dev:$(echo "$BASE" | tr ':/' '__')-r$DEV_REV"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "scripts/cargo.sh: building dev image $IMAGE (one-off)..." >&2
  docker build -q -t "$IMAGE" - >/dev/null <<DOCKERFILE
FROM $BASE
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake make pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN rustup component add clippy rustfmt
DOCKERFILE
fi

TARGET="${NL_TARGET:-docker}"
TTY=(); [ -t 1 ] && TTY=(-t)
exec docker run --rm -i ${TTY[@]+"${TTY[@]}"} \
  -v "$ROOT":/work -w /work \
  -v neurolithe-cargo-registry:/usr/local/cargo/registry \
  -v neurolithe-cargo-git:/usr/local/cargo/git \
  -v neurolithe-cache:/root/.cache \
  -e CARGO_TARGET_DIR="/work/target/$TARGET" \
  -e CARGO_TERM_COLOR=always \
  "$IMAGE" cargo "$@"
