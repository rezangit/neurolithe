# NeuroLithe in Kafka mode: a container image for `neurolithe daemon`.
#
# Stage 1 compiles a release binary with the default features (including
# offline local embeddings via a statically linked ONNX Runtime) plus `kafka`.
# It builds librdkafka, the SQLite amalgamation and the sqlite-vec C extension
# from source. Stage 2 is a slim runtime with just that binary.
#
# Both stages use Debian 13 (trixie): the prebuilt ONNX Runtime needs
# glibc >= 2.38 and GCC 14's libstdc++, so bookworm fails to link.
#
# The standalone MCP server does not need this image; install it with
# `cargo install --path .` instead.

# ── build ────────────────────────────────────────────────────────────────────
FROM rust:1.94-slim-trixie AS builder
WORKDIR /app
# Build deps:
#   - cmake/g++/make: the vendored librdkafka (rdkafka-sys), the bundled SQLite
#     amalgamation + sqlite-vec, the rustls crypto backend (aws-lc-rs), and
#     linking ONNX Runtime (libstdc++).
#   - zlib: librdkafka's only default dependency; pkg-config wires it up.
#   - ca-certificates: TLS for build-time downloads (crates, the prebuilt
#     ONNX Runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake g++ make pkg-config zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock neurolithe.example.toml ./
COPY src ./src
RUN cargo build --release --features kafka --bin neurolithe && strip target/release/neurolithe

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:trixie-slim AS runtime
# zlib1g: the librdkafka runtime. libstdc++6: ONNX Runtime (C++).
# ca-certificates: TLS to brokers, cloud LLM providers and the one-time model
# download. SQLite + sqlite-vec are statically linked.
RUN apt-get update \
    && apt-get install -y --no-install-recommends zlib1g libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 -s /usr/sbin/nologin neurolithe \
    && mkdir -p /data && chown neurolithe:neurolithe /data && chmod 0700 /data
COPY --from=builder /app/target/release/neurolithe /usr/local/bin/neurolithe
USER neurolithe
# The home directory holds: config (optional /data/neurolithe.toml), .env,
# workspaces (the SQLite stores) and the local embedding model cache
# (/data/models, ~130 MB, downloaded on first start). Mount a persistent volume
# here, or every new container re-downloads the model. Pre-seed /data/models to
# run without network access.
ENV NEUROLITHE_HOME=/data
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/neurolithe"]
# Long-running daemon: MCP (stdio) + feeder + command/query consumers +
# schedulers. Configure it with NEUROLITHE__SECTION__KEY env vars.
CMD ["daemon"]
