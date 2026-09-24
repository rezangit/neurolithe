<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="content/image/neurolithe-logo-full.png">
    <img width="500" alt="NeuroLithe" src="content/image/neurolithe-logo-full.png">
  </picture>
</p>

<p align="center">
  <b>A fast, embedded, and efficient contextual memory database for AI agents.</b>
</p>

<p align="center">
  <a href="https://github.com/rezangit/neurolithe/releases"><img src="https://img.shields.io/github/v/release/rezangit/neurolithe?color=cyan" alt="Release"></a>
  <a href="https://github.com/rezangit/neurolithe/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-cyan.svg" alt="License"></a>
  <a href="https://docs.neurolithe.com"><img src="https://img.shields.io/badge/docs-neurolithe.com-cyan.svg" alt="Docs"></a>
  <a href="https://neurolithe.com"><img src="https://img.shields.io/badge/website-neurolithe.com-cyan.svg" alt="Website"></a>
</p>

**NeuroLithe** is built in 🦀 Rust to solve the **context memory problem** for AI agents. It gives an agent **two distinct memory regimes** — a fast, *decaying* **short-term memory** (STM) for "what's happening right now," and a permanent, non-decaying **long-term memory** (LTM) knowledge tree — both searchable by meaning (`sqlite-vec`) and by keyword (FTS5), so an agent recalls the right context without drowning the prompt in full history.

Run it as an embedded **MCP server** over STDIO (drop-in agent memory). You can also run it as an optional long-running **Kafka daemon** that ingests documents from an event stream and answers memory queries over the bus.

<p align="center">
<strong><a href="#-quick-start">Quick Start</a> • <a href="#-features">Features</a> • <a href="#-tech-stack">Tech Stack</a> • <a href="#-contributing">Contributing</a> • <a href="https://docs.neurolithe.com">Documentation</a></strong>
</p>

## 🚀 Quick Start

### 1. Install

NeuroLithe is built from source with a native [Rust](https://rustup.rs/) toolchain:

```bash
git clone https://github.com/rezangit/neurolithe.git
cd neurolithe
cargo install --path .        # standalone MCP server (default)
neurolithe init               # creates ~/.neurolithe + a commented config, prints an MCP snippet
```

The default build needs no Kafka/`librdkafka` toolchain. It includes offline
**local embeddings** (`bge-small-en-v1.5` via ONNX Runtime), so no API key is
needed. With no chat model configured (`[llm] provider = "none"`, the default),
fact extraction and summaries are skipped, but memory is still stored and
searched. Kafka mode is an opt-in build (see [Kafka mode](#-kafka-mode-optional)).

<details>
<summary><b>Platform notes for the default build (local embeddings)</b></summary>
<br>

- **First build:** downloads a prebuilt ONNX Runtime from `cdn.pyke.io`
  (SHA-verified) and links it statically. For an offline build, point
  `ORT_LIB_LOCATION` at a local onnxruntime.
- **First run:** downloads the model (~130 MB) into `<home>/models`. After that it
  runs fully offline.
- **macOS Apple Silicon:** works. It needs the Xcode Command Line Tools, which the
  bundled SQLite already requires.
- **macOS Intel (x86_64):** there is no prebuilt ONNX Runtime. Build with
  `--no-default-features` and use a remote embedder (`openai`, `gemini`, or
  `custom` with Ollama), or supply your own onnxruntime via `ORT_LIB_LOCATION`.
- **Linux:** needs glibc ≥ 2.38 and GCC 14's libstdc++ (Debian 13, Ubuntu 24.10+,
  Fedora 40+). Ubuntu 24.04 ships GCC 13 and fails to link. Use
  `--no-default-features` or the Docker image instead.
- **Windows x86_64:** a prebuilt runtime exists (not tested).

Other local models include `bge-small-en-v1.5-q`, `bge-base-en-v1.5`,
`bge-large-en-v1.5`, `all-minilm-l6-v2`, `multilingual-e5-small`, `bge-m3`,
`embeddinggemma-300m` and `mxbai-embed-large-v1`. Changing the model on an existing
store requires `neurolithe reembed`.

</details>

`neurolithe init` accepts `--home <dir>`, and so does every other command. The home
directory holds `neurolithe.toml`, an optional `.env` for API keys, the workspaces
(the SQLite stores) and the local model cache. The current directory is never read
for config, `.env` or data.

### 2. Connect your AI agent

Paste the snippet that `neurolithe init` printed into your MCP client (*Claude
Desktop, Cursor, Claude Code, …*). It uses the absolute path of the installed binary:

```json
{
  "mcpServers": {
    "neurolithe": {
      "command": "/Users/you/.cargo/bin/neurolithe",
      "args": ["mcp"]
    }
  }
}
```

Add `"--workspace", "<name>"` to `args` to give a client its own separate memory.
Logs go to stderr only (stdout is the MCP transport). Set the level with `RUST_LOG`
or `[log] level` (default `info`). SIGINT/SIGTERM shut down gracefully, with a WAL
checkpoint; see the [configuration docs](docs/src/configuration.md).
To use a cloud chat model (fact extraction, summaries), set its provider in
`~/.neurolithe/neurolithe.toml` and its key in `~/.neurolithe/.env`, e.g.
`OPENAI_API_KEY=…`. Keys are never read from the config file itself.

## 🔀 Run modes

```bash
neurolithe mcp      # MCP server over STDIO (embedded agent memory) — always available
neurolithe daemon   # Kafka mode: MCP + document feeder + bus memory API + schedulers
                    #   (only in builds compiled with --features kafka)
```

- **`mcp`** is the drop-in option for an MCP client. It needs no Kafka and no broker.
- **`daemon`** is the event-driven mode, described below.

## 📨 Kafka mode (optional)

Kafka mode is a generic, optional integration for feeding documents to NeuroLithe
from an event stream and for reading and writing memory over the bus. Build it with:

```bash
cargo install --path . --features kafka   # needs cmake + a C++ compiler for librdkafka
neurolithe daemon
```

or run the container example in [`docker-compose.yml`](docker-compose.yml):

```bash
docker compose --profile broker up -d --build   # includes a local single-node broker
```

> [!NOTE]
> In aarch64 Linux containers or VMs on Apple Silicon, the container may print
> `onnxruntime cpuid_info warning: Unknown CPU vendor` at startup. The message is
> known and harmless: it comes from an ONNX Runtime static initializer and does not
> affect embeddings.

| Topic (`[kafka.topics]` key) | Default | Direction | Payload |
|---|---|---|---|
| `documents` | `document.completed` | in | `{"data_id": "...", "title"?: "...", "text": "...", "tags"?: [...], "ts"?: "..."}`. The key is the `data_id`; a null payload (tombstone) forgets it |
| `commands` | `memory.command` | in | `remember` / `forget` / `reset_soft` / `reset_hard` |
| `queries` → `results` | `memory.query` → `memory.result` | in → out | recall request/reply, keyed by `correlationId` |
| `metrics` | `memory.metrics` | out | periodic store snapshot (single key; use a compacted topic) |
| `dlq` / `parking` | `dlq.memory` / `parking.lot` | out | failed / un-parseable messages |

Handling rules:

- An event without `text` is skipped with a warning.
- An event without `data_id` goes to the DLQ.
- The daemon serves exactly one workspace. A `tenant` field on bus messages is
  accepted for compatibility and ignored.

**Configuration:**

```toml
[kafka]
brokers = "broker.example:9093"
group_id = "neurolithe"            # command/query consumers use <group_id>-cmd / -query

[kafka.topics]                     # optional; any subset
documents = "docs.ingest"

[kafka.client]                     # passthrough librdkafka properties for every client
"security.protocol" = "SASL_SSL"
"sasl.mechanism" = "SCRAM-SHA-512"
"sasl.username" = "neurolithe"
"ssl.ca.location" = "/etc/ssl/certs/ca.pem"
```

Secrets can come from the environment instead. Write the key with underscores for
dots, e.g. `NEUROLITHE__KAFKA__CLIENT__SASL_PASSWORD=…`. `[kafka.client]` cannot
override `bootstrap.servers`, `group.id` or the offset settings.

> [!WARNING]
> Anyone who can produce to the commands topic can write, forget and soft-reset
> memory. Use SASL/TLS and broker ACLs. **Hard reset** (wipes both stores, then
> replays the documents topic) is disabled unless `NEUROLITHE_RESET_TOKEN` is set
> to at least 16 characters. The command's `confirm` value must match it, and the
> comparison is constant-time.

## ✨ Features

- 🧠 **Dual-memory architecture (V2):** two independent SQLite stores, so the two problems don't fight each other:
  - **Short-Term Memory (STM)** — a fast, *decaying* fact engine for recent, relevant context. Facts fade on a half-life curve unless reinforced.
  - **Long-Term Memory (LTM)** — a permanent, *non-decaying* knowledge tree. Documents/notes are placed as leaves under a growing concept hierarchy; the forgetting curve never touches it.
- 🔎 **Hybrid retrieval:** semantic vector search (`sqlite-vec`) + BM25 keyword search (FTS5) + 1-hop graph traversal, natively in SQL. LTM recall is *reference-returning* (`dataId` + provenance) so you can fetch originals.
- 🧭 **Working memory (situational awareness):** a connected **session graph** — the agent's recent *turns* linked by `about` edges to the documents/entities they touched, with a **focus** so follow-ups like "what is *its* id?" resolve from context, not a fuzzy re-search.
- ⏱️ **Adaptive forgetting curve:** real-elapsed, **per-layer** exponential decay — situational notes fade in minutes-to-hours, durable facts over days; reads reinforce (reset the clock). A sweep or restart never wipes fresh memory.
- 🌌 **Cognitive Context Layers (CCL):** segregate memories by conceptual layer (`reality`, `working`, `dream`, `simulation`, …) to prevent cross-talk and enable counterfactual reasoning.
- 🔌 **Two run modes:** an embedded **MCP server** over STDIO (drop-in agent memory), or an optional **Kafka daemon** that ingests document events and serves reads/writes over the bus. See [Kafka mode](#-kafka-mode-optional).
- 🩻 **Introspection ("CT scan"):** read-only tools (and, in Kafka mode, a metrics snapshot) to see exactly what memory holds (STM/LTM counts, sizes, layers).
- 🛠️ **Bring your own LLM:** embeddings run **locally and offline** by default. A chat model is optional; chat and embeddings can be configured separately: OpenAI, Google (Gemini / Vertex), Anthropic Claude, or any OpenAI-compatible endpoint such as Ollama. Keys come from the environment or `<home>/.env`.
- 🗄️ **Zero external infra (MCP mode):** runs locally as an embedded database — nothing to manage. (Kafka mode adds a broker when you want the event-driven pipeline.)

## 🛠️ Tech Stack

NeuroLithe is built for speed, safety, and conciseness using modern technologies:

- **Language:** [Rust](https://www.rust-lang.org/) — Ensuring memory safety, high performance, and fearless concurrency.
- **Database:** [SQLite](https://sqlite.org/) + `rusqlite` — Fast, file-based SQL database optimized with WAL mode.
- **Vector Search:** `sqlite-vec` & FTS5 — Powering hybrid search (semantic vector embeddings + BM25 full-text search) natively in SQL.
- **Async Runtime:** `tokio` — Handling concurrent operations efficiently.
- **LLM Integration:** `reqwest` & `serde` — Provider-agnostic clients for OpenAI, Google (Gemini / Vertex AI), Anthropic Claude, and local OpenAI-compatible endpoints (Ollama). Chat and embedding providers are configured independently.
- **Event Backbone (optional):** `rdkafka` behind the `kafka` feature. It consumes document events and serves a request/reply memory API over Kafka. Not compiled into the standalone build.
- **Protocol:** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) — Operating seamlessly as an intelligent MCP server over standard input/output (STDIO).

## 🤝 Contributing

We welcome contributions! NeuroLithe is built using **Domain-Driven Design (DDD)**. To keep the project clean, scalable, and testable, contributors must adhere to this architectural pattern.

<details>
<summary><b>View Project Architecture Overview</b></summary>
<br>

- **`src/domain/`**: The core of the application. Contains business models, logic (e.g., decay math), and interfaces (`ports`). Zero external networking or database logic belongs here.
- **`src/infrastructure/`**: Concrete implementations of the `ports`. This is where `rusqlite` database connections, `reqwest` LLM clients, and raw SQL schemas live.
- **`src/application/`**: Use cases and orchestrators (like `RetrievalService` or `SleepWorker`). This layer wires the domain and infrastructure together.
- **`src/interfaces/`**: The outer boundary. Contains the MCP server, JSON-RPC parsing, and STDIO handlers.

</details>

### Contribution Guidelines

1. **Test Early, Test Often:** We expect comprehensive unit tests within your modules alongside integration tests targeting the database. Include tests *in the same PR* as your feature.
2. **Feature Branches:** Never commit directly to `master`/`main`. Create a descriptive branch from the latest root:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Pull Requests:** Open a Pull Request outlining *what* changed and *why*. Make sure the quality gate passes before requesting a review.

### Quality gate (no CI, run it locally)

```bash
scripts/check.sh           # fmt --check, clippy -D warnings, tests (default + kafka features)
scripts/check.sh --quick   # default features only
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit   # optional: quick gate on every commit
```

`scripts/cargo.sh` runs the pinned toolchain (`rust-toolchain.toml`) natively, or in Docker
(`rust:1.94`) when no host toolchain is installed. The end-to-end tests in `tests/` spawn the
real `neurolithe mcp` binary over STDIO against an in-process fake OpenAI-compatible LLM, so
they need no API key or network.

For local runs, `neurolithe init` writes a config from `neurolithe.example.toml` into your home directory. Use `--home` to keep a test setup separate.

## 📝 License

MIT License
