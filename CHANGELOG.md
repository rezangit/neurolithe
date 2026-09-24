# Changelog

All notable changes to NeuroLithe are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.3.0] — 2026-09-24

A clean standalone product: one user, isolated **workspaces**, **local
embeddings with no API key**, a proper home directory, versioned stores, and no
JARVIS-specific coupling. **Breaking**: config location, store layout, and the
MCP tool surface all change. See *Upgrading* below.

### Upgrading from 0.2.x

1. `cargo install --path .` (the prebuilt-binary installers are retired), then
   `neurolithe init`. This writes `~/.neurolithe/neurolithe.toml`, downloads the
   local embedding model (~130 MB, once), and prints an MCP client snippet.
2. Bring an old store over with
   `neurolithe workspace import <name> --stm <old-stm.sqlite> --ltm <old-ltm.sqlite>`.
   The source files are left untouched and a backup is taken before migrating.
   All old tenants are merged into the workspace.
3. If the old store was embedded with a different model, run
   `neurolithe reembed --workspace <name>`.
4. Update MCP client configs to the snippet `init` prints (`args: ["mcp", …]`).
   `tenant_id` arguments, `delete_tenant` and `export_tenant` are gone.

### Added

- **Workspaces**: fully separate STM+LTM store pairs under
  `<home>/workspaces/<name>/`.
  - Select one with `--workspace`, `NEUROLITHE_WORKSPACE` or config.
  - MCP tools: `workspace_current/list/create/switch/export/delete`. Delete
    requires `confirm` and refuses the active workspace; switching can be
    disabled with `[mcp] allow_workspace_switch`.
  - CLI: `neurolithe workspace list|create|delete|export|backup|import`.
    Backups use timestamped `VACUUM INTO`.
- **Local embeddings by default** (`local-embeddings` feature, on by default):
  fastembed `bge-small-en-v1.5` (384-d). The model is cached in
  `<home>/models` and works offline after the first download.
- **The chat LLM is optional** (`provider = "none"`). Memory still stores and
  searches without it; `push_dialogue` reports `learning_error: "LLM not
  configured"`.
- **`remember_document`** MCP tool, so long-term memory works in standalone
  mode: summarize (or excerpt), embed, place in the concept tree, upsert by
  `data_id`.
- **`neurolithe init`** and **`neurolithe reembed`**.
- **Store metadata and migrations**: each store records its schema version
  and embedding model/dimension.
  - An automatic backup is taken before any migration.
  - A store built with a different embedder, a newer schema, or of the wrong
    kind (STM vs LTM) is refused.
- **Configurable concept spine** via `[[ltm.spine]]`. The default is
  `notes` / `documents` / `inbox`.
- **Kafka mode is a generic optional tool**:
  - `[kafka.topics]` sets topic names.
  - `[kafka.client]` passes settings through to librdkafka (SASL/TLS).
  - Documents carry their text in the event.
- **Structured logging** via `tracing`, to stderr only. The level comes from
  `RUST_LOG`, then `[log] level`.
- **Graceful shutdown** on SIGINT/SIGTERM in both modes, with a WAL
  checkpoint. The daemon also drains its Kafka loops, commits and flushes.

### Changed

- **Home directory**: config comes from `--home` / `NEUROLITHE_HOME` /
  `~/.neurolithe`, and `--config` / `NEUROLITHE_CONFIG`. `.env` is read only
  from the home dir. The current working directory is never read, so a cloned
  repo can no longer redirect your API key. Created dirs are 0700 and files
  0600.
- **Store paths and dimensions are no longer configured**: they come from the
  workspace and the embedder. Stale `path` / `vector_dimension` keys are
  ignored with a warning.
- **MCP startup never blocks on the embedder**: `initialize`, `tools/list`
  and `ping` answer immediately while the workspace opens.
- **Hard reset over Kafka** is disabled unless `NEUROLITHE_RESET_TOKEN` is set
  to at least 16 characters, and it is compared in constant time.
- **Build image** moved to Debian trixie. The ONNX Runtime prebuilt needs
  glibc ≥ 2.38 and GCC 14.
- **sqlite-vec** upgraded to 0.1.9.

### Removed

- Tenancy on the MCP and Kafka surfaces (`tenant_id`, `delete_tenant`,
  `export_tenant`).
- The Pithos archive client and `[pithos]` config.
- The hard-coded personal-life spine, private LAN/GCP defaults and
  host-specific compose paths.
- The `install.sh` / `install.ps1` binary installers.

## [0.2.1] — 2026-09-23 (not tagged; ships as part of 0.3.0)

Correctness and hardening from the 2026-09 multi-discipline review.
Phase 0 (safety net) + Phase 1 (make it correct).

### Fixed

- **`push_dialogue` never learned anything.** Extracted facts were stored under a
  placeholder episode id `0`, failed the foreign key, and the error was discarded.
  Facts now persist. An extraction failure no longer fails the call; it is reported
  as `learning_error` on a successful context window, so a retry does not duplicate
  the turn.
- **Conflict resolver corrupted memories.** A "modify" replaced a node's text but
  kept its old embedding and dropped payload keys such as `dataId`. It now
  re-embeds, preserves payload, unions tags, and never merges nodes with different
  `dataId`s. `assimilation_threshold` is now used.
- **`delete_tenant` failed for any tenant with graph edges** (FK error). It now
  deletes edges, vectors, nodes, episodes and CCL registry rows in one
  transaction. It also **requires** `tenant_id` plus a matching `confirm`
  instead of defaulting to the `jarvis` tenant.
- **MCP clients saw every tool failure as success.** Tool results now carry
  `isError` (camelCase) as the spec requires. An unknown tool is a JSON-RPC
  `-32602` error.
- **Search ignored `k` and could return nothing.** The hard-coded `LIMIT 5` is
  gone, filters run before the limit, zero-vector anchors and archived nodes are
  kept out of the vector index (existing stores are cleaned on startup), and LTM
  recall returns the top-k hits instead of one.
- **"database is locked" with several processes.** `busy_timeout` is set before
  WAL, write transactions begin `IMMEDIATE`, and store open retries briefly.
- The Anthropic JSON extraction could panic, and a zero half-life produced NaN.

### Changed

- **Every LLM call has timeouts** via one shared HTTP client
  (`llm.request_timeout_secs`, default 120).
- **No more `dummy_key`.** A missing key logs a startup warning, and LLM tools
  return "LLM not configured: …".
- **Config is validated at load** and every problem is reported at once.
- **MCP:**
  - Supports `ping`, advertises `listChanged: false` and negotiates the protocol version.
  - Validates and clamps tool arguments.
  - Caps sizes: message 64 KiB, fact 16 KiB, query 4 KiB, line 4 MiB.
  - Tool schemas declare all accepted params and include read-only/destructive hints.
- **Sessions are keyed by (tenant, session)**, with LRU (256) and idle-TTL (6 h)
  eviction. Each extraction is capped at 32 facts and 32 relationships.
- **Secrets:**
  - The Gemini key is sent in the `x-goog-api-key` header, not the URL.
  - `provider = "custom"` only reads `NEUROLITHE_API_KEY`.
  - Errors are stripped of URLs and upstream bodies are truncated.

### Added

- `scripts/check.sh`: local quality gate (fmt, clippy `-D warnings`, tests on
  default and `kafka` features), with an optional `scripts/pre-commit` hook.
  `scripts/cargo.sh` runs cargo in Docker when no local toolchain exists.
- End-to-end MCP STDIO tests (`tests/`) driving the real binary against an
  in-process fake LLM, including regression tests tagged by issue ID.
- `rust-toolchain.toml` (1.94.1) and a generic `neurolithe.example.toml`. The
  private `neurolithe.toml` is no longer tracked.

### Removed

- The dead CI badge.

## [0.2.0] — 2026-07-10

A reliability release driven by real-world MCP testing: search actually works
now, and NeuroLithe is **standalone by default** — install the binary, run
`neurolithe mcp`, and its MCP server is ready with no Kafka or broker.

### Changed

- **Standalone by default; Kafka is opt-in.** The default build is just the
  embedded stores + the MCP server — **no `rdkafka`/librdkafka**, so it compiles
  and installs cleanly on every platform (fixing the release build that failed on
  the Kafka dependency, and restoring the prebuilt **Windows** binary). The full
  JARVIS daemon (feeder + `memory.command`/`memory.query` + metrics) is now
  behind a `kafka` cargo feature (`cargo build --release --features kafka`); the
  Docker image enables it. `neurolithe mcp` is the standalone entry point.

### Fixed

- **`query_memory` results carried no `data_id` (round 2).** Search hits now
  include the archive `data_id` (both the MCP `MemoryResult` and the bus
  `StmEntry`), so an agent goes straight from a hit to fetching the source
  instead of a second `stm_list` scan — closing the search → trace → fetch loop.
- **Ranking ignored match quality (round 2).** Results were ordered by decay
  `relevance_score`, so once reads reinforced several facts to 1.0 the best hit
  no longer ranked first (a reinforced but weaker keyword match outranked a
  stronger one). Now ranked by the hybrid vector+keyword score — direct matches
  first, graph neighbours after, relevance only as a tiebreak.
- **`query_memory` returned nothing over MCP (tenant mismatch).** The feeder
  ingests under tenant `jarvis`, but the MCP door defaulted queries to `default`,
  so the tenant filter dropped every row. The MCP door now routes through the
  **same `QueryService`** as the `memory.query` bus door and both default to one
  shared `DEFAULT_TENANT` constant, so they can't drift again.
- **Hybrid keyword search failed to zero.** The raw query was passed straight to
  FTS5 (implicit-AND; syntax error on punctuation). It's now sanitized into an
  OR of quoted terms, so search degrades to keyword-any instead of erroring or
  over-matching; the keyword leg is skipped (vector-only) when there are no terms.
- **Document summaries truncated ~2,500 chars.** Anthropic `compress_context`
  capped output at 1024 tokens; raised to 8192 so long-document summaries
  complete instead of cutting off mid-word.
- **Every document landed in the inbox.** The spine was seeded with no
  embeddings, so placement could never match a concept. The daemon now embeds
  each concept from its curated identity at startup, and seeding is **additive**
  with generic personal-life branches (health, home, vehicles, insurance,
  family, admin, …). The placement distance threshold was then **tuned from
  measured data** (new `placement_debug` CT-scan tool): real document→concept
  distances cluster ~1.05 (cosine ~0.4) for normalized `text-embedding-004`, so
  the threshold is 1.10 — earlier guesses (0.5, 0.85) sat below the whole
  distribution and filed 100% to the inbox.
- **Inbox gardener.** A startup pass re-homes inbox documents that now match a
  concept, using their **stored** embeddings — so a threshold change or a new
  branch re-files existing docs with **no replay and no LLM calls** (idempotent;
  the ambiguous tail stays in the inbox). Also exposed as `garden_inbox`.
- **Hard reset silently re-broke placement.** `hard_reset` re-seeds the spine but
  seeding leaves concepts un-embedded, so a post-reset replay filed everything
  back into the inbox. The command consumer now **re-embeds the spine after a
  hard reset** (before rewinding the feeder), so a reset → replay rebuilds a
  correctly-filed tree.
- **Personal data removed.** The seed root node is renamed from a personal name
  to the generic `"root"`, and all test fixtures use fictional data — the crate
  is reusable and carries no PII (it is a public repository).
- **Rolling summaries were degenerate.** A container's summary was one child's
  truncated text (which propagated to the root). Container summaries now describe
  the collection (`"N items: title; title; …"`).

### Added

- **`placement_debug` CT-scan tool** — reports, for a sample of document leaves, the distance to their nearest concept, so the placement threshold can be tuned to real embedding distances instead of guessed.
- **`recall_ltm` MCP tool** — reference-returning search of the permanent archive
  (dataId + provenance + ancestor concepts), the primary way to find a document.
- **Leaf titles + `ingested_at`** — leaves get a human title from the summary's
  first line (not the raw dataId), and `provenance.ingested_at` is populated from
  the leaf's creation time.
- **CT-scan ergonomics** — `stm_list` gains `offset` pagination and a `contains`
  substring filter; `inspect_node` pages children/leaves (`child_limit`/
  `child_offset`), caps summaries (`summary_max_chars`), and reports
  `child_count`/`leaf_count`. `feeder_lag: -1` is documented as "unknown".

## [0.1.2] — 2026-07-03

A large release: NeuroLithe grows from a single decaying store into a **dual
memory** brain (decaying STM + permanent LTM), gains a **daemon** mode with a
Kafka-native memory API, and a **connected working-memory graph** so an agent
can reconstruct what it just did.

### Added

- **V2 dual-memory architecture** — two independent SQLite stores: a decaying
  **STM** fact engine and a permanent, non-decaying **LTM** knowledge tree
  (concept hierarchy with document leaves). The forgetting curve can never touch
  LTM. Each store locks its own vector dimension at init.
- **Daemon run mode** — `neurolithe daemon`: a Kafka feeder consumes
  `document.completed` and dual-writes each item into STM + LTM; a **bus memory
  API** answers `memory.query` → `memory.result` and applies `memory.command`
  (remember / forget / soft+hard reset); a `memory.metrics` snapshot + read-only
  introspection tools give a "CT scan" of the brain. Dockerized
  (`docker-compose.yml`).
- **Working memory (situational awareness)** — a connected **session graph**:
  the agent's recent *turns* linked by `about` edges to the documents/entities
  they touched, with a **focus** so follow-ups ("what is *its* id?") resolve
  from context. New `working` CCL with its own short half-life; recency-first
  recall.
- **Configurable, splittable LLM** — chat and embeddings can use different
  providers: OpenAI, Google (Gemini / Vertex AI `text-embedding-004`), Anthropic
  Claude, or fully local/offline via Ollama (`nomic-embed-text`).
- **Reference-returning LTM recall** — results carry `dataId` + provenance so the
  caller can fetch originals.
- `CHANGELOG.md`.

### Changed

- STM and LTM are now **distinct regimes** (previously STM "compressed into"
  LTM). LTM is permanent; STM decays.
- **Decay is real-elapsed and per-layer** — a note decays by its true age since
  last touched (not a fixed one-day-per-sweep pass); reads reset the clock. A
  sweep or restart no longer wipes freshly-written memory. `working` notes fade
  in minutes–hours, durable facts over days.
- The **working-memory map is pure context recency** (no vector search) —
  semantic/knowledge recall stays behind the explicit `memory.query` tool.
- **MCP config must pass `args: ["mcp"]`.** Running the binary with no subcommand
  now starts the **daemon**.

### Fixed

- **Installers** wrote a V1 config (`[database]`, fixed 1536 dim) that mismatched
  the V2 stores' dimension → now write valid `[stm]` / `[ltm]` sections.
- **Install one-liner** pointed at a nonexistent `main` branch (404) → `master`.
- LTM recall could return empty (documents indexed in a separate `vec_leaves`
  table now searched on recall).
- `stm_map` semantic enrichment dredged unrelated documents/other threads into
  the situational map → scoped out.

## [0.1.1] — 2026-03-08

- Hybrid vector + FTS retrieval, adaptive forgetting curve, Cognitive Context
  Layers, MCP server over STDIO, one-line installers, release binaries.

## [0.1.0] — 2026-02-24

- First release.

[0.2.0]: https://github.com/rezangit/neurolithe/releases/tag/v0.2.0
[0.1.2]: https://github.com/rezangit/neurolithe/releases/tag/v0.1.2
[0.1.1]: https://github.com/rezangit/neurolithe/releases/tag/v0.1.1
[0.1.0]: https://github.com/rezangit/neurolithe/releases/tag/v0.1.0
