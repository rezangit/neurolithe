# Changelog

All notable changes to NeuroLithe are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.4.2] — 2026-10-18

### Fixed

- Kafka mode: the command consumer (`<group_id>-cmd`) no longer replays old
  commands. Before, a group without a committed offset (first start, a renamed
  group, or offsets the broker expired) read `memory.command` from the
  beginning, so an old `reset_soft`, `forget` or `remember` could run again
  against the current memory. Now, before joining, such a group starts at the
  end of the topic and commits that position, so a later restart still applies
  the commands sent while the daemon was down. A group that already has
  committed offsets is left as it is. The startup log names the command group
  and where each partition starts. If the topic doesn't exist yet, or the
  position can't be read, a warning is logged and the daemon starts anyway.

### Documentation

- Corrected the 0.4.0 notes: a redelivered pre-0.4.0 document is fully
  reprocessed once (summary, embedding, placement and STM fact), not only
  given its full text. Thanks to the user report.

### Added

- An "Operating and upgrading" section in the Kafka docs covers consumer
  groups, topics and keying, test instances, post-start checks and rollback.
  Kafka mode was tested against a real broker: ingest, duplicates, changed and
  deleted documents, crashes and SIGTERM mid-burst, poison messages, hard
  reset, a rebalance, and an upgrade from 0.3.0 with rollback.

## [0.4.1] — 2026-10-17

### Changed

- New `[retrieval]` defaults: `rrf_k = 5` (was 60) and `keyword_weight = 1.05`
  (was 1.0), for both `query_memory` and `recall_ltm`. In our retrieval
  evaluation a small `rrf_k` ranked better on every search path. The slightly
  higher keyword weight stops the two lists' top hits from tying exactly
  (ties always went to the vector hit), which had cost rank-1 precision.
  To keep 0.4.0's ranking, set `rrf_k = 60` and `keyword_weight = 1.0`.

## [0.4.0] — 2026-10-17

**Better search.** `query_memory` and `recall_ltm` rank results by
reciprocal rank fusion (RRF) of meaning-based and keyword matches, and
`recall_ltm` searches a document's full text instead of its first ~1,000
characters. Expect a different result order after upgrading.

**Breaking**: the eight inspection tools are merged into one `inspect` tool.

### Upgrading from 0.3.x

Replace calls to the old inspection tools with `inspect` and a `view`:

| Old tool | `inspect` view | Argument changes |
|---|---|---|
| `memory_stats` | `stats` | |
| `health` | `health` | |
| `stm_list` | `stm` | |
| `ltm_map` | `ltm_map` | |
| `inspect_node` | `node` | |
| `subtree` | `subtree` | `node` is now `id` |
| `trace_dataId` | `trace` | `dataId` is now `data_id` |
| `placement_debug` | `placement` | |

The old names now return an "unknown tool" error. Prompts or agent
instructions that mention them need updating.

`llm.provider = "vertex"` is now rejected at startup, because Vertex only
computes embeddings here. Set `llm.provider` to `none` or a chat provider,
and use `llm.embedding_provider = "vertex"` for Vertex embeddings.

Both stores migrate on first start, each with the usual backup first. STM
moves to schema v3, which adds indexes only. LTM moves to schema v2, which
adds a full-text index of documents. LTM never stored a document's full text
before, so existing documents are indexed by their title and summary only.
To make one searchable by its full text, send it again with
`remember_document` and the same `data_id` (or re-send its Kafka event).

Kafka mode: documents stored by 0.3.x have no content fingerprint, so the
first redelivery of each one after the upgrade is not recognised as a
duplicate. That includes a replay of the documents topic with a new
`group_id`. Each such document is reprocessed once, as if it had changed:
- it is summarised again (a chat-model call, if one is configured) and
  embedded again;
- it is filed again under the current placement rules, so it may move to
  another branch;
- its STM fact is replaced by a fresh one, which resets that fact's decay
  and relevance, makes it active again if it had been archived, and drops
  graph links other facts had to it.

After that, redeliveries of the same content are free. Budget for one full
ingest per old document if you replay the topic.

Both migrations are one-way: 0.3.x refuses the new stores, so to go back,
restore both backups made at migration.

### Changed

- `query_memory` ranks by RRF instead of adding vector distance to the
  keyword score. Before, any keyword match outranked every meaning-based
  match. English stop-words are left out of the keyword query, unless the
  query is nothing but stop-words.
- `query_memory` honours `k` up to 100; it was silently capped at 10.
- `recall_ltm` ranks by RRF of meaning-based search and a keyword search over
  each document's full text. Hits found only by keyword still report their
  real vector `distance`.
- `workspace_export` includes each document's full text.
- The MCP server is built on the official Rust SDK, `rmcp`. It negotiates
  protocol versions up to 2025-11-25. Clients on 2025-06-18 or later also
  receive `structuredContent` alongside the unchanged text result; arrays are
  wrapped as `{"items": [...]}`.
- Every tool carries a `title` and annotations: `readOnlyHint`,
  `destructiveHint` (on `workspace_delete`), `idempotentHint` and
  `openWorldHint`. `query_memory` is not marked read-only, because reading a
  fact reinforces it.
- With `[mcp] allow_workspace_switch = false`, `workspace_switch`,
  `workspace_create` and `workspace_delete` are no longer listed. Calling one
  still returns an explanatory error.
- `initialize` must include `capabilities` and `clientInfo`, as the spec
  requires. Malformed params get a -32602 error.
- A request sent before `initialize` (other than `ping`) is rejected with a
  -32602 error instead of being answered.
- `tools/call` with non-object `arguments` is a JSON-RPC -32602 error instead
  of an `isError` tool result.
- Shutdown waits for tool calls in flight. At stdin EOF, every accepted call
  finishes and is answered before the server exits (up to 300 s); a call the
  client cancelled gets no answer but still finishes its work. On SIGTERM or
  SIGINT, the running call gets up to 8 s, and calls not yet started get a
  "shutting down" `isError` result.
- A request that reuses the id of a request still in flight is rejected with
  a -32600 error.
- Default logging shows the MCP SDK's (`rmcp`) messages only at `warn` and
  above. An explicit `RUST_LOG` or a `[log] level` that names `rmcp`
  overrides this.
- `inspect` with `view: "node"` or `"subtree"` returns an error for an unknown
  id instead of a `null` result. An argument that belongs to another view is
  rejected.
- A stored vector whose byte length isn't a multiple of 4 is reported as
  corrupt instead of being silently truncated.
- The STM decay sweep runs in batches of 500 facts, each in its own
  transaction, instead of one pass over every fact.
- Without a chat model, the startup warning says fact extraction and
  compression "are skipped" (it said "will fail"; they were already skipped).

### Performance

- STM schema v3 adds indexes on graph edges and on `dataId`. Graph expansion
  in `query_memory`, subject lookups and `dataId` upserts no longer scan
  whole tables.

### Added

- A `[retrieval]` config section tunes the ranking of both stores:
  `rrf_k` (default 60), `vector_weight` and `keyword_weight` (1.0 each;
  see 0.4.1 for the current defaults),
  `leg_k_factor` (4), `fts_stopwords` (true) and `include_scores` (false).
  Invalid values and unknown keys are rejected at startup.
- With `[retrieval] include_scores = true`, `query_memory` and `recall_ltm`
  hits carry their fused `score`.
- `[ltm] placement_min_margin`: file a document only if its nearest concept
  beats the runner-up by at least this distance, otherwise send it to the
  inbox. The default is `0` (off) for every model.
- `inspect` with `view: "placement"` reports, per probe, the runner-up concept
  and its distance, the margin, and the placement `decision` under the current
  settings. The thresholds now include `placement_min_margin`.
- `neurolithe tools [--markdown|--json]` prints the tool reference from the
  same registry the server uses. `docs/src/mcp/tools.md` is generated from it.

### Removed

- Unused dependencies `thiserror` and `libsqlite3-sys` (direct); the `config`
  crate is built with TOML support only.

### Fixed

- The root of the LTM tree is no longer a placement target. Documents that
  were filed directly under `root` now go to a real branch or to the inbox,
  and `inspect`'s placement probes no longer report `root` as the nearest
  concept.
- `query_memory` with no searchable words (for example just `?`) returned a
  SQLite error. It now returns vector-only results.
- A relevance boost made while the decay sweep was running could be
  overwritten by the sweep.
- `forget` removes a document's stored full text too, even when the
  document was never filed.
- Kafka ingest is idempotent. Document events upsert by `data_id`: a
  redelivered event adds nothing, and a changed event replaces the stored
  version in both stores instead of adding a second LTM leaf and STM fact. A
  crash between the two stores' writes is repaired when the event is
  redelivered. Documents stored before 0.4.0 carry no content fingerprint,
  so the first redelivery of each one after the upgrade (for example a
  replay with a new `group_id`) fully reprocesses it once; see *Upgrading
  from 0.3.x*.
- The Kafka documents feeder commits an offset only once the message is
  written or forwarded to `dlq`/`parking`. A failed forward is retried instead
  of skipped, and shutdown no longer commits a message that wasn't handled.
- A hard reset commits the feeder's rewind straight away, so a restart before
  the replay's first commit no longer resumes at the old position (leaving the
  wiped stores unfilled).
- Removing an old LTM leaf (on update or forget) rewrites its parent's summary
  in the same transaction, so a crash can't leave the parent listing a
  document that is gone.

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
- **Configurable distance thresholds**: `[ltm] placement_max_distance`, `[stm] assimilation_threshold` and `[stm] accommodation_threshold`, with per-embedding-model defaults (`bge-small-en-v1.5`, `text-embedding-004`) and a warned fallback for other models. `placement_debug` now returns `{thresholds, probes}` and shows where each value came from.

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
