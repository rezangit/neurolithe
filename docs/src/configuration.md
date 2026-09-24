# Configuration

## Where NeuroLithe looks

| What | Resolution order |
|---|---|
| Home directory | `--home <dir>` → `NEUROLITHE_HOME` → `~/.neurolithe` |
| Config file | `--config <file>` → `NEUROLITHE_CONFIG` → `<home>/neurolithe.toml` (optional) |
| API keys | the environment, then `<home>/.env` (variables already set win) |
| Workspace | `--workspace <name>` → `NEUROLITHE_WORKSPACE` → config `workspace` → `default` |

The current directory is never read for config, `.env` or data. Each workspace is
a separate memory whose two stores live in `<home>/workspaces/<name>/`.

`neurolithe init` writes a fully commented default config. Its source,
[`neurolithe.example.toml`](https://github.com/rezangit/neurolithe/blob/master/neurolithe.example.toml),
is the reference for every LLM and embedding option.

## Environment variables

Any config value can be overridden with `NEUROLITHE__<SECTION>__<KEY>`:

```bash
export NEUROLITHE__LLM__PROVIDER=gemini
export NEUROLITHE__KAFKA__BROKERS=broker.example:9093
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | Key for that provider |
| `NEUROLITHE_API_KEY` | Key for `custom` (OpenAI-compatible) endpoints; fallback for the others |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service-account key file for the `vertex` embedder |
| `NEUROLITHE_RESET_TOKEN` | Kafka mode only: enables hard reset (≥16 characters) |

Keys are never read from `neurolithe.toml`.

## Logging

Logs go to **stderr only**, because stdout is the MCP transport and carries
nothing but JSON-RPC frames. The level is chosen in this order:

1. `RUST_LOG`, if set. It accepts full filter syntax, e.g.
   `RUST_LOG=neurolithe=debug,rdkafka=warn`.
2. `[log] level` in the config file.
3. `info`.

```toml
[log]
level = "debug"        # or e.g. "neurolithe=debug,warn"
```

## Shutdown

Both modes shut down gracefully on SIGINT (Ctrl-C) or SIGTERM (e.g. `docker stop`).

- **`neurolithe mcp`:** stops when stdin closes (the client exits) or on a
  signal. It then checkpoints the stores' WAL.
- **`neurolithe daemon`:** runs these steps in order:
  1. Stops the MCP session and the schedulers.
  2. Lets each Kafka loop finish its in-flight message and commit its offset
     (up to 8 s).
  3. Flushes the last metrics snapshot.
  4. Checkpoints the WAL.

  Give supervisors a stop grace period longer than that. The example
  `docker-compose.yml` uses `stop_grace_period: 15s`, because Docker's default is
  10 s.

## The long-term memory spine

The LTM tree starts from a small curated spine. New documents are filed under
the best-matching branch, or under `inbox` when nothing fits. The default spine is
`notes` + `documents` (+ the implicit `inbox`). You can replace it with your own
branches. `path` is `/`-separated under the root:

```toml
[[ltm.spine]]
path = "work/projects"
description = "Active projects: plans, decisions, status."

[[ltm.spine]]
path = "research"
description = "Papers, articles and reading notes."
```

Seeding is additive: new branches are added to an existing tree, and existing
branches keep their documents. Describe each branch well, because the
description is what documents are matched against.

With the generic default spine, placement is essentially "inbox": every
document is about equally far from `notes` and `documents`, so few are filed
under either. For filing to work, define descriptive `[[ltm.spine]]` branches,
each with a `description` of what belongs there. See also
[Distance thresholds](#distance-thresholds).

## Distance thresholds

Three vector-distance thresholds decide what NeuroLithe does with a new piece of
memory. Their right values depend on the embedding model, because every model
has its own distance scale:

| Key | Section | Decides |
|---|---|---|
| `placement_max_distance` | `[ltm]` | A document is filed under its nearest spine concept only within this distance; otherwise it goes to `inbox`. |
| `assimilation_threshold` | `[stm]` | A new fact at or below this distance from an existing fact *is* that fact: the existing one is reinforced and keeps its text. |
| `accommodation_threshold` | `[stm]` | At or below this distance (but beyond assimilation), the new fact *refines* the existing one, which takes the new text. Beyond it, a new fact is created. Must be larger than `assimilation_threshold`. |

All three are optional. For each one, NeuroLithe uses the first of:

1. the value in your config;
2. the embedding model's default from the table below;
3. the generic fallback (the `bge-small-en-v1.5` values). NeuroLithe then logs
   one warning at startup that suggests calibrating.

| Embedding model (canonical id) | placement | assimilation | accommodation |
|---|---|---|---|
| `local:bge-small-en-v1.5` (default), `local:bge-small-en-v1.5-q` | 0.96 | 0.40 | 0.58 |
| `text-embedding-004` (any provider, e.g. `vertex:`) | 1.10 | 0.15 | 0.35 |
| anything else | fallback: 0.96 | 0.40 | 0.58 |

The `bge-small-en-v1.5` values were calibrated with the real model on a small
English test set (accurate to about ±0.02). At these values, 70% of clear
documents are filed correctly, paraphrases merge, and distinct facts about the
same entity are not overwritten.

> **Placement needs a descriptive spine.** With the generic default spine
> (`notes`, `documents`), every document is about equally far from both
> concepts (≈ 0.96), so placement is essentially "everything goes to `inbox`",
> whatever the threshold. For filing to work, define your own
> [`[[ltm.spine]]`](#the-long-term-memory-spine) branches, each with a
> `description` that says what belongs there.

Distances are sqlite-vec L2 distances (for unit-length embeddings,
L2 = √(2 − 2·cosine)). All values must be finite and greater than 0. A key in
the wrong section is rejected, so it can't be silently ignored.

```toml
[ltm]
placement_max_distance = 0.9

[stm]
assimilation_threshold = 0.25
accommodation_threshold = 0.40
```

### Calibrating with `placement_debug`

The `placement_debug` MCP tool reports the effective `thresholds`, each with its
`source` (`config`, `model_default` or `fallback`). It also reports `probes`: for
a sample of stored documents, the distance to their nearest concept. To tune
placement:

1. Store a representative set of documents (for example with
   `remember_document`).
2. Call `placement_debug` with `sample` 100–500 and look at the `distance`
   values.
3. Set `placement_max_distance` just above the distances of documents that
   clearly belong to their concept. Documents with larger distances go to
   `inbox`. A value below almost all probes files everything into `inbox`.
4. Restart. The resolved values are also logged at startup.

The STM thresholds are tuned the same way, on the distance between facts you
would or wouldn't want merged. Keep `assimilation_threshold` small (near
duplicates only), because merged facts lose their separate wording.

## Kafka mode

`[kafka]`, `[kafka.topics]` and `[kafka.client]` configure the optional daemon.
See [Kafka Mode](./kafka.md).
