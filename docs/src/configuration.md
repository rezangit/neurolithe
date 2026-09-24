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

## Kafka mode

`[kafka]`, `[kafka.topics]` and `[kafka.client]` configure the optional daemon.
See [Kafka Mode](./kafka.md).
