# Kafka Mode (optional)

Kafka mode is an optional integration. NeuroLithe consumes document events from
a topic, files them into both memory stores, and serves reads and writes over the
bus. It is compiled only with `--features kafka` and runs as `neurolithe daemon`.
The standalone MCP server needs none of it.

```bash
cargo install --path . --features kafka   # needs cmake + a C++ compiler (librdkafka)
neurolithe daemon
```

A container example (with an optional local single-node broker) is in
`docker-compose.yml`: `docker compose --profile broker up -d --build`.

In aarch64 Linux containers or VMs on Apple Silicon, the container may print
`onnxruntime cpuid_info warning: Unknown CPU vendor` at startup. The message is
known and harmless: it comes from an ONNX Runtime static initializer and does not
affect embeddings.

Cause: ONNX Runtime is linked statically for local embeddings. A C++ global
constructor in its MLAS layer (`platform.cpp`) detects the CPU before `main` runs,
in every process, whichever embedder is configured. When ONNX Runtime's Linux
cpuinfo doesn't recognise the CPU vendor (e.g. Apple Silicon under Docker, MIDR
implementer `0x61`), it writes the notice straight to stderr, because no logger
exists yet. It is harmless: generic kernels are used. Native macOS and
x86_64/ARM-server Linux don't print it.

## Topics

| `[kafka.topics]` key | Default | Direction | Payload |
|---|---|---|---|
| `documents` | `document.completed` | in | document event (below); a null payload (tombstone) forgets the message key's `data_id` |
| `commands` | `memory.command` | in | `remember`, `forget`, `reset_soft`, `reset_hard` |
| `queries` | `memory.query` | in | recall request with a `correlationId` |
| `results` | `memory.result` | out | reply keyed by `correlationId` |
| `metrics` | `memory.metrics` | out | periodic store snapshot under one fixed key (use a compacted topic) |
| `dlq` | `dlq.memory` | out | messages that failed or had no `data_id` |
| `parking` | `parking.lot` | out | un-parseable messages |

Topics are not created automatically.

## Document events

```json
{"data_id": "doc_42", "title": "Lease agreement", "text": "The lease runs …", "tags": ["home"], "ts": "2026-09-01T10:00:00Z"}
```

`data_id` and `text` are required. `title`, `tags` and `ts` are optional, and
`dataId` is accepted as an alias. The text is summarized when a chat model is
configured, then embedded. It is filed as a permanent LTM leaf under the
best-matching concept (or the inbox), and also stored as a decaying STM fact.

- An event without `text` is skipped with a warning.
- An event without `data_id` is dead-lettered.

## Workspaces and tenants

The daemon serves exactly one workspace (`--workspace`, `NEUROLITHE_WORKSPACE` or
config `workspace`). A `tenant` field on bus messages is accepted for
compatibility and ignored.

## Configuration

```toml
[kafka]
brokers = "broker.example:9093"
group_id = "neurolithe"            # command/query consumers use <group_id>-cmd / -query

[kafka.topics]                     # optional; any subset
documents = "docs.ingest"

[kafka.client]                     # passthrough librdkafka properties, applied to every client
"security.protocol" = "SASL_SSL"
"sasl.mechanism" = "SCRAM-SHA-512"
"sasl.username" = "neurolithe"
"ssl.ca.location" = "/etc/ssl/certs/ca.pem"
```

librdkafka property names contain no `_`, so an underscore in a `[kafka.client]`
key is read as a dot. That lets secrets come from the environment:
`NEUROLITHE__KAFKA__CLIENT__SASL_PASSWORD=…`.

`[kafka.client]` cannot override `bootstrap.servers`, `group.id`,
`enable.auto.commit` or `auto.offset.reset`.

## Security

- **Anyone who can produce to the commands topic can write, forget and
  soft-reset memory.** Anyone who can read the results topic sees recall
  answers. Use SASL/TLS (`[kafka.client]`) and broker ACLs so only trusted
  services can produce to `commands`/`queries` and consume `results`.
- **Hard reset** wipes both stores, re-seeds the spine, and replays the documents
  topic. It is disabled unless `NEUROLITHE_RESET_TOKEN` is set to at least 16
  characters. A `reset_hard` command must carry `"confirm": "<that token>"`, and
  the comparison is constant-time.
