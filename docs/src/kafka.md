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

## Redelivery and offsets

The documents consumer delivers at least once, and ingest is idempotent, so a
redelivered event does no harm.

- **Upsert by `data_id`.** A document is stored once: one LTM leaf and one STM
  fact per `data_id`. An event whose `title`, `text` and `tags` match the
  stored version is ignored, without summarizing or embedding it again. `ts`
  isn't compared. A changed event replaces the stored version in both stores;
  the old content is removed, not merged. The new STM fact starts fresh
  (support count 1, full relevance).
- **Crash safety.** The two stores are separate SQLite files, so one event
  can't be written in a single transaction. LTM is written first and STM last.
  Each step is its own transaction and is safe to repeat. The STM fact carries
  a fingerprint of the content, and that marks the event as done. If a crash
  comes before it, the redelivered event redoes the whole write and removes any
  duplicate leaf.
- **Offsets.** An offset is committed only after its message is handled:
  written, skipped, or forwarded to `dlq` or `parking`. Offsets aren't stored
  automatically, so the commit at shutdown covers handled messages only. A
  message cut off by a crash, a rebalance or a shutdown is delivered again.
- **Poison messages don't block the partition.** Un-parseable bytes go to
  `parking`, and events without a `data_id` go to `dlq`, straight away. An
  ingest or forget that fails 3 times (0.5 s apart) goes to `dlq`, with the
  error in the `x-reason` header. Only forwarding to `dlq` or `parking` is
  retried until it works (backing off to 30 s between tries), because
  committing first would lose the message. While it keeps failing, the
  partition waits and each failure is logged as an error.
- **Hard reset** rewinds the feeder and commits the start position at once,
  so a restart before the replay commits anything still replays from the
  beginning. A message that was being handled when the reset arrived isn't
  committed afterwards.
- **Commands are not replayed.** The documents topic is a replay source, but
  `memory.command` is not. When the command group (`<group_id>-cmd`) has no
  committed offset, it starts at the end of the topic and commits that
  position at once. That happens on a first start, with a new `group_id`, and
  after the broker expires the offsets (`offsets.retention.minutes`). Commands
  already on the topic are skipped. After that, a restart resumes where the
  group stopped, so commands sent while the daemon was down are still applied.
  The startup log shows where each partition starts:
  `command consumer: group '<group_id>-cmd' on 'memory.command': partition 0 resumes at offset N`.
- **Tombstones** are idempotent too. Forgetting a `data_id` that is unknown
  or already forgotten succeeds.

A soft reset clears STM, so a later redelivery or replay of an unchanged
event writes it again; its LTM leaf is replaced by an identical one.

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
`enable.auto.commit`, `enable.auto.offset.store` or `auto.offset.reset`
(or their aliases), because offset handling depends on them.

## Security

- **Anyone who can produce to the commands topic can write, forget and
  soft-reset memory.** Anyone who can read the results topic sees recall
  answers. Use SASL/TLS (`[kafka.client]`) and broker ACLs so only trusted
  services can produce to `commands`/`queries` and consume `results`.
- **Hard reset** wipes both stores, re-seeds the spine, and replays the documents
  topic. It is disabled unless `NEUROLITHE_RESET_TOKEN` is set to at least 16
  characters. A `reset_hard` command must carry `"confirm": "<that token>"`, and
  the comparison is constant-time.

## Operating and upgrading

### Before the first start

1. **Back up.** Stop the old instance (SIGTERM), then copy its home directory:
   the workspaces, `.env` and `neurolithe.toml`. An upgrade also backs up each
   store before migrating it (`stm.sqlite.bak-v<N>-<ts>`,
   `ltm.sqlite.bak-v<N>-<ts>`), but keep your own copy.
2. **Build with Kafka:** `cargo install --path . --features kafka`, or the
   repository `Dockerfile`.
3. **Create the topics.** They are not created automatically:
   - `document.completed`
   - `memory.command`
   - `memory.query`
   - `memory.result`
   - `dlq.memory`
   - `parking.lot`
   - `memory.metrics` (compacted)

   Producers must **key document events by `data_id`**. Tombstones need the
   key, and keying keeps every version of a document in order on one
   partition.
4. **Choose the consumer group (`[kafka] group_id`).**
   - Keep the old `group_id` to resume where the previous instance stopped.
   - A new `group_id` replays the whole documents topic from the start. That's
     safe, because ingest is idempotent, and earlier poison messages are
     forwarded to `dlq.memory` / `parking.lot` again.
     - Documents stored by 0.4.0 or later are recognised by their content
       fingerprint and skipped at almost no cost.
     - **Documents stored by 0.3.x have no fingerprint, so each one is fully
       reprocessed once.** It is summarised again (a chat-model call, if
       configured), embedded again and filed again under the current
       placement rules. Its STM fact is also replaced by a fresh one, which
       resets its decay, makes it active again if it had been archived, and
       drops graph links other facts had to it.
     - On a store upgraded from 0.3.x, budget one full ingest per old
       document for a replay.
   - The commands group (`<group_id>-cmd`) starts at the end of
     `memory.command` when it has no committed offset, so old commands are
     never replayed (see [Redelivery and offsets](#redelivery-and-offsets)).
   - **0.4.1 and earlier replayed every retained command in that case.** That
     included `reset_soft`, which needs no token and wipes STM. On those
     versions, pin the commands group to the end of the topic before the first
     start, and after any outage longer than the broker's
     `offsets.retention.minutes`:
     ```sh
     kafka-consumer-groups.sh --bootstrap-server <broker> --group <group_id>-cmd \
       --topic memory.command --reset-offsets --to-latest --execute
     ```
5. **Use the embedder the stores were written with.** It is recorded in each
   store, and a change needs `neurolithe reembed`. With local embeddings, the
   first start downloads about 130 MB into `<home>/models`.
6. **Hard reset.** Leave `NEUROLITHE_RESET_TOKEN` unset unless you need hard
   reset. `reset_soft` needs no token, so restrict who can produce to
   `memory.command`; see [Security](#security).
7. **Shutdown grace.** Give the process or container at least 15 s
   (`stop_grace_period`). SIGTERM drains the in-flight message and commits its
   offset.

### Test instances

- **Never** start a test instance against production topics with the
  production `group_id`. It joins the group, takes partitions away from the
  production instance, and files those documents into its own store.
- A test instance with a different `group_id` replays the whole documents
  topic into its own store.
- For anything that isn't production, use a separate broker, or separate topic
  names (`[kafka.topics]`) and a separate `group_id`.
- Run exactly one daemon per `group_id` and workspace.

### After the start

- **The log** shows any store migrations, with their backup paths, and
  `daemon running (workspace: …, feeder: true, …)`.
- **The group drains:** `kafka-consumer-groups.sh --describe --group
  <group_id>` shows the lag falling to 0 on every partition. After a crash
  (rather than a clean stop), the broker keeps the dead member until its
  session times out (45 s by default), so the restarted daemon can wait that
  long before it gets its partitions back.
- **The data is there:** MCP `inspect` with `view: "stats"` shows the expected
  counts, and `view: "trace"` with a known `data_id` shows one leaf and one STM
  fact.
- **The commands group** is logged as
  `command consumer: group '<group_id>-cmd' on 'memory.command': …`, with each
  partition's start: it resumes at an offset, or starts at the end.
- **No unexplained errors:** `dlq.memory` and `parking.lot` get no new
  entries, or each new entry's `x-reason` header explains it. After a hard
  reset, the replay forwards earlier poison messages to `dlq.memory` and
  `parking.lot` again, so those topics can hold the same message twice.

### Rollback

1. Stop the new version (SIGTERM).
2. An older version refuses a migrated store ("schema version … supports up
   to …"). Put the automatic backups back in place of `stm.sqlite` and
   `ltm.sqlite`, and delete any `-wal`/`-shm` files next to them. Or restore
   your own copy.
3. Documents that arrived after the upgrade are still on the documents topic.
   To re-ingest them, move the group back to the upgrade time with
   `--reset-offsets --to-datetime <time>` on `document.completed`. Versions
   before 0.4.0 don't ingest idempotently, so documents already in the backup
   get duplicates.
4. Commands sent after the upgrade are not replayed.
