# Introspection tools ("CT scan")

These tools are read-only. They show what the active workspace's memory holds
and never call an LLM.

| Tool | Input | Returns |
|---|---|---|
| `memory_stats` | none | Full metrics snapshot of both stores: STM active/archived counts, average relevance, decay histogram; LTM tree nodes, leaves, edges, inbox size, orphan leaves, max depth; DB sizes |
| `health` | none | Compact summary: STM/LTM counts, orphan leaves, DB sizes, `feeder_lag` (`-1` = unknown here; Kafka mode publishes live lag on the metrics topic) |
| `stm_list` | `limit` (1–500, default 20), `offset` (default 0), `status` (`active`\|`archived`), `contains` (case-insensitive substring) | STM facts, most relevant first, with score, status, support count, layer, last access and `data_id` |
| `ltm_map` | `depth` (1–10, default 3) | The top concept layers of the LTM tree (a table of contents) |
| `inspect_node` | `id` (required), `child_limit` (1–500, default 50), `child_offset`, `summary_max_chars` (0 = full, default 200) | One LTM node: summary, parents, children and document leaves (paged), plus `child_count` / `leaf_count` totals |
| `subtree` | `node` (required), `depth` (1–10, default 2) | A branch of concepts below a node |
| `trace_dataId` | `dataId` (required) | Where a document lives: its LTM leaf, ancestor branch, and how many STM facts carry it |
| `placement_debug` | `sample` (1–500, default 30) | For a sample of document leaves, the distance to their nearest concept, for tuning the placement threshold |
