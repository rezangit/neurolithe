# Tool: query_memory

Search **short-term** working memory for relevant context. Hybrid search
(vector + FTS5 keyword) plus 1-hop graph expansion, with optional temporal and
layer filters. For the permanent document archive, use
[`recall_ltm`](./recall-ltm.md).

## Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | string | yes | Non-blank (≤ 4 KiB) |
| `k` | integer | no | Max results, 1–100, default 10 |
| `time_filter` | object | no | `{ "after"?: "YYYY-MM-DD", "before"?: "YYYY-MM-DD" }` |
| `ccl_filter` | string[] | no | Layers to search, default `["reality"]` |

## Output

A JSON array, best match first:

```json
[
  {
    "fact": "User lives in Berlin",
    "ccl": "reality",
    "last_updated": "2026-02-23T14:00:00",
    "connections": [
      { "relation": "LIVES_IN", "entity": "Berlin", "ccl": "reality",
        "valid_from": "2026-01-15", "valid_until": null }
    ],
    "data_id": "only present for facts that came from a document"
  }
]
```

## Behavior

- Direct hits are ranked by combined vector and keyword score. Their 1-hop
  neighbours follow.
- Reading a fact reinforces it: its relevance resets and its decay clock
  restarts.
- Read-only hint: yes. The reinforcement is the only side effect.
