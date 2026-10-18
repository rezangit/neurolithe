# Tool: query_memory

Search **short-term** working memory for relevant context. Hybrid search
(vector + FTS5 keyword) plus 1-hop graph expansion, with optional temporal and
layer filters. For the permanent document archive, use
[`recall_ltm`](./recall-ltm.md).

## Input

See [`query_memory` in the tool reference](./tools.md#query_memory) for the arguments,
their limits and the tool's annotations.

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
- Not read-only, because of that reinforcement (`readOnlyHint: false`,
  `destructiveHint: false`, `idempotentHint: true`).
