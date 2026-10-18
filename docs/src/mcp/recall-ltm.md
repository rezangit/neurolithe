# Tool: recall_ltm

Search the **permanent** long-term archive by meaning. Recall is
reference-returning: each hit carries the document's `data_id`, its provenance
and its ancestor concepts, so the caller can fetch the original.

## Input

See [`recall_ltm` in the tool reference](./tools.md#recall_ltm) for the arguments,
their limits and the tool's annotations.

## Output

A JSON array with one entry per document leaf. A concept hit with no leaves
appears once, with a null `data_id`.

```json
[
  {
    "concept": "documents",
    "summary": "Lease agreement for the flat, runs to 2027 …",
    "data_id": "doc_42",
    "provenance": { "source": "remember_document", "ingested_at": "2026-09-01T10:00:00", "confidence": 1.0 },
    "distance": 0.41,
    "ancestors": ["root"]
  }
]
```

## Behavior

- Searches concept and document vectors together, nearest first.
- No distance cutoff is applied: the nearest `k` hits are always returned, each
  with its `distance`. Judge relevance from that value. An unrelated archive
  still returns its nearest nodes.
- At most 50 hits are returned, whatever `k` asks for.
- A concept hit lists at most 20 child documents. Use
  [`inspect`](./tools.md#inspect) with `view: "node"` to page further.
