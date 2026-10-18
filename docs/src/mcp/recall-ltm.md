# Tool: recall_ltm

Search the **permanent** long-term archive by meaning. Recall is
reference-returning: each hit carries the document's `data_id`, its provenance
and its ancestor concepts, so the caller can fetch the original.

## Input

See [`recall_ltm` in the tool reference](./tools.md#recall_ltm) for the arguments,
their limits and the tool's annotations.

## Output

A JSON array with one entry per document leaf, best first. A concept hit with
no leaves appears once, with a null `dataId`.

```json
[
  {
    "concept": "documents",
    "summary": "Lease agreement for the flat, runs to 2027 …",
    "dataId": "doc_42",
    "provenance": { "source": "remember_document", "ingested_at": "2026-09-01T10:00:00", "confidence": 1.0 },
    "distance": 0.41,
    "ancestors": ["root"],
    "score": 0.0325
  }
]
```

`score` (the fused rank score, higher is better) is present only when the
server sets `[retrieval] include_scores = true`.

## Behavior

- Two searches are fused into one ranking by reciprocal rank fusion, with the
  same `[retrieval]` settings as `query_memory` (`rrf_k`, `vector_weight`,
  `keyword_weight`, `leg_k_factor`, `fts_stopwords`):
  - **by meaning:** concept and document vectors, nearest first;
  - **by words:** each document's **full text**, not just its summary. Without
    a chat LLM a document's summary and vector only see its first ~1,000
    characters; the keyword search still finds words anywhere in it.
  - A weight of 0 switches that search off entirely (pure meaning, or pure
    keyword; keyword-only hits still report their vector `distance`).
- No distance cutoff is applied: the best `k` hits are always returned, each
  with its vector `distance` (also for hits found only by their words). An
  unrelated archive still returns its nearest nodes.
- At most 50 hits are returned, whatever `k` asks for.
- A concept hit lists at most 20 child documents. Use
  [`inspect`](./tools.md#inspect) with `view: "node"` to page further.
- Documents filed before full-text recall existed are keyword-searchable by
  their title and summary only. To index a document's whole text, file it
  again with `remember_document` and the same `data_id`.
