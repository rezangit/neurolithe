# Tool: remember_document

File a document or note into the **permanent** long-term archive. Its summary is
embedded and placed under the best-matching concept of the tree, or the inbox
when nothing fits.

## Input

See [`remember_document` in the tool reference](./tools.md#remember_document) for the arguments,
their limits and the tool's annotations.

## Output

```json
{
  "data_id": "doc_0192…",
  "leaf_id": 57,
  "concept_path": ["root", "documents"],
  "updated": false,
  "summarized": true
}
```

- `concept_path`: where the leaf was filed (`["root", "inbox"]` if nothing matched).
- `updated`: an earlier version with this `data_id` was replaced.
- `summarized`: `false` when no chat LLM is configured. The first 1,000
  characters of the text are used as the summary instead.

## Behavior

The document's full text (with its title and tags) is kept for keyword recall,
so `recall_ltm` finds words anywhere in it, not just in the summary.

Idempotent by `data_id`: passing an existing id replaces that document (upsert),
its kept text included.
The new copy is placed before the old one is removed, so a failure never loses
the previous version. Find the document again with
[`recall_ltm`](./recall-ltm.md), or locate it with
[`inspect`](./tools.md#inspect) `view: "trace"`.
