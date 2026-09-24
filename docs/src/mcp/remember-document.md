# Tool: remember_document

File a document or note into the **permanent** long-term archive. Its summary is
embedded and placed under the best-matching concept of the tree, or the inbox
when nothing fits.

## Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | The document text (≤ 512 KiB) |
| `title` | string | no | Leaf name; defaults to the first line of `text` |
| `data_id` | string | no | Stable id; if omitted, a new `doc_<uuidv7>` is minted |
| `tags` | string[] | no | Up to 64 tags |

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

Idempotent by `data_id`: passing an existing id replaces that document (upsert).
The new copy is placed before the old one is removed, so a failure never loses
the previous version. Find the document again with
[`recall_ltm`](./recall-ltm.md) or `trace_dataId`.
