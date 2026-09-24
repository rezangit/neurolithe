# Tool: store_memory

Explicitly store a fact in short-term memory right away, bypassing extraction.
Works without a chat LLM (only the embedder is used).

## Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `fact_text` | string | yes | The fact, non-blank (≤ 16 KiB) |
| `tags` | string[] | no | Up to 64 tags |
| `ccl` | string | no | Cognitive context layer, default `reality` |

## Output

Text: `Memory fact explicitly stored.` On failure, an `isError` result.

## Behavior

The fact is embedded and stored as-is (`is_explicit = true`) in the active
workspace's STM. Like any STM fact, it decays unless it is read or reinforced.
