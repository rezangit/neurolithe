# Tool: store_memory

Explicitly store a fact in short-term memory right away, bypassing extraction.
Works without a chat LLM (only the embedder is used).

## Input

See [`store_memory` in the tool reference](./tools.md#store_memory) for the arguments,
their limits and the tool's annotations.

## Output

Text: `Memory fact explicitly stored.` On failure, an `isError` result.

## Behavior

The fact is embedded and stored as-is (`is_explicit = true`) in the active
workspace's STM. Like any STM fact, it decays unless it is read or reinforced.
