# Tool: push_dialogue

Push the latest conversation turn to short-term memory. The message is archived,
facts are extracted from it (when a chat LLM is configured), and the optimized
context window is returned.

## Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | Conversation id (≤ 256 bytes) |
| `new_message` | string | yes | The new turn (≤ 64 KiB) |
| `ccl` | string | no | Cognitive context layer, default `reality` |

## Output

```json
{
  "summary": "Dense summary of compressed older messages (null until compression runs)",
  "recent_messages": ["Most recent raw messages still in the buffer"],
  "relevant_facts": [
    {
      "fact": "User lives in Berlin",
      "ccl": "reality",
      "last_updated": "2026-02-23T14:00:00",
      "connections": [
        { "relation": "LIVES_IN", "entity": "Berlin", "ccl": "reality",
          "valid_from": "2026-01-15", "valid_until": null }
      ]
    }
  ],
  "learning_error": "only present if fact extraction failed",
  "warnings": ["only present if e.g. compression or recall was unavailable"]
}
```

## Behavior

1. Archives the raw message as an episode (ground truth). This always happens.
2. Adds it to the in-memory session buffer. At most 256 sessions are kept; idle
   ones are evicted after 6 hours. The archived episodes are unaffected.
3. If the buffer exceeds about 4,000 tokens, compresses the oldest messages into
   a summary. This needs a chat LLM.
4. Extracts facts and relationships into the graph. This needs a chat LLM.
5. Returns `[summary] + [recent messages] + [relevant graph facts]`.

If extraction fails, or no chat LLM is configured, the call **still succeeds**
(the message is safely archived) and reports the problem in `learning_error` or
`warnings`. Don't retry because of it: a retry would archive the message twice.
