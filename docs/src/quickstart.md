# Quickstart Guide

Get NeuroLithe running in a few minutes.

## Install

NeuroLithe is built from source with a native [Rust](https://rustup.rs/) toolchain:

```bash
git clone https://github.com/rezangit/neurolithe.git
cd neurolithe
cargo install --path .
neurolithe init
```

`neurolithe init` creates the home directory (default `~/.neurolithe`, mode `0700`),
writes a commented `neurolithe.toml`, and prints a ready-to-paste MCP client
snippet that uses the absolute path of the installed binary.

The default build embeds text locally and offline (`bge-small-en-v1.5` via ONNX
Runtime), so no API key is needed to start. The first build downloads a prebuilt
ONNX Runtime, and the first run downloads the model (~130 MB) into
`<home>/models`. Platform limits apply:

- **macOS Intel:** has no prebuilt runtime.
- **Linux:** needs glibc ≥ 2.38 and GCC 14's libstdc++ (e.g. Debian 13,
  Ubuntu 24.10+).

On other platforms, build with `--no-default-features` and configure a remote
embedder. Kafka mode is a separate, opt-in build (`--features kafka`, see
[Kafka mode](./kafka.md)).

## Connect your AI agent

NeuroLithe talks to clients over **STDIO** using the Model Context Protocol (MCP).
Paste the snippet from `neurolithe init` into your client config (Claude Desktop,
Cursor, Claude Code, …):

```json
{
  "mcpServers": {
    "neurolithe": {
      "command": "/Users/you/.cargo/bin/neurolithe",
      "args": ["mcp"]
    }
  }
}
```

Add `"--workspace", "<name>"` to `args` to give a client its own, separate memory.

## Optional: a chat model

Fact extraction and summaries need a chat model. Set the provider in
`~/.neurolithe/neurolithe.toml` (see [Configuration](./configuration.md)) and its
API key in `~/.neurolithe/.env`:

```bash
OPENAI_API_KEY=sk-...
```

## Try it out

Once connected, your agent can use tools such as:

1. **Store a fact:** `store_memory({fact_text: "User prefers dark mode", tags: ["preference"]})`
2. **Query memory:** `query_memory({query: "What does the user prefer?"})`
3. **Push dialogue:** `push_dialogue({session_id: "chat-1", new_message: "I just moved to Berlin"})`
