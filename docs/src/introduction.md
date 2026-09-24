# NeuroLithe

> A plug-and-play, embedded hybrid Graph-Vector database built in Rust and SQLite that gives AI agents human-like memory with zero external infrastructure.

## What is NeuroLithe?

NeuroLithe is an embedded memory service designed specifically for AI agents. It acts as a **hybrid Graph-Vector database** that mimics human cognition — automatically extracting facts from conversations, building a knowledge graph with temporal relationships, and retrieving relevant context using combined semantic and keyword search.

## Key Features

- **Hybrid Search** — Combines vector embeddings (semantic) with FTS5 (keyword) for precise retrieval
- **Knowledge Graph** — Automatically extracts entities and relationships with temporal bounds
- **Adaptive Forgetting** — Memories decay over time via an exponential function, mimicking human memory
- **Context Compression** — Keeps agent context windows lean by summarizing old dialogue
- **Conflict Resolution** — Tri-modal cognitive adaptation: assimilate, accommodate-modify, or accommodate-create
- **Workspaces** — Separate, independent memories per workspace, selected per process
- **Zero Infrastructure** — Everything runs in a single embedded SQLite database
- **Model Context Protocol** — Exposes tools via MCP JSON-RPC 2.0 over STDIO

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| Storage | rusqlite (bundled SQLite) |
| Vectors | sqlite-vec |
| Full-Text | FTS5 |
| Protocol | MCP over STDIO |
