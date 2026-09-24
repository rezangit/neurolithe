# Long-Term Memory

Long-term memory is stored in an embedded SQLite database using a hybrid Graph-Vector architecture.

> Isolation is per **workspace**: each workspace has its own store files in
> `<home>/workspaces/<name>/`. The `tenant_id` columns shown below are a legacy
> of the old multi-tenant design. They always hold `default` and are not exposed
> through any tool. A full V2 architecture rewrite of these pages is planned.

## Storage Layers

### Episodes (Ground Truth)

Raw conversation logs, permanently archived:

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    raw_dialogue TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Nodes (Facts)

Structured knowledge with cognitive attributes:

```sql
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    source_episode_id INTEGER,
    payload JSON NOT NULL,
    status TEXT DEFAULT 'active',       -- 'active', 'superseded', 'archived'
    is_explicit BOOLEAN DEFAULT 0,
    support_count INTEGER DEFAULT 1,
    relevance_score REAL DEFAULT 1.0,   -- Decays over time
    last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Edges (Relationships)

Temporal relationships between nodes:

```sql
CREATE TABLE edges (
    source_id INTEGER,
    target_id INTEGER,
    relation TEXT NOT NULL,
    valid_from DATETIME,    -- When did this become true?
    valid_until DATETIME,   -- When did this stop being true?
    weight REAL DEFAULT 1.0
);
```

### Hybrid Indices

- **`vec_nodes`** — sqlite-vec virtual table for semantic vector search
- **`fts_nodes`** — FTS5 virtual table for keyword search (auto-synced via triggers)

## Retrieval Query

Retrieval uses a combined score of FTS5 (BM25) and Vector Cosine Distance, followed by a **1-hop graph traversal** that respects temporal boundaries. Reading a node automatically boosts its `relevance_score` back to 1.0.
