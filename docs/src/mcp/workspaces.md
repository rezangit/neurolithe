# Workspace tools

A **workspace** is a completely separate memory: its own STM and LTM stores in
`<home>/workspaces/<name>/`. Each server process has one *active* workspace,
chosen at startup by `--workspace`, `NEUROLITHE_WORKSPACE`, config `workspace`,
or `default`. All memory tools read and write the active workspace. There is no
per-call tenant.

Workspace names match `^[a-z0-9][a-z0-9_-]{0,63}$`.

The same operations exist on the CLI:
`neurolithe workspace list | create | delete --yes | export | backup | import`.

## workspace_current (read-only)

No input. Returns the active workspace:

```json
{ "name": "default", "path": "/Users/you/.neurolithe/workspaces/default",
  "stm_bytes": 204800, "ltm_bytes": 409600, "active": true }
```

## workspace_list (read-only)

No input. Returns every workspace in the same shape, sorted by name. The active
one has `"active": true`.

## workspace_create

| Field | Type | Required |
|---|---|---|
| `name` | string | yes |

Creates an empty workspace and returns its info. It does **not** switch to it.

## workspace_switch

| Field | Type | Required |
|---|---|---|
| `name` | string | yes |

Makes an existing workspace active: its stores are opened and the session
buffers start empty. Returns the new active workspace's info. The operator can
disable it with `[mcp] allow_workspace_switch = false`; the call then returns an
`isError` result. In Kafka mode a switch only moves the MCP session, and the
Kafka loops stay on the startup workspace.

## workspace_export (read-only)

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | no | Defaults to the active workspace |

Returns a JSON dump:

```json
{
  "workspace": "default",
  "stm_facts": [ { "payload": {"fact": "…", "tags": []}, "ccl": "reality", "status": "active",
                   "relevance_score": 0.8, "support_count": 1, "created_at": "…" } ],
  "ltm_leaves": [ { "data_id": "doc_42", "title": "…", "summary": "…", "provenance": {…} } ]
}
```

## workspace_delete (destructive)

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | |
| `confirm` | string | yes | Must equal `name` exactly |

Permanently deletes the workspace directory and all of its memory. The active
workspace cannot be deleted; switch away first. The tool is annotated
`destructiveHint: true`.
