# Workspace tools

A **workspace** is a completely separate memory: its own STM and LTM stores in
`<home>/workspaces/<name>/`. Each server process has one *active* workspace,
chosen at startup by `--workspace`, `NEUROLITHE_WORKSPACE`, config `workspace`,
or `default`. All memory tools read and write the active workspace. There is no
per-call tenant.

Workspace names match `^[a-z0-9][a-z0-9_-]{0,63}$`. Arguments, limits and
annotations of each tool are in the [tool reference](./tools.md#workspace-tools).

The same operations exist on the CLI:
`neurolithe workspace list | create | delete --yes | export | backup | import`.

## Pinned servers

With `[mcp] allow_workspace_switch = false` a server is **pinned** to its
startup workspace. It can't switch, and it can't reach any other workspace:
`workspace_switch`, `workspace_create` and `workspace_delete` are not listed in
`tools/list`, and calling them anyway returns an `isError` result explaining
why. `workspace_export` works only for the pinned workspace. Use this to give
each MCP-client entry its own fixed memory.

## workspace_current (read-only)

Returns the active workspace:

```json
{ "name": "default", "path": "/Users/you/.neurolithe/workspaces/default",
  "stm_bytes": 204800, "ltm_bytes": 409600, "active": true }
```

## workspace_list (read-only)

Returns every workspace in the same shape, sorted by name. The active one has
`"active": true`.

## workspace_create

Creates an empty workspace and returns its info. It does **not** switch to it.
Creating a name that already exists is an error.

## workspace_switch

Makes an existing workspace active: its stores are opened and the session
buffers start empty. Returns the new active workspace's info. In Kafka mode a
switch only moves the MCP session; the Kafka loops stay on the startup
workspace.

## workspace_export (read-only)

Returns a JSON dump of the named workspace (default: the active one):

```json
{
  "workspace": "default",
  "stm_facts": [ { "payload": {"fact": "…", "tags": []}, "ccl": "reality", "status": "active",
                   "relevance_score": 0.8, "support_count": 1, "created_at": "…" } ],
  "ltm_leaves": [ { "data_id": "doc_42", "title": "…", "summary": "…",
                   "body": "the document's full text (null if none is kept)", "provenance": {…} } ]
}
```

## workspace_delete (destructive)

Permanently deletes the workspace directory and all of its memory. `confirm`
must equal `name` exactly. The active workspace cannot be deleted; switch away
first. A workspace another process has open can't be deleted either. The tool
is annotated `destructiveHint: true`.
