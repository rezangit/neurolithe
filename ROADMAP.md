# NeuroLithe — Roadmap

Recommended features, captured from real-world MCP testing so they can be picked
up later. Ordered roughly by value. None are blocking; search + placement work
today.

## 1. Reference resolver (memory → original document)

Retrieval is **reference-returning**: `recall_ltm` / `query_memory` hand back a
`dataId` + provenance, but nothing in this crate can turn a `dataId` back into
the original bytes. Close the loop with a fetch surface so an agent can go
`search → dataId → read the source`.

- An optional `fetch` capability (e.g. read the artifact from a configured
  store / local path by `dataId`), or a companion document-store MCP server
  that owns the originals. Bonus: have `trace_dataId` include a ready-made
  fetch URI.

## 2. Sharper concept placement

Placement is best-match-or-inbox against concept vectors derived from short
category labels. Two rough edges observed:

- **Coarse concept vectors** — a few branches act as "magnets" and attract
  loosely-related docs. Improve by embedding **richer concept descriptions**
  (representative examples/keywords per branch), or by averaging the branch's
  filed leaves into a centroid the concept vector tracks.
- **Configurable threshold** — the placement distance cutoff is a tuned constant
  (`DEFAULT_MAX_DISTANCE`, calibrated with the `placement_debug` tool). Promote
  it to config (`[ltm] placement_max_distance`) so it can be retuned per corpus
  without a recompile.

## 3. Smart tree growth (AI-grown concepts)

V2 only does best-match-or-inbox; it never grows the tree. Add growth rules:
split a fat node, merge similar branches, and **spawn new concepts**
from clusters of inbox documents (reusing the conflict-resolver's
assimilate/modify/create idea at the branch level). This is the real fix for the
inbox tail and for concept coarseness.

## 4. Document-subject graph links (cluster recall)

`query_memory` hits come back with empty `connections` because feeder-ingested
document facts have no `about`/subject edges — those are only built on the
`push_dialogue` path. Link document facts by subject (e.g. all records about the
same entity) so **one hit surfaces the whole cluster** via 1-hop expansion. The
LTM tree is the natural home once docs file under shared concepts; an STM
subject-graph is the lighter alternative.

## 5. Title/heading-boosted ranking

Ranking is by hybrid vector+keyword score (direct matches first). A further
refinement: boost matches that hit a document's **title / first heading**, so the
canonical document outranks incidental mentions of the same terms.

## 6. STM ↔ LTM hydration

When an agent focuses a branch, optionally pre-load that branch's summaries into
STM so fast recall covers the current topic (`hydrate(branch)`), with
a smart policy for what to pull and when.
