//! Bus memory query API — wire envelopes for `memory.query` / `memory.result`.
//!
//! This is the delivery-layer contract for agents reading NeuroLithe over Kafka.
//! It defines:
//!
//! - [`MemoryQuery`] — the inbound request envelope (lenient camelCase deserialize).
//! - [`MemoryReply`] — the outbound labelled result (STM token-optimized, LTM
//!   reference-returning).
//! - [`parse_query`] — a three-way routing parser: a good request, a *rejectable*
//!   request (has a `correlationId` we can reply an error to), or an *unroutable*
//!   one (no `correlationId` → the parking topic).
//!
//! The mapping helpers turn the existing application types ([`MemoryResult`],
//! [`RecallResult`]) into wire entries without leaking STM internal ids.

use crate::application::ltm_retrieval::RecallResult;
use crate::application::query_service::QueryScope;
use crate::domain::ltm::Provenance;
use crate::domain::models::{MemoryResult, TimeFilter};
use serde::{Deserialize, Serialize};

/// Default recall breadth when the request omits `k`.
pub const DEFAULT_K: usize = 5;

fn default_k() -> usize {
    DEFAULT_K
}

/// Inbound `memory.query` envelope. Lenient: unknown fields are ignored and
/// omitted optionals fall back to the documented defaults. A legacy `tenant`
/// field is accepted and ignored — one process serves one workspace.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryQuery {
    pub correlation_id: String,
    pub scope: QueryScope,
    /// Query text. Optional for `stm_map` (recency needs no text); defaults to
    /// empty. Required in practice for the semantic scopes — the service treats
    /// an empty query as "no semantic search".
    #[serde(default)]
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub time_filter: Option<TimeFilter>,
    #[serde(default)]
    pub ccl: Vec<String>,
    /// Working-memory thread key for `stm_map`.
    #[serde(default)]
    pub context_key: Option<String>,
}

/// The outcome of parsing raw `memory.query` bytes — drives the loop's routing:
/// reply, error-reply, or park.
#[derive(Debug)]
pub enum ParseOutcome {
    /// A well-formed request. Dispatch it and reply.
    Query(Box<MemoryQuery>),
    /// Has a `correlationId` but is otherwise malformed (bad scope, missing
    /// `query`, …). We can and must reply `status:"error"` to this id.
    Rejected {
        correlation_id: String,
        reason: String,
    },
    /// No usable `correlationId` (not JSON, or the field is absent/non-string).
    /// There is no one to reply to → `parking.lot`.
    Unroutable { reason: String },
}

/// Parse raw request bytes into a routing outcome.
///
/// Two-stage on purpose: first fish out `correlationId` from loose JSON so a
/// malformed-but-addressable request still yields an error *reply* rather than a
/// silent park; only a request with no reply address becomes `Unroutable`.
pub fn parse_query(bytes: &[u8]) -> ParseOutcome {
    let value: serde_json::Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => {
            return ParseOutcome::Unroutable {
                reason: format!("not valid JSON: {e}"),
            };
        }
    };

    let correlation_id = match value.get("correlationId").and_then(|v| v.as_str()) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            return ParseOutcome::Unroutable {
                reason: "missing or non-string correlationId".to_string(),
            };
        }
    };

    match serde_json::from_value::<MemoryQuery>(value) {
        Ok(q) => ParseOutcome::Query(Box::new(q)),
        Err(e) => ParseOutcome::Rejected {
            correlation_id,
            reason: format!("invalid memory.query: {e}"),
        },
    }
}

/// Result status on `memory.result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyStatus {
    Ok,
    Empty,
    Error,
}

/// A 1-hop connection carried alongside an STM fact (relation → entity only;
/// the rest of `MemoryConnection` stays internal).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StmConnection {
    pub relation: String,
    pub entity: String,
}

/// An STM fact on the wire — token-optimized, no internal ids.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StmEntry {
    pub fact: String,
    pub ccl: String,
    pub last_updated: String,
    pub connections: Vec<StmConnection>,
    /// Archive reference (`dataId`) this fact came from, when present — so a bus
    /// consumer can trace/fetch the source document straight from a
    /// search hit. Omitted on the wire when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_id: Option<String>,
    /// Working-memory thread this note belongs to. Present only for `stm_map`
    /// entries (situational notes); omitted for ordinary STM facts — so old
    /// consumers and non-working replies are unchanged on the wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_key: Option<String>,
}

impl StmEntry {
    pub fn from_memory_result(m: &MemoryResult) -> Self {
        Self {
            fact: m.fact.clone(),
            ccl: m.ccl.clone(),
            last_updated: m.last_updated.clone(),
            connections: m
                .connections
                .iter()
                .map(|c| StmConnection {
                    relation: c.relation.clone(),
                    entity: c.entity.clone(),
                })
                .collect(),
            data_id: m.data_id.clone(),
            context_key: m.context_key.clone(),
        }
    }
}

/// An LTM concept+document on the wire — reference-returning: the
/// located concept, the document `dataId` + `provenance` beneath it, distance,
/// and the ancestor concept names for context.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LtmEntry {
    pub concept: String,
    pub summary: String,
    pub data_id: Option<String>,
    pub provenance: Option<Provenance>,
    pub distance: f64,
    pub ancestors: Vec<String>,
}

/// Flatten one `RecallResult` into one wire entry per document leaf. A concept
/// with no leaves still surfaces once (with a null `dataId`) so the recall isn't
/// silently dropped.
pub fn flatten_recall(r: &RecallResult) -> Vec<LtmEntry> {
    let ancestors: Vec<String> = r.ancestors.iter().map(|n| n.name.clone()).collect();
    let concept = r.entry.name.clone();
    let summary = r.entry.summary.clone();

    if r.leaves.is_empty() {
        return vec![LtmEntry {
            concept,
            summary,
            data_id: None,
            provenance: None,
            distance: r.distance,
            ancestors,
        }];
    }

    r.leaves
        .iter()
        .map(|leaf| LtmEntry {
            concept: concept.clone(),
            summary: summary.clone(),
            data_id: Some(leaf.data_id.clone()),
            provenance: Some(leaf.provenance.clone()),
            distance: r.distance,
            ancestors: ancestors.clone(),
        })
        .collect()
}

/// Outbound `memory.result` envelope.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryReply {
    pub correlation_id: String,
    pub status: ReplyStatus,
    pub stm: Vec<StmEntry>,
    pub ltm: Vec<LtmEntry>,
    pub seeded_by: Vec<String>,
    pub error: Option<String>,
}

impl MemoryReply {
    /// A successful reply. `status` is `empty` when both sections are empty,
    /// else `ok`.
    pub fn success(
        correlation_id: impl Into<String>,
        stm: Vec<StmEntry>,
        ltm: Vec<LtmEntry>,
        seeded_by: Vec<String>,
    ) -> Self {
        let status = if stm.is_empty() && ltm.is_empty() {
            ReplyStatus::Empty
        } else {
            ReplyStatus::Ok
        };
        Self {
            correlation_id: correlation_id.into(),
            status,
            stm,
            ltm,
            seeded_by,
            error: None,
        }
    }

    /// An error reply — always emitted rather than hanging.
    pub fn error(correlation_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            correlation_id: correlation_id.into(),
            status: ReplyStatus::Error,
            stm: Vec::new(),
            ltm: Vec::new(),
            seeded_by: Vec::new(),
            error: Some(message.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::ltm_retrieval::{LeafRef, NodeView};
    use crate::domain::ltm::TreeNodeKind;
    use crate::domain::models::MemoryConnection;
    use serde_json::json;

    fn node(name: &str) -> NodeView {
        NodeView {
            id: 1,
            name: name.to_string(),
            summary: format!("summary of {name}"),
            kind: TreeNodeKind::Grown,
        }
    }

    #[test]
    fn parses_each_scope() {
        for (raw, want) in [
            ("stm", QueryScope::Stm),
            ("ltm", QueryScope::Ltm),
            ("both", QueryScope::Both),
            ("ltm_via_stm", QueryScope::LtmViaStm),
        ] {
            let bytes =
                format!(r#"{{"correlationId":"c1","scope":"{raw}","query":"hi"}}"#).into_bytes();
            match parse_query(&bytes) {
                ParseOutcome::Query(q) => assert_eq!(q.scope, want),
                other => panic!("expected Query for {raw}, got {other:?}"),
            }
        }
    }

    #[test]
    fn applies_defaults_when_omitted() {
        let bytes = br#"{"correlationId":"c1","scope":"stm","query":"hi"}"#;
        let ParseOutcome::Query(q) = parse_query(bytes) else {
            panic!("expected Query");
        };
        assert_eq!(q.k, 5);
        assert!(q.time_filter.is_none());
        assert!(q.ccl.is_empty());
    }

    /// A legacy `tenant` field still parses (it is ignored: one workspace per
    /// process).
    #[test]
    fn legacy_tenant_field_is_accepted_and_ignored() {
        let bytes = br#"{"correlationId":"c1","tenant":"someone-else","scope":"stm","query":"hi"}"#;
        assert!(matches!(parse_query(bytes), ParseOutcome::Query(_)));
    }

    #[test]
    fn stm_map_parses_without_query_and_carries_context_key() {
        // Recency needs no query text; contextKey scopes the thread.
        let bytes = br#"{"correlationId":"c1","scope":"stm_map","contextKey":"chat.jid:42"}"#;
        let ParseOutcome::Query(q) = parse_query(bytes) else {
            panic!("expected Query");
        };
        assert_eq!(q.scope, QueryScope::StmMap);
        assert_eq!(q.query, "", "query defaults empty for stm_map");
        assert_eq!(q.context_key.as_deref(), Some("chat.jid:42"));
    }

    #[test]
    fn old_envelope_without_new_fields_still_parses() {
        // A pre-working-memory request (no contextKey) keeps its defaults.
        let bytes = br#"{"correlationId":"c1","scope":"stm","query":"hi"}"#;
        let ParseOutcome::Query(q) = parse_query(bytes) else {
            panic!("expected Query");
        };
        assert!(q.context_key.is_none());
    }

    #[test]
    fn stm_entry_context_key_serializes_only_when_present() {
        let base = StmEntry {
            fact: "found report = doc_1".into(),
            ccl: "working".into(),
            last_updated: "2026-07-02T00:00:00Z".into(),
            connections: vec![],
            data_id: None,
            context_key: Some("chat.jid:42".into()),
        };
        let v = serde_json::to_value(&base).unwrap();
        assert_eq!(v["contextKey"], "chat.jid:42");

        let without = StmEntry {
            context_key: None,
            ..base
        };
        let v = serde_json::to_value(&without).unwrap();
        assert!(
            v.get("contextKey").is_none(),
            "omitted on the wire when absent"
        );
    }

    #[test]
    fn ignores_unknown_fields() {
        let bytes = br#"{"correlationId":"c1","scope":"stm","query":"hi","surprise":42}"#;
        assert!(matches!(parse_query(bytes), ParseOutcome::Query(_)));
    }

    #[test]
    fn missing_correlation_id_is_unroutable() {
        let bytes = br#"{"scope":"stm","query":"hi"}"#;
        assert!(matches!(
            parse_query(bytes),
            ParseOutcome::Unroutable { .. }
        ));
    }

    #[test]
    fn non_json_is_unroutable() {
        assert!(matches!(
            parse_query(b"not json at all"),
            ParseOutcome::Unroutable { .. }
        ));
    }

    #[test]
    fn addressable_but_malformed_is_rejected() {
        // Has correlationId but an invalid scope — we can reply an error to it.
        let bytes = br#"{"correlationId":"c9","scope":"telepathy","query":"hi"}"#;
        match parse_query(bytes) {
            ParseOutcome::Rejected { correlation_id, .. } => assert_eq!(correlation_id, "c9"),
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn stm_entry_drops_internal_connection_fields() {
        let m = MemoryResult {
            fact: "the user's dentist is Dr. Lee".to_string(),
            ccl: "reality".to_string(),
            last_updated: "2026-06-30T00:00:00Z".to_string(),
            connections: vec![MemoryConnection {
                relation: "at".to_string(),
                entity: "Kerrisdale".to_string(),
                ccl: "reality".to_string(),
                valid_from: Some("2026-01-01".to_string()),
                valid_until: None,
            }],
            data_id: Some("doc_dentist".to_string()),
            context_key: None,
        };
        let entry = StmEntry::from_memory_result(&m);
        let v = serde_json::to_value(&entry).unwrap();
        assert_eq!(
            v["connections"][0],
            json!({"relation":"at","entity":"Kerrisdale"})
        );
        assert!(v["connections"][0].get("ccl").is_none());
    }

    #[test]
    fn flatten_recall_one_entry_per_leaf() {
        let r = RecallResult {
            entry: node("Health / Dental"),
            ancestors: vec![node("Health")],
            children: vec![],
            leaves: vec![
                LeafRef {
                    data_id: "doc_1".to_string(),
                    provenance: Provenance {
                        source: "scan".to_string(),
                        ingested_at: None,
                        confidence: 0.9,
                    },
                },
                LeafRef {
                    data_id: "doc_2".to_string(),
                    provenance: Provenance {
                        source: "scan".to_string(),
                        ingested_at: None,
                        confidence: 0.8,
                    },
                },
            ],
            distance: 0.12,
        };
        let entries = flatten_recall(&r);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].data_id.as_deref(), Some("doc_1"));
        assert_eq!(entries[0].ancestors, vec!["Health".to_string()]);
        assert_eq!(entries[1].data_id.as_deref(), Some("doc_2"));
    }

    #[test]
    fn flatten_recall_leafless_concept_still_surfaces() {
        let r = RecallResult {
            entry: node("Finance"),
            ancestors: vec![],
            children: vec![],
            leaves: vec![],
            distance: 0.3,
        };
        let entries = flatten_recall(&r);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].data_id.is_none());
        assert!(entries[0].provenance.is_none());
    }

    #[test]
    fn reply_status_is_empty_when_both_sections_empty() {
        let reply = MemoryReply::success("c1", vec![], vec![], vec![]);
        assert_eq!(reply.status, ReplyStatus::Empty);
    }

    #[test]
    fn reply_serialization_matches_design_section_6() {
        let stm = vec![StmEntry {
            fact: "the user's dentist is Dr. Lee".to_string(),
            ccl: "reality".to_string(),
            last_updated: "2026-06-30T00:00:00Z".to_string(),
            connections: vec![StmConnection {
                relation: "at".to_string(),
                entity: "Kerrisdale".to_string(),
            }],
            data_id: None,
            context_key: None,
        }];
        let ltm = vec![LtmEntry {
            concept: "Health / Dental".to_string(),
            summary: "dental care".to_string(),
            data_id: Some("doc_123".to_string()),
            provenance: Some(Provenance {
                source: "scan".to_string(),
                ingested_at: None,
                confidence: 0.9,
            }),
            distance: 0.12,
            ancestors: vec!["Health".to_string()],
        }];
        let reply = MemoryReply::success(
            "abc",
            stm,
            ltm,
            vec!["the user's dentist is Dr. Lee".into()],
        );
        let v = serde_json::to_value(&reply).unwrap();

        assert_eq!(v["correlationId"], "abc");
        assert_eq!(v["status"], "ok");
        assert_eq!(v["stm"][0]["lastUpdated"], "2026-06-30T00:00:00Z");
        assert_eq!(v["ltm"][0]["dataId"], "doc_123");
        assert_eq!(v["ltm"][0]["concept"], "Health / Dental");
        assert_eq!(v["seededBy"][0], "the user's dentist is Dr. Lee");
        assert_eq!(v["error"], serde_json::Value::Null);
    }

    #[test]
    fn error_reply_carries_message_and_status() {
        let reply = MemoryReply::error("c1", "boom");
        let v = serde_json::to_value(&reply).unwrap();
        assert_eq!(v["status"], "error");
        assert_eq!(v["error"], "boom");
    }
}
