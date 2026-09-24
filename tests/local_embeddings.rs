//! Phase 2 §3: the default local embedder (fastembed/ONNX, bge-small-en-v1.5),
//! end to end with the real model. Opt-in: it downloads ~130 MB into
//! `<home>/models` on first run, so it never runs by default.
//!
//! ```sh
//! NL_TARGET=qa scripts/cargo.sh test --test local_embeddings -- --ignored
//! ```
mod common;

use common::Home;
use serde_json::json;

#[test]
#[ignore = "downloads the ~130 MB default embedding model"]
fn default_local_embedder_works_offline_after_download() {
    // Pure defaults: no chat LLM, local embeddings. No config file at all.
    let home = Home::empty();
    let mut server = home.spawn_mcp(&[], &[]);
    // The model loads lazily on the first embed (well within the harness's
    // response timeout on a normal connection).
    server.initialize();

    for fact in [
        "The quarterly budget review is on Friday",
        "My cat is called Miso and loves tuna",
        "The car needs new winter tyres",
    ] {
        let res = server.call_tool("store_memory", json!({"fact_text": fact}));
        assert!(!res.is_error, "{} / {}", res.text, server.stderr());
    }
    // Semantic, not lexical: no shared keyword with the stored fact.
    let res = server.call_tool("query_memory", json!({"query": "pet food", "k": 1}));
    assert!(!res.is_error, "{}", res.text);
    assert!(
        res.text.contains("Miso"),
        "semantic recall failed: {}",
        res.text
    );

    let res = server.call_tool(
        "remember_document",
        json!({"title": "Tyre invoice", "text": "Invoice for four winter tyres, fitted in November."}),
    );
    assert!(!res.is_error, "{}", res.text);
    assert_eq!(res.json()["summarized"], false, "no chat LLM by default");
    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "{stderr}");

    // The model is cached under the home; the stores record a 384-dim identity.
    assert!(
        home.path().join("models").is_dir(),
        "model cache not under <home>/models"
    );
    let db = home.workspace_dir("default").join("stm.sqlite");
    let conn =
        rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let dim: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embedding_dim'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(dim, "384");
}
