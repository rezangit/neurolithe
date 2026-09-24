//! `neurolithe.example.toml` must stay a valid, loadable config.
mod common;

use common::McpProcess;
use serde_json::json;

#[test]
fn example_config_starts_the_mcp_server() {
    let dir = tempfile::tempdir().unwrap();
    let example = concat!(env!("CARGO_MANIFEST_DIR"), "/neurolithe.example.toml");
    std::fs::copy(example, dir.path().join("neurolithe.toml")).expect("copy example config");

    let mut server = McpProcess::spawn_with_key(dir.path());
    let init = server.initialize();
    assert_eq!(
        init["serverInfo"]["name"],
        "NeuroLithe",
        "stderr: {}",
        server.stderr()
    );
    // Introspection needs no LLM round-trip, so a fake key is fine.
    let res = server.call_tool("health", json!({}));
    assert!(!res.is_error, "{}", res.text);

    // Stores were created where the example says (relative to the CWD).
    assert!(dir.path().join("neurolithe-stm.sqlite").exists());
    assert!(dir.path().join("neurolithe-ltm.sqlite").exists());
}

#[test]
fn example_config_has_no_private_or_deployment_specific_values() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/neurolithe.example.toml"
    ))
    .unwrap()
    .to_lowercase();
    for needle in [
        "jarvis",
        "pithos",
        "192.168.",
        "cadmus",
        "embedding_project",
    ] {
        assert!(!text.contains(needle), "example config mentions {needle:?}");
    }
}
