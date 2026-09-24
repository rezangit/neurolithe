//! `neurolithe.example.toml` (what `neurolithe init` writes) must stay a valid,
//! loadable, generic config.
mod common;

use common::{FakeLlm, Home};
use serde_json::json;

#[test]
fn example_config_starts_the_mcp_server() {
    let llm = FakeLlm::start();
    let home = Home::empty();
    let example = concat!(env!("CARGO_MANIFEST_DIR"), "/neurolithe.example.toml");
    std::fs::copy(example, home.path().join("neurolithe.toml")).expect("copy example config");

    // Keep the test offline whatever the example configures (the default
    // embedder is `local`, which downloads a model): point chat + embeddings
    // at the fake via env overrides (env wins over the file).
    let url = llm.base_url();
    let mut server = home.spawn_mcp(
        &[],
        &[
            ("NEUROLITHE__LLM__PROVIDER", "openai"),
            ("NEUROLITHE__LLM__MODEL", "fake-chat"),
            ("NEUROLITHE__LLM__BASE_URL", url.as_str()),
            ("NEUROLITHE__LLM__EMBEDDING_PROVIDER", "openai"),
            ("NEUROLITHE__LLM__EMBEDDING_MODEL", "fake-embed"),
            ("NEUROLITHE__LLM__EMBEDDING_BASE_URL", url.as_str()),
        ],
    );
    let init = server.initialize();
    assert_eq!(
        init["serverInfo"]["name"],
        "NeuroLithe",
        "stderr: {}",
        server.stderr()
    );
    let res = server.call_tool("health", json!({}));
    assert!(!res.is_error, "{}", res.text);
    let res = server.call_tool("store_memory", json!({"fact_text": "example works"}));
    assert!(!res.is_error, "{} / stderr: {}", res.text, server.stderr());

    // Stores live in the default workspace under the home, not the CWD.
    let ws = home.workspace_dir("default");
    assert!(ws.join("stm.sqlite").exists());
    assert!(ws.join("ltm.sqlite").exists());
    assert_eq!(std::fs::read_dir(home.cwd.path()).unwrap().count(), 0);
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
    // Phase 2: store paths derive from the workspace; the example must not
    // carry per-store paths any more.
    assert!(
        !text.lines().any(|l| l.trim_start().starts_with("path =")),
        "example still sets a store path"
    );
}
