//! Phase 2 §3: the chat LLM is optional. With `provider = "none"` (and a
//! working embedder), memory is stored and searched, nothing crashes, and the
//! chat-dependent steps report "LLM not configured".
mod common;

use common::{FakeLlm, Home};
use serde_json::json;

fn no_chat_home(llm: &FakeLlm) -> Home {
    let url = llm.base_url();
    Home::with_config(&format!(
        r#"[llm]
provider = "none"
model = "unused"
embedding_provider = "openai"
embedding_model = "fake-embed"
embedding_base_url = "{url}"
"#
    ))
}

#[test]
fn memory_works_without_a_chat_llm() {
    let llm = FakeLlm::start();
    let home = no_chat_home(&llm);
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();

    // push_dialogue archives the turn and reports the missing LLM.
    let res = server.call_tool(
        "push_dialogue",
        json!({"session_id": "s1", "new_message": "Remember my locker is 42"}),
    );
    assert!(!res.is_error, "{}", res.text);
    let ctx = res.json();
    assert_eq!(ctx["learning_error"], "LLM not configured", "{ctx}");
    assert!(
        ctx["recent_messages"]
            .as_array()
            .is_some_and(|m| m.iter().any(|x| x == "Remember my locker is 42")),
        "{ctx}"
    );

    // store_memory stores the fact as given; query finds it.
    let res = server.call_tool("store_memory", json!({"fact_text": "Locker number is 42"}));
    assert!(!res.is_error, "{}", res.text);
    let res = server.call_tool("query_memory", json!({"query": "locker"}));
    assert!(!res.is_error, "{}", res.text);
    assert!(res.text.contains("Locker number is 42"), "{}", res.text);

    // remember_document falls back to an excerpt instead of a summary.
    let res = server.call_tool(
        "remember_document",
        json!({"title": "Gym", "text": "Gym membership card, locker 42."}),
    );
    assert!(!res.is_error, "{}", res.text);
    assert_eq!(res.json()["summarized"], false, "{}", res.text);

    // Introspection is unaffected.
    assert!(!server.call_tool("health", json!({})).is_error);

    // The chat endpoint was never called; embeddings were.
    assert_eq!(llm.count("/chat/completions"), 0);
    assert!(llm.count("/embeddings") > 0);
    let (code, stderr) = server.shutdown();
    assert_eq!(code, Some(0), "{stderr}");
}
