use crate::domain::ports::{ExtractedFact, LlmClient};
use crate::infrastructure::config::{LlmConfig, LlmProvider};
use anyhow::{Result, anyhow};
use reqwest::{Client, RequestBuilder};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

// ==========================================
// HTTP plumbing shared by every provider
// ==========================================

/// TCP/TLS connect timeout for provider calls. Short: an unreachable host should
/// fail fast rather than hold the serial MCP loop (DEV-5 / SEC-11).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Upstream error bodies are truncated to this many characters before they are
/// wrapped into an error that may reach an MCP client or a log (SEC-09).
const MAX_ERROR_BODY_CHARS: usize = 300;

/// Build the one HTTP client every provider shares (connection pool + TLS
/// session reuse), with a connect timeout and a total per-request timeout so a
/// hung provider can never freeze the process (DEV-5 / SEC-11).
pub fn build_http_client(request_timeout: Duration) -> Client {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(request_timeout)
        .build()
        // Builder only fails on TLS backend init; fall back to defaults rather
        // than abort (the timeouts are then enforced by the caller's budget).
        .unwrap_or_else(|_| Client::new())
}

/// Truncate an upstream body to [`MAX_ERROR_BODY_CHARS`], collapsing whitespace
/// so a multi-line HTML error page stays a one-liner.
fn truncate_for_error(body: &str) -> String {
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX_ERROR_BODY_CHARS {
        flat
    } else {
        let head: String = flat.chars().take(MAX_ERROR_BODY_CHARS).collect();
        format!("{head}… [truncated]")
    }
}

/// A transport error with the request URL stripped — reqwest's `Display`
/// includes the URL, which for some providers carried credentials and in all
/// cases leaks endpoint details to the MCP client (SEC-08 / SEC-09).
fn transport_error(provider: &str, e: reqwest::Error) -> anyhow::Error {
    let kind = if e.is_timeout() {
        " (timed out)"
    } else if e.is_connect() {
        " (connection failed)"
    } else {
        ""
    };
    anyhow!("{provider} request failed{kind}: {}", e.without_url())
}

/// Send a request and return the JSON body, mapping every failure to a
/// sanitized error: URL stripped, status included, body truncated.
async fn send_json(provider: &str, request: RequestBuilder) -> Result<serde_json::Value> {
    let resp = request
        .send()
        .await
        .map_err(|e| transport_error(provider, e))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "{provider} API error ({status}): {}",
            truncate_for_error(&body)
        ));
    }
    resp.json().await.map_err(|e| transport_error(provider, e))
}

// ==========================================
// API keys
// ==========================================

/// Environment variables consulted (in order) for a provider's API key.
///
/// `custom` deliberately reads **only** `NEUROLITHE_API_KEY`: its `base_url`
/// can point anywhere (OpenRouter, a LAN box, plain HTTP), so it must never
/// pick up — and forward — a real `OPENAI_API_KEY` (SEC-09). Vertex uses
/// service-account auth, not a key.
pub fn api_key_env_vars(provider: &LlmProvider) -> &'static [&'static str] {
    match provider {
        LlmProvider::Openai => &["OPENAI_API_KEY", "NEUROLITHE_API_KEY"],
        LlmProvider::Gemini => &["GEMINI_API_KEY", "NEUROLITHE_API_KEY"],
        LlmProvider::Anthropic => &["ANTHROPIC_API_KEY", "NEUROLITHE_API_KEY"],
        LlmProvider::Custom => &["NEUROLITHE_API_KEY"],
        LlmProvider::Vertex => &[],
    }
}

/// Whether a provider cannot work without an API key. `custom` endpoints are
/// often local (Ollama, LM Studio) and keyless; Vertex authenticates via
/// `GOOGLE_APPLICATION_CREDENTIALS`.
fn key_required(provider: &LlmProvider) -> bool {
    matches!(
        provider,
        LlmProvider::Openai | LlmProvider::Gemini | LlmProvider::Anthropic
    )
}

/// Resolve a provider's API key via `lookup` (the process env in production,
/// a map in tests). Empty values count as missing.
pub fn resolve_api_key(
    provider: &LlmProvider,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    api_key_env_vars(provider)
        .iter()
        .filter_map(|name| lookup(name))
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

/// Stand-in for a provider half whose API key is missing. Every call fails with
/// a short, actionable message instead of sending a fake key upstream (the old
/// `dummy_key` fallback, QA-10 / DEV-13). Introspection never calls the LLM, so
/// it keeps working.
struct UnconfiguredLlm {
    message: String,
}

impl UnconfiguredLlm {
    fn for_provider(role: &str, provider: &LlmProvider) -> Self {
        let vars = api_key_env_vars(provider).join(" or ");
        Self {
            message: format!("LLM not configured: set {vars} ({role} provider)"),
        }
    }

    fn fail<T>(&self) -> Result<T> {
        Err(anyhow!("{}", self.message))
    }
}

#[async_trait::async_trait]
impl LlmClient for UnconfiguredLlm {
    async fn extract_facts(
        &self,
        _dialogue: &str,
        _valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        self.fail()
    }

    async fn generate_ccl_description(&self, _ccl_name: &str, _context: &str) -> Result<String> {
        self.fail()
    }

    async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
        self.fail()
    }

    async fn compress_context(&self, _messages: &str) -> Result<String> {
        self.fail()
    }
}

// ==========================================
// Factory
// ==========================================

/// The assembled client plus any startup warnings (missing keys, keys sent in
/// clear text) for the caller to log.
pub struct LlmSetup {
    pub client: Arc<dyn LlmClient>,
    pub warnings: Vec<String>,
}

/// Build the LLM client, delegating chat and embeddings to (possibly) different
/// providers. Chat uses `config.provider`/`config.model`/`config.base_url`;
/// embeddings use `config.effective_embedding_provider()` /
/// `config.embedding_model` / `config.embedding_base_url` (falling back to
/// `base_url`). This lets chat run on Claude (which has no embeddings endpoint)
/// while embeddings run on Vertex/Google/OpenAI/local — see `SplitLlmClient`.
///
/// Keys are resolved through `lookup` (pass `|k| std::env::var(k).ok()`). A
/// missing required key does **not** abort startup: that half becomes an
/// [`UnconfiguredLlm`] and a warning is returned. Both halves share one HTTP
/// client with timeouts.
pub fn create_llm_client(config: &LlmConfig, lookup: &dyn Fn(&str) -> Option<String>) -> LlmSetup {
    let http = build_http_client(Duration::from_secs(config.request_timeout_secs));
    let mut warnings = Vec::new();

    let chat_provider = &config.provider;
    let chat_key = resolve_api_key(chat_provider, lookup);
    let chat: Arc<dyn LlmClient> = if key_required(chat_provider) && chat_key.is_none() {
        let stub = UnconfiguredLlm::for_provider("chat", chat_provider);
        warnings.push(format!(
            "{} — fact extraction and compression will fail",
            stub.message
        ));
        Arc::new(stub)
    } else {
        build_chat_client(config, http.clone(), chat_key.clone())
    };

    let embed_provider = config.effective_embedding_provider();
    let embed_key = resolve_api_key(embed_provider, lookup);
    let embed_base_url = config
        .embedding_base_url
        .clone()
        .or_else(|| config.base_url.clone());
    let embed: Arc<dyn LlmClient> = if key_required(embed_provider) && embed_key.is_none() {
        let stub = UnconfiguredLlm::for_provider("embedding", embed_provider);
        warnings.push(format!(
            "{} — storing and searching memory will fail",
            stub.message
        ));
        Arc::new(stub)
    } else {
        build_embed_client(config, http, embed_key.clone(), embed_base_url.clone())
    };

    // SEC-09: a key sent to a non-TLS, non-loopback endpoint travels in clear.
    // Only openai/custom honour a base URL; anthropic/gemini/vertex ignore it,
    // so checking their key against `base_url` would warn about a request that
    // never happens (REV-7).
    let pairs = [
        (chat_provider, &chat_key, &config.base_url),
        (embed_provider, &embed_key, &embed_base_url),
    ];
    for (provider, key, url) in pairs {
        if uses_base_url(provider)
            && let (Some(_), Some(url)) = (key, url)
            && is_plaintext_remote(url)
        {
            warnings.push(format!(
                "an API key will be sent over plain HTTP to {url}; use https"
            ));
        }
    }

    LlmSetup {
        client: Arc::new(SplitLlmClient { chat, embed }),
        warnings,
    }
}

/// Providers whose requests go to the configured `base_url`.
fn uses_base_url(provider: &LlmProvider) -> bool {
    matches!(provider, LlmProvider::Openai | LlmProvider::Custom)
}

/// Plain `http` to a host that is not loopback. Parsed with `reqwest::Url`, so
/// scheme case (`HTTP://`), IPv6 literals (`[::1]`), the whole 127/8 block and
/// `localhost` in any case are all handled (REV-7). An unparseable URL is not
/// judged here (the request itself will fail loudly).
fn is_plaintext_remote(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if parsed.scheme() != "http" {
        return false;
    }
    let Some(host) = parsed.host_str() else {
        return false;
    };
    // IPv6 literals come back bracketed (`[::1]`).
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let local = host.eq_ignore_ascii_case("localhost")
        || bare
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    !local
}

/// Default Vertex AI region when `embedding_location` is unset. `us-central1`
/// serves `text-embedding-004`; the `global` endpoint does not host it.
const DEFAULT_VERTEX_LOCATION: &str = "us-central1";

fn build_chat_client(
    config: &LlmConfig,
    http: Client,
    api_key: Option<String>,
) -> Arc<dyn LlmClient> {
    match config.provider {
        LlmProvider::Openai | LlmProvider::Custom => Arc::new(OpenAiClient::new(
            http,
            api_key,
            config.model.clone(),
            config.embedding_model.clone(),
            config.base_url.clone(),
        )),
        LlmProvider::Gemini => Arc::new(GeminiClient::new(
            http,
            api_key.unwrap_or_default(),
            config.model.clone(),
            config.embedding_model.clone(),
        )),
        LlmProvider::Anthropic => Arc::new(AnthropicClient::new(
            http,
            api_key.unwrap_or_default(),
            config.model.clone(),
        )),
        LlmProvider::Vertex => Arc::new(vertex_client(config, http, config.model.clone())),
    }
}

fn build_embed_client(
    config: &LlmConfig,
    http: Client,
    api_key: Option<String>,
    base_url: Option<String>,
) -> Arc<dyn LlmClient> {
    match config.effective_embedding_provider() {
        LlmProvider::Openai | LlmProvider::Custom => Arc::new(OpenAiClient::new(
            http,
            api_key,
            config.model.clone(),
            config.embedding_model.clone(),
            base_url,
        )),
        LlmProvider::Gemini => Arc::new(GeminiClient::new(
            http,
            api_key.unwrap_or_default(),
            config.model.clone(),
            config.embedding_model.clone(),
        )),
        LlmProvider::Anthropic => Arc::new(AnthropicClient::new(
            http,
            api_key.unwrap_or_default(),
            config.model.clone(),
        )),
        LlmProvider::Vertex => {
            Arc::new(vertex_client(config, http, config.embedding_model.clone()))
        }
    }
}

/// Construct a `VertexClient` from the config's project/location, defaulting the
/// region to `us-central1`.
fn vertex_client(config: &LlmConfig, http: Client, model: String) -> VertexClient {
    VertexClient::new(
        http,
        model,
        config.embedding_project.clone().unwrap_or_default(),
        config
            .embedding_location
            .clone()
            .unwrap_or_else(|| DEFAULT_VERTEX_LOCATION.to_string()),
    )
}

/// Routes chat/reasoning calls to one provider and embeddings to another.
///
/// Required because Claude has no embeddings endpoint: the reasoning half runs
/// on Anthropic while `embed_text` is served by Google/OpenAI/a local model.
/// When both halves resolve to the same provider the two inner clients are
/// simply equivalent, so single-provider setups behave exactly as before.
struct SplitLlmClient {
    chat: Arc<dyn LlmClient>,
    embed: Arc<dyn LlmClient>,
}

#[async_trait::async_trait]
impl LlmClient for SplitLlmClient {
    async fn extract_facts(
        &self,
        dialogue: &str,
        valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        self.chat.extract_facts(dialogue, valid_ccls).await
    }

    async fn generate_ccl_description(&self, ccl_name: &str, context: &str) -> Result<String> {
        self.chat.generate_ccl_description(ccl_name, context).await
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed.embed_text(text).await
    }

    async fn compress_context(&self, messages: &str) -> Result<String> {
        self.chat.compress_context(messages).await
    }
}

// ==========================================
// Prompts (shared by every chat provider)
// ==========================================

fn extraction_prompt(valid_ccls: &[crate::domain::models::CclDefinition]) -> String {
    let ccl_descriptions = valid_ccls
        .iter()
        .map(|c| format!("- '{}' ({})", c.name, c.description))
        .collect::<Vec<_>>()
        .join("\n");

    format!("
            Extract independent factual statements from the user's dialogue.
            Only extract facts that represent long-term knowledge, preferences, or identifiers.
            For each fact, also extract any relationships to other entities with temporal bounds.
            The available Cognitive Context Layers (CCL) are:
{}
            You MUST assign a valid 'ccl' to each fact and relationship based on these definitions.
            Return format: {{\"facts\": [{{\"fact\": \"...\", \"ccl\": \"reality\", \"tags\": [...], \"relationships\": [{{\"target_entity\": \"...\", \"relation\": \"WORKS_AT\", \"ccl\": \"reality\", \"valid_from\": \"YYYY-MM-DD or null\", \"valid_until\": \"YYYY-MM-DD or null\"}}]}}]}}
            If no facts are present, return {{\"facts\": []}}.
            Output ONLY valid JSON.
        ", ccl_descriptions)
}

const CCL_DESCRIPTION_SYSTEM: &str = "You are a helpful assistant. Generate a generic one-line description for the conceptual memory layer requested.";

fn ccl_description_prompt(ccl_name: &str, context: &str) -> String {
    format!(
        "Generate a generic one-line description for the conceptual memory layer '{}' based on the following context:\n{}",
        ccl_name, context
    )
}

const COMPRESS_SYSTEM: &str = "Compress the following conversation into a dense, factual summary. Preserve all key facts, decisions, and context. Remove filler and redundancy. Output only the summary text.";

/// Parse `{"facts": [...]}` out of a model reply.
fn parse_facts(json_text: &str) -> Result<Vec<ExtractedFact>> {
    let parsed: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| anyhow!("model returned invalid JSON for fact extraction: {e}"))?;
    Ok(serde_json::from_value(parsed["facts"].clone())?)
}

// ==========================================
// OpenAI Client (also handles custom URLs)
// ==========================================
pub struct OpenAiClient {
    client: Client,
    /// `None` for keyless local endpoints (Ollama): no Authorization header.
    api_key: Option<String>,
    model: String,
    embedding_model: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(
        client: Client,
        api_key: Option<String>,
        model: String,
        embedding_model: String,
        base_url: Option<String>,
    ) -> Self {
        Self {
            client,
            api_key,
            model,
            embedding_model,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
        }
    }

    fn post(&self, path: &str) -> RequestBuilder {
        let url = format!("{}/{}", self.base_url.trim_end_matches('/'), path);
        let mut rb = self
            .client
            .post(url)
            .header("HTTP-Referer", "https://neurolithe.com")
            .header("X-Title", "NeuroLithe");
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        rb
    }

    async fn chat(&self, payload: serde_json::Value) -> Result<String> {
        let resp_json = send_json("OpenAI", self.post("chat/completions").json(&payload)).await?;
        Ok(resp_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAiClient {
    async fn extract_facts(
        &self,
        dialogue: &str,
        valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        let content = self
            .chat(json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": extraction_prompt(valid_ccls)},
                    {"role": "user", "content": dialogue}
                ],
                "response_format": {"type": "json_object"}
            }))
            .await?;
        let content = if content.trim().is_empty() {
            "{\"facts\": []}"
        } else {
            content.as_str()
        };
        parse_facts(content)
    }

    async fn generate_ccl_description(&self, ccl_name: &str, context: &str) -> Result<String> {
        let content = self
            .chat(json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": CCL_DESCRIPTION_SYSTEM},
                    {"role": "user", "content": ccl_description_prompt(ccl_name, context)}
                ]
            }))
            .await?;
        Ok(content.trim().to_string())
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let payload = json!({
            "model": self.embedding_model,
            "input": text
        });
        let resp_json = send_json("OpenAI", self.post("embeddings").json(&payload)).await?;
        let embedding: Vec<f32> = serde_json::from_value(resp_json["data"][0]["embedding"].clone())
            .map_err(|e| anyhow!("OpenAI embeddings: unexpected response shape: {e}"))?;
        Ok(embedding)
    }

    async fn compress_context(&self, messages: &str) -> Result<String> {
        self.chat(json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPRESS_SYSTEM},
                {"role": "user", "content": messages}
            ]
        }))
        .await
    }
}

// ==========================================
// Google Gemini Client
// ==========================================
const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub struct GeminiClient {
    client: Client,
    api_key: String,
    model: String,
    embedding_model: String,
    base_url: String,
}

impl GeminiClient {
    pub fn new(client: Client, api_key: String, model: String, embedding_model: String) -> Self {
        Self {
            client,
            api_key,
            model,
            embedding_model,
            base_url: GEMINI_BASE_URL.to_string(),
        }
    }

    /// POST `models/{model}:{method}`. The key travels in the
    /// `x-goog-api-key` header — never in the `?key=` query string, where it
    /// leaked into reqwest error text, logs, and MCP replies (DEV-14 / SEC-08).
    fn post(&self, model: &str, method: &str) -> RequestBuilder {
        self.client
            .post(format!("{}/models/{model}:{method}", self.base_url))
            .header("x-goog-api-key", &self.api_key)
    }

    async fn generate(&self, payload: serde_json::Value) -> Result<String> {
        let resp_json = send_json(
            "Gemini",
            self.post(&self.model, "generateContent").json(&payload),
        )
        .await?;
        Ok(resp_json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}

#[async_trait::async_trait]
impl LlmClient for GeminiClient {
    async fn extract_facts(
        &self,
        dialogue: &str,
        valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        let content = self
            .generate(json!({
                "system_instruction": {
                    "parts": [{"text": extraction_prompt(valid_ccls)}]
                },
                "contents": [{
                    "parts": [{"text": dialogue}]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }))
            .await?;
        let content = if content.trim().is_empty() {
            "{\"facts\": []}"
        } else {
            content.as_str()
        };
        parse_facts(content)
    }

    async fn generate_ccl_description(&self, ccl_name: &str, context: &str) -> Result<String> {
        let content = self
            .generate(json!({
                "system_instruction": {
                    "parts": [{"text": CCL_DESCRIPTION_SYSTEM}]
                },
                "contents": [{
                    "parts": [{"text": ccl_description_prompt(ccl_name, context)}]
                }]
            }))
            .await?;
        Ok(content.trim().to_string())
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let payload = json!({
            "model": format!("models/{}", self.embedding_model),
            "content": {
                "parts": [{"text": text}]
            }
        });
        let resp_json = send_json(
            "Gemini",
            self.post(&self.embedding_model, "embedContent")
                .json(&payload),
        )
        .await?;
        let embedding: Vec<f32> = serde_json::from_value(resp_json["embedding"]["values"].clone())
            .map_err(|e| anyhow!("Gemini embeddings: unexpected response shape: {e}"))?;
        Ok(embedding)
    }

    async fn compress_context(&self, messages: &str) -> Result<String> {
        self.generate(json!({
            "system_instruction": {
                "parts": [{"text": COMPRESS_SYSTEM}]
            },
            "contents": [{
                "parts": [{"text": messages}]
            }]
        }))
        .await
    }
}

// ==========================================
// Anthropic Client
// ==========================================

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Return the text of the first `text` content block in an Anthropic Messages
/// response. Claude 4.6+ models (incl. Sonnet 5) can emit a `thinking` block at
/// index 0, so indexing `content[0].text` is unsafe — scan for the text block
/// instead. (We also disable thinking on these requests, but this keeps parsing
/// robust regardless of model/config.)
fn anthropic_first_text(resp_json: &serde_json::Value) -> String {
    resp_json["content"]
        .as_array()
        .and_then(|blocks| blocks.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .unwrap_or("")
        .to_string()
}

/// The outermost `{…}` span of a chatty model reply, if there is a well-ordered
/// one. Replaces an unchecked slice that panicked when a `}` preceded the first
/// `{` (e.g. `"} sorry {"`) — DEV-6.
fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end > start).then(|| &content[start..=end])
}

/// Output-token ceiling for **document summarization** (`compress_context`).
///
/// The old value (1024, shared with fact extraction) truncated long-document
/// summaries mid-word around ~2,500 chars — losing the decision-relevant tail
/// of multi-page reports (field-report §2). A summary is a bounded artifact, so
/// a generous cap (well within Sonnet's output limit) lets it complete; short
/// inputs stop early on their own and cost nothing extra.
const COMPRESS_MAX_TOKENS: u32 = 8192;

/// Output-token ceiling for the small structured calls (fact extraction, CCL
/// descriptions) whose responses are inherently short.
const SHORT_MAX_TOKENS: u32 = 1024;

pub struct AnthropicClient {
    client: Client,
    api_key: String,
    model: String,
}

impl AnthropicClient {
    pub fn new(client: Client, api_key: String, model: String) -> Self {
        Self {
            client,
            api_key,
            model,
        }
    }

    /// One Messages call; thinking disabled so the reply is a single `text`
    /// block (Sonnet 5 otherwise runs adaptive thinking first).
    async fn message(&self, system: &str, user: &str, max_tokens: u32) -> Result<String> {
        let payload = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
            "system": system,
            "messages": [
                {"role": "user", "content": user}
            ]
        });
        let resp_json = send_json(
            "Anthropic",
            self.client
                .post(ANTHROPIC_MESSAGES_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&payload),
        )
        .await?;
        Ok(anthropic_first_text(&resp_json))
    }
}

#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn extract_facts(
        &self,
        dialogue: &str,
        valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        let text = self
            .message(&extraction_prompt(valid_ccls), dialogue, SHORT_MAX_TOKENS)
            .await?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        // Claude may wrap the JSON in prose; take the outermost object.
        let json_text = extract_json_object(&text)
            .ok_or_else(|| anyhow!("Anthropic reply contained no JSON object"))?;
        parse_facts(json_text)
    }

    async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
        Err(anyhow!(
            "Anthropic does not offer a native embedding API. Please use OpenAI/Gemini or an OpenAI-compatible custom endpoint for embeddings."
        ))
    }

    async fn generate_ccl_description(&self, ccl_name: &str, context: &str) -> Result<String> {
        let text = self
            .message(
                CCL_DESCRIPTION_SYSTEM,
                &ccl_description_prompt(ccl_name, context),
                SHORT_MAX_TOKENS,
            )
            .await?;
        Ok(text.trim().to_string())
    }

    async fn compress_context(&self, messages: &str) -> Result<String> {
        self.message(COMPRESS_SYSTEM, messages, COMPRESS_MAX_TOKENS)
            .await
    }
}

// ==========================================
// Google Vertex AI Client (embeddings)
// ==========================================

/// Vertex AI access-token scope (same broad scope Cadmus uses).
const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Google Vertex AI client, used for embeddings so NeuroLithe can share
/// Cadmus's Gemini/Vertex access. Authenticates with a service-account key via
/// `gcp_auth` (reads `GOOGLE_APPLICATION_CREDENTIALS`), minting short-lived
/// bearer tokens — no API key. Chat methods are unimplemented (use
/// anthropic/gemini/custom for reasoning).
pub struct VertexClient {
    client: Client,
    model: String,
    project: String,
    location: String,
    /// Lazily-initialised token provider, cached so we don't re-read the
    /// service-account key on every embed (the token itself is cached by
    /// `gcp_auth` until near expiry).
    token_provider: tokio::sync::OnceCell<Arc<dyn gcp_auth::TokenProvider>>,
}

impl VertexClient {
    pub fn new(client: Client, model: String, project: String, location: String) -> Self {
        Self {
            client,
            model,
            project,
            location,
            token_provider: tokio::sync::OnceCell::new(),
        }
    }

    /// The Vertex `:predict` embeddings endpoint. `global` uses the unprefixed
    /// host; any other location pins data residency to `{loc}-aiplatform...`
    /// (mirrors Cadmus's `generate_content_url`).
    fn predict_url(&self) -> String {
        let host = if self.location == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.location)
        };
        format!(
            "https://{host}/v1/projects/{proj}/locations/{loc}\
             /publishers/google/models/{model}:predict",
            proj = self.project,
            loc = self.location,
            model = self.model,
        )
    }

    /// Mint (or reuse a cached) short-lived Vertex access token from the
    /// service-account key at `GOOGLE_APPLICATION_CREDENTIALS`.
    async fn access_token(&self) -> Result<String> {
        let provider = self
            .token_provider
            .get_or_try_init(|| async {
                gcp_auth::provider().await.map_err(|e| {
                    anyhow!(
                        "Vertex auth init failed — is GOOGLE_APPLICATION_CREDENTIALS a valid \
                         service-account key? {e}"
                    )
                })
            })
            .await?;
        let token = provider
            .token(&[VERTEX_SCOPE])
            .await
            .map_err(|e| anyhow!("fetching Vertex access token: {e}"))?;
        Ok(token.as_str().to_string())
    }
}

#[async_trait::async_trait]
impl LlmClient for VertexClient {
    async fn extract_facts(
        &self,
        _dialogue: &str,
        _valid_ccls: &[crate::domain::models::CclDefinition],
    ) -> Result<Vec<ExtractedFact>> {
        Err(anyhow!(
            "Vertex client is embeddings-only in NeuroLithe; use anthropic/gemini/custom for chat."
        ))
    }

    async fn generate_ccl_description(&self, _ccl_name: &str, _context: &str) -> Result<String> {
        Err(anyhow!(
            "Vertex client is embeddings-only in NeuroLithe; use anthropic/gemini/custom for chat."
        ))
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let token = self.access_token().await?;
        let payload = json!({ "instances": [{ "content": text }] });

        let resp_json = send_json(
            "Vertex",
            self.client
                .post(self.predict_url())
                .bearer_auth(token)
                .json(&payload),
        )
        .await?;
        let values = resp_json["predictions"][0]["embeddings"]["values"].clone();
        if values.is_null() {
            return Err(anyhow!(
                "Vertex embeddings: no values in response (unexpected shape): {}",
                truncate_for_error(&resp_json.to_string())
            ));
        }
        let embedding: Vec<f32> = serde_json::from_value(values)?;
        Ok(embedding)
    }

    async fn compress_context(&self, _messages: &str) -> Result<String> {
        Err(anyhow!(
            "Vertex client is embeddings-only in NeuroLithe; use anthropic/gemini/custom for chat."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn test_config(provider: LlmProvider, embed: Option<LlmProvider>) -> LlmConfig {
        LlmConfig {
            provider,
            model: "m".into(),
            embedding_model: "e".into(),
            base_url: None,
            embedding_provider: embed,
            embedding_base_url: None,
            embedding_project: None,
            embedding_location: None,
            request_timeout_secs: 5,
        }
    }

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    /// One-shot HTTP server: captures the raw request, replies with `status`
    /// and `body`. Returns the base URL and a handle yielding the request text.
    async fn one_shot_server(
        status: u16,
        body: String,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // Read headers, then the declared body.
            loop {
                let n = sock.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(idx) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() >= idx + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&buf).to_string()
        });
        (format!("http://{addr}"), handle)
    }

    /// DEV-6: a reply where `}` precedes the first `{` used to panic on an
    /// out-of-order slice; now it's simply "no JSON object".
    #[test]
    fn test_extract_json_object_never_panics() {
        assert_eq!(extract_json_object("} sorry {"), None);
        assert_eq!(extract_json_object("no braces"), None);
        assert_eq!(extract_json_object("}"), None);
        assert_eq!(
            extract_json_object("Sure! {\"facts\": []} hope that helps"),
            Some("{\"facts\": []}")
        );
    }

    /// QA-10 / DEV-13: a missing key never becomes `dummy_key`; the half is
    /// replaced by a stub that fails with an actionable message, and a startup
    /// warning is produced.
    #[tokio::test]
    async fn test_missing_key_yields_not_configured_error() {
        let setup = create_llm_client(&test_config(LlmProvider::Openai, None), &env(&[]));
        assert!(
            setup
                .warnings
                .iter()
                .any(|w| w.contains("LLM not configured") && w.contains("OPENAI_API_KEY"))
        );
        let err = setup.client.embed_text("hi").await.unwrap_err().to_string();
        assert!(
            err.starts_with("LLM not configured: set OPENAI_API_KEY"),
            "{err}"
        );
        assert!(!err.contains("dummy"));
    }

    /// SEC-09: `custom` must never read `OPENAI_API_KEY` (it may point at any
    /// host); only `NEUROLITHE_API_KEY`, and it is optional (keyless Ollama).
    #[test]
    fn test_custom_provider_never_reads_openai_key() {
        let lookup = env(&[("OPENAI_API_KEY", "sk-real-openai")]);
        assert_eq!(resolve_api_key(&LlmProvider::Custom, &lookup), None);
        let lookup = env(&[
            ("OPENAI_API_KEY", "sk-real-openai"),
            ("NEUROLITHE_API_KEY", "nl-key"),
        ]);
        assert_eq!(
            resolve_api_key(&LlmProvider::Custom, &lookup).as_deref(),
            Some("nl-key")
        );
        // Keyless custom is a valid setup: no warning, no stub.
        let setup = create_llm_client(
            &test_config(LlmProvider::Custom, None),
            &env(&[("OPENAI_API_KEY", "sk-real-openai")]),
        );
        assert!(setup.warnings.is_empty(), "{:?}", setup.warnings);
    }

    /// SEC-09: the keyless custom client sends no Authorization header at all,
    /// even when OPENAI_API_KEY is present in the environment.
    #[tokio::test]
    async fn test_custom_request_carries_no_openai_key() {
        let (url, server) =
            one_shot_server(200, r#"{"data":[{"embedding":[0.5,0.25]}]}"#.into()).await;
        let mut cfg = test_config(LlmProvider::Custom, None);
        cfg.base_url = Some(url);
        let setup = create_llm_client(&cfg, &env(&[("OPENAI_API_KEY", "sk-real-openai")]));
        let v = setup.client.embed_text("hello").await.unwrap();
        assert_eq!(v, vec![0.5, 0.25]);
        let request = server.await.unwrap().to_ascii_lowercase();
        assert!(!request.contains("sk-real-openai"));
        assert!(!request.contains("authorization:"), "{request}");
    }

    /// DEV-14 / SEC-08: the Gemini key goes in the `x-goog-api-key` header,
    /// never the URL; and an upstream error body is truncated (SEC-09).
    #[tokio::test]
    async fn test_gemini_key_in_header_and_error_body_truncated() {
        let huge = format!("{{\"error\":\"{}\"}}", "x".repeat(5000));
        let (url, server) = one_shot_server(500, huge).await;
        let mut client = GeminiClient::new(
            build_http_client(Duration::from_secs(5)),
            "SECRET-GEMINI-KEY".into(),
            "chat".into(),
            "emb".into(),
        );
        client.base_url = url;

        let err = client.embed_text("hi").await.unwrap_err().to_string();
        let request = server.await.unwrap();
        let request_line = request.lines().next().unwrap_or("");
        assert!(
            !request_line.contains("SECRET-GEMINI-KEY"),
            "key leaked into URL: {request_line}"
        );
        assert!(!request_line.contains("key="));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("x-goog-api-key: secret-gemini-key")
        );
        assert!(err.contains("500"), "{err}");
        assert!(
            err.len() < 500,
            "error body not truncated ({} chars)",
            err.len()
        );
        assert!(!err.contains("SECRET-GEMINI-KEY"));
    }

    /// SEC-08: transport errors don't echo the request URL (which can carry
    /// credentials or internal hosts) back to the caller.
    #[tokio::test]
    async fn test_transport_error_strips_url() {
        // Bind then drop, so the port refuses connections.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let client = OpenAiClient::new(
            build_http_client(Duration::from_secs(5)),
            None,
            "m".into(),
            "e".into(),
            Some(format!("http://127.0.0.1:{port}/v1?token=SECRET-IN-URL")),
        );
        let err = client.embed_text("hi").await.unwrap_err().to_string();
        assert!(!err.contains("SECRET-IN-URL"), "{err}");
        assert!(!err.contains("127.0.0.1"), "{err}");
    }

    /// DEV-5: a provider that accepts the connection but never answers must
    /// time out instead of hanging the caller forever.
    #[tokio::test]
    async fn test_hung_provider_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let _hold = tokio::spawn(async move {
            let (_sock, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let client = OpenAiClient::new(
            build_http_client(Duration::from_millis(300)),
            None,
            "m".into(),
            "e".into(),
            Some(format!("http://{addr}")),
        );
        let started = std::time::Instant::now();
        let err = tokio::time::timeout(Duration::from_secs(5), client.embed_text("hi"))
            .await
            .expect("request must time out on its own")
            .unwrap_err()
            .to_string();
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(err.contains("timed out"), "{err}");
    }

    #[test]
    fn test_plaintext_remote_detection() {
        assert!(is_plaintext_remote("http://openrouter.example/api"));
        assert!(!is_plaintext_remote("http://localhost:11434/v1"));
        assert!(!is_plaintext_remote("http://127.0.0.1:11434/v1"));
        assert!(!is_plaintext_remote("https://api.openai.com/v1"));
    }

    /// REV-7: IPv6 loopback, the whole 127/8 block and `localhost` in any case
    /// are local; an uppercase scheme is still plain HTTP.
    #[test]
    fn test_plaintext_remote_parses_urls() {
        assert!(!is_plaintext_remote("http://[::1]:11434/v1"));
        assert!(!is_plaintext_remote("http://127.0.0.2:11434/v1"));
        assert!(!is_plaintext_remote("http://LOCALHOST:11434"));
        assert!(is_plaintext_remote("HTTP://openrouter.example/api"));
        assert!(is_plaintext_remote("http://[2001:db8::1]/v1"));
        assert!(is_plaintext_remote("http://10.0.0.5:8080/v1"));
    }

    /// REV-7: anthropic/gemini ignore `base_url`, so an http base URL must not
    /// trigger a warning for their key; openai with the same URL must.
    #[test]
    fn test_plaintext_warning_only_for_providers_using_base_url() {
        let keys = env(&[
            ("ANTHROPIC_API_KEY", "a"),
            ("OPENAI_API_KEY", "o"),
            ("GEMINI_API_KEY", "g"),
        ]);
        let mut cfg = test_config(LlmProvider::Anthropic, Some(LlmProvider::Gemini));
        cfg.base_url = Some("http://10.0.0.5:8080/v1".into());
        let setup = create_llm_client(&cfg, &keys);
        assert!(
            !setup.warnings.iter().any(|w| w.contains("plain HTTP")),
            "{:?}",
            setup.warnings
        );

        let mut cfg = test_config(LlmProvider::Openai, None);
        cfg.base_url = Some("http://10.0.0.5:8080/v1".into());
        let setup = create_llm_client(&cfg, &keys);
        assert!(
            setup.warnings.iter().any(|w| w.contains("plain HTTP")),
            "{:?}",
            setup.warnings
        );
    }
}
