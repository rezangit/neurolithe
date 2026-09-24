use serde::{Deserialize, Serialize};
use serde_json::Value;

// JSON-RPC 2.0 Base Types

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// MCP Specific Payloads

#[derive(Debug, Deserialize)]
pub struct StoreMemoryParams {
    pub fact_text: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct QueryMemoryParams {
    pub query: String,
}

/// A `tools/call` result. The MCP spec names the failure flag `isError`
/// (camelCase); a snake_case `is_error` is ignored by clients, which then treat
/// every tool failure as a success (DEV-1 / QA-4).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    pub is_error: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
}

impl McpToolResult {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text { text: text.into() }],
            is_error: false,
        }
    }

    pub fn err(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text { text: text.into() }],
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DEV-1 / QA-4: the failure flag must serialize as the spec's `isError`.
    #[test]
    fn test_tool_result_serializes_is_error_camel_case() {
        let v = serde_json::to_value(McpToolResult::err("boom")).unwrap();
        assert_eq!(v["isError"], serde_json::json!(true));
        assert!(v.get("is_error").is_none(), "snake_case flag leaked: {v}");
        assert_eq!(v["content"][0]["type"], "text");

        let ok = serde_json::to_value(McpToolResult::ok("fine")).unwrap();
        assert_eq!(ok["isError"], serde_json::json!(false));
    }
}
