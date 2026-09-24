pub mod config;
pub mod database;
pub mod llm;
// tracing subscriber (stderr only; stdout is the MCP transport).
pub mod logging;
// Offline fastembed/ONNX embedder — only with the `local-embeddings` feature.
#[cfg(feature = "local-embeddings")]
pub mod local_embed;
pub mod ltm_repository;
// Shared rdkafka client settings (passthrough `[kafka.client]`) — `kafka` only.
#[cfg(feature = "kafka")]
pub mod kafka_client;
// Kafka metrics-snapshot producer — only with the `kafka` feature.
#[cfg(feature = "kafka")]
pub mod metrics_publisher;
pub mod repository;
pub mod schema;
