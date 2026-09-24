pub mod bus_query;
pub mod cli;
// Kafka consumers/producers — only compiled with the `kafka` feature (the
// long-running daemon). The standalone MCP build omits them entirely (no rdkafka).
#[cfg(feature = "kafka")]
pub mod command_consumer;
#[cfg(feature = "kafka")]
pub mod kafka_feeder;
pub mod mcp_server;
pub mod mcp_types;
#[cfg(feature = "kafka")]
pub mod query_consumer;
