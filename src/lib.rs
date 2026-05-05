//! Universal LLM protocol middleware for OpenAI, Anthropic, Claude Code, and
//! OpenAI-compatible backends.
//!
//! ferryllm translates client protocol requests into a shared internal
//! representation, routes them by model name, and translates them into the
//! selected provider protocol. The crate can be embedded as a Rust library or
//! run as a standalone HTTP server through the `ferryllm` binary.
//!
//! Main modules:
//!
//! - [`ir`]: shared request, response, content block, tool, and streaming types.
//! - [`entry`]: client protocol translators.
//! - [`adapters`]: backend provider adapters.
//! - [`router`]: model routing and rewrite rules.
//! - [`config`]: TOML configuration support.
//! - [`server`]: Axum HTTP server when the `http` feature is enabled.

pub mod adapter;
pub mod adapters;
pub mod config;
pub mod entry;
pub mod ir;
pub mod router;
pub mod token_observability;

#[cfg(feature = "http")]
pub mod server;
