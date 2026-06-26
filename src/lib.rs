//! Desktop-first LLM gateway with N+M architecture.
//!
//! ferryllm translates client protocol requests into a shared internal
//! representation, routes them by model name, and translates them into the
//! selected provider protocol. The crate can be embedded as a Rust library or
//! run as a standalone HTTP server through the `ferryllm` binary.
//!
//! ## Architecture
//!
//! ```text
//! Client Request (OpenAI/Anthropic/Responses)
//!     ↓
//! Entry Adapter (client → IR)
//!     ↓
//! Router (resolve model → adapter)
//!     ↓
//! Backend Adapter (IR → provider)
//!     ↓
//! Provider API (OpenAI/Anthropic/Gemini)
//! ```
//!
//! Main modules:
//!
//! - [`ir`]: shared request, response, content block, tool, and streaming types.
//! - [`entry`]: client protocol translators.
//! - [`adapters`]: backend provider adapters.
//! - [`router`]: model routing and rewrite rules.
//! - [`gateway`]: health registry and circuit breaker (enterprise features).
//! - [`config`]: TOML configuration support.
//! - [`server`]: Axum HTTP server when the `http` feature is enabled.

pub mod adapter;
pub mod adapters;
pub mod circuit;
pub mod config;
pub mod entry;
pub mod health;
pub mod ir;
pub mod router;
pub mod token_observability;

#[cfg(feature = "http")]
pub mod server;

// Re-export gateway types for convenience
pub use circuit::{BreakerConfig, CircuitBreaker, CircuitMetrics, CircuitState};
pub use health::{HealthConfig, HealthRegistry, HealthStatus, TargetHealth};
