use std::pin::Pin;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::Stream;

use crate::ir::{ChatRequest, ChatResponse, StreamEvent, EXTRA_HTTP_USER_AGENT};

pub const DEFAULT_USER_AGENT: &str = concat!("ferryllm/", env!("CARGO_PKG_VERSION"));

pub fn outbound_user_agent(request: &ChatRequest) -> &str {
    normalized_user_agent(
        request
            .extra
            .get(EXTRA_HTTP_USER_AGENT)
            .and_then(|value| value.as_str()),
    )
}

pub fn normalized_user_agent(user_agent: Option<&str>) -> &str {
    user_agent
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_USER_AGENT)
}

/// Wire protocol used by a client endpoint or backend adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAI,
    Anthropic,
    OpenAIResponses,
    Gemini,
}

/// Raw HTTP response returned by passthrough methods.
pub struct RawResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl RawResponse {
    /// Convert into an [`axum::response::Response`].
    pub fn into_axum(self) -> axum::response::Response {
        let status = axum::http::StatusCode::from_u16(self.status)
            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
        let mut builder = axum::response::Response::builder().status(status);
        for (k, v) in &self.headers {
            if k.to_lowercase() != "transfer-encoding" {
                builder = builder.header(k.as_str(), v.as_str());
            }
        }
        builder
            .body(axum::body::Body::from(self.body))
            .unwrap_or_else(|_| axum::response::Response::new(axum::body::Body::empty()))
    }
}

/// Thread-safe, hot-reloadable string field wrapper.
#[derive(Clone)]
pub struct WatchedField(Arc<RwLock<String>>);

impl WatchedField {
    pub fn new(value: String) -> Self {
        Self(Arc::new(RwLock::new(value)))
    }

    pub fn read(&self) -> String {
        self.0.read().unwrap().clone()
    }

    pub fn update(&self, new_value: String) {
        *self.0.write().unwrap() = new_value;
    }
}

/// Backward-compatible alias.
pub type ApiKey = WatchedField;

/// Errors that can occur during protocol translation or backend communication.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("backend request failed: {0}")]
    BackendError(String),
    #[error("protocol translation error: {0}")]
    TranslationError(String),
    #[error("stream error: {0}")]
    StreamError(String),
    #[error("feature not supported: {feature} (provider: {provider})")]
    UnsupportedFeature { provider: String, feature: String },
}

/// The core trait every provider backend implements.
///
/// Each adapter:
/// 1. Translates the unified [`ChatRequest`] into the provider's native wire format
/// 2. Sends the request to the provider
/// 3. Translates the provider's response (or SSE stream) back into our IR types
#[async_trait]
pub trait Adapter: Send + Sync {
    /// Human-readable provider name, e.g. "openai", "anthropic".
    fn provider_name(&self) -> &str;

    /// Check whether this adapter supports a given model.
    fn supports_model(&self, model: &str) -> bool;

    /// Send a non-streaming chat request.
    ///
    /// Default implementation: forward the request to the backend, deserialize
    /// the raw response, and translate it to [`ChatResponse`].
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError>;

    /// Send a streaming chat request.
    ///
    /// Returns a stream of [`StreamEvent`] values produced by translating the
    /// backend's native SSE stream.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>;

    /// Wire protocol this adapter speaks (e.g. `Protocol::Anthropic`).
    fn protocol(&self) -> Protocol;

    /// Forward a non-streaming request as raw bytes (passthrough mode).
    ///
    /// Default implementation returns `UnsupportedFeature`.
    async fn chat_raw(
        &self,
        _body: &[u8],
        _user_agent: Option<&str>,
    ) -> Result<RawResponse, AdapterError> {
        Err(AdapterError::UnsupportedFeature {
            provider: self.provider_name().to_string(),
            feature: "raw passthrough".into(),
        })
    }

    /// Forward a streaming request as raw bytes (passthrough mode).
    ///
    /// Default implementation returns `UnsupportedFeature`.
    async fn chat_stream_raw(
        &self,
        _body: &[u8],
        _user_agent: Option<&str>,
    ) -> Result<RawResponse, AdapterError> {
        Err(AdapterError::UnsupportedFeature {
            provider: self.provider_name().to_string(),
            feature: "raw stream passthrough".into(),
        })
    }

    /// Update the API key at runtime (for hot-reload from file watcher).
    fn update_api_key(&self, _new_key: String) {}

    /// Update the base URL at runtime (for hot-reload from file watcher).
    fn update_base_url(&self, _new_url: String) {}
}

/// Convenience: every `Box<dyn Adapter>` is itself an [`Adapter`].
#[async_trait]
impl<T: Adapter + ?Sized> Adapter for Box<T> {
    fn provider_name(&self) -> &str {
        (**self).provider_name()
    }

    fn supports_model(&self, model: &str) -> bool {
        (**self).supports_model(model)
    }

    fn protocol(&self) -> Protocol {
        (**self).protocol()
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError> {
        (**self).chat(request).await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>
    {
        (**self).chat_stream(request).await
    }

    async fn chat_raw(
        &self,
        body: &[u8],
        user_agent: Option<&str>,
    ) -> Result<RawResponse, AdapterError> {
        (**self).chat_raw(body, user_agent).await
    }

    async fn chat_stream_raw(
        &self,
        body: &[u8],
        user_agent: Option<&str>,
    ) -> Result<RawResponse, AdapterError> {
        (**self).chat_stream_raw(body, user_agent).await
    }
}
