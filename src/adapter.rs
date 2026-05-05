use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::ir::{ChatRequest, ChatResponse, StreamEvent};

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
}
