//! Run with:
//!   ANTHROPIC_KEY=sk-ant-... OPENAI_KEY=sk-... cargo run --example basic_server --features http
//!
//! Then test:
//!   curl -X POST http://localhost:3000/v1/chat/completions \
//!     -H "Content-Type: application/json" \
//!     -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hi"}],"stream":true}'

use std::sync::Arc;

use ferryllm::adapters::anthropic::AnthropicAdapter;
use ferryllm::adapters::openai::OpenaiAdapter;
use ferryllm::router::Router;
use ferryllm::server::{AppState, build_router};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .try_init();

    let mut router = Router::new();

    // Register Anthropic backend
    if let Ok(api_key) = std::env::var("ANTHROPIC_KEY") {
        let adapter = Arc::new(AnthropicAdapter::new(
            "https://api.anthropic.com".into(),
            api_key,
        ));
        router.add_route("claude-", "anthropic");
        router.register_adapter(adapter);
        println!("[ferry] Anthropic backend registered");
    }

    // Register OpenAI backend
    if let Ok(api_key) = std::env::var("OPENAI_KEY") {
        let adapter = Arc::new(OpenaiAdapter::new(
            "https://api.openai.com".into(),
            api_key,
        ));
        router.add_route("gpt-", "openai");
        router.add_route("o1", "openai");
        router.add_route("o3", "openai");
        router.register_adapter(adapter);
        println!("[ferry] OpenAI backend registered");
    }

    // Register vLLM / Ollama (OpenAI-compatible) backend
    if let Ok(base_url) = std::env::var("OLLAMA_BASE_URL").or_else(|_| std::env::var("VLLM_BASE_URL")) {
        let api_key = std::env::var("OLLAMA_KEY").unwrap_or_else(|_| "ollama".into());
        let adapter = Arc::new(OpenaiAdapter::new(base_url, api_key));
        router.add_route("llama", "openai");
        router.add_route("qwen", "openai");
        router.add_route("deepseek", "openai");
        router.register_adapter(adapter);
        println!("[ferry] Ollama/vLLM backend registered");
    }

    let state = Arc::new(AppState { router });
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    println!("[ferry] listening on http://0.0.0.0:3000");
    println!("[ferry] OpenAI endpoint:  POST /v1/chat/completions");
    println!("[ferry] Anthropic endpoint: POST /v1/messages");

    axum::serve(listener, app).await.unwrap();
}
