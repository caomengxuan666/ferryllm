//! Run ferryllm as a server with codexapis backend.
//!
//!   export CODX_API_KEY="sk-..." && cargo run --example codexapis_server --features http
//!
//! Then Claude Code can use it via: ANTHROPIC_BASE_URL=http://localhost:3000

use std::sync::Arc;

use ferryllm::adapters::openai::OpenaiAdapter;
use ferryllm::router::Router;
use ferryllm::server::{AppState, Metrics, ServerOptions, build_router};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .try_init();

    let api_key = std::env::var("CODX_API_KEY").expect("CODX_API_KEY not set");
    let base_url = std::env::var("CODX_BASE_URL").unwrap_or_else(|_| "https://codexapis.com".into());

    tracing::info!(base_url = %base_url, "starting codexapis proxy");

    let adapter = Arc::new(OpenaiAdapter::new(base_url.clone(), api_key));
    let mut router = Router::new();
    router.register_adapter(adapter);

    // Route GPT and Grok models directly (no rewrite needed).
    router.add_route("gpt-", "openai");
    router.add_route("grok-", "openai");

    // Rewrite Claude and DeepSeek model requests to gpt-5.5 on codexapis.
    router.add_route("claude-", "openai");
    router.add_rewrite("claude-", "gpt-5.5");
    router.add_route("deepseek-", "openai");
    router.add_rewrite("deepseek-", "gpt-5.5");

    // Catch-all: anything unmatched goes to openai, rewritten to gpt-5.5.
    router.set_default_provider("openai");
    router.add_route("", "openai");
    router.add_rewrite("", "gpt-5.5");

    let state = Arc::new(AppState {
        router,
        options: ServerOptions::default(),
        metrics: Metrics::default(),
    });
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    println!("=== ferryllm + codexapis ===");
    println!("listening on http://0.0.0.0:3000");
    println!("");
    println!("Test with curl:");
    println!("  # OpenAI format (native)");
    println!("  curl -s http://localhost:3000/v1/chat/completions \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"model\":\"gpt-5.5\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}' | jq .");
    println!("");
    println!("  # Anthropic format (Claude Code compatible!)");
    println!("  curl -s http://localhost:3000/v1/messages \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"model\":\"claude-sonnet-4-20250514\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":100}}' | jq .");

    axum::serve(listener, app).await.unwrap();
}
