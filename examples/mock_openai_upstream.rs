//! Local mock OpenAI-compatible upstream for load testing ferryllm without
//! spending provider tokens.
//!
//! Run with:
//!   MOCK_DELAY_MS=20 cargo run --example mock_openai_upstream --features http

use std::time::Duration;

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde_json::{json, Value};

#[derive(Clone)]
struct MockState {
    delay: Duration,
}

#[tokio::main]
async fn main() {
    let delay_ms = std::env::var("MOCK_DELAY_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(20);
    let state = MockState {
        delay: Duration::from_millis(delay_ms),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:4010")
        .await
        .expect("bind mock upstream");

    println!("mock OpenAI upstream listening on http://127.0.0.1:4010");
    axum::serve(listener, app)
        .await
        .expect("serve mock upstream");
}

async fn chat_completions(
    State(state): State<MockState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    tokio::time::sleep(state.delay).await;

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("mock-model");
    let response = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "pong"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    });
    Ok(Json(response))
}
