//! Test ferryllm against codexapis.com.
//!
//! Prerequisites:
//!   export CODX_API_KEY="sk-..."
//!   export CODX_BASE_URL="https://codexapis.com/v1"
//!
//! Run:
//!   cargo run --example codexapis_test --features http

use std::sync::Arc;

use ferryllm::adapter::Adapter;
use ferryllm::adapters::openai::OpenaiAdapter;
use ferryllm::entry::{anthropic, openai};
use ferryllm::ir::{ChatRequest, ContentBlock, Message, Role};

fn api_key() -> String {
    std::env::var("CODX_API_KEY").expect("CODX_API_KEY not set")
}

fn base_url() -> String {
    std::env::var("CODX_BASE_URL").unwrap_or_else(|_| "https://codexapis.com/v1".into())
}

#[tokio::main]
async fn main() {
    let key = api_key();
    let url = base_url();

    // ─── Test 1: Direct adapter call (OpenAI exit) ───────────────────────
    println!("=== Test 1: OpenAI adapter → codexapis (non-streaming) ===");
    let adapter = Arc::new(OpenaiAdapter::new(url.clone(), key.clone()));
    let req = ChatRequest {
        model: "grok-4.20-0309-non-reasoning".into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Say hello in exactly one sentence.".into(),
            }],
        }],
        system: None,
        temperature: None,
        max_tokens: Some(100),
        stop_sequences: vec![],
        tools: vec![],
        tool_choice: None,
        stream: false,
        extra: Default::default(),
    };

    match adapter.chat(&req).await {
        Ok(resp) => {
            println!("  id:     {}", resp.id);
            println!("  model:  {}", resp.model);
            for c in &resp.choices {
                if let Some(msg) = &c.message {
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => println!("  text:   {}", text),
                            other => println!("  block:  {:?}", other),
                        }
                    }
                }
            }
            println!("  usage:  {:?} tokens", resp.usage.total_tokens);
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    // ─── Test 2: OpenAI entry → IR → OpenAI adapter (full pipeline) ─────
    println!("\n=== Test 2: OpenAI entry → IR → codexapis ===");
    let openai_body = serde_json::json!({
        "model": "grok-4.20-0309-non-reasoning",
        "messages": [
            {"role": "user", "content": "What is 2+2? Reply in one word."}
        ],
        "max_tokens": 50
    });

    let entry_req: openai::OpenAIChatRequest =
        serde_json::from_value(openai_body).unwrap();
    let ir_req = openai::openai_to_ir(&entry_req);

    match adapter.chat(&ir_req).await {
        Ok(ir_resp) => {
            let openai_resp = openai::ir_to_openai_response(ir_resp);
            println!("  response: {}", serde_json::to_string_pretty(&openai_resp).unwrap());
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    // ─── Test 3: Anthropic entry → IR → OpenAI adapter (cross-protocol) ──
    println!("\n=== Test 3: Anthropic entry → IR → OpenAI adapter ===");
    // Cross-protocol: Anthropic-format request → IR → OpenAI adapter → codexapis.
    // The client speaks Anthropic, but the backend only accepts OpenAI format.
    // ferryllm translates both ways transparently.
    let anthro_body = serde_json::json!({
        "model": "grok-4.20-0309-non-reasoning",
        "messages": [
            {"role": "user", "content": "Reply with just the word 'pong'"}
        ],
        "max_tokens": 50
    });

    let entry_req: anthropic::AnthropicMessageRequest =
        serde_json::from_value(anthro_body).unwrap();
    let ir_req = anthropic::anthropic_to_ir(&entry_req);
    println!("  IR model: {}", ir_req.model);
    println!("  IR messages: {}", ir_req.messages.len());
    if let Some(sys) = &ir_req.system {
        println!("  IR system:  {}", sys);
    }

    // The model is grok, which codexapis only serves via OpenAI format.
    // ferryllm's IR decouples the entry protocol from the exit protocol.
    match adapter.chat(&ir_req).await {
        Ok(ir_resp) => {
            let anthro_resp = anthropic::ir_to_anthropic_response(ir_resp);
            println!("  response: {}", serde_json::to_string_pretty(&anthro_resp).unwrap());
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    // ─── Test 4: Streaming ──────────────────────────────────────────────
    println!("\n=== Test 4: OpenAI entry → IR → codexapis (streaming) ===");
    let stream_req = ChatRequest {
        model: "grok-4.20-0309-non-reasoning".into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Count from 1 to 5, one per line.".into(),
            }],
        }],
        system: None,
        temperature: None,
        max_tokens: Some(100),
        stop_sequences: vec![],
        tools: vec![],
        tool_choice: None,
        stream: true,
        extra: Default::default(),
    };

    match adapter.chat_stream(&stream_req).await {
        Ok(mut stream) => {
            use futures::StreamExt;
            println!("  stream started:");
            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                        let sse = openai::ir_to_openai_sse(event, "test-id", "grok");
                        if let Some(line) = sse {
                            print!("  {}", line);
                        }
                    }
                    Err(e) => println!("  stream error: {}", e),
                }
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    println!("\n=== All tests done ===");
}
