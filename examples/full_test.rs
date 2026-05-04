//! Comprehensive model test against codexapis.
//!
//! export CODX_API_KEY="sk-..." && cargo run --example full_test --features http

use std::sync::Arc;

use ferryllm::adapter::Adapter;
use ferryllm::adapters::anthropic::AnthropicAdapter;
use ferryllm::adapters::openai::OpenaiAdapter;
use ferryllm::entry::{anthropic, openai};
use ferryllm::ir::{ChatRequest, ContentBlock, Message, Role};

fn api_key() -> String {
    std::env::var("CODX_API_KEY").expect("CODX_API_KEY not set")
}

fn base_url() -> String {
    std::env::var("CODX_BASE_URL").unwrap_or_else(|_| "https://codexapis.com".into())
}

fn simple_req(model: &str, prompt: &str) -> ChatRequest {
    ChatRequest {
        model: model.into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: prompt.into() }],
        }],
        system: None,
        temperature: None,
        max_tokens: Some(60),
        stop_sequences: vec![],
        tools: vec![],
        tool_choice: None,
        stream: false,
        extra: Default::default(),
    }
}

macro_rules! test_model {
    ($adapter:expr, $model:expr, $label:expr) => {
        println!("  {} ({}):", $label, $model);
        match $adapter.chat(&simple_req($model, "Reply with a single short sentence.")).await {
            Ok(resp) => {
                let text = resp.choices.first()
                    .and_then(|c| c.message.as_ref())
                    .and_then(|m| m.content.iter().find_map(|b| match b {
                        ContentBlock::Text { text } if !text.is_empty() => Some(text.trim().to_string()),
                        _ => None,
                    }))
                    .unwrap_or_else(|| "(no text)".into());
                println!("    -> {}", text);
            }
            Err(e) => println!("    -> ERROR: {}", e),
        }
    };
}

#[tokio::main]
async fn main() {
    let key = api_key();
    let url = base_url();
    let openai = Arc::new(OpenaiAdapter::new(url.clone(), key.clone()));
    let anthro = Arc::new(AnthropicAdapter::new(url.clone(), key.clone()));

    println!("=== OpenAI Adapter Tests ===\n");

    test_model!(openai, "gpt-5.5", "GPT-5.5");
    test_model!(openai, "gpt-5.4", "GPT-5.4");
    test_model!(openai, "gpt-5.4-mini", "GPT-5.4-mini");
    test_model!(openai, "gpt-5.3-codex", "GPT-5.3-codex");
    test_model!(openai, "gpt-5.2", "GPT-5.2");
    test_model!(openai, "grok-4.20-0309-non-reasoning", "Grok non-reasoning");
    test_model!(openai, "grok-4.20-0309-reasoning", "Grok reasoning");
    test_model!(openai, "claude-opus-4-6", "Claude via OpenAI");

    println!("\n=== Anthropic Adapter Tests ===\n");

    test_model!(anthro, "claude-opus-4-6", "Claude via Anthropic");

    println!("\n=== Cross-Protocol: Anthropic entry → IR → OpenAI exit ===\n");

    {
        let anthro_body = serde_json::json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Say hello in French."}],
            "max_tokens": 40
        });
        let entry_req: anthropic::AnthropicMessageRequest =
            serde_json::from_value(anthro_body).unwrap();
        let ir_req = anthropic::anthropic_to_ir(&entry_req);
        println!("  Anthropic→GPT-5.5:");
        match openai.chat(&ir_req).await {
            Ok(ir_resp) => {
                let anthro_resp = anthropic::ir_to_anthropic_response(ir_resp);
                println!("    -> {}", serde_json::to_string_pretty(&anthro_resp).unwrap());
            }
            Err(e) => println!("    -> ERROR: {}", e),
        }
    }

    println!("\n=== Streaming: Grok count 1-3 ===\n");

    let stream_req = ChatRequest {
        model: "grok-4.20-0309-non-reasoning".into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Count 1 2 3".into(),
            }],
        }],
        system: None,
        temperature: None,
        max_tokens: Some(40),
        stop_sequences: vec![],
        tools: vec![],
        tool_choice: None,
        stream: true,
        extra: Default::default(),
    };

    match openai.chat_stream(&stream_req).await {
        Ok(mut stream) => {
            use futures::StreamExt;
            print!("  stream: ");
            while let Some(result) = stream.next().await {
                if let Ok(event) = result {
                    if let Some(line) = openai::ir_to_openai_sse(event, "test", "grok") {
                        // Extract just the content text from the SSE line
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data.trim()) {
                                if let Some(text) = v["choices"][0]["delta"]["content"].as_str() {
                                    if !text.is_empty() {
                                        print!("{}", text);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            println!();
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    println!("\n=== Done ===");
}
