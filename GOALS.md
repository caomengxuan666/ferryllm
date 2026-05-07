# ferryllm

A universal LLM protocol middleware written in Rust. Ferry your requests between OpenAI, Anthropic, and beyond — one unified internal representation, N+M adapters instead of an N×M matrix.

## Design goals

1. **Unified IR** — A self-built superset internal representation that captures the semantics of all major LLM protocols (text, images, tool calls, thinking blocks, streaming lifecycle), not tied to any single provider's format.

2. **N+M adapter architecture** — Entry adapters translate client protocols into IR; exit adapters translate IR into backend-native requests. Adding a new provider requires only one exit adapter, and all entry formats automatically gain access.

3. **Zero-copy, async, streaming-first** — Built on tokio + axum + reqwest. SSE streams are translated on the fly without buffering the entire response. Designed for <1ms proxy overhead.

4. **Dual entry points** — Exposes both `POST /v1/chat/completions` (OpenAI format) and `POST /v1/messages` (Anthropic format). Route by model name to the appropriate backend.

5. **Embeddable library + standalone binary** — Core types and adapters are a library (`lib.rs`); the HTTP server is behind a `http` feature flag. Usable as a Rust dependency or as a standalone proxy.

## Architecture

```
Client (OpenAI format) ──→ POST /v1/chat/completions ──→ entry/openai ──→ IR
Client (Anthropic format) ─→ POST /v1/messages ──────────→ entry/anthropic ─→ IR
                                                                                │
                                                                         router.resolve(model)
                                                                                │
                                   ┌────────────────────────────────────────────┤
                                   ▼                                            ▼
                            adapters/anthropic                           adapters/openai
                            IR → Anthropic native → HTTP                  IR → OpenAI native → HTTP
                                   │                                            │
                                   ▼                                            ▼
                              Anthropic API                              OpenAI API / vLLM
```

## Project structure

```
src/
├── ir.rs           # Unified IR: ChatRequest, ChatResponse, Message, ContentBlock,
│                   #   StreamEvent (8 event types), Tool, Usage
├── adapter.rs      # Adapter trait: chat() + chat_stream()
├── router.rs       # Model router: prefix-match + default fallback
├── server.rs       # axum HTTP server (behind "http" feature)
├── adapters/
│   ├── openai.rs   # Exit adapter: IR ↔ OpenAI wire format
│   └── anthropic.rs# Exit adapter: IR ↔ Anthropic wire format
└── entry/
    ├── openai.rs   # Entry translator: OpenAI client JSON ↔ IR
    └── anthropic.rs# Entry translator: Anthropic client JSON ↔ IR
```

## Roadmap

- [x] Unified IR types
- [x] Adapter trait
- [x] OpenAI adapter (backend)
- [x] Anthropic adapter (backend)
- [x] Entry translation (client-side)
- [x] Model router
- [x] HTTP server with SSE streaming
- [x] Gemini adapter
- [x] Stateful tool-call JSON accumulation in streams
- [ ] Connection pooling & keep-alive
- [ ] Hot-reload configuration
- [x] Observability (tracing / metrics)
- [x] Rate limiting
