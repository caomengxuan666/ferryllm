# Architecture

ferryllm uses an `N + M` adapter architecture.

Instead of implementing every client protocol against every provider protocol directly, ferryllm translates client requests into one unified internal representation, then translates that representation into the selected provider format.

## Flow

```text
OpenAI client
  -> POST /v1/chat/completions
  -> entry/openai
  -> IR
  -> router.resolve(model)
  -> backend adapter
  -> provider API

Anthropic client
  -> POST /v1/messages
  -> entry/anthropic
  -> IR
  -> router.resolve(model)
  -> backend adapter
  -> provider API
```

## Core Components

`ir.rs`

Defines the shared request, response, message, content block, tool, usage, and streaming event types.

`entry/openai.rs`

Translates OpenAI-compatible client requests into IR and IR responses back into OpenAI-compatible responses.

`entry/anthropic.rs`

Translates Anthropic-compatible client requests into IR and IR responses back into Anthropic-compatible responses.

`adapter.rs`

Defines the backend adapter trait. Each provider adapter implements non-streaming chat and streaming chat.

`adapters/openai.rs`

Translates IR into OpenAI-compatible backend requests and parses OpenAI-compatible responses.

`adapters/anthropic.rs`

Translates IR into Anthropic backend requests and parses Anthropic responses.

`router.rs`

Selects the provider and backend model based on the incoming model name.

`server.rs`

Exposes HTTP routes and connects entry translation, routing, backend calls, and response translation.

## Model Routing

Routing is prefix-based in the current implementation.

Example:

```text
incoming model: claude-opus-4-6
matched prefix: claude-
provider: openai
backend model: gpt-5.5
```

This lets an Anthropic-compatible client use an OpenAI-compatible backend without changing the client.

## Streaming

ferryllm is designed to translate streaming events without buffering the full response.

The IR includes streaming events such as:

- `MessageStart`
- `ContentBlockStart`
- `ContentBlockDelta`
- `ContentBlockStop`
- `MessageDelta`
- `MessageStop`
- `Error`

Entry adapters convert these events into the client protocol's SSE format.

## Library and Server Separation

The library should remain reusable by other Rust projects.

The standalone server should be a thin production wrapper around the same library components:

```text
config file
  -> providers
  -> routes
  -> Router
  -> AppState
  -> axum server
```

This keeps embedded use cases and standalone deployments aligned.
