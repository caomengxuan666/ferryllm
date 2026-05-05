# ferryllm

ferryllm is a universal LLM protocol middleware written in Rust.

It lets clients and backends speak different LLM protocols while routing through one unified internal representation. For example, an Anthropic-compatible client such as Claude Code can talk to ferryllm, while ferryllm forwards the request to an OpenAI-compatible backend using a different model such as `gpt-5.5`.

## Why ferryllm

- Accept OpenAI-compatible requests at `POST /v1/chat/completions`.
- Accept Anthropic-compatible requests at `POST /v1/messages`.
- Translate both formats into a shared internal representation.
- Route and rewrite model names before forwarding to upstream providers.
- Use the project as an embeddable Rust library or as a standalone proxy server.
- Build production middleware for gateways, model hubs, private deployments, and API relay services.

## Current Status

The library already includes:

- Unified IR types for chat requests, responses, content blocks, tools, usage, and streaming events.
- OpenAI-compatible entry translation.
- Anthropic-compatible entry translation.
- OpenAI-compatible backend adapter.
- Anthropic-compatible backend adapter.
- Prefix-based model routing and model rewriting.
- Axum HTTP server with OpenAI and Anthropic entry points.
- Configurable tracing logs through `RUST_LOG` in the examples.
- Config-driven server options for body limits, request timeout, API-key auth, and metrics.

A production-grade standalone binary and config-file driven server are planned. See `docs/server-plan.md`.

## Claude Code With GPT-5.5

Claude Code normally sends Anthropic-format requests and model names such as `claude-opus-4-6` or `claude-haiku-4-5-20251001`.

With ferryllm, Claude Code can still use the Anthropic wire protocol, while ferryllm rewrites the backend model to `gpt-5.5` and forwards the request to an OpenAI-compatible provider.

Verified flow:

```text
Claude Code
  -> POST /v1/messages, model = claude-*
  -> ferryllm Anthropic entry
  -> unified IR
  -> route match: claude-
  -> rewrite backend model: gpt-5.5
  -> OpenAI-compatible backend
```

Example with the config-driven server:

```bash
export CODX_API_KEY="your-api-key"
cargo run --bin ferryllm -- serve --config examples/config/codexapis.toml
```

In another shell:

```bash
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_BASE_URL=http://127.0.0.1:3000 \
claude --bare --print --model claude-opus-4-6 "Reply with exactly one short word: pong"
```

For persistent Claude Code or cc-switch setup, see `docs/claude-code.md`.

Expected result:

```text
pong
```

The server logs should show the important part:

```text
incoming request entry="anthropic" model=claude-opus-4-6
resolved route entry="anthropic" display_model=claude-opus-4-6 backend_model=gpt-5.5 provider="openai"
sending chat request provider="openai" model=gpt-5.5
```

This means Claude Code is the client, but the backend model used for inference is `gpt-5.5`.

## User-Defined Model Aliases

Users can expose their own model names and map them to upstream provider models in the config file.

```toml
[[routes]]
match = "cc-gpt55"
match_type = "exact"
provider = "codexapis"
rewrite_model = "gpt-5.5"
```

A client can request `cc-gpt55`, while ferryllm sends `gpt-5.5` to the selected provider.

## Library Usage

ferryllm is designed as an `N + M` adapter system instead of an `N x M` matrix.

- Entry adapters translate client protocols into IR.
- Exit adapters translate IR into provider-native requests.
- The router decides which provider and backend model to use.

High-level library structure:

```text
src/
  adapter.rs        Adapter trait
  ir.rs             Unified request/response/event types
  router.rs         Prefix routes and model rewrites
  server.rs         Axum HTTP server
  entry/            Client protocol translators
  adapters/         Backend provider adapters
```

## Documentation

- `docs/claude-code.md`: Claude Code setup and verification.
- `docs/configuration.md`: Planned configuration file format.
- `docs/server-plan.md`: Standalone server and production roadmap.
- `docs/deployment.md`: Deployment model and operational guidance.
- `docs/load-testing.md`: Local mock-upstream load testing without provider token spend.

## Roadmap

- Release packaging for the config-file driven standalone binary.
- Structured access logs with request IDs and upstream latency.
- Health, readiness, and metrics endpoints.
- Retry, timeout, fallback, and circuit breaker policies.
- Rate limits, concurrency limits, and provider pools.
- API-key authentication for relay deployments.
- Docker and Kubernetes deployment examples.
