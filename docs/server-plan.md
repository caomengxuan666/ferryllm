# Standalone Server Plan

ferryllm should be released as both a Rust library and a standalone server.

The library enables deep integration. The standalone server enables users to deploy ferryllm without writing code.

## Product Shape

```bash
ferryllm serve --config ferryllm.toml
ferryllm check-config --config ferryllm.toml
ferryllm version
```

## Core Responsibilities

The standalone server should:

- Listen for OpenAI-compatible requests at `/v1/chat/completions`.
- Listen for Anthropic-compatible requests at `/v1/messages`.
- Translate incoming requests into ferryllm IR.
- Resolve the route based on the incoming model name.
- Optionally rewrite the backend model name.
- Forward the request to the selected provider adapter.
- Translate the backend response back into the client protocol.
- Stream SSE responses without buffering the full response.

## MVP Scope

The first public server release should include:

- `ferryllm serve --config ferryllm.toml`.
- TOML configuration.
- OpenAI-compatible provider adapter.
- Anthropic-compatible provider adapter.
- Prefix-based model routing.
- Model rewriting.
- `trace`, `debug`, `info`, `warn`, and `error` logging.
- `/healthz` endpoint.
- `/readyz` endpoint.
- Request timeout.
- Body size limit.
- Graceful shutdown.

## Logging Requirements

The server must make routing decisions visible.

Required request logs:

```text
incoming request entry="anthropic" model=claude-opus-4-6 stream=false
resolved route entry="anthropic" display_model=claude-opus-4-6 backend_model=gpt-5.5 provider="openai"
sending chat request provider="openai" model=gpt-5.5 stream=false
backend response ok provider="openai" status="200 OK" latency_ms=1234
```

Log level guidance:

- `trace`: Wire-level streaming chunks, parser details, very verbose internals.
- `debug`: Route matching decisions, provider selection, retries, connection pool behavior.
- `info`: Request summary, resolved backend model, upstream success.
- `warn`: Fallbacks, recoverable upstream failures, rate-limit pressure.
- `error`: Failed requests, invalid config, upstream hard failures.

Default level should be `info` for local development and can be set to `warn` for production images.

## Health and Readiness

`/healthz` should only prove that the process is alive.

`/readyz` should prove that the server is ready to handle traffic. It should validate:

- Configuration loaded successfully.
- Providers are configured.
- Required secrets are present.
- Listener is active.

Provider network checks should be optional because they can slow startup or create external dependency failures.

## Production Features

After the MVP, the server should add:

- Request IDs.
- Structured JSON logs.
- Access logs.
- Upstream latency measurement.
- Retry policy.
- Circuit breaker.
- Provider fallback.
- Connection pool tuning.
- Max concurrency.
- Per-provider rate limits.
- API-key authentication.
- Prometheus metrics.

## High Availability Roadmap

High-throughput deployments need more than protocol translation.

Planned HA features:

- Multiple provider targets per route.
- Weighted load balancing.
- Round-robin load balancing.
- Least-latency routing.
- Fallback routing after upstream errors.
- Circuit breaker per provider.
- Configurable retry budget.
- Queueing or rejection under overload.

## Relay and Gateway Features

For API relay services and internal model gateways, ferryllm should eventually support:

- API key authentication.
- User or project identity.
- Per-key model allowlists.
- Per-key quotas.
- Per-key rate limits.
- Audit logs.
- Usage accounting hooks.
- Optional cost accounting based on provider usage fields.

## Implementation Steps

Recommended build order:

1. Add `src/bin/ferryllm.rs` with `serve`, `check-config`, and `version` commands.
2. Add `src/config.rs` and TOML parsing.
3. Convert example server wiring into config-driven provider and route registration.
4. Add health and readiness endpoints.
5. Add request IDs and structured logs.
6. Add timeouts, body limits, and graceful shutdown.
7. Add Dockerfile and deployment examples.
8. Add metrics and advanced routing.
