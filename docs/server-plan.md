# Standalone Server

ferryllm ships as both a Rust library and a config-driven standalone server.
The standalone binary lets users deploy the gateway without writing Rust code:

```bash
ferryllm serve --config ferryllm.toml
ferryllm check-config --config ferryllm.toml
ferryllm version
```

## Current Scope

The server currently provides:

- OpenAI-compatible chat entrypoint: `POST /v1/chat/completions`.
- OpenAI Responses API entrypoints: `POST /v1/responses` and `POST /responses`.
- Anthropic-compatible messages entrypoint: `POST /v1/messages`.
- OpenAI-compatible model listing: `GET /v1/models`.
- Health and readiness endpoints: `GET /health`, `GET /healthz`, and `GET /readyz`.
- Prometheus-style metrics: `GET /metrics`.
- TOML configuration for server, logging, auth, metrics, prompt cache, providers, and routes.
- Exact and prefix route matching, model rewriting, fallback providers, retry, and circuit breaker controls.
- API key authentication plus global and per-key rate/concurrency limits.
- Text or JSON logging through the `[logging]` config section.

## Request Flow

```text
client protocol
  -> entry adapter
  -> ferryllm IR
  -> model route resolution
  -> backend provider adapter
  -> provider protocol
```

Example route log shape:

```text
incoming request entry="anthropic" model=claude-opus-4-6 stream=false
resolved route entry="anthropic" display_model=claude-opus-4-6 backend_model=gpt-5.4 provider="codexapis"
backend response ok provider="codexapis" status="200 OK" latency_ms=1234
```

## Remaining Roadmap

- Weighted, round-robin, and latency-aware provider pools.
- Full config hot reload without restarting the managed process.
- Per-key quota and usage-accounting hooks.
- Packaged Docker images and deployment templates.
