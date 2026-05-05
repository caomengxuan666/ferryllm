# ferryllm

Universal LLM protocol middleware for OpenAI, Anthropic, Claude Code, and OpenAI-compatible backends.

[![CI](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml/badge.svg)](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/ferryllm.svg)](https://crates.io/crates/ferryllm)
[![Docs.rs](https://docs.rs/ferryllm/badge.svg)](https://docs.rs/ferryllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ferryllm is a Rust gateway that lets clients and providers speak different LLM protocols through one shared internal representation. Use it as a local Claude Code bridge, a private model gateway, or an embeddable adapter library.

## Highlights

- OpenAI-compatible entrypoint: `POST /v1/chat/completions`
- Anthropic-compatible entrypoint: `POST /v1/messages`
- OpenAI-compatible and Anthropic backend adapters
- Claude Code to OpenAI-compatible backend routing
- Model aliases, prefix routing, and model rewrite rules
- Streaming SSE translation with tool-call support
- Config-driven standalone server: `ferryllm serve --config ferryllm.toml`
- Request timeout, body limit, API-key auth, rate limits, concurrency caps, metrics, retry, fallback, and circuit breaker support
- Library-first architecture for adding new entry protocols and provider adapters

## Why

Most LLM gateways become an `N x M` matrix: every client protocol needs custom code for every provider protocol. ferryllm uses an `N + M` design instead.

```text
Client protocol -> ferryllm IR -> provider protocol
```

That means a new backend adapter can immediately serve OpenAI-style clients, Anthropic-style clients, and Claude Code without rewriting every path.

## Quick Start

Install from crates.io:

```bash
cargo install ferryllm
```

Or run from source:

```bash
git clone https://github.com/caomengxuan666/ferryllm.git
cd ferryllm
cargo run --features http --bin ferryllm -- serve --config examples/config/codexapis.toml
```

Use an OpenAI-compatible provider key:

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=info ferryllm serve --config examples/config/codexapis.toml
```

Smoke test the Anthropic-compatible endpoint:

```bash
curl -s http://127.0.0.1:3000/v1/messages \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer local-test-token' \
  -d '{"model":"cc-gpt55","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}'
```

## Claude Code With GPT-5.4

Claude Code sends Anthropic-format requests. ferryllm can receive those requests, rewrite the model, and forward them to an OpenAI-compatible backend.

```text
Claude Code
  -> POST /v1/messages, model = claude-*
  -> ferryllm Anthropic entry
  -> unified IR
  -> route match: claude-
  -> rewrite backend model: gpt-5.4
  -> OpenAI-compatible backend
```

Start ferryllm:

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=ferryllm=info,tower_http=info \
  ferryllm serve --config examples/config/codexapis.toml
```

Point Claude Code at ferryllm:

```bash
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_BASE_URL=http://127.0.0.1:3000 \
claude --bare --print --model claude-opus-4-6 \
  "Reply with exactly one short word: pong"
```

Expected output:

```text
pong
```

See [docs/claude-code.md](docs/claude-code.md) for persistent Claude Code and cc-switch setup.

## Configuration

ferryllm uses TOML configuration. Secrets stay in environment variables.

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32

[logging]
level = "info"
format = "text"

[metrics]
enabled = true

[[providers]]
name = "codexapis"
type = "openai"
base_url = "https://codexapis.com"
api_key_env = "CODX_API_KEY"

[[routes]]
match = "cc-gpt55"
match_type = "exact"
provider = "codexapis"
rewrite_model = "gpt-5.4"

[[routes]]
match = "claude-"
provider = "codexapis"
rewrite_model = "gpt-5.4"
```

Check a config without starting the server:

```bash
ferryllm check-config --config examples/config/codexapis.toml
```

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages |
| `GET /health` | Simple health check |
| `GET /healthz` | Kubernetes-style liveness check |
| `GET /readyz` | Readiness check |
| `GET /metrics` | Prometheus-style metrics |

## Architecture

```text
src/
  adapter.rs        Adapter trait
  ir.rs             Unified request, response, content, tool, and stream types
  router.rs         Exact and prefix model routing
  server.rs         Axum HTTP server
  config.rs         TOML config loader and validator
  entry/            Client protocol translators
  adapters/         Backend provider adapters
```

More detail: [docs/architecture.md](docs/architecture.md).

## Load Testing

ferryllm ships a benchmark-style load tester for local mock-upstream testing:

```bash
MOCK_DELAY_MS=20 cargo run --example mock_openai_upstream --features http
cargo run --release --example load_test --features http -- \
  --preset mock-anthropic \
  --requests 10000 \
  --concurrency 512
```

See [docs/load-testing.md](docs/load-testing.md).

## Documentation

- [Chinese README](README.zh-CN.md)
- [Architecture](docs/architecture.md)
- [Claude Code setup](docs/claude-code.md)
- [Configuration](docs/configuration.md)
- [Compatibility notes](docs/compatibility.md)
- [Deployment](docs/deployment.md)
- [Load testing](docs/load-testing.md)
- [Prompt caching and token observability](docs/prompt-caching.md)

## Prompt Cache

With `prompt-observability` enabled, ferryllm exposes prompt-cache usage in
logs and `/metrics`. In the current Claude Code + Codex relay setup, we have
observed cache read rates around 99.8% on stable prompts when the system prefix
is normalized and volatile transport metadata is stripped.

The exact result depends on the upstream provider, prompt shape, and how stable
the prefix is across requests. See [docs/prompt-caching.md](docs/prompt-caching.md)
for the cache-key rules and the current tuning knobs.

## Roadmap

- More provider adapters, including Gemini
- Weighted and latency-aware provider pools
- Hot-reload configuration
- Richer Prometheus metrics labels
- Per-key quota and usage accounting hooks
- Packaged Docker images and deployment templates

## License

MIT. See [LICENSE](LICENSE).
