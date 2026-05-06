# ferryllm

Universal LLM protocol middleware for OpenAI, Anthropic, Claude Code, and OpenAI-compatible backends.

[![CI](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml/badge.svg)](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/ferryllm.svg)](https://crates.io/crates/ferryllm)
[![Docs.rs](https://docs.rs/ferryllm/badge.svg)](https://docs.rs/ferryllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ferryllm is a Rust gateway that normalizes client and provider LLM protocols into one internal representation. Use it as a Claude Code bridge, a private model gateway, or an embeddable adapter layer.

## What It Does

- Accepts OpenAI-compatible chat requests at `POST /v1/chat/completions`
- Accepts Anthropic-compatible messages at `POST /v1/messages`
- Rewrites model names with exact and prefix routing rules
- Forwards to OpenAI-compatible or Anthropic backend adapters
- Preserves tool calls and SSE streaming behavior
- Keeps prompt-cache keys stable while stripping transport metadata
- Maps reasoning control through the IR and provider adapters

## Why ferryllm

Most gateways end up as an `N x M` matrix: every client protocol needs custom code for every provider protocol.

ferryllm uses `N + M` routing instead:

```text
Client protocol -> ferryllm IR -> provider protocol
```

That makes it easier to:

- put Claude Code behind a stable backend
- expose one gateway to multiple client protocols
- keep cache behavior predictable
- add new providers without rewriting every client path

## Fast Start

Install from crates.io:

```bash
cargo install ferryllm
```

Run from source:

```bash
git clone https://github.com/caomengxuan666/ferryllm.git
cd ferryllm
cargo run --features http --bin ferryllm -- serve --config examples/config/codexapis.toml
```

Set the provider key and start the server:

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

## Claude Code Bridge

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

ferryllm uses TOML config. Secrets stay in environment variables.

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32
default_reasoning_effort = "medium"
# Optional. Uncomment to cap in-flight requests.
# max_concurrent_requests = 128
# Optional. Uncomment to cap total requests per minute.
# rate_limit_per_minute = 600
# Optional non-streaming upstream resilience. Streaming requests are not retried.
# retry_attempts = 2
# retry_backoff_ms = 100
# circuit_breaker_failures = 5
# circuit_breaker_cooldown_secs = 30

[logging]
level = "info"
format = "text"

[auth]
enabled = false
# api_keys_env = "FERRYLLM_API_KEYS"
# Optional per-client caps, keyed by the authenticated API key.
# per_key_rate_limit_per_minute = 120
# per_key_max_concurrent_requests = 8

[metrics]
enabled = true

[prompt_cache]
auto_inject_anthropic_cache_control = true
cache_system = true
cache_tools = true
cache_last_user_message = true
openai_prompt_cache_key = "ferryllm"
# openai_prompt_cache_retention = "24h"
debug_log_request_shape = true
relocate_system_prefix_range = "0..1"
log_relocated_system_text = false
strip_system_line_prefixes = ["x-anthropic-billing-header:"]

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

[[routes]]
match = "gpt-"
provider = "codexapis"

[[routes]]
match = "grok-"
provider = "codexapis"

[[routes]]
match = "*"
provider = "codexapis"
rewrite_model = "gpt-5.4"
```

Check a config without starting the server:

```bash
ferryllm check-config --config examples/config/codexapis.toml
```

## Reasoning Effort

Set the default model reasoning depth in TOML:

```toml
[server]
default_reasoning_effort = "medium"
```

Valid values are `none`, `low`, `medium`, `high`, `xhigh`, and `x_high`.

This default is applied only when the client request does not already include an explicit reasoning or thinking control. For Claude Code today, changing this in TOML is the practical way to control the forwarded OpenAI-compatible `reasoning.effort`.

Run with debug logging and look for `reasoning=effort=...` in the outbound request-shape log to confirm what ferryllm sent upstream.

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages |
| `GET /health` | Simple health check |
| `GET /healthz` | Kubernetes-style liveness check |
| `GET /readyz` | Readiness check |
| `GET /metrics` | Prometheus-style metrics |

## Prompt Cache

ferryllm keeps prompt-cache keys stable while stripping transport metadata and normalizing the prompt prefix.

With `prompt-observability` enabled, ferryllm logs prompt-cache usage and exposes it through `/metrics`.

For Claude Code deployments, the important knobs are:

- `relocate_system_prefix_range`
- `strip_system_line_prefixes`
- `openai_prompt_cache_key`
- `default_reasoning_effort`

See [docs/prompt-caching.md](docs/prompt-caching.md) and [docs/reasoning-control.md](docs/reasoning-control.md).

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
- [Reasoning control](docs/reasoning-control.md)

## Roadmap

- More provider adapters, including Gemini
- Weighted and latency-aware provider pools
- Hot-reload configuration
- Richer Prometheus metrics labels
- Per-key quota and usage accounting hooks
- Packaged Docker images and deployment templates

## License

MIT. See [LICENSE](LICENSE).
