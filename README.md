# ferryllm

Desktop-first LLM gateway and launcher for Codex, Claude Code, OpenCode, OpenAI, Anthropic, and OpenAI-compatible backends.

[![CI](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml/badge.svg)](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/ferryllm.svg)](https://crates.io/crates/ferryllm)
[![Docs.rs](https://docs.rs/ferryllm/badge.svg)](https://docs.rs/ferryllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ferryllm is a native desktop control panel backed by a Rust LLM gateway. The GUI is the primary workflow: configure providers, start the local gateway, inspect runtime metrics, map client-visible model names to upstream models, and launch Codex, Claude Code, OpenCode, or VS Code with the right environment.

The CLI and GUI are thin shells around the same core gateway engine. Protocol translation, routing, prompt-cache handling, reasoning policy, and provider adapters live in the Rust core so desktop and command-line behavior stay consistent.

## What It Does

- Provides a Tauri desktop app with Dashboard, Providers, Launcher, Usage Logs, and Settings
- Ships provider presets, provider testing, best-effort usage probes, and `/v1/models` model discovery
- Launches Codex, Claude Code, OpenCode, and VS Code against the local gateway
- Remembers workspaces, provider bindings, recent launches, and discovered local AI sessions
- Accepts OpenAI-compatible chat requests at `POST /v1/chat/completions`
- Accepts OpenAI Responses API requests at `POST /v1/responses`
- Accepts Anthropic-compatible messages at `POST /v1/messages`
- Rewrites model names with exact and prefix routing rules, including user-editable model aliases
- Forwards to OpenAI-compatible, OpenAI Responses, Anthropic, or optional Gemini backend adapters
- Preserves tool calls and SSE streaming behavior
- Forwards or synthesizes `User-Agent` headers for upstream requests
- Keeps prompt-cache keys stable while stripping transport metadata
- Applies configurable reasoning policy: preserve, fill missing, cap, or force

## Why ferryllm

Most gateways end up as an `N x M` matrix: every client protocol needs custom code for every provider protocol.

ferryllm uses `N + M` routing instead:

```text
Client protocol -> ferryllm IR -> provider protocol
```

That makes it easier to:

- operate the gateway from a GUI instead of hand-editing TOML for every change
- put Claude Code behind a stable backend
- expose one local gateway to multiple client protocols
- keep cache behavior predictable
- add new providers without rewriting every client path

## Fast Start

Recommended path: install the desktop app from
[GitHub Releases](https://github.com/caomengxuan666/ferryllm/releases/latest).

- Windows: download and run the `.exe` or `.msi` installer.
- macOS: download and open the `.dmg`.
- Linux: download and install the `.deb`.

Open the app, add a provider from the preset grid or custom form, test it, fetch models if the provider exposes `/v1/models`, choose model mappings, then start the gateway from the GUI. The app runs the same engine as the CLI:

```bash
ferryllm serve --config <generated-config.toml>
```

CLI-only install is still available:

```bash
cargo install ferryllm
```

This installs only the `ferryllm` gateway CLI, not the desktop app.

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

## Desktop GUI

The desktop app is the main ferryllm experience.

![ferryllm desktop provider overview](docs/assets/gui-main.png)

It includes:

- **Launcher**: project list, workspace creation/opening, project-to-provider binding, per-project reasoning selection, Codex/Claude/OpenCode launch, VS Code launch, and session resume.
- **Providers**: logo preset grid, custom providers, provider test, usage probe, copy/delete actions, key-source status, model mappings, and model discovery from provider model endpoints.
- **Dashboard**: gateway state, health/readiness, requests, success/error counts, latency, cache hit ratio, per-provider/model table, prompt-cache bar, and recent logs.
- **Usage Logs**: recent gateway and launcher events in one table.
- **Settings**: runtime limits, retry/circuit breaker options, reasoning policy, auth, prompt-cache controls, logging, and desktop preferences.

After opening the app, configure a provider, save the config, and start the gateway. The GUI writes a runnable TOML config and launches:

```bash
ferryllm serve --config <generated-config.toml>
```

The packaged app first looks for the bundled `ferryllm` sidecar, then falls back
to a `ferryllm` executable on `PATH`. Launcher actions start Codex, Claude Code,
OpenCode, or VS Code with `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, model aliases,
and reasoning defaults pointing at the local gateway.

![ferryllm desktop launcher](docs/assets/gui-launcher.png)

![ferryllm desktop provider settings](docs/assets/gui-provider-detail.png)

See [docs/claude-code.md](docs/claude-code.md) for persistent Claude Code and cc-switch setup.

## Configuration

ferryllm uses TOML config. Secrets stay in environment variables.

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32
reasoning_policy = "fill_missing"
default_reasoning_effort = "medium"
# Optional. With reasoning_policy = "cap", prevents clients from exceeding this.
# max_reasoning_effort = "high"
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
ansi = false

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
# Default path for controlling reasoning effort today.
type = "openai_responses"
base_url = "https://codexapis.com"
api_key_env = "CODX_API_KEY"
# If you want the legacy Chat Completions path instead, switch this back to:
# type = "openai"

# Or use key_watch for hot-reload from external config files:
# [[providers.key_watch]]
# file = "C:/Users/hzz/.claude/settings.json"
# path = "env.ANTHROPIC_AUTH_TOKEN"

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

For hot-reload API key configuration (e.g., from cc-switch settings), see the [key_watch](docs/configuration.md#key-watch-hot-reload-api-keys) section in the configuration docs.

To route OpenAI-compatible upstream calls through the Responses API instead of
Chat Completions, use provider type `openai_responses`. Default builds,
including `cargo install ferryllm`, include this adapter. If you build with
`--no-default-features`, add the `openai-responses` feature explicitly:

```bash
cargo build --release --features http,prompt-observability,openai-responses --bin ferryllm
```

```toml
[[providers]]
name = "codexapis"
type = "openai_responses"
base_url = "https://codexapis.com"
api_key_env = "CODX_API_KEY"
```

See [examples/config/codexapis-responses.toml](examples/config/codexapis-responses.toml).

## Reasoning Effort

Configure model reasoning in TOML or from the GUI Settings page:

```toml
[server]
reasoning_policy = "cap"
default_reasoning_effort = "medium"
max_reasoning_effort = "high"
```

Valid effort values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, and `ultracode`.

`reasoning_policy` controls how ferryllm handles client-provided reasoning:

- `preserve`: leave client reasoning untouched.
- `fill_missing`: apply `default_reasoning_effort` only when the client omits reasoning.
- `cap`: let clients choose effort, but clamp it at `max_reasoning_effort`.
- `force`: replace client reasoning with `default_reasoning_effort`.

Launcher can also pass a project-level reasoning choice to Codex and Claude Code. The gateway still applies the server policy, so GUI launches and CLI calls are governed by the same core rules.

At `info` level, ferryllm logs `requested_reasoning` and `applied_reasoning`. With request-shape debug enabled, it also logs outbound `reasoning=effort=...` or Anthropic thinking budget summaries.

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /responses` | Responses API compatibility alias |
| `POST /v1/messages` | Anthropic-compatible messages |
| `GET /v1/models` | OpenAI-compatible model listing |
| `GET /health` | Simple health check |
| `GET /healthz` | Kubernetes-style liveness check |
| `GET /readyz` | Readiness check |
| `GET /metrics` | Prometheus-style metrics with per-provider/model labels |

## Prompt Cache

ferryllm keeps prompt-cache keys stable while stripping transport metadata and normalizing the prompt prefix.

With `prompt-observability` enabled, ferryllm logs prompt-cache usage and exposes it through `/metrics`.

For Claude Code deployments, the important knobs are:

- `relocate_system_prefix_range`
- `strip_system_line_prefixes`
- `openai_prompt_cache_key`
- `default_reasoning_effort`
- `reasoning_policy`
- `max_reasoning_effort`

See [docs/prompt-caching.md](docs/prompt-caching.md) and [docs/reasoning-control.md](docs/reasoning-control.md).

## Architecture

```text
desktop/            Tauri GUI, launcher, dashboard, provider editor
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

- More provider adapters and provider-specific tuning
- Weighted and latency-aware provider pools
- Provider usage/balance adapters for NewAPI/OneAPI-style dashboards
- Full config hot reload without managed-process restart
- Richer Prometheus metrics dimensions
- Per-key quota and usage accounting hooks
- Packaged Docker images and deployment templates

## License

MIT. See [LICENSE](LICENSE).
