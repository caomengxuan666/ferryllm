# Configuration

This document describes the configuration model for the standalone ferryllm server.

ferryllm includes a config-file driven binary, so users can run the server without writing custom Rust code:

```bash
ferryllm serve --config ferryllm.toml
```

## Design Goals

- Keep secrets out of committed files.
- Make model routing explicit and auditable.
- Support multiple providers and model rewrite rules.
- Support local proxy use cases and production relay deployments.
- Allow advanced routing features such as fallback and weighted balancing later without breaking the basic format.

## Format

TOML is recommended for the first public server release.

Reasons:

- It is familiar in the Rust ecosystem.
- It is easy to read and review.
- It maps cleanly to structured configuration.
- It is less error-prone than large environment-variable based setups.

Environment variables should still be used for secrets and runtime overrides.

## Minimal Example

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32
max_concurrent_requests = 128
rate_limit_per_minute = 600
default_reasoning_effort = "medium"

[logging]
level = "info"
format = "text"

[auth]
enabled = false

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
rewrite_model = "gpt-5.5"

[[routes]]
match = "claude-"
provider = "codexapis"
rewrite_model = "gpt-5.5"
fallback_providers = ["backup-openai"]

[[routes]]
match = "gpt-"
provider = "codexapis"

[[routes]]
match = "*"
provider = "codexapis"
rewrite_model = "gpt-5.5"
```

## Server Section

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32
max_concurrent_requests = 128
rate_limit_per_minute = 600
graceful_shutdown_secs = 30
```

Fields:

- `listen`: Address and port to bind.
- `request_timeout_secs`: Maximum request duration before ferryllm returns an error.
- `body_limit_mb`: Maximum request body size.
- `max_concurrent_requests`: Optional maximum number of in-flight OpenAI/Anthropic chat requests. Requests above the limit return `429`.
- `rate_limit_per_minute`: Optional global request rate cap. Requests above the limit return `429`.
- `graceful_shutdown_secs`: Time allowed for in-flight requests during shutdown.
- `default_reasoning_effort`: Optional default reasoning effort applied only when the client does not send an explicit reasoning or thinking control. Valid values are `none`, `low`, `medium`, `high`, `xhigh`, and `x_high`.

`default_reasoning_effort` is a control-plane setting. It is not included in ferryllm's prompt cache key.

## Logging Section

```toml
[logging]
level = "info"
format = "json"
```

Fields:

- `level`: One of `trace`, `debug`, `info`, `warn`, or `error`.
- `format`: `text` for local development, `json` for production log pipelines.

Expected logs should include:

- Incoming protocol entry.
- Incoming model name.
- Stream flag.
- Selected provider.
- Rewritten backend model.
- Upstream status.
- Upstream latency.
- Error category and message.

## Providers

```toml
[[providers]]
name = "openai"
type = "openai"
base_url = "https://api.openai.com"
api_key_env = "OPENAI_API_KEY"
```

Provider fields:

- `name`: Unique provider name used by routes.
- `type`: Adapter type, for example `openai` or `anthropic`.
- `base_url`: Provider base URL without endpoint path rewriting in routes.
- `api_key_env`: Environment variable containing the secret.

Provider-specific options can be added later:

```toml
connect_timeout_secs = 10
request_timeout_secs = 120
max_idle_connections = 256
```

## Routes

```toml
[[routes]]
match = "claude-"
provider = "codexapis"
rewrite_model = "gpt-5.5"
```

Route fields:

- `match`: Prefix match. `*` means catch-all.
- `match_type`: Optional. `prefix` by default, or `exact` for user-defined model aliases.
- `provider`: Provider name.
- `rewrite_model`: Optional backend model override.
- `fallback_providers`: Optional provider names tried in order for non-streaming requests when the primary provider fails.

Routes should be evaluated by longest-prefix match. This lets specific rules override broad defaults.

## User-Defined Model Aliases

Users can expose their own model names to clients and map those names to provider models.

Exact alias example:

```toml
[[routes]]
match = "cc-gpt55"
match_type = "exact"
provider = "codexapis"
rewrite_model = "gpt-5.5"
```

A client can now request `cc-gpt55`, while ferryllm sends `gpt-5.5` to the upstream provider.

Prefix mapping example:

```toml
[[routes]]
match = "claude-"
match_type = "prefix"
provider = "codexapis"
rewrite_model = "gpt-5.5"
```

This is useful for clients such as Claude Code, which send Anthropic model names even when the backend is not Anthropic.

## Fallback Routing

Simple fallback routing is supported for non-streaming requests:

```toml
[[providers]]
name = "primary"
type = "openai"
base_url = "https://primary.example.com"
api_key_env = "PRIMARY_API_KEY"

[[providers]]
name = "backup"
type = "openai"
base_url = "https://backup.example.com"
api_key_env = "BACKUP_API_KEY"

[[routes]]
match = "claude-"
provider = "primary"
rewrite_model = "gpt-5.5"
fallback_providers = ["backup"]
```

Fallbacks use the same rewritten backend model. Streaming fallback is intentionally not attempted after a stream has started.

## Future Route Strategies

The basic route format should leave room for advanced strategies.

Advanced fallback example:

```toml
[[routes]]
match = "claude-"
strategy = "fallback"

[[routes.targets]]
provider = "primary"
model = "gpt-5.5"

[[routes.targets]]
provider = "backup"
model = "gpt-5.4"
```

Weighted example:

```toml
[[routes]]
match = "gpt-"
strategy = "weighted"

[[routes.targets]]
provider = "provider-a"
weight = 80

[[routes.targets]]
provider = "provider-b"
weight = 20
```

## Authentication

For local usage, authentication can be disabled.

For public relay deployments, API key authentication should be enabled.

```toml
[auth]
enabled = true
api_keys_env = "FERRYLLM_API_KEYS"
per_key_rate_limit_per_minute = 120
per_key_max_concurrent_requests = 8
```

`FERRYLLM_API_KEYS` contains comma-separated keys:

```bash
export FERRYLLM_API_KEYS="key-one,key-two"
```

Clients can authenticate with either header:

```text
Authorization: Bearer key-one
x-api-key: key-one
```

Per-key limits are optional and only apply when authentication is enabled. They are tracked in-memory per server process and keyed by a hash of the authenticated API key, not by the raw key string.

## Prompt Cache

```toml
[prompt_cache]
auto_inject_anthropic_cache_control = true
cache_system = true
cache_tools = true
cache_last_user_message = true
openai_prompt_cache_key = "ferryllm"
# openai_prompt_cache_retention = "24h"
debug_log_request_shape = true
# relocate_system_prefix_range = "64..128"
# log_relocated_system_text = false
# strip_system_line_prefixes = ["x-anthropic-billing-header:"]
```

Fields:

- `auto_inject_anthropic_cache_control`: Preserve explicit Anthropic `cache_control`, and when missing, inject `{"type":"ephemeral"}` on stable cache breakpoints.
- `cache_system`: Add a breakpoint to top-level Anthropic system text.
- `cache_tools`: Add a breakpoint to the last Anthropic tool definition.
- `cache_last_user_message`: Add a breakpoint to the last cacheable block in the latest user message.
- `openai_prompt_cache_key`: Stable, low-cardinality key sent to OpenAI-compatible backends when supported.
- `openai_prompt_cache_retention`: Optional retention hint sent to OpenAI-compatible backends when supported.
- `debug_log_request_shape`: Log outbound request structure, lengths, and stable hashes without logging prompt text. Keep this enabled when diagnosing provider-side prompt cache misses.
- `relocate_system_prefix_range`: Optional `start..end` byte range. ferryllm moves the full system line intersecting this range into a user context block at the end of the message list, preserving the text while keeping stable prompt content first for provider prompt caches.
- `log_relocated_system_text`: Print the relocated text verbatim for diagnosis. This can expose prompt content; keep it disabled outside short investigations.
- `strip_system_line_prefixes`: Remove system lines that start with one of these prefixes and append them to trailing user context messages. Use this for transport metadata or other non-semantic boilerplate that should not affect cache prefix stability. A common Claude Code example is `x-anthropic-billing-header: cc_version=...; cc_entrypoint=...;`, where stripping the line keeps volatile metadata out of the stable system prefix while preserving the relocated instruction context.

This follows LiteLLM-style prompt caching practice: cache stable prefixes, do not mark every block, and avoid injecting Anthropic-only metadata into OpenAI-compatible outbound requests.

## Metrics

```toml
[metrics]
enabled = true
```

When enabled, ferryllm exposes Prometheus-style counters at `/metrics`:

```text
ferryllm_requests_total
ferryllm_requests_ok_total
ferryllm_requests_error_total
ferryllm_upstream_errors_total
```

## Provider Resilience

Non-streaming upstream requests can be retried with exponential backoff:

```toml
[server]
retry_attempts = 2
retry_backoff_ms = 100
circuit_breaker_failures = 5
circuit_breaker_cooldown_secs = 30
```

`retry_attempts` is the number of retries after the first attempt. The default is `0`, which preserves fail-fast behavior. Streaming requests are not retried because ferryllm may already have started sending tokens to the client.

When `circuit_breaker_failures` is set, ferryllm tracks consecutive failures per provider. Once the threshold is reached, that provider is short-circuited until `circuit_breaker_cooldown_secs` has elapsed. Fallback providers can still be tried while the primary provider circuit is open.

## Validation Rules

The server should fail fast during startup when:

- A provider referenced by a route does not exist.
- A provider secret environment variable is missing.
- Two providers have the same name.
- A route has neither a provider nor a strategy.
- A timeout, body size, or weight value is invalid.

## Security Notes

- Never commit real API keys.
- Prefer `api_key_env` over inline `api_key`.
- Redact authorization headers in logs.
- Redact request bodies by default.
- Log model names, provider names, status codes, and latency instead of full prompts.
