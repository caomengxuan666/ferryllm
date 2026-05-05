# Deployment Guide

This document describes the recommended deployment model for ferryllm as a standalone LLM middleware server.

The current repository provides example servers. A production release should expose a dedicated binary such as:

```bash
ferryllm serve --config /etc/ferryllm/config.toml
```

## Local Development

For local testing with the current example server:

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=info cargo run --example codexapis_server --features http
```

Health check:

```bash
curl http://127.0.0.1:3000/health
```

Claude Code through ferryllm:

```bash
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_BASE_URL=http://127.0.0.1:3000 \
claude --bare --print --model claude-opus-4-6 \
  "Reply with exactly one short word: pong"
```

## Environment Variables

Use environment variables for secrets.

Example:

```bash
export CODX_API_KEY="..."
export RUST_LOG=info
```

Do not store provider keys directly in committed config files.

## Logging

Recommended local logging:

```bash
RUST_LOG=info ferryllm serve --config ferryllm.toml
```

Recommended debugging:

```bash
RUST_LOG=debug ferryllm serve --config ferryllm.toml
```

Recommended deep protocol troubleshooting:

```bash
RUST_LOG=trace ferryllm serve --config ferryllm.toml
```

For local Claude Code against codexapis with full trace logs:

```bash
export CODX_API_KEY="..."
RUST_LOG=ferryllm=trace,tower_http=debug,reqwest=debug \
  cargo run --features http --bin ferryllm -- serve --config examples/config/codexapis.toml
```

Then in another shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3000
export ANTHROPIC_API_KEY=local-test-token
claude --model cc-gpt55
```

Production deployments should prefer JSON logs once supported:

```toml
[logging]
level = "info"
format = "json"
```

## Docker

Build the image locally:

```bash
docker build -t ferryllm:local .
```

Run with the example Codex APIs config:

```bash
docker run --rm \
  -p 3000:3000 \
  -e CODX_API_KEY="..." \
  ferryllm:local
```

Run with a custom config:

```bash
docker run --rm \
  -p 3000:3000 \
  -e CODX_API_KEY="..." \
  -v ./ferryllm.toml:/etc/ferryllm/config.toml:ro \
  ferryllm:local \
  serve --config /etc/ferryllm/config.toml
```

Recommended container behavior:

- Run as a non-root user.
- Expose only the configured listen port.
- Emit logs to stdout/stderr.
- Support graceful shutdown on `SIGTERM`.
- Avoid writing secrets to disk.

## systemd Plan

Example service:

```ini
[Unit]
Description=ferryllm LLM protocol middleware
After=network-online.target
Wants=network-online.target

[Service]
User=ferryllm
Group=ferryllm
Environment=RUST_LOG=info
EnvironmentFile=/etc/ferryllm/secrets.env
ExecStart=/usr/local/bin/ferryllm serve --config /etc/ferryllm/config.toml
Restart=always
RestartSec=3
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

## Kubernetes Plan

Recommended Kubernetes endpoints:

- Liveness probe: `/healthz`
- Readiness probe: `/readyz`
- Metrics endpoint: `/metrics` once implemented

Example probe shape:

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /readyz
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 10
```

Secrets should be injected through Kubernetes Secrets and referenced by environment variables.

## Operational Checklist

Before exposing ferryllm to users:

- Confirm provider keys are supplied through environment variables.
- Confirm model routes are explicit and reviewed.
- Confirm logs show incoming model and backend model.
- Confirm request body logging is disabled or redacted.
- Confirm health and readiness probes work.
- Confirm `/metrics` works when metrics are enabled.
- Confirm timeouts are configured.
- Confirm max body size is configured.
- Confirm API authentication is enabled for public deployments.
- Confirm rate limits are configured if serving untrusted clients.

## Public Relay Authentication

Enable API key authentication for public deployments:

```toml
[auth]
enabled = true
api_keys_env = "FERRYLLM_API_KEYS"
```

Then set keys at runtime:

```bash
export FERRYLLM_API_KEYS="key-one,key-two"
```

Clients should send one of:

```text
Authorization: Bearer key-one
x-api-key: key-one
```

## Security Notes

- Never log `Authorization`, `x-api-key`, or provider API keys.
- Do not log full prompts by default.
- Redact request bodies unless explicitly debugging in a safe environment.
- Use TLS at the ingress layer for production deployments.
- Restrict access to local-only deployments with firewall rules or `127.0.0.1` binding.
