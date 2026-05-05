# Claude Code Setup

This document explains how to use Claude Code as an Anthropic-compatible client while sending the actual upstream request to an OpenAI-compatible backend model such as `gpt-5.4`.

## Key Idea

Claude Code does not need to know about `gpt-5.4` directly.

Claude Code sends Anthropic-format requests to ferryllm. ferryllm receives the request at `/v1/messages`, translates it into the internal representation, rewrites the backend model, and forwards the request to an OpenAI-compatible provider.

```text
Claude Code
  -> Anthropic API format
  -> ferryllm /v1/messages
  -> model route/rewrite
  -> OpenAI-compatible backend
  -> backend model gpt-5.4
```

## Verified Behavior

In a logged-out Claude Code environment, Claude Code fails without a configured API endpoint:

```text
Not logged in - Please run /login
```

When `ANTHROPIC_BASE_URL` points to ferryllm and a placeholder `ANTHROPIC_API_KEY` is provided, Claude Code can send requests through ferryllm successfully.

Observed server logs:

```text
incoming request entry="anthropic" model=claude-opus-4-6 stream=false
resolved route entry="anthropic" display_model=claude-opus-4-6 backend_model=gpt-5.4 provider="openai"
sending chat request provider="openai" model=gpt-5.4 stream=false
```

This proves that Claude Code is still the client, but the upstream inference model is `gpt-5.4`.

## Config-Driven Server

The repository includes `examples/config/codexapis.toml`, which demonstrates this flow with an OpenAI-compatible provider.

Start the proxy:

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=info cargo run --bin ferryllm -- serve --config examples/config/codexapis.toml
```

Run Claude Code through the proxy:

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

## Start Claude Code Normally

For day-to-day usage, start ferryllm first:

```bash
export CODX_API_KEY="your-api-key"
ferryllm serve --config examples/config/codexapis.toml
```

Then configure Claude Code to use ferryllm as its Anthropic-compatible API endpoint. After that, users can launch Claude Code normally:

```bash
claude
```

Claude Code will still send Anthropic-format requests, but ferryllm will rewrite the backend model to `gpt-5.4` according to the configured routes.

## Claude Code Settings

Claude Code can be configured through environment variables or through its settings file.

Environment variables:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:3000"
export ANTHROPIC_API_KEY="dummy"
```

Equivalent `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:3000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001"
  }
}
```

The model names above are client-side Anthropic model names. ferryllm rewrites them to the configured backend model, such as `gpt-5.4`.

## cc-switch Setup

cc-switch can manage Claude Code providers and write the corresponding Claude Code environment configuration.

Create a new Anthropic-compatible provider in cc-switch with these values:

```text
Name: ferryllm-gpt55
Provider type: Anthropic-compatible
Base URL: http://127.0.0.1:3000
API key / Auth token: dummy
Opus model: claude-opus-4-6
Sonnet model: claude-opus-4-6
Haiku model: claude-haiku-4-5-20251001
```

If cc-switch exposes environment variable fields directly, use:

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:3000
ANTHROPIC_API_KEY=dummy
ANTHROPIC_AUTH_TOKEN=dummy
ANTHROPIC_MODEL=claude-opus-4-6
ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-6
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-opus-4-6
ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5-20251001
```

Then switch to the `ferryllm-gpt55` provider in cc-switch and launch Claude Code normally:

```bash
claude
```

Expected ferryllm logs:

```text
incoming request entry="anthropic" model=claude-opus-4-6
resolved route entry="anthropic" display_model=claude-opus-4-6 backend_model=gpt-5.4 provider="openai"
sending streaming request provider="openai" model=gpt-5.4
```

If Claude Code says `Not logged in`, verify that cc-switch has written the provider environment into Claude Code settings and that `ANTHROPIC_BASE_URL` points to ferryllm.

## Streaming

ferryllm supports streaming for Claude Code.

Claude Code sends `stream: true` requests to `/v1/messages`. ferryllm forwards a streaming request to the configured backend and translates the upstream stream back into Anthropic-compatible SSE events.

Typical streamed response events look like:

```text
event: content_block_delta
data: {"delta":{"text":"1","type":"text_delta"},"index":0,"type":"content_block_delta"}
```

## Model Rewriting

The example server uses route rules equivalent to:

```text
if incoming model starts with "claude-":
  provider = openai
  backend model = gpt-5.4
```

Therefore, Claude Code may send `claude-opus-4-6`, `claude-haiku-4-5-20251001`, or another Anthropic model name, but the upstream request can still use `gpt-5.4`.

## Recommended Production Setup

For production, use the planned standalone server with a config file instead of hard-coded example code:

```toml
[server]
listen = "0.0.0.0:3000"

[logging]
level = "info"
format = "json"

[[providers]]
name = "codexapis"
type = "openai"
base_url = "https://codexapis.com"
api_key_env = "CODX_API_KEY"

[[routes]]
match = "claude-"
provider = "codexapis"
rewrite_model = "gpt-5.4"
```

Users can also define their own exact aliases:

```toml
[[routes]]
match = "cc-gpt55"
match_type = "exact"
provider = "codexapis"
rewrite_model = "gpt-5.4"
```

This lets clients request `cc-gpt55` while ferryllm sends `gpt-5.4` upstream.

Then point Claude Code to ferryllm:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:3000"
export ANTHROPIC_API_KEY="dummy"
```

## Notes

- The Claude Code UI or CLI may still show a Claude model name because that is the client-side model name.
- The authoritative backend model is the one logged by ferryllm as `backend_model` or adapter `model`.
- Do not put real provider keys in documentation, shell history, or committed config files. Use environment variables for secrets.
