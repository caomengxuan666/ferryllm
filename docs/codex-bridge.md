# Codex → Claude Bridge

This document explains how to use ferryllm as a bridge between OpenAI Codex and Anthropic Claude.

## Overview

OpenAI Codex uses the **Responses API** (`/v1/responses`) to communicate with LLM backends. ferryllm accepts these requests, translates them into the Anthropic Messages API format, forwards them to Claude, and translates the responses back.

```text
Codex (Responses API)
  → ferryllm /v1/responses
  → model route + rewrite (gpt-4o → claude-sonnet-4-20250514)
  → Anthropic Messages API
  → Claude response
  → translated back to Responses API format
  → Codex receives native Responses API response
```

## Quick Start

### 1. Start ferryllm

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
RUST_LOG=info cargo run --bin ferryllm -- serve --config examples/config/codex-bridge-claude.toml
```

### 2. Configure Codex

Point Codex at ferryllm instead of the default OpenAI API:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:3000"
export OPENAI_API_KEY="dummy"
```

Or configure in `~/.codex/config.toml`:

```toml
model = "gpt-4o"
api_base = "http://127.0.0.1:3000"
```

### 3. Run Codex

```bash
codex
```

Codex will send Requests API requests to ferryllm, which translates them to Claude.

## How It Works

### Request Flow

1. **Codex** sends a Responses API request:
   ```json
   POST /v1/responses
   {
     "model": "gpt-4o",
     "instructions": "You are a helpful coding assistant.",
     "input": [
       {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
     ],
     "tools": [...],
     "stream": true
   }
   ```

2. **ferryllm entry layer** (`entry/openai_responses.rs`) converts this to the unified IR:
   - `instructions` → `system`
   - `input` items → `messages`
   - `tools` → IR `Tool` objects
   - `reasoning.effort` → `ReasoningControl`

3. **Router** resolves the model name:
   - `gpt-4o` matches the exact route → rewrites to `claude-sonnet-4-20250514`

4. **Anthropic adapter** (`adapters/anthropic.rs`) translates IR to Anthropic wire format:
   - IR `ChatRequest` → `AnthropicRequest` at `/v1/messages`
   - System prompt, messages, tools, thinking config all translated

5. **Claude** responds (streaming or non-streaming)

6. **Anthropic adapter** parses the response back into IR `StreamEvent`/`ChatResponse`

7. **Responses entry layer** converts IR back to Responses API format:
   - `StreamEvent::ContentBlockDelta` → `response.output_text.delta`
   - `StreamEvent::MessageStop` → `response.completed`

8. **Codex** receives a native Responses API response

### Model Mapping

| Codex Model | Claude Model | Notes |
|---|---|---|
| `gpt-4o` | `claude-sonnet-4-20250514` | Fast, capable (default) |
| `gpt-4o-mini` | `claude-haiku-4-5-20251001` | Lightweight, fast |
| `o3` / `o3-mini` / `o4-mini` | `claude-sonnet-4-20250514` | Reasoning models |
| `gpt-*` (other) | `claude-sonnet-4-20250514` | Catch-all |
| `claude-*` | Passed through | No rewrite needed |

You can customize these mappings in the config file's `[[routes]]` section.

## Configuration Reference

The config file is at `examples/config/codex-bridge-claude.toml`. Key sections:

### Provider

```toml
[[providers]]
name = "claude"
type = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
```

- `type = "anthropic"` — uses the Anthropic Messages API adapter
- `base_url` — Anthropic API endpoint (or a compatible proxy)
- `api_key_env` — environment variable containing the API key

### Routes

```toml
[[routes]]
match = "gpt-4o"           # Model name pattern to match
match_type = "exact"        # "exact" or "prefix" (default)
provider = "claude"          # Which provider to route to
rewrite_model = "claude-sonnet-4-20250514"  # Model name sent to Claude
```

- **Exact matches** take priority over prefix matches
- **Longer prefixes** take priority over shorter ones
- **`rewrite_model`** is required for non-Claude model names (the Anthropic adapter only accepts models starting with `claude-`)

### Reasoning / Extended Thinking

```toml
[server]
default_reasoning_effort = "high"
```

When Codex sends `reasoning.effort`, ferryllm maps it to Anthropic's thinking config:
- `low` → 1024 token budget
- `medium` → 4096 token budget
- `high` → 8192 token budget
- `xhigh` → 16384 token budget

If Codex doesn't specify reasoning effort, `default_reasoning_effort` from the server config is used.

### Prompt Caching

```toml
[prompt_cache]
auto_inject_anthropic_cache_control = true
cache_system = true
cache_tools = true
cache_last_user_message = true
```

Automatically injects Anthropic `cache_control` breakpoints for cost savings on repeated prompts.

## Known Limitations

### Tool Call Results

ferryllm supports the `function_call_output` input item type used by Codex for tool results. These are automatically converted to IR `ToolResult` blocks and forwarded to Claude as tool results in the Anthropic Messages API format.

### Image Content

The Responses API may send `input_image` content parts. The current entry layer does not handle image content — if Codex sends images, the request may fail to parse. This is a known limitation for future improvement.

### Authentication

The Responses API uses `Authorization: Bearer <key>` headers. Since `auth.enabled = false` in the default config, any key is accepted. For production use, enable authentication:

```toml
[auth]
enabled = true
api_keys_env = "FERRYLLM_API_KEYS"
```

Set `FERRYLLM_API_KEYS` to a comma-separated list of accepted keys.

## Advanced: Hot-Reload API Keys

If you use cc-switch or another tool to manage API keys, configure `key_watch` for automatic key reloading:

```toml
[[providers]]
name = "claude"
type = "anthropic"
base_url = "https://api.anthropic.com"

[[providers.key_watch]]
file = "C:/Users/you/.claude/settings.json"
path = "env.ANTHROPIC_AUTH_TOKEN"
```

When the key file changes, ferryllm automatically reloads the API key without restarting.

## Troubleshooting

### Check ferryllm logs

```bash
RUST_LOG=debug cargo run --bin ferryllm -- serve --config examples/config/codex-bridge-claude.toml
```

Expected log output for a successful request:

```text
incoming request entry="responses" model=gpt-4o stream=true
resolved route entry="responses" display_model=gpt-4o backend_model=claude-sonnet-4-20250514 provider="claude"
sending streaming request provider="anthropic" model=claude-sonnet-4-20250514
```

### Verify the config

```bash
cargo run --bin ferryllm -- check-config --config examples/config/codex-bridge-claude.toml
```

### Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `model not supported` | Route missing for the model Codex sends | Add a route with `rewrite_model` starting with `claude-` |
| `401 Unauthorized` | Missing or invalid `ANTHROPIC_API_KEY` | Set the environment variable |
| `connection refused` | ferryllm not running or wrong port | Check `listen` address in config |
| Request timeout | Claude taking too long | Increase `request_timeout_secs` |

## Architecture Diagram

```text
┌──────────┐     Responses API      ┌───────────┐    Anthropic API    ┌───────────┐
│          │  POST /v1/responses     │           │  POST /v1/messages  │           │
│  Codex   │ ─────────────────────→  │  ferryllm │ ─────────────────→  │  Claude   │
│          │  {model: "gpt-4o",      │           │  {model: "claude-   │           │
│          │   instructions: "...",  │  ┌─────┐  │   sonnet-4-20250514"│           │
│          │   input: [...],         │  │ IR  │  │   messages: [...],  │           │
│          │   tools: [...]}         │  └─────┘  │   tools: [...]}     │           │
│          │                         │           │                     │           │
│          │  ← SSE stream events    │           │  ← SSE stream       │           │
│          │  response.output_text   │           │  content_block_delta│           │
│          │  .delta, ...            │           │  ...                │           │
└──────────┘                         └───────────┘                     └───────────┘
```
