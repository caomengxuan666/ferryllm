# Prompt Caching and Token Observability

ferryllm does not implement a local response cache. For Claude Code and other
tool-heavy clients, the safer and more useful optimization is to preserve
provider-side prompt caching and make cache behavior observable.

## Recommended Model

Use ferryllm as a cache-friendly protocol bridge:

```text
client request -> ferryllm IR -> provider-native request -> upstream prompt cache
```

The upstream provider or relay remains the authority for cache lookup,
eviction, billing, and model-specific tokenization. ferryllm should avoid
injecting volatile request-body fields that would make the upstream prompt
prefix unstable.

## Local Token Estimation

Local prompt token estimation is enabled by default through the
`prompt-observability` feature. It is designed for observability, not
billing-grade accounting.

```bash
cargo run --bin ferryllm -- serve --config examples/config/codexapis.toml
```

To build without local token estimation:

```bash
cargo run --no-default-features --features http --bin ferryllm -- \
  serve --config examples/config/codexapis.toml
```

Without `prompt-observability`, ferryllm still parses upstream usage and cache
fields when providers return them, but `estimated_prompt_tokens` is not emitted.

## Metrics

The `/metrics` endpoint includes cache-related counters:

```text
ferryllm_estimated_prompt_tokens_total
ferryllm_upstream_prompt_tokens_total
ferryllm_prompt_cached_tokens_total
ferryllm_prompt_cache_creation_tokens_total
ferryllm_prompt_cache_read_tokens_total
ferryllm_prompt_cache_hit_ratio
```

`ferryllm_prompt_cache_hit_ratio` is calculated as:

```text
prompt_cached_tokens_total / cache_prompt_tokens_total
```

For OpenAI-compatible usage, `prompt_tokens` already includes cached prompt
tokens, so cached/read tokens are counted in the numerator and are not added
again to the denominator.

For Anthropic-compatible usage, cache creation and cache read tokens are
reported separately from `input_tokens`, so `cache_prompt_tokens_total` uses
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.

This keeps the aggregate ratio meaningful for both Anthropic-compatible usage
and OpenAI-compatible `prompt_tokens_details.cached_tokens`.

This is a process-wide aggregate. For per-provider or per-model dashboards,
prefer structured logs until label cardinality policy is defined.

## Automatic Anthropic Cache Breakpoints

ferryllm follows the same practical pattern used by LiteLLM-style prompt
caching: preserve any client-provided `cache_control`, and when it is missing,
place `{"type":"ephemeral"}` on the last stable cacheable block.

The default policy is:

- cache top-level Anthropic `system` text,
- cache the last tool definition,
- cache the last cacheable block in the latest user message,
- never inject cache metadata into OpenAI-compatible outbound requests.

Configure it in TOML:

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

Set `auto_inject_anthropic_cache_control = false` if the client already manages
all Anthropic cache breakpoints explicitly.

For OpenAI-compatible backends, ferryllm sets a stable `prompt_cache_key` when
configured. Keep this value stable and low-cardinality. It should identify the
application or route family, not a specific conversation. ferryllm also
canonicalizes JSON object key order for tool schemas and tool-use inputs before
forwarding requests, which reduces accidental cache-key churn.

If a client places volatile context near the beginning of `system`, set
`relocate_system_prefix_range = "start..end"`. ferryllm moves the full line
intersecting that byte range into a user context block at the end of the message
list. This preserves the content but keeps stable prompt content first, which is
usually better for provider-side prompt cache prefix reuse. During short
investigations, `log_relocated_system_text = true` prints the moved text
verbatim; disable it after confirming the boundary.

For transport metadata or other non-semantic boilerplate, prefer
`strip_system_line_prefixes`. It removes matching system lines and appends them
as trailing user context, which is safer than byte slicing when a line boundary
matters.

## Request Shape Debugging

When `debug_log_request_shape = true`, ferryllm emits one outbound request-shape
log before sending the provider-native request. This is safe to keep on during
cache investigations because it records only structure, lengths, and stable
hashes:

```text
provider=openai
request_shape="model=gpt-5.5,stream=true,include_usage=true,messages=42,tools=12,..."
```

The shape log includes:

- backend model and stream mode,
- whether OpenAI streaming usage was requested,
- tool count and tool schema hash,
- message roles, content block types, lengths, and hashes,
- `prompt_cache_key` shape,
- full outbound request body hash.

It intentionally does not print prompt text, file contents, tool-result bodies,
image data, or tool argument bodies. Use it to compare ferryllm's outbound
prefix against a direct client/provider path. If only a small prefix such as
`13,312` tokens is cached, compare the first few message shapes and tool schema
hashes across adjacent requests; the first changing hash usually marks where the
upstream cache can no longer reuse the prefix.

## Logs

Successful requests emit a `prompt cache observation` log with fields such as:

```text
entry=anthropic
model=claude-opus-4-6
estimated_prompt_tokens=180000
upstream_prompt_tokens=181240
cached_tokens=164000
cache_creation_input_tokens=0
cache_read_input_tokens=164000
cache_hit_ratio=0.9048
```

For OpenAI-compatible upstreams, ferryllm reads
`usage.prompt_tokens_details.cached_tokens`.

For Anthropic-compatible upstreams, ferryllm reads
`usage.cache_creation_input_tokens` and `usage.cache_read_input_tokens`.

## Cache-Key Stability Guidelines

Provider caches usually depend on a stable prompt prefix. Some relays cache by
token prefix, while simpler relays may hash large request-body fragments. Keep
the serialized request stable where possible.

Recommended practices:

- Keep stable content first: system prompts, tool schemas, and static project
  context.
- Put dynamic content later: the current user message, tool results, fresh file
  excerpts, and timestamps.
- Do not add random request IDs, trace IDs, timestamps, or session markers to
  the upstream request body.
- Keep model rewrites stable. The same alias should map to the same backend
  model if you want cache reuse.
- Keep tool schemas stable. Avoid reordering or generating equivalent schemas
  differently between requests.
- Prefer omitting absent optional fields over alternating between `null`,
  empty strings, and missing fields.
- Avoid embedding volatile logging or diagnostics inside prompts.

## Claude Code Notes

Claude Code is a good prompt-cache workload because its system prompt, tool
definitions, and much of the workspace context are large and repeated. ferryllm
preserves the message order and does not add volatile fields to OpenAI-compatible
upstream request bodies, so provider-side cache hit rates should remain high
when the upstream relay supports prompt caching.

The cache will not necessarily be shared with direct Anthropic-format requests,
because ferryllm converts Anthropic client requests into the selected provider's
native wire format.
