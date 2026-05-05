# Reasoning Control Design

This document proposes a first-class IR design for controlling model reasoning depth without affecting prompt-prefix cache stability.

## Goal

ferryllm should let a client express reasoning intensity once, then translate that intent into the correct provider-specific parameter.

The design must satisfy two constraints:

- Reasoning control must be real and enforceable at the upstream provider.
- Reasoning control must not alter the cached prompt prefix used for cache-key generation.

## Core Principle

Reasoning control is a semantic request attribute, not an opaque passthrough field.

It should be modeled in IR as a formal field, not stored in `extra`.

## Proposed IR Shape

```rust
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub tools: Vec<Tool>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<String>,
    pub reasoning: Option<ReasoningControl>,
    pub extra: HashMap<String, Value>,
}

pub struct ReasoningControl {
    pub effort: ReasoningEffort,
    pub budget_tokens: Option<u32>,
}

pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    XHigh,
}
```

`reasoning` is the canonical cross-provider control surface.

`extra` remains available for experimental or provider-specific fields that do not belong in the core model.

## Provider Mapping

The adapter layer should translate the IR field into the correct backend parameter.

### OpenAI-compatible backends

Map `ReasoningEffort` to the provider's reasoning control field.

Example target shape:

```json
{
  "reasoning": {
    "effort": "high"
  }
}
```

### Anthropic-compatible backends

Map `ReasoningControl` to `thinking` configuration.

Example target shape:

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4096
  }
}
```

The adapter may derive `budget_tokens` from `effort` when the client does not provide an explicit budget.

## Mapping Policy

Suggested default mapping:

| IR effort | OpenAI | Anthropic |
| --- | --- | --- |
| `none` | disabled | disabled |
| `low` | low | small budget |
| `medium` | medium | medium budget |
| `high` | high | large budget |
| `xhigh` | xhigh if supported | larger budget |

The exact token budgets should be configurable per provider and per model family.

## Cache Policy

Reasoning control must not participate in prompt-prefix cache construction.

Cache identity should be derived from stable prompt inputs only:

- system text
- message content
- tools
- route-selected prompt cache key

Reasoning control is a control-plane concern and should stay out of the cached prefix.

This preserves cache behavior when a caller changes reasoning intensity but keeps the same prompt.

## Request Flow

```text
client protocol
  -> entry adapter
  -> IR.ChatRequest.reasoning
  -> router.resolve(model)
  -> provider adapter
  -> provider-specific reasoning field
```

## Implementation Notes

- Add `reasoning: Option<ReasoningControl>` to IR.
- Extend entry adapters to parse client-side reasoning hints into IR.
- Extend provider adapters to emit provider-native reasoning parameters.
- Keep reasoning out of cache-key generation.
- If a backend does not support reasoning control, degrade gracefully to the provider default.

## Non-Goals

- Do not add a separate HTTP endpoint for reasoning control.
- Do not store reasoning in `extra` as the long-term API.
- Do not couple reasoning with prompt-cache prefix text.

## Outcome

This design gives ferryllm a single cross-provider reasoning control surface while preserving prompt-cache stability and keeping provider translation isolated in adapters.

## Operational Guidance

For Claude Code deployments, treat reasoning depth as a user-facing control and treat `MAX_THINKING_TOKENS` as the provider-side budget cap.

Recommended practice:

- Use a stable default effort for most sessions.
- Raise effort only when the task genuinely needs deeper reasoning.
- Keep reasoning out of cache-key construction.
- Let ferryllm normalize provider-specific inputs into `IR.reasoning`.

This keeps behavior predictable while avoiding unnecessary cache fragmentation.

In practice:

- Claude Code sets the thinking budget through its own configuration.
- ferryllm reads Anthropic `thinking` input when present.
- ferryllm can apply a stable `server.default_reasoning_effort` when the client omits reasoning controls.
- ferryllm maps that into a formal IR field.
- downstream adapters translate that field into the target provider's native parameter.

That preserves a single semantic control path without turning reasoning into opaque passthrough data.

Example server default:

```toml
[server]
default_reasoning_effort = "medium"
```

To inspect the translated provider parameter during troubleshooting, run with debug logging and look for `reasoning=effort=...` in the outbound request shape log.
