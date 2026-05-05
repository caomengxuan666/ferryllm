# Compatibility Notes

This document tracks compatibility findings for translating between Anthropic-style clients, OpenAI-compatible backends, and ferryllm's internal IR. It is intentionally written as engineering notes: only proven gaps should become runtime changes.

## Current Position

ferryllm already has the core structures needed for Claude Code to talk to OpenAI-compatible models:

- Anthropic `tool_use` maps to IR `ContentBlock::ToolUse` with the original `id` preserved.
- Anthropic `tool_result` maps to IR `ContentBlock::ToolResult` with `tool_use_id` preserved as `id`.
- OpenAI backend requests split IR `ToolResult` blocks into `role: "tool"` messages with `tool_call_id`.
- Assistant tool calls are emitted as OpenAI `tool_calls` with the same ID used later by tool results.
- Empty assistant/user content is guarded so OpenAI-compatible providers do not receive `content: null`.

Because these basics are already implemented, compatibility work should prioritize regression tests and narrowly scoped fixes over speculative rewrites.

## LiteLLM Reference Findings

LiteLLM handles Anthropic/OpenAI compatibility with a few practical rules that are useful references for ferryllm.

### Tool IDs Are Authoritative

LiteLLM preserves Anthropic `tool_use.id` as the OpenAI tool call `id`, then uses the same value as the OpenAI `tool_call_id` for tool outputs. ferryllm follows the same principle through IR.

### Tool Results Need Matching Tool Calls

LiteLLM has explicit repair logic for histories where a tool output exists but the preceding assistant message lacks the corresponding `tool_calls`. It attempts to recover the call from cached previous responses or the declared tool list.

ferryllm should not add similar repair logic unless a real request trace proves the assistant call is missing after IR conversion. The preferred near-term action is to add regression tests that assert the normal Anthropic Claude Code shape remains valid after conversion.

### Empty Tool IDs Are Unsafe

LiteLLM skips or removes tool result messages with empty call IDs when possible, because providers cannot match them to a previous tool call. If ferryllm observes empty `tool_use_id` from a client, the safest behavior is likely to reject or drop that block with a clear debug log rather than forward an invalid tool message.

### Consecutive Tool Calls Must Stay Grouped

LiteLLM merges consecutive function call items into a single assistant message because Anthropic requires tool use blocks to be grouped before the following tool result blocks. ferryllm's IR can represent multiple `ToolUse` blocks in one assistant message, so tests should ensure the OpenAI backend request preserves that grouping as a single assistant message with multiple `tool_calls`.

### Tool Outputs Should Follow Tool Calls

Several providers require tool result messages to immediately follow the assistant tool call they answer. ferryllm should avoid inserting unrelated user or assistant text between an assistant `tool_calls` message and the corresponding `role: "tool"` messages.

### Streaming Is Provider-Specific

LiteLLM treats Anthropic message streaming and OpenAI chat streaming separately, then wraps chunks into the target protocol's SSE format. ferryllm follows the same broad design. The remaining compatibility risk is OpenAI streaming tool call deltas, where `id`, `name`, and `arguments` may arrive incrementally and must be accumulated before emitting complete Anthropic `tool_use` blocks.

## ferryllm Verification Checklist

Add regression tests before adding more runtime repair code:

- Anthropic request with assistant `tool_use` followed by user `tool_result` converts to OpenAI assistant `tool_calls` followed by `role: "tool"`.
- Multiple Anthropic `tool_use` blocks in one assistant message convert to one OpenAI assistant message with multiple `tool_calls`.
- Multiple Anthropic `tool_result` blocks convert to consecutive OpenAI `role: "tool"` messages with matching `tool_call_id` values.
- Empty text around tool calls does not serialize as `content: null`.
- Empty `tool_use_id` is handled deliberately, either by rejection or safe dropping with logs.
- OpenAI non-streaming tool calls convert back to Anthropic `tool_use` blocks with the original IDs.
- OpenAI streaming tool call deltas are accumulated into complete Anthropic `tool_use` blocks before the client receives them.

## Non-Goals For Now

- Do not add broad LiteLLM-style history repair unless real Claude Code traces show missing assistant tool calls after ferryllm conversion.
- Do not add a response/session cache until there is a demonstrated need for `previous_response_id`-style reconstruction.
- Do not normalize every provider-specific tool schema variant until a supported client emits it.
