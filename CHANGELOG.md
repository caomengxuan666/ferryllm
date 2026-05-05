# Changelog

All notable changes to ferryllm are documented here.

## 0.1.0 - 2026-05-05

Initial public release.

### Added

- Unified internal representation for chat requests, responses, content blocks, tool calls, usage, and stream events.
- OpenAI-compatible client entrypoint at `/v1/chat/completions`.
- Anthropic-compatible client entrypoint at `/v1/messages`.
- OpenAI-compatible backend adapter.
- Anthropic backend adapter.
- Config-driven standalone binary with `serve`, `check-config`, and `version`.
- Exact and prefix model routing with model rewriting.
- Claude Code to OpenAI-compatible backend flow.
- SSE streaming translation with accumulated tool-call JSON.
- API key authentication, global and per-key rate limits, concurrency limits, metrics, retry, fallback, and circuit breaker support.
- Dockerfile, local mock upstream, and load-test examples.
- English and Chinese README files.
