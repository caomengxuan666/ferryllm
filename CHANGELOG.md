# Changelog

All notable changes to ferryllm are documented here.

## 0.1.2 - 2026-05-06

### Added

- README and README.zh-CN now document the full standalone server example,
  including auth, resilience, and prompt-cache settings.
- README and README.zh-CN now explain stripping
  `x-anthropic-billing-header:` lines to move volatile Claude Code metadata such
  as `cc_version` and `cc_entrypoint` out of the stable system prefix.
- Release workflow permissions now include `id-token: write` for crates.io
  trusted publishing setups.

## 0.1.1 - 2026-05-05

### Fixed

- Correct prompt-cache hit-ratio metrics so cached/read tokens are not counted
  twice in the denominator.

### Added

- Prompt-cache observability metrics and structured request-shape diagnostics.
- Configurable Anthropic cache-control injection for stable system, tool, and
  user-message blocks.
- OpenAI-compatible prompt cache key and retention forwarding.
- Claude Code system-prefix cleanup options for moving volatile metadata out of
  the stable prompt prefix.
- Prompt caching documentation with Claude Code cache-stability guidance.

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
