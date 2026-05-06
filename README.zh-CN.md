# ferryllm

面向 OpenAI、Anthropic、Claude Code 和 OpenAI-compatible 后端的通用 LLM 协议中间层。

[![CI](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml/badge.svg)](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/ferryllm.svg)](https://crates.io/crates/ferryllm)
[![Docs.rs](https://docs.rs/ferryllm/badge.svg)](https://docs.rs/ferryllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ferryllm 是一个 Rust 编写的 LLM 协议网关。它把不同客户端协议统一成一套内部表示，再转发到不同 provider。你可以把它当作 Claude Code 本地代理、私有模型网关，或者可嵌入的 adapter 层。

## 它能做什么

- 提供 OpenAI-compatible 入口：`POST /v1/chat/completions`
- 提供 Anthropic-compatible 入口：`POST /v1/messages`
- 支持模型别名、精确路由和前缀路由
- 支持转发到 OpenAI-compatible 或 Anthropic 后端
- 保留工具调用和 SSE 流式行为
- 在清理 transport metadata 的同时保持 prompt-cache 前缀稳定
- 通过 IR 和 adapter 统一 reasoning control

## 为什么是 ferryllm

很多网关最终都会变成 `N x M` 矩阵：每一种客户端协议都要为每一种 provider 协议写一套适配。

ferryllm 采用 `N + M` 路由方式：

```text
客户端协议 -> ferryllm IR -> provider 协议
```

这样更适合下面这些场景：

- 把 Claude Code 放到稳定的后端之上
- 用一个网关同时服务多种客户端协议
- 让缓存行为保持可预测
- 新增 provider 时不需要重写所有客户端路径

## 快速开始

从 crates.io 安装：

```bash
cargo install ferryllm
```

从源码运行：

```bash
git clone https://github.com/caomengxuan666/ferryllm.git
cd ferryllm
cargo run --features http --bin ferryllm -- serve --config examples/config/codexapis.toml
```

设置 provider key 并启动：

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=info ferryllm serve --config examples/config/codexapis.toml
```

测试 Anthropic-compatible endpoint：

```bash
curl -s http://127.0.0.1:3000/v1/messages \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer local-test-token' \
  -d '{"model":"cc-gpt55","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}'
```

## Claude Code 代理场景

Claude Code 发送的是 Anthropic 格式请求。ferryllm 可以接收这类请求，重写模型名，再转发给 OpenAI-compatible 后端。

```text
Claude Code
  -> POST /v1/messages, model = claude-*
  -> ferryllm Anthropic entry
  -> unified IR
  -> route match: claude-
  -> rewrite backend model: gpt-5.4
  -> OpenAI-compatible backend
```

启动 ferryllm：

```bash
export CODX_API_KEY="your-api-key"
RUST_LOG=ferryllm=info,tower_http=info \
  ferryllm serve --config examples/config/codexapis.toml
```

让 Claude Code 指向 ferryllm：

```bash
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_BASE_URL=http://127.0.0.1:3000 \
claude --bare --print --model claude-opus-4-6 \
  "Reply with exactly one short word: pong"
```

预期输出：

```text
pong
```

更多 Claude Code 和 cc-switch 配置见 [docs/claude-code.md](docs/claude-code.md)。

## 配置

ferryllm 使用 TOML 配置，密钥通过环境变量注入。

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32
default_reasoning_effort = "medium"
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
type = "openai"
base_url = "https://codexapis.com"
api_key_env = "CODX_API_KEY"

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

不启动服务也可以检查配置：

```bash
ferryllm check-config --config examples/config/codexapis.toml
```

如果要让 OpenAI-compatible 上游走 Responses API，而不是 Chat Completions，
需要开启可选 feature，并把 provider type 改成 `openai_responses`：

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

示例见 [examples/config/codexapis-responses.toml](examples/config/codexapis-responses.toml)。

## 思考强度

在 TOML 里设置默认模型思考强度：

```toml
[server]
default_reasoning_effort = "medium"
```

可选值是 `none`、`low`、`medium`、`high`、`xhigh` 和 `x_high`。

这个默认值只会在客户端请求没有显式携带 reasoning 或 thinking 控制时生效。对现在的 Claude Code 场景来说，通过 TOML 改这个值，是控制转发到 OpenAI-compatible 后端的 `reasoning.effort` 的实际方式。

排查时可以开启 debug 日志，在 outbound request-shape 日志里看 `reasoning=effort=...`，确认 ferryllm 实际发给上游的值。

## 接口

| Endpoint | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages |
| `GET /health` | 简单健康检查 |
| `GET /healthz` | Kubernetes 风格 liveness check |
| `GET /readyz` | readiness check |
| `GET /metrics` | Prometheus 风格 metrics |

## Prompt Cache

ferryllm 会在清理 transport metadata 的同时保持 prompt-cache 前缀稳定。

开启 `prompt-observability` 后，ferryllm 会把 prompt-cache 使用情况写入日志和 `/metrics`。

对 Claude Code 场景，最重要的几个参数是：

- `relocate_system_prefix_range`
- `strip_system_line_prefixes`
- `openai_prompt_cache_key`
- `default_reasoning_effort`

详见 [docs/prompt-caching.md](docs/prompt-caching.md) 和 [docs/reasoning-control.md](docs/reasoning-control.md)。

## 项目结构

```text
src/
  adapter.rs        Adapter trait
  ir.rs             统一请求、响应、内容块、工具和流式事件类型
  router.rs         精确和前缀路由
  server.rs         Axum HTTP server
  config.rs         TOML 配置加载和校验
  entry/            客户端协议转换
  adapters/         后端 provider adapter
```

更多细节见 [docs/architecture.md](docs/architecture.md)。

## 压测

ferryllm 内置了一个本地 mock upstream 的压测工具：

```bash
MOCK_DELAY_MS=20 cargo run --example mock_openai_upstream --features http
cargo run --release --example load_test --features http -- \
  --preset mock-anthropic \
  --requests 10000 \
  --concurrency 512
```

更多信息见 [docs/load-testing.md](docs/load-testing.md)。

## 文档

- [English README](README.md)
- [Architecture](docs/architecture.md)
- [Claude Code setup](docs/claude-code.md)
- [Configuration](docs/configuration.md)
- [Compatibility notes](docs/compatibility.md)
- [Deployment](docs/deployment.md)
- [Load testing](docs/load-testing.md)
- [Prompt caching and token observability](docs/prompt-caching.md)
- [Reasoning control](docs/reasoning-control.md)

## 路线图

- 更多 provider adapter，包括 Gemini
- 支持加权和延迟感知 provider pool
- 配置热加载
- 更丰富的 Prometheus metrics 标签
- per-key quota 和 usage accounting hooks
- 打包好的 Docker 镜像和部署模板

## License

MIT. See [LICENSE](LICENSE).
