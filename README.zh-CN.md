# ferryllm

面向 OpenAI、Anthropic、Claude Code 和 OpenAI-compatible 后端的通用 LLM 协议中间层。

[![CI](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml/badge.svg)](https://github.com/caomengxuan666/ferryllm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/ferryllm.svg)](https://crates.io/crates/ferryllm)
[![Docs.rs](https://docs.rs/ferryllm/badge.svg)](https://docs.rs/ferryllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ferryllm 是一个 Rust 编写的 LLM 协议网关。它把不同客户端协议转换为统一内部表示，再转发到不同 provider。你可以把它当作 Claude Code 本地代理、内网模型网关，或者作为 Rust adapter 库嵌入到自己的项目里。

## 核心能力

- OpenAI-compatible 入口：`POST /v1/chat/completions`
- Anthropic-compatible 入口：`POST /v1/messages`
- OpenAI-compatible 和 Anthropic 后端 adapter
- 支持 Claude Code 请求转发到 OpenAI-compatible 后端
- 支持模型别名、前缀路由和模型名重写
- 支持 SSE 流式协议转换和工具调用
- 配置驱动独立服务：`ferryllm serve --config ferryllm.toml`
- 支持超时、body 限制、API key 鉴权、限流、并发限制、metrics、retry、fallback 和 circuit breaker
- `N + M` adapter 架构，方便继续接入新协议和新 provider

## 为什么需要 ferryllm

传统 LLM 网关很容易变成 `N x M` 矩阵：每一种客户端协议都要适配每一种后端协议。ferryllm 使用 `N + M` 设计：

```text
客户端协议 -> ferryllm IR -> provider 协议
```

新增一个后端 adapter 后，OpenAI 风格客户端、Anthropic 风格客户端和 Claude Code 都可以通过同一条 IR 链路访问它。

## 快速开始

从 crates.io 安装：

```bash
cargo install ferryllm
```

或者从源码运行：

```bash
git clone https://github.com/caomengxuan666/ferryllm.git
cd ferryllm
cargo run --features http --bin ferryllm -- serve --config examples/config/codexapis.toml
```

设置 OpenAI-compatible provider key：

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

## 让 Claude Code 使用 GPT-5.4 后端

Claude Code 发送 Anthropic 格式请求。ferryllm 可以接收这个请求，重写模型名，然后转发给 OpenAI-compatible 后端。

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

## 配置示例

ferryllm 使用 TOML 配置，密钥通过环境变量注入。

```toml
[server]
listen = "0.0.0.0:3000"
request_timeout_secs = 120
body_limit_mb = 32

[logging]
level = "info"
format = "text"

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
rewrite_model = "gpt-5.4"

[[routes]]
match = "claude-"
provider = "codexapis"
rewrite_model = "gpt-5.4"
```

检查配置：

```bash
ferryllm check-config --config examples/config/codexapis.toml
```

## 接口

| Endpoint | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages |
| `GET /health` | 简单健康检查 |
| `GET /healthz` | Kubernetes liveness check |
| `GET /readyz` | readiness check |
| `GET /metrics` | Prometheus 风格 metrics |

## 项目结构

```text
src/
  adapter.rs        Adapter trait
  ir.rs             统一请求、响应、内容块、工具和流式事件类型
  router.rs         exact/prefix 模型路由
  server.rs         Axum HTTP server
  config.rs         TOML 配置加载和校验
  entry/            客户端协议转换
  adapters/         后端 provider adapter
```

## 压测

本仓库内置一个类似 `redis-benchmark` 的本地压测工具：

```bash
MOCK_DELAY_MS=20 cargo run --example mock_openai_upstream --features http
cargo run --release --example load_test --features http -- \
  --preset mock-anthropic \
  --requests 10000 \
  --concurrency 512
```

更多信息见 [docs/load-testing.md](docs/load-testing.md)。

## Prompt Cache 观测

默认构建会开启 `prompt-observability`，ferryllm 会估算本地 prompt token，并把上游返回的 cache 命中字段写入日志和 `/metrics`：

```bash
cargo run --bin ferryllm -- serve --config examples/config/codexapis.toml
```

更多缓存命中率、cache-key 稳定性和 Claude Code 最佳实践见 [docs/prompt-caching.md](docs/prompt-caching.md)。

## Prompt Cache

在当前 Claude Code + Codex relay 组合下，只要 system prefix 稳定、并且把 transport metadata 从前缀中移走，实测缓存读命中率可以到接近 99.8%。

这个结果依赖上游 provider、prompt 结构和前缀稳定性，不是无条件保证。具体的规则和可调参数见 [docs/prompt-caching.md](docs/prompt-caching.md)。

## 后续路线

- Gemini 等更多 provider adapter
- 加权和延迟感知 provider pool
- 配置热加载
- 更细粒度 Prometheus metrics 标签
- per-key quota 和 usage accounting hooks
- Docker 镜像和部署模板

## License

MIT. See [LICENSE](LICENSE).
