# Benchmarking

ferryllm includes a benchmark-style load tester similar to `redis-benchmark`.
It is meant to measure your own server, machine, and deployment settings without spending real provider tokens.

## Start A Mock Upstream

```bash
MOCK_DELAY_MS=20 cargo run --example mock_openai_upstream --features http
```

The mock listens on `127.0.0.1:4010` and returns a small OpenAI-compatible chat response.

## Start ferryllm

In another shell:

```bash
export MOCK_OPENAI_API_KEY=dummy
cargo run --bin ferryllm -- serve --config examples/config/mock-openai.toml
```

This starts ferryllm on `127.0.0.1:3000` and routes all models to the mock upstream.

## Run The Benchmark Tool

The Rust benchmark example is the main tool for load testing:

```bash
cargo run --release --example load_test --features http -- \
  --protocol anthropic \
  --url http://127.0.0.1:3000/v1/messages \
  --requests 10000 \
  --concurrency 512
```

Useful flags:

- `--protocol anthropic|openai`
- `--url`
- `--model`
- `--prompt`
- `--max-tokens`
- `--requests`
- `--concurrency`
- `--timeout`
- `--bearer`

Examples:

```bash
cargo run --release --example load_test --features http -- --protocol anthropic --requests 1000 --concurrency 128
cargo run --release --example load_test --features http -- --protocol openai --url http://127.0.0.1:3000/v1/chat/completions --requests 1000 --concurrency 128
cargo run --release --example load_test --features http -- --requests 5000 --concurrency 512 --timeout 20
```

The benchmark prints JSON-like output with:

- `requests_per_second`
- status code counts
- error count
- latency min/mean/p50/p95/p99/max

## Notes

- Do not use the benchmark against paid providers unless you intentionally want to spend tokens.
- Increase `MOCK_DELAY_MS` to simulate slower upstreams.
- Use `max_concurrent_requests` and `rate_limit_per_minute` in the config to validate `429` behavior.
- For public deployments, run this on hardware close to the real deployment target.
