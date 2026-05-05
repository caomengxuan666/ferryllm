# Load Testing

This workflow measures ferryllm itself without spending real provider tokens. It uses a local mock OpenAI-compatible upstream.

## Start The Mock Upstream

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

## Run The Python Load Test

The Python script is dependency-free and useful for quick checks:

```bash
python3 scripts/load_test.py --requests 1000 --concurrency 32
```

Try a few concurrency levels:

```bash
python3 scripts/load_test.py --requests 200 --concurrency 1
python3 scripts/load_test.py --requests 1000 --concurrency 8
python3 scripts/load_test.py --requests 2000 --concurrency 32
python3 scripts/load_test.py --requests 5000 --concurrency 128
```

## Run The Rust Load Test

For higher QPS testing, use the async Rust example:

```bash
cargo run --release --example load_test --features http -- \
  --requests 10000 \
  --concurrency 512
```

Try higher concurrency levels:

```bash
cargo run --release --example load_test --features http -- --requests 10000 --concurrency 128
cargo run --release --example load_test --features http -- --requests 20000 --concurrency 512
cargo run --release --example load_test --features http -- --requests 50000 --concurrency 1024 --timeout 20
```

The scripts print JSON-like output with:

- `requests_per_second`
- status code counts
- error count
- latency min/mean/p50/p95/p99/max

## Notes

- Do not use this script against paid providers unless you intentionally want to spend tokens.
- Increase `MOCK_DELAY_MS` to simulate slower upstreams.
- Use `max_concurrent_requests` and `rate_limit_per_minute` in the config to validate `429` behavior.
- For public deployments, run this on hardware close to the real deployment target.
