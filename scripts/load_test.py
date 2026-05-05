#!/usr/bin/env python3
"""Small standard-library load test for ferryllm.

This is intended for local mock-upstream testing, not for real paid providers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request


def request_once(url: str, timeout: float) -> tuple[int, float]:
    body = json.dumps(
        {
            "model": "claude-load-test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json", "authorization": "Bearer load-test"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    except Exception:
        status = 0
    return status, (time.perf_counter() - started) * 1000.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test ferryllm with a mock upstream")
    parser.add_argument("--url", default="http://127.0.0.1:3000/v1/messages")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    started = time.perf_counter()
    statuses: dict[int, int] = {}
    latencies: list[float] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request_once, args.url, args.timeout) for _ in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            status, latency_ms = future.result()
            statuses[status] = statuses.get(status, 0) + 1
            latencies.append(latency_ms)

    elapsed = time.perf_counter() - started
    ok = statuses.get(200, 0)
    errors = args.requests - ok
    print(json.dumps(
        {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "elapsed_seconds": round(elapsed, 3),
            "requests_per_second": round(args.requests / elapsed, 2) if elapsed else 0,
            "ok": ok,
            "errors": errors,
            "statuses": statuses,
            "latency_ms": {
                "min": round(min(latencies), 2) if latencies else 0,
                "mean": round(statistics.mean(latencies), 2) if latencies else 0,
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
                "max": round(max(latencies), 2) if latencies else 0,
            },
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
