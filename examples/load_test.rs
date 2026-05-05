//! Async Rust load tester for ferryllm.
//!
//! Example:
//!   cargo run --release --example load_test --features http -- --requests 10000 --concurrency 512

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::json;
use tokio::sync::Semaphore;

#[derive(Clone)]
struct Args {
    url: String,
    requests: usize,
    concurrency: usize,
    timeout_secs: u64,
}

#[derive(Debug)]
struct ResultSample {
    status: u16,
    latency_ms: f64,
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let client = Client::builder()
        .pool_max_idle_per_host(args.concurrency)
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()
        .expect("build client");
    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(args.requests);

    for _ in 0..args.requests {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("acquire concurrency permit");
        let client = client.clone();
        let url = args.url.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            request_once(&client, &url).await
        }));
    }

    let mut samples = Vec::with_capacity(args.requests);
    for task in tasks {
        samples.push(task.await.expect("join request task"));
    }

    print_summary(&args, started.elapsed(), &samples);
}

async fn request_once(client: &Client, url: &str) -> ResultSample {
    let body = json!({
        "model": "claude-load-test",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    });
    let started = Instant::now();
    let status = match client
        .post(url)
        .header("content-type", "application/json")
        .header("authorization", "Bearer load-test")
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let _ = resp.bytes().await;
            status
        }
        Err(_) => 0,
    };

    ResultSample {
        status,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn print_summary(args: &Args, elapsed: Duration, samples: &[ResultSample]) {
    let mut statuses = BTreeMap::new();
    let mut latencies = samples
        .iter()
        .map(|sample| sample.latency_ms)
        .collect::<Vec<_>>();
    latencies.sort_by(|a, b| a.total_cmp(b));

    for sample in samples {
        *statuses.entry(sample.status).or_insert(0usize) += 1;
    }

    let ok = statuses.get(&200).copied().unwrap_or(0);
    let errors = args.requests.saturating_sub(ok);
    let elapsed_secs = elapsed.as_secs_f64();
    println!("{{");
    println!("  \"requests\": {},", args.requests);
    println!("  \"concurrency\": {},", args.concurrency);
    println!("  \"elapsed_seconds\": {:.3},", elapsed_secs);
    println!(
        "  \"requests_per_second\": {:.2},",
        args.requests as f64 / elapsed_secs
    );
    println!("  \"ok\": {ok},");
    println!("  \"errors\": {errors},");
    println!("  \"statuses\": {:?},", statuses);
    println!("  \"latency_ms\": {{");
    println!(
        "    \"min\": {:.2},",
        latencies.first().copied().unwrap_or(0.0)
    );
    println!("    \"mean\": {:.2},", mean(&latencies));
    println!("    \"p50\": {:.2},", percentile(&latencies, 50.0));
    println!("    \"p95\": {:.2},", percentile(&latencies, 95.0));
    println!("    \"p99\": {:.2},", percentile(&latencies, 99.0));
    println!(
        "    \"max\": {:.2}",
        latencies.last().copied().unwrap_or(0.0)
    );
    println!("  }}");
    println!("}}");
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((pct / 100.0) * (values.len().saturating_sub(1)) as f64).round() as usize;
    values[index.min(values.len() - 1)]
}

fn parse_args() -> Args {
    let mut args = Args {
        url: "http://127.0.0.1:3000/v1/messages".into(),
        requests: 10_000,
        concurrency: 128,
        timeout_secs: 10,
    };
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--url" => args.url = iter.next().expect("--url value"),
            "--requests" => {
                args.requests = iter
                    .next()
                    .expect("--requests value")
                    .parse()
                    .expect("valid --requests")
            }
            "--concurrency" => {
                args.concurrency = iter
                    .next()
                    .expect("--concurrency value")
                    .parse()
                    .expect("valid --concurrency")
            }
            "--timeout" => {
                args.timeout_secs = iter
                    .next()
                    .expect("--timeout value")
                    .parse()
                    .expect("valid --timeout")
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }

    args
}

fn print_help() {
    println!("Usage: load_test [--url URL] [--requests N] [--concurrency N] [--timeout SECS]");
}
