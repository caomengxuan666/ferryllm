//! ferryllm-benchmark style async load tester.
//!
//! Example:
//!   cargo run --release --example load_test --features http -- --requests 10000 --concurrency 512

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

#[derive(Clone)]
struct Args {
    preset: String,
    url: String,
    protocol: Protocol,
    model: String,
    prompt: String,
    max_tokens: u32,
    requests: usize,
    concurrency: usize,
    timeout_secs: u64,
    bearer: Option<String>,
}

#[derive(Clone, Copy)]
enum Protocol {
    Anthropic,
    Openai,
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
        let args = args.clone();
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            request_once(&client, &args).await
        }));
    }

    let mut samples = Vec::with_capacity(args.requests);
    for task in tasks {
        samples.push(task.await.expect("join request task"));
    }

    print_summary(&args, started.elapsed(), &samples);
}

async fn request_once(client: &Client, args: &Args) -> ResultSample {
    let body = request_body(args);
    let started = Instant::now();
    let mut builder = client
        .post(&args.url)
        .header("content-type", "application/json")
        .json(&body);
    if let Some(token) = &args.bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }

    let status = match builder.send().await {
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

fn request_body(args: &Args) -> Value {
    match args.protocol {
        Protocol::Anthropic => json!({
            "model": args.model,
            "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": args.prompt}],
        }),
        Protocol::Openai => json!({
            "model": args.model,
            "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": args.prompt}],
        }),
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
    println!("  \"preset\": \"{}\",", args.preset);
    println!("  \"url\": \"{}\",", args.url);
    println!("  \"protocol\": \"{}\",", args.protocol.as_str());
    println!("  \"model\": \"{}\",", args.model);
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

impl Protocol {
    fn as_str(self) -> &'static str {
        match self {
            Protocol::Anthropic => "anthropic",
            Protocol::Openai => "openai",
        }
    }
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
        preset: "custom".into(),
        url: "http://127.0.0.1:3000/v1/messages".into(),
        protocol: Protocol::Anthropic,
        model: "claude-load-test".into(),
        prompt: "ping".into(),
        max_tokens: 16,
        requests: 10_000,
        concurrency: 128,
        timeout_secs: 10,
        bearer: None,
    };
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--preset" => apply_preset(&mut args, &iter.next().expect("--preset value")),
            "--url" => args.url = iter.next().expect("--url value"),
            "--protocol" => args.protocol = parse_protocol(&iter.next().expect("--protocol value")),
            "--model" => args.model = iter.next().expect("--model value"),
            "--prompt" => args.prompt = iter.next().expect("--prompt value"),
            "--max-tokens" => args.max_tokens = parse_next(&mut iter, "--max-tokens"),
            "--requests" => args.requests = parse_next(&mut iter, "--requests"),
            "--concurrency" => args.concurrency = parse_next(&mut iter, "--concurrency"),
            "--timeout" => args.timeout_secs = parse_next(&mut iter, "--timeout"),
            "--bearer" => args.bearer = Some(iter.next().expect("--bearer value")),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }

    args
}

fn apply_preset(args: &mut Args, preset: &str) {
    args.preset = preset.into();
    match preset {
        "mock-anthropic" | "anthropic-messages" => {
            args.protocol = Protocol::Anthropic;
            args.url = "http://127.0.0.1:3000/v1/messages".into();
            args.model = "claude-load-test".into();
        }
        "mock-openai" | "openai-chat" => {
            args.protocol = Protocol::Openai;
            args.url = "http://127.0.0.1:3000/v1/chat/completions".into();
            args.model = "gpt-load-test".into();
        }
        other => panic!("unknown preset {other}"),
    }
}

fn parse_next<T: std::str::FromStr>(iter: &mut impl Iterator<Item = String>, name: &str) -> T {
    iter.next()
        .unwrap_or_else(|| panic!("{name} value"))
        .parse()
        .unwrap_or_else(|_| panic!("valid {name}"))
}

fn parse_protocol(value: &str) -> Protocol {
    match value {
        "anthropic" => Protocol::Anthropic,
        "openai" => Protocol::Openai,
        other => panic!("unknown protocol {other}; expected anthropic or openai"),
    }
}

fn print_help() {
    println!("Usage: load_test [OPTIONS]");
    println!();
    println!("Options:");
    println!(
        "  --preset NAME             mock-anthropic|mock-openai|anthropic-messages|openai-chat"
    );
    println!("  --url URL                 Target URL (default: http://127.0.0.1:3000/v1/messages)");
    println!("  --protocol anthropic|openai  Request body shape (default: anthropic)");
    println!("  --model MODEL             Model name (default: claude-load-test)");
    println!("  --prompt TEXT             Prompt text (default: ping)");
    println!("  --max-tokens N            max_tokens value (default: 16)");
    println!("  --requests N              Total request count (default: 10000)");
    println!("  --concurrency N           In-flight request count (default: 128)");
    println!("  --timeout SECS            Per-request timeout (default: 10)");
    println!("  --bearer TOKEN            Optional Authorization bearer token");
}
