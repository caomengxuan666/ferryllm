use std::net::SocketAddr;
use std::sync::Arc;

use ferryllm::config::Config;
use ferryllm::server::{build_router, AppState, Metrics};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());

    match command.as_str() {
        "serve" => {
            let config_path = parse_config_path(args)?;
            let config = Config::from_file(&config_path)?;
            init_logging(
                &config.logging.level,
                &config.logging.format,
                config.logging.ansi,
            );
            config.validate()?;

            let listen: SocketAddr = config.server.listen.parse()?;
            let router = config.build_router()?;
            let _watchers = ferryllm::config::start_key_watchers(&config, &router);
            let options = config.runtime_options()?;
            let state = Arc::new(AppState::new(router, options, Metrics::default()));
            let app = build_router(state);
            let listener = tokio::net::TcpListener::bind(listen).await?;

            info!(listen = %listen, "ferryllm server starting");
            axum::serve(listener, app).await?;
        }
        "check-config" => {
            let config_path = parse_config_path(args)?;
            let config = Config::from_file(&config_path)?;
            config.validate()?;
            println!("config ok: {config_path}");
        }
        "version" | "--version" | "-V" => {
            println!("ferryllm {}", env!("CARGO_PKG_VERSION"));
        }
        "help" | "--help" | "-h" => print_help(),
        other => {
            print_help();
            return Err(format!("unknown command '{other}'").into());
        }
    }

    Ok(())
}

fn parse_config_path(
    args: impl Iterator<Item = String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut config_path = None;
    let mut iter = args.peekable();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                config_path = iter.next();
            }
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }

    config_path.ok_or_else(|| "missing --config <path>".into())
}

fn init_logging(level: &str, format: &str, ansi: bool) {
    // Always try to construct a new EnvFilter with the specified level
    let filter = EnvFilter::new(level);
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi && format != "json");

    let result = if format == "json" {
        builder.json().try_init()
    } else {
        builder.try_init()
    };

    let _ = result;

    // Log after initialization so it actually prints
    tracing::info!(level = %level, format = %format, ansi, "logging initialized");
}

fn print_help() {
    println!("ferryllm - universal LLM protocol middleware");
    println!();
    println!("USAGE:");
    println!("  ferryllm serve --config <path>");
    println!("  ferryllm check-config --config <path>");
    println!("  ferryllm version");
}
