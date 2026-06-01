use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use crate::adapter::AdapterError;
#[cfg(feature = "gemini")]
use crate::adapters::gemini::GeminiAdapter;
#[cfg(feature = "openai-responses")]
use crate::adapters::openai_responses::OpenaiResponsesAdapter;
use crate::adapters::{anthropic::AnthropicAdapter, openai::OpenaiAdapter};
use crate::ir::ReasoningEffort;
use crate::router::Router;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub prompt_cache: PromptCacheConfig,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub routes: Vec<RouteConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_body_limit_mb")]
    pub body_limit_mb: u64,
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
    #[serde(default)]
    pub rate_limit_per_minute: Option<u64>,
    #[serde(default)]
    pub retry_attempts: u32,
    #[serde(default = "default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,
    #[serde(default)]
    pub circuit_breaker_failures: Option<u64>,
    #[serde(default = "default_circuit_breaker_cooldown_secs")]
    pub circuit_breaker_cooldown_secs: u64,
    #[serde(default)]
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            request_timeout_secs: default_request_timeout_secs(),
            body_limit_mb: default_body_limit_mb(),
            max_concurrent_requests: None,
            rate_limit_per_minute: None,
            retry_attempts: 0,
            retry_backoff_ms: default_retry_backoff_ms(),
            circuit_breaker_failures: None,
            circuit_breaker_cooldown_secs: default_circuit_breaker_cooldown_secs(),
            default_reasoning_effort: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub api_keys_env: Option<String>,
    #[serde(default)]
    pub per_key_rate_limit_per_minute: Option<u64>,
    #[serde(default)]
    pub per_key_max_concurrent_requests: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct MetricsConfig {
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: default_metrics_enabled(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PromptCacheConfig {
    #[serde(default = "default_prompt_cache_auto_inject_anthropic_cache_control")]
    pub auto_inject_anthropic_cache_control: bool,
    #[serde(default = "default_prompt_cache_cache_system")]
    pub cache_system: bool,
    #[serde(default = "default_prompt_cache_cache_tools")]
    pub cache_tools: bool,
    #[serde(default = "default_prompt_cache_cache_last_user_message")]
    pub cache_last_user_message: bool,
    #[serde(default = "default_prompt_cache_openai_prompt_cache_key")]
    pub openai_prompt_cache_key: String,
    #[serde(default)]
    pub openai_prompt_cache_retention: Option<String>,
    #[serde(default = "default_prompt_cache_debug_log_request_shape")]
    pub debug_log_request_shape: bool,
    #[serde(default = "default_prompt_cache_relocate_system_prefix_range")]
    pub relocate_system_prefix_range: Option<String>,
    #[serde(default = "default_prompt_cache_log_relocated_system_text")]
    pub log_relocated_system_text: bool,
    #[serde(default = "default_prompt_cache_strip_system_line_prefixes")]
    pub strip_system_line_prefixes: Vec<String>,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            auto_inject_anthropic_cache_control:
                default_prompt_cache_auto_inject_anthropic_cache_control(),
            cache_system: default_prompt_cache_cache_system(),
            cache_tools: default_prompt_cache_cache_tools(),
            cache_last_user_message: default_prompt_cache_cache_last_user_message(),
            openai_prompt_cache_key: default_prompt_cache_openai_prompt_cache_key(),
            openai_prompt_cache_retention: None,
            debug_log_request_shape: default_prompt_cache_debug_log_request_shape(),
            relocate_system_prefix_range: default_prompt_cache_relocate_system_prefix_range(),
            log_relocated_system_text: default_prompt_cache_log_relocated_system_text(),
            strip_system_line_prefixes: default_prompt_cache_strip_system_line_prefixes(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    pub base_url: String,
    /// Direct API key value defined in the config file.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_key_url: Option<String>,
    #[serde(default)]
    pub api_key_file: Option<String>,
    /// List of key file sources to watch. When any file changes, the key is
    /// re-extracted using the `path` field (dotted JSON path, e.g. `OPENAI_API_KEY`
    /// or `env.ANTHROPIC_AUTH_TOKEN`). Supports JSON and TOML files.
    #[serde(default)]
    pub key_watch: Vec<KeyWatchConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct KeyWatchConfig {
    /// Absolute path to the file to watch.
    pub file: String,
    /// Dotted path to the API key value, e.g. `OPENAI_API_KEY` or `env.ANTHROPIC_AUTH_TOKEN`.
    pub path: String,
    /// Dotted path to the base URL value, e.g. `env.ANTHROPIC_BASE_URL`.
    #[serde(default)]
    pub url_path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Openai,
    #[serde(rename = "openai_responses")]
    #[cfg(feature = "openai-responses")]
    OpenaiResponses,
    #[cfg(feature = "gemini")]
    Gemini,
    Anthropic,
}

#[derive(Debug, Deserialize)]
pub struct RouteConfig {
    #[serde(rename = "match")]
    pub match_prefix: String,
    #[serde(default)]
    pub match_type: RouteMatchType,
    pub provider: String,
    #[serde(default)]
    pub fallback_providers: Vec<String>,
    #[serde(default)]
    pub rewrite_model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteMatchType {
    #[default]
    Prefix,
    Exact,
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let raw = fs::read_to_string(path).map_err(ConfigError::Read)?;
        toml::from_str(&raw).map_err(ConfigError::Parse)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.providers.is_empty() {
            return Err(ConfigError::Invalid(
                "at least one provider is required".into(),
            ));
        }

        let mut providers = HashSet::new();
        for provider in &self.providers {
            if provider.name.trim().is_empty() {
                return Err(ConfigError::Invalid("provider name cannot be empty".into()));
            }
            if !providers.insert(provider.name.as_str()) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate provider name '{}'",
                    provider.name
                )));
            }
            if provider.base_url.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "provider '{}' base_url cannot be empty",
                    provider.name
                )));
            }
            let key_sources = [
                provider.api_key.is_some(),
                provider.api_key_env.is_some(),
                provider.api_key_url.is_some(),
                provider.api_key_file.is_some(),
                !provider.key_watch.is_empty(),
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            if key_sources > 1 {
                return Err(ConfigError::Invalid(format!(
                    "provider '{}' must specify only one of: api_key, api_key_env, api_key_url, api_key_file, key_watch",
                    provider.name
                )));
            }
            if let Some(env) = &provider.api_key_env {
                if env.trim().is_empty() {
                    return Err(ConfigError::Invalid(format!(
                        "provider '{}' api_key_env cannot be empty",
                        provider.name
                    )));
                }
                if std::env::var(env).is_err() {
                    return Err(ConfigError::Invalid(format!(
                        "environment variable '{}' is required for provider '{}'",
                        env, provider.name
                    )));
                }
            }
            if let Some(file) = &provider.api_key_file {
                if file.trim().is_empty() {
                    return Err(ConfigError::Invalid(format!(
                        "provider '{}' api_key_file cannot be empty",
                        provider.name
                    )));
                }
                if !std::path::Path::new(file).exists() {
                    return Err(ConfigError::Invalid(format!(
                        "api_key_file '{}' for provider '{}' does not exist",
                        file, provider.name
                    )));
                }
            }
        }

        for route in &self.routes {
            if route.match_prefix.trim().is_empty() && route.match_prefix != "*" {
                return Err(ConfigError::Invalid("route match cannot be empty".into()));
            }
            if !providers.contains(route.provider.as_str()) {
                return Err(ConfigError::Invalid(format!(
                    "route '{}' references unknown provider '{}'",
                    route.match_prefix, route.provider
                )));
            }
            for fallback in &route.fallback_providers {
                if !providers.contains(fallback.as_str()) {
                    return Err(ConfigError::Invalid(format!(
                        "route '{}' references unknown fallback provider '{}'",
                        route.match_prefix, fallback
                    )));
                }
            }
        }

        if self.auth.enabled {
            let Some(env) = &self.auth.api_keys_env else {
                return Err(ConfigError::Invalid(
                    "auth.api_keys_env is required when auth.enabled is true".into(),
                ));
            };
            let keys = std::env::var(env).map_err(|_| {
                ConfigError::Invalid(format!(
                    "environment variable '{}' is required when auth is enabled",
                    env
                ))
            })?;
            if parse_csv(&keys).is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "environment variable '{}' must contain at least one API key",
                    env
                )));
            }
        }

        Ok(())
    }

    pub fn runtime_options(&self) -> Result<crate::server::ServerOptions, ConfigError> {
        self.validate()?;
        let auth_keys = if self.auth.enabled {
            let env = self.auth.api_keys_env.as_ref().ok_or_else(|| {
                ConfigError::Invalid(
                    "auth.api_keys_env is required when auth.enabled is true".into(),
                )
            })?;
            let raw = std::env::var(env).map_err(|_| {
                ConfigError::Invalid(format!(
                    "environment variable '{}' is required when auth is enabled",
                    env
                ))
            })?;
            parse_csv(&raw)
        } else {
            Vec::new()
        };

        Ok(crate::server::ServerOptions {
            request_timeout_secs: self.server.request_timeout_secs,
            body_limit_bytes: self.server.body_limit_mb.saturating_mul(1024 * 1024) as usize,
            max_concurrent_requests: self.server.max_concurrent_requests,
            rate_limit_per_minute: self.server.rate_limit_per_minute,
            retry_attempts: self.server.retry_attempts,
            retry_backoff_ms: self.server.retry_backoff_ms,
            circuit_breaker_failures: self.server.circuit_breaker_failures,
            circuit_breaker_cooldown_secs: self.server.circuit_breaker_cooldown_secs,
            default_reasoning_effort: self.server.default_reasoning_effort.clone(),
            auth_enabled: self.auth.enabled,
            auth_keys,
            per_key_rate_limit_per_minute: self.auth.per_key_rate_limit_per_minute,
            per_key_max_concurrent_requests: self.auth.per_key_max_concurrent_requests,
            metrics_enabled: self.metrics.enabled,
            prompt_cache: crate::server::PromptCacheOptions {
                auto_inject_anthropic_cache_control: self
                    .prompt_cache
                    .auto_inject_anthropic_cache_control,
                cache_system: self.prompt_cache.cache_system,
                cache_tools: self.prompt_cache.cache_tools,
                cache_last_user_message: self.prompt_cache.cache_last_user_message,
                openai_prompt_cache_key: self.prompt_cache.openai_prompt_cache_key.clone(),
                openai_prompt_cache_retention: self
                    .prompt_cache
                    .openai_prompt_cache_retention
                    .clone(),
                debug_log_request_shape: self.prompt_cache.debug_log_request_shape,
                relocate_system_prefix_range: self
                    .prompt_cache
                    .relocate_system_prefix_range
                    .as_deref()
                    .and_then(parse_byte_range),
                log_relocated_system_text: self.prompt_cache.log_relocated_system_text,
                strip_system_line_prefixes: self.prompt_cache.strip_system_line_prefixes.clone(),
            },
        })
    }

    pub fn build_router(&self) -> Result<Router, ConfigError> {
        self.validate()?;

        let mut router = Router::new();
        for provider in &self.providers {
            let api_key = if let Some(key) = &provider.api_key {
                key.clone()
            } else if let Some(env) = &provider.api_key_env {
                std::env::var(env).map_err(|_| {
                    ConfigError::Invalid(format!(
                        "environment variable '{}' is required for provider '{}'",
                        env, provider.name
                    ))
                })?
            } else if let Some(url) = &provider.api_key_url {
                fetch_api_key(url, &provider.name)?
            } else if let Some(file) = &provider.api_key_file {
                read_key_file(file, &provider.name)?
            } else if !provider.key_watch.is_empty() {
                resolve_key_from_watch(&provider.key_watch, &provider.name)?
            } else {
                // Auto-detect API key from cc switch config files.
                match cc_switch_resolve(&provider.provider_type) {
                    Some(r) => r.key,
                    None => {
                        return Err(ConfigError::Invalid(format!(
                            "provider '{}' has no key source and no cc switch config found",
                            provider.name
                        )));
                    }
                }
            };

            let key_prefix = &api_key[..api_key.len().min(8)];
            let key_source = if provider.api_key.is_some() {
                "direct"
            } else if provider.api_key_env.is_some() {
                "env"
            } else if provider.api_key_url.is_some() {
                "url"
            } else if provider.api_key_file.is_some() {
                "file"
            } else if !provider.key_watch.is_empty() {
                "key_watch"
            } else {
                "cc_switch_auto"
            };
            tracing::trace!(provider = %provider.name, key_source, key_prefix = %key_prefix, "api key loaded");

            match provider.provider_type {
                ProviderType::Openai => {
                    let adapter = Arc::new(OpenaiAdapter::new(provider.base_url.clone(), api_key));
                    router.register_adapter_as(&provider.name, adapter);
                }
                #[cfg(feature = "openai-responses")]
                ProviderType::OpenaiResponses => {
                    let adapter = Arc::new(OpenaiResponsesAdapter::new(
                        provider.base_url.clone(),
                        api_key,
                    ));
                    router.register_adapter_as(&provider.name, adapter);
                }
                #[cfg(feature = "gemini")]
                ProviderType::Gemini => {
                    let adapter = Arc::new(GeminiAdapter::new(provider.base_url.clone(), api_key));
                    router.register_adapter_as(&provider.name, adapter);
                }
                ProviderType::Anthropic => {
                    let adapter =
                        Arc::new(AnthropicAdapter::new(provider.base_url.clone(), api_key));
                    router.register_adapter_as(&provider.name, adapter);
                }
            }
        }

        for route in &self.routes {
            let prefix = if route.match_prefix == "*" {
                ""
            } else {
                route.match_prefix.as_str()
            };
            match route.match_type {
                RouteMatchType::Prefix => {
                    router.add_prefix_route_with_fallbacks(
                        prefix,
                        &route.provider,
                        route.rewrite_model.clone(),
                        route.fallback_providers.clone(),
                    );
                }
                RouteMatchType::Exact => {
                    router.add_exact_route_with_fallbacks(
                        prefix,
                        &route.provider,
                        route.rewrite_model.clone(),
                        route.fallback_providers.clone(),
                    );
                }
            }
        }

        Ok(router)
    }
}

fn read_key_file(path: &str, provider_name: &str) -> Result<String, ConfigError> {
    let key = std::fs::read_to_string(path).map_err(|e| {
        ConfigError::Invalid(format!(
            "failed to read api_key_file '{}' for provider '{}': {}",
            path, provider_name, e
        ))
    })?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(ConfigError::Invalid(format!(
            "api_key_file '{}' for provider '{}' is empty",
            path, provider_name
        )));
    }
    Ok(key)
}

/// Try each `key_watch` entry in order; return the first key found.
fn resolve_key_from_watch(
    watches: &[KeyWatchConfig],
    provider_name: &str,
) -> Result<String, ConfigError> {
    for watch in watches {
        match extract_key_from_file(&watch.file, &watch.path) {
            Ok(key) if !key.is_empty() => {
                tracing::trace!(file = %watch.file, path = %watch.path, key_prefix = %&key[..key.len().min(8)], "resolved api key from watched file");
                return Ok(key);
            }
            Ok(_) => {
                tracing::debug!(file = %watch.file, path = %watch.path, "key_watch file exists but key is empty");
            }
            Err(e) => {
                tracing::debug!(file = %watch.file, path = %watch.path, error = %e, "failed to extract key from key_watch file");
            }
        }
    }
    Err(ConfigError::Invalid(format!(
        "provider '{}': no key found in any key_watch file",
        provider_name
    )))
}

/// Try each `key_watch` entry in order; return the first base URL found.
/// Returns `None` if no `url_path` is configured or no value is found.
fn resolve_url_from_watch(watches: &[KeyWatchConfig], _provider_name: &str) -> Option<String> {
    for watch in watches {
        let url_path = watch.url_path.as_ref()?;
        match extract_key_from_file(&watch.file, url_path) {
            Ok(url) if !url.is_empty() => return Some(url),
            _ => continue,
        }
    }
    None
}

/// Read a JSON or TOML file and extract a value at the given dotted path.
fn extract_key_from_file(file_path: &str, dotted_path: &str) -> Result<String, ConfigError> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| ConfigError::Invalid(format!("failed to read '{}': {}", file_path, e)))?;

    let value: serde_json::Value = if file_path.ends_with(".json") {
        serde_json::from_str(&content).map_err(|e| {
            ConfigError::Invalid(format!("failed to parse JSON '{}': {}", file_path, e))
        })?
    } else if file_path.ends_with(".toml") {
        let toml_val: toml::Value = toml::from_str(&content).map_err(|e| {
            ConfigError::Invalid(format!("failed to parse TOML '{}': {}", file_path, e))
        })?;
        serde_json::to_value(toml_val).map_err(|e| {
            ConfigError::Invalid(format!("failed to convert TOML '{}': {}", file_path, e))
        })?
    } else {
        return Err(ConfigError::Invalid(format!(
            "unsupported file type '{}': expected .json or .toml",
            file_path
        )));
    };

    extract_json_path(&value, dotted_path).ok_or_else(|| {
        ConfigError::Invalid(format!(
            "path '{}' not found in '{}'",
            dotted_path, file_path
        ))
    })
}

/// Walk a `serde_json::Value` by a dotted path like `env.ANTHROPIC_AUTH_TOKEN`.
fn extract_json_path(value: &serde_json::Value, dotted_path: &str) -> Option<String> {
    let mut current = value;
    for segment in dotted_path.split('.') {
        current = current.get(segment)?;
    }
    current.as_str().map(String::from)
}

/// Auto-detected cc switch configuration for a provider type.
struct CcSwitchResolved {
    key: String,
    /// Files to watch for hot-reload: (file, key_path)
    watch_files: Vec<(String, String)>,
}

/// Resolve API key from well-known cc switch config files.
/// base_url is NOT read from cc switch — it stays in our config (the client
/// should point to ferryllm, not to the backend directly).
fn cc_switch_resolve(provider_type: &ProviderType) -> Option<CcSwitchResolved> {
    let home = dirs_home()?;

    match provider_type {
        ProviderType::Anthropic => {
            let file = format!("{}/.claude/settings.json", home);
            let key = extract_key_from_file(&file, "env.ANTHROPIC_AUTH_TOKEN").ok()?;
            Some(CcSwitchResolved {
                key,
                watch_files: vec![(file, "env.ANTHROPIC_AUTH_TOKEN".into())],
            })
        }
        ProviderType::Openai => {
            let file = format!("{}/.codex/auth.json", home);
            let key = extract_key_from_file(&file, "OPENAI_API_KEY").ok()?;
            Some(CcSwitchResolved {
                key,
                watch_files: vec![(file, "OPENAI_API_KEY".into())],
            })
        }
        #[cfg(feature = "openai-responses")]
        ProviderType::OpenaiResponses => {
            let file = format!("{}/.codex/auth.json", home);
            let key = extract_key_from_file(&file, "OPENAI_API_KEY").ok()?;
            Some(CcSwitchResolved {
                key,
                watch_files: vec![(file, "OPENAI_API_KEY".into())],
            })
        }
        #[cfg(feature = "gemini")]
        ProviderType::Gemini => None,
    }
}

fn dirs_home() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok()
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok()
    }
}

fn fetch_api_key(url: &str, provider_name: &str) -> Result<String, ConfigError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| {
            ConfigError::Invalid(format!(
                "failed to create HTTP client for provider '{}': {}",
                provider_name, e
            ))
        })?;
    let resp = client.get(url).send().map_err(|e| {
        ConfigError::Invalid(format!(
            "failed to fetch api_key_url for provider '{}': {}",
            provider_name, e
        ))
    })?;
    if !resp.status().is_success() {
        return Err(ConfigError::Invalid(format!(
            "api_key_url for provider '{}' returned status {}",
            provider_name,
            resp.status()
        )));
    }
    let key = resp.text().map_err(|e| {
        ConfigError::Invalid(format!(
            "failed to read api_key_url response for provider '{}': {}",
            provider_name, e
        ))
    })?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(ConfigError::Invalid(format!(
            "api_key_url for provider '{}' returned empty response",
            provider_name
        )));
    }
    Ok(key)
}

/// Opaque handle that keeps a file-system watcher alive.
/// Dropping this stops watching.
pub struct KeyFileWatcher {
    _watcher: notify::RecommendedWatcher,
}

/// Start file-system watchers for every provider that uses `api_key_file` or `key_watch`.
/// Returns a Vec of handles; dropping a handle stops that watcher.
pub fn start_key_watchers(config: &Config, router: &Router) -> Vec<KeyFileWatcher> {
    use notify::{
        Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    };
    use std::sync::mpsc;

    let mut watchers = Vec::new();

    for provider in &config.providers {
        let adapter = match router.get_adapter(&provider.name) {
            Some(a) => a,
            None => continue,
        };
        let provider_name = provider.name.clone();

        // Collect all (file, reload_fn) pairs for this provider.
        let mut watch_entries: Vec<(String, Box<dyn Fn() + Send + 'static>)> = Vec::new();

        // Case 1: api_key_file — simple file read
        if let Some(file_path) = &provider.api_key_file {
            let path = file_path.clone();
            let name = provider_name.clone();
            let a = Arc::clone(&adapter);
            watch_entries.push((
                path.clone(),
                Box::new(move || {
                    tracing::trace!(file = %path, "file changed, checking for api key");
                    match read_key_file(&path, &name) {
                        Ok(key) => {
                            a.update_api_key(key);
                            tracing::info!(provider = %name, "api key reloaded from file");
                        }
                        Err(e) => {
                            tracing::warn!(provider = %name, error = %e, "failed to reload api key")
                        }
                    }
                }),
            ));
        }

        // Case 2: key_watch — extract from JSON/TOML at dotted path
        if !provider.key_watch.is_empty() {
            let watches = provider.key_watch.clone();
            let name = provider_name.clone();
            let a = Arc::clone(&adapter);
            for w in &watches {
                let file = w.file.clone();
                let watches_inner = watches.clone();
                let name_inner = name.clone();
                let a_inner = Arc::clone(&a);
                let file_inner = file.clone();
                watch_entries.push((
                    file,
                    Box::new(move || {
                        tracing::trace!(file = %file_inner, "file changed, checking for api key");
                        // Reload API key
                        match resolve_key_from_watch(&watches_inner, &name_inner) {
                            Ok(key) => {
                                a_inner.update_api_key(key);
                                tracing::info!(provider = %name_inner, "api key reloaded from watched file");
                            }
                            Err(e) => {
                                tracing::debug!(provider = %name_inner, error = %e, "failed to reload api key from watched file");
                            }
                        }
                        // Reload base URL (if url_path is configured)
                        if let Some(url) = resolve_url_from_watch(&watches_inner, &name_inner) {
                            a_inner.update_base_url(url);
                            tracing::info!(provider = %name_inner, "base url reloaded from watched file");
                        }
                    }),
                ));
            }
        }

        // Case 3: auto-detect cc switch files (no explicit key source)
        let has_explicit_source = provider.api_key_env.is_some()
            || provider.api_key_url.is_some()
            || provider.api_key_file.is_some()
            || !provider.key_watch.is_empty();
        if !has_explicit_source {
            if let Some(cc) = cc_switch_resolve(&provider.provider_type) {
                let name = provider_name.clone();
                let a = Arc::clone(&adapter);
                for (file, key_path) in cc.watch_files {
                    let kp = key_path.clone();
                    let file_for_fn = file.clone();
                    let name_inner = name.clone();
                    let a_inner = Arc::clone(&a);
                    watch_entries.push((
                        file,
                        Box::new(move || {
                            tracing::trace!(file = %file_for_fn, "cc switch file changed, checking for api key");
                            match extract_key_from_file(&file_for_fn, &kp) {
                                Ok(key) if !key.is_empty() => {
                                    a_inner.update_api_key(key);
                                    tracing::info!(provider = %name_inner, "api key reloaded from cc switch");
                                }
                                Ok(_) => {
                                    tracing::debug!(provider = %name_inner, file = %file_for_fn, "cc switch file changed but no valid key found");
                                }
                                Err(e) => {
                                    tracing::debug!(provider = %name_inner, file = %file_for_fn, error = %e, "cc switch file changed but failed to extract key");
                                }
                            }
                        }),
                    ));
                }
            }
        }

        // Deduplicate by file path (same file may appear in both api_key_file and key_watch).
        let mut seen_files = std::collections::HashSet::new();
        for (file_path, reload_fn) in watch_entries {
            if !seen_files.insert(file_path.clone()) {
                continue;
            }

            let (tx, rx) = mpsc::channel::<notify::Result<Event>>();
            let mut watcher = match RecommendedWatcher::new(
                move |res: notify::Result<Event>| {
                    let _ = tx.send(res);
                },
                NotifyConfig::default(),
            ) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!(provider = %provider_name, error = %e, "failed to create file watcher");
                    break;
                }
            };

            if let Err(e) = watcher.watch(Path::new(&file_path), RecursiveMode::NonRecursive) {
                tracing::error!(provider = %provider_name, file = %file_path, error = %e, "failed to watch file");
                continue;
            }

            std::thread::spawn(move || {
                while let Ok(Ok(event)) = rx.recv() {
                    if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        reload_fn();
                    }
                }
            });

            tracing::info!(provider = %provider_name, file = %file_path, "key file watcher started");
            watchers.push(KeyFileWatcher { _watcher: watcher });
        }
    }

    watchers
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config: {0}")]
    Read(std::io::Error),
    #[error("failed to parse config: {0}")]
    Parse(toml::de::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
    #[error("failed to build router: {0}")]
    Router(#[from] AdapterError),
}

fn default_listen() -> String {
    "0.0.0.0:3000".into()
}

fn default_request_timeout_secs() -> u64 {
    120
}

fn default_body_limit_mb() -> u64 {
    32
}

fn default_retry_backoff_ms() -> u64 {
    100
}

fn default_circuit_breaker_cooldown_secs() -> u64 {
    30
}

fn default_log_level() -> String {
    "info".into()
}

fn default_log_format() -> String {
    "text".into()
}

fn default_metrics_enabled() -> bool {
    true
}

fn default_prompt_cache_auto_inject_anthropic_cache_control() -> bool {
    true
}

fn default_prompt_cache_cache_system() -> bool {
    true
}

fn default_prompt_cache_cache_tools() -> bool {
    true
}

fn default_prompt_cache_cache_last_user_message() -> bool {
    true
}

fn default_prompt_cache_openai_prompt_cache_key() -> String {
    "ferryllm".into()
}

fn default_prompt_cache_debug_log_request_shape() -> bool {
    true
}

fn default_prompt_cache_relocate_system_prefix_range() -> Option<String> {
    None
}

fn default_prompt_cache_log_relocated_system_text() -> bool {
    false
}

fn default_prompt_cache_strip_system_line_prefixes() -> Vec<String> {
    vec!["x-anthropic-billing-header:".into()]
}

fn parse_byte_range(value: &str) -> Option<(usize, usize)> {
    let (start, end) = value.split_once("..")?;
    let start = start.trim().parse::<usize>().ok()?;
    let end = end.trim().parse::<usize>().ok()?;
    (start < end).then_some((start, end))
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
