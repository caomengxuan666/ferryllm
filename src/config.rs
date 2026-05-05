use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use crate::adapter::AdapterError;
use crate::adapters::{anthropic::AnthropicAdapter, openai::OpenaiAdapter};
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            request_timeout_secs: default_request_timeout_secs(),
            body_limit_mb: default_body_limit_mb(),
            max_concurrent_requests: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub api_keys_env: Option<String>,
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
    pub api_key_env: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Openai,
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
            if provider.api_key_env.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "provider '{}' api_key_env cannot be empty",
                    provider.name
                )));
            }
            if std::env::var(&provider.api_key_env).is_err() {
                return Err(ConfigError::Invalid(format!(
                    "environment variable '{}' is required for provider '{}'",
                    provider.api_key_env, provider.name
                )));
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
            auth_enabled: self.auth.enabled,
            auth_keys,
            metrics_enabled: self.metrics.enabled,
        })
    }

    pub fn build_router(&self) -> Result<Router, ConfigError> {
        self.validate()?;

        let mut router = Router::new();
        for provider in &self.providers {
            let api_key = std::env::var(&provider.api_key_env).map_err(|_| {
                ConfigError::Invalid(format!(
                    "environment variable '{}' is required for provider '{}'",
                    provider.api_key_env, provider.name
                ))
            })?;

            match provider.provider_type {
                ProviderType::Openai => {
                    let adapter = Arc::new(OpenaiAdapter::new(provider.base_url.clone(), api_key));
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

fn default_log_level() -> String {
    "info".into()
}

fn default_log_format() -> String {
    "text".into()
}

fn default_metrics_enabled() -> bool {
    true
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
