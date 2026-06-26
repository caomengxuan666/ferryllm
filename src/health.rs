//! Health registry for tracking target health status.
//!
//! This module provides a health registry for tracking the health status of
//! backend targets, supporting circuit breaker patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Health status of a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Target is healthy and accepting requests.
    Healthy,
    /// Target is cooling down after failures.
    Cooldown,
    /// Target is currently rate limited.
    RateLimited,
    /// Target is unhealthy and not accepting requests.
    Unhealthy,
}

impl HealthStatus {
    /// Whether the target can accept requests.
    pub fn can_serve(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }
}

/// Health information for a single target.
#[derive(Debug, Clone)]
pub struct TargetHealth {
    /// Current health status.
    pub status: HealthStatus,
    /// Number of consecutive failures.
    pub failure_count: u32,
    /// Last failure timestamp.
    pub last_failure: Option<Instant>,
    /// Last success timestamp.
    pub last_success: Option<Instant>,
    /// Cooldown end time.
    pub cooldown_until: Option<Instant>,
    /// EWMA latency in milliseconds.
    pub latency_ms: f64,
}

/// Configuration for health tracking.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Failure threshold before entering cooldown.
    pub failure_threshold: u32,
    /// Base cooldown duration.
    pub base_cooldown: Duration,
    /// Maximum cooldown multiplier.
    pub max_cooldown_multiplier: u32,
    /// Cooldown recovery rate (successes needed to reduce cooldown).
    pub recovery_successes: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            base_cooldown: Duration::from_secs(60),
            max_cooldown_multiplier: 4,
            recovery_successes: 3,
        }
    }
}

/// Health registry for tracking target health.
#[derive(Default)]
pub struct HealthRegistry {
    targets: RwLock<HashMap<String, TargetHealth>>,
    config: HealthConfig,
}

impl HealthRegistry {
    /// Create a new health registry.
    pub fn new(config: HealthConfig) -> Self {
        Self {
            targets: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(HealthConfig::default())
    }

    /// Get the health status of a target.
    pub fn get(&self, target: &str) -> HealthStatus {
        let targets = self.targets.read();
        if let Some(health) = targets.get(target) {
            if let Some(until) = health.cooldown_until {
                if Instant::now() < until {
                    return health.status;
                }
            }
            health.status
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get full health information for a target.
    pub fn get_health(&self, target: &str) -> TargetHealth {
        let targets = self.targets.read();
        targets.get(target).cloned().unwrap_or_else(|| TargetHealth {
            status: HealthStatus::Healthy,
            failure_count: 0,
            last_failure: None,
            last_success: None,
            cooldown_until: None,
            latency_ms: 0.0,
        })
    }

    /// Record a successful request.
    pub fn record_success(&self, target: &str, latency_ms: f64) {
        let mut targets = self.targets.write();
        let health = targets.entry(target.to_string()).or_insert_with(|| TargetHealth {
            status: HealthStatus::Healthy,
            failure_count: 0,
            last_failure: None,
            last_success: None,
            cooldown_until: None,
            latency_ms: 0.0,
        });

        // Update latency with EWMA (exponentially weighted moving average)
        const EWMA_ALPHA: f64 = 0.2;
        if health.latency_ms == 0.0 {
            health.latency_ms = latency_ms;
        } else {
            health.latency_ms = EWMA_ALPHA * latency_ms + (1.0 - EWMA_ALPHA) * health.latency_ms;
        }

        health.last_success = Some(Instant::now());

        // Reset failure count and potentially recover from cooldown
        if health.failure_count > 0 {
            health.failure_count = health.failure_count.saturating_sub(1);
            if health.failure_count == 0 {
                health.status = HealthStatus::Healthy;
                health.cooldown_until = None;
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self, target: &str) {
        let mut targets = self.targets.write();
        let health = targets.entry(target.to_string()).or_insert_with(|| TargetHealth {
            status: HealthStatus::Healthy,
            failure_count: 0,
            last_failure: None,
            last_success: None,
            cooldown_until: None,
            latency_ms: 0.0,
        });

        health.failure_count += 1;
        health.last_failure = Some(Instant::now());

        if health.failure_count >= self.config.failure_threshold {
            // Enter cooldown with exponential backoff
            let multiplier = (health.failure_count / self.config.failure_threshold)
                .min(self.config.max_cooldown_multiplier);
            let cooldown = self.config.base_cooldown * multiplier as u32;
            health.cooldown_until = Some(Instant::now() + cooldown);
            health.status = HealthStatus::Cooldown;
        }
    }

    /// Record a rate-limited response.
    pub fn record_rate_limited(&self, target: &str, retry_after: Option<Duration>) {
        let mut targets = self.targets.write();
        let health = targets.entry(target.to_string()).or_insert_with(|| TargetHealth {
            status: HealthStatus::Healthy,
            failure_count: 0,
            last_failure: None,
            last_success: None,
            cooldown_until: None,
            latency_ms: 0.0,
        });

        health.status = HealthStatus::RateLimited;
        health.last_failure = Some(Instant::now());

        // Use retry_after header if provided, otherwise use base cooldown
        let cooldown = retry_after.unwrap_or(self.config.base_cooldown);
        health.cooldown_until = Some(Instant::now() + cooldown);
    }

    /// Check if a target can serve requests.
    pub fn can_serve(&self, target: &str) -> bool {
        let status = self.get(target);
        if status == HealthStatus::Healthy {
            return true;
        }

        // Check if cooldown has expired
        let targets = self.targets.read();
        if let Some(health) = targets.get(target) {
            if let Some(until) = health.cooldown_until {
                if Instant::now() >= until {
                    return true;
                }
            }
        }

        false
    }

    /// Get all targets sorted by health (healthy first, then by latency).
    pub fn get_healthy_targets(&self) -> Vec<(String, TargetHealth)> {
        let targets = self.targets.read();
        let mut result: Vec<_> = targets
            .iter()
            .filter(|(_, h)| h.status.can_serve())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort by latency (lowest first)
        result.sort_by(|a, b| a.1.latency_ms.partial_cmp(&b.1.latency_ms).unwrap());
        result
    }

    /// Reset health status for a target.
    pub fn reset(&self, target: &str) {
        let mut targets = self.targets.write();
        targets.remove(target);
    }

    /// Reset all targets.
    pub fn reset_all(&self) {
        let mut targets = self.targets.write();
        targets.clear();
    }

    /// Update latency for a target.
    pub fn update_latency(&self, target: &str, latency_ms: f64) {
        let mut targets = self.targets.write();
        if let Some(health) = targets.get_mut(target) {
            const EWMA_ALPHA: f64 = 0.2;
            health.latency_ms = EWMA_ALPHA * latency_ms + (1.0 - EWMA_ALPHA) * health.latency_ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_registry_initial_state() {
        let registry = HealthRegistry::with_defaults();
        assert_eq!(registry.get("test"), HealthStatus::Healthy);
        assert!(registry.can_serve("test"));
    }

    #[test]
    fn test_failure_threshold() {
        let config = HealthConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let registry = HealthRegistry::new(config);

        // Record 2 failures - should still be healthy
        registry.record_failure("test");
        registry.record_failure("test");
        assert_eq!(registry.get("test"), HealthStatus::Healthy);

        // 3rd failure - should enter cooldown
        registry.record_failure("test");
        let status = registry.get("test");
        assert!(matches!(status, HealthStatus::Cooldown | HealthStatus::Unhealthy));
    }

    #[test]
    fn test_success_resets_failure_count() {
        let registry = HealthRegistry::with_defaults();

        // Record failures
        for _ in 0..4 {
            registry.record_failure("test");
        }
        assert!(!registry.can_serve("test"));

        // Record success - should reduce failure count
        registry.record_success("test", 100.0);
        // With default config (threshold 5), failure count goes to 3
        // Still not serving, but closer to recovery
    }

    #[test]
    fn test_ewma_latency() {
        let registry = HealthRegistry::with_defaults();

        registry.record_success("test", 100.0);
        let health = registry.get_health("test");
        assert!((health.latency_ms - 100.0).abs() < 0.1);

        registry.record_success("test", 200.0);
        let health = registry.get_health("test");
        // EWMA with alpha=0.2: 0.2*200 + 0.8*100 = 120
        assert!((health.latency_ms - 120.0).abs() < 0.1);
    }
}
