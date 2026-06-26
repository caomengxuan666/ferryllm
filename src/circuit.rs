//! Circuit breaker implementation.
//!
//! This module provides a circuit breaker pattern implementation for
//! handling backend failures gracefully.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    /// Circuit is closed, requests pass through normally.
    Closed,
    /// Circuit is open, requests are rejected immediately.
    Open,
    /// Circuit is half-open, allowing test requests.
    HalfOpen,
}

impl CircuitState {
    /// Whether requests can pass through.
    pub fn allows_request(&self) -> bool {
        matches!(self, CircuitState::Closed | CircuitState::HalfOpen)
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct BreakerConfig {
    /// Failure threshold before opening the circuit.
    pub failure_threshold: u32,
    /// Success threshold in half-open state to close the circuit.
    pub success_threshold: u32,
    /// Time to wait before transitioning from open to half-open.
    pub timeout: Duration,
    /// Half-open request budget (max concurrent test requests).
    pub half_open_budget: u32,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_budget: 1,
        }
    }
}

/// Circuit breaker for a single target.
pub struct CircuitBreaker {
    name: String,
    config: BreakerConfig,
    state: AtomicU8,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    half_open_count: AtomicU32,
    last_failure_time: RwLock<Option<Instant>>,
    last_state_change: RwLock<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(name: impl Into<String>, config: BreakerConfig) -> Self {
        Self {
            name: name.into(),
            config,
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            half_open_count: AtomicU32::new(0),
            last_failure_time: RwLock::new(None),
            last_state_change: RwLock::new(Instant::now()),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self::new(name, BreakerConfig::default())
    }

    /// Get current circuit state.
    pub fn state(&self) -> CircuitState {
        let state = self.state.load(Ordering::SeqCst);
        CircuitState::try_from(state).unwrap_or(CircuitState::Closed)
    }

    /// Check if a request can proceed.
    pub fn can_request(&self) -> bool {
        let state = self.state();
        if state.allows_request() {
            // For half-open state, check if we have budget
            if state == CircuitState::HalfOpen {
                self.half_open_count.load(Ordering::SeqCst) < self.config.half_open_budget
            } else {
                true
            }
        } else {
            // Check if timeout has elapsed
            let last_change = *self.last_state_change.read();
            if last_change.elapsed() >= self.config.timeout {
                // Transition to half-open
                self.to_half_open();
                true
            } else {
                false
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        let state = self.state();

        match state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if successes >= self.config.success_threshold {
                    self.to_closed();
                }
                self.half_open_count.fetch_sub(1, Ordering::SeqCst);
            }
            CircuitState::Open => {
                // Unexpected - should not happen
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        let state = self.state();

        *self.last_failure_time.write() = Some(Instant::now());

        match state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                if failures >= self.config.failure_threshold {
                    self.to_open();
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state opens the circuit
                self.to_open();
                self.half_open_count.fetch_sub(1, Ordering::SeqCst);
            }
            CircuitState::Open => {
                // Reset timeout
                *self.last_state_change.write() = Instant::now();
            }
        }
    }

    /// Record that a request was rejected (circuit open).
    pub fn record_rejection(&self) {
        // No state change, just tracking
    }

    /// Get circuit metrics.
    pub fn metrics(&self) -> CircuitMetrics {
        CircuitMetrics {
            name: self.name.clone(),
            state: self.state(),
            failure_count: self.failure_count.load(Ordering::SeqCst),
            success_count: self.success_count.load(Ordering::SeqCst),
            half_open_count: self.half_open_count.load(Ordering::SeqCst),
            time_until_half_open: self.time_until_half_open(),
        }
    }

    /// Transition to closed state.
    fn to_closed(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        self.half_open_count.store(0, Ordering::SeqCst);
        *self.last_state_change.write() = Instant::now();
    }

    /// Transition to open state.
    fn to_open(&self) {
        self.state.store(CircuitState::Open as u8, Ordering::SeqCst);
        *self.last_state_change.write() = Instant::now();
    }

    /// Transition to half-open state.
    fn to_half_open(&self) {
        self.state
            .store(CircuitState::HalfOpen as u8, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        self.half_open_count.store(0, Ordering::SeqCst);
        *self.last_state_change.write() = Instant::now();
    }

    /// Calculate time until circuit can transition to half-open.
    fn time_until_half_open(&self) -> Option<Duration> {
        if self.state() == CircuitState::Open {
            let elapsed = self.last_state_change.read().elapsed();
            if elapsed < self.config.timeout {
                Some(self.config.timeout - elapsed)
            } else {
                Some(Duration::ZERO)
            }
        } else {
            None
        }
    }

    /// Reset the circuit breaker.
    pub fn reset(&self) {
        self.to_closed();
    }
}

/// Circuit breaker metrics.
#[derive(Debug, Clone)]
pub struct CircuitMetrics {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub half_open_count: u32,
    pub time_until_half_open: Option<Duration>,
}

impl CircuitState {
    fn try_from(value: u8) -> Result<Self, ()> {
        match value {
            0 => Ok(CircuitState::Closed),
            1 => Ok(CircuitState::Open),
            2 => Ok(CircuitState::HalfOpen),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let breaker = CircuitBreaker::with_defaults("test");
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.can_request());
    }

    #[test]
    fn test_circuit_opens_on_failures() {
        let config = BreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test", config);

        for _ in 0..2 {
            breaker.record_failure();
        }
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.can_request());
    }

    #[test]
    fn test_circuit_closes_on_success_in_half_open() {
        let config = BreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_budget: 1,
        };
        let breaker = CircuitBreaker::new("test", config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // Should transition to half-open
        assert!(breaker.can_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes
        breaker.record_success();
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
}
