use std::collections::HashMap;
use std::sync::Arc;

use crate::adapter::{Adapter, AdapterError};
use tracing::{debug, trace, warn};

/// Result of model resolution: which adapter to use, and the (possibly
/// rewritten) model name to send to the backend.
pub struct ResolvedRoute {
    pub adapter: Arc<dyn Adapter>,
    pub model: String,
}

/// Manages a set of protocol adapters and routes requests to the right
/// backend based on the model name.
pub struct Router {
    adapters: HashMap<String, Arc<dyn Adapter>>,
    model_routes: Vec<ModelRoute>,
    default_provider: Option<String>,
}

#[derive(Clone, Debug)]
pub enum MatchType {
    Prefix,
    Exact,
}

#[derive(Clone, Debug)]
struct ModelRoute {
    pattern: String,
    provider: String,
    rewrite_model: Option<String>,
    match_type: MatchType,
}

impl Router {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            model_routes: Vec::new(),
            default_provider: None,
        }
    }

    pub fn register_adapter(&mut self, adapter: Arc<dyn Adapter>) {
        let name = adapter.provider_name().to_string();
        self.register_adapter_as(&name, adapter);
    }

    pub fn register_adapter_as(&mut self, name: &str, adapter: Arc<dyn Adapter>) {
        let name = name.to_string();
        self.adapters.insert(name.clone(), adapter);
        if self.default_provider.is_none() {
            self.default_provider = Some(name);
        }
    }

    pub fn add_route(&mut self, prefix: &str, provider: &str) {
        self.add_prefix_route(prefix, provider, None);
    }

    /// When the incoming model starts with `prefix`, rewrite it to
    /// `target_model` before forwarding to the backend.
    pub fn add_rewrite(&mut self, prefix: &str, target_model: &str) {
        if let Some(route) = self
            .model_routes
            .iter_mut()
            .rev()
            .find(|route| matches!(route.match_type, MatchType::Prefix) && route.pattern == prefix)
        {
            route.rewrite_model = Some(target_model.to_string());
            return;
        }
        self.add_prefix_route(prefix, "", Some(target_model.to_string()));
    }

    pub fn add_prefix_route(&mut self, prefix: &str, provider: &str, rewrite_model: Option<String>) {
        self.model_routes.push(ModelRoute {
            pattern: prefix.to_string(),
            provider: provider.to_string(),
            rewrite_model,
            match_type: MatchType::Prefix,
        });
    }

    pub fn add_exact_route(&mut self, model: &str, provider: &str, rewrite_model: Option<String>) {
        self.model_routes.push(ModelRoute {
            pattern: model.to_string(),
            provider: provider.to_string(),
            rewrite_model,
            match_type: MatchType::Exact,
        });
    }

    pub fn set_default_provider(&mut self, name: &str) {
        self.default_provider = Some(name.to_string());
    }

    /// Resolve a model name to an adapter, applying any rewrite rules.
    pub fn resolve(&self, model: &str) -> Result<ResolvedRoute, AdapterError> {
        trace!(model = %model, routes = self.model_routes.len(), "resolving model");
        // Sort routes by specificity so exact aliases beat prefixes, and "claude-" beats "".
        let mut sorted: Vec<_> = self.model_routes.iter().collect();
        sorted.sort_by_key(|route| {
            let exact_bonus = if matches!(route.match_type, MatchType::Exact) { 1_000_000 } else { 0 };
            -((exact_bonus + route.pattern.len()) as isize)
        });

        for route in &sorted {
            if route.matches(model) {
                if let Some(adapter) = self.adapters.get(&route.provider) {
                    let rewritten = route
                        .rewrite_model
                        .clone()
                        .unwrap_or_else(|| model.to_string());
                    debug!(model = %model, pattern = %route.pattern, match_type = ?route.match_type, provider = %route.provider, rewritten = %rewritten, "matched route");
                    return Ok(ResolvedRoute {
                        adapter: Arc::clone(adapter),
                        model: rewritten,
                    });
                }
            }
        }

        // 2. Ask each adapter
        for adapter in self.adapters.values() {
            if adapter.supports_model(model) {
                debug!(model = %model, provider = adapter.provider_name(), "adapter supports model");
                return Ok(ResolvedRoute {
                    adapter: Arc::clone(adapter),
                    model: model.to_string(),
                });
            }
        }

        // 3. Fall back to default
        if let Some(ref default) = self.default_provider {
            if let Some(adapter) = self.adapters.get(default) {
                warn!(model = %model, provider = %default, "falling back to default provider");
                return Ok(ResolvedRoute {
                    adapter: Arc::clone(adapter),
                    model: model.to_string(),
                });
            }
        }

        warn!(model = %model, "no adapter found for model");
        Err(AdapterError::BackendError(format!(
            "no adapter found for model '{}'",
            model
        )))
    }

}

impl ModelRoute {
    fn matches(&self, model: &str) -> bool {
        match self.match_type {
            MatchType::Prefix => self.pattern.is_empty() || model.starts_with(&self.pattern),
            MatchType::Exact => model == self.pattern,
        }
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}
