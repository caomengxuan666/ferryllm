use std::collections::HashMap;
use std::sync::Arc;

use crate::adapter::{Adapter, AdapterError};
use tracing::{debug, trace, warn};

/// Result of model resolution: which adapter to use, and the (possibly
/// rewritten) model name to send to the backend.
pub struct ResolvedRoute {
    pub adapter: Arc<dyn Adapter>,
    pub provider: String,
    pub model: String,
    pub fallbacks: Vec<ResolvedFallback>,
}

pub struct ResolvedFallback {
    pub adapter: Arc<dyn Adapter>,
    pub provider: String,
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
    fallback_providers: Vec<String>,
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
        if let Some(route) =
            self.model_routes.iter_mut().rev().find(|route| {
                matches!(route.match_type, MatchType::Prefix) && route.pattern == prefix
            })
        {
            route.rewrite_model = Some(target_model.to_string());
            return;
        }
        self.add_prefix_route(prefix, "", Some(target_model.to_string()));
    }

    pub fn add_prefix_route(
        &mut self,
        prefix: &str,
        provider: &str,
        rewrite_model: Option<String>,
    ) {
        self.add_prefix_route_with_fallbacks(prefix, provider, rewrite_model, Vec::new());
    }

    pub fn add_prefix_route_with_fallbacks(
        &mut self,
        prefix: &str,
        provider: &str,
        rewrite_model: Option<String>,
        fallback_providers: Vec<String>,
    ) {
        self.model_routes.push(ModelRoute {
            pattern: prefix.to_string(),
            provider: provider.to_string(),
            fallback_providers,
            rewrite_model,
            match_type: MatchType::Prefix,
        });
    }

    pub fn add_exact_route(&mut self, model: &str, provider: &str, rewrite_model: Option<String>) {
        self.add_exact_route_with_fallbacks(model, provider, rewrite_model, Vec::new());
    }

    pub fn add_exact_route_with_fallbacks(
        &mut self,
        model: &str,
        provider: &str,
        rewrite_model: Option<String>,
        fallback_providers: Vec<String>,
    ) {
        self.model_routes.push(ModelRoute {
            pattern: model.to_string(),
            provider: provider.to_string(),
            fallback_providers,
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
            let exact_bonus = if matches!(route.match_type, MatchType::Exact) {
                1_000_000
            } else {
                0
            };
            -((exact_bonus + route.pattern.len()) as isize)
        });

        for route in &sorted {
            if route.matches(model) {
                if let Some(adapter) = self.adapters.get(&route.provider) {
                    let rewritten = route
                        .rewrite_model
                        .clone()
                        .unwrap_or_else(|| model.to_string());
                    let fallbacks = route
                        .fallback_providers
                        .iter()
                        .filter_map(|provider| {
                            self.adapters.get(provider).map(|adapter| ResolvedFallback {
                                adapter: Arc::clone(adapter),
                                provider: provider.clone(),
                                model: rewritten.clone(),
                            })
                        })
                        .collect();
                    debug!(model = %model, pattern = %route.pattern, match_type = ?route.match_type, provider = %route.provider, rewritten = %rewritten, "matched route");
                    return Ok(ResolvedRoute {
                        adapter: Arc::clone(adapter),
                        provider: route.provider.clone(),
                        model: rewritten,
                        fallbacks,
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
                    provider: adapter.provider_name().to_string(),
                    model: model.to_string(),
                    fallbacks: Vec::new(),
                });
            }
        }

        // 3. Fall back to default
        if let Some(ref default) = self.default_provider {
            if let Some(adapter) = self.adapters.get(default) {
                warn!(model = %model, provider = %default, "falling back to default provider");
                return Ok(ResolvedRoute {
                    adapter: Arc::clone(adapter),
                    provider: default.clone(),
                    model: model.to_string(),
                    fallbacks: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct MockAdapter {
        name: &'static str,
    }

    #[async_trait]
    impl Adapter for MockAdapter {
        fn provider_name(&self) -> &str {
            self.name
        }

        fn supports_model(&self, _model: &str) -> bool {
            false
        }

        async fn chat(
            &self,
            _request: &crate::ir::ChatRequest,
        ) -> Result<crate::ir::ChatResponse, AdapterError> {
            Err(AdapterError::BackendError("unused".into()))
        }

        async fn chat_stream(
            &self,
            _request: &crate::ir::ChatRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<crate::ir::StreamEvent, AdapterError>> + Send,
                >,
            >,
            AdapterError,
        > {
            Err(AdapterError::BackendError("unused".into()))
        }
    }

    #[test]
    fn resolve_includes_rewrite_and_fallback_providers() {
        let mut router = Router::new();
        router.register_adapter_as("primary", Arc::new(MockAdapter { name: "primary" }));
        router.register_adapter_as("backup-a", Arc::new(MockAdapter { name: "backup-a" }));
        router.register_adapter_as("backup-b", Arc::new(MockAdapter { name: "backup-b" }));
        router.add_exact_route_with_fallbacks(
            "cc-gpt55",
            "primary",
            Some("gpt-5.5".into()),
            vec!["backup-a".into(), "backup-b".into()],
        );

        let resolved = router.resolve("cc-gpt55").expect("resolve route");

        assert_eq!(resolved.provider, "primary");
        assert_eq!(resolved.model, "gpt-5.5");
        assert_eq!(resolved.fallbacks.len(), 2);
        assert_eq!(resolved.fallbacks[0].provider, "backup-a");
        assert_eq!(resolved.fallbacks[0].model, "gpt-5.5");
        assert_eq!(resolved.fallbacks[1].provider, "backup-b");
        assert_eq!(resolved.fallbacks[1].model, "gpt-5.5");
    }
}
