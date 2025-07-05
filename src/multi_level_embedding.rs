//! Multi-level embedding generation for contextual and hierarchical semantic search
//!
//! This module implements NodeSpace's sophisticated multi-level embedding architecture:
//! 1. Individual Embedding - Just node content (baseline)
//! 2. Contextual Embedding - Enhanced with relationship context (parent/siblings/mentions)
//! 3. Hierarchical Embedding - Full path context from root to node

use crate::error::NLPError;
use crate::utils::metrics::Timer;

use nodespace_core_types::{
    ContextStrategy, EmbeddingGenerationMetrics, MultiLevelEmbeddings, NodeContext,
};

use nodespace_core_types::Node;
use std::collections::HashMap;

/// Status information about sibling context availability for enhanced embeddings
#[derive(Debug, Clone)]
struct SiblingContextStatus {
    /// Whether any sibling context is available
    pub has_siblings: bool,
    /// Total number of sibling nodes
    pub total_siblings: usize,
    /// Context weight multiplier based on sibling availability
    pub context_weight: f32,
}

/// Multi-level embedding generator that creates contextual and hierarchical embeddings
#[derive(Default)]
pub struct MultiLevelEmbeddingGenerator {
    /// Cache for generated context strings to avoid regeneration
    context_cache: HashMap<String, String>,
}

impl MultiLevelEmbeddingGenerator {
    /// Create a new multi-level embedding generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate contextual embedding with relationship context
    pub async fn generate_contextual_embedding(
        &mut self,
        node: &Node,
        context: &NodeContext,
        base_generator: &dyn EmbeddingProvider,
    ) -> Result<Vec<f32>, NLPError> {
        let _timer = Timer::new("contextual_embedding_generation");

        // Monitor sibling context availability for enhanced embeddings
        let sibling_context_status = self.analyze_sibling_context(context);
        if sibling_context_status.has_siblings {
            tracing::info!(
                "✅ Enhanced sibling context available - rich contextual embeddings enabled"
            );
            tracing::debug!(
                "Sibling context: total_siblings={}, weight={:.2}",
                sibling_context_status.total_siblings,
                sibling_context_status.context_weight
            );
        }

        // Generate contextual text based on strategy
        let contextual_text = match &context.strategy {
            ContextStrategy::RuleBased => self.generate_rule_based_context(node, context)?,
            ContextStrategy::Phi4Enhanced => {
                tracing::warn!(
                    "Phi4Enhanced strategy not yet implemented, falling back to RuleBased"
                );
                self.generate_rule_based_context(node, context)?
            }
            ContextStrategy::Adaptive => {
                tracing::warn!("Adaptive strategy not yet implemented, falling back to RuleBased");
                self.generate_rule_based_context(node, context)?
            }
        };

        tracing::debug!(
            "Generated contextual text (length: {})",
            contextual_text.len()
        );

        // Generate embedding for the contextual text
        base_generator.generate_embedding(&contextual_text).await
    }

    /// Generate hierarchical embedding with full path context from root
    pub async fn generate_hierarchical_embedding(
        &mut self,
        node: &Node,
        path: &[Node],
        base_generator: &dyn EmbeddingProvider,
    ) -> Result<Vec<f32>, NLPError> {
        let _timer = Timer::new("hierarchical_embedding_generation");

        // Build hierarchical context from root to current node
        let hierarchical_text = self.generate_hierarchical_context(node, path)?;

        tracing::debug!(
            "Generated hierarchical text (length: {}, path_depth: {})",
            hierarchical_text.len(),
            path.len()
        );

        // Generate embedding for the hierarchical text
        base_generator.generate_embedding(&hierarchical_text).await
    }

    /// Generate all embedding levels for a node
    pub async fn generate_all_embeddings(
        &mut self,
        node: &Node,
        context: &NodeContext,
        path: &[Node],
        base_generator: &dyn EmbeddingProvider,
    ) -> Result<MultiLevelEmbeddings, NLPError> {
        let total_timer = Timer::new("multi_level_embedding_generation");
        let mut metrics = EmbeddingGenerationMetrics::default();

        // Generate individual embedding (baseline)
        let individual_timer = std::time::Instant::now();
        let node_text = extract_node_text(node)?;
        let individual = base_generator.generate_embedding(&node_text).await?;
        metrics.individual_time_ms = individual_timer.elapsed().as_millis() as u64;

        // Initialize multi-level embeddings
        let mut embeddings = MultiLevelEmbeddings::new(individual, context.strategy.clone());

        // Generate contextual embedding if context is available
        if has_context(context) {
            let contextual_timer = std::time::Instant::now();
            match self
                .generate_contextual_embedding(node, context, base_generator)
                .await
            {
                Ok(contextual) => {
                    metrics.contextual_time_ms =
                        Some(contextual_timer.elapsed().as_millis() as u64);
                    embeddings = embeddings.with_contextual(contextual);
                }
                Err(e) => {
                    tracing::warn!("Failed to generate contextual embedding: {}", e);
                }
            }
        }

        // Generate hierarchical embedding if path is available
        if !path.is_empty() {
            let hierarchical_timer = std::time::Instant::now();
            match self
                .generate_hierarchical_embedding(node, path, base_generator)
                .await
            {
                Ok(hierarchical) => {
                    metrics.hierarchical_time_ms =
                        Some(hierarchical_timer.elapsed().as_millis() as u64);
                    metrics.path_depth = Some(path.len());
                    embeddings = embeddings.with_hierarchical(hierarchical);
                }
                Err(e) => {
                    tracing::warn!("Failed to generate hierarchical embedding: {}", e);
                }
            }
        }

        // Finalize metrics
        metrics.total_time_ms = total_timer.elapsed().as_millis() as u64;
        embeddings = embeddings.with_metrics(metrics);

        Ok(embeddings)
    }

    /// Generate rule-based context by combining parent, sibling, and mention information
    fn generate_rule_based_context(
        &mut self,
        node: &Node,
        context: &NodeContext,
    ) -> Result<String, NLPError> {
        let mut context_parts = Vec::new();

        // Add node content as base
        let node_text = extract_node_text(node)?;
        context_parts.push(format!("Content: {}", node_text));

        // Add parent context
        if let Some(parent) = &context.parent {
            let parent_text = extract_node_text(parent)?;
            context_parts.push(format!("Parent: {}", truncate_text(&parent_text, 200)));
        }

        // Add collective sibling context
        if !context.siblings.is_empty() {
            let sibling_texts: Vec<String> = context.siblings
                .iter()
                .take(3) // Limit to avoid context explosion
                .filter_map(|sibling| extract_node_text(sibling).ok())
                .map(|text| truncate_text(&text, 100))
                .collect();

            if !sibling_texts.is_empty() {
                context_parts.push(format!("Siblings: {}", sibling_texts.join("; ")));
                tracing::debug!("✅ Collective sibling context added: {} siblings", sibling_texts.len());
            }
        }

        // Add mention context (nodes that reference this node)
        if !context.mentions.is_empty() {
            let mention_texts: Vec<String> = context
                .mentions
                .iter()
                .take(3) // Limit to 3 mentions to avoid context explosion
                .filter_map(|mention| extract_node_text(mention).ok())
                .map(|text| truncate_text(&text, 100))
                .collect();

            if !mention_texts.is_empty() {
                context_parts.push(format!("Mentions: {}", mention_texts.join("; ")));
            }
        }

        // Add related nodes context (by topic or similarity)
        if !context.related_nodes.is_empty() {
            let related_texts: Vec<String> = context
                .related_nodes
                .iter()
                .take(2) // Limit to 2 related nodes
                .filter_map(|related| extract_node_text(related).ok())
                .map(|text| truncate_text(&text, 100))
                .collect();

            if !related_texts.is_empty() {
                context_parts.push(format!("Related: {}", related_texts.join("; ")));
            }
        }

        // Combine all context parts
        let full_context = context_parts.join("\n");

        // Truncate if too long (embedding models have token limits)
        let final_context = if full_context.len() > 2000 {
            truncate_text(&full_context, 2000)
        } else {
            full_context
        };

        Ok(final_context)
    }

    /// Generate hierarchical context by traversing from root to current node
    fn generate_hierarchical_context(
        &mut self,
        node: &Node,
        path: &[Node],
    ) -> Result<String, NLPError> {
        let mut hierarchical_parts = Vec::new();

        // Add path context from root to parent
        for (depth, path_node) in path.iter().enumerate() {
            let node_text = extract_node_text(path_node)?;
            let prefix = match depth {
                0 => "Root",
                1 => "Level 1",
                2 => "Level 2",
                3 => "Level 3",
                _ => "Deep Level",
            };

            // Truncate path nodes to avoid context explosion
            let truncated_text = truncate_text(&node_text, 150);
            hierarchical_parts.push(format!("{}: {}", prefix, truncated_text));
        }

        // Add current node
        let node_text = extract_node_text(node)?;
        hierarchical_parts.push(format!("Current: {}", node_text));

        // Combine hierarchical context
        let full_hierarchical = hierarchical_parts.join("\n");

        // Truncate if necessary
        let final_hierarchical = if full_hierarchical.len() > 2500 {
            truncate_text(&full_hierarchical, 2500)
        } else {
            full_hierarchical
        };

        Ok(final_hierarchical)
    }

    /// Clear the context cache
    pub fn clear_cache(&mut self) {
        self.context_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.context_cache.len(), self.context_cache.capacity())
    }

    /// Analyze sibling context availability and quality
    fn analyze_sibling_context(&self, context: &NodeContext) -> SiblingContextStatus {
        let has_siblings = !context.siblings.is_empty();
        let total_siblings = context.siblings.len();
        
        let context_weight = if has_siblings {
            1.1 + (total_siblings.min(3) as f32 * 0.1) // Weight increases with more siblings
        } else {
            1.0 // Isolated node (normal weight)
        };

        SiblingContextStatus {
            has_siblings,
            total_siblings,
            context_weight,
        }
    }
}

/// Trait for providing base embedding generation functionality
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for the given text
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError>;
}

/// Extract text content from a Node's JSON content
fn extract_node_text(node: &Node) -> Result<String, NLPError> {
    // Try different common content patterns
    if let Some(text) = node.content.get("text").and_then(|v| v.as_str()) {
        return Ok(text.to_string());
    }

    if let Some(content) = node.content.get("content").and_then(|v| v.as_str()) {
        return Ok(content.to_string());
    }

    if let Some(title) = node.content.get("title").and_then(|v| v.as_str()) {
        return Ok(title.to_string());
    }

    if let Some(name) = node.content.get("name").and_then(|v| v.as_str()) {
        return Ok(name.to_string());
    }

    // Fallback: convert entire content to string
    Ok(node.content.to_string())
}

/// Truncate text to specified length with ellipsis
fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        format!("{}...", &text[..max_length.saturating_sub(3)])
    }
}

/// Check if context has any meaningful information
fn has_context(context: &NodeContext) -> bool {
    context.parent.is_some()
        || !context.siblings.is_empty()
        || !context.mentions.is_empty()
        || !context.related_nodes.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodespace_core_types::Node;
    use serde_json::json;

    /// Mock embedding provider for testing
    struct MockEmbeddingProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
            // Generate a deterministic mock embedding based on text length
            let length = text.len();
            Ok(vec![length as f32 / 100.0; 384])
        }
    }

    #[test]
    fn test_extract_node_text() {
        let node = Node::new("text".to_string(), json!({"text": "Hello, world!"}));
        let extracted = extract_node_text(&node).unwrap();
        assert_eq!(extracted, "Hello, world!");

        let node = Node::new("content".to_string(), json!({"content": "Different field"}));
        let extracted = extract_node_text(&node).unwrap();
        assert_eq!(extracted, "Different field");

        let node = Node::new("title".to_string(), json!({"title": "Node Title"}));
        let extracted = extract_node_text(&node).unwrap();
        assert_eq!(extracted, "Node Title");
    }

    #[test]
    fn test_truncate_text() {
        let short_text = "Short";
        assert_eq!(truncate_text(short_text, 10), "Short");

        let long_text = "This is a very long text that should be truncated";
        let truncated = truncate_text(long_text, 20);
        assert_eq!(truncated.len(), 20);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_has_context() {
        let empty_context = NodeContext::default();
        assert!(!has_context(&empty_context));

        let context_with_parent = NodeContext::default().with_parent(Node::new(
            "text".to_string(),
            json!({"text": "Parent node"}),
        ));
        assert!(has_context(&context_with_parent));

        let context_with_mentions = NodeContext::default().with_mentions(vec![Node::new(
            "text".to_string(),
            json!({"text": "Mention"}),
        )]);
        assert!(has_context(&context_with_mentions));
    }

    #[tokio::test]
    async fn test_multi_level_embedding_generation() {
        let mut generator = MultiLevelEmbeddingGenerator::new();
        let provider = MockEmbeddingProvider;

        let node = Node::new("text".to_string(), json!({"text": "Current node content"}));
        let parent = Node::new("text".to_string(), json!({"text": "Parent node content"}));
        let path = vec![parent.clone()];

        let context = NodeContext::default()
            .with_parent(parent)
            .with_mentions(vec![Node::new(
                "text".to_string(),
                json!({"text": "Reference node"}),
            )]);

        let embeddings = generator
            .generate_all_embeddings(&node, &context, &path, &provider)
            .await
            .unwrap();

        assert_eq!(embeddings.individual.len(), 384);
        assert!(embeddings.contextual.is_some());
        assert!(embeddings.hierarchical.is_some());
        assert!(embeddings.is_complete());
        assert_eq!(embeddings.embedding_levels(), 3);
    }

    #[tokio::test]
    async fn test_contextual_embedding_generation() {
        let mut generator = MultiLevelEmbeddingGenerator::new();
        let provider = MockEmbeddingProvider;

        let node = Node::new("text".to_string(), json!({"text": "Main content"}));
        let parent = Node::new("text".to_string(), json!({"text": "Parent content"}));
        let sibling = Node::new("text".to_string(), json!({"text": "Sibling content"}));

        let context =
            NodeContext::default()
                .with_parent(parent)
                .with_siblings(vec![sibling]);

        let embedding = generator
            .generate_contextual_embedding(&node, &context, &provider)
            .await
            .unwrap();

        assert_eq!(embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_hierarchical_embedding_generation() {
        let mut generator = MultiLevelEmbeddingGenerator::new();
        let provider = MockEmbeddingProvider;

        let node = Node::new("text".to_string(), json!({"text": "Leaf node"}));
        let root = Node::new("text".to_string(), json!({"text": "Root node"}));
        let middle = Node::new("text".to_string(), json!({"text": "Middle node"}));
        let path = vec![root, middle];

        let embedding = generator
            .generate_hierarchical_embedding(&node, &path, &provider)
            .await
            .unwrap();

        assert_eq!(embedding.len(), 384);
    }
}
