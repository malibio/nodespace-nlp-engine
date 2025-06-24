//! Main NLP Engine implementation

use crate::embedding::EmbeddingGenerator;
use crate::error::NLPError;
use crate::models::{DeviceType, NLPConfig};
use crate::surrealql::SurrealQLGenerator;
use crate::text_generation::TextGenerator;
use crate::utils::metrics::Timer;
use crate::NLPEngine;

use async_trait::async_trait;
use nodespace_core_types::{NodeSpaceError, NodeSpaceResult};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Local NLP Engine implementation using Mistral.rs and Candle
pub struct LocalNLPEngine {
    config: NLPConfig,
    embedding_generator: Arc<RwLock<Option<EmbeddingGenerator>>>,
    text_generator: Arc<RwLock<Option<TextGenerator>>>,
    surrealql_generator: Arc<RwLock<Option<SurrealQLGenerator>>>,
    device_type: DeviceType,
    initialized: Arc<RwLock<bool>>,
}

impl LocalNLPEngine {
    /// Create a new LocalNLPEngine with default configuration
    pub fn new() -> Self {
        Self::with_config(NLPConfig::default())
    }

    /// Create a new LocalNLPEngine with custom configuration
    pub fn with_config(config: NLPConfig) -> Self {
        let device_type = config.device.device_type.clone();

        Self {
            config,
            embedding_generator: Arc::new(RwLock::new(None)),
            text_generator: Arc::new(RwLock::new(None)),
            surrealql_generator: Arc::new(RwLock::new(None)),
            device_type,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize all components of the NLP engine
    pub async fn initialize(&self) -> Result<(), NLPError> {
        let _timer = Timer::new("nlp_engine_initialization");

        // Check if already initialized
        {
            let initialized = self.initialized.read().await;
            if *initialized {
                return Ok(());
            }
        }

        tracing::info!("Initializing NodeSpace NLP Engine...");

        // Initialize embedding generator
        let mut embedding_generator = EmbeddingGenerator::new(
            self.config.models.embedding.clone(),
            self.device_type.clone(),
        )?;
        embedding_generator.initialize().await?;

        // Initialize text generator
        let mut text_generator = TextGenerator::new(
            self.config.models.text_generation.clone(),
            self.device_type.clone(),
        )?;
        text_generator.initialize().await?;

        // Initialize SurrealQL generator
        let surrealql_generator = SurrealQLGenerator::new();

        // Store the initialized components
        *self.embedding_generator.write().await = Some(embedding_generator);
        *self.text_generator.write().await = Some(text_generator);
        *self.surrealql_generator.write().await = Some(surrealql_generator);

        // Mark as initialized
        *self.initialized.write().await = true;

        tracing::info!("NodeSpace NLP Engine initialized successfully");
        Ok(())
    }

    /// Check if the engine is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Get engine status and information
    pub async fn status(&self) -> EngineStatus {
        let initialized = self.is_initialized().await;

        let embedding_info = if initialized {
            self.embedding_generator
                .read()
                .await
                .as_ref()
                .map(|gen| EmbeddingInfo {
                    model_name: self.config.models.embedding.model_name.clone(),
                    dimensions: gen.dimensions(),
                    cache_stats: gen.cache_stats(),
                })
        } else {
            None
        };

        let text_generation_info = if initialized {
            self.text_generator
                .read()
                .await
                .as_ref()
                .map(|gen| gen.model_info())
        } else {
            None
        };

        EngineStatus {
            initialized,
            device_type: self.device_type.clone(),
            embedding_info,
            text_generation_info,
        }
    }

    /// Ensure the engine is initialized
    async fn ensure_initialized(&self) -> Result<(), NLPError> {
        if !self.is_initialized().await {
            self.initialize().await?;
        }
        Ok(())
    }

    /// Get the embedding generator (ensuring it's initialized)
    async fn get_embedding_generator(
        &self,
    ) -> Result<Arc<RwLock<Option<EmbeddingGenerator>>>, NLPError> {
        self.ensure_initialized().await?;
        Ok(self.embedding_generator.clone())
    }

    /// Get the text generator (ensuring it's initialized)
    async fn get_text_generator(&self) -> Result<Arc<RwLock<Option<TextGenerator>>>, NLPError> {
        self.ensure_initialized().await?;
        Ok(self.text_generator.clone())
    }

    /// Get the SurrealQL generator (ensuring it's initialized)
    async fn get_surrealql_generator(
        &self,
    ) -> Result<Arc<RwLock<Option<SurrealQLGenerator>>>, NLPError> {
        self.ensure_initialized().await?;
        Ok(self.surrealql_generator.clone())
    }

    /// Generate embeddings with advanced preprocessing
    pub async fn generate_embedding_advanced(
        &self,
        text: &str,
        preprocess: bool,
    ) -> Result<Vec<f32>, NLPError> {
        let generator = self.get_embedding_generator().await?;
        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Embedding generator not initialized".to_string(),
        })?;

        let processed_text = if preprocess {
            crate::utils::text::preprocess_for_embedding(text)
        } else {
            text.to_string()
        };

        generator.generate_embedding(&processed_text).await
    }

    /// Generate SurrealQL with detailed options
    pub async fn generate_surrealql_advanced(
        &self,
        natural_query: &str,
        schema_context: &str,
        safety_checks: bool,
    ) -> Result<SurrealQLResult, NLPError> {
        let surrealql_generator = self.get_surrealql_generator().await?;
        let surrealql_generator = surrealql_generator.read().await;
        let surrealql_generator =
            surrealql_generator
                .as_ref()
                .ok_or_else(|| NLPError::ModelLoading {
                    message: "SurrealQL generator not initialized".to_string(),
                })?;

        let text_generator = self.get_text_generator().await?;
        let mut text_generator = text_generator.write().await;
        let text_generator = text_generator
            .as_mut()
            .ok_or_else(|| NLPError::ModelLoading {
                message: "Text generator not initialized".to_string(),
            })?;

        let _timer = Timer::new("advanced_surrealql_generation");

        let surrealql = surrealql_generator
            .generate_surrealql(text_generator, natural_query, schema_context, safety_checks)
            .await?;

        Ok(SurrealQLResult {
            query: surrealql,
            safety_checks_applied: safety_checks,
            estimated_complexity: self.estimate_query_complexity(natural_query),
        })
    }

    /// Estimate query complexity for performance optimization
    fn estimate_query_complexity(&self, query: &str) -> QueryComplexity {
        let word_count = query.split_whitespace().count();
        let has_joins =
            query.to_lowercase().contains("with") || query.to_lowercase().contains("related");
        let has_aggregation =
            query.to_lowercase().contains("count") || query.to_lowercase().contains("sum");

        if word_count > 20 || has_joins || has_aggregation {
            QueryComplexity::High
        } else if word_count > 10 {
            QueryComplexity::Medium
        } else {
            QueryComplexity::Low
        }
    }

    /// Clear all caches
    pub async fn clear_caches(&self) -> Result<(), NLPError> {
        if let Some(embedding_gen) = self.embedding_generator.read().await.as_ref() {
            embedding_gen.clear_cache();
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let embedding_cache = self.embedding_generator.read().await.as_ref().map(|embedding_gen| embedding_gen.cache_stats());

        CacheStats {
            embedding_cache_size: embedding_cache.map(|(size, _)| size).unwrap_or(0),
            embedding_cache_capacity: embedding_cache.map(|(_, cap)| cap).unwrap_or(0),
        }
    }
}

#[async_trait]
impl NLPEngine for LocalNLPEngine {
    /// Generate vector embedding for text content
    async fn generate_embedding(&self, text: &str) -> NodeSpaceResult<Vec<f32>> {
        let generator = self
            .get_embedding_generator()
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))?;

        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| {
            NodeSpaceError::ProcessingError("Embedding generator not initialized".to_string())
        })?;

        generator
            .generate_embedding(text)
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))
    }

    /// Generate embeddings for multiple texts (batch operation)
    async fn batch_embeddings(&self, texts: &[String]) -> NodeSpaceResult<Vec<Vec<f32>>> {
        let generator = self
            .get_embedding_generator()
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))?;

        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| {
            NodeSpaceError::ProcessingError("Embedding generator not initialized".to_string())
        })?;

        generator
            .batch_embeddings(texts)
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))
    }

    /// Generate text using the local LLM (Mistral.rs)
    async fn generate_text(&self, prompt: &str) -> NodeSpaceResult<String> {
        let generator = self
            .get_text_generator()
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))?;

        let mut generator = generator.write().await;
        let generator = generator.as_mut().ok_or_else(|| {
            NodeSpaceError::ProcessingError("Text generator not initialized".to_string())
        })?;

        generator
            .generate_text(prompt)
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))
    }

    /// Generate SurrealQL from natural language query
    async fn generate_surrealql(
        &self,
        natural_query: &str,
        schema_context: &str,
    ) -> NodeSpaceResult<String> {
        let result = self
            .generate_surrealql_advanced(natural_query, schema_context, true)
            .await
            .map_err(|e| NodeSpaceError::ProcessingError(e.to_string()))?;

        Ok(result.query)
    }

    /// Get embedding model dimensions
    fn embedding_dimensions(&self) -> usize {
        self.config.models.embedding.dimensions
    }
}

impl Default for LocalNLPEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine status information
#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub initialized: bool,
    pub device_type: DeviceType,
    pub embedding_info: Option<EmbeddingInfo>,
    pub text_generation_info: Option<crate::text_generation::TextGenerationModelInfo>,
}

/// Embedding model information
#[derive(Debug, Clone)]
pub struct EmbeddingInfo {
    pub model_name: String,
    pub dimensions: usize,
    pub cache_stats: (usize, usize), // (size, capacity)
}

/// SurrealQL generation result
#[derive(Debug, Clone)]
pub struct SurrealQLResult {
    pub query: String,
    pub safety_checks_applied: bool,
    pub estimated_complexity: QueryComplexity,
}

/// Query complexity estimation
#[derive(Debug, Clone, PartialEq)]
pub enum QueryComplexity {
    Low,
    Medium,
    High,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub embedding_cache_size: usize,
    pub embedding_cache_capacity: usize,
}
