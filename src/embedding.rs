//! Embedding generation using fastembed-rs and ONNX Runtime
//! Uses unified ONNX stack for local-first AI processing

use crate::error::NLPError;
use crate::models::{DeviceType, EmbeddingModelConfig};
use crate::utils::{metrics::Timer, vector};

use dashmap::DashMap;
use std::sync::Arc;

// Real ML dependencies (ONNX Runtime + fastembed)
#[cfg(feature = "real-ml")]
use fastembed::{EmbeddingModel, FlagEmbedding, InitOptions, TextEmbedding};

/// Embedding generator using fastembed-rs with ONNX Runtime
pub struct EmbeddingGenerator {
    config: EmbeddingModelConfig,
    device_type: DeviceType,
    #[cfg(feature = "real-ml")]
    model: Option<FlagEmbedding>,
    cache: Arc<DashMap<String, Vec<f32>>>,
    initialized: bool,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new(config: EmbeddingModelConfig, device_type: DeviceType) -> Result<Self, NLPError> {
        Ok(Self {
            config,
            device_type,
            #[cfg(feature = "real-ml")]
            model: None,
            cache: Arc::new(DashMap::new()),
            initialized: false,
        })
    }

    /// Initialize the model
    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        let _timer = Timer::new("embedding_model_initialization");

        if self.initialized {
            return Ok(());
        }

        #[cfg(feature = "real-ml")]
        {
            tracing::info!("Loading fastembed model: {}", self.config.model_name);
            self.load_fastembed_model().await?;
            tracing::info!(
                "Fastembed model initialized successfully: {}",
                self.config.model_name
            );
        }

        #[cfg(not(feature = "real-ml"))]
        {
            tracing::info!(
                "STUB: Embedding model initialized: {}",
                self.config.model_name
            );
        }

        self.initialized = true;
        Ok(())
    }

    #[cfg(feature = "real-ml")]
    async fn load_fastembed_model(&mut self) -> Result<(), NLPError> {
        // Map model name to fastembed EmbeddingModel enum
        let embedding_model = match self.config.model_name.as_str() {
            "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
            _ => {
                return Err(NLPError::ModelLoading {
                    message: format!("Unsupported embedding model: {}", self.config.model_name),
                });
            }
        };

        // Configure initialization options
        let init_options = InitOptions {
            model_name: embedding_model,
            show_download_progress: true,
            ..Default::default()
        };

        // Initialize the model
        let model = FlagEmbedding::try_new(init_options)
            .map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to initialize fastembed model: {}", e),
            })?;

        self.model = Some(model);
        Ok(())
    }

    /// Generate embedding for a single text
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        let _timer = Timer::new("embedding_generation");

        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        #[cfg(feature = "real-ml")]
        {
            let embedding = self.generate_fastembed_embedding(text).await?;
            self.cache.insert(text.to_string(), embedding.clone());
            Ok(embedding)
        }

        #[cfg(not(feature = "real-ml"))]
        {
            let embedding = self.generate_stub_embedding(text)?;
            self.cache.insert(text.to_string(), embedding.clone());
            Ok(embedding)
        }
    }

    #[cfg(feature = "real-ml")]
    async fn generate_fastembed_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
        let model = self.model.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Model not loaded".to_string(),
        })?;

        // Generate embedding using fastembed
        let documents = vec![text];
        let embeddings = model
            .embed(documents, None)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Embedding generation failed: {}", e),
            })?;

        // Extract the first (and only) embedding
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| NLPError::ProcessingError {
                message: "No embedding generated".to_string(),
            })?;

        // Convert to Vec<f32> and normalize if configured
        let mut final_embedding = embedding;
        if self.config.normalize {
            vector::normalize_vector(&mut final_embedding)?;
        }

        Ok(final_embedding)
    }

    #[cfg(not(feature = "real-ml"))]
    fn generate_stub_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
        // STUB: Generate a deterministic fake embedding based on text hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Create a fake embedding vector
        let mut embedding = Vec::with_capacity(self.config.dimensions);
        let mut seed = hash;
        for _ in 0..self.config.dimensions {
            // Simple pseudo-random number generation
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed >> 16) as f32) / 32768.0 - 1.0; // Range [-1, 1]
            embedding.push(value);
        }

        // Normalize if configured
        if self.config.normalize {
            vector::normalize_vector(&mut embedding)?;
        }

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn batch_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NLPError> {
        let _timer = Timer::new("batch_embedding_generation");

        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        #[cfg(feature = "real-ml")]
        {
            let model = self.model.as_ref().ok_or_else(|| NLPError::ModelLoading {
                message: "Model not loaded".to_string(),
            })?;

            // Check cache for all texts
            let mut cached_results = Vec::with_capacity(texts.len());
            let mut uncached_texts = Vec::new();
            let mut uncached_indices = Vec::new();

            for (i, text) in texts.iter().enumerate() {
                if let Some(cached) = self.cache.get(text) {
                    cached_results.push((i, cached.clone()));
                } else {
                    uncached_texts.push(text.as_str());
                    uncached_indices.push(i);
                }
            }

            // Generate embeddings for uncached texts in batch
            let new_embeddings = if !uncached_texts.is_empty() {
                model
                    .embed(uncached_texts.clone(), None)
                    .map_err(|e| NLPError::ProcessingError {
                        message: format!("Batch embedding generation failed: {}", e),
                    })?
            } else {
                Vec::new()
            };

            // Cache new embeddings and combine results
            let mut final_results = vec![Vec::new(); texts.len()];
            
            // Add cached results
            for (index, embedding) in cached_results {
                final_results[index] = embedding;
            }

            // Add new results and cache them
            for (i, mut embedding) in new_embeddings.into_iter().enumerate() {
                if self.config.normalize {
                    vector::normalize_vector(&mut embedding)?;
                }
                
                let text_index = uncached_indices[i];
                let text = &texts[text_index];
                
                self.cache.insert(text.clone(), embedding.clone());
                final_results[text_index] = embedding;
            }

            Ok(final_results)
        }

        #[cfg(not(feature = "real-ml"))]
        {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                let embedding = self.generate_embedding(text).await?;
                results.push(embedding);
            }
            Ok(results)
        }
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let len = self.cache.len();
        let capacity = self.cache.capacity();
        (len, capacity)
    }
}