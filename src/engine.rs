//! Main NLP Engine implementation

use crate::embedding::EmbeddingGenerator;
use crate::error::NLPError;
#[cfg(feature = "multimodal")]
use crate::image_processing::{ImageEmbeddingGenerator, ImageMetadataExtractor};
use crate::models::{DeviceType, NLPConfig};
use crate::multi_level_embedding::{EmbeddingProvider, MultiLevelEmbeddingGenerator};
#[cfg(feature = "ollama")]
use crate::ollama::OllamaTextGenerator;
use crate::text_generation::TextGenerator;
use crate::utils::metrics::Timer;
use crate::NLPEngine;
use nodespace_core_types::{MultiLevelEmbeddings, NodeContext};

use async_trait::async_trait;
use nodespace_core_types::{NodeSpaceError, NodeSpaceResult, ProcessingError};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Local NLP Engine implementation with optional Ollama HTTP client
pub struct LocalNLPEngine {
    config: NLPConfig,
    embedding_generator: Arc<RwLock<Option<EmbeddingGenerator>>>,
    text_generator: Arc<RwLock<Option<TextGenerator>>>,
    #[cfg(feature = "ollama")]
    ollama_generator: Arc<RwLock<Option<OllamaTextGenerator>>>,
    #[cfg(feature = "multimodal")]
    image_embedding_generator: Arc<RwLock<Option<ImageEmbeddingGenerator>>>,
    multi_level_generator: Arc<RwLock<MultiLevelEmbeddingGenerator>>,
    device_type: DeviceType,
    initialized: Arc<RwLock<bool>>,
}

impl LocalNLPEngine {
    /// Create a new LocalNLPEngine with default configuration
    pub fn new() -> Self {
        Self::with_config(NLPConfig::default())
    }

    /// Create a new LocalNLPEngine with custom model directory
    pub fn with_model_directory<P: Into<std::path::PathBuf>>(model_dir: P) -> Self {
        Self::with_config(NLPConfig::with_model_directory(model_dir))
    }

    /// Create a new LocalNLPEngine with custom configuration
    pub fn with_config(config: NLPConfig) -> Self {
        let device_type = config.device.device_type.clone();

        Self {
            config,
            embedding_generator: Arc::new(RwLock::new(None)),
            text_generator: Arc::new(RwLock::new(None)),
            #[cfg(feature = "ollama")]
            ollama_generator: Arc::new(RwLock::new(None)),
            #[cfg(feature = "multimodal")]
            image_embedding_generator: Arc::new(RwLock::new(None)),
            multi_level_generator: Arc::new(RwLock::new(MultiLevelEmbeddingGenerator::new())),
            device_type,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Set the model path for text generation (convenience method for clients)
    pub fn set_model_path<P: Into<std::path::PathBuf>>(&mut self, model_path: P) {
        self.config.models.text_generation.model_path = Some(model_path.into());
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

        // Initialize Ollama HTTP client (if ollama feature is enabled)
        #[cfg(feature = "ollama")]
        let ollama_generator = {
            let mut generator = OllamaTextGenerator::new(self.config.models.ollama.clone())?;
            match generator.initialize().await {
                Ok(()) => {
                    tracing::info!("Ollama HTTP client initialized successfully");
                    Some(generator)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize Ollama client (will use ONNX fallback): {}",
                        e
                    );
                    None
                }
            }
        };

        // Initialize image embedding generator (if multimodal feature is enabled)
        #[cfg(feature = "multimodal")]
        let image_embedding_generator = {
            let mut generator = ImageEmbeddingGenerator::new(self.device_type.clone())?;
            match generator.initialize().await {
                Ok(()) => Some(generator),
                Err(e) => {
                    tracing::warn!("Failed to initialize image embedding generator (multimodal features will be disabled): {}", e);
                    None
                }
            }
        };

        // Store the initialized components
        *self.embedding_generator.write().await = Some(embedding_generator);
        *self.text_generator.write().await = Some(text_generator);
        #[cfg(feature = "ollama")]
        {
            *self.ollama_generator.write().await = ollama_generator;
        }
        #[cfg(feature = "multimodal")]
        {
            *self.image_embedding_generator.write().await = image_embedding_generator;
        }

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

    /// Get the Ollama text generator (ensuring it's initialized)
    #[cfg(feature = "ollama")]
    async fn get_ollama_generator(
        &self,
    ) -> Result<Arc<RwLock<Option<OllamaTextGenerator>>>, NLPError> {
        self.ensure_initialized().await?;
        Ok(self.ollama_generator.clone())
    }

    /// Get the image embedding generator (ensuring it's initialized)
    #[cfg(feature = "multimodal")]
    async fn get_image_embedding_generator(
        &self,
    ) -> Result<Arc<RwLock<Option<ImageEmbeddingGenerator>>>, NLPError> {
        self.ensure_initialized().await?;
        Ok(self.image_embedding_generator.clone())
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

    /// Clear all caches
    pub async fn clear_caches(&self) -> Result<(), NLPError> {
        if let Some(embedding_gen) = self.embedding_generator.read().await.as_ref() {
            embedding_gen.clear_cache();
        }

        #[cfg(feature = "multimodal")]
        if let Some(image_embedding_gen) = self.image_embedding_generator.read().await.as_ref() {
            image_embedding_gen.clear_cache();
        }

        // Clear multi-level embedding cache
        self.multi_level_generator.write().await.clear_cache();

        Ok(())
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let embedding_cache = self
            .embedding_generator
            .read()
            .await
            .as_ref()
            .map(|embedding_gen| embedding_gen.cache_stats());

        #[cfg(feature = "multimodal")]
        let image_embedding_cache = self
            .image_embedding_generator
            .read()
            .await
            .as_ref()
            .map(|image_embedding_gen| image_embedding_gen.cache_stats());

        CacheStats {
            embedding_cache_size: embedding_cache.map(|(size, _)| size).unwrap_or(0),
            embedding_cache_capacity: embedding_cache.map(|(_, cap)| cap).unwrap_or(0),
            #[cfg(feature = "multimodal")]
            image_embedding_cache_size: image_embedding_cache.map(|(size, _)| size).unwrap_or(0),
            #[cfg(feature = "multimodal")]
            image_embedding_cache_capacity: image_embedding_cache.map(|(_, cap)| cap).unwrap_or(0),
            #[cfg(not(feature = "multimodal"))]
            image_embedding_cache_size: 0,
            #[cfg(not(feature = "multimodal"))]
            image_embedding_cache_capacity: 0,
        }
    }
}

/// Wrapper to provide EmbeddingProvider trait for EmbeddingGenerator
struct EmbeddingGeneratorProvider {
    generator: Arc<RwLock<Option<EmbeddingGenerator>>>,
}

#[async_trait]
impl EmbeddingProvider for EmbeddingGeneratorProvider {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
        let generator = self.generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Embedding generator not initialized".to_string(),
        })?;

        generator.generate_embedding(text).await
    }
}

#[async_trait]
impl NLPEngine for LocalNLPEngine {
    /// Generate vector embedding for text content
    async fn generate_embedding(&self, text: &str) -> NodeSpaceResult<Vec<f32>> {
        let generator = self.get_embedding_generator().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                "Embedding generator not initialized",
            ))
        })?;

        generator.generate_embedding(text).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::embedding_failed(&e.to_string(), "text"))
        })
    }

    /// Generate embeddings for multiple texts (batch operation)
    async fn batch_embeddings(&self, texts: &[String]) -> NodeSpaceResult<Vec<Vec<f32>>> {
        let generator = self.get_embedding_generator().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                "Embedding generator not initialized",
            ))
        })?;

        generator.batch_embeddings(texts).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::embedding_failed(&e.to_string(), "text"))
        })
    }

    /// Generate text using Ollama (preferred) or local ONNX fallback
    async fn generate_text(&self, prompt: &str) -> NodeSpaceResult<String> {
        // Try Ollama first (if available)
        #[cfg(feature = "ollama")]
        {
            let ollama_generator = self.get_ollama_generator().await.map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::model_error(
                    "nlp-engine",
                    "ollama-generator",
                    &e.to_string(),
                ))
            })?;

            let ollama_generator = ollama_generator.read().await;
            if let Some(generator) = ollama_generator.as_ref() {
                tracing::debug!("Using Ollama HTTP client for text generation");
                return generator.generate_text(prompt).await.map_err(|e| {
                    NodeSpaceError::Processing(ProcessingError::embedding_failed(
                        &e.to_string(),
                        "text",
                    ))
                });
            } else {
                tracing::debug!("Ollama not available, falling back to ONNX text generator");
            }
        }

        // Fallback to ONNX text generator
        let generator = self.get_text_generator().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                &e.to_string(),
            ))
        })?;

        let mut generator = generator.write().await;
        let generator = generator.as_mut().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                "No text generator available (Ollama and ONNX both failed)",
            ))
        })?;

        tracing::debug!("Using ONNX text generator as fallback");
        generator.generate_text(prompt).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::embedding_failed(&e.to_string(), "text"))
        })
    }

    /// Enhanced text generation with RAG context support using Ollama (preferred) or ONNX fallback
    async fn generate_text_enhanced(
        &self,
        request: crate::TextGenerationRequest,
    ) -> NodeSpaceResult<crate::EnhancedTextGenerationResponse> {
        // Try Ollama first (if available)
        #[cfg(feature = "ollama")]
        {
            let ollama_generator = self.get_ollama_generator().await.map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::model_error(
                    "nlp-engine",
                    "ollama-generator",
                    &e.to_string(),
                ))
            })?;

            let ollama_generator = ollama_generator.read().await;
            if let Some(generator) = ollama_generator.as_ref() {
                tracing::debug!("Using Ollama HTTP client for enhanced text generation");
                return generator
                    .generate_text_enhanced(request)
                    .await
                    .map_err(|e| {
                        NodeSpaceError::Processing(ProcessingError::embedding_failed(
                            &e.to_string(),
                            "text",
                        ))
                    });
            } else {
                tracing::debug!(
                    "Ollama not available, falling back to ONNX for enhanced generation"
                );
            }
        }

        // Fallback to ONNX text generator
        let generator = self.get_text_generator().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                &e.to_string(),
            ))
        })?;

        let mut generator = generator.write().await;
        let generator = generator.as_mut().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                "No text generator available (Ollama and ONNX both failed)",
            ))
        })?;

        tracing::debug!("Using ONNX text generator for enhanced generation");
        generator
            .generate_text_enhanced(request)
            .await
            .map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::embedding_failed(
                    &e.to_string(),
                    "text",
                ))
            })
    }
    /// Extract structured data from natural language text
    async fn extract_structured_data(
        &self,
        text: &str,
        schema_hint: &str,
    ) -> NodeSpaceResult<serde_json::Value> {
        let text_generator = self.get_text_generator().await?;
        let mut text_generator = text_generator.write().await;
        let text_generator = text_generator.as_mut().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                "Text generator not initialized",
            ))
        })?;

        let prompt = format!(
            "Extract structured data from the following text based on this schema hint: {}\n\nText: {}\n\nPlease return only valid JSON:",
            schema_hint, text
        );

        let response = text_generator.generate_text(&prompt).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                &e.to_string(),
            ))
        })?;

        // Try to parse the response as JSON, with fallback for stub implementation
        match serde_json::from_str(&response) {
            Ok(json) => Ok(json),
            Err(_) => {
                // Fallback: create a simple JSON structure from the response
                let fallback_data = serde_json::json!({
                    "extracted_text": response,
                    "schema_hint": schema_hint,
                    "extraction_method": "fallback_text_extraction"
                });
                Ok(fallback_data)
            }
        }
    }

    /// Generate intelligent text summarization
    async fn generate_summary(
        &self,
        text: &str,
        max_length: Option<usize>,
    ) -> NodeSpaceResult<String> {
        let text_generator = self.get_text_generator().await?;
        let mut text_generator = text_generator.write().await;
        let text_generator = text_generator.as_mut().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                "Text generator not initialized",
            ))
        })?;

        let length_constraint = max_length
            .map(|len| format!(" in approximately {} words", len))
            .unwrap_or_default();

        let prompt = format!(
            "Please provide a concise summary of the following text{}:\n\n{}\n\nSummary:",
            length_constraint, text
        );

        text_generator.generate_text(&prompt).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                &e.to_string(),
            ))
        })
    }

    /// Analyze and classify content semantically
    async fn analyze_content(
        &self,
        text: &str,
        analysis_type: &str,
    ) -> NodeSpaceResult<crate::ContentAnalysis> {
        let text_generator = self.get_text_generator().await?;
        let mut text_generator = text_generator.write().await;
        let text_generator = text_generator.as_mut().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                "Text generator not initialized",
            ))
        })?;

        let prompt = format!(
            "Analyze the following text for {}. Provide a JSON response with classification, confidence (0.0-1.0), topics, sentiment, and entities:\n\nText: {}\n\nAnalysis:",
            analysis_type, text
        );

        let start_time = std::time::Instant::now();
        let response = text_generator.generate_text(&prompt).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "text-generator",
                &e.to_string(),
            ))
        })?;
        let processing_time = start_time.elapsed().as_millis() as u64;

        // Parse response or provide fallback analysis
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
            Ok(crate::ContentAnalysis {
                classification: parsed["classification"]
                    .as_str()
                    .unwrap_or("general")
                    .to_string(),
                confidence: parsed["confidence"].as_f64().unwrap_or(0.7) as f32,
                topics: parsed["topics"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                sentiment: parsed["sentiment"].as_str().map(|s| s.to_string()),
                entities: parsed["entities"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                processing_time_ms: processing_time,
            })
        } else {
            // Fallback: simple rule-based analysis
            Ok(crate::ContentAnalysis {
                classification: analysis_type.to_string(),
                confidence: 0.6,
                topics: Vec::new(),
                sentiment: None,
                entities: Vec::new(),
                processing_time_ms: processing_time,
            })
        }
    }

    /// Get embedding model dimensions
    fn embedding_dimensions(&self) -> usize {
        self.config.models.embedding.dimensions
    }

    /// Generate contextual embedding enhanced with relationship context
    async fn generate_contextual_embedding(
        &self,
        node: &nodespace_core_types::Node,
        context: &NodeContext,
    ) -> NodeSpaceResult<Vec<f32>> {
        self.ensure_initialized().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let provider = EmbeddingGeneratorProvider {
            generator: self.embedding_generator.clone(),
        };

        let mut multi_level_gen = self.multi_level_generator.write().await;
        multi_level_gen
            .generate_contextual_embedding(node, context, &provider)
            .await
            .map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::embedding_failed(
                    &e.to_string(),
                    "text",
                ))
            })
    }

    /// Generate hierarchical embedding with full path context from root
    async fn generate_hierarchical_embedding(
        &self,
        node: &nodespace_core_types::Node,
        path: &[nodespace_core_types::Node],
    ) -> NodeSpaceResult<Vec<f32>> {
        self.ensure_initialized().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let provider = EmbeddingGeneratorProvider {
            generator: self.embedding_generator.clone(),
        };

        let mut multi_level_gen = self.multi_level_generator.write().await;
        multi_level_gen
            .generate_hierarchical_embedding(node, path, &provider)
            .await
            .map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::embedding_failed(
                    &e.to_string(),
                    "text",
                ))
            })
    }

    /// Generate all embedding levels for a node (individual, contextual, hierarchical)
    async fn generate_all_embeddings(
        &self,
        node: &nodespace_core_types::Node,
        context: &NodeContext,
        path: &[nodespace_core_types::Node],
    ) -> NodeSpaceResult<MultiLevelEmbeddings> {
        self.ensure_initialized().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let provider = EmbeddingGeneratorProvider {
            generator: self.embedding_generator.clone(),
        };

        let mut multi_level_gen = self.multi_level_generator.write().await;
        multi_level_gen
            .generate_all_embeddings(node, context, path, &provider)
            .await
            .map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::embedding_failed(
                    &e.to_string(),
                    "text",
                ))
            })
    }

    /// Generate vector embedding for image content (multimodal)
    #[cfg(feature = "multimodal")]
    async fn generate_image_embedding(&self, image_data: &[u8]) -> NodeSpaceResult<Vec<f32>> {
        let generator = self.get_image_embedding_generator().await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "embedding-generator",
                &e.to_string(),
            ))
        })?;

        let generator = generator.read().await;
        let generator = generator.as_ref().ok_or_else(|| {
            NodeSpaceError::Processing(ProcessingError::model_error(
                "nlp-engine",
                "image-embedding-generator",
                "Image embedding generator not available (multimodal models failed to load)",
            ))
        })?;

        generator.generate_embedding(image_data).await.map_err(|e| {
            NodeSpaceError::Processing(ProcessingError::embedding_failed(&e.to_string(), "text"))
        })
    }

    /// Extract comprehensive metadata from image
    #[cfg(feature = "multimodal")]
    async fn extract_image_metadata(
        &self,
        image_data: &[u8],
    ) -> NodeSpaceResult<crate::ImageMetadata> {
        ImageMetadataExtractor::extract_metadata(image_data)
            .await
            .map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::embedding_failed(
                    &e.to_string(),
                    "text",
                ))
            })
    }

    /// Generate multimodal response with text and image understanding using Ollama (preferred) or fallback
    #[cfg(feature = "multimodal")]
    async fn generate_multimodal_response(
        &self,
        request: crate::MultimodalRequest,
    ) -> NodeSpaceResult<crate::MultimodalResponse> {
        let _timer = Timer::new("multimodal_response_generation");

        // Try Ollama multimodal first (if available)
        #[cfg(feature = "ollama")]
        {
            let ollama_generator = self.get_ollama_generator().await.map_err(|e| {
                NodeSpaceError::Processing(ProcessingError::model_error(
                    "nlp-engine",
                    "ollama-generator",
                    &e.to_string(),
                ))
            })?;

            let ollama_generator = ollama_generator.read().await;
            if let Some(generator) = ollama_generator.as_ref() {
                tracing::debug!("Using Ollama HTTP client for multimodal response generation");
                return generator
                    .generate_multimodal_response(request)
                    .await
                    .map_err(|e| {
                        NodeSpaceError::Processing(ProcessingError::embedding_failed(
                            &e.to_string(),
                            "multimodal",
                        ))
                    });
            } else {
                tracing::debug!("Ollama not available, using fallback multimodal implementation");
            }
        }

        // Fallback implementation: Generate embeddings and use text-only generation
        let mut image_embeddings = Vec::new();
        let mut image_references = Vec::new();

        for (i, image_input) in request.images.iter().enumerate() {
            let embedding = self.generate_image_embedding(&image_input.data).await?;
            image_embeddings.push(embedding);

            // Create basic image reference
            let image_ref = crate::ImageReference {
                id: image_input
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("img_{}", i)),
                description: image_input
                    .description
                    .clone()
                    .unwrap_or_else(|| "Image".to_string()),
                confidence: 0.6, // Lower confidence for fallback
            };
            image_references.push(image_ref);
        }

        // Use text generation with image context (fallback approach)
        let enhanced_prompt = self.build_multimodal_prompt(&request, &image_references);

        let text_request = crate::TextGenerationRequest {
            prompt: enhanced_prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            context_window: 8192,
            conversation_mode: false,
            rag_context: None,
            enable_link_generation: request.enable_smart_links,
            node_metadata: Vec::new(),
        };

        let text_response = self.generate_text_enhanced(text_request).await?;

        // Build multimodal response
        let response = crate::MultimodalResponse {
            text: text_response.text,
            image_sources: image_references,
            smart_links: Vec::new(), // TODO: Extract from enhanced response
            generation_metrics: text_response.generation_metrics,
            image_utilization: crate::ImageUtilization {
                images_referenced: !request.images.is_empty(),
                images_used: request.images.len(),
                understanding_confidence: 0.5, // Lower confidence for fallback approach
            },
        };

        tracing::debug!("Using fallback multimodal implementation (text-only with image context)");
        Ok(response)
    }
}

#[cfg(feature = "multimodal")]
impl LocalNLPEngine {
    /// Build multimodal prompt combining text query with image context
    fn build_multimodal_prompt(
        &self,
        request: &crate::MultimodalRequest,
        image_refs: &[crate::ImageReference],
    ) -> String {
        let mut prompt = "# Multimodal Query\n\n".to_string();
        prompt.push_str(&format!("User query: {}\n\n", request.text_query));

        if !image_refs.is_empty() {
            prompt.push_str("## Images provided:\n");
            for (i, img_ref) in image_refs.iter().enumerate() {
                prompt.push_str(&format!(
                    "{}. Image ID: {} - {}\n",
                    i + 1,
                    img_ref.id,
                    img_ref.description
                ));
            }
            prompt.push_str("\nPlease analyze the provided images in context of the user's query and provide a comprehensive response.\n\n");
        }

        prompt.push_str("Response:");
        prompt
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

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub embedding_cache_size: usize,
    pub embedding_cache_capacity: usize,
    pub image_embedding_cache_size: usize,
    pub image_embedding_cache_capacity: usize,
}
