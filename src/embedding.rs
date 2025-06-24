//! Embedding generation using Candle and transformers
//! Uses stable Candle stack (0.9.x) for local-first AI processing

use crate::error::NLPError;
use crate::models::{DeviceType, EmbeddingModelConfig};
use crate::utils::{metrics::Timer, vector};
#[cfg(feature = "real-ml")]
use crate::utils::device;

use dashmap::DashMap;
use std::sync::Arc;

// Real ML dependencies (stable versions)
#[cfg(feature = "real-ml")]
use candle_core::{Device, Tensor};
#[cfg(feature = "real-ml")]
use candle_nn::VarBuilder;
#[cfg(feature = "real-ml")]
use candle_transformers::models::bert::BertModel;
#[cfg(feature = "real-ml")]
use hf_hub::api::tokio::Api;
#[cfg(feature = "real-ml")]
use tokenizers::Tokenizer;

/// Embedding generator using transformer models
pub struct EmbeddingGenerator {
    config: EmbeddingModelConfig,
    #[cfg(feature = "real-ml")]
    device: Device,
    #[cfg(not(feature = "real-ml"))]
    #[allow(dead_code)]
    device_type: DeviceType,
    #[cfg(feature = "real-ml")]
    model: Option<BertModel>,
    #[cfg(feature = "real-ml")]
    tokenizer: Option<Tokenizer>,
    cache: Arc<DashMap<String, Vec<f32>>>,
    initialized: bool,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new(config: EmbeddingModelConfig, device_type: DeviceType) -> Result<Self, NLPError> {
        #[cfg(feature = "real-ml")]
        {
            let device = device::create_device(device_type)?;
            Ok(Self {
                config,
                device,
                model: None,
                tokenizer: None,
                cache: Arc::new(DashMap::new()),
                initialized: false,
            })
        }

        #[cfg(not(feature = "real-ml"))]
        {
            Ok(Self {
                config,
                device_type,
                cache: Arc::new(DashMap::new()),
                initialized: false,
            })
        }
    }


    /// Initialize the model and tokenizer
    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        let _timer = Timer::new("embedding_model_initialization");

        if self.initialized {
            return Ok(());
        }

        #[cfg(feature = "real-ml")]
        {
            tracing::info!("Loading real embedding model: {}", self.config.model_name);
            self.load_real_model().await?;
            tracing::info!(
                "Real embedding model initialized successfully: {}",
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
    async fn load_real_model(&mut self) -> Result<(), NLPError> {
        // Load tokenizer from HuggingFace Hub
        let api = Api::new().map_err(|e| NLPError::ModelLoading {
            message: format!("Failed to initialize HF API: {}", e),
        })?;

        let repo = api.model(self.config.model_name.clone());

        // Download tokenizer
        let tokenizer_file =
            repo.get("tokenizer.json")
                .await
                .map_err(|e| NLPError::ModelLoading {
                    message: format!("Failed to download tokenizer: {}", e),
                })?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_file).map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to load tokenizer: {}", e),
            })?;

        // Download model weights
        let model_file =
            repo.get("model.safetensors")
                .await
                .map_err(|e| NLPError::ModelLoading {
                    message: format!("Failed to download model: {}", e),
                })?;

        // Load model configuration
        let config_file = repo
            .get("config.json")
            .await
            .map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to download config: {}", e),
            })?;

        let config_content =
            std::fs::read_to_string(&config_file).map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to read config: {}", e),
            })?;

        let bert_config: serde_json::Value =
            serde_json::from_str(&config_content).map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to parse config: {}", e),
            })?;

        // Load model weights
        let weights = candle_core::safetensors::load(&model_file, &self.device).map_err(|e| {
            NLPError::ModelLoading {
                message: format!("Failed to load model weights: {}", e),
            }
        })?;

        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, &self.device);

        // Create BERT model with basic config
        let model_config = candle_transformers::models::bert::Config {
            vocab_size: bert_config["vocab_size"].as_u64().unwrap_or(30522) as usize,
            hidden_size: bert_config["hidden_size"].as_u64().unwrap_or(384) as usize,
            num_hidden_layers: bert_config["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            num_attention_heads: bert_config["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: bert_config["intermediate_size"].as_u64().unwrap_or(1536) as usize,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: bert_config["max_position_embeddings"]
                .as_u64()
                .unwrap_or(512) as usize,
            type_vocab_size: bert_config["type_vocab_size"].as_u64().unwrap_or(2) as usize,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            pad_token_id: 0,
            position_embedding_type:
                candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };

        let model = BertModel::load(vb, &model_config).map_err(|e| NLPError::ModelLoading {
            message: format!("Failed to create BERT model: {}", e),
        })?;

        self.model = Some(model);
        self.tokenizer = Some(tokenizer);
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
            let embedding = self.generate_real_embedding(text).await?;
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
    async fn generate_real_embedding(&self, text: &str) -> Result<Vec<f32>, NLPError> {
        let model = self.model.as_ref().ok_or_else(|| NLPError::ModelLoading {
            message: "Model not loaded".to_string(),
        })?;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| NLPError::ModelLoading {
                message: "Tokenizer not loaded".to_string(),
            })?;

        // Tokenize the input text
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Tokenization failed: {}", e),
            })?;

        let tokens = encoding.get_ids();
        let token_type_ids = encoding.get_type_ids();

        // Convert to tensors
        let input_ids = Tensor::new(tokens, &self.device)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to create input tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to unsqueeze input tensor: {}", e),
            })?;

        let token_type_ids = Tensor::new(token_type_ids, &self.device)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to create token type tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to unsqueeze token type tensor: {}", e),
            })?;

        // Forward pass through the model (with attention mask as None)
        let embeddings = model
            .forward(&input_ids, &token_type_ids, None)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Model forward pass failed: {}", e),
            })?;

        // Mean pooling: average over sequence length (dim 1)
        let pooled = embeddings.mean(1).map_err(|e| NLPError::ProcessingError {
            message: format!("Pooling failed: {}", e),
        })?;

        // Convert to Vec<f32>
        let embedding_vec: Vec<f32> = pooled
            .squeeze(0)
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to squeeze tensor: {}", e),
            })?
            .to_vec1()
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to convert tensor to vec: {}", e),
            })?;

        // Normalize if configured
        let mut final_embedding = embedding_vec;
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

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch (STUB)
    pub async fn batch_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, NLPError> {
        let _timer = Timer::new("batch_embedding_generation");

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let embedding = self.generate_embedding(text).await?;
            results.push(embedding);
        }

        Ok(results)
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
