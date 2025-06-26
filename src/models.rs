//! Model configuration and management

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the NLP Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLPConfig {
    /// Model configurations
    pub models: ModelConfigs,

    /// Device configuration (CPU, CUDA, Metal)
    pub device: DeviceConfig,

    /// Cache configuration
    pub cache: CacheConfig,

    /// Performance settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    /// Embedding model configuration
    pub embedding: EmbeddingModelConfig,

    /// Text generation model configuration (ONNX Runtime)
    pub text_generation: TextGenerationModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelConfig {
    /// Model name or path
    pub model_name: String,

    /// Local model path (if downloaded)
    pub model_path: Option<PathBuf>,

    /// Embedding dimensions
    pub dimensions: usize,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Normalization settings
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationModelConfig {
    /// Model name (Gemma 3 1B Instruct)
    pub model_name: String,

    /// Local model path (ONNX format)
    pub model_path: Option<PathBuf>,

    /// Maximum context length
    pub max_context_length: usize,

    /// Default generation parameters
    pub default_temperature: f32,
    pub default_max_tokens: u32,
    pub default_top_p: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Preferred device type
    pub device_type: DeviceType,

    /// GPU device ID (for CUDA)
    pub gpu_device_id: Option<usize>,

    /// Memory limits
    pub max_memory_gb: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable model caching
    pub enable_model_cache: bool,

    /// Enable embedding caching
    pub enable_embedding_cache: bool,

    /// Maximum cache size (in MB)
    pub max_cache_size_mb: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for CPU inference
    pub cpu_threads: Option<usize>,

    /// Batch size for embedding generation
    pub embedding_batch_size: usize,

    /// Enable async processing
    pub enable_async_processing: bool,

    /// Connection pool size for concurrent requests
    pub pool_size: usize,
}

impl Default for NLPConfig {
    fn default() -> Self {
        Self {
            models: ModelConfigs {
                embedding: EmbeddingModelConfig {
                    model_name: "BAAI/bge-small-en-v1.5".to_string(),
                    model_path: None,
                    dimensions: 384,
                    max_sequence_length: 512,
                    normalize: true,
                },
                text_generation: TextGenerationModelConfig {
                    model_name: "google/gemma-3-1b-instruct".to_string(),
                    model_path: Some("/models/gemma-3-1b-it-onnx/".into()),
                    max_context_length: 8192, // Gemma 3 1B context length
                    default_temperature: 0.7,
                    default_max_tokens: 1024,
                    default_top_p: 0.95,
                },
            },
            device: DeviceConfig {
                device_type: DeviceType::Auto,
                gpu_device_id: None,
                max_memory_gb: None,
            },
            cache: CacheConfig {
                enable_model_cache: true,
                enable_embedding_cache: true,
                max_cache_size_mb: 1024,
                cache_ttl_seconds: 3600,
            },
            performance: PerformanceConfig {
                cpu_threads: None,
                embedding_batch_size: 32,
                enable_async_processing: true,
                pool_size: 4,
            },
        }
    }
}

/// Available model types for discovery and validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AvailableModels {
    TextGeneration(TextGenerationModel),
    Embedding(EmbeddingModel),
}

/// Available text generation models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TextGenerationModel {
    Gemma3_1bOnnx,
    // Future models can be added here
}

/// Available embedding models  
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    BgeSmallEn,
    // Future models can be added here
}

/// Model metadata and state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: AvailableModels,
    pub path: String,
    pub loaded: bool,
    pub device: DeviceType,
    pub memory_usage_mb: Option<f32>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub capabilities: ModelCapabilities,
}

/// Model capabilities and specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub max_context_length: usize,
    pub dimensions: Option<usize>, // For embedding models
    pub supports_streaming: bool,
    pub memory_requirements_mb: usize,
}
