//! Model configuration and management

use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};

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

    /// Ollama HTTP client configuration
    #[cfg(feature = "ollama")]
    pub ollama: OllamaConfig,
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

/// Configuration for Ollama HTTP client
#[cfg(feature = "ollama")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server base URL
    pub base_url: String,

    /// Default model for text generation
    pub default_model: String,

    /// Multimodal model for vision tasks
    pub multimodal_model: String,

    /// HTTP request timeout in seconds
    pub timeout_secs: u64,

    /// Maximum tokens for generation
    pub max_tokens: usize,

    /// Default temperature for generation
    pub temperature: f32,

    /// Retry attempts for failed requests
    pub retry_attempts: usize,

    /// Enable streaming responses
    pub stream: bool,
}

#[cfg(feature = "ollama")]
impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            default_model: "gemma3:12b".to_string(),
            multimodal_model: "gemma3:12b".to_string(), // Same model for now
            timeout_secs: 120,
            max_tokens: 4000,
            temperature: 0.7,
            retry_attempts: 3,
            stream: false,
        }
    }
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

impl NLPConfig {
    /// Create configuration with custom model directory
    pub fn with_model_directory<P: Into<PathBuf>>(model_dir: P) -> Self {
        let model_dir = model_dir.into();

        Self {
            models: ModelConfigs {
                embedding: EmbeddingModelConfig {
                    model_name: "BAAI/bge-small-en-v1.5".to_string(),
                    model_path: None, // Future: local embedding model path
                    dimensions: 384,
                    max_sequence_length: 512,
                    normalize: true,
                },
                text_generation: TextGenerationModelConfig {
                    model_name: "local/gemma-3-1b-it-onnx".to_string(),
                    model_path: Some(model_dir.join("gemma-3-1b-it-onnx/model.onnx")),
                    max_context_length: 8192,
                    default_temperature: 0.7,
                    default_max_tokens: 1024,
                    default_top_p: 0.95,
                },
                #[cfg(feature = "ollama")]
                ollama: OllamaConfig::default(),
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

    /// Get default model path for a given model file
    fn get_default_model_path(model_file: &str) -> Option<PathBuf> {
        // Try environment variable first
        if let Ok(models_dir) = std::env::var("NODESPACE_MODELS_DIR") {
            return Some(PathBuf::from(models_dir).join(model_file));
        }

        // Try workspace-relative path (for development)
        let current_dir = std::env::current_dir().ok()?;
        if let Some(workspace_models) = current_dir
            .parent()
            .map(|p| p.join("models").join(model_file))
        {
            if workspace_models.exists() {
                return Some(workspace_models);
            }
        }

        // Fall back to cache directory
        let home_dir = std::env::var("HOME").ok()?;
        let model_cache_dir = PathBuf::from(home_dir)
            .join(".cache")
            .join("nodespace")
            .join("models");

        Some(model_cache_dir.join(model_file))
    }
}

impl Default for NLPConfig {
    fn default() -> Self {
        Self {
            models: ModelConfigs {
                embedding: EmbeddingModelConfig {
                    model_name: "BAAI/bge-small-en-v1.5".to_string(), // Text-only for now, plan multimodal upgrade
                    model_path: None, // Future: local embedding model path when available
                    dimensions: 384,
                    max_sequence_length: 512,
                    normalize: true,
                },
                text_generation: TextGenerationModelConfig {
                    model_name: "local/gemma-3-1b-it-onnx".to_string(),
                    model_path: NLPConfig::get_default_model_path("gemma-3-1b-it-onnx/model.onnx"),
                    max_context_length: 8192, // Gemma 3 1B context length
                    default_temperature: 0.7,
                    default_max_tokens: 1024,
                    default_top_p: 0.95,
                },
                #[cfg(feature = "ollama")]
                ollama: OllamaConfig::default(),
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

/// Model metadata and state information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub loaded: bool,
    pub device: DeviceType,
    pub memory_usage_mb: Option<f32>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}
