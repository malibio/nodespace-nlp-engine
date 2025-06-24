//! Utility functions for the NLP Engine

/// Text preprocessing utilities
pub mod text {
    use regex::Regex;

    /// Clean and normalize text for embedding generation
    pub fn preprocess_for_embedding(text: &str) -> String {
        let mut processed = text.trim().to_lowercase();

        // Remove extra whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        processed = whitespace_regex.replace_all(&processed, " ").to_string();

        // Normalize punctuation (remove no-effect replacements)
        // Note: Placeholder for actual punctuation normalization

        processed
    }

    /// Split text into chunks for processing
    pub fn chunk_text(text: &str, max_length: usize, overlap: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        if words.len() <= max_length {
            return vec![text.to_string()];
        }

        let mut start = 0;
        while start < words.len() {
            let end = (start + max_length).min(words.len());
            let chunk = words[start..end].join(" ");
            chunks.push(chunk);

            if end >= words.len() {
                break;
            }

            start = end - overlap;
        }

        chunks
    }

    /// Extract key phrases from text
    pub fn extract_key_phrases(text: &str) -> Vec<String> {
        // Simple extraction based on common patterns
        let mut phrases = Vec::new();

        // Extract quoted phrases
        let quote_regex = Regex::new(r#""([^"]+)"|'([^']+)'"#).unwrap();
        for capture in quote_regex.captures_iter(text) {
            if let Some(phrase) = capture.get(1).or_else(|| capture.get(2)) {
                phrases.push(phrase.as_str().to_string());
            }
        }

        // Extract capitalized phrases (potential proper nouns)
        let proper_noun_regex = Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b").unwrap();
        for capture in proper_noun_regex.find_iter(text) {
            phrases.push(capture.as_str().to_string());
        }

        phrases
    }
}

/// Vector operations utilities
pub mod vector {
    use crate::error::NLPError;

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, NLPError> {
        if a.len() != b.len() {
            return Err(NLPError::InvalidInput {
                message: "Vectors must have the same dimension".to_string(),
            });
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Normalize a vector to unit length
    pub fn normalize_vector(vector: &mut [f32]) -> Result<(), NLPError> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return Err(NLPError::InvalidInput {
                message: "Cannot normalize zero vector".to_string(),
            });
        }

        for value in vector.iter_mut() {
            *value /= norm;
        }

        Ok(())
    }

    /// Calculate the mean of multiple vectors
    pub fn mean_pooling(vectors: &[Vec<f32>]) -> Result<Vec<f32>, NLPError> {
        if vectors.is_empty() {
            return Err(NLPError::InvalidInput {
                message: "Cannot calculate mean of empty vector set".to_string(),
            });
        }

        let dim = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dim) {
            return Err(NLPError::InvalidInput {
                message: "All vectors must have the same dimension".to_string(),
            });
        }

        let mut mean = vec![0.0; dim];
        for vector in vectors {
            for (i, &value) in vector.iter().enumerate() {
                mean[i] += value;
            }
        }

        let count = vectors.len() as f32;
        for value in mean.iter_mut() {
            *value /= count;
        }

        Ok(mean)
    }
}

/// Unified device management for AI models
pub mod device {
    use crate::error::NLPError;
    use crate::models::DeviceType;
    
    #[cfg(feature = "real-ml")]
    use candle_core::Device;

    /// Create a Candle device based on the device type preference
    #[cfg(feature = "real-ml")]
    pub fn create_device(device_type: DeviceType) -> Result<Device, NLPError> {
        match device_type {
            DeviceType::CPU => Ok(Device::Cpu),
            DeviceType::CUDA => Device::new_cuda(0).map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to initialize CUDA device: {}", e),
            }),
            DeviceType::Metal => Device::new_metal(0).map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to initialize Metal device: {}", e),
            }),
            DeviceType::Auto => {
                // Try Metal first (for Apple Silicon), then CUDA, then CPU
                if Device::new_metal(0).is_ok() {
                    tracing::info!("Auto device selection: Using Metal (Apple Silicon)");
                    Device::new_metal(0).map_err(|e| NLPError::ModelLoading {
                        message: format!("Auto device selection failed: {}", e),
                    })
                } else if Device::new_cuda(0).is_ok() {
                    tracing::info!("Auto device selection: Using CUDA");
                    Device::new_cuda(0).map_err(|e| NLPError::ModelLoading {
                        message: format!("Auto device selection failed: {}", e),
                    })
                } else {
                    tracing::info!("Auto device selection: Using CPU");
                    Ok(Device::Cpu)
                }
            }
        }
    }

    /// Get device type from a Candle Device for reporting
    #[cfg(feature = "real-ml")]
    pub fn device_to_type(device: &Device) -> DeviceType {
        match device {
            Device::Cpu => DeviceType::CPU,
            Device::Cuda(_) => DeviceType::CUDA,
            Device::Metal(_) => DeviceType::Metal,
        }
    }

    /// Check if Metal acceleration is available
    #[cfg(feature = "real-ml")]
    pub fn is_metal_available() -> bool {
        Device::new_metal(0).is_ok()
    }

    /// Check if CUDA acceleration is available
    #[cfg(feature = "real-ml")]
    pub fn is_cuda_available() -> bool {
        Device::new_cuda(0).is_ok()
    }

    /// Get the best available device automatically
    #[cfg(feature = "real-ml")]
    pub fn best_available_device() -> Result<Device, NLPError> {
        create_device(DeviceType::Auto)
    }

    /// Detect the best available device for inference (stub version)
    #[cfg(not(feature = "real-ml"))]
    pub fn detect_best_device() -> DeviceType {
        DeviceType::CPU
    }

    /// Device memory information
    #[derive(Debug, Clone)]
    pub struct DeviceMemoryInfo {
        pub total_memory: Option<u64>,      // Total memory in bytes
        pub available_memory: Option<u64>,  // Available memory in bytes
        pub device_name: String,
    }

    /// Get device memory information (stub implementation)
    #[cfg(feature = "real-ml")]
    pub fn get_device_memory_info(device: &Device) -> DeviceMemoryInfo {
        match device {
            Device::Cpu => DeviceMemoryInfo {
                total_memory: None, // CPU memory is system dependent
                available_memory: None,
                device_name: "CPU".to_string(),
            },
            Device::Cuda(_) => DeviceMemoryInfo {
                total_memory: Some(8 * 1024 * 1024 * 1024), // Placeholder: 8GB
                available_memory: Some(6 * 1024 * 1024 * 1024), // Placeholder: 6GB
                device_name: "CUDA GPU".to_string(),
            },
            Device::Metal(_) => DeviceMemoryInfo {
                total_memory: Some(16 * 1024 * 1024 * 1024), // Placeholder: 16GB unified memory
                available_memory: Some(12 * 1024 * 1024 * 1024), // Placeholder: 12GB
                device_name: "Apple Silicon Metal".to_string(),
            },
        }
    }
}


/// Performance monitoring utilities
pub mod metrics {
    use std::time::{Duration, Instant};

    /// Simple performance timer
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }

        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        pub fn log_elapsed(&self) {
            tracing::debug!("{} took {:?}", self.name, self.elapsed());
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            self.log_elapsed();
        }
    }
}
