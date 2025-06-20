//! Utility functions for the NLP Engine

use crate::error::NLPError;

/// Text preprocessing utilities
pub mod text {
    use regex::Regex;
    
    /// Clean and normalize text for embedding generation
    pub fn preprocess_for_embedding(text: &str) -> String {
        let mut processed = text.trim().to_lowercase();
        
        // Remove extra whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        processed = whitespace_regex.replace_all(&processed, " ").to_string();
        
        // Normalize punctuation
        processed = processed.replace("'", "'");
        processed = processed.replace('"', "\"");
        processed = processed.replace('"', "\"");
        
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

/// Device detection and management
pub mod device {
    use crate::models::DeviceType;
    
    /// Detect the best available device for inference
    pub fn detect_best_device() -> DeviceType {
        // Check for CUDA
        #[cfg(feature = "cuda")]
        {
            if is_cuda_available() {
                return DeviceType::CUDA;
            }
        }
        
        // Check for Metal (macOS)
        #[cfg(feature = "metal")]
        {
            if is_metal_available() {
                return DeviceType::Metal;
            }
        }
        
        DeviceType::CPU
    }
    
    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> bool {
        // Simple check - in real implementation, would use CUDA runtime API
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }
    
    #[cfg(feature = "metal")]
    fn is_metal_available() -> bool {
        // Check if we're on macOS with Metal support
        cfg!(target_os = "macos")
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