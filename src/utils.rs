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

        // Normalize punctuation

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
    use crate::models::DeviceType;

    /// Get device info for logging
    #[cfg(feature = "real-ml")]
    pub fn get_device_info(device_type: DeviceType) -> String {
        match device_type {
            DeviceType::CPU => "CPU".to_string(),
            DeviceType::CUDA => "CUDA (if available)".to_string(),
            DeviceType::Metal => "Metal (if available)".to_string(),
            DeviceType::Auto => "Auto (fastembed selects optimal)".to_string(),
        }
    }

    /// Stub device info for non-ML builds
    #[cfg(not(feature = "real-ml"))]
    pub fn get_device_info(device_type: DeviceType) -> String {
        format!("STUB: {:?}", device_type)
    }

    /// Check if Metal acceleration is available
    #[cfg(feature = "real-ml")]
    pub fn is_metal_available() -> bool {
        cfg!(target_os = "macos")
    }

    /// Check if CUDA acceleration is available
    #[cfg(feature = "real-ml")]
    pub fn is_cuda_available() -> bool {
        false // Conservative default
    }

    /// Get the best available device type automatically  
    #[cfg(feature = "real-ml")]
    pub fn best_available_device_type() -> DeviceType {
        if is_metal_available() {
            DeviceType::Metal
        } else if is_cuda_available() {
            DeviceType::CUDA
        } else {
            DeviceType::CPU
        }
    }

    /// Detect the best available device for inference (stub version)
    #[cfg(not(feature = "real-ml"))]
    pub fn detect_best_device() -> DeviceType {
        DeviceType::CPU
    }

    /// Device memory information
    #[derive(Debug, Clone)]
    pub struct DeviceMemoryInfo {
        pub total_memory: Option<u64>,     // Total memory in bytes
        pub available_memory: Option<u64>, // Available memory in bytes
        pub device_name: String,
    }

    /// Get device memory information
    #[cfg(feature = "real-ml")]
    pub fn get_device_memory_info(device_type: DeviceType) -> DeviceMemoryInfo {
        match device_type {
            DeviceType::CPU => DeviceMemoryInfo {
                total_memory: None, // CPU memory is system dependent
                available_memory: None,
                device_name: "CPU".to_string(),
            },
            DeviceType::CUDA => DeviceMemoryInfo {
                total_memory: Some(8 * 1024 * 1024 * 1024), // Placeholder: 8GB
                available_memory: Some(6 * 1024 * 1024 * 1024), // Placeholder: 6GB
                device_name: "CUDA GPU".to_string(),
            },
            DeviceType::Metal => DeviceMemoryInfo {
                total_memory: Some(16 * 1024 * 1024 * 1024), // Placeholder: 16GB unified memory
                available_memory: Some(12 * 1024 * 1024 * 1024), // Placeholder: 12GB
                device_name: "Apple Silicon Metal".to_string(),
            },
            DeviceType::Auto => DeviceMemoryInfo {
                total_memory: None,
                available_memory: None,
                device_name: "Auto-selected".to_string(),
            },
        }
    }
}

/// Token counting and budget management utilities for RAG
pub mod tokens {
    use crate::error::NLPError;

    /// Token budget management for RAG context assembly
    #[derive(Debug, Clone)]
    pub struct TokenBudget {
        pub total_available: usize,        // Model's context window
        pub reserved_for_response: usize,  // Tokens reserved for AI response
        pub system_prompt: usize,          // Tokens used by system instructions
        pub conversation_history: usize,   // Tokens for chat history
        pub knowledge_context: usize,      // Tokens for retrieved information
    }

    impl TokenBudget {
        /// Create a new token budget with default allocations
        pub fn new(total_context_window: usize) -> Self {
            let reserved_for_response = (total_context_window as f32 * 0.25) as usize; // 25% for response
            let system_prompt = 200; // Typical system prompt size
            
            Self {
                total_available: total_context_window,
                reserved_for_response,
                system_prompt,
                conversation_history: 0,
                knowledge_context: 0,
            }
        }

        /// Calculate remaining tokens available for context
        pub fn available_for_context(&self) -> usize {
            self.total_available
                .saturating_sub(self.reserved_for_response)
                .saturating_sub(self.system_prompt)
        }

        /// Update conversation history token usage
        pub fn set_conversation_tokens(&mut self, tokens: usize) {
            self.conversation_history = tokens;
        }

        /// Update knowledge context token usage
        pub fn set_knowledge_tokens(&mut self, tokens: usize) {
            self.knowledge_context = tokens;
        }

        /// Check if the current allocation is within budget
        pub fn is_within_budget(&self) -> bool {
            let used = self.reserved_for_response + self.system_prompt + 
                      self.conversation_history + self.knowledge_context;
            used <= self.total_available
        }

        /// Get the excess tokens if over budget
        pub fn excess_tokens(&self) -> usize {
            let used = self.reserved_for_response + self.system_prompt + 
                      self.conversation_history + self.knowledge_context;
            used.saturating_sub(self.total_available)
        }
    }

    /// Estimate token count for text (rough approximation)
    /// Note: This is a simple estimation. For precise counting, use the actual tokenizer.
    pub fn estimate_token_count(text: &str) -> usize {
        // Rough estimation: ~4 characters per token for English text
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();
        
        // Use the maximum of char/4 and word*1.3 as a conservative estimate
        let char_estimate = (char_count as f32 / 4.0).ceil() as usize;
        let word_estimate = (word_count as f32 * 1.3).ceil() as usize;
        
        char_estimate.max(word_estimate)
    }

    /// Truncate text to fit within token budget
    pub fn truncate_to_token_budget(text: &str, max_tokens: usize) -> String {
        let estimated_tokens = estimate_token_count(text);
        
        if estimated_tokens <= max_tokens {
            return text.to_string();
        }

        // Calculate approximate character limit
        let target_chars = (max_tokens * 4).saturating_sub(10); // Small buffer
        
        if text.len() <= target_chars {
            return text.to_string();
        }

        // Find the last complete word within the character limit
        let truncated = &text[..target_chars];
        if let Some(last_space) = truncated.rfind(' ') {
            truncated[..last_space].to_string()
        } else {
            truncated.to_string()
        }
    }

    /// Prioritize and truncate multiple text segments to fit budget
    pub fn allocate_budget_to_segments(
        segments: Vec<(String, f32)>, // (text, priority_score)
        total_budget: usize,
    ) -> Result<Vec<String>, NLPError> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by priority (highest first)
        let mut prioritized: Vec<_> = segments.into_iter().enumerate().collect();
        prioritized.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = Vec::new();
        let mut remaining_budget = total_budget;

        for (_, (text, _priority)) in prioritized {
            let estimated_tokens = estimate_token_count(&text);
            
            if estimated_tokens <= remaining_budget {
                // Include the full text
                remaining_budget = remaining_budget.saturating_sub(estimated_tokens);
                result.push(text);
            } else if remaining_budget > 10 {
                // Truncate to fit remaining budget
                let truncated = truncate_to_token_budget(&text, remaining_budget);
                result.push(truncated);
                break; // No more budget available
            } else {
                break; // Not enough budget for any more content
            }
        }

        Ok(result)
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
