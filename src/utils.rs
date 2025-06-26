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

/// Smart link detection and injection utilities
pub mod links {
    use crate::{SmartLink, LinkType, NodeMetadata};
    use crate::error::NLPError;
    use regex::Regex;
    use std::collections::HashMap;

    /// Response processor for smart link injection
    pub struct ResponseProcessor {
        link_patterns: HashMap<LinkType, Vec<Regex>>,
    }

    impl ResponseProcessor {
        /// Create a new response processor with pre-compiled patterns
        pub fn new() -> Self {
            let mut link_patterns = HashMap::new();
            
            // Date patterns
            link_patterns.insert(LinkType::DateReference, vec![
                Regex::new(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b").unwrap(),
                Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").unwrap(),
                Regex::new(r"\b(?:today|yesterday|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b").unwrap(),
            ]);
            
            // Document patterns
            link_patterns.insert(LinkType::DocumentReference, vec![
                Regex::new(r"\b(?:proposal|document|report|notes?|meeting|presentation)\b").unwrap(),
                Regex::new(r"\b[A-Z][a-zA-Z0-9\s]{2,30}(?:proposal|report|document|notes?)\b").unwrap(),
            ]);
            
            // Task patterns
            link_patterns.insert(LinkType::TaskReference, vec![
                Regex::new(r"\b(?:task|todo|action item|follow[- ]up)\b").unwrap(),
                Regex::new(r"\b(?:complete|finish|do|handle)\s+[a-zA-Z0-9\s]{3,20}\b").unwrap(),
            ]);
            
            // Entity patterns (general)
            link_patterns.insert(LinkType::EntityReference, vec![
                Regex::new(r"\b[A-Z][a-zA-Z]{2,}\s+[A-Z][a-zA-Z]{2,}\b").unwrap(), // Person names
                Regex::new(r"\b[A-Z][a-zA-Z0-9\s&]{3,30}(?:Inc|LLC|Corp|Company|Ltd)\b").unwrap(), // Company names
            ]);
            
            Self { link_patterns }
        }

        /// Inject smart links into AI response content
        pub fn inject_smart_links(&self, content: &str, available_nodes: &[NodeMetadata]) -> Result<String, NLPError> {
            let mut enhanced_content = content.to_string();
            let detected_links = self.detect_potential_links(&enhanced_content, available_nodes)?;
            
            // Sort by confidence (highest first) to prioritize best matches
            let mut sorted_links = detected_links;
            sorted_links.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            
            // Apply links in reverse order to maintain string positions
            for link in sorted_links.iter().rev() {
                if link.confidence > 0.6 { // Only apply high-confidence links
                    let markdown_link = format!("[{}](nodespace://{})", link.text, link.node_id);
                    enhanced_content = enhanced_content.replace(&link.text, &markdown_link);
                }
            }
            
            Ok(enhanced_content)
        }

        /// Detect potential links in text based on available node metadata
        pub fn detect_potential_links(&self, content: &str, available_nodes: &[NodeMetadata]) -> Result<Vec<SmartLink>, NLPError> {
            let mut potential_links = Vec::new();
            
            for node in available_nodes {
                // Check for exact title matches (highest confidence)
                if content.contains(&node.title) {
                    potential_links.push(SmartLink {
                        text: node.title.clone(),
                        node_id: node.id.clone(),
                        link_type: self.classify_link_type(&node.node_type),
                        confidence: 0.95,
                    });
                }
                
                // Check for partial title matches
                let title_words: Vec<&str> = node.title.split_whitespace().collect();
                if title_words.len() > 1 {
                    for window in title_words.windows(2) {
                        let partial_title = window.join(" ");
                        if content.contains(&partial_title) && partial_title.len() > 4 {
                            potential_links.push(SmartLink {
                                text: partial_title,
                                node_id: node.id.clone(),
                                link_type: self.classify_link_type(&node.node_type),
                                confidence: 0.75,
                            });
                        }
                    }
                }
                
                // Check for pattern-based matches in snippet
                if let Some(pattern_match) = self.find_pattern_matches(content, &node.snippet, &node.node_type) {
                    potential_links.push(SmartLink {
                        text: pattern_match,
                        node_id: node.id.clone(),
                        link_type: self.classify_link_type(&node.node_type),
                        confidence: 0.65,
                    });
                }
            }
            
            // Remove duplicates and overlapping matches
            let deduplicated = self.deduplicate_links(potential_links)?;
            
            Ok(deduplicated)
        }

        /// Classify node type into LinkType
        fn classify_link_type(&self, node_type: &str) -> LinkType {
            match node_type.to_lowercase().as_str() {
                "customer" | "person" | "contact" | "company" => LinkType::EntityReference,
                "meeting" | "date" | "event" | "calendar" => LinkType::DateReference,
                "document" | "note" | "proposal" | "report" => LinkType::DocumentReference,
                "task" | "todo" | "action" | "item" => LinkType::TaskReference,
                _ => LinkType::EntityReference, // Default
            }
        }

        /// Find pattern-based matches between content and node data
        fn find_pattern_matches(&self, content: &str, snippet: &str, node_type: &str) -> Option<String> {
            let link_type = self.classify_link_type(node_type);
            
            if let Some(patterns) = self.link_patterns.get(&link_type) {
                for pattern in patterns {
                    if let Some(mat) = pattern.find(content) {
                        // Check if this match might relate to the node's snippet content
                        let matched_text = mat.as_str();
                        if self.content_similarity(matched_text, snippet) > 0.3 {
                            return Some(matched_text.to_string());
                        }
                    }
                }
            }
            
            None
        }

        /// Calculate content similarity between two strings (simple keyword overlap)
        fn content_similarity(&self, text1: &str, text2: &str) -> f32 {
            let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
            let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
            
            let intersection = words1.intersection(&words2).count();
            let union = words1.union(&words2).count();
            
            if union == 0 {
                0.0
            } else {
                intersection as f32 / union as f32
            }
        }

        /// Remove duplicate and overlapping smart links
        fn deduplicate_links(&self, mut links: Vec<SmartLink>) -> Result<Vec<SmartLink>, NLPError> {
            // Sort by text length (longer matches preferred)
            links.sort_by(|a, b| b.text.len().cmp(&a.text.len()));
            
            let mut deduplicated = Vec::new();
            let mut used_text_ranges = Vec::new();
            
            for link in links {
                let overlaps = used_text_ranges.iter().any(|used: &String| {
                    used.contains(&link.text) || link.text.contains(used)
                });
                
                if !overlaps {
                    used_text_ranges.push(link.text.clone());
                    deduplicated.push(link);
                }
            }
            
            Ok(deduplicated)
        }
    }

    impl Default for ResponseProcessor {
        fn default() -> Self {
            Self::new()
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
