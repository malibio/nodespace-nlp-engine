//! Ollama HTTP client for real text generation
//! Connects to external Ollama server for production AI responses

use crate::error::NLPError;
use crate::models::OllamaConfig;
use crate::utils::metrics::Timer;
use crate::{
    ContextUtilization, EnhancedTextGenerationResponse, GenerationMetrics, RAGContext,
    TextGenerationRequest,
};

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "ollama")]
use reqwest::Client;

/// Ollama API request for text generation
#[derive(Debug, Clone, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: Option<OllamaOptions>,
}

/// Ollama generation options
#[derive(Debug, Clone, Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32, // max tokens
    top_p: Option<f32>,
}

/// Ollama API response for text generation
#[derive(Debug, Clone, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    done: bool,
    #[allow(dead_code)]
    total_duration: Option<u64>,
    #[allow(dead_code)]
    load_duration: Option<u64>,
    #[allow(dead_code)]
    prompt_eval_count: Option<u32>,
    #[allow(dead_code)]
    eval_count: Option<u32>,
}

/// Ollama multimodal request with image support
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize)]
struct OllamaMultimodalRequest {
    model: String,
    prompt: String,
    images: Vec<String>, // Base64 encoded images
    stream: bool,
    options: Option<OllamaOptions>,
}

/// Real Ollama HTTP client for text generation
#[cfg(feature = "ollama")]
pub struct OllamaTextGenerator {
    client: Client,
    config: OllamaConfig,
    initialized: bool,
}

#[cfg(feature = "ollama")]
impl OllamaTextGenerator {
    /// Create new Ollama text generator
    pub fn new(config: OllamaConfig) -> Result<Self, NLPError> {
        let timeout = Duration::from_secs(config.timeout_secs);
        let client = Client::builder()
            .timeout(timeout)
            .user_agent("NodeSpace-NLP-Engine/0.1.0")
            .build()
            .map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        Ok(Self {
            client,
            config,
            initialized: false,
        })
    }

    /// Initialize connection to Ollama server
    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        let _timer = Timer::new("ollama_initialization");

        if self.initialized {
            return Ok(());
        }

        tracing::info!("Initializing Ollama client...");
        tracing::info!("Ollama server: {}", self.config.base_url);
        tracing::info!("Default model: {}", self.config.default_model);

        // Test connection to Ollama server
        self.test_connection().await?;

        self.initialized = true;
        tracing::info!("✅ Ollama client initialized successfully");
        Ok(())
    }

    /// Test connection to Ollama server
    async fn test_connection(&self) -> Result<(), NLPError> {
        let url = format!("{}/api/tags", self.config.base_url);
        
        tracing::debug!("Testing Ollama connection: {}", url);
        
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NLPError::ModelLoading {
                message: format!("Failed to connect to Ollama server at {}: {}", self.config.base_url, e),
            })?;

        if !response.status().is_success() {
            return Err(NLPError::ModelLoading {
                message: format!(
                    "Ollama server returned error status: {} - {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                ),
            });
        }

        tracing::info!("✅ Ollama server connection successful");
        Ok(())
    }

    /// Generate text using Ollama API
    pub async fn generate_text(&self, prompt: &str) -> Result<String, NLPError> {
        self.generate_text_with_params(
            prompt,
            self.config.max_tokens as u32,
            self.config.temperature,
            0.9, // default top_p
        )
        .await
    }

    /// Generate text with custom parameters
    pub async fn generate_text_with_params(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, NLPError> {
        self.generate_with_model(&self.config.default_model, prompt, max_tokens, temperature, top_p)
            .await
    }

    /// Generate text with specific model
    pub async fn generate_with_model(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, NLPError> {
        let _timer = Timer::new("ollama_text_generation");

        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Ollama client not initialized".to_string(),
            });
        }

        tracing::debug!("Generating text with Ollama model: {}", model);
        tracing::debug!("Prompt length: {} chars", prompt.len());
        tracing::debug!("Max tokens: {}, Temperature: {}, Top-p: {}", max_tokens, temperature, top_p);

        let request = OllamaGenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: self.config.stream,
            options: Some(OllamaOptions {
                temperature,
                num_predict: max_tokens as i32,
                top_p: Some(top_p),
            }),
        };

        let url = format!("{}/api/generate", self.config.base_url);
        
        // Perform request with retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.retry_attempts {
            tracing::debug!("Ollama request attempt {}/{}", attempt, self.config.retry_attempts);
            
            match self.make_request(&url, &request).await {
                Ok(response) => {
                    tracing::info!("✅ Ollama text generation successful");
                    tracing::debug!("Response length: {} chars", response.len());
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.retry_attempts {
                        tracing::warn!("Ollama request failed (attempt {}), retrying...", attempt);
                        tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| NLPError::ProcessingError {
            message: "All retry attempts exhausted".to_string(),
        }))
    }

    /// Make HTTP request to Ollama API
    async fn make_request(
        &self,
        url: &str,
        request: &OllamaGenerateRequest,
    ) -> Result<String, NLPError> {
        let response = self
            .client
            .post(url)
            .json(request)
            .send()
            .await
            .map_err(|e| NLPError::ProcessingError {
                message: format!("HTTP request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(NLPError::ProcessingError {
                message: format!("Ollama API error {}: {}", status, error_text),
            });
        }

        let ollama_response: OllamaGenerateResponse = response
            .json()
            .await
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to parse Ollama response: {}", e),
            })?;

        if !ollama_response.done {
            tracing::warn!("Ollama response not marked as done - may be incomplete");
        }

        Ok(ollama_response.response)
    }

    /// Enhanced text generation with RAG context support
    pub async fn generate_text_enhanced(
        &self,
        request: TextGenerationRequest,
    ) -> Result<EnhancedTextGenerationResponse, NLPError> {
        let start_time = std::time::Instant::now();
        let _timer = Timer::new("ollama_enhanced_text_generation");

        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Ollama client not initialized".to_string(),
            });
        }

        // Phase 1: Context validation and prompt assembly
        let (enhanced_prompt, context_tokens) = self.assemble_rag_prompt(&request)?;

        // Phase 2: Token management - ensure we stay within context window
        let available_tokens = request
            .context_window
            .saturating_sub(context_tokens as usize);
        let max_response_tokens = std::cmp::min(request.max_tokens, available_tokens);

        if max_response_tokens < 10 {
            return Err(NLPError::ProcessingError {
                message: "Insufficient tokens available for response after RAG context".to_string(),
            });
        }

        // Phase 3: Generate text with conversation optimizations
        let generated_text = if request.conversation_mode {
            self.generate_conversational_text(
                &enhanced_prompt,
                max_response_tokens as u32,
                request.temperature,
            )
            .await?
        } else {
            self.generate_text_with_params(
                &enhanced_prompt,
                max_response_tokens as u32,
                request.temperature,
                0.9, // Default top_p
            )
            .await?
        };

        // Phase 4: Post-processing and quality validation
        let context_utilization =
            self.analyze_context_utilization(&generated_text, &request.rag_context);

        // Phase 5: Compile metrics and response
        let generation_time = start_time.elapsed();
        let generation_metrics = GenerationMetrics {
            generation_time_ms: generation_time.as_millis() as u64,
            context_tokens,
            response_tokens: self.estimate_token_count(&generated_text),
            temperature_used: request.temperature,
        };

        Ok(EnhancedTextGenerationResponse {
            text: generated_text,
            tokens_used: generation_metrics.response_tokens,
            generation_metrics,
            context_utilization,
        })
    }

    /// Generate multimodal response with image support
    #[cfg(feature = "multimodal")]
    pub async fn generate_multimodal_response(
        &self,
        request: crate::MultimodalRequest,
    ) -> Result<crate::MultimodalResponse, NLPError> {
        let _timer = Timer::new("ollama_multimodal_generation");

        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Ollama client not initialized".to_string(),
            });
        }

        // Encode images to base64
        let encoded_images: Result<Vec<String>, _> = request
            .images
            .iter()
            .map(|img| {
                use base64::prelude::*;
                Ok(BASE64_STANDARD.encode(&img.data))
            })
            .collect();

        let encoded_images = encoded_images.map_err(|e| NLPError::ProcessingError {
            message: format!("Failed to encode images: {:?}", e),
        })?;

        let multimodal_request = OllamaMultimodalRequest {
            model: self.config.multimodal_model.clone(),
            prompt: request.text_query.clone(),
            images: encoded_images,
            stream: self.config.stream,
            options: Some(OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens as i32,
                top_p: None,
            }),
        };

        let url = format!("{}/api/generate", self.config.base_url);

        let response = self
            .client
            .post(&url)
            .json(&multimodal_request)
            .send()
            .await
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Multimodal HTTP request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(NLPError::ProcessingError {
                message: format!("Ollama multimodal API error {}: {}", status, error_text),
            });
        }

        let ollama_response: OllamaGenerateResponse = response
            .json()
            .await
            .map_err(|e| NLPError::ProcessingError {
                message: format!("Failed to parse multimodal response: {}", e),
            })?;

        // Build image references
        let image_references: Vec<crate::ImageReference> = request
            .images
            .iter()
            .enumerate()
            .map(|(i, img)| crate::ImageReference {
                id: img.id.clone().unwrap_or_else(|| format!("img_{}", i)),
                description: img
                    .description
                    .clone()
                    .unwrap_or_else(|| "Image".to_string()),
                confidence: 0.8, // Default confidence
            })
            .collect();

        Ok(crate::MultimodalResponse {
            text: ollama_response.response,
            image_sources: image_references,
            smart_links: Vec::new(), // TODO: Extract from enhanced response
            generation_metrics: GenerationMetrics {
                generation_time_ms: ollama_response.total_duration.unwrap_or(0) / 1_000_000, // Convert from nanoseconds
                context_tokens: ollama_response.prompt_eval_count.unwrap_or(0),
                response_tokens: ollama_response.eval_count.unwrap_or(0),
                temperature_used: request.temperature,
            },
            image_utilization: crate::ImageUtilization {
                images_referenced: !request.images.is_empty(),
                images_used: request.images.len(),
                understanding_confidence: 0.7, // Placeholder
            },
        })
    }

    /// Get model information
    pub fn model_info(&self) -> crate::text_generation::TextGenerationModelInfo {
        crate::text_generation::TextGenerationModelInfo {
            model_name: self.config.default_model.clone(),
            max_context_length: self.config.max_tokens,
            device_type: crate::models::DeviceType::Auto, // Ollama handles device selection
        }
    }

    // Helper methods for RAG context processing

    /// Assemble enhanced prompt with RAG context
    fn assemble_rag_prompt(
        &self,
        request: &TextGenerationRequest,
    ) -> Result<(String, u32), NLPError> {
        let mut enhanced_prompt = String::new();

        // Add RAG context if provided
        if let Some(ref rag_context) = request.rag_context {
            enhanced_prompt.push_str("# Context Information\n\n");
            enhanced_prompt.push_str(&format!("Summary: {}\n\n", rag_context.context_summary));

            if !rag_context.knowledge_sources.is_empty() {
                enhanced_prompt.push_str("Sources:\n");
                for (i, source) in rag_context.knowledge_sources.iter().enumerate() {
                    enhanced_prompt.push_str(&format!("{}. {}\n", i + 1, source));
                }
                enhanced_prompt.push('\n');
            }

            enhanced_prompt.push_str("# Query\n\n");
        }

        // Add the main prompt
        enhanced_prompt.push_str(&request.prompt);

        // Add conversation formatting if in conversation mode
        if request.conversation_mode {
            enhanced_prompt.push_str(
                "\n\nPlease respond naturally and reference the provided context where relevant.",
            );
        }

        let total_tokens = self.estimate_token_count(&enhanced_prompt);
        Ok((enhanced_prompt, total_tokens))
    }

    /// Generate text optimized for conversational responses
    async fn generate_conversational_text(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String, NLPError> {
        // Use slightly higher temperature for more natural conversation
        let conv_temperature = (temperature + 0.1).min(1.0);
        let conv_top_p = 0.95; // Encourage diverse but coherent responses

        self.generate_text_with_params(prompt, max_tokens, conv_temperature, conv_top_p)
            .await
    }

    /// Analyze how well the generated text utilized the provided context
    fn analyze_context_utilization(
        &self,
        generated_text: &str,
        rag_context: &Option<RAGContext>,
    ) -> ContextUtilization {
        let mut context_referenced = false;
        let mut sources_mentioned = Vec::new();
        let mut relevance_score = 0.0;

        if let Some(ref context) = rag_context {
            // Check if response references context keywords
            let context_keywords: Vec<&str> = context
                .context_summary
                .split_whitespace()
                .filter(|word| word.len() > 4) // Only meaningful words
                .collect();

            let generated_lower = generated_text.to_lowercase();
            let mut keyword_matches = 0;

            for keyword in &context_keywords {
                if generated_lower.contains(&keyword.to_lowercase()) {
                    keyword_matches += 1;
                    context_referenced = true;
                }
            }

            // Check for source references
            for (i, source) in context.knowledge_sources.iter().enumerate() {
                let source_keywords: Vec<&str> = source
                    .split_whitespace()
                    .filter(|word| word.len() > 4)
                    .take(3) // First few significant words
                    .collect();

                for keyword in source_keywords {
                    if generated_lower.contains(&keyword.to_lowercase()) {
                        sources_mentioned.push(format!("Source {}", i + 1));
                        break;
                    }
                }
            }

            // Calculate relevance score
            if !context_keywords.is_empty() {
                relevance_score = (keyword_matches as f32) / (context_keywords.len() as f32);
                relevance_score = relevance_score.min(1.0);

                // Boost score if sources were mentioned
                if !sources_mentioned.is_empty() {
                    relevance_score = (relevance_score + 0.2).min(1.0);
                }
            }
        }

        ContextUtilization {
            context_referenced,
            sources_mentioned,
            relevance_score,
        }
    }

    /// Estimate token count for text (rough approximation)
    fn estimate_token_count(&self, text: &str) -> u32 {
        // Rough approximation: 1 token ≈ 0.75 words for English
        let word_count = text.split_whitespace().count();
        ((word_count as f32) / 0.75) as u32
    }
}

// Stub implementation when ollama feature is disabled
#[cfg(not(feature = "ollama"))]
pub struct OllamaTextGenerator;

#[cfg(not(feature = "ollama"))]
impl OllamaTextGenerator {
    pub fn new(_config: OllamaConfig) -> Result<Self, NLPError> {
        Err(NLPError::ModelLoading {
            message: "Ollama feature not enabled. Compile with --features ollama".to_string(),
        })
    }

    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        Err(NLPError::ModelLoading {
            message: "Ollama feature not enabled".to_string(),
        })
    }

    pub async fn generate_text(&self, _prompt: &str) -> Result<String, NLPError> {
        Err(NLPError::ModelLoading {
            message: "Ollama feature not enabled".to_string(),
        })
    }
}