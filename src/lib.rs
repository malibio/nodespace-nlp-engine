//! NodeSpace NLP Engine
//!
//! AI/ML processing and SurrealDB integration for NodeSpace.
//! Provides embedding generation, LLM integration, SurrealQL generation, and semantic processing.

use async_trait::async_trait;
use nodespace_core_types::NodeSpaceResult;
use serde::{Deserialize, Serialize};

// Re-export core types for convenience
pub use nodespace_core_types;

// Module declarations
pub mod embedding;
pub mod engine;
pub mod error;
pub mod models;
pub mod surrealql;
pub mod text_generation;
pub mod utils;

// Re-export main types for consumers
pub use engine::LocalNLPEngine;
pub use error::NLPError;

/// NLP Engine Service Interface
///
/// Minimal interface for AI/ML operations using Mistral.rs and embedding generation.
/// This is re-exported from the contracts for implementation.
#[async_trait]
pub trait NLPEngine: Send + Sync {
    /// Generate vector embedding for text content
    async fn generate_embedding(&self, text: &str) -> NodeSpaceResult<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch operation)
    async fn batch_embeddings(&self, texts: &[String]) -> NodeSpaceResult<Vec<Vec<f32>>>;

    /// Generate text using the local LLM (Mistral.rs)
    async fn generate_text(&self, prompt: &str) -> NodeSpaceResult<String>;

    /// Generate SurrealQL from natural language query
    async fn generate_surrealql(
        &self,
        natural_query: &str,
        schema_context: &str,
    ) -> NodeSpaceResult<String>;

    /// Get embedding model dimensions
    fn embedding_dimensions(&self) -> usize;
}

/// Request/response types for service boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateEmbeddingRequest {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSurrealQLRequest {
    pub natural_query: String,
    pub schema_context: String,
    pub safety_checks: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationResponse {
    pub text: String,
    pub tokens_used: u32,
}
