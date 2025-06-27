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
#[cfg(feature = "evaluation")]
pub mod evaluation;
#[cfg(feature = "multimodal")]
pub mod image_processing;
pub mod models;
pub mod surrealql;
pub mod text_generation;
pub mod utils;

// Re-export main types for consumers
pub use engine::LocalNLPEngine;
pub use error::NLPError;

// Re-export evaluation types when feature is enabled
#[cfg(feature = "evaluation")]
pub use evaluation::{
    BLEUConfig, BLEUScores, EvaluationFramework, EvaluationReport, RAGEvaluationResult,
    ROUGEConfig, ROUGEScores, SemanticSearchEvaluation, SimilarityScores,
};

// Re-export configuration types for external configuration
pub use models::{
    CacheConfig, DeviceConfig, DeviceType, EmbeddingModelConfig, ModelConfigs, ModelInfo,
    NLPConfig, PerformanceConfig, TextGenerationModelConfig,
};

// Re-export smart link utilities for intelligent response processing
pub use utils::links::ResponseProcessor;

/// NLP Engine Service Interface
///
/// Interface for AI/ML operations with RAG context-aware generation support.
/// NLP Engine Service Interface - owned and exported by this repository.
#[async_trait]
pub trait NLPEngine: Send + Sync {
    /// Generate vector embedding for text content
    async fn generate_embedding(&self, text: &str) -> NodeSpaceResult<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch operation)
    async fn batch_embeddings(&self, texts: &[String]) -> NodeSpaceResult<Vec<Vec<f32>>>;

    /// Generate text using the local LLM (backward compatibility)
    async fn generate_text(&self, prompt: &str) -> NodeSpaceResult<String>;

    /// Enhanced text generation with RAG context support
    async fn generate_text_enhanced(
        &self,
        request: TextGenerationRequest,
    ) -> NodeSpaceResult<EnhancedTextGenerationResponse>;
    /// Generate SurrealQL from natural language query
    async fn generate_surrealql(
        &self,
        natural_query: &str,
        schema_context: &str,
    ) -> NodeSpaceResult<String>;

    /// Get embedding model dimensions
    fn embedding_dimensions(&self) -> usize;

    /// Generate vector embedding for image content (multimodal)
    #[cfg(feature = "multimodal")]
    async fn generate_image_embedding(&self, image_data: &[u8]) -> NodeSpaceResult<Vec<f32>>;

    /// Extract comprehensive metadata from image
    #[cfg(feature = "multimodal")]
    async fn extract_image_metadata(&self, image_data: &[u8]) -> NodeSpaceResult<ImageMetadata>;

    /// Generate multimodal response with text and image understanding
    #[cfg(feature = "multimodal")]
    async fn generate_multimodal_response(
        &self,
        request: MultimodalRequest,
    ) -> NodeSpaceResult<MultimodalResponse>;
}

/// Future-ready streaming interface (for future implementation)
///
/// This trait is prepared for streaming text generation but not yet implemented.
/// Will be activated when streaming support is added to the underlying models.
#[async_trait]
pub trait StreamingNLPEngine: NLPEngine {
    // Future streaming method - commented out until streaming infrastructure is ready
    // async fn generate_text_stream(
    //     &self,
    //     request: TextGenerationRequest,
    // ) -> NodeSpaceResult<Box<dyn Stream<Item = NodeSpaceResult<TextChunk>> + Send + Unpin>>;
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

/// Node metadata for smart link generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub id: nodespace_core_types::NodeId,
    pub title: String,
    pub node_type: String, // "customer", "date", "task", etc.
    pub created_date: String,
    pub snippet: String, // Brief content preview
}

/// Smart link types for different kinds of references
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LinkType {
    EntityReference,   // Customer, project, person
    DateReference,     // Specific dates or meetings
    DocumentReference, // Notes, proposals, documents
    TaskReference,     // Action items, todos
}

/// Smart link structure for generated links in AI responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartLink {
    pub text: String,                          // Display text
    pub node_id: nodespace_core_types::NodeId, // Target node
    pub link_type: LinkType,                   // Reference, Date, Entity
    pub confidence: f32,                       // Link relevance score
}

/// Enhanced text generation request with RAG context and smart link support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationRequest {
    pub prompt: String,                   // Complete prompt with RAG context
    pub max_tokens: usize,                // Response length limit
    pub temperature: f32,                 // Response creativity
    pub context_window: usize,            // Total context tokens
    pub conversation_mode: bool,          // Optimize for dialogue
    pub rag_context: Option<RAGContext>,  // Knowledge context metadata
    pub enable_link_generation: bool,     // NEW: Enable smart link generation
    pub node_metadata: Vec<NodeMetadata>, // NEW: Available nodes for linking
}

/// RAG context metadata for enhanced generation with smart linking capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGContext {
    pub knowledge_sources: Vec<String>,  // Source summaries
    pub retrieval_confidence: f32,       // Overall relevance
    pub context_summary: String,         // What context includes
    pub suggested_links: Vec<SmartLink>, // NEW: Generated smart links
}

/// Enhanced text generation response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTextGenerationResponse {
    pub text: String,
    pub tokens_used: u32,
    pub generation_metrics: GenerationMetrics,
    pub context_utilization: ContextUtilization,
}

/// Generation quality and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub generation_time_ms: u64,
    pub context_tokens: u32,
    pub response_tokens: u32,
    pub temperature_used: f32,
}

/// Analysis of how well the generated text used provided context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUtilization {
    pub context_referenced: bool,
    pub sources_mentioned: Vec<String>,
    pub relevance_score: f32,
}

// Future-ready streaming support types
/// Streaming text generation chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub content: String,
    pub is_final: bool,
    pub token_count: usize,
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

/// Image metadata extracted from EXIF and analysis
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Image dimensions (width, height)
    pub dimensions: (u32, u32),
    /// File format (JPEG, PNG, etc.)
    pub format: String,
    /// File size in bytes
    pub file_size: usize,
    /// EXIF timestamp if available
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// GPS coordinates if available (latitude, longitude)
    pub gps_coordinates: Option<(f64, f64)>,
    /// Camera information from EXIF
    pub camera_info: Option<CameraInfo>,
    /// Color space information
    pub color_space: Option<String>,
    /// Image orientation
    pub orientation: Option<u8>,
    /// Processing performance metrics
    pub processing_time_ms: u64,
}

/// Camera information extracted from EXIF data
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    /// Camera make (e.g., "Canon", "Apple")
    pub make: Option<String>,
    /// Camera model (e.g., "iPhone 15 Pro", "EOS R5")
    pub model: Option<String>,
    /// Lens information
    pub lens_model: Option<String>,
    /// Focal length in mm
    pub focal_length: Option<f32>,
    /// Aperture f-number
    pub aperture: Option<f32>,
    /// ISO sensitivity
    pub iso: Option<u32>,
    /// Exposure time in seconds
    pub exposure_time: Option<f32>,
    /// Flash information
    pub flash: Option<bool>,
}

/// Multimodal request combining text and images
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalRequest {
    /// Text query or prompt
    pub text_query: String,
    /// Image data for analysis
    pub images: Vec<ImageInput>,
    /// Context from previous conversation
    pub context_nodes: Vec<nodespace_core_types::NodeId>,
    /// Enable smart link generation in response
    pub enable_smart_links: bool,
    /// Maximum tokens for response
    pub max_tokens: usize,
    /// Generation temperature
    pub temperature: f32,
}

/// Input image with optional metadata
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    /// Raw image data
    pub data: Vec<u8>,
    /// Optional description or context
    pub description: Option<String>,
    /// Image identifier for reference
    pub id: Option<String>,
}

/// Multimodal response with text and image references
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalResponse {
    /// Generated text response
    pub text: String,
    /// Images referenced in the response
    pub image_sources: Vec<ImageReference>,
    /// Smart links generated in response
    pub smart_links: Vec<SmartLink>,
    /// Performance and quality metrics
    pub generation_metrics: GenerationMetrics,
    /// How well images were understood and used
    pub image_utilization: ImageUtilization,
}

/// Reference to an image used in multimodal response
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageReference {
    /// Image identifier
    pub id: String,
    /// Description of what was understood from the image
    pub description: String,
    /// Confidence in image understanding
    pub confidence: f32,
}

/// Analysis of how well images were utilized in response
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUtilization {
    /// Whether images were referenced in response
    pub images_referenced: bool,
    /// Number of images actually used
    pub images_used: usize,
    /// Overall confidence in image understanding
    pub understanding_confidence: f32,
}
