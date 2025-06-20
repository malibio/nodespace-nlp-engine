//! Error types for the NLP Engine

use thiserror::Error;

/// Errors that can occur in the NLP Engine
#[derive(Error, Debug)]
pub enum NLPError {
    #[error("Model loading failed: {message}")]
    ModelLoading { message: String },

    #[error("Inference failed: {message}")]
    Inference { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Tokenization error: {message}")]
    Tokenization { message: String },

    #[error("Embedding generation failed: {message}")]
    EmbeddingGeneration { message: String },

    #[error("Text generation failed: {message}")]
    TextGeneration { message: String },

    #[error("SurrealQL generation failed: {message}")]
    SurrealQLGeneration { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: serde_json::Error,
    },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Device error: {message}")]
    Device { message: String },

    #[error("Batch processing error: {message}")]
    BatchProcessing { message: String },
}

impl From<NLPError> for nodespace_core_types::NodeSpaceError {
    fn from(err: NLPError) -> Self {
        nodespace_core_types::NodeSpaceError::ProcessingError(err.to_string())
    }
}