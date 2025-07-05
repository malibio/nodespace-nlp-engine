# NodeSpace NLP Engine

**Multimodal AI/ML processing for NodeSpace - text embeddings, image understanding, and semantic search**

## Overview

The NodeSpace NLP Engine provides AI/ML capabilities for the NodeSpace system, using FastEmbed for embeddings and Ollama for text generation. Currently in Phase 1 (text-only) with multimodal capabilities planned for Phase 2.

### Current Features (Phase 1)
- **Text embeddings** - BGE-small model via FastEmbed (384 dimensions)
- **Text generation** - Gemma 3 model via Ollama HTTP API
- **Semantic search** - Vector similarity and caching
- **Apple MPS acceleration** - FastEmbed optimized for Apple Silicon
- **Evaluation framework** - Basic quality assessment (ROUGE/BLEU in development)

### External Dependencies
- **Ollama server** - Required for text generation (runs locally on port 11434)

### Planned Features (Phase 2)
- **Image embeddings** - CLIP model for visual content
- **Visual Q&A** - Gemma 3 multimodal integration
- **Cross-modal search** - Text-to-image and image-to-text retrieval
- **PDF processing** - Text and image extraction

## Quick Start

### Prerequisites
```bash
# Install and start Ollama for text generation
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:12b
```

### Installation
```bash
# Add to your Cargo.toml
[dependencies]
nodespace-nlp-engine = { git = "https://github.com/malibio/nodespace-nlp-engine" }
nodespace-core-types = { git = "https://github.com/malibio/nodespace-core-types" }
```

```rust
use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = LocalNLPEngine::new();
    engine.initialize().await?;
    
    // Generate text embedding
    let embedding = engine.generate_embedding("Meeting notes about Q3 planning").await?;
    println!("Generated embedding with {} dimensions", embedding.len());
    
    Ok(())
}
```

## Development

### Setup
```bash
# Clone the repository
git clone https://github.com/malibio/nodespace-nlp-engine
cd nodespace-nlp-engine

# Run tests
cargo test

# Run examples
cargo run --example generate_embeddings
cargo run --example text_generation

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Technology Stack
- **Language**: Rust 1.86.0
- **AI/ML**: FastEmbed 4.9 (embeddings), Ollama HTTP API (text generation)
- **Models**: BGE-small-en-v1.5 (text), Gemma 3 (generation via Ollama)
- **Hardware**: Apple MPS via FastEmbed, Ollama external processing
- **Testing**: Unit tests and integration tests

## Architecture

The engine implements the `NLPEngine` trait from [nodespace-core-types](https://github.com/malibio/nodespace-core-types) and provides:

- **Embedding Generation** - Text-to-vector conversion with caching
- **Text Generation** - LLM inference via Ollama HTTP API
- **Multi-level Embeddings** - Individual, contextual, and hierarchical embeddings
- **Performance Optimization** - Lazy initialization and smart caching

### Core Components
- `LocalNLPEngine` - Main engine implementation
- `EmbeddingGenerator` - FastEmbed integration for text embeddings
- `TextGenerator` - Ollama HTTP API integration for LLM inference
- `MultiLevelEmbeddingGenerator` - Contextual and hierarchical embedding support

## Current Status

**Phase 1: Mostly Complete**
- Text embeddings working with real BGE-small model
- Text generation with Gemma 3 via Ollama HTTP API
- Full trait compliance with nodespace-core-types
- Comprehensive test suite with 19 passing tests
- Basic evaluation framework (ROUGE/BLEU stubs)

**Phase 2: In Progress**
- Image embeddings via CLIP model (in development)
- Multimodal LLM with Gemma 3 (in development)
- Cross-modal search capabilities (planned)
- PDF processing pipeline (planned)

## Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test trait_compliance
cargo test --test test_ollama_integration

# Basic evaluation tests
cargo test --features evaluation

# Examples
cargo run --example generate_embeddings
cargo run --example text_generation
```

## Architecture Context

Part of the NodeSpace system architecture:

1. [nodespace-core-types](https://github.com/malibio/nodespace-core-types) - Shared data structures and interfaces
2. [nodespace-data-store](https://github.com/malibio/nodespace-data-store) - Vector storage and retrieval
3. **[nodespace-nlp-engine](https://github.com/malibio/nodespace-nlp-engine)** ← **You are here** (AI/ML processing and LLM integration)
4. [nodespace-workflow-engine](https://github.com/malibio/nodespace-workflow-engine) - Automation and event processing
5. [nodespace-core-logic](https://github.com/malibio/nodespace-core-logic) - Business logic orchestration
6. [nodespace-core-ui](https://github.com/malibio/nodespace-core-ui) - React components and UI
7. [nodespace-desktop-app](https://github.com/malibio/nodespace-desktop-app) - Tauri application shell

## License

See LICENSE file for details.