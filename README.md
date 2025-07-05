# NodeSpace NLP Engine

**Multimodal AI/ML processing for NodeSpace - text embeddings, image understanding, and semantic search**

## 🎯 Overview

The NodeSpace NLP Engine provides AI/ML capabilities for the NodeSpace system, focusing on multimodal processing with ONNX Runtime and Apple MPS acceleration. Currently in Phase 1 (text-only) with multimodal capabilities planned for Phase 2.

### Current Features (Phase 1)
- **Text embeddings** - BGE-small model via FastEmbed (384 dimensions)
- **Text generation** - Gemma 3 1B IT ONNX model with local inference
- **Semantic search** - Vector similarity and caching
- **Apple MPS acceleration** - ONNX Runtime CoreML execution provider
- **Evaluation framework** - ROUGE/BLEU metrics for quality assessment

### Planned Features (Phase 2)
- **Image embeddings** - CLIP model for visual content
- **Visual Q&A** - Phi-4 multimodal ONNX integration
- **Cross-modal search** - Text-to-image and image-to-text retrieval
- **PDF processing** - Text and image extraction

## 🚀 Quick Start

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

## 🛠️ Development

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
- **Language**: Rust 1.88.0
- **AI/ML**: ONNX Runtime 2.0.0-rc.10, FastEmbed 4.9.1
- **Models**: BGE-small-en-v1.5 (text), Gemma 3 1B IT (generation)
- **Hardware**: Apple MPS via ONNX Runtime CoreML EP
- **Testing**: ROUGE/BLEU evaluation metrics

## 🏗️ Architecture

The engine implements the `NLPEngine` trait from [nodespace-core-types](https://github.com/malibio/nodespace-core-types) and provides:

- **Embedding Generation** - Text-to-vector conversion with caching
- **Text Generation** - LLM inference with ONNX Runtime
- **Multi-level Embeddings** - Individual, contextual, and hierarchical embeddings
- **Performance Optimization** - Lazy initialization and smart caching

### Core Components
- `LocalNLPEngine` - Main engine implementation
- `EmbeddingGenerator` - FastEmbed integration for text embeddings
- `TextGenerator` - ONNX Runtime integration for LLM inference
- `MultiLevelEmbeddingGenerator` - Contextual and hierarchical embedding support

## 📊 Current Status

**Phase 1: Complete ✅**
- Text embeddings working with real BGE-small model
- Text generation with Gemma 3 1B ONNX model
- Full trait compliance with nodespace-core-types
- Comprehensive test suite with 22 passing tests
- ROUGE/BLEU evaluation framework

**Phase 2: Planned 📋**
- Image embeddings via CLIP model
- Multimodal LLM with Phi-4
- Cross-modal search capabilities
- PDF processing pipeline

## 🧪 Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test trait_compliance
cargo test --test test_ollama_integration

# Evaluation framework
cargo test --features evaluation

# Examples
cargo run --example generate_embeddings
cargo run --example text_generation
```

## 🏗️ Architecture Context

Part of the NodeSpace system architecture:

1. [nodespace-core-types](https://github.com/malibio/nodespace-core-types) - Shared data structures and interfaces
2. [nodespace-data-store](https://github.com/malibio/nodespace-data-store) - Vector storage and retrieval
3. **[nodespace-nlp-engine](https://github.com/malibio/nodespace-nlp-engine)** ← **You are here** (AI/ML processing and LLM integration)
4. [nodespace-workflow-engine](https://github.com/malibio/nodespace-workflow-engine) - Automation and event processing
5. [nodespace-core-logic](https://github.com/malibio/nodespace-core-logic) - Business logic orchestration
6. [nodespace-core-ui](https://github.com/malibio/nodespace-core-ui) - React components and UI
7. [nodespace-desktop-app](https://github.com/malibio/nodespace-desktop-app) - Tauri application shell

## 📝 License

See LICENSE file for details.