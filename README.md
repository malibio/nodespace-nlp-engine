# NodeSpace NLP Engine

**AI/ML processing and LLM integration for NodeSpace**

This repository implements the complete AI/ML layer for NodeSpace, providing embedding generation, LLM integration, and semantic processing capabilities. It serves as the **intelligence layer** of the distributed system.

## 🎯 Purpose

- **Embedding generation** - Convert text content to vector representations
- **LLM integration** - Interface with large language models for RAG queries
- **Semantic processing** - Advanced text analysis and understanding
- **AI orchestration** - Manage AI model interactions and caching

## 📦 Key Features

- **Multiple embedding models** - Support for various embedding backends
- **LLM abstraction** - Unified interface for different language models
- **Async processing** - Background embedding generation for performance
- **Smart caching** - Optimize AI model usage and reduce latency
- **Batch processing** - Efficient handling of multiple requests

## 🔗 Dependencies

- **`nodespace-core-types`** - Data structures and `NLPEngine` trait interface
- **Embedding libraries** - Various embedding model integrations
- **LLM APIs** - OpenAI, Anthropic, or local model interfaces

## 🏗️ Architecture Context

Part of the [NodeSpace system architecture](https://github.com/malibio/nodespace-system-design):

1. `nodespace-core-types` - Shared data structures and interfaces
2. `nodespace-data-store` - Database and vector storage
3. **`nodespace-nlp-engine`** ← **You are here**
4. `nodespace-workflow-engine` - Automation and event processing
5. `nodespace-core-logic` - Business logic orchestration
6. `nodespace-core-ui` - React components and UI
7. `nodespace-desktop-app` - Tauri application shell

## 🚀 Getting Started

```bash
# Add to your Cargo.toml
[dependencies]
nodespace-nlp-engine = { git = "https://github.com/malibio/nodespace-nlp-engine" }

# Use in your code
use nodespace_nlp_engine::LocalNLPEngine;
use nodespace_core_types::{NLPEngine, EmbeddingRequest};

let engine = LocalNLPEngine::new().await?;
let embedding = engine.generate_embedding(request).await?;
```

## 🔄 MVP Implementation

The initial implementation focuses on the core RAG workflow:

1. **Generate embeddings** - Convert text to vector representations
2. **LLM queries** - Process RAG requests with context
3. **Async processing** - Background embedding generation
4. **Error handling** - Robust AI service error management

## 🧪 Testing

```bash
# Run all tests including AI model tests
cargo test

# Test with sample content
cargo run --example generate_embeddings

# Benchmark AI performance
cargo bench
```

## 📋 Development Status

- [ ] Implement `NLPEngine` trait from core-types
- [ ] Set up embedding model integration
- [ ] Add LLM client implementation
- [ ] Implement async processing pipeline
- [ ] Add comprehensive test suite
- [ ] Performance optimization and caching

---

**Project Management:** All tasks tracked in [NodeSpace Project](https://github.com/users/malibio/projects/4)