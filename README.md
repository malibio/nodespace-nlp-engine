# NodeSpace NLP Engine

**AI/ML processing and SurrealDB integration for NodeSpace**

This repository implements the complete AI/ML layer for NodeSpace, providing embedding generation, LLM integration, SurrealQL generation, and semantic processing capabilities. It serves as the **intelligence layer** of the distributed system with native SurrealDB integration.

## 🎯 Purpose

- **Embedding generation** - Convert text content to SurrealDB vector<float, DIM> format
- **LLM integration** - Interface with large language models for RAG queries
- **SurrealQL generation** - Convert natural language to safe SurrealQL queries
- **Semantic processing** - Advanced text analysis and understanding
- **AI orchestration** - Manage AI model interactions and caching

## 📦 Key Features

- **SurrealDB-native vectors** - Generate embeddings in SurrealDB vector<float, DIM> format
- **Natural Language to SurrealQL** - Convert user queries to safe, validated SurrealQL
- **Graph relationship extraction** - AI-powered RELATE statement generation
- **Multiple embedding models** - Support for various embedding backends
- **LLM abstraction** - Unified interface for different language models
- **Async processing** - Background embedding generation for performance
- **Smart caching** - Optimize AI model usage and reduce latency
- **Batch processing** - Efficient handling of multiple requests

## 🔗 Dependencies

- **`nodespace-core-types`** - Shared data structures and error types
- **SurrealDB client** - For schema introspection and query validation
- **mistral.rs** - High-performance LLM inference with Metal acceleration on Apple Silicon
- **Candle** - Rust-native ML framework for embedding generation
- **GGUF models** - Quantized model support for efficient local inference

## 🚀 Getting Started

### **New to NodeSpace? Start Here:**
1. **Read [NodeSpace System Design](../nodespace-system-design/README.md)** - Understand the full architecture
2. **Check [Linear workspace](https://linear.app/nodespace)** - Find your current tasks (filter by `nodespace-nlp-engine`)
3. **Review [Development Workflow](../nodespace-system-design/docs/development/workflow.md)** - Process and procedures
4. **Study [NLP Engine Interface](src/lib.rs)** - Interface definitions owned by this repository
5. **See [MVP User Flow](../nodespace-system-design/examples/mvp-user-flow.md)** - What you're building

### **Development Setup:**
```bash
# Add to your Cargo.toml
[dependencies]
nodespace-nlp-engine = { git = "https://github.com/malibio/nodespace-nlp-engine" }

# Use in your code
use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use nodespace_core_types::NodeSpaceResult;

let engine = LocalNLPEngine::new().await?;
let embedding = engine.generate_embedding(request).await?;
```

## 🏗️ Architecture Context

Part of the [NodeSpace system architecture](../nodespace-system-design/README.md):

1. `nodespace-core-types` - Shared data structures and interfaces
2. `nodespace-data-store` - SurrealDB-based entity storage with graph relationships
3. **`nodespace-nlp-engine`** ← **You are here** (SurrealDB + AI integration)
4. `nodespace-workflow-engine` - Automation and event processing
5. `nodespace-core-logic` - Business logic orchestration
6. `nodespace-core-ui` - React components and UI
7. `nodespace-desktop-app` - Tauri application shell

## 🔄 MVP Implementation

The initial implementation focuses on SurrealDB-integrated AI workflow:

1. **Generate embeddings** - Convert text to SurrealDB vector<float, DIM> format
2. **Natural Language to SurrealQL** - Convert user queries to validated SurrealQL
3. **Graph relationship extraction** - AI-powered RELATE statement generation
4. **LLM queries** - Process RAG requests with SurrealDB context
5. **Async processing** - Background embedding generation and query validation
6. **Error handling** - Robust AI service and SurrealDB integration error management

## ✅ Implementation Status

**Real AI/ML implementation with Apple Silicon Metal acceleration!**

The engine now includes production-ready AI capabilities:

- ✅ **Real embeddings** - sentence-transformers/all-MiniLM-L6-v2 via Candle
- ✅ **Real text generation** - mistralai/Magistral-Small-2506 (23B) GGUF via mistral.rs
- ✅ **Metal acceleration** - Optimized for Apple Silicon (36GB model on Metal)
- ✅ **Fast inference** - ~13ms response time after model warm-up
- ✅ **Trait compliance** - Full `NLPEngine` trait implementation
- ✅ **GGUF support** - Q8_0 quantization for performance/quality balance

### Model Configuration

Current models:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Text Generation**: `mistralai/Magistral-Small-2506_gguf` (23.6B parameters, Q8_0 quantization)
- **Cache Location**: `~/.cache/huggingface/hub/` (25.1GB download, one-time)

### Features by Implementation

**With `real-ml` feature (default):**
- Real AI model inference
- Metal acceleration on Apple Silicon
- High-quality embeddings and text generation
- Large initial download (25.1GB for text model)

**Without `real-ml` feature:**
- Stub implementations for development
- Fast compilation and testing
- Deterministic responses for consistent testing

## 🧪 Testing

```bash
# Run all tests with stub implementations (fast)
cargo test --no-default-features

# Run all tests with real ML models (requires model download)
cargo test --features real-ml

# Test embedding generation
cargo run --example generate_embeddings

# Test text generation with real AI (requires 25.1GB model download)
cargo run --example text_generation --features real-ml

# Test without downloading models (uses stubs)
cargo run --example text_generation --no-default-features
```

### Performance Expectations

**First run with real-ml:**
- Model download: ~4 minutes (25.1GB at 100MB/s)
- Model loading: ~30 seconds
- First inference: ~24 seconds (includes warm-up)

**Subsequent runs:**
- Model loading: ~15 seconds (from cache)
- Inference: ~13-14ms per request

---

**Project Management:** All development tasks tracked in [Linear workspace](https://linear.app/nodespace)