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

- **`nodespace-core-types`** - Data structures and `NLPEngine` trait interface
- **SurrealDB client** - For schema introspection and query validation
- **Mistral.rs** - Local LLM execution with Magistral-Small-2506 (23.6B, 128k context)
- **Embedding libraries** - Various embedding model integrations

## 🚀 Getting Started

### **New to NodeSpace? Start Here:**
1. **Read [NodeSpace System Design](../nodespace-system-design/README.md)** - Understand the full architecture
2. **Check [Linear workspace](https://linear.app/nodespace)** - Find your current tasks (filter by `nodespace-nlp-engine`)
3. **Review [Development Workflow](../nodespace-system-design/docs/development-workflow.md)** - Process and procedures
4. **Study [Key Contracts](../nodespace-system-design/contracts/)** - Interface definitions you'll implement
5. **See [MVP User Flow](../nodespace-system-design/examples/mvp-user-flow.md)** - What you're building

### **Development Setup:**
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

## ⚠️ Current Implementation Status

**This is currently a STUB IMPLEMENTATION for contract compliance and compilation.**

The heavy ML dependencies (Candle, Mistral.rs) are commented out in `Cargo.toml` to enable compilation without requiring large model downloads. The current implementation provides:

- ✅ **Contract-compliant interface** - Implements all required `NLPEngine` trait methods
- ✅ **Deterministic stub responses** - Generates fake but consistent embeddings and text
- ✅ **Full compilation** - Builds successfully with minimal dependencies
- ✅ **Test structure** - Complete test suite for validation
- 🚧 **Real ML models** - TODO: Enable actual Mistral.rs and Candle integration

### Enabling Real ML Implementation

To use actual AI models, uncomment the dependencies in `Cargo.toml`:

```toml
# Uncomment these lines for real ML functionality
candle-core = "0.6"
candle-nn = "0.6" 
candle-transformers = "0.6"
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", features = ["cuda", "metal"] }
tokenizers = "0.19"
hf-hub = "0.3"
```

Then replace the stub implementations in `src/embedding.rs` and `src/text_generation.rs` with actual model loading and inference code.

## 🧪 Testing

```bash
# Run all tests (currently using stub implementations)
cargo test

# Test with sample content (stub responses)
cargo run --example generate_embeddings

# Benchmark performance (stub timing)
cargo run --example text_generation
```

---

**Project Management:** All development tasks tracked in [Linear workspace](https://linear.app/nodespace)