# NodeSpace NLP Engine

**Multimodal AI/ML processing and LanceDB integration for NodeSpace**

This repository implements the complete multimodal AI/ML layer for NodeSpace, providing text and image embedding generation, LLM integration, PDF processing, and semantic search capabilities. It serves as the **intelligence layer** of the distributed system with native vector database integration.

## 🎯 Purpose

- **Multimodal embeddings** - Convert text and images to vector format for semantic search
- **Image Q&A** - Visual question answering using advanced multimodal LLMs
- **PDF processing** - Extract and understand text and images from PDF documents
- **Semantic search** - Find relevant content across text and visual modalities
- **RAG capabilities** - Context-aware generation using retrieved multimodal content
- **Local-first AI** - All processing on-device using Apple MPS acceleration

## 📦 Key Features

### Multimodal Capabilities
- **Dual embedding models** - BGE-small for text, CLIP for images (512 dimensions)
- **Visual Question Answering** - Answer questions about images using Phi-4 multimodal
- **Cross-modal search** - Find images using text queries and vice versa
- **Metadata integration** - Utilize EXIF data (GPS, timestamps, camera info)
- **PDF multimodal processing** - Extract and understand both text and images from PDFs

### Performance & Infrastructure
- **Apple MPS acceleration** - Optimized for Apple Silicon using ONNX Runtime
- **LanceDB vector storage** - Embedded, efficient columnar database for vector operations
- **Smart caching** - Optimize AI model usage and reduce latency
- **Batch processing** - Efficient handling of multiple requests
- **Async processing** - Background processing for performance

### Developer Experience
- **ONNX Runtime stack** - Proven, cross-platform inference without Metal compilation issues
- **FastEmbed integration** - High-level Rust embedding library with model management
- **Comprehensive evaluation** - ROUGE/BLEU for text + VQA/retrieval metrics for multimodal
- **Feature-gated compilation** - Optional multimodal features for flexible deployment

## 🔗 Dependencies & Technology Stack

### Core Framework
- **Application**: Tauri 2.6.0 (desktop application with Rust backend)
- **Language**: Rust 1.88.0 (performance, safety, memory management)
- **Hardware**: Apple MPS via ONNX Runtime CoreML Execution Provider

### AI/ML Stack
- **LLM Inference**: ONNX Runtime (`ort = "2.0.0-rc.10"`)
- **Multimodal LLM**: `microsoft/Phi-4-multimodal-instruct-onnx`
- **Embedding Library**: `fastembed = "0.4"` (ONNX-based, high-performance)
- **Text Embeddings**: `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Image Embeddings**: `Qdrant/clip-ViT-B-32-vision` (512 dimensions)

### Data & Storage
- **Vector Database**: `lancedb = "0.20.0"` (Rust-native, embedded)
- **PDF Processing**: `pdfium-render = "0.8.33"` (text and image extraction)
- **Image Processing**: `image = "0.25"`, `exif = "0.6.1"` (metadata extraction)
- **Tokenization**: `tokenizers = "0.21.2"` (compatible with FastEmbed)

## 🚀 Getting Started

### **New to NodeSpace? Start Here:**
1. **Read [Multimodal Architecture Guide](docs/multimodal-architecture.md)** - Comprehensive technical architecture
2. **Check [Linear workspace](https://linear.app/nodespace)** - Find your current tasks (filter by `nodespace-nlp-engine`)
3. **Review [Development Workflow](../nodespace-system-design/docs/development/workflow.md)** - Process and procedures
4. **Study [NLP Engine Interface](src/lib.rs)** - Interface definitions owned by this repository
5. **See [Evaluation Strategy](docs/multimodal-evaluation-strategy.md)** - Quality assurance framework

### **Development Setup:**
```bash
# Add to your Cargo.toml
[dependencies]
nodespace-nlp-engine = { git = "https://github.com/malibio/nodespace-nlp-engine" }

# Basic usage
use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use nodespace_core_types::NodeSpaceResult;

let engine = LocalNLPEngine::new().await?;

// Text embedding
let text_embedding = engine.generate_embedding("Meeting notes about Q3 planning").await?;

// Multimodal capabilities (Phase 2)
let image_embedding = engine.generate_image_embedding(&image_bytes).await?;
let vqa_response = engine.answer_image_question(&image_bytes, "What do you see?").await?;
```

## 🏗️ Architecture Context

Part of the [NodeSpace system architecture](../nodespace-system-design/README.md):

1. `nodespace-core-types` - Shared data structures and interfaces
2. `nodespace-data-store` - SurrealDB-based entity storage with graph relationships  
3. **`nodespace-nlp-engine`** ← **You are here** (Multimodal AI + vector storage)
4. `nodespace-workflow-engine` - Automation and event processing
5. `nodespace-core-logic` - Business logic orchestration
6. `nodespace-core-ui` - React components and UI
7. `nodespace-desktop-app` - Tauri application shell

## 🔄 Implementation Roadmap

### Phase 1: Text Foundation (✅ Complete)
- ✅ **Text embeddings** - BGE-small via FastEmbed with ONNX Runtime
- ✅ **Text generation** - Gemma 3 1B IT ONNX model
- ✅ **Vector operations** - Basic embedding generation and caching
- ✅ **Evaluation framework** - ROUGE/BLEU metrics (NS-71)

### Phase 2: Multimodal Core (🚧 In Progress)
- 🚧 **Image embeddings** - CLIP via FastEmbed ImageEmbedding
- 🚧 **Multimodal LLM** - Phi-4 multimodal ONNX integration
- 🚧 **LanceDB migration** - Vector database for dual embedding storage
- 🚧 **Image Q&A** - Visual question answering capabilities

### Phase 3: Advanced Features (📋 Planned)
- 📋 **PDF processing** - Text and image extraction pipeline
- 📋 **Cross-modal RAG** - Semantic search across text and images
- 📋 **Metadata integration** - EXIF data utilization in responses
- 📋 **Performance optimization** - Apple MPS tuning and caching strategies

### Phase 4: Production Ready (🔮 Future)
- 🔮 **Audio capabilities** - Speech-to-text and audio Q&A
- 🔮 **Fine-tuning** - LoRA adaptation for personal data
- 🔮 **Advanced evaluation** - Comprehensive multimodal quality metrics
- 🔮 **Deployment optimization** - Model bundling and distribution

## ✅ Current Implementation Status

**Phase 1 Complete: Text-Only RAG with ONNX Runtime**

### Working Features
- ✅ **Real text embeddings** - BAAI/bge-small-en-v1.5 (384 dimensions) via FastEmbed
- ✅ **Real text generation** - Gemma 3 1B IT ONNX model with local inference
- ✅ **ONNX Runtime integration** - Proven Apple MPS acceleration without Metal compilation issues
- ✅ **Evaluation framework** - ROUGE/BLEU metrics for text quality assessment
- ✅ **Trait compliance** - Full `NLPEngine` interface implementation
- ✅ **Shared model storage** - `/Users/malibio/nodespace/models/` for cross-service access

### Model Configuration
Current models (Phase 1):
- **Text Embeddings**: `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Text Generation**: `local/gemma-3-1b-it-onnx` (1B parameters, ONNX format)
- **Cache Location**: FastEmbed manages model downloads automatically

Target models (Phase 2):
- **Text Embeddings**: `BAAI/bge-small-en-v1.5` (384 dimensions) - keep for text-only tasks
- **Image Embeddings**: `Qdrant/clip-ViT-B-32-vision` (512 dimensions) - add for vision
- **Multimodal LLM**: `microsoft/Phi-4-multimodal-instruct-onnx` - upgrade for vision Q&A

## 🧪 Testing & Evaluation

### Current Testing (Phase 1)
```bash
# Run text embedding tests
cargo run --example generate_embeddings

# Run text generation tests  
cargo run --example text_generation

# Run evaluation framework
cargo test --features evaluation

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Multimodal Testing (Phase 2)
```bash
# Test image embedding generation (planned)
cargo run --example generate_image_embeddings --features multimodal

# Test visual question answering (planned)
cargo run --example image_qa --features multimodal

# Test multimodal RAG pipeline (planned)
cargo run --example multimodal_rag --features multimodal

# Run comprehensive evaluation (planned)
cargo test --features "evaluation,multimodal"
```

### Performance Expectations

**Current Performance (Phase 1):**
- **Text embedding generation**: ~2-5ms per text
- **Text generation**: Stub responses (~5µs) or real ONNX inference
- **Memory usage**: Minimal with lazy loading and caching

**Target Performance (Phase 2):**
- **Image embedding generation**: ~10-50ms per image
- **Multimodal LLM inference**: ~100-500ms per response
- **Memory usage**: Optimized for Apple MPS with efficient model loading

### Evaluation Framework
- **Text Quality**: ROUGE-1, ROUGE-2, ROUGE-L, BLEU-1 through BLEU-4
- **Multimodal Quality**: VQA accuracy, cross-modal retrieval metrics, metadata integration
- **Performance**: Response time, memory usage, cache efficiency
- **See**: [Multimodal Evaluation Strategy](docs/multimodal-evaluation-strategy.md)

## 📋 Development Workflow

**Current Status**: Building multimodal foundations on proven ONNX Runtime stack

**Next Steps**:
1. **Review** [Multimodal Architecture Guide](docs/multimodal-architecture.md)
2. **Implement** Phase 2 features according to roadmap
3. **Test** with comprehensive evaluation framework
4. **Optimize** for Apple MPS performance

---

**Project Management:** All development tasks tracked in [Linear workspace](https://linear.app/nodespace)