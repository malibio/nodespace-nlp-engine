# CLAUDE.md

🚨 **STOP - READ WORKFLOW FIRST** 🚨
Before doing ANYTHING else, you MUST read the development workflow:
1. Read: `../nodespace-system-design/docs/development/workflow.md`
2. Check Linear for current tasks
3. Then return here for implementation guidance

❌ **FORBIDDEN:** Any code analysis, planning, or implementation before reading the workflow

## Development Workflow
**ALWAYS start with README.md** - This file contains the authoritative development workflow and setup instructions for this repository.

**Then return here** for repository-specific guidance and architecture details.

## Project Overview

This file provides guidance to Claude Code (claude.ai/code) when working with the NodeSpace NLP Engine - a multimodal AI/ML processing system built on ONNX Runtime for Apple MPS acceleration.

## Development Commands

### Build and Test
```bash
# Build the project
cargo build

# Run tests (works with current text-only implementation)
cargo test

# Run specific test
cargo test test_nlp_engine_trait_compliance

# Run examples (demonstrate functionality)
cargo run --example generate_embeddings
cargo run --example text_generation

# Check code formatting
cargo fmt --check

# Run clippy lints
cargo clippy -- -D warnings

# Test evaluation framework
cargo test --features evaluation
```

### Multimodal Development (Phase 2)
```bash
# Test image embedding generation (planned)
cargo run --example generate_image_embeddings --features multimodal

# Test visual question answering (planned)  
cargo run --example image_qa --features multimodal

# Run comprehensive multimodal evaluation (planned)
cargo test --features "evaluation,multimodal"
```

### Development Status
This is a **REAL AI/ML IMPLEMENTATION** with proven ONNX Runtime stack. Phase 1 (text-only) is complete and functional. Phase 2 (multimodal) is in planning/implementation.

**Current Stack**:
- **FastEmbed + ONNX Runtime** for embeddings and inference
- **BGE-small-en-v1.5** for text embeddings (384 dimensions)
- **Gemma 3 1B IT ONNX** for text generation
- **Apple MPS acceleration** via ONNX Runtime CoreML EP

**Target Stack (Phase 2)**:
- **Dual embedding strategy**: BGE-small (text) + CLIP (images)
- **Phi-4 multimodal ONNX** for visual question answering
- **LanceDB** for vector storage and retrieval
- **PDF processing** with pdfium-render

## Architecture Overview

### Core Components

**`LocalNLPEngine`** (`src/engine.rs`): Main implementation of the `NLPEngine` trait. Orchestrates multimodal embedding generation, text generation, image processing, and semantic search with lazy initialization and caching.

**Modular Design**:
- **`EmbeddingGenerator`** (`src/embedding.rs`): Handles text-to-vector conversion using FastEmbed + ONNX Runtime
- **`ImageEmbeddingGenerator`** (planned): CLIP-based image-to-vector conversion
- **`TextGenerator`** (`src/text_generation.rs`): LLM text generation using ONNX Runtime
- **`MultimodalGenerator`** (planned): Phi-4 multimodal for visual Q&A
- **`Configuration`** (`src/models.rs`): Device detection, model configs, performance tuning

### Key Architectural Patterns

**ONNX Runtime Foundation**: All AI/ML operations use ONNX Runtime for consistent Apple MPS acceleration without Metal compilation issues.

**Dual Embedding Strategy**: 
- Text embeddings (BGE-small, 384 dims) for text-only semantic search
- Image embeddings (CLIP, 512 dims) for visual and cross-modal search
- Unified caching and retrieval interface

**Lazy Initialization**: Components are initialized on first use via `ensure_initialized()` to avoid startup delays.

**Async-First Design**: All operations are async with `Arc<RwLock<>>` for safe concurrent access.

**Multimodal Caching Strategy**: 
- Text embedding cache using `DashMap` for thread-safe concurrent access
- Image embedding cache with metadata integration
- Cross-modal search result caching

**Error Handling**: Custom `NLPError` types with conversion to `NodeSpaceError` at boundaries.

**Contract Compliance**: Implements the `NLPEngine` trait from `nodespace-core-types` with complete method coverage.

### Device and Model Management

The engine supports automatic device detection (CPU/CUDA/Metal) and lazy model loading. Configuration is handled through `NLPConfig` with sensible defaults:

**Current (Phase 1)**:
- Text Embedding: BAAI/bge-small-en-v1.5 (384 dimensions)
- Text generation: local/gemma-3-1b-it-onnx (1B parameters, ONNX format)

**Target (Phase 2)**:
- Text Embedding: BAAI/bge-small-en-v1.5 (384 dimensions) - keep for text tasks
- Image Embedding: Qdrant/clip-ViT-B-32-vision (512 dimensions) - add for vision
- Multimodal LLM: microsoft/Phi-4-multimodal-instruct-onnx (text + image Q&A)

### Vector Database Integration

**Current**: SurrealDB integration for basic vector operations
**Target**: LanceDB for optimized vector storage and retrieval

The engine is designed for efficient vector operations:
- Embeddings stored in optimized columnar format
- Semantic search across text and image modalities
- Metadata integration (EXIF data, document structure)
- Cross-modal retrieval capabilities

## Testing Strategy

### Current Test Coverage (`tests/contract_compliance.rs`)
- Contract compliance for all `NLPEngine` trait methods
- Embedding consistency and determinism  
- Batch processing performance comparison
- Error handling and initialization edge cases
- Caching functionality and statistics
- Evaluation framework (ROUGE/BLEU) validation

### Multimodal Test Strategy (Planned)
- Visual Question Answering (VQA) accuracy testing
- Cross-modal retrieval precision/recall metrics
- Image metadata extraction and utilization
- PDF processing (text + image extraction)
- Performance benchmarking on Apple MPS
- End-to-end multimodal RAG pipeline testing

See [Multimodal Evaluation Strategy](docs/multimodal-evaluation-strategy.md) for comprehensive testing framework.

## 🎯 FINDING YOUR NEXT TASK

**Current Phase**: Implementing multimodal capabilities (Phase 2)

**Priority Tasks**:
1. **Review** [Multimodal Architecture Guide](docs/multimodal-architecture.md)
2. **Implement** image embedding support via FastEmbed ImageEmbedding
3. **Add** Phi-4 multimodal ONNX integration
4. **Migrate** to LanceDB for vector storage
5. **Test** with comprehensive evaluation framework

**See [development-workflow.md](../nodespace-system-design/docs/development-workflow.md)** for task management workflow.

## Implementation Phases

### ✅ Phase 1: Text Foundation (Complete)
- FastEmbed + ONNX Runtime integration
- BGE-small text embeddings
- Gemma 3 1B text generation  
- ROUGE/BLEU evaluation framework
- Apple MPS acceleration proven

### 🚧 Phase 2: Multimodal Core (In Progress)
- FastEmbed ImageEmbedding for CLIP-based image embeddings
- Phi-4 multimodal ONNX integration for visual Q&A
- LanceDB migration for dual embedding storage
- Image preprocessing and metadata extraction
- Cross-modal semantic search

### 📋 Phase 3: Advanced Features (Planned)
- PDF processing pipeline (pdfium-render)
- EXIF metadata integration and utilization
- Performance optimization for Apple MPS
- Advanced multimodal evaluation metrics
- Production deployment optimization

### 🔮 Phase 4: Production Ready (Future)
- Audio capabilities (speech-to-text, audio Q&A)
- LoRA fine-tuning for personal data adaptation
- Model quantization and optimization
- Advanced caching and performance tuning

## Integration Context

This is part of the NodeSpace distributed system architecture:
1. `nodespace-core-types` - Shared interfaces and data structures
2. `nodespace-data-store` - SurrealDB entity storage + future LanceDB integration
3. **`nodespace-nlp-engine`** - Multimodal AI/ML processing layer (this repository)
4. `nodespace-workflow-engine` - Automation and events
5. `nodespace-core-logic` - Business logic orchestration
6. `nodespace-core-ui` - React components
7. `nodespace-desktop-app` - Tauri application

The engine provides multimodal AI capabilities to the workflow and logic layers, with vector storage integration for semantic search across text and visual content.

## Key Technical References

- **[Multimodal Architecture Guide](docs/multimodal-architecture.md)** - Comprehensive implementation strategy
- **[Multimodal Evaluation Strategy](docs/multimodal-evaluation-strategy.md)** - Quality assurance framework
- **FastEmbed Documentation** - For embedding model integration
- **ONNX Runtime Documentation** - For model inference and Apple MPS optimization
- **LanceDB Documentation** - For vector database operations