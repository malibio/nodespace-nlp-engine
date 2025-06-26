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

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Test
```bash
# Build the project
cargo build

# Run tests (works with stub implementation)
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
```

### Development Workflow
This is currently a STUB IMPLEMENTATION for contract compliance. The heavy ML dependencies (Candle, Mistral.rs) are commented out in `Cargo.toml` to enable compilation without requiring large model downloads.

To enable real ML functionality, uncomment the ML dependencies in `Cargo.toml`:
```toml
# Uncomment these for real ML implementation
candle-core = "0.6"
candle-nn = "0.6" 
candle-transformers = "0.6"
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs", features = ["cuda", "metal"] }
tokenizers = "0.19"
hf-hub = "0.3"
```

## Architecture Overview

### Core Components

**`LocalNLPEngine`** (`src/engine.rs`): Main implementation of the `NLPEngine` trait. Orchestrates embedding generation, text generation, and SurrealQL generation with lazy initialization and caching.

**Modular Design**:
- **`EmbeddingGenerator`** (`src/embedding.rs`): Handles text-to-vector conversion with caching
- **`TextGenerator`** (`src/text_generation.rs`): Enhanced LLM text generation with RAG context support
- **`SurrealQLGenerator`** (`src/surrealql.rs`): Natural language to SurrealQL conversion
- **Token Utilities** (`src/utils.rs`): RAG token budget management and content allocation
- **Configuration** (`src/models.rs`): Device detection, model configs, performance tuning

### Key Architectural Patterns

**Lazy Initialization**: Components are initialized on first use via `ensure_initialized()` to avoid startup delays.

**Async-First Design**: All operations are async with `Arc<RwLock<>>` for safe concurrent access.

**Caching Strategy**: 
- Embedding cache using `DashMap` for thread-safe concurrent access
- Cache statistics and management through engine status

**Error Handling**: Custom `NLPError` types with conversion to `NodeSpaceError` at boundaries.

**Contract Compliance**: Implements the `NLPEngine` trait from `nodespace-core-types` with complete method coverage.

### Device and Model Management

The engine supports automatic device detection (CPU/CUDA/Metal) and lazy model loading. Configuration is handled through `NLPConfig` with sensible defaults:
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Text generation: mistralai/Magistral-Small-2506 (128k context)

### SurrealDB Integration

The engine is designed for SurrealDB-native operations:
- Embeddings generated in SurrealDB `vector<float, DIM>` format
- Natural language to SurrealQL conversion with safety checks
- Schema-aware query generation

## RAG Integration Architecture

The NLP engine provides comprehensive RAG (Retrieval-Augmented Generation) support for `nodespace-core-logic`:

### Enhanced Text Generation
- **`generate_text_enhanced`**: RAG-aware text generation with context assembly
- **Token Budget Management**: Automatic token allocation for context + response
- **Conversation Mode**: Optimized parameters for conversational AI
- **Context Utilization Analysis**: Tracks how well responses use provided context

### Token Management Utilities
```rust
use nodespace_nlp_engine::{TokenBudget, estimate_token_count, allocate_budget_to_segments};

// Smart content allocation within token limits
let budget = TokenBudget::new(8192); // Model context window
let allocated = allocate_budget_to_segments(knowledge_segments, budget.available_for_context())?;
```

### RAG Request/Response Pattern
```rust
let request = TextGenerationRequest {
    prompt: user_query,
    rag_context: Some(RAGContext {
        knowledge_sources: retrieved_content,
        context_summary: "What knowledge includes",
        retrieval_confidence: 0.88,
    }),
    conversation_mode: true,
    // ... other params
};

let response = nlp_engine.generate_text_enhanced(request).await?;
// Returns enhanced response with metrics and context analysis
```

**Example**: See `examples/rag_integration.rs` for complete RAG workflow demonstration.

## Testing Strategy

Comprehensive test suite in `tests/contract_compliance.rs` covers:
- Contract compliance for all `NLPEngine` trait methods
- Embedding consistency and determinism
- Batch processing performance comparison
- Error handling and initialization edge cases
- Caching functionality and statistics
- SurrealQL safety features (injection prevention)

Tests work with stub implementations and provide a foundation for real ML model testing.

## 🎯 FINDING YOUR NEXT TASK

**See [development-workflow.md](../nodespace-system-design/docs/development-workflow.md)** for task management workflow.

## Integration Context

This is part of the NodeSpace distributed system architecture:
1. `nodespace-core-types` - Shared interfaces and data structures
2. `nodespace-data-store` - SurrealDB entity storage
3. **`nodespace-nlp-engine`** - AI/ML processing layer (this repository)
4. `nodespace-workflow-engine` - Automation and events
5. `nodespace-core-logic` - Business logic orchestration
6. `nodespace-core-ui` - React components
7. `nodespace-desktop-app` - Tauri application

The engine integrates with the data store for semantic search and provides AI capabilities to the workflow and logic layers.