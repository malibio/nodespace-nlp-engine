# Legacy Documentation

This directory contains documentation from the previous text-only implementation phases, preserved for reference during the transition to multimodal capabilities.

## Contents

- **`TOKENIZER_COMPATIBILITY.md`** - Original tokenizer compatibility research and findings
- **Historical architecture decisions** - Context for current ONNX Runtime adoption
- **Migration notes** - Lessons learned during the transition from Candle/Mistral.rs

## Historical Context

### Original Stack (Pre-ONNX Runtime)
- **Candle + Mistral.rs** for text generation
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings  
- **Metal compilation challenges** that led to ONNX Runtime adoption

### Key Learnings
1. **Metal Compilation Issues**: Direct Metal kernel compilation in Tauri environment proved problematic
2. **Tokenizer Compatibility**: Magistral-Small-2506's tekken.json v11 format was incompatible with Rust libraries
3. **ONNX Runtime Success**: Provided reliable Apple MPS acceleration without compilation issues

### Migration Path
The transition to ONNX Runtime with FastEmbed provided:
- **Proven Apple MPS acceleration** without Metal compilation complexity
- **Broad model compatibility** with standard tokenizer formats
- **High-level abstractions** through FastEmbed library
- **Foundation for multimodal** capabilities with unified inference runtime

## Current Implementation

See the main documentation for current multimodal architecture:
- **[README.md](../README.md)** - Current system overview
- **[docs/multimodal-architecture.md](../multimodal-architecture.md)** - Comprehensive architecture guide
- **[CLAUDE.md](../CLAUDE.md)** - Development guidance

This legacy documentation is maintained for historical context and to inform future architectural decisions.