# Architectural Guide: Multimodal RAG Application in Rust on Apple MPS (Updated)

This document provides an updated architectural guide, reflecting the latest stable/release candidate versions of models, databases, and libraries as of June 27, 2025. The core strategy remains robust, leveraging Rust for performance, ONNX Runtime for MPS acceleration, and LanceDB for embedded vector storage.

## 1. Project Goal

The primary goal is to create a robust, performant, and locally runnable Q&A system capable of answering questions based on personal text documents (including PDFs) and images, leveraging advanced large language models (LLMs) and vector embeddings. Future expansion to include audio is also considered.

## 2. Core Requirements

- **Application Framework**: Tauri (for cross-platform desktop application with Rust backend and web frontend)
- **Core Language**: Rust (for performance, safety, and control)
- **Hardware Acceleration**: Apple MPS (Metal Performance Shaders) for efficient on-device inference on Apple Silicon Macs
- **Multimodal Capabilities**: Support for text and image input (with future audio potential)

## 3. R&D Journey Summary: Overcoming Key Challenges

Our R&D journey encountered and successfully navigated several critical roadblocks:

### Initial Tokenizer Incompatibility
- **Problem**: Mistral's Magistral-Small-2506 uses a proprietary tekken.json v11 tokenizer format, which kitoken (the only Rust library supporting tekken.json) does not yet support. Other Rust tokenizer libraries only handle standard tokenizer.json.
- **Solution**: Pivoted away from Magistral for now.

### Metal Compilation Issues with Native Rust ML Libraries
- **Problem**: Direct compilation of custom Metal kernels or specific C++/Metal linking in mistral.rs and mlx-rs failed consistently within the cargo tauri dev build environment on Apple MPS. This was identified as a build system (CMake) configuration issue not correctly passing Metal language standard flags.
- **Solution**: Explored and validated ONNX Runtime as an alternative.

### ONNX Runtime Breakthrough
- **Finding**: ONNX Runtime compiled successfully and ran without any Metal-related compilation errors in the Tauri Dev environment. This confirmed that using a pre-compiled, robust runtime for Metal operations effectively bypasses the native Rust ML framework's build complexities.
- **Impact**: This became the cornerstone of our inference strategy.

## 4. Chosen Technology Stack & Rationale

Based on the R&D findings, the following technology stack has been selected:

### Application Framework: Tauri
- **Rationale**: Provides a native desktop application experience with a Rust backend and web frontend, enabling cross-platform deployment.
- **Current Version**: `tauri = "2.6.0"` (Latest release as of June 26, 2025)

### Core Language: Rust
- **Rationale**: Meets the performance, memory safety, and control requirements for on-device AI inference.
- **Current Version**: Rust 1.88.0 (Latest stable release as of June 26, 2025)

### Hardware Acceleration: Apple MPS
- **Rationale**: Leverages Apple Silicon's dedicated ML hardware for highly efficient inference, proven to compile cleanly with ONNX Runtime.
- **Implementation**: Via ONNX Runtime's CoreML Execution Provider

### LLM Inference Engine: ONNX Runtime (ort crate)
- **Rationale**: Solves the Metal compilation issues by providing a pre-compiled, cross-platform runtime. Offers excellent performance and broad model compatibility.
- **Current Version**: `ort = "=2.0.0-rc.10"` (Latest Release Candidate, considered production-ready by maintainers)

### Multimodal LLM: microsoft/Phi-4-multimodal-instruct-onnx
- **Rationale**: Directly available in ONNX format (no complex conversion needed). Supports text, image, and audio inputs within a single, unified model. Designed for efficiency and low-latency inference on various hardware, including Apple MPS. Still the most straightforward ONNX-ready multimodal model from Microsoft.

### Embedding Library: fastembed-rs
- **Rationale**: A high-level Rust library built on ONNX Runtime (ort) and huggingface/tokenizers. Simplifies text and image embedding generation, handles model downloading, and provides optimized performance with parallelism.
- **Current Version**: `fastembed = "0.4"` (Latest major version, check crates.io for minor updates like 0.4.x)

### Text Embedding Model: BAAI/bge-small-en-v1.5
- **Rationale**: Chosen for its state-of-the-art performance in text retrieval for RAG applications. It's efficient and natively supported by fastembed-rs. Still the recommended choice for text embeddings.

### Image Embedding Model: Qdrant/clip-ViT-B-32-vision
- **Rationale**: A dedicated vision encoder (part of CLIP) for generating general-purpose image embeddings. Natively supported by fastembed-rs and runs on ONNX. Used for semantic image search. Still the recommended choice for image embeddings.

### Vector Database: LanceDB (lancedb crate)
- **Rationale**: A Rust-native, embedded, and efficient columnar database optimized for vector embeddings. Ideal for local RAG applications and managing vector data alongside metadata.
- **Current Version**: `lancedb = "0.20.0"` (Latest release as of June 4, 2025)

### Image Metadata Extraction: exif and chrono crates
- **Rationale**: For extracting valuable metadata (timestamp, GPS, camera info) from image files.
- **Current Versions**: `exif = "0.6.1"` (Latest as of 8 months ago, still standard), `chrono = "0.4.41"` (Latest as of June 26, 2025)

### PDF Processing: pdfium_render and image crates
- **Rationale**: pdfium_render (bindings to Google's Pdfium) for robust text and image extraction from PDFs. image for rendering PDF pages to bitmaps if OCR is needed.
- **Current Versions**: `pdfium-render = "0.8.33"` (Latest as of June 14, 2025), `image = "0.25"` (Latest)

### OCR (Conditional): tesseract-rs bindings for Tesseract
- **Rationale**: Only needed if the application must extract text from scanned or image-based PDFs where direct text extraction is not possible.

### Tokenizer (General): tokenizers crate
- **Rationale**: Used by fastembed-rs and directly for LLM input preparation.
- **Current Version**: `tokenizers = "0.21.2"` (Latest as of June 26, 2025)

## 5. Feature Implementation Approach: Multimodal RAG Q&A

The core feature will allow users to ask questions about their personal text documents (including PDFs) and images.

### 5.1. Data Ingestion & Indexing Phase (Offline Processing)

This process prepares your knowledge base for fast retrieval.

#### ImageNode Structure
```rust
// In your `src/models.rs` or similar
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageNode {
    pub id: String, // Unique ID for LanceDB record
    pub raw_data: Vec<u8>, // Raw image bytes (for display/LLM input)
    pub embedding: Vec<f32>, // Vector embedding for semantic search
    pub filename: String,
    pub filepath: String, // Original path
    pub description: Option<String>, // AI-generated or manual description
    pub timestamp: Option<DateTime<Utc>>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub keywords: Vec<String>, // Tags, AI-generated
}
```

#### Image Processing Pipeline
1. **Load Raw Image**: Read image file into `Vec<u8>` using `std::fs::read` or `image::io::Reader`
2. **Extract Metadata**: Use `exif` crate to parse EXIF data (timestamp, GPS, camera info) from the raw image bytes
3. **Generate Image Embedding**: Use `fastembed::ImageEmbedding` (with `Qdrant/clip-ViT-B-32-vision` or `NomicEmbedVisionV1_5`) to create a `Vec<f32>` embedding from the image
4. **Create ImageNode**: Populate the ImageNode struct with raw data, embedding, and extracted metadata
5. **Save to LanceDB**: Insert the ImageNode into a LanceDB table (e.g., `image_nodes`). LanceDB will automatically use the embedding field for vector indexing

#### Text Document Processing Pipeline (including PDFs)
1. **PDF/Document Parsing**:
   - Use `pdfium_render` to extract text content page by page from PDFs
   - For scanned/image-based PDFs, render pages to images using `pdfium_render` and then apply an OCR solution (e.g., `tesseract-rs`)
   - For other text formats (.txt, Markdown), simply read the file content
2. **Text Chunking**: Split the extracted text into semantically coherent chunks (e.g., 256-512 tokens with 10-20% overlap) using a custom Rust implementation (e.g., based on character/token count, or sentence boundaries)
3. **Generate Text Embedding**: Use `fastembed::TextEmbedding` (with `BAAI/bge-small-en-v1.5`) to create a `Vec<f32>` embedding for each text chunk
4. **Store to LanceDB**: Insert each text chunk (original text, embedding, and any relevant metadata like source document ID, page number) into a separate LanceDB table (e.g., `text_chunks`)

#### Knowledge Graph Integration (Optional Enhancement)
- After initial text/image indexing, use your LLM (Phi-4 Multimodal) to perform entity and relationship extraction from the text chunks and image descriptions
- Populate a graph database (or use LanceDB's graph capabilities if it evolves to support them) with these structured facts
- This graph can then be used in the retrieval phase for multi-hop reasoning

### 5.2. Querying & RAG Phase (Local, Interactive Processing)

This process handles user questions and generates answers, all running on the local device.

#### User Query Embedding
When a user asks a question (text), use `fastembed::TextEmbedding` (with `BAAI/bge-small-en-v1.5`) to generate an embedding for the query.

#### Semantic Search & Retrieval
1. **Text Retrieval**: Query the `text_chunks` table in LanceDB using the user's query embedding to retrieve the top K most semantically similar text chunks
2. **Image Retrieval**: (If relevant to the query) Query the `image_nodes` table in LanceDB using the user's query embedding to retrieve the top M most semantically similar ImageNodes
3. **Context Aggregation**: Combine the raw text from the retrieved text chunks and the raw image data (and relevant metadata/descriptions) from the retrieved ImageNodes
4. **Knowledge Graph Expansion** (if implemented): If a knowledge graph is used, extract entities from the query and retrieved chunks/images. Traverse the graph to find additional, fact-based context

#### Multimodal Prompt Construction for Phi-4
Construct a single prompt for `microsoft/Phi-4-multimodal-instruct-onnx`. This prompt will include:
- A clear instruction (e.g., "Answer the following question based on the provided context and image(s).")
- The retrieved text chunks
- Crucially, the preprocessed raw image data for the top relevant images, inserted using Phi-4's specific image placeholder (`<|image_1|>`, etc.)
- The user's original question

#### Image Preprocessing for LLM
For each retrieved image, use the `preprocess_image_for_phi4` function to convert raw image bytes into the `Vec<f32>` tensor format expected by Phi-4.

#### Phi-4 Inference via ONNX Runtime
1. Load `microsoft/Phi-4-multimodal-instruct-onnx` using `ort::Session`
2. Feed the combined `input_ids` (text tokens with image placeholders) and `pixel_values` (image tensors) to the ONNX session
3. Implement the token generation loop (sampling one token at a time, managing KV cache, and feeding output back as input)
4. **Response Generation**: Receive the generated text response from Phi-4

## 6. Key Challenges & Considerations for Architect

1. **Multimodal ONNX Input**: The exact input names, shapes, and interleaving strategy for `microsoft/Phi-4-multimodal-instruct-onnx` in Rust via `ort` will require careful attention and likely translation from Python onnxruntime-genai examples. This is the most critical technical detail.

2. **PDF Parsing Robustness**: Handling diverse PDF structures (text, images, tables, complex layouts) can be challenging. `pdfium_render` is a strong start, but be prepared for edge cases and the need for OCR.

3. **LanceDB Schema & Data Types**: Ensure your ImageNode and text chunk structs correctly map to LanceDB's Arrow schema for efficient storage and retrieval.

4. **Performance Tuning**: Optimize ONNX Runtime Execution Providers (CoreML for MPS), batching, and potentially model quantization for production performance.

5. **LoRA Fine-tuning Workflow**: Remember that LoRA training will likely happen in Python, and you'll need a strategy to load the fine-tuned model (either merged ONNX or separate LoRA adapters if `ort` supports them directly) in Rust.

6. **Model & Data Bundling (Tauri)**: For desktop deployment, you'll need a strategy to bundle the ONNX model files, tokenizer files, and potentially LanceDB data with your Tauri application.

7. **Error Handling & User Feedback**: Robust error handling and clear user feedback are essential for a smooth application experience, especially with complex ML pipelines.

## 7. Sample Test Questions (Personal Photos Scenario)

These questions aim to test various aspects of the multimodal RAG system:

### Image Description (VQA)
- **Image**: A photo of a specific landmark (e.g., Eiffel Tower)
- **Question**: "Describe what you see in this image."
- **Expected AI**: A detailed visual description of the landmark and its surroundings

### Metadata Retrieval + VQA
- **Image**: A photo taken on a specific date with GPS data
- **Question**: "When and where was this photo taken, and what is the main subject?"
- **Expected AI**: "This photo was taken on [Date] at [Location derived from GPS]. The main subject appears to be [description of main subject from VQA]."

### Semantic Image Search + VQA
- **Query**: "Show me photos related to nature and greenery."
- **System Action**: Retrieves image(s) of parks/gardens via fastembed-rs semantic search
- **Image (retrieved)**: A photo of a botanical garden
- **Question (user follow-up)**: "What kind of plants are visible here?"
- **Expected AI**: "Based on the image, I can see [list types of plants/flowers/trees visible]."

### PDF Text RAG
- **Document**: A PDF containing a travel itinerary
- **Question**: "What is the planned activity for the afternoon of day 3 in the itinerary?"
- **Expected AI**: "According to the itinerary, the planned activity for the afternoon of day 3 is [activity from PDF text]."

### PDF Image/Chart RAG (if OCR/Chart understanding is implemented)
- **Document**: A PDF with a simple bar chart showing monthly expenses
- **Question**: "Looking at the chart on page X, what was the total expense in March?"
- **Expected AI**: "Based on the chart, the total expense in March was [amount from chart]."

### Combined Multimodal Reasoning
- **Image**: A photo of a person holding a specific product
- **Text Context (from RAG on documents)**: A retrieved text chunk describing the product's features
- **Question**: "Based on the image and the product description, what are the two main features of this product?"
- **Expected AI**: "The product in the image appears to be [product name]. Its two main features, as described, are [feature 1] and [feature 2]."

This guide provides a solid foundation for implementing this exciting multimodal RAG application.