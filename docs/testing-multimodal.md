# Testing Guide: Multimodal RAG System

This document provides comprehensive testing strategies and implementation guidelines for validating the multimodal RAG system across all phases of development.

## Testing Philosophy

The multimodal RAG system requires testing across multiple dimensions:
- **Functional correctness**: Does each component work as designed?
- **Performance benchmarks**: Does the system meet performance requirements?
- **Quality metrics**: Does the AI output meet quality standards?
- **Integration testing**: Do components work together correctly?
- **End-to-end validation**: Does the complete system solve user problems?

## Testing Architecture

### Test Pyramid Structure

```
    ┌─────────────────────────────────────┐
    │         E2E Tests (Few)             │  ← Complete user workflows
    │    Multimodal RAG Integration       │
    └─────────────────────────────────────┘
           ┌─────────────────────────┐
           │   Integration Tests     │      ← Component interactions
           │  (Cross-modal, VQA)     │
           └─────────────────────────┘
                  ┌─────────────┐
                  │ Unit Tests  │           ← Individual functions
                  │  (Many)     │
                  └─────────────┘
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component interaction testing
3. **Performance Tests**: Benchmarking and optimization
4. **Quality Tests**: AI output evaluation
5. **End-to-End Tests**: Complete user journey validation

## Phase 1: Text Foundation Testing (✅ Complete)

### Current Test Coverage

#### Unit Tests (`tests/contract_compliance.rs`)
```rust
#[tokio::test]
async fn test_nlp_engine_trait_compliance() {
    let engine = LocalNLPEngine::new();
    assert!(engine.initialize().await.is_ok());
    
    // Test all NLPEngine trait methods
    let embedding = engine.generate_embedding("test text").await?;
    assert_eq!(embedding.len(), 384); // BGE-small dimensions
}

#[tokio::test]
async fn test_embedding_consistency() {
    let engine = LocalNLPEngine::new();
    let text = "consistent test input";
    
    let embedding1 = engine.generate_embedding(text).await?;
    let embedding2 = engine.generate_embedding(text).await?;
    
    // Should be identical for same input
    assert_eq!(embedding1, embedding2);
}

#[tokio::test]
async fn test_caching_functionality() {
    let engine = LocalNLPEngine::new();
    let text = "cached test input";
    
    // First call - cache miss
    let start1 = Instant::now();
    let _embedding1 = engine.generate_embedding(text).await?;
    let duration1 = start1.elapsed();
    
    // Second call - cache hit
    let start2 = Instant::now();
    let _embedding2 = engine.generate_embedding(text).await?;
    let duration2 = start2.elapsed();
    
    // Cache hit should be significantly faster
    assert!(duration2 < duration1 / 2);
}
```

#### Evaluation Tests (`cargo test --features evaluation`)
```rust
#[tokio::test]
async fn test_rouge_evaluation() {
    let framework = EvaluationFramework::new();
    let generated = "The meeting covered Q3 planning and budget allocation.";
    let reference = "Q3 planning and budget allocation were discussed in the meeting.";
    
    let result = framework.evaluate_rag(generated, reference)?;
    assert!(result.rouge.rouge_1.f_score > 0.5);
    assert!(result.overall_quality > 0.6);
}

#[tokio::test]
async fn test_bleu_evaluation() {
    let framework = EvaluationFramework::new();
    let generated = "Paris is the capital of France.";
    let reference = "The capital of France is Paris.";
    
    let result = framework.evaluate_rag(generated, reference)?;
    assert!(result.bleu.bleu_1 > 0.7);
}
```

## Phase 2: Multimodal Testing (🚧 In Progress)

### Image Embedding Tests

#### Unit Tests
```rust
#[tokio::test]
async fn test_image_embedding_generation() {
    let generator = ImageEmbeddingGenerator::new(ImageEmbeddingConfig::default())?;
    generator.initialize().await?;
    
    let image_bytes = load_test_image("test_images/sample.jpg");
    let embedding = generator.generate_embedding(&image_bytes).await?;
    
    assert_eq!(embedding.len(), 512); // CLIP dimensions
    assert!(embedding.iter().any(|&x| x != 0.0)); // Non-zero values
}

#[tokio::test]
async fn test_image_metadata_extraction() {
    let image_bytes = load_test_image("test_images/with_exif.jpg");
    let metadata = extract_image_metadata(&image_bytes)?;
    
    assert!(metadata.timestamp.is_some());
    assert!(metadata.gps_latitude.is_some());
    assert!(metadata.camera_make.is_some());
}

#[tokio::test]
async fn test_image_preprocessing() {
    let image_bytes = load_test_image("test_images/large_image.jpg");
    let processed = preprocess_image_for_phi4(&image_bytes, (512, 512))?;
    
    assert_eq!(processed.len(), 3 * 512 * 512); // CHW format
    assert!(processed.iter().all(|&x| x >= 0.0 && x <= 1.0)); // Normalized
}
```

#### Cross-Modal Search Tests
```rust
#[tokio::test]
async fn test_text_to_image_search() {
    let engine = LocalNLPEngine::new();
    
    // Index test images
    let beach_image = load_test_image("test_images/beach.jpg");
    let mountain_image = load_test_image("test_images/mountain.jpg");
    
    engine.index_image("beach_vacation", &beach_image).await?;
    engine.index_image("mountain_hike", &mountain_image).await?;
    
    // Search with text query
    let results = engine.search_images("sunny beach vacation").await?;
    
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "beach_vacation"); // Beach should rank higher
}

#[tokio::test]
async fn test_image_to_text_search() {
    let engine = LocalNLPEngine::new();
    
    // Index text documents
    engine.index_text("doc1", "Beach vacation memories from last summer").await?;
    engine.index_text("doc2", "Mountain hiking expedition report").await?;
    
    // Search with image query
    let beach_image = load_test_image("test_images/beach.jpg");
    let results = engine.search_texts_by_image(&beach_image).await?;
    
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "doc1"); // Beach document should rank higher
}
```

### Visual Question Answering Tests

#### Basic VQA Tests
```rust
#[tokio::test]
async fn test_image_description() {
    let vqa = Phi4MultimodalLLM::new().await?;
    let image_bytes = load_test_image("test_images/eiffel_tower.jpg");
    
    let response = vqa.answer_question(
        "Describe what you see in this image.",
        &image_bytes
    ).await?;
    
    assert!(response.to_lowercase().contains("eiffel tower"));
    assert!(response.to_lowercase().contains("paris"));
}

#[tokio::test]
async fn test_object_counting() {
    let vqa = Phi4MultimodalLLM::new().await?;
    let image_bytes = load_test_image("test_images/three_cats.jpg");
    
    let response = vqa.answer_question(
        "How many cats are in this image?",
        &image_bytes
    ).await?;
    
    assert!(response.contains("three") || response.contains("3"));
}

#[tokio::test]
async fn test_metadata_integration() {
    let vqa = Phi4MultimodalLLM::new().await?;
    let image_bytes = load_test_image("test_images/timestamped_photo.jpg");
    
    let response = vqa.answer_question(
        "When was this photo taken?",
        &image_bytes
    ).await?;
    
    // Should extract and use EXIF timestamp
    assert!(response.contains("2024") || response.contains("June"));
}
```

#### Multimodal RAG Tests
```rust
#[tokio::test]
async fn test_multimodal_rag_query() {
    let engine = LocalNLPEngine::new();
    
    // Index a product image and description
    let product_image = load_test_image("test_images/laptop.jpg");
    let product_description = "High-performance laptop with 16GB RAM and SSD storage";
    
    engine.index_image("laptop_photo", &product_image).await?;
    engine.index_text("laptop_specs", product_description).await?;
    
    // Ask a question that requires both image and text understanding
    let response = engine.multimodal_query(
        "Based on the image and specifications, what are the main features of this laptop?",
        Some(&product_image)
    ).await?;
    
    assert!(response.contains("16GB RAM"));
    assert!(response.contains("laptop"));
}
```

### Performance Benchmarks

#### Latency Tests
```rust
#[tokio::test]
async fn test_image_embedding_performance() {
    let generator = ImageEmbeddingGenerator::new(ImageEmbeddingConfig::default())?;
    generator.initialize().await?;
    
    let image_bytes = load_test_image("test_images/medium_image.jpg");
    let iterations = 10;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _embedding = generator.generate_embedding(&image_bytes).await?;
    }
    let duration = start.elapsed();
    
    let avg_duration = duration / iterations;
    assert!(avg_duration < Duration::from_millis(50)); // <50ms target
}

#[tokio::test]
async fn test_vqa_performance() {
    let vqa = Phi4MultimodalLLM::new().await?;
    let image_bytes = load_test_image("test_images/simple_scene.jpg");
    
    let start = Instant::now();
    let _response = vqa.answer_question("What do you see?", &image_bytes).await?;
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(500)); // <500ms target
}
```

#### Memory Usage Tests
```rust
#[tokio::test]
async fn test_memory_usage() {
    let initial_memory = get_memory_usage();
    
    let engine = LocalNLPEngine::new();
    engine.initialize().await?;
    
    // Process multiple images
    for i in 0..10 {
        let image_bytes = load_test_image(&format!("test_images/image_{}.jpg", i));
        let _embedding = engine.generate_image_embedding(&image_bytes).await?;
    }
    
    let final_memory = get_memory_usage();
    let memory_increase = final_memory - initial_memory;
    
    assert!(memory_increase < 2_000_000_000); // <2GB increase
}
```

## Test Data Management

### Test Image Collection

#### Standard Test Images
```
test_images/
├── landmarks/
│   ├── eiffel_tower.jpg
│   ├── statue_of_liberty.jpg
│   └── big_ben.jpg
├── nature/
│   ├── mountain_landscape.jpg
│   ├── beach_sunset.jpg
│   └── forest_path.jpg
├── objects/
│   ├── laptop.jpg
│   ├── smartphone.jpg
│   └── coffee_cup.jpg
├── people/
│   ├── family_portrait.jpg
│   ├── business_meeting.jpg
│   └── children_playing.jpg
└── technical/
    ├── chart_bar.jpg
    ├── diagram_flowchart.jpg
    └── text_document.jpg
```

#### Metadata Test Cases
```rust
pub struct TestImageWithMetadata {
    pub filename: String,
    pub expected_timestamp: Option<DateTime<Utc>>,
    pub expected_location: Option<(f64, f64)>,
    pub expected_camera: Option<String>,
    pub description: String,
    pub test_questions: Vec<String>,
    pub expected_answers: Vec<String>,
}

pub fn load_test_images_with_metadata() -> Vec<TestImageWithMetadata> {
    vec![
        TestImageWithMetadata {
            filename: "eiffel_tower_sunset.jpg".to_string(),
            expected_timestamp: Some(DateTime::parse_from_rfc3339("2024-06-15T19:30:00Z")?.into()),
            expected_location: Some((48.8584, 2.2945)), // Paris coordinates
            expected_camera: Some("iPhone 15 Pro".to_string()),
            description: "Eiffel Tower at sunset with golden lighting".to_string(),
            test_questions: vec![
                "What landmark is shown in this image?".to_string(),
                "What time of day was this photo taken?".to_string(),
                "Where was this photo taken?".to_string(),
            ],
            expected_answers: vec![
                "Eiffel Tower".to_string(),
                "sunset".to_string(),
                "Paris".to_string(),
            ],
        },
        // Additional test cases...
    ]
}
```

### PDF Test Documents

#### Test PDF Collection
```
test_documents/
├── simple/
│   ├── text_only.pdf
│   ├── images_only.pdf
│   └── mixed_content.pdf
├── complex/
│   ├── technical_manual.pdf
│   ├── financial_report.pdf
│   └── research_paper.pdf
└── scanned/
    ├── old_document.pdf
    ├── handwritten_notes.pdf
    └── low_quality_scan.pdf
```

## Quality Assurance Testing

### VQA Quality Metrics

#### Accuracy Testing
```rust
#[derive(Debug)]
pub struct VQATestSuite {
    pub name: String,
    pub test_cases: Vec<VQATestCase>,
}

#[derive(Debug)]
pub struct VQATestCase {
    pub image_path: String,
    pub question: String,
    pub expected_answer: String,
    pub category: VQACategory,
    pub difficulty: VQADifficulty,
}

#[tokio::test]
async fn test_vqa_accuracy_suite() {
    let vqa = Phi4MultimodalLLM::new().await?;
    let test_suite = load_vqa_test_suite("comprehensive_vqa_tests.json")?;
    
    let mut correct_answers = 0;
    let mut total_questions = 0;
    
    for test_case in test_suite.test_cases {
        let image_bytes = load_test_image(&test_case.image_path);
        let response = vqa.answer_question(&test_case.question, &image_bytes).await?;
        
        let is_correct = evaluate_answer_similarity(&response, &test_case.expected_answer);
        if is_correct {
            correct_answers += 1;
        }
        total_questions += 1;
        
        println!("Q: {} | Expected: {} | Got: {} | Correct: {}", 
                test_case.question, test_case.expected_answer, response, is_correct);
    }
    
    let accuracy = correct_answers as f32 / total_questions as f32;
    println!("VQA Accuracy: {:.2}%", accuracy * 100.0);
    
    assert!(accuracy >= 0.70); // 70% minimum accuracy requirement
}
```

#### Semantic Similarity Evaluation
```rust
fn evaluate_answer_similarity(generated: &str, expected: &str) -> bool {
    // Use embedding similarity for semantic comparison
    let similarity = calculate_semantic_similarity(generated, expected);
    similarity >= 0.8 // 80% similarity threshold
}

fn calculate_semantic_similarity(text1: &str, text2: &str) -> f32 {
    // Implementation using text embeddings
    let embedding1 = generate_text_embedding(text1);
    let embedding2 = generate_text_embedding(text2);
    cosine_similarity(&embedding1, &embedding2)
}
```

## Integration Testing

### Cross-Modal Integration Tests

#### End-to-End RAG Pipeline
```rust
#[tokio::test]
async fn test_complete_multimodal_rag_pipeline() {
    let engine = LocalNLPEngine::new();
    
    // Phase 1: Index diverse content
    let documents = vec![
        ("Travel itinerary for Japan trip", load_test_document("japan_itinerary.pdf")),
        ("Photo from Tokyo", load_test_image("tokyo_skyline.jpg")),
        ("Restaurant recommendations", "Best sushi restaurants in Tokyo: ..."),
    ];
    
    for (description, content) in documents {
        match content {
            Content::Text(text) => engine.index_text(description, &text).await?,
            Content::Image(bytes) => engine.index_image(description, &bytes).await?,
            Content::PDF(bytes) => engine.index_pdf(description, &bytes).await?,
        }
    }
    
    // Phase 2: Complex multimodal query
    let query = "Based on my travel itinerary and photos, recommend the best time to visit Tokyo for sushi dining";
    let response = engine.multimodal_query(query, None).await?;
    
    // Phase 3: Validate response quality
    assert!(response.contains("Tokyo"));
    assert!(response.contains("sushi"));
    assert!(!response.is_empty());
    
    // Phase 4: Validate source attribution
    let sources = engine.get_response_sources(&response).await?;
    assert!(sources.iter().any(|s| s.contains("itinerary")));
    assert!(sources.iter().any(|s| s.contains("Tokyo")));
}
```

### Performance Integration Tests

#### Concurrent Processing
```rust
#[tokio::test]
async fn test_concurrent_multimodal_processing() {
    let engine = Arc::new(LocalNLPEngine::new());
    let tasks = 10;
    let mut handles = Vec::new();
    
    for i in 0..tasks {
        let engine_clone = Arc::clone(&engine);
        let handle = tokio::spawn(async move {
            let image_bytes = load_test_image(&format!("test_images/concurrent_{}.jpg", i));
            let embedding = engine_clone.generate_image_embedding(&image_bytes).await?;
            Ok::<Vec<f32>, NLPError>(embedding)
        });
        handles.push(handle);
    }
    
    let results = futures::future::try_join_all(handles).await?;
    assert_eq!(results.len(), tasks);
    
    // All embeddings should have correct dimensions
    for result in results {
        let embedding = result?;
        assert_eq!(embedding.len(), 512);
    }
}
```

## Automated Testing Pipeline

### Continuous Integration

#### GitHub Actions Workflow
```yaml
name: Multimodal Testing Pipeline

on: [push, pull_request]

jobs:
  test-phase-1:
    name: Text Foundation Tests
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run unit tests
        run: cargo test
      - name: Run evaluation tests
        run: cargo test --features evaluation

  test-phase-2:
    name: Multimodal Tests
    runs-on: macos-latest
    needs: test-phase-1
    steps:
      - uses: actions/checkout@v3
      - name: Download test images
        run: ./scripts/download_test_data.sh
      - name: Run multimodal tests
        run: cargo test --features multimodal
      - name: Run integration tests
        run: cargo test --features "multimodal,evaluation" --test integration

  performance-benchmarks:
    name: Performance Benchmarks
    runs-on: macos-latest
    needs: test-phase-2
    steps:
      - uses: actions/checkout@v3
      - name: Run performance tests
        run: cargo test --features "multimodal,benchmarks" --test performance
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

### Test Data Management

#### Automated Test Data Download
```bash
#!/bin/bash
# scripts/download_test_data.sh

echo "Downloading test images..."
curl -L https://github.com/nodespace/test-assets/releases/download/v1.0/test_images.tar.gz | tar -xz

echo "Downloading test documents..."
curl -L https://github.com/nodespace/test-assets/releases/download/v1.0/test_documents.tar.gz | tar -xz

echo "Downloading VQA test suite..."
curl -L https://github.com/nodespace/test-assets/releases/download/v1.0/vqa_test_suite.json -o test_data/vqa_test_suite.json

echo "Test data download complete."
```

This comprehensive testing strategy ensures that the multimodal RAG system maintains high quality, performance, and reliability across all development phases.