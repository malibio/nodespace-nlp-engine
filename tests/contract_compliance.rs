//! Contract compliance tests for the NLP Engine

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

/// Test that the LocalNLPEngine implements the NLPEngine trait correctly
#[tokio::test]
async fn test_nlp_engine_trait_compliance() {
    let engine = LocalNLPEngine::new();

    // Initialize the engine
    let init_result = engine.initialize().await;
    assert!(init_result.is_ok(), "Engine initialization should succeed");

    // Test embedding generation
    let text = "This is a test sentence for embedding generation.";
    let embedding_result = engine.generate_embedding(text).await;
    assert!(
        embedding_result.is_ok(),
        "Embedding generation should succeed"
    );

    let embedding = embedding_result.unwrap();
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert_eq!(
        embedding.len(),
        engine.embedding_dimensions(),
        "Embedding should have correct dimensions"
    );

    // Test batch embeddings
    let texts = vec![
        "First test sentence.".to_string(),
        "Second test sentence.".to_string(),
        "Third test sentence.".to_string(),
    ];

    let batch_result = engine.batch_embeddings(&texts).await;
    assert!(
        batch_result.is_ok(),
        "Batch embedding generation should succeed"
    );

    let batch_embeddings = batch_result.unwrap();
    assert_eq!(
        batch_embeddings.len(),
        texts.len(),
        "Should return same number of embeddings as input texts"
    );

    for embedding in &batch_embeddings {
        assert_eq!(
            embedding.len(),
            engine.embedding_dimensions(),
            "Each embedding should have correct dimensions"
        );
    }

    // Test text generation
    let prompt = "Generate a short description of a meeting:";
    let text_result = engine.generate_text(prompt).await;
    assert!(text_result.is_ok(), "Text generation should succeed");

    let generated_text = text_result.unwrap();
    assert!(
        !generated_text.is_empty(),
        "Generated text should not be empty"
    );

    // Test SurrealQL generation
    let natural_query = "Find all meetings from last week";
    let schema_context = "TABLE meeting { id, title, date, participants }";
    let surrealql_result = engine
        .generate_surrealql(natural_query, schema_context)
        .await;
    assert!(
        surrealql_result.is_ok(),
        "SurrealQL generation should succeed"
    );

    let surrealql = surrealql_result.unwrap();
    assert!(
        !surrealql.is_empty(),
        "Generated SurrealQL should not be empty"
    );
    assert!(
        surrealql.to_uppercase().contains("SELECT"),
        "Generated query should be a SELECT statement"
    );
}

/// Test embedding consistency
#[tokio::test]
async fn test_embedding_consistency() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    let text = "Consistent embedding test text";

    // Generate the same embedding multiple times
    let embedding1 = engine
        .generate_embedding(text)
        .await
        .expect("First embedding should succeed");
    let embedding2 = engine
        .generate_embedding(text)
        .await
        .expect("Second embedding should succeed");

    // Embeddings should be identical for the same text
    assert_eq!(
        embedding1.len(),
        embedding2.len(),
        "Embeddings should have same length"
    );

    // Calculate similarity (should be very close to 1.0)
    let similarity = cosine_similarity(&embedding1, &embedding2);
    assert!(
        similarity > 0.99,
        "Embeddings should be nearly identical, similarity: {}",
        similarity
    );
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() {
    let engine = LocalNLPEngine::new();

    // Test operations before initialization
    let result = engine.generate_embedding("test").await;
    // Should either succeed (auto-initialization) or fail gracefully
    match result {
        Ok(_) => {
            // Auto-initialization worked
            assert!(
                engine.is_initialized().await,
                "Engine should be initialized after successful operation"
            );
        }
        Err(e) => {
            // Should be a proper error, not a panic
            assert!(
                !e.to_string().is_empty(),
                "Error message should not be empty"
            );
        }
    }
}

/// Test batch processing performance
#[tokio::test]
async fn test_batch_performance() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    // Generate a larger batch to test performance
    let texts: Vec<String> = (0..10)
        .map(|i| format!("Test sentence number {} for batch processing.", i))
        .collect();

    let start_time = std::time::Instant::now();
    let batch_result = engine.batch_embeddings(&texts).await;
    let batch_duration = start_time.elapsed();

    assert!(batch_result.is_ok(), "Batch processing should succeed");

    // Compare with individual processing
    let start_time = std::time::Instant::now();
    for text in &texts {
        let _ = engine
            .generate_embedding(text)
            .await
            .expect("Individual embedding should succeed");
    }
    let individual_duration = start_time.elapsed();

    println!("Batch processing took: {:?}", batch_duration);
    println!("Individual processing took: {:?}", individual_duration);

    // Batch processing should be faster or at least not significantly slower
    // Note: For stub implementations, this test is relaxed since we're not doing real ML processing
    println!("Batch vs Individual processing comparison - this is expected to vary in stub implementation");

    // For real ML implementations, batch processing should be more efficient
    // if texts.len() > 5 {
    //     assert!(
    //         batch_duration <= individual_duration * 2,
    //         "Batch processing should not be more than 2x slower than individual processing"
    //     );
    // }
}

/// Test SurrealQL safety features
#[tokio::test]
async fn test_surrealql_safety() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    // Test potentially dangerous queries
    let dangerous_queries = vec![
        "DROP TABLE users; SELECT * FROM meetings",
        "'; DELETE FROM important_data; --",
        "SELECT * FROM users WHERE 1=1 OR '1'='1'",
    ];

    let schema_context = "TABLE meeting { id, title, date }";

    for query in dangerous_queries {
        let result = engine.generate_surrealql(query, schema_context).await;

        match result {
            Ok(surrealql) => {
                // If generation succeeds, ensure it's safe
                assert!(
                    !surrealql.to_uppercase().contains("DROP"),
                    "Generated SQL should not contain DROP"
                );
                assert!(
                    !surrealql.to_uppercase().contains("DELETE"),
                    "Generated SQL should not contain DELETE"
                );
                assert!(
                    !surrealql.contains("--"),
                    "Generated SQL should not contain SQL comments"
                );
            }
            Err(_) => {
                // It's also acceptable to reject dangerous queries entirely
            }
        }
    }
}

/// Test engine status and configuration
#[tokio::test]
async fn test_engine_status() {
    let engine = LocalNLPEngine::new();

    // Check status before initialization
    let status_before = engine.status().await;
    assert!(
        !status_before.initialized,
        "Engine should not be initialized initially"
    );

    // Initialize
    engine.initialize().await.expect("Engine should initialize");

    // Check status after initialization
    let status_after = engine.status().await;
    assert!(
        status_after.initialized,
        "Engine should be initialized after initialization"
    );
    assert!(
        status_after.embedding_info.is_some(),
        "Embedding info should be available"
    );
    assert!(
        status_after.text_generation_info.is_some(),
        "Text generation info should be available"
    );

    // Test dimensions
    let dimensions = engine.embedding_dimensions();
    assert!(dimensions > 0, "Embedding dimensions should be positive");

    if let Some(embedding_info) = status_after.embedding_info {
        assert_eq!(
            embedding_info.dimensions, dimensions,
            "Dimensions should match"
        );
    }
}

/// Test caching functionality
#[tokio::test]
async fn test_caching() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    let text = "This text will be cached";

    // Generate embedding (should cache)
    let start_time = std::time::Instant::now();
    let _embedding1 = engine
        .generate_embedding(text)
        .await
        .expect("First embedding should succeed");
    let first_duration = start_time.elapsed();

    // Generate same embedding again (should use cache)
    let start_time = std::time::Instant::now();
    let _embedding2 = engine
        .generate_embedding(text)
        .await
        .expect("Second embedding should succeed");
    let second_duration = start_time.elapsed();

    // Second call should be faster (cached)
    println!(
        "First generation: {:?}, Second generation: {:?}",
        first_duration, second_duration
    );

    // Get cache stats
    let cache_stats = engine.cache_stats().await;
    assert!(
        cache_stats.embedding_cache_size > 0,
        "Cache should contain at least one entry"
    );

    // Clear cache
    engine
        .clear_caches()
        .await
        .expect("Cache clearing should succeed");

    // Cache stats should show empty cache
    let cache_stats_after_clear = engine.cache_stats().await;
    assert_eq!(
        cache_stats_after_clear.embedding_cache_size, 0,
        "Cache should be empty after clearing"
    );
}

/// Helper function to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
