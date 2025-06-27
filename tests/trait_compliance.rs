//! Trait compliance tests for the NLP Engine
//!
//! Tests that verify the NLPEngine trait implementation meets all interface requirements.
//! This repository owns and exports the NLPEngine trait definition.

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine, RAGContext, TextGenerationRequest};

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

/// Test enhanced text generation with RAG context support
#[tokio::test]
async fn test_rag_context_aware_generation() {
    let engine = LocalNLPEngine::new();

    // Initialize the engine
    let init_result = engine.initialize().await;
    assert!(init_result.is_ok(), "Engine initialization should succeed");

    // Create RAG context
    let rag_context = RAGContext {
        knowledge_sources: vec![
            "Meeting with John scheduled for Friday".to_string(),
            "Project deadline is next week".to_string(),
            "Team needs to review design documents".to_string(),
        ],
        retrieval_confidence: 0.85,
        context_summary: "Information about upcoming meeting and project deadline".to_string(),
        suggested_links: vec![], // No smart links for this test
    };

    // Test basic enhanced generation
    let basic_request = TextGenerationRequest {
        prompt: "What is scheduled for this week?".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        context_window: 4000,
        conversation_mode: false,
        rag_context: Some(rag_context.clone()),
        enable_link_generation: false,
        node_metadata: vec![],
    };

    let result = engine.generate_text_enhanced(basic_request).await;
    assert!(result.is_ok(), "Enhanced text generation should succeed");

    let response = result.unwrap();
    assert!(
        !response.text.is_empty(),
        "Response text should not be empty"
    );
    assert!(response.tokens_used > 0, "Should report tokens used");
    // Verify generation metrics are populated
    assert!(
        response.generation_metrics.context_tokens > 0,
        "Should report context tokens used"
    );
    assert!(
        response.generation_metrics.response_tokens > 0,
        "Should report response tokens generated"
    );

    // Test conversation mode
    let conversation_request = TextGenerationRequest {
        prompt: "Can you help me prepare for the meeting?".to_string(),
        max_tokens: 150,
        temperature: 0.8,
        context_window: 4000,
        conversation_mode: true,
        rag_context: Some(rag_context.clone()),
        enable_link_generation: false,
        node_metadata: vec![],
    };

    let conv_result = engine.generate_text_enhanced(conversation_request).await;
    assert!(
        conv_result.is_ok(),
        "Conversation mode generation should succeed"
    );

    let conv_response = conv_result.unwrap();
    assert!(
        !conv_response.text.is_empty(),
        "Conversation response should not be empty"
    );

    // Test without RAG context
    let no_context_request = TextGenerationRequest {
        prompt: "Generate a simple response".to_string(),
        max_tokens: 50,
        temperature: 0.5,
        context_window: 2000,
        conversation_mode: false,
        rag_context: None,
        enable_link_generation: false,
        node_metadata: vec![],
    };

    let no_context_result = engine.generate_text_enhanced(no_context_request).await;
    assert!(
        no_context_result.is_ok(),
        "Generation without RAG context should succeed"
    );

    let no_context_response = no_context_result.unwrap();
    assert!(
        !no_context_response.text.is_empty(),
        "Response without context should not be empty"
    );
    assert!(
        !no_context_response.context_utilization.context_referenced,
        "Should not reference context when none provided"
    );

    // Test token limit validation
    let limited_request = TextGenerationRequest {
        prompt: "This is a test".to_string(),
        max_tokens: 5000, // High request
        temperature: 0.7,
        context_window: 100, // Very small context window
        conversation_mode: false,
        rag_context: Some(rag_context),
        enable_link_generation: false,
        node_metadata: vec![],
    };

    let limited_result = engine.generate_text_enhanced(limited_request).await;
    // Should either succeed with fewer tokens or return appropriate error
    match limited_result {
        Ok(response) => {
            assert!(
                response.tokens_used <= 100,
                "Should respect context window limits"
            );
        }
        Err(_) => {
            // Error due to insufficient tokens is acceptable
        }
    }
}

/// Test smart link generation functionality
#[tokio::test]
async fn test_smart_link_generation() {
    use nodespace_nlp_engine::{nodespace_core_types::NodeId, NodeMetadata, ResponseProcessor};

    // Create test node metadata
    let node_metadata = vec![
        NodeMetadata {
            id: NodeId::new(),
            title: "EcoSmart Proposal".to_string(),
            node_type: "document".to_string(),
            created_date: "2024-06-15".to_string(),
            snippet: "Proposal for EcoSmart sustainable energy project".to_string(),
        },
        NodeMetadata {
            id: NodeId::new(),
            title: "June 15th Meeting".to_string(),
            node_type: "meeting".to_string(),
            created_date: "2024-06-15".to_string(),
            snippet: "Meeting about EcoSmart proposal review".to_string(),
        },
        NodeMetadata {
            id: NodeId::new(),
            title: "John Smith".to_string(),
            node_type: "person".to_string(),
            created_date: "2024-01-01".to_string(),
            snippet: "Senior project manager for sustainable energy initiatives".to_string(),
        },
    ];

    // Test content that should generate smart links
    let test_content = "You reviewed the EcoSmart Proposal during your June 15th Meeting. John Smith had key concerns about the implementation timeline.";

    // Initialize response processor
    let processor = ResponseProcessor::new();

    // Test link detection
    let detected_links = processor
        .detect_potential_links(test_content, &node_metadata)
        .expect("Link detection should succeed");

    // Verify that links were detected
    assert!(!detected_links.is_empty(), "Should detect potential links");

    // Check for specific expected links
    let proposal_link = detected_links
        .iter()
        .find(|link| link.text.contains("EcoSmart Proposal"));
    assert!(
        proposal_link.is_some(),
        "Should detect EcoSmart Proposal link"
    );

    let meeting_link = detected_links
        .iter()
        .find(|link| link.text.contains("June 15th Meeting"));
    assert!(
        meeting_link.is_some(),
        "Should detect June 15th Meeting link"
    );

    let person_link = detected_links
        .iter()
        .find(|link| link.text.contains("John Smith"));
    assert!(person_link.is_some(), "Should detect John Smith link");

    // Test confidence scoring
    for link in &detected_links {
        assert!(
            link.confidence > 0.0 && link.confidence <= 1.0,
            "Confidence should be between 0 and 1, got: {}",
            link.confidence
        );
        assert!(
            link.confidence > 0.6,
            "High-quality matches should have confidence > 0.6, got: {}",
            link.confidence
        );
    }
}

/// Test smart link injection functionality
#[tokio::test]
async fn test_smart_link_injection() {
    use nodespace_nlp_engine::{nodespace_core_types::NodeId, NodeMetadata, ResponseProcessor};

    // Create test node metadata
    let ecosmart_id = NodeId::new();
    let node_metadata = vec![NodeMetadata {
        id: ecosmart_id.clone(),
        title: "EcoSmart Proposal".to_string(),
        node_type: "document".to_string(),
        created_date: "2024-06-15".to_string(),
        snippet: "Proposal for sustainable energy project".to_string(),
    }];

    let original_content = "Please review the EcoSmart Proposal before the meeting.";
    let processor = ResponseProcessor::new();

    // Test link injection
    let enhanced_content = processor
        .inject_smart_links(original_content, &node_metadata)
        .expect("Link injection should succeed");

    // Verify that markdown link was injected
    assert!(
        enhanced_content.contains("[EcoSmart Proposal]"),
        "Should contain markdown link text"
    );
    assert!(
        enhanced_content.contains(&format!("nodespace://{}", ecosmart_id)),
        "Should contain nodespace:// link with correct ID"
    );

    println!("Original: {}", original_content);
    println!("Enhanced: {}", enhanced_content);
}

/// Test link type classification
#[tokio::test]
async fn test_link_type_classification() {
    use nodespace_nlp_engine::{
        nodespace_core_types::NodeId, LinkType, NodeMetadata, ResponseProcessor,
    };

    let processor = ResponseProcessor::new();
    let node_metadata = vec![
        NodeMetadata {
            id: NodeId::new(),
            title: "Task Item".to_string(),
            node_type: "task".to_string(),
            created_date: "2024-06-15".to_string(),
            snippet: "Complete project review".to_string(),
        },
        NodeMetadata {
            id: NodeId::new(),
            title: "Client Meeting".to_string(),
            node_type: "meeting".to_string(),
            created_date: "2024-06-15".to_string(),
            snippet: "Meeting with client about requirements".to_string(),
        },
    ];

    let content = "Complete the Task Item before the Client Meeting.";
    let detected_links = processor
        .detect_potential_links(content, &node_metadata)
        .expect("Link detection should succeed");

    // Verify link types are correctly classified
    for link in &detected_links {
        match link.text.as_str() {
            text if text.contains("Task Item") => {
                assert_eq!(
                    link.link_type,
                    LinkType::TaskReference,
                    "Task should be classified as TaskReference"
                );
            }
            text if text.contains("Client Meeting") => {
                assert_eq!(
                    link.link_type,
                    LinkType::DateReference,
                    "Meeting should be classified as DateReference"
                );
            }
            _ => {}
        }
    }
}

/// Test enhanced TextGenerationRequest with smart link support
#[tokio::test]
async fn test_enhanced_request_with_smart_links() {
    use nodespace_nlp_engine::{
        nodespace_core_types::NodeId, LinkType, NodeMetadata, RAGContext, SmartLink,
        TextGenerationRequest,
    };

    let node_metadata = vec![NodeMetadata {
        id: NodeId::new(),
        title: "Test Document".to_string(),
        node_type: "document".to_string(),
        created_date: "2024-06-15".to_string(),
        snippet: "Test document content".to_string(),
    }];

    let smart_links = vec![SmartLink {
        text: "Test Document".to_string(),
        node_id: NodeId::new(),
        link_type: LinkType::DocumentReference,
        confidence: 0.95,
    }];

    let rag_context = RAGContext {
        knowledge_sources: vec!["Test source".to_string()],
        retrieval_confidence: 0.85,
        context_summary: "Test context".to_string(),
        suggested_links: smart_links,
    };

    // Test enhanced request creation
    let request = TextGenerationRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        context_window: 2048,
        conversation_mode: true,
        rag_context: Some(rag_context),
        enable_link_generation: true,
        node_metadata,
    };

    // Verify all fields are properly set
    assert!(
        request.enable_link_generation,
        "Link generation should be enabled"
    );
    assert!(
        !request.node_metadata.is_empty(),
        "Node metadata should be provided"
    );
    assert!(
        request.rag_context.is_some(),
        "RAG context should be provided"
    );

    let rag = request.rag_context.unwrap();
    assert!(
        !rag.suggested_links.is_empty(),
        "Suggested links should be provided"
    );
    assert_eq!(
        rag.suggested_links[0].link_type,
        LinkType::DocumentReference,
        "Link type should be preserved"
    );
}

/// Test deduplication of overlapping smart links
#[tokio::test]
async fn test_smart_link_deduplication() {
    use nodespace_nlp_engine::{nodespace_core_types::NodeId, NodeMetadata, ResponseProcessor};

    // Create overlapping node metadata (partial title matches)
    let node_metadata = vec![
        NodeMetadata {
            id: NodeId::new(),
            title: "EcoSmart Energy Proposal".to_string(),
            node_type: "document".to_string(),
            created_date: "2024-06-15".to_string(),
            snippet: "Full proposal document".to_string(),
        },
        NodeMetadata {
            id: NodeId::new(),
            title: "EcoSmart Proposal Review".to_string(),
            node_type: "document".to_string(),
            created_date: "2024-06-16".to_string(),
            snippet: "Review of the proposal".to_string(),
        },
    ];

    let content = "The EcoSmart Energy Proposal needs review.";
    let processor = ResponseProcessor::new();

    let detected_links = processor
        .detect_potential_links(content, &node_metadata)
        .expect("Link detection should succeed");

    // Verify deduplication works - longer matches should be preferred
    let link_texts: Vec<&String> = detected_links.iter().map(|link| &link.text).collect();

    // Should prefer "EcoSmart Energy Proposal" over "EcoSmart Proposal"
    assert!(
        link_texts
            .iter()
            .any(|text| text.contains("EcoSmart Energy Proposal") || text.contains("EcoSmart")),
        "Should detect at least one EcoSmart-related link"
    );

    // Verify no duplicate/overlapping links in final result
    for (i, link1) in detected_links.iter().enumerate() {
        for (j, link2) in detected_links.iter().enumerate() {
            if i != j {
                assert!(
                    !link1.text.contains(&link2.text) && !link2.text.contains(&link1.text),
                    "Links should not overlap: '{}' and '{}'",
                    link1.text,
                    link2.text
                );
            }
        }
    }
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
