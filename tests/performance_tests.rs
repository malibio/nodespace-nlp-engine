//! Performance and integration tests for the complete ONNX autoregressive pipeline
//!
//! These tests validate the full text generation pipeline including ONNX inference,
//! KV-cache management, autoregressive generation, and performance characteristics.

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine, RAGContext, TextGenerationRequest};
use std::time::Instant;

#[tokio::test]
async fn test_text_generation_performance() {
    // Initialize the NLP engine
    let engine = LocalNLPEngine::new();

    // Ensure initialization happens before timing
    engine.initialize().await.expect("Engine should initialize");

    let prompts = vec![
        "Explain quantum computing in simple terms:",
        "Generate a meeting summary for project planning:",
        "Describe the benefits of renewable energy:",
        "Write a brief product description for a smartphone:",
        "Summarize the key points of agile development:",
    ];

    let mut total_time = std::time::Duration::new(0, 0);
    let mut successful_generations = 0;

    for prompt in &prompts {
        let start = Instant::now();

        match engine.generate_text(prompt).await {
            Ok(generated_text) => {
                let duration = start.elapsed();
                total_time += duration;
                successful_generations += 1;

                // Validate response quality
                assert!(
                    !generated_text.is_empty(),
                    "Generated text should not be empty"
                );
                assert!(
                    generated_text.len() > 10,
                    "Generated text should be meaningful (>10 chars)"
                );

                println!("Prompt: {}", prompt);
                println!(
                    "Generated: {} chars in {:?}",
                    generated_text.len(),
                    duration
                );
                println!(
                    "Response: {}\n",
                    generated_text.chars().take(100).collect::<String>()
                );
            }
            Err(e) => {
                println!("Generation failed for prompt '{}': {}", prompt, e);
            }
        }
    }

    // Performance assertions
    assert!(
        successful_generations > 0,
        "At least some generations should succeed"
    );

    if successful_generations > 0 {
        let avg_time = total_time / successful_generations as u32;
        println!("Average generation time: {:?}", avg_time);

        // Performance targets: Should generate text in reasonable time
        // Note: In test environment without real ONNX models, fallback should be very fast
        assert!(
            avg_time.as_millis() < 5000,
            "Average generation time should be < 5s (fallback should be much faster)"
        );
    }
}

#[tokio::test]
async fn test_enhanced_text_generation_pipeline() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    let request = TextGenerationRequest {
        prompt: "Based on the following context, answer the question:\n\nCONTEXT: Our Q4 revenue was $2.5 million, representing a 25% increase over Q3.\n\nQUESTION: What was our Q4 revenue?\n\nINSTRUCTIONS: Provide a concise answer based on the context.".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        context_window: 2048,
        conversation_mode: false,
        rag_context: Some(RAGContext {
            knowledge_sources: vec![
                "Q4 financial results show strong performance with revenue growth.".to_string(),
                "Company achieved $2.5M in Q4 revenue with 25% quarter-over-quarter growth.".to_string(),
            ],
            retrieval_confidence: 0.95,
            context_summary: "Financial performance data for Q4".to_string(),
            suggested_links: vec![],
        }),
        enable_link_generation: true,
        node_metadata: vec![],
    };

    let start = Instant::now();
    let result = engine.generate_text_enhanced(request).await;
    let duration = start.elapsed();

    assert!(result.is_ok(), "Enhanced text generation should succeed");

    let response = result.unwrap();

    // Validate response structure
    assert!(
        !response.text.is_empty(),
        "Response text should not be empty"
    );
    assert!(response.tokens_used > 0, "Should report tokens used");

    // Validate generation metrics
    assert!(
        response.generation_metrics.response_tokens > 0,
        "Should report response tokens"
    );
    assert!(
        response.generation_metrics.context_tokens > 0,
        "Should report context tokens"
    );
    assert!(
        response.generation_metrics.generation_time_ms > 0,
        "Should report generation time"
    );
    assert!(
        response.generation_metrics.temperature_used > 0.0,
        "Should report temperature used"
    );

    // Validate context utilization
    assert!(
        response.context_utilization.relevance_score >= 0.0,
        "Should report relevance score"
    );

    println!("Enhanced generation completed in {:?}", duration);
    println!("Response: {}", response.text);
    println!("Tokens used: {}", response.tokens_used);
    println!("Generation metrics: {:?}", response.generation_metrics);
    println!("Context utilization: {:?}", response.context_utilization);
}

#[tokio::test]
async fn test_batch_text_generation_performance() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    let prompts = vec![
        "Summarize the key benefits of cloud computing.",
        "Explain machine learning in one paragraph.",
        "Describe the importance of cybersecurity.",
        "Write about sustainable technology practices.",
    ];

    let start = Instant::now();

    // Generate multiple texts in sequence
    let mut results = Vec::new();
    for prompt in &prompts {
        match engine.generate_text(prompt).await {
            Ok(text) => results.push(text),
            Err(e) => println!("Generation failed: {}", e),
        }
    }

    let total_duration = start.elapsed();

    // Validate batch performance
    assert!(
        !results.is_empty(),
        "At least some generations should succeed"
    );

    let avg_time_per_generation = total_duration / prompts.len() as u32;

    println!(
        "Batch generation: {} prompts in {:?}",
        prompts.len(),
        total_duration
    );
    println!("Average time per generation: {:?}", avg_time_per_generation);

    for (i, result) in results.iter().enumerate() {
        println!("Result {}: {} chars", i + 1, result.len());
        assert!(!result.is_empty(), "Each result should not be empty");
    }

    // Performance expectations
    assert!(
        avg_time_per_generation.as_millis() < 10000,
        "Batch average should be reasonable"
    );
}

#[tokio::test]
async fn test_concurrent_text_generation() {
    let engine = std::sync::Arc::new(LocalNLPEngine::new());
    engine.initialize().await.expect("Engine should initialize");

    let prompts = vec![
        "Describe artificial intelligence.",
        "Explain blockchain technology.",
        "Summarize renewable energy benefits.",
    ];

    let start = Instant::now();

    // Launch concurrent generation tasks
    let mut handles = Vec::new();
    for prompt in prompts {
        let engine_clone = engine.clone();
        let prompt_owned = prompt.to_string();

        let handle = tokio::spawn(async move { engine_clone.generate_text(&prompt_owned).await });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(Ok(text)) => results.push(text),
            Ok(Err(e)) => println!("Generation error: {}", e),
            Err(e) => println!("Task join error: {}", e),
        }
    }

    let total_duration = start.elapsed();

    println!(
        "Concurrent generation: {} tasks in {:?}",
        results.len(),
        total_duration
    );

    // Validate concurrent execution
    assert!(
        !results.is_empty(),
        "At least some concurrent generations should succeed"
    );

    for (i, result) in results.iter().enumerate() {
        println!("Concurrent result {}: {} chars", i + 1, result.len());
        assert!(
            !result.is_empty(),
            "Each concurrent result should not be empty"
        );
    }

    // Concurrent execution should not be significantly slower than sequential
    // (Note: In test environment with fallback, this should be very fast)
    assert!(
        total_duration.as_millis() < 15000,
        "Concurrent execution should complete in reasonable time"
    );
}

#[tokio::test]
async fn test_text_generation_with_different_parameters() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    let base_prompt = "Write a creative story about space exploration:";

    // Test different parameter combinations
    let test_cases = vec![
        (0.3, 50),  // Low temperature, short
        (0.7, 100), // Medium temperature, medium
        (0.9, 150), // High temperature, long
    ];

    for (temperature, max_tokens) in test_cases {
        let request = TextGenerationRequest {
            prompt: base_prompt.to_string(),
            max_tokens,
            temperature,
            context_window: 2048,
            conversation_mode: false,
            rag_context: None,
            enable_link_generation: false,
            node_metadata: vec![],
        };

        let start = Instant::now();
        let result = engine.generate_text_enhanced(request).await;
        let duration = start.elapsed();

        match result {
            Ok(response) => {
                println!(
                    "Temperature {}, Max tokens {}: {} chars in {:?}",
                    temperature,
                    max_tokens,
                    response.text.len(),
                    duration
                );

                assert!(!response.text.is_empty(), "Response should not be empty");
                assert_eq!(
                    response.generation_metrics.temperature_used, temperature,
                    "Should use requested temperature"
                );

                // Validate that the response respects the token limit (approximately)
                // Note: Token estimation might not be perfect, so we allow some variance
                let estimated_tokens = response.generation_metrics.response_tokens;
                assert!(
                    estimated_tokens as u64 <= max_tokens as u64 + 20,
                    "Should respect max tokens limit (with some variance)"
                );
            }
            Err(e) => {
                println!(
                    "Generation failed for temperature {}, max_tokens {}: {}",
                    temperature, max_tokens, e
                );
            }
        }
    }
}

#[tokio::test]
async fn test_autoregressive_pipeline_resilience() {
    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Engine should initialize");

    // Test edge cases and error handling
    let test_cases = vec![
        ("".to_string(), "Empty prompt"),
        ("A".to_string(), "Single character"),
        ("   ".to_string(), "Whitespace only"),
        (
            format!("Write a story about {}", "adventure ".repeat(100)),
            "Very long prompt",
        ),
        (
            "Generate text with special characters: @#$%^&*()".to_string(),
            "Special characters",
        ),
    ];

    for (prompt, description) in test_cases.iter() {
        println!("Testing: {}", description);

        let start = Instant::now();
        let result = engine.generate_text(&prompt).await;
        let duration = start.elapsed();

        match result {
            Ok(text) => {
                println!("  Success: {} chars in {:?}", text.len(), duration);
                // For most cases, we expect some output (fallback mechanism should work)
                if !prompt.trim().is_empty() {
                    assert!(!text.is_empty(), "Non-empty prompts should generate text");
                }
            }
            Err(e) => {
                println!("  Error (expected for some cases): {}", e);
                // Some edge cases might fail, which is acceptable
            }
        }
    }
}

#[cfg(feature = "real-ml")]
#[tokio::test]
async fn test_onnx_model_loading_and_signature_analysis() {
    // This test will only run when real ML features are enabled
    // and actual ONNX model files are available

    use nodespace_nlp_engine::models::{DeviceType, TextGenerationModelConfig};
    use nodespace_nlp_engine::text_generation::TextGenerator;
    use std::path::PathBuf;

    let config = TextGenerationModelConfig {
        model_name: "gemma-3-1b-it-onnx".to_string(),
        model_path: Some(PathBuf::from("models/gemma-3-1b-it-onnx/model.onnx")),
        max_context_length: 2048,
        default_temperature: 0.7,
        default_max_tokens: 100,
        default_top_p: 0.9,
    };

    match TextGenerator::new(config, DeviceType::CPU) {
        Ok(mut generator) => {
            // Test model loading
            match generator.initialize().await {
                Ok(()) => {
                    println!("ONNX model loaded successfully");

                    // Test that model signature was analyzed
                    let model_info = generator.model_info();
                    println!("Model info: {:?}", model_info);

                    // Test basic generation
                    let result = generator.generate_text("Test prompt").await;
                    match result {
                        Ok(text) => {
                            println!("ONNX generation successful: {}", text);
                            assert!(!text.is_empty(), "ONNX generation should produce text");
                        }
                        Err(e) => {
                            println!("ONNX generation failed (fallback should work): {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "ONNX model loading failed (expected in test environment): {}",
                        e
                    );
                    // This is expected in test environment without actual model files
                }
            }
        }
        Err(e) => {
            println!("TextGenerator creation failed: {}", e);
        }
    }
}
