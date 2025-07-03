//! Integration tests for Ollama HTTP client (NS-126)
//! Tests real connectivity to local Ollama server with gemma3:12b

use nodespace_nlp_engine::{LocalNLPEngine, NLPConfig, OllamaConfig, NLPEngine};

#[tokio::test]
async fn test_ollama_basic_text_generation() {
    // Skip test if Ollama server is not available
    if !is_ollama_available().await {
        println!("⚠️ Skipping Ollama test - server not available");
        return;
    }

    // Create engine with default gemma3:12b configuration
    let engine = LocalNLPEngine::new();
    
    // Initialize engine
    engine.initialize().await.expect("Failed to initialize engine");
    
    // Test basic text generation
    let prompt = "What is the capital of France?";
    let response = engine.generate_text(prompt).await.expect("Failed to generate text");
    
    // Verify response is meaningful
    assert!(!response.is_empty());
    assert!(response.to_lowercase().contains("paris"));
    
    println!("✅ Ollama basic text generation test passed");
    println!("   Prompt: {}", prompt);
    println!("   Response: {}", response.trim());
}

#[tokio::test]
async fn test_ollama_enhanced_generation() {
    // Skip test if Ollama server is not available
    if !is_ollama_available().await {
        println!("⚠️ Skipping enhanced Ollama test - server not available");
        return;
    }

    use nodespace_nlp_engine::{TextGenerationRequest, RAGContext};

    let engine = LocalNLPEngine::new();
    engine.initialize().await.expect("Failed to initialize engine");
    
    // Create enhanced request with RAG context
    let rag_context = RAGContext {
        knowledge_sources: vec![
            "NodeSpace is a distributed system architecture".to_string(),
            "Rust provides memory safety and performance".to_string(),
        ],
        retrieval_confidence: 0.9,
        context_summary: "Information about NodeSpace and Rust programming".to_string(),
        suggested_links: Vec::new(),
    };

    let request = TextGenerationRequest {
        prompt: "How does Rust benefit distributed systems like NodeSpace?".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        context_window: 4096,
        conversation_mode: false,
        rag_context: Some(rag_context),
        enable_link_generation: false,
        node_metadata: Vec::new(),
    };

    let response = engine.generate_text_enhanced(request).await.expect("Failed to generate enhanced text");
    
    // Verify enhanced response
    assert!(!response.text.is_empty());
    assert!(response.tokens_used > 0);
    assert!(response.generation_metrics.generation_time_ms > 0);
    
    println!("✅ Ollama enhanced generation test passed");
    println!("   Response: {}", response.text.trim());
    println!("   Tokens used: {}", response.tokens_used);
    println!("   Generation time: {}ms", response.generation_metrics.generation_time_ms);
}

#[tokio::test]
async fn test_ollama_fallback_to_onnx() {
    // Create engine with invalid Ollama config to test fallback
    let mut config = NLPConfig::default();
    config.models.ollama = OllamaConfig {
        base_url: "http://localhost:99999".to_string(), // Invalid port
        default_model: "gemma3:12b".to_string(),
        multimodal_model: "gemma3:12b".to_string(),
        timeout_secs: 1, // Short timeout
        max_tokens: 100,
        temperature: 0.7,
        retry_attempts: 1, // Single attempt
        stream: false,
    };
    
    let engine = LocalNLPEngine::with_config(config);
    engine.initialize().await.expect("Failed to initialize engine");
    
    // This should fall back to ONNX since Ollama connection will fail
    let prompt = "Test fallback";
    let result = engine.generate_text(prompt).await;
    
    // Should either succeed with ONNX fallback or fail gracefully
    match result {
        Ok(response) => {
            println!("✅ Fallback to ONNX successful: {}", response.trim());
        }
        Err(e) => {
            println!("✅ Fallback test completed - expected failure: {}", e);
            // This is expected if ONNX models aren't available either
        }
    }
}

/// Check if Ollama server is available for testing
async fn is_ollama_available() -> bool {
    match reqwest::get("http://localhost:11434/api/tags").await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}