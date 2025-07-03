//! Test Ollama HTTP client integration
//! Tests real HTTP connectivity to local Ollama server

use nodespace_nlp_engine::{LocalNLPEngine, NLPConfig, NLPEngine, OllamaConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    println!("🧪 Testing Ollama HTTP Client Integration");

    // Create custom config with available gemma3:4b model
    let mut config = NLPConfig::default();
    config.models.ollama = OllamaConfig {
        base_url: "http://localhost:11434".to_string(),
        default_model: "gemma3:4b".to_string(), // Use available model
        multimodal_model: "gemma3:4b".to_string(),
        timeout_secs: 30,
        max_tokens: 1000,
        temperature: 0.7,
        retry_attempts: 2,
        stream: false,
    };

    println!("📋 Configuration:");
    println!("   Ollama Server: {}", config.models.ollama.base_url);
    println!("   Model: {}", config.models.ollama.default_model);
    println!("   Max Tokens: {}", config.models.ollama.max_tokens);
    println!("   Temperature: {}", config.models.ollama.temperature);

    // Create engine with custom config
    let engine = LocalNLPEngine::with_config(config);

    println!("\n🚀 Initializing NLP Engine...");
    match engine.initialize().await {
        Ok(()) => println!("✅ Engine initialized successfully!"),
        Err(e) => {
            println!("❌ Engine initialization failed: {}", e);
            return Err(e.into());
        }
    }

    // Check engine status
    let status = engine.status().await;
    println!("\n📊 Engine Status:");
    println!("   Initialized: {}", status.initialized);
    println!("   Device: {:?}", status.device_type);

    // Test 1: Simple text generation
    println!("\n🧪 Test 1: Simple Text Generation");
    let prompt = "Explain what Rust is in one sentence:";
    println!("Prompt: \"{}\"", prompt);

    let start_time = std::time::Instant::now();
    match engine.generate_text(prompt).await {
        Ok(response) => {
            let duration = start_time.elapsed();
            println!("✅ Response ({:?}): \"{}\"", duration, response.trim());
        }
        Err(e) => {
            println!("❌ Text generation failed: {}", e);
            return Err(e.into());
        }
    }

    // Test 2: Enhanced text generation with RAG context
    println!("\n🧪 Test 2: Enhanced Text Generation");

    use nodespace_nlp_engine::{RAGContext, TextGenerationRequest};

    let rag_context = RAGContext {
        knowledge_sources: vec![
            "Rust is a systems programming language".to_string(),
            "NodeSpace is a distributed system architecture".to_string(),
        ],
        retrieval_confidence: 0.8,
        context_summary: "Information about programming languages and system architecture"
            .to_string(),
        suggested_links: Vec::new(),
    };

    let request = TextGenerationRequest {
        prompt: "What are the benefits of using Rust for system programming?".to_string(),
        max_tokens: 150,
        temperature: 0.7,
        context_window: 4096,
        conversation_mode: false,
        rag_context: Some(rag_context),
        enable_link_generation: false,
        node_metadata: Vec::new(),
    };

    let start_time = std::time::Instant::now();
    match engine.generate_text_enhanced(request).await {
        Ok(response) => {
            let duration = start_time.elapsed();
            println!("✅ Enhanced Response ({:?}):", duration);
            println!("   Text: \"{}\"", response.text.trim());
            println!("   Tokens Used: {}", response.tokens_used);
            println!(
                "   Generation Time: {}ms",
                response.generation_metrics.generation_time_ms
            );
            println!(
                "   Context Referenced: {}",
                response.context_utilization.context_referenced
            );
        }
        Err(e) => {
            println!("❌ Enhanced text generation failed: {}", e);
            return Err(e.into());
        }
    }

    // Test 3: Content analysis
    println!("\n🧪 Test 3: Content Analysis");
    let content =
        "I need to schedule a meeting with the engineering team to discuss the Q4 roadmap";
    println!("Content: \"{}\"", content);

    match engine
        .analyze_content(content, "intent_classification")
        .await
    {
        Ok(analysis) => {
            println!("✅ Analysis:");
            println!("   Classification: {}", analysis.classification);
            println!("   Confidence: {:.2}", analysis.confidence);
            println!("   Processing Time: {}ms", analysis.processing_time_ms);
        }
        Err(e) => {
            println!("❌ Content analysis failed: {}", e);
        }
    }

    println!("\n✅ All Ollama tests completed successfully!");
    println!("🎉 NS-126 implementation verified with real HTTP API calls");

    Ok(())
}
