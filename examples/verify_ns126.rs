//! Verification script for NS-126: Replace ONNX Text Generation with Real Ollama HTTP Client
//! This demonstrates the successful implementation of real HTTP connectivity to Ollama

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 NS-126 Verification: Ollama HTTP Client Implementation");
    println!("=========================================================");

    // Test 1: Direct Ollama HTTP client functionality
    println!("\n📡 Test 1: Direct Ollama HTTP Client");
    test_direct_ollama_client().await?;

    // Test 2: Verify Ollama configuration
    println!("\n⚙️  Test 2: Ollama Configuration");
    test_ollama_config();

    // Test 3: Integration with LocalNLPEngine
    println!("\n🔧 Test 3: NLP Engine Integration");
    test_nlp_engine_integration().await?;

    println!("\n✅ NS-126 Implementation Verification Complete!");
    println!("🎉 Real Ollama HTTP client successfully replaces ONNX text generation");
    
    Ok(())
}

/// Test direct Ollama HTTP client functionality
async fn test_direct_ollama_client() -> Result<(), Box<dyn std::error::Error>> {
    use nodespace_nlp_engine::{OllamaTextGenerator, OllamaConfig};

    // Create Ollama configuration for gemma3:12b
    let config = OllamaConfig {
        base_url: "http://localhost:11434".to_string(),
        default_model: "gemma3:12b".to_string(),
        multimodal_model: "gemma3:12b".to_string(),
        timeout_secs: 30,
        max_tokens: 150,
        temperature: 0.7,
        retry_attempts: 2,
        stream: false,
    };

    println!("   Server: {}", config.base_url);
    println!("   Model: {}", config.default_model);

    // Create and initialize Ollama client
    let mut ollama_client = OllamaTextGenerator::new(config)?;
    
    let start_time = Instant::now();
    ollama_client.initialize().await?;
    let init_duration = start_time.elapsed();
    
    println!("   ✅ Ollama client initialized ({:?})", init_duration);

    // Test text generation
    let prompt = "Explain what makes Rust memory-safe in one sentence:";
    println!("   Prompt: \"{}\"", prompt);
    
    let start_time = Instant::now();
    let response = ollama_client.generate_text(prompt).await?;
    let gen_duration = start_time.elapsed();
    
    println!("   ✅ Response ({:?}): \"{}\"", gen_duration, response.trim());
    
    // Verify response quality
    assert!(!response.is_empty());
    assert!(response.len() > 20); // Meaningful response
    
    Ok(())
}

/// Test Ollama configuration
fn test_ollama_config() {
    use nodespace_nlp_engine::{OllamaConfig, NLPConfig};

    // Test default configuration
    let default_config = OllamaConfig::default();
    println!("   Default URL: {}", default_config.base_url);
    println!("   Default Model: {}", default_config.default_model);
    println!("   Max Tokens: {}", default_config.max_tokens);
    println!("   Temperature: {}", default_config.temperature);
    
    // Test NLP config includes Ollama
    let nlp_config = NLPConfig::default();
    println!("   ✅ Ollama config included in NLPConfig");
    println!("   Server: {}", nlp_config.models.ollama.base_url);
    println!("   Model: {}", nlp_config.models.ollama.default_model);
}

/// Test integration with LocalNLPEngine
async fn test_nlp_engine_integration() -> Result<(), Box<dyn std::error::Error>> {
    use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

    // Create engine with default configuration (includes Ollama)
    let engine = LocalNLPEngine::new();
    
    println!("   Initializing LocalNLPEngine with Ollama support...");
    let start_time = Instant::now();
    engine.initialize().await?;
    let init_duration = start_time.elapsed();
    
    println!("   ✅ Engine initialized ({:?})", init_duration);

    // Check engine status
    let status = engine.status().await;
    println!("   Initialized: {}", status.initialized);
    println!("   Device: {:?}", status.device_type);

    // Test text generation (should use Ollama, not ONNX)
    let prompt = "What is the key advantage of Rust's ownership system?";
    println!("   Testing text generation with prompt: \"{}\"", prompt);
    
    let start_time = Instant::now();
    let response = engine.generate_text(prompt).await?;
    let gen_duration = start_time.elapsed();
    
    println!("   ✅ Generated text ({:?}):", gen_duration);
    println!("      \"{}\"", response.trim());
    
    // Verify response quality
    assert!(!response.is_empty());
    assert!(response.len() > 30);
    assert!(response.to_lowercase().contains("memory") || 
           response.to_lowercase().contains("safety") ||
           response.to_lowercase().contains("ownership"));
    
    println!("   ✅ Response contains expected content about Rust");
    
    Ok(())
}