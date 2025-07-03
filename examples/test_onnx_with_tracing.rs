//! Test ONNX text generation with full tracing enabled

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber to see debug logs
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    println!("🔍 Testing ONNX inference with full logging...");

    // Create engine with explicit model directory
    let model_dir = PathBuf::from("../models");
    let engine = LocalNLPEngine::with_model_directory(model_dir);

    println!("📡 Initializing NLP engine...");
    engine.initialize().await?;

    println!("🎯 Testing text generation...");

    // Test with a simple prompt
    let prompt = "What is machine learning?";
    println!("📝 Prompt: {}", prompt);

    let start = std::time::Instant::now();
    let response = engine.generate_text(prompt).await?;
    let duration = start.elapsed();

    println!("⏱️  Duration: {:?}", duration);
    println!("🤖 Response: {}", response);

    // Analyze the response
    if response.contains("good team meeting requires")
        || response.contains("task requires careful planning")
        || response.contains("This is a generated response from the NodeSpace")
    {
        println!("❌ DETECTED: Canned/stub response - ONNX not working");
    } else if response.contains("ONNX Runtime working")
        || response.contains("inference attempted")
        || response.contains("single inference pass")
    {
        println!("✅ DETECTED: ONNX inference response");
    } else {
        println!("🤔 UNKNOWN: Response pattern doesn't match known canned responses");
        println!("   This could be real ONNX output or a new fallback pattern");
    }

    Ok(())
}
