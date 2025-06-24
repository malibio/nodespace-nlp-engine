//! Debug text generation with TinyLlama

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Testing NodeSpace NLP Engine with TinyLlama model");

    // Create engine
    let engine = LocalNLPEngine::new();

    // Initialize
    println!("📋 Initializing engine...");
    match engine.initialize().await {
        Ok(_) => println!("✅ Engine initialized successfully"),
        Err(e) => {
            println!("❌ Engine initialization failed: {:?}", e);
            return Err(format!("Engine initialization failed: {:?}", e).into());
        }
    }

    // Test text generation
    println!("📝 Testing text generation...");
    let prompt = "What is a meeting?";

    match engine.generate_text(prompt).await {
        Ok(response) => {
            println!("✅ Text generation successful!");
            println!("📤 Prompt: {}", prompt);
            println!("📥 Response: {}", response);
        }
        Err(e) => {
            println!("❌ Text generation failed: {:?}", e);
            return Err(format!("Text generation failed: {:?}", e).into());
        }
    }

    println!("🎉 All tests completed successfully!");
    Ok(())
}
