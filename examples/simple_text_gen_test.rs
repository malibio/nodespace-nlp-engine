//! Simple test to isolate text generation model loading

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber with simpler output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    println!("🔍 Testing text generation model loading...");

    // Create engine
    let model_dir = PathBuf::from("../models");
    let engine = LocalNLPEngine::with_model_directory(model_dir);

    println!("📡 Initializing NLP engine...");
    match engine.initialize().await {
        Ok(()) => {
            println!("✅ NLP engine initialized successfully");
        }
        Err(e) => {
            println!("❌ NLP engine initialization failed: {}", e);
            return Err(e.into());
        }
    }

    println!("🎯 Testing simple text generation...");

    // Test with a very simple prompt to isolate issues
    let prompt = "Hello";
    println!("📝 Prompt: '{}'", prompt);

    let start = std::time::Instant::now();
    match engine.generate_text(prompt).await {
        Ok(response) => {
            let duration = start.elapsed();
            println!("⏱️  Duration: {:?}", duration);
            println!("🤖 Response: '{}'", response);

            // Very specific analysis
            if response == "A good team meeting requires clear objectives, structured agenda, active participation from all members, and defined action items." {
                println!("❌ DETECTED: Canned meeting response - using stub implementation");
            } else if response == "Based on the available information, this requires further analysis to provide a comprehensive response." {
                println!("❌ DETECTED: Generic canned response - using stub implementation");
            } else if response.contains("ONNX Runtime working") || response.contains("Generated response") {
                println!("✅ DETECTED: ONNX inference response");
            } else {
                println!("🤔 UNKNOWN: Response '{}' - could be real ONNX", response);
            }
        }
        Err(e) => {
            println!("❌ Text generation failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
