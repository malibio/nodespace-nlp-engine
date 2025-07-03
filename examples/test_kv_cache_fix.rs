//! Test the KV cache fix for ONNX inference
//!
//! This example tests the new KV cache implementation to ensure
//! all 54 required inputs are provided to the Gemma-3-1B ONNX model.

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use std::path::PathBuf;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize comprehensive logging
    tracing_subscriber::fmt::init();

    println!("🚀 Testing KV Cache Fix for ONNX Inference");
    println!("==========================================");
    println!("");

    // Create NLP engine with explicit model path
    let model_dir = PathBuf::from("/Users/malibio/nodespace/models");
    println!("🔍 Using model directory: {}", model_dir.display());

    let nlp_engine = LocalNLPEngine::with_model_directory(model_dir);
    println!("✅ LocalNLPEngine created successfully");

    println!("🔧 Initializing NLP engine components...");
    if let Err(e) = nlp_engine.initialize().await {
        println!("❌ Failed to initialize LocalNLPEngine: {}", e);
        return Err(e.into());
    }
    println!("✅ LocalNLPEngine initialized successfully");

    println!("");
    println!("🔧 Testing simple text generation with KV cache fix...");

    // Test with a simple prompt
    let test_prompt = "Hello, how are you today?";
    println!("🔤 Test prompt: '{}'", test_prompt);

    match nlp_engine.generate_text(test_prompt).await {
        Ok(response) => {
            println!("✅ Text generation succeeded!");
            println!("📝 Response: {}", response);
            println!("📏 Response length: {} characters", response.len());

            // Check if this is still a fallback response
            if response.contains("fallback") || response.contains("technical difficulties") {
                println!("⚠️ WARNING: This appears to be a fallback response");
                println!("🔍 This indicates ONNX inference still failed despite KV cache fix");
                return Err("ONNX inference still failing".into());
            } else {
                println!("🎉 SUCCESS: This appears to be a real LLM-generated response!");
                println!("🎉 KV cache fix worked - ONNX inference is now functioning!");
            }
        }
        Err(e) => {
            println!("❌ Text generation failed: {}", e);
            println!("🔍 Error details: {:?}", e);
            return Err(e.into());
        }
    }

    println!("");
    println!("🏁 KV cache fix test completed successfully!");

    Ok(())
}
