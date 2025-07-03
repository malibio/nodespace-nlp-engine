//! Test RAG context processing fix
//! This example tests the specific marketing team resources question that was failing

use std::env;
use tokio;
use tracing::{info, warn, Level};
use tracing_subscriber;

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🧪 Testing RAG context processing fix...");

    // Create and initialize the NLP engine
    let mut engine = LocalNLPEngine::new();
    engine.initialize().await?;

    // Test the exact prompt format from the issue
    let test_prompt = r#"You are a helpful AI assistant. Please read the provided context carefully and answer the question based on the information given.

CONTEXT:
How much of the marketing team's resources would we need to support the Product Launch

**Marketing Technology and Tools**: 10,000

**Influencer and Partnership Marketing**: 35,000

**Total Budget**: $180,000

**Content Creation and Production**: 45,000

**Public Relations and Events**: 25,000

**Campaign Management**: 40% of marketing team capacity for 12 weeks

QUESTION: How much of the marketing team's resources would we need to support the Product Launch

INSTRUCTIONS:
1. Read the provided context carefully
2. If the context contains information that directly answers the question, extract and use that specific information
3. Base your response primarily on the provided context
4. If the context doesn't contain relevant information, acknowledge this clearly
5. Be specific and direct in your answer
6. Quote specific details from the context when relevant

Please provide a clear, specific answer based on the context above:"#;

    info!("📝 Testing with marketing team resources question...");
    info!("Expected: Should extract '40% of marketing team capacity for 12 weeks'");

    match engine.generate_text(test_prompt).await {
        Ok(response) => {
            info!("✅ LLM Response received:");
            println!("\n🤖 Response: {}\n", response);

            // Check if the response contains the expected information
            if response.contains("40%") && response.contains("marketing team") && response.contains("12 weeks") {
                info!("✅ SUCCESS: Response correctly extracted marketing team capacity information!");
            } else if response.contains("Based on the provided context, I can see relevant information about the topic, but the ONNX model needs proper configuration") {
                warn!("❌ STILL FAILING: Getting the old generic error message");
            } else {
                warn!("⚠️  PARTIAL: Response generated but didn't extract specific information:");
                println!("   Expected: '40% of marketing team capacity for 12 weeks'");
                println!("   Got: {}", response);
            }
        }
        Err(e) => {
            warn!("❌ Text generation failed: {}", e);
            return Err(e.into());
        }
    }

    // Test a simpler case to verify basic functionality
    info!("\n📝 Testing simpler RAG context...");
    let simple_prompt = r#"CONTEXT:
**Campaign Management**: 40% of marketing team capacity for 12 weeks

QUESTION: How much of the marketing team's resources would we need?"#;

    match engine.generate_text(simple_prompt).await {
        Ok(response) => {
            info!("✅ Simple test response:");
            println!("🤖 Response: {}\n", response);
        }
        Err(e) => {
            warn!("❌ Simple test failed: {}", e);
        }
    }

    info!("🎯 Test completed");
    Ok(())
}
