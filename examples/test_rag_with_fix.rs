use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine, RAGContext, TextGenerationRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 Testing RAG with Fixed ONNX Model");
    println!("====================================");

    let engine = LocalNLPEngine::new();

    // Simulate a real RAG context
    let rag_context = RAGContext {
        query: "What income range segment are we targeting for the Product Launch strategy?"
            .to_string(),
        retrieved_snippets: vec![
            "Income: 75,000-150,000 annually".to_string(),
            "Target customers with professional jobs".to_string(),
            "Mid-to-upper income bracket for our premium products".to_string(),
        ],
        source_documents: vec!["strategy_doc.md".to_string()],
        confidence_score: 0.85,
    };

    // Create a proper RAG prompt
    let prompt = format!(
        "You are a helpful AI assistant. Please read the provided context carefully and answer the question based on the information given.

CONTEXT:
{}

QUESTION: {}

Please provide a clear, specific answer based on the context above:",
        rag_context.retrieved_snippets.join("\n\n"),
        rag_context.query
    );

    println!("🔤 Testing with RAG prompt:");
    println!("📝 Prompt: {}", prompt);
    println!();

    // Test with different sampling parameters to avoid repetition
    let test_configs = vec![
        ("Low temperature, high diversity", 0.3, 0.9),
        ("Medium temperature, balanced", 0.7, 0.95),
        ("High temperature, creative", 1.0, 0.98),
    ];

    for (name, temperature, top_p) in test_configs {
        println!(
            "🔧 Testing: {} (temp={}, top_p={})",
            name, temperature, top_p
        );

        let request = TextGenerationRequest {
            prompt: prompt.clone(),
            max_tokens: Some(50), // Shorter to avoid repetition loops
            temperature: Some(temperature),
            top_p: Some(top_p),
            stop_sequences: Some(vec!["</answer>".to_string(), "\n\n".to_string()]),
        };

        match engine.generate_text(request).await {
            Ok(response) => {
                println!("✅ Response: '{}'", response.generated_text);
                println!("📏 Length: {} characters", response.generated_text.len());

                // Check if it's a real response
                if response.generated_text.len() > 10
                    && !response.generated_text.contains("hal hal hal")
                    && !response.generated_text.chars().any(|c| c as u32 > 127)
                {
                    println!("🎉 SUCCESS: Got a real response!");
                } else {
                    println!("⚠️  Still getting repetitive/invalid output");
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
        println!();
    }

    println!("🏁 RAG testing completed!");
    Ok(())
}
