//! RAG Integration Example
//!
//! Demonstrates how to use the NLP engine for RAG (Retrieval-Augmented Generation) operations.
//! This example shows the pattern that nodespace-core-logic will use for orchestrating
//! semantic search + AI generation workflows.

use nodespace_nlp_engine::{
    LocalNLPEngine, NLPEngine, RAGContext, TextGenerationRequest,
    TokenBudget, estimate_token_count, allocate_budget_to_segments
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better debugging
    tracing_subscriber::fmt::init();

    println!("🚀 NodeSpace RAG Integration Example");
    println!("=====================================\n");

    // Step 1: Initialize NLP Engine
    println!("1️⃣ Initializing NLP Engine...");
    let nlp_engine = LocalNLPEngine::new();
    nlp_engine.initialize().await?;
    println!("   ✅ NLP Engine initialized\n");

    // Step 2: Simulate retrieved knowledge from data store
    println!("2️⃣ Simulating Knowledge Retrieval...");
    let retrieved_knowledge = vec![
        ("NodeSpace is a distributed system for knowledge management with AI integration.", 0.95),
        ("The system uses SurrealDB for data storage and local LLMs for AI processing.", 0.88),
        ("RAG (Retrieval-Augmented Generation) combines search with AI generation.", 0.82),
    ];
    
    for (text, score) in &retrieved_knowledge {
        println!("   📄 Knowledge (score: {:.2}): {}", score, text);
    }
    println!();

    // Step 3: Token Budget Management
    println!("3️⃣ Token Budget Management...");
    let context_window = 8192; // Typical model context window
    let mut budget = TokenBudget::new(context_window);
    
    // Estimate tokens for retrieved knowledge
    let knowledge_texts: Vec<String> = retrieved_knowledge.iter()
        .map(|(text, _)| text.to_string())
        .collect();
    
    let total_knowledge_tokens: usize = knowledge_texts.iter()
        .map(|text| estimate_token_count(text))
        .sum();
    
    budget.set_knowledge_tokens(total_knowledge_tokens);
    
    println!("   📊 Context window: {} tokens", context_window);
    println!("   📊 Knowledge content: {} tokens", total_knowledge_tokens);
    println!("   📊 Available for context: {} tokens", budget.available_for_context());
    println!("   📊 Within budget: {}\n", budget.is_within_budget());

    // Step 4: Smart content allocation if over budget
    println!("4️⃣ Smart Content Allocation...");
    let max_knowledge_budget = budget.available_for_context() / 2; // Reserve half for conversation
    
    let prioritized_segments: Vec<(String, f32)> = retrieved_knowledge.iter()
        .map(|(text, score)| (text.to_string(), *score))
        .collect();
    
    let allocated_content = allocate_budget_to_segments(
        prioritized_segments,
        max_knowledge_budget
    )?;
    
    println!("   🎯 Allocated {} knowledge segments within budget", allocated_content.len());
    for (i, content) in allocated_content.iter().enumerate() {
        println!("   📝 Segment {}: {} chars", i + 1, content.len());
    }
    println!();

    // Step 5: Assemble RAG Context
    println!("5️⃣ Assembling RAG Context...");
    let rag_context = RAGContext {
        knowledge_sources: allocated_content,
        retrieval_confidence: 0.88,
        context_summary: "NodeSpace distributed knowledge management system with AI capabilities".to_string(),
    };
    
    println!("   🧠 Context summary: {}", rag_context.context_summary);
    println!("   🎯 Retrieval confidence: {:.2}", rag_context.retrieval_confidence);
    println!("   📚 Knowledge sources: {}\n", rag_context.knowledge_sources.len());

    // Step 6: Enhanced Text Generation with RAG
    println!("6️⃣ Enhanced RAG Generation...");
    let user_query = "How does NodeSpace use AI for knowledge management?";
    
    let request = TextGenerationRequest {
        prompt: user_query.to_string(),
        max_tokens: 200,
        temperature: 0.7,
        context_window,
        conversation_mode: true,
        rag_context: Some(rag_context),
    };
    
    println!("   ❓ User query: {}", user_query);
    println!("   ⚙️ Generation params: {} max tokens, {:.1} temperature", 
             request.max_tokens, request.temperature);
    
    let response = nlp_engine.generate_text_enhanced(request).await?;
    
    println!("\n   🤖 AI Response:");
    println!("   {}\n", response.text);
    
    // Step 7: Analysis and Metrics
    println!("7️⃣ Response Analysis...");
    println!("   ⏱️  Generation time: {}ms", response.generation_metrics.generation_time_ms);
    println!("   📊 Context tokens: {}", response.generation_metrics.context_tokens);
    println!("   📊 Response tokens: {}", response.generation_metrics.response_tokens);
    println!("   🎯 Context referenced: {}", response.context_utilization.context_referenced);
    println!("   📚 Sources mentioned: {:?}", response.context_utilization.sources_mentioned);
    println!("   📈 Relevance score: {:.2}", response.context_utilization.relevance_score);

    println!("\n✨ RAG Integration Example Complete!");
    println!("   This pattern can be used by nodespace-core-logic for:");
    println!("   • AIChatNode query processing");
    println!("   • Semantic search + generation workflows");  
    println!("   • Context-aware AI responses");
    println!("   • Token budget management");

    Ok(())
}