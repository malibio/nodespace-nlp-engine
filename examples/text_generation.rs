//! Example demonstrating text generation and SurrealQL generation

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Skip tracing initialization - mistralrs handles this

    println!("NodeSpace NLP Engine - Text Generation Example");

    // Create and initialize the NLP engine
    let engine = LocalNLPEngine::new();
    println!("Initializing NLP engine...");
    engine.initialize().await?;

    // Check engine status
    let status = engine.status().await;
    println!("Engine initialized successfully!");
    if let Some(text_info) = status.text_generation_info {
        println!("   Model: {}", text_info.model_name);
        println!("   Max context: {} tokens", text_info.max_context_length);
        println!("   Device: {:?}", text_info.device_type);
    }

    // Example 1: Basic text generation
    println!("\nExample 1: Basic Text Generation");
    let prompt = "Write a brief summary of what makes a good team meeting:";
    println!("Prompt: \"{}\"", prompt);

    let start_time = std::time::Instant::now();
    let generated_text = engine.generate_text(prompt).await?;
    let duration = start_time.elapsed();

    println!("Generated text ({:?}):", duration);
    println!("\"{}\"", generated_text);

    // Example 2: Content Analysis and Intent Extraction
    println!("\nExample 2: Content Analysis and Intent Extraction");

    let _schema_context = r#"
    TABLE meeting {
        id: string,
        title: string,
        date: datetime,
        participants: array<string>,
        status: string,
        notes: string
    }
    
    TABLE task {
        id: string,
        title: string,
        description: string,
        assignee: string,
        due_date: datetime,
        priority: string,
        completed: bool
    }
    "#;

    let natural_queries = [
        "Find all meetings from last week",
        "Get tasks assigned to John that are high priority",
        "Show meetings with more than 5 participants",
        "Find incomplete tasks due this week",
        "Get all meetings about planning or roadmap",
    ];

    for (i, query) in natural_queries.iter().enumerate() {
        println!("\nQuery {}: \"{}\"", i + 1, query);

        let start_time = std::time::Instant::now();
        let analysis = engine.analyze_content(query, "query_intent").await?;
        let duration = start_time.elapsed();

        println!("Content Analysis ({:?}):", duration);
        println!("Classification: {}", analysis.classification);
        println!("Confidence: {:.2}", analysis.confidence);
        if !analysis.topics.is_empty() {
            println!("Topics: {:?}", analysis.topics);
        }
    }

    // Example 3: Advanced structured data extraction
    println!("\nExample 3: Advanced Structured Data Extraction");

    let entity_texts = [
        "Create a meeting about Q4 budget review with finance team for next Thursday at 2 PM",
        "Task: Implement user authentication with OAuth2 - high priority, due Friday",
        "Customer feedback session with Acme Corp about product roadmap",
        "Code review for the new payment processing module",
    ];

    for (i, text) in entity_texts.iter().enumerate() {
        println!("\nText {}: \"{}\"", i + 1, text);

        // Use the advanced text generation to analyze entity creation
        let analysis_prompt = format!(
            r#"Analyze this text and extract structured information:
"{}"

Extract:
- Entity type (Meeting, Task, Person, Document, etc.)
- Key fields and values
- Suggested tags
- Priority or urgency level

Provide a structured analysis:"#,
            text
        );

        let analysis = engine.generate_text(&analysis_prompt).await?;
        println!("Analysis:\n{}", analysis);
    }

    // Example 4: Natural language to structured query
    println!("\nExample 4: Complex Query Generation");

    let complex_queries = [
        "Find all meetings from the last month where John was a participant and the status is completed",
        "Get high priority tasks that are overdue and assigned to members of the engineering team",
        "Show me meetings and related tasks for the Q4 planning project",
    ];

    for (i, query) in complex_queries.iter().enumerate() {
        println!("\nComplex Query {}: \"{}\"", i + 1, query);

        let structured_data = engine
            .extract_structured_data(query, "search_request")
            .await?;
        println!("Extracted Structured Data:");
        println!(
            "```json\n{}\n```",
            serde_json::to_string_pretty(&structured_data)?
        );

        let summary = engine.generate_summary(query, Some(15)).await?;
        println!("Query Summary: {}", summary);
    }

    println!("\nText generation examples completed successfully!");

    Ok(())
}
