//! Example demonstrating text generation and SurrealQL generation

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 NodeSpace NLP Engine - Text Generation Example");

    // Create and initialize the NLP engine
    let engine = LocalNLPEngine::new();
    println!("📊 Initializing NLP engine...");
    engine.initialize().await?;

    // Check engine status
    let status = engine.status().await;
    println!("✅ Engine initialized successfully!");
    if let Some(text_info) = status.text_generation_info {
        println!("   Model: {}", text_info.model_name);
        println!("   Max context: {} tokens", text_info.max_context_length);
        println!("   Device: {:?}", text_info.device_type);
    }

    // Example 1: Basic text generation
    println!("\n💬 Example 1: Basic Text Generation");
    let prompt = "Write a brief summary of what makes a good team meeting:";
    println!("Prompt: \"{}\"", prompt);

    let start_time = std::time::Instant::now();
    let generated_text = engine.generate_text(prompt).await?;
    let duration = start_time.elapsed();

    println!("Generated text ({:?}):", duration);
    println!("\"{}\"", generated_text);

    // Example 2: SurrealQL generation
    println!("\n🗄️  Example 2: SurrealQL Generation");

    let schema_context = r#"
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

    let natural_queries = vec![
        "Find all meetings from last week",
        "Get tasks assigned to John that are high priority",
        "Show meetings with more than 5 participants",
        "Find incomplete tasks due this week",
        "Get all meetings about planning or roadmap",
    ];

    for (i, query) in natural_queries.iter().enumerate() {
        println!("\nQuery {}: \"{}\"", i + 1, query);

        let start_time = std::time::Instant::now();
        let surrealql = engine.generate_surrealql(query, schema_context).await?;
        let duration = start_time.elapsed();

        println!("Generated SurrealQL ({:?}):", duration);
        println!("```sql\n{}\n```", surrealql);
    }

    // Example 3: Entity analysis
    println!("\n🏷️  Example 3: Entity Analysis (Advanced)");

    let entity_texts = vec![
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
    println!("\n🔍 Example 4: Complex Query Generation");

    let complex_queries = vec![
        "Find all meetings from the last month where John was a participant and the status is completed",
        "Get high priority tasks that are overdue and assigned to members of the engineering team",
        "Show me meetings and related tasks for the Q4 planning project",
    ];

    for (i, query) in complex_queries.iter().enumerate() {
        println!("\nComplex Query {}: \"{}\"", i + 1, query);

        let surrealql = engine.generate_surrealql(query, schema_context).await?;
        println!("Generated SurrealQL:");
        println!("```sql\n{}\n```", surrealql);
    }

    println!("\n🎉 Text generation examples completed successfully!");

    Ok(())
}
