//! Regenerate embeddings for the sample database using fastembed 4.9
//! This example demonstrates using the NLP engine to generate real embeddings
//! for the existing sample database created by the data-store repo.

// Removed unused imports for this example
use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("NodeSpace Sample Database Embedding Regeneration");
    println!("Using fastembed 4.9 with BAAI/bge-small-en-v1.5 model");
    println!("============================================================");

    // Initialize the NLP engine with real ML
    println!("\nInitializing NLP engine with fastembed...");
    let nlp_engine = create_nlp_engine().await?;

    // Simulate sample database content (in real usage, this would come from data-store)
    println!("\nLoading sample database content...");
    let sample_content = create_sample_content();
    println!("Found {} content items to embed", sample_content.len());

    // Generate embeddings for all content
    println!("\nGenerating embeddings with fastembed...");
    let embeddings = generate_embeddings_for_content(&nlp_engine, &sample_content).await?;

    // Display results
    println!("\nEmbedding Generation Results:");
    println!("======================================");
    display_embedding_results(&sample_content, &embeddings).await?;

    // Demonstrate semantic search
    println!("\nSemantic Search Demonstration:");
    println!("=======================================");
    demonstrate_semantic_search(&nlp_engine, &sample_content, &embeddings).await?;

    println!("\nSample database embedding regeneration completed!");
    println!("Integration status: Ready for data-store connection");

    Ok(())
}

async fn create_nlp_engine() -> Result<LocalNLPEngine, Box<dyn std::error::Error>> {
    let engine = LocalNLPEngine::new();
    engine.initialize().await?;

    println!("NLP engine initialized successfully");
    println!("   Model: BAAI/bge-small-en-v1.5");
    println!("   Dimensions: 384");
    println!("   Device: Auto-selected");

    Ok(engine)
}

fn create_sample_content() -> Vec<SampleContentItem> {
    vec![
        SampleContentItem {
            id: "marketing_strategy_001".to_string(),
            content: "Develop comprehensive marketing campaign strategy for Q4 product launch focusing on digital channels and customer engagement".to_string(),
            category: "Strategy".to_string(),
        },
        SampleContentItem {
            id: "user_feedback_002".to_string(),
            content: "Customer feedback indicates strong demand for mobile app features including offline sync and dark mode support".to_string(),
            category: "Feedback".to_string(),
        },
        SampleContentItem {
            id: "tech_review_003".to_string(),
            content: "Technical review of authentication module shows excellent security practices but performance optimization needed for high-load scenarios".to_string(),
            category: "Technical".to_string(),
        },
        SampleContentItem {
            id: "meeting_notes_004".to_string(),
            content: "Weekly team meeting covered sprint retrospective, planning for next iteration, and discussion of API architecture improvements".to_string(),
            category: "Meeting".to_string(),
        },
        SampleContentItem {
            id: "product_roadmap_005".to_string(),
            content: "Product roadmap for 2025 includes AI integration, enhanced analytics dashboard, and expansion to European markets".to_string(),
            category: "Planning".to_string(),
        },
        SampleContentItem {
            id: "bug_report_006".to_string(),
            content: "Critical bug in payment processing system causing transaction failures during peak traffic hours needs immediate attention".to_string(),
            category: "Issue".to_string(),
        },
        SampleContentItem {
            id: "design_spec_007".to_string(),
            content: "UI design specification for new onboarding flow emphasizes simplicity and accessibility compliance across all user touchpoints".to_string(),
            category: "Design".to_string(),
        },
        SampleContentItem {
            id: "data_analysis_008".to_string(),
            content: "Data analysis reveals user engagement patterns with highest activity during evening hours and weekend peak usage periods".to_string(),
            category: "Analytics".to_string(),
        },
        SampleContentItem {
            id: "security_audit_009".to_string(),
            content: "Security audit completed with recommendations for enhanced encryption, access controls, and vulnerability scanning automation".to_string(),
            category: "Security".to_string(),
        },
        SampleContentItem {
            id: "customer_success_010".to_string(),
            content: "Customer success team reports high satisfaction scores and identifies training opportunities for improved feature adoption".to_string(),
            category: "Success".to_string(),
        },
    ]
}

async fn generate_embeddings_for_content(
    engine: &LocalNLPEngine,
    content_items: &[SampleContentItem],
) -> Result<HashMap<String, Vec<f32>>, Box<dyn std::error::Error>> {
    let mut embeddings = HashMap::new();
    let texts: Vec<String> = content_items
        .iter()
        .map(|item| item.content.clone())
        .collect();

    println!("Generating embeddings in batch...");
    let start_time = std::time::Instant::now();

    // Use batch processing for efficiency
    let batch_embeddings = engine.batch_embeddings(&texts).await?;

    let elapsed = start_time.elapsed();
    println!(
        "Generated {} embeddings in {:.2}s",
        batch_embeddings.len(),
        elapsed.as_secs_f64()
    );

    // Map embeddings back to content IDs
    for (item, embedding) in content_items.iter().zip(batch_embeddings.iter()) {
        embeddings.insert(item.id.clone(), embedding.clone());
    }

    Ok(embeddings)
}

async fn display_embedding_results(
    content_items: &[SampleContentItem],
    embeddings: &HashMap<String, Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    for item in content_items {
        if let Some(embedding) = embeddings.get(&item.id) {
            let preview = if item.content.len() > 60 {
                format!("{}...", &item.content[..57])
            } else {
                item.content.clone()
            };

            println!("{}: {} dimensions", item.id, embedding.len());
            println!("  Category: {}", item.category);
            println!("  Content: {}", preview);
            println!(
                "  Sample values: [{:.3}, {:.3}, {:.3}, ...]",
                embedding[0], embedding[1], embedding[2]
            );
            println!();
        }
    }

    Ok(())
}

async fn demonstrate_semantic_search(
    engine: &LocalNLPEngine,
    content_items: &[SampleContentItem],
    embeddings: &HashMap<String, Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let search_queries = vec![
        "product planning and roadmap",
        "security and authentication",
        "customer feedback analysis",
        "team meeting and collaboration",
    ];

    for query in search_queries {
        println!("\nQuery: \"{}\"", query);

        // Generate embedding for the query
        let query_embedding = engine.generate_embedding(query).await?;

        // Calculate similarities
        let mut similarities = Vec::new();
        for item in content_items {
            if let Some(item_embedding) = embeddings.get(&item.id) {
                let similarity = cosine_similarity(&query_embedding, item_embedding);
                similarities.push((item, similarity));
            }
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Display top 3 results
        println!("   Top matches:");
        for (i, (item, score)) in similarities.iter().take(3).enumerate() {
            let preview = if item.content.len() > 80 {
                format!("{}...", &item.content[..77])
            } else {
                item.content.clone()
            };
            println!(
                "   {}. {:.3} - {} ({})",
                i + 1,
                score,
                preview,
                item.category
            );
        }
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[derive(Debug, Clone)]
struct SampleContentItem {
    id: String,
    content: String,
    category: String,
}
