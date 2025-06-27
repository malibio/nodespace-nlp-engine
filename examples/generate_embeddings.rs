//! Example demonstrating embedding generation

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Skip tracing initialization when using real-ml feature
    #[cfg(not(feature = "real-ml"))]
    tracing_subscriber::fmt::init();

    println!("🚀 NodeSpace NLP Engine - Embedding Generation Example");

    // Create and initialize the NLP engine
    let engine = LocalNLPEngine::new();
    println!("📊 Initializing NLP engine...");
    engine.initialize().await?;

    // Check engine status
    let status = engine.status().await;
    println!("✅ Engine initialized successfully!");
    println!("   Device: {:?}", status.device_type);
    if let Some(embedding_info) = status.embedding_info {
        println!("   Model: {}", embedding_info.model_name);
        println!("   Dimensions: {}", embedding_info.dimensions);
    }

    // Example texts
    let texts = [
        "Meeting about Q3 planning with the engineering team",
        "Customer feedback review session",
        "Sprint retrospective for the last two weeks",
        "Product roadmap discussion for next quarter",
        "Code review for the authentication module",
    ];

    println!("\n🔍 Generating embeddings for example texts...");

    // Generate individual embeddings
    for (i, text) in texts.iter().enumerate() {
        println!("\nText {}: \"{}\"", i + 1, text);

        let start_time = std::time::Instant::now();
        let embedding = engine.generate_embedding(text).await?;
        let duration = start_time.elapsed();

        println!(
            "   ✓ Generated embedding with {} dimensions in {:?}",
            embedding.len(),
            duration
        );
        println!(
            "   First 5 values: {:?}",
            &embedding[..5.min(embedding.len())]
        );
    }

    // Demonstrate batch processing
    println!("\n📦 Demonstrating batch embedding generation...");

    let start_time = std::time::Instant::now();
    let batch_embeddings = engine
        .batch_embeddings(&texts.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .await?;
    let batch_duration = start_time.elapsed();

    println!(
        "   ✓ Generated {} embeddings in batch in {:?}",
        batch_embeddings.len(),
        batch_duration
    );

    // Calculate similarities between texts
    println!("\n🔗 Calculating similarities between texts...");

    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            let similarity = cosine_similarity(&batch_embeddings[i], &batch_embeddings[j]);
            println!("   Text {} <-> Text {}: {:.3}", i + 1, j + 1, similarity);
        }
    }

    // Demonstrate semantic search scenario
    println!("\n🎯 Semantic search demonstration...");

    let query = "team meeting planning";
    let query_embedding = engine.generate_embedding(query).await?;

    println!("Query: \"{}\"", query);
    println!("Most similar texts:");

    let mut similarities: Vec<(usize, f32)> = batch_embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (text_idx, similarity)) in similarities.iter().enumerate() {
        println!(
            "   {}. \"{}\" (similarity: {:.3})",
            i + 1,
            texts[*text_idx],
            similarity
        );
    }

    // Show cache statistics
    let cache_stats = engine.cache_stats().await;
    println!("\n📈 Cache Statistics:");
    println!(
        "   Cached embeddings: {}/{}",
        cache_stats.embedding_cache_size, cache_stats.embedding_cache_capacity
    );

    println!("\n🎉 Example completed successfully!");

    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
