//! Multimodal capabilities demonstration
//!
//! This example demonstrates the image processing and multimodal capabilities
//! of the NodeSpace NLP Engine, including:
//! - Image embedding generation using CLIP
//! - EXIF metadata extraction from images
//! - Multimodal response generation

use nodespace_nlp_engine::{LocalNLPEngine, NLPEngine};

#[cfg(feature = "multimodal")]
use nodespace_nlp_engine::{
    ImageInput, MultimodalRequest, ImageMetadata
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better logging
    tracing_subscriber::fmt::init();

    println!("🎯 NodeSpace Multimodal Capabilities Demo");
    println!("==========================================\n");

    // Create and initialize the NLP engine
    let engine = LocalNLPEngine::new();
    println!("📋 Initializing NLP Engine...");
    
    match engine.initialize().await {
        Ok(()) => println!("✅ Engine initialized successfully!"),
        Err(e) => {
            println!("⚠️  Engine initialization had issues: {}", e);
            println!("📝 Note: This is expected if CLIP models aren't available offline");
            println!("🔄 Continuing with available capabilities...\n");
        }
    }

    // Test basic text capabilities first
    println!("🔤 Testing Text Capabilities");
    println!("----------------------------");
    
    let text = "This is a test sentence for embedding generation.";
    match engine.generate_embedding(text).await {
        Ok(embedding) => {
            println!("✅ Text embedding generated successfully");
            println!("📊 Embedding dimensions: {}", embedding.len());
            println!("🎯 First 5 values: {:?}\n", &embedding[..5.min(embedding.len())]);
        }
        Err(e) => println!("❌ Text embedding failed: {}\n", e),
    }

    // Test image capabilities if multimodal feature is enabled
    #[cfg(feature = "multimodal")]
    {
        println!("🖼️  Testing Multimodal Capabilities");
        println!("----------------------------------");

        // Create a simple test image (1x1 PNG)
        let test_image_data = create_test_image();
        
        // Test image metadata extraction
        println!("📋 Extracting image metadata...");
        match engine.extract_image_metadata(&test_image_data).await {
            Ok(metadata) => {
                println!("✅ Image metadata extracted successfully");
                display_image_metadata(&metadata);
            }
            Err(e) => println!("❌ Image metadata extraction failed: {}", e),
        }

        // Test image embedding generation
        println!("\n🔢 Generating image embeddings...");
        match engine.generate_image_embedding(&test_image_data).await {
            Ok(embedding) => {
                println!("✅ Image embedding generated successfully");
                println!("📊 Embedding dimensions: {}", embedding.len());
                println!("🎯 First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
            }
            Err(e) => {
                println!("❌ Image embedding failed: {}", e);
                println!("📝 Note: This is expected if CLIP models aren't available");
            }
        }

        // Test multimodal response generation
        println!("\n🤖 Testing multimodal response generation...");
        let multimodal_request = MultimodalRequest {
            text_query: "What can you tell me about this image?".to_string(),
            images: vec![ImageInput {
                data: test_image_data,
                description: Some("Test image".to_string()),
                id: Some("test_img_1".to_string()),
            }],
            context_nodes: vec![],
            enable_smart_links: false,
            max_tokens: 100,
            temperature: 0.7,
        };

        match engine.generate_multimodal_response(multimodal_request).await {
            Ok(response) => {
                println!("✅ Multimodal response generated successfully");
                println!("💬 Response: {}", response.text);
                println!("🖼️  Images used: {}", response.image_utilization.images_used);
                println!("🔗 Smart links: {}", response.smart_links.len());
            }
            Err(e) => {
                println!("❌ Multimodal response failed: {}", e);
                println!("📝 Note: This is expected if models aren't fully available");
            }
        }
    }

    #[cfg(not(feature = "multimodal"))]
    {
        println!("🔒 Multimodal features disabled");
        println!("💡 Run with: cargo run --example multimodal_demo --features multimodal");
    }

    // Display engine status
    println!("\n📊 Engine Status");
    println!("----------------");
    let status = engine.status().await;
    println!("🔧 Initialized: {}", status.initialized);
    println!("💻 Device: {:?}", status.device_type);
    
    if let Some(embedding_info) = status.embedding_info {
        println!("📐 Text embeddings: {} dimensions", embedding_info.dimensions);
        println!("🗂️  Cache: {} items", embedding_info.cache_stats.0);
    }

    if let Some(text_info) = status.text_generation_info {
        println!("🤖 Text model: {}", text_info.model_name);
    }

    println!("\n🎉 Demo completed successfully!");
    Ok(())
}

#[cfg(feature = "multimodal")]
fn display_image_metadata(metadata: &ImageMetadata) {
    println!("📏 Dimensions: {}x{}", metadata.dimensions.0, metadata.dimensions.1);
    println!("📁 Format: {}", metadata.format);
    println!("💾 File size: {} bytes", metadata.file_size);
    
    if let Some(timestamp) = &metadata.timestamp {
        println!("📅 Timestamp: {}", timestamp);
    }
    
    if let Some((lat, lon)) = metadata.gps_coordinates {
        println!("📍 GPS: {:.6}, {:.6}", lat, lon);
    }
    
    if let Some(camera_info) = &metadata.camera_info {
        if let Some(make) = &camera_info.make {
            println!("📸 Camera: {}", make);
        }
        if let Some(model) = &camera_info.model {
            println!("📷 Model: {}", model);
        }
    }
    
    if let Some(color_space) = &metadata.color_space {
        println!("🎨 Color space: {}", color_space);
    }
    
    println!("⏱️  Processing time: {} ms", metadata.processing_time_ms);
}

/// Create a minimal test image (1x1 pixel PNG)
#[cfg(feature = "multimodal")]
fn create_test_image() -> Vec<u8> {
    // Minimal PNG file data (1x1 red pixel)
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // bit depth, color type, etc.
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x00, 0x00, // compressed image data
        0x00, 0xFF, 0xFF, 0x03, 0x00, 0x00, 0x06, 0x00, // (red pixel)
        0x05, 0x02, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x49, // IEND chunk
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ]
}