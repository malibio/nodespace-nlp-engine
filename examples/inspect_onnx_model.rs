//! Inspect ONNX model structure and input/output signatures

use std::path::PathBuf;

#[cfg(feature = "real-ml")]
use ort::{session::builder::SessionBuilder, value::ValueType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber to see debug logs
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    #[cfg(feature = "real-ml")]
    {
        let model_path = PathBuf::from("../models/gemma-3-1b-it-onnx/model.onnx");

        println!("🔍 Inspecting ONNX model at: {}", model_path.display());

        if !model_path.exists() {
            println!("❌ Model file not found at: {}", model_path.display());
            return Ok(());
        }

        // Load the ONNX session
        let session = SessionBuilder::new()?.commit_from_file(&model_path)?;

        // Get model metadata
        let metadata = session.metadata()?;
        println!("📊 Model metadata:");
        println!("  Producer: {:?}", metadata.producer());
        println!("  Version: {:?}", metadata.version());
        println!("  Description: {:?}", metadata.description());

        // Get input information
        println!("\n📥 Model inputs:");
        for (i, input) in session.inputs.iter().enumerate() {
            println!("  Input {}: {}", i, input.name);
            println!("    Type: {:?}", input.input_type);

            // Try to get shape information from the type
            match &input.input_type {
                ValueType::Tensor { ty, dimensions, .. } => {
                    println!("    Tensor type: {:?}", ty);
                    println!("    Shape: {:?}", dimensions);
                }
                _ => {
                    println!("    Type: {:?}", input.input_type);
                }
            }
        }

        // Get output information
        println!("\n📤 Model outputs:");
        for (i, output) in session.outputs.iter().enumerate() {
            println!("  Output {}: {}", i, output.name);
            println!("    Type: {:?}", output.output_type);

            // Try to get shape information from the type
            match &output.output_type {
                ValueType::Tensor { ty, dimensions, .. } => {
                    println!("    Tensor type: {:?}", ty);
                    println!("    Shape: {:?}", dimensions);
                }
                _ => {
                    println!("    Type: {:?}", output.output_type);
                }
            }
        }

        // Try to get more detailed information
        println!("\n🔧 Model session info:");
        println!("  Number of inputs: {}", session.inputs.len());
        println!("  Number of outputs: {}", session.outputs.len());

        // Check if we can determine the expected input format
        println!("\n🎯 Analysis:");

        // Look for common input patterns
        let input_names: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
        println!("  Input names: {:?}", input_names);

        let output_names: Vec<&str> = session.outputs.iter().map(|o| o.name.as_str()).collect();
        println!("  Output names: {:?}", output_names);

        // Check for KV cache patterns
        let has_past_key_values = input_names.iter().any(|name| {
            name.contains("past_key_values") || name.contains("past_key") || name.contains("cache")
        });

        let has_attention_mask = input_names
            .iter()
            .any(|name| name.contains("attention_mask"));

        println!("  Has past key values: {}", has_past_key_values);
        println!("  Has attention mask: {}", has_attention_mask);

        // Analyze each input in detail
        for input in &session.inputs {
            if input.name == "input_ids" {
                println!("  Found input_ids input");
            }
        }

        // Analyze each output in detail
        for output in &session.outputs {
            if output.name == "logits" || output.name.contains("logits") {
                println!("  Found logits output");
            }
        }
    }

    #[cfg(not(feature = "real-ml"))]
    {
        println!("❌ This example requires the 'real-ml' feature to be enabled");
        println!("Run with: cargo run --example inspect_onnx_model --features real-ml");
    }

    Ok(())
}
