//! ONNX Runtime Diagnostic Example
//!
//! This example systematically tests ONNX model loading and inference to isolate
//! issues in the NodeSpace NLP engine text generation pipeline.
//!
//! It tests:
//! - Model loading and session creation
//! - CPU vs CoreML execution providers  
//! - Tensor input/output shapes
//! - Forward pass with various input combinations
//! - Error messages and debugging information
//!
//! Run with: cargo run --example test_onnx_inference_debug --features real-ml

use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider},
    inputs,
    session::{builder::SessionBuilder, Session},
    value::Value,
};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    println!("🔍 ONNX Runtime Diagnostic Tool");
    println!("================================");

    // Test 1: Check available execution providers
    println!("\n📋 Test 1: Available Execution Providers");
    test_execution_providers().await?;

    // Test 2: Locate and validate model files
    println!("\n📋 Test 2: Model File Validation");
    let model_path = find_model_files().await?;

    // Test 3: Load model with different execution providers
    println!("\n📋 Test 3: Model Loading Tests");
    test_model_loading(&model_path).await?;

    // Test 4: Analyze model signature
    println!("\n📋 Test 4: Model Signature Analysis");
    let session = load_model_cpu(&model_path).await?;
    analyze_model_signature(&session).await?;

    // Test 5: Test tensor creation and basic inference
    println!("\n📋 Test 5: Tensor Creation and Basic Inference");
    test_tensor_operations(&session).await?;

    // Test 6: Test different input combinations
    println!("\n📋 Test 6: Input Combination Tests");
    test_input_combinations(&session).await?;

    // Test 7: Test with realistic text input
    println!("\n📋 Test 7: Realistic Text Input Test");
    test_realistic_inference(&session).await?;

    println!("\n✅ Diagnostic tests completed!");
    println!("Check the logs above for detailed error messages and debugging information.");

    Ok(())
}

async fn test_execution_providers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Available execution providers:");

    // Test CoreML (Apple MPS)
    match SessionBuilder::new() {
        Ok(builder) => {
            match builder.with_execution_providers([CoreMLExecutionProvider::default().build()]) {
                Ok(_) => println!("✅ CoreML execution provider available"),
                Err(e) => println!("❌ CoreML execution provider failed: {}", e),
            }
        }
        Err(e) => println!("❌ Failed to create session builder: {}", e),
    }

    // Test CUDA
    match SessionBuilder::new() {
        Ok(builder) => {
            match builder.with_execution_providers([CUDAExecutionProvider::default().build()]) {
                Ok(_) => println!("✅ CUDA execution provider available"),
                Err(e) => println!("❌ CUDA execution provider failed: {}", e),
            }
        }
        Err(e) => println!("❌ Failed to create session builder: {}", e),
    }

    // Test CPU (baseline)
    match SessionBuilder::new() {
        Ok(builder) => {
            println!("✅ CPU execution provider available");
            // Test creating a session without any model
            drop(builder);
        }
        Err(e) => println!("❌ CPU execution provider failed: {}", e),
    }

    Ok(())
}

async fn find_model_files() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Look for model files in common locations
    let possible_paths = [
        "/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/model.onnx",
        "./models/gemma-3-1b-it-onnx/model.onnx",
        "../models/gemma-3-1b-it-onnx/model.onnx",
        "models/gemma-3-1b-it-onnx/model.onnx",
    ];

    for path_str in &possible_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            println!("✅ Found model at: {}", path.display());

            // Check file size
            let metadata = std::fs::metadata(&path)?;
            println!("   Model size: {} MB", metadata.len() / 1024 / 1024);

            // Check for tokenizer
            let tokenizer_path = path.parent().unwrap().join("tokenizer.json");
            if tokenizer_path.exists() {
                println!("✅ Found tokenizer at: {}", tokenizer_path.display());
            } else {
                println!("⚠️  Tokenizer not found at: {}", tokenizer_path.display());
            }

            return Ok(path);
        } else {
            println!("❌ Model not found at: {}", path.display());
        }
    }

    Err("No ONNX model file found in any of the expected locations".into())
}

async fn test_model_loading(model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing model loading with different execution providers...");

    // Test 1: CPU execution provider
    println!("\n🔄 Testing CPU execution provider...");
    match load_model_cpu(model_path).await {
        Ok(session) => {
            println!("✅ CPU loading successful");
            println!("   Input count: {}", session.inputs.len());
            println!("   Output count: {}", session.outputs.len());
        }
        Err(e) => {
            println!("❌ CPU loading failed: {}", e);
            return Err(e);
        }
    }

    // Test 2: CoreML execution provider
    println!("\n🔄 Testing CoreML execution provider...");
    match load_model_coreml(model_path).await {
        Ok(session) => {
            println!("✅ CoreML loading successful");
            println!("   Input count: {}", session.inputs.len());
            println!("   Output count: {}", session.outputs.len());
        }
        Err(e) => {
            println!("❌ CoreML loading failed: {}", e);
            println!("   This is expected if CoreML EP is not available");
        }
    }

    Ok(())
}

async fn load_model_cpu(model_path: &PathBuf) -> Result<Session, Box<dyn std::error::Error>> {
    println!("Loading model with CPU execution provider...");
    let session = SessionBuilder::new()?.commit_from_file(model_path)?;
    Ok(session)
}

async fn load_model_coreml(model_path: &PathBuf) -> Result<Session, Box<dyn std::error::Error>> {
    println!("Loading model with CoreML execution provider...");
    let session = SessionBuilder::new()?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file(model_path)?;
    Ok(session)
}

async fn analyze_model_signature(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing model signature...");

    println!("\n📥 Model Inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("  {}. Name: {}", i + 1, input.name);
        println!("     Type: {:?}", input.input_type);

        // Print element type information
        println!("     Element type: {:?}", input.input_type);
    }

    println!("\n📤 Model Outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("  {}. Name: {}", i + 1, output.name);
        println!("     Type: {:?}", output.output_type);
    }

    // Check for common input patterns
    let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.to_string()).collect();
    println!("\n🔍 Input Pattern Analysis:");

    if input_names.contains(&"input_ids".to_string()) {
        println!("✅ Standard transformer input 'input_ids' found");
    }
    if input_names.contains(&"attention_mask".to_string()) {
        println!("✅ Standard transformer input 'attention_mask' found");
    }
    if input_names.contains(&"position_ids".to_string()) {
        println!("✅ Standard transformer input 'position_ids' found");
    }

    // Check for KV cache inputs
    let kv_cache_inputs: Vec<_> = input_names
        .iter()
        .filter(|name| {
            name.contains("past_key_values") || name.contains("key") || name.contains("value")
        })
        .collect();

    if !kv_cache_inputs.is_empty() {
        println!("✅ KV cache inputs detected: {:?}", kv_cache_inputs);
    } else {
        println!("ℹ️  No KV cache inputs detected");
    }

    Ok(())
}

async fn test_tensor_operations(_session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tensor creation and operations...");

    // Test creating basic tensors
    println!("\n🔄 Creating test tensors...");

    // Create input_ids tensor
    let input_ids = vec![1i64, 2, 3, 4, 5]; // Simple token sequence
    let batch_size = 1;
    let seq_len = input_ids.len();

    match Value::from_array(([batch_size, seq_len], input_ids.clone())) {
        Ok(_tensor) => {
            println!("✅ input_ids tensor created successfully");
            println!("   Shape: [{}, {}]", batch_size, seq_len);
        }
        Err(e) => {
            println!("❌ input_ids tensor creation failed: {}", e);
            return Err(e.into());
        }
    }

    // Create attention_mask tensor
    let attention_mask = vec![1i64; seq_len]; // All ones
    match Value::from_array(([batch_size, seq_len], attention_mask.clone())) {
        Ok(_tensor) => {
            println!("✅ attention_mask tensor created successfully");
        }
        Err(e) => {
            println!("❌ attention_mask tensor creation failed: {}", e);
            return Err(e.into());
        }
    }

    // Create position_ids tensor
    let position_ids: Vec<i64> = (0..seq_len).map(|i| i as i64).collect();
    match Value::from_array(([batch_size, seq_len], position_ids.clone())) {
        Ok(_tensor) => {
            println!("✅ position_ids tensor created successfully");
        }
        Err(e) => {
            println!("❌ position_ids tensor creation failed: {}", e);
            return Err(e.into());
        }
    }

    println!("✅ All basic tensors created successfully");
    Ok(())
}

async fn test_input_combinations(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing different input combinations...");

    let input_ids = vec![1i64, 2, 3, 4, 5];
    let batch_size = 1;
    let seq_len = input_ids.len();
    let attention_mask = vec![1i64; seq_len];
    let position_ids: Vec<i64> = (0..seq_len).map(|i| i as i64).collect();

    // Test 1: Just input_ids
    println!("\n🔄 Test 1: input_ids only");
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids.clone()))?;
    let inputs = inputs! { "input_ids" => input_ids_tensor }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ input_ids only: SUCCESS");
            println!("   Outputs: {}", outputs.len());
        }
        Err(e) => println!("❌ input_ids only: FAILED - {}", e),
    }

    // Test 2: input_ids + attention_mask
    println!("\n🔄 Test 2: input_ids + attention_mask");
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids.clone()))?;
    let attention_mask_tensor = Value::from_array(([batch_size, seq_len], attention_mask.clone()))?;
    let inputs = inputs! {
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ input_ids + attention_mask: SUCCESS");
            println!("   Outputs: {}", outputs.len());
        }
        Err(e) => println!("❌ input_ids + attention_mask: FAILED - {}", e),
    }

    // Test 3: input_ids + attention_mask + position_ids
    println!("\n🔄 Test 3: input_ids + attention_mask + position_ids");
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids.clone()))?;
    let attention_mask_tensor = Value::from_array(([batch_size, seq_len], attention_mask.clone()))?;
    let position_ids_tensor = Value::from_array(([batch_size, seq_len], position_ids.clone()))?;
    let inputs = inputs! {
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
        "position_ids" => position_ids_tensor,
    }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ Full input set: SUCCESS");
            println!("   Outputs: {}", outputs.len());
        }
        Err(e) => println!("❌ Full input set: FAILED - {}", e),
    }

    Ok(())
}

async fn test_realistic_inference(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing with realistic text input...");

    // Simulate realistic token sequence (similar to what tokenizer would produce)
    let realistic_tokens = vec![
        1i64,  // BOS token
        7803,  // "Hello"
        1849,  // "world"
        29892, // ","
        278,   // "the"
        4720,  // "quick"
        17354, // "brown"
        1701,  // "fox"
        2,     // EOS token
    ];

    let batch_size = 1;
    let seq_len = realistic_tokens.len();
    let attention_mask = vec![1i64; seq_len];
    let _position_ids: Vec<i64> = (0..seq_len).map(|i| i as i64).collect();

    println!("🔄 Testing with realistic token sequence:");
    println!("   Token count: {}", seq_len);
    println!("   Tokens: {:?}", realistic_tokens);

    // Test the most likely working input combination
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], realistic_tokens.clone()))?;
    let attention_mask_tensor = Value::from_array(([batch_size, seq_len], attention_mask.clone()))?;
    let inputs = inputs! {
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ Realistic inference successful!");

            // Try to extract and analyze logits
            if let Ok(logits_tensor) = outputs["logits"].try_extract_tensor::<f32>() {
                let shape = logits_tensor.shape();
                println!("   🎯 Logits shape: {:?}", shape);

                if shape.len() >= 2 {
                    let vocab_size = shape[shape.len() - 1];
                    println!("   📚 Vocabulary size: {}", vocab_size);

                    // Extract last token's logits for next token prediction
                    if shape.len() == 3 {
                        // [batch_size, seq_len, vocab_size]
                        let last_pos = shape[1] - 1;
                        println!("   🔍 Extracting logits for position {}", last_pos);

                        // Get a slice of logits for the last token
                        let logits_slice = logits_tensor.slice(ndarray::s![0, last_pos, ..100]);
                        println!("   📊 First 100 logits: {:?}", logits_slice);

                        // Find the token with highest probability
                        let max_idx = logits_slice
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);

                        println!("   🎯 Predicted next token ID: {}", max_idx);
                    }
                }
            } else {
                println!("   ⚠️  Could not extract logits as f32 tensor");

                // Try to find any output that might be logits
                for (name, tensor) in outputs.iter() {
                    if let Ok(float_tensor) = tensor.try_extract_tensor::<f32>() {
                        let shape = float_tensor.shape();
                        println!("   🔍 Output '{}' shape: {:?}", name, shape);

                        if shape.len() >= 2 && shape[shape.len() - 1] > 1000 {
                            println!(
                                "   🎯 '{}' might be logits (vocab_size={})",
                                name,
                                shape[shape.len() - 1]
                            );
                        }
                    }
                }
            }

            return Ok(());
        }
        Err(e) => {
            println!("❌ Realistic inference failed: {}", e);

            // Provide specific guidance based on the error
            let error_str = format!("{}", e);
            if error_str.contains("CoreML") {
                println!("   💡 Try running with CPU execution provider instead");
            }
            if error_str.contains("shape") {
                println!("   💡 Check if the model expects different input shapes");
            }
            if error_str.contains("input") {
                println!("   💡 Check if the model expects different input names");
            }

            return Err(e.into());
        }
    }
}
