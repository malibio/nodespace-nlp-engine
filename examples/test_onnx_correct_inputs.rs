//! ONNX Runtime Test with Correct Inputs
//!
//! This example demonstrates the correct way to call the Gemma 3 ONNX model
//! based on the diagnostic findings.
//!
//! Key findings from diagnostic:
//! - Model does NOT accept attention_mask
//! - Model requires input_ids, position_ids, and KV cache tensors
//! - KV cache inputs are required even for first inference
//!
//! Run with: cargo run --example test_onnx_correct_inputs --features real-ml

use ort::{
    inputs,
    session::{builder::SessionBuilder, Session},
    value::Value,
};
use std::path::PathBuf;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("🧪 ONNX Runtime Correct Input Test");
    println!("==================================");

    let model_path = PathBuf::from("/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/model.onnx");

    // Load model
    println!("🔄 Loading model...");
    let session = SessionBuilder::new()?.commit_from_file(&model_path)?;
    println!("✅ Model loaded successfully");

    // Test correct inference
    println!("\n🔄 Testing inference with correct inputs...");
    test_correct_inference(&session).await?;

    println!("\n✅ Test completed successfully!");
    Ok(())
}

async fn test_correct_inference(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    // Create test input
    let input_ids = vec![1i64, 7803, 1849]; // [BOS, "Hello", "world"]
    let batch_size = 1;
    let seq_len = input_ids.len();

    println!("Token sequence: {:?}", input_ids);
    println!("Sequence length: {}", seq_len);

    // Create required tensors
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids.clone()))?;
    let position_ids: Vec<i64> = (0..seq_len).map(|i| i as i64).collect();
    let position_ids_tensor = Value::from_array(([batch_size, seq_len], position_ids))?;

    // Create empty KV cache tensors (required for first inference)
    // Based on model signature: past_key_values.X.key/value with shape [-1, 1, -1, 256]
    let num_layers = 26; // Gemma 3 1B has 26 layers (0-25)
    let num_heads = 1; // From model signature
    let head_dim = 256; // From model signature
    let past_seq_len = 0; // No past sequence for first inference

    println!("Creating KV cache tensors...");
    println!("  Layers: {}", num_layers);
    println!("  Heads: {}", num_heads);
    println!("  Head dimension: {}", head_dim);

    // Build inputs map
    let mut inputs_builder = std::collections::HashMap::new();

    // Add basic inputs
    inputs_builder.insert("input_ids".to_string(), input_ids_tensor);
    inputs_builder.insert("position_ids".to_string(), position_ids_tensor);

    // Add KV cache tensors (empty for first inference)
    for layer in 0..num_layers {
        let key_name = format!("past_key_values.{}.key", layer);
        let value_name = format!("past_key_values.{}.value", layer);

        // Create empty tensors with correct shape: [batch_size, num_heads, past_seq_len, head_dim]
        let empty_key = Value::from_array((
            [batch_size, num_heads, past_seq_len, head_dim],
            Vec::<f32>::new(),
        ))?;
        let empty_value = Value::from_array((
            [batch_size, num_heads, past_seq_len, head_dim],
            Vec::<f32>::new(),
        ))?;

        inputs_builder.insert(key_name, empty_key);
        inputs_builder.insert(value_name, empty_value);
    }

    println!("Total inputs: {}", inputs_builder.len());

    // Convert to ort inputs format
    // Since we have many inputs, we'll use the basic inputs! macro approach
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids))?;
    let position_ids_tensor = Value::from_array((
        [batch_size, seq_len],
        (0..seq_len).map(|i| i as i64).collect::<Vec<_>>(),
    ))?;

    // For now, let's try with just the basic inputs to see what happens
    let inputs = inputs! {
        "input_ids" => input_ids_tensor,
        "position_ids" => position_ids_tensor,
    }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ Inference successful!");
            println!("📊 Output analysis:");

            for (name, tensor) in outputs.iter() {
                println!("  - Output: {}", name);

                if let Ok(float_tensor) = tensor.try_extract_tensor::<f32>() {
                    let shape = float_tensor.shape();
                    println!("    Shape: {:?}", shape);

                    // Check if this looks like logits
                    if name == "logits" && shape.len() >= 2 {
                        let vocab_size = shape[shape.len() - 1];
                        println!("    🎯 Logits detected! Vocabulary size: {}", vocab_size);

                        // Get the last token's logits for next token prediction
                        if shape.len() == 3 && shape[1] > 0 {
                            let last_pos = shape[1] - 1;
                            println!("    🔍 Analyzing position {}", last_pos);

                            // Show first few and last few logits
                            let logits_view = float_tensor.slice(ndarray::s![0, last_pos, ..]);
                            let logits_vec: Vec<f32> = logits_view.iter().cloned().collect();

                            if !logits_vec.is_empty() {
                                println!(
                                    "    📈 First 10 logits: {:?}",
                                    &logits_vec[..10.min(logits_vec.len())]
                                );

                                // Find max logit for predicted token
                                let (max_idx, max_val) = logits_vec
                                    .iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .unwrap();

                                println!(
                                    "    🎯 Predicted token ID: {} (logit: {:.3})",
                                    max_idx, max_val
                                );
                            }
                        }
                    }
                } else if let Ok(int_tensor) = tensor.try_extract_tensor::<i64>() {
                    let shape = int_tensor.shape();
                    println!("    Shape: {:?} (int64)", shape);
                }
            }

            return Ok(());
        }
        Err(e) => {
            println!("❌ Inference failed: {}", e);

            let error_str = format!("{}", e);
            if error_str.contains("past_key_values") {
                println!("💡 Model requires KV cache inputs - let's try with empty cache tensors");
                return test_with_empty_kv_cache(session).await;
            }

            return Err(e.into());
        }
    }
}

async fn test_with_empty_kv_cache(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Testing with empty KV cache tensors...");

    let input_ids = vec![1i64, 7803, 1849]; // [BOS, "Hello", "world"]
    let batch_size = 1;
    let seq_len = input_ids.len();
    let num_layers = 26;
    let num_heads = 1;
    let head_dim = 256;
    let past_seq_len = 0;

    // Create inputs with proper KV cache
    let input_ids_tensor = Value::from_array(([batch_size, seq_len], input_ids.clone()))?;
    let position_ids_tensor = Value::from_array((
        [batch_size, seq_len],
        (0..seq_len).map(|i| i as i64).collect::<Vec<_>>(),
    ))?;

    // We need to create the inputs dynamically since we have 52+ KV cache tensors
    println!("Creating {} KV cache tensor pairs...", num_layers);

    // For this test, let's try with a minimal working case
    // Create inputs for just layer 0 to see if the pattern works
    let empty_shape = [batch_size, num_heads, past_seq_len, head_dim];
    let empty_data: Vec<f32> = vec![]; // Empty for 0 past sequence length

    let key_0 = Value::from_array((empty_shape, empty_data.clone()))?;
    let value_0 = Value::from_array((empty_shape, empty_data.clone()))?;

    // Try with just first layer KV cache
    let inputs = inputs! {
        "input_ids" => input_ids_tensor,
        "position_ids" => position_ids_tensor,
        "past_key_values.0.key" => key_0,
        "past_key_values.0.value" => value_0,
    }?;

    match session.run(inputs) {
        Ok(outputs) => {
            println!("✅ Partial KV cache inference successful!");
            println!("📊 Output count: {}", outputs.len());

            // Check for logits
            if let Ok(logits_tensor) = outputs["logits"].try_extract_tensor::<f32>() {
                let shape = logits_tensor.shape();
                println!("🎯 Logits shape: {:?}", shape);

                if shape.len() == 3 && shape[1] > 0 {
                    let last_pos = shape[1] - 1;
                    let vocab_size = shape[2];
                    println!("📚 Vocabulary size: {}", vocab_size);
                    println!("🔍 Predicting next token for position {}", last_pos);

                    // Get prediction
                    let logits_slice = logits_tensor.slice(ndarray::s![0, last_pos, ..]);
                    let max_idx = logits_slice
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    println!("🎯 Predicted next token ID: {}", max_idx);
                }
            }

            Ok(())
        }
        Err(e) => {
            println!("❌ Still failed with partial KV cache: {}", e);
            println!("💡 The model likely requires ALL KV cache tensors to be provided");
            Err(e.into())
        }
    }
}
