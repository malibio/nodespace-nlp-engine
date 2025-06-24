//! Download Gemma 3 1B ONNX Model
//! Downloads the real onnx-community/gemma-3-1b-it-ONNX model for production use

use std::process::Command;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Downloading Gemma 3 1B ONNX Model");
    println!("{}", "=".repeat(50));

    // Create models directory
    println!("📁 Creating models directory...");
    std::fs::create_dir_all("models/gemma-3-1b-it-onnx")?;

    // Download the optimal quantized model (998 MB)
    println!("⬇️  Downloading model_q4f16.onnx (998 MB)...");
    let output = Command::new("curl")
        .args([
            "-L",
            "-o", "models/gemma-3-1b-it-onnx/model.onnx",
            "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/onnx/model_q4f16.onnx"
        ])
        .output()?;

    if !output.status.success() {
        eprintln!(
            "❌ Failed to download model: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Err("Model download failed".into());
    }

    // Download the tokenizer
    println!("⬇️  Downloading tokenizer.json...");
    let output = Command::new("curl")
        .args([
            "-L",
            "-o",
            "models/gemma-3-1b-it-onnx/tokenizer.json",
            "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/tokenizer.json",
        ])
        .output()?;

    if !output.status.success() {
        eprintln!(
            "❌ Failed to download tokenizer: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Err("Tokenizer download failed".into());
    }

    // Download config.json for reference
    println!("⬇️  Downloading config.json...");
    let output = Command::new("curl")
        .args([
            "-L",
            "-o",
            "models/gemma-3-1b-it-onnx/config.json",
            "https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX/resolve/main/config.json",
        ])
        .output()?;

    if !output.status.success() {
        eprintln!(
            "❌ Failed to download config: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Err("Config download failed".into());
    }

    // Verify downloads
    println!("✅ Verifying downloads...");
    let model_path = "models/gemma-3-1b-it-onnx/model.onnx";
    let tokenizer_path = "models/gemma-3-1b-it-onnx/tokenizer.json";
    let config_path = "models/gemma-3-1b-it-onnx/config.json";

    if std::path::Path::new(model_path).exists() {
        let metadata = std::fs::metadata(model_path)?;
        println!("  📦 Model: {} MB", metadata.len() / 1024 / 1024);
    } else {
        return Err("Model file not found after download".into());
    }

    if std::path::Path::new(tokenizer_path).exists() {
        println!("  🔤 Tokenizer: ✅");
    } else {
        return Err("Tokenizer file not found after download".into());
    }

    if std::path::Path::new(config_path).exists() {
        println!("  ⚙️  Config: ✅");
    } else {
        return Err("Config file not found after download".into());
    }

    println!("");
    println!("🎉 Download Complete!");
    println!("{}", "=".repeat(50));
    println!("✅ Gemma 3 1B ONNX model ready for inference");
    println!("📍 Location: models/gemma-3-1b-it-onnx/");
    println!("🧪 Run: cargo run --example onnx_gemma3_test --features onnx");
    println!("🚀 Real ONNX inference now available!");

    Ok(())
}
