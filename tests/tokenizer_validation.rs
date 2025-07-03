// tests/tokenizer_validation.rs
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

#[test]
fn validate_local_tokenizer_decoding() {
    // Compare original vs official tokenizer
    let original_path =
        "/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/tokenizer.json.original";
    let official_path =
        "/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/tokenizer.json.official";

    let original_tokenizer =
        Tokenizer::from_file(original_path).expect("Failed to load original tokenizer");
    let official_tokenizer =
        Tokenizer::from_file(official_path).expect("Failed to load official tokenizer");

    println!("=== TOKENIZER COMPARISON ===");
    println!(
        "Original vocab size: {}",
        original_tokenizer.get_vocab(true).len()
    );
    println!(
        "Official vocab size: {}",
        official_tokenizer.get_vocab(true).len()
    );

    // Test the problematic token
    let test_token_id = 2011;
    let original_decode = original_tokenizer
        .decode(&[test_token_id], true)
        .expect("Decoding failed");
    let official_decode = official_tokenizer
        .decode(&[test_token_id], true)
        .expect("Decoding failed");

    println!("Token 2011 comparison:");
    println!("  Original: '{}'", original_decode);
    println!("  Official: '{}'", official_decode);

    // Test a realistic response
    let test_text = "Based on the provided context, the income range segment we are targeting is 75,000-150,000 annually.";

    let original_encoded = original_tokenizer
        .encode(test_text, false)
        .expect("Encoding failed");
    let official_encoded = official_tokenizer
        .encode(test_text, false)
        .expect("Encoding failed");

    println!("\nTest encoding comparison:");
    println!(
        "  Original tokens: {} -> {:?}",
        original_encoded.get_ids().len(),
        &original_encoded.get_ids()[..5]
    );
    println!(
        "  Official tokens: {} -> {:?}",
        official_encoded.get_ids().len(),
        &official_encoded.get_ids()[..5]
    );

    let original_roundtrip = original_tokenizer
        .decode(original_encoded.get_ids(), true)
        .expect("Decoding failed");
    let official_roundtrip = official_tokenizer
        .decode(official_encoded.get_ids(), true)
        .expect("Decoding failed");

    println!("\nRound-trip test:");
    println!("  Original: '{}'", original_roundtrip);
    println!("  Official: '{}'", official_roundtrip);

    if original_roundtrip == official_roundtrip {
        println!("✅ Tokenizers are identical!");
    } else {
        println!("❌ Tokenizers are different!");
    }
}

#[test]
fn inspect_tokenizer_vocabulary_sample() {
    let tokenizer_path = "/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    let vocab = tokenizer.get_vocab(true);
    println!("Vocabulary size: {}", vocab.len());

    // Show first 20 tokens to see what we're dealing with
    let mut vocab_vec: Vec<(String, u32)> = vocab.into_iter().collect();
    vocab_vec.sort_by_key(|(_, id)| *id);

    println!("First 20 vocabulary entries:");
    for (token, id) in vocab_vec.iter().take(20) {
        println!("  ID {}: '{}'", id, token);
    }

    println!("\nLast 20 vocabulary entries:");
    for (token, id) in vocab_vec.iter().rev().take(20) {
        println!("  ID {}: '{}'", id, token);
    }

    // Check if our problematic token IDs exist
    let problematic_ids = [2011, 236776, 236761];
    for token_id in problematic_ids {
        if let Some((token, _)) = vocab_vec.iter().find(|(_, id)| *id == token_id) {
            println!("Found problematic token ID {}: '{}'", token_id, token);
        } else {
            println!("Token ID {} NOT FOUND in vocabulary!", token_id);
        }
    }
}

#[tokio::test]
async fn fetch_and_validate_official_tokenizer() {
    // --- 1. Fetch the official tokenizer ---
    let api = Api::new().unwrap();

    // IMPORTANT: We need to identify the exact model used for ONNX export
    // Common Gemma variants that could be our source:
    let possible_models = [
        "google/gemma-1.1-2b-it",
        "google/gemma-2-2b-it",
        "google/gemma-3-1b-it", // This seems most likely based on filename
        "google/gemma-7b-it",
        "google/gemma-2-9b-it",
    ];

    // Let's try models that might have 262,144 vocab size
    let model_name = "google/gemma-7b-it"; // Larger models might have larger vocab
    println!("Fetching official tokenizer from: {}", model_name);

    let official_tokenizer_path = api
        .model(model_name.to_string())
        .get("tokenizer.json")
        .await
        .expect("Failed to download official tokenizer");

    let official_tokenizer =
        Tokenizer::from_file(&official_tokenizer_path).expect("Failed to load official tokenizer");

    // --- 2. Load your local tokenizer ---
    let local_tokenizer_path = "/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/tokenizer.json";
    let local_tokenizer =
        Tokenizer::from_file(local_tokenizer_path).expect("Failed to load local tokenizer");

    // --- 3. Compare their vocabularies ---
    let official_vocab = official_tokenizer.get_vocab(true); // true = include added tokens
    let local_vocab = local_tokenizer.get_vocab(true);

    println!("Official vocab size: {}", official_vocab.len());
    println!("Local vocab size: {}", local_vocab.len());

    if official_vocab == local_vocab {
        println!("✅ Vocabularies are identical!");
    } else {
        println!("❌ Vocabularies DO NOT MATCH.");

        // Show some differences
        let mut official_vec: Vec<(String, u32)> = official_vocab.into_iter().collect();
        let mut local_vec: Vec<(String, u32)> = local_vocab.into_iter().collect();
        official_vec.sort_by_key(|(_, id)| *id);
        local_vec.sort_by_key(|(_, id)| *id);

        println!("\nFirst 10 official tokens:");
        for (token, id) in official_vec.iter().take(10) {
            println!("  ID {}: '{}'", id, token);
        }

        println!("\nFirst 10 local tokens:");
        for (token, id) in local_vec.iter().take(10) {
            println!("  ID {}: '{}'", id, token);
        }

        // Test our problematic token
        let test_token_id = 2011;
        let official_decode = official_tokenizer
            .decode(&[test_token_id], true)
            .unwrap_or_default();
        let local_decode = local_tokenizer
            .decode(&[test_token_id], true)
            .unwrap_or_default();

        println!("\nToken 2011 comparison:");
        println!("  Official: '{}'", official_decode);
        println!("  Local: '{}'", local_decode);

        // This assertion will fail, proving the mismatch
        assert_eq!(
            official_decode, local_decode,
            "Token 2011 decoding mismatch proves tokenizer incompatibility!"
        );
    }
}
