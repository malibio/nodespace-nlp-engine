//! Phi-4 Multimodal Model Isolated Testing
//!
//! This example tests the Phi-4 multimodal model in isolation to verify:
//! 1. Model loading and initialization
//! 2. Basic text inference capability
//! 3. Context understanding and curation
//! 4. Enhanced context generation for embedding purposes
//!
//! Usage:
//!   cargo run --example test_phi4 --features phi4-experimental
//!
//! Environment Variables:
//!   PHI4_MODEL_PATH - Override default model path
//!   RUST_LOG - Set logging level (e.g., info, debug)

use std::env;
use std::time::Instant;
use tokio;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[cfg(feature = "phi4-experimental")]
use nodespace_nlp_engine::phi4_test::{Phi4TestHarness, TestResults};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    info!("🚀 Starting Phi-4 Multimodal Model Testing");
    info!("===========================================");

    #[cfg(not(feature = "phi4-experimental"))]
    {
        error!("❌ Phi-4 experimental feature not enabled!");
        error!("Run with: cargo run --example test_phi4 --features phi4-experimental");
        return Ok(());
    }

    #[cfg(feature = "phi4-experimental")]
    {
        // Check for custom model path
        let model_path = env::var("PHI4_MODEL_PATH").unwrap_or_else(|_| {
            "/Users/malibio/nodespace/models/gpu/gpu-int4-rtn-block-32".to_string()
        });

        info!("📁 Model path: {}", model_path);

        // Create test harness
        let mut harness = if env::var("PHI4_MODEL_PATH").is_ok() {
            Phi4TestHarness::with_model_path(model_path)
        } else {
            Phi4TestHarness::new()
        };

        // Display model info
        let model_info = harness.model_info();
        info!("📊 Model Info:");
        info!("   Path: {:?}", model_info.model_path);
        info!("   Initialized: {}", model_info.initialized);

        // Run comprehensive tests
        let test_results = run_comprehensive_tests(&mut harness).await;

        // Display results
        display_test_results(&test_results);

        // Summary
        info!("===========================================");
        if test_results.all_passed() {
            info!("🎉 All tests passed! Phi-4 is ready for integration.");
        } else {
            warn!("⚠️  Some tests failed. Check output above for details.");
            info!(
                "✅ Passed: {}/{}",
                test_results.success_count(),
                test_results.total_tests()
            );
        }
    }

    Ok(())
}

#[cfg(feature = "phi4-experimental")]
async fn run_comprehensive_tests(harness: &mut Phi4TestHarness) -> TestResults {
    info!("🧪 Running comprehensive Phi-4 tests...");

    // Test 1: Basic inference
    info!("\n📝 Test 1: Basic Inference");
    info!("---------------------------");
    let start = Instant::now();
    let basic_result = match harness.test_basic_inference().await {
        Ok(response) => {
            let duration = start.elapsed();
            info!(
                "✅ Basic inference successful ({:.2}s)",
                duration.as_secs_f32()
            );
            info!("📤 Response: {}", truncate_response(&response, 200));
            Ok(response)
        }
        Err(e) => {
            error!("❌ Basic inference failed: {}", e);
            Err(e.to_string())
        }
    };

    // Test 2: Context understanding
    info!("\n🧠 Test 2: Context Understanding");
    info!("----------------------------------");
    let start = Instant::now();
    let context_result = match harness.test_context_understanding().await {
        Ok(response) => {
            let duration = start.elapsed();
            info!(
                "✅ Context understanding successful ({:.2}s)",
                duration.as_secs_f32()
            );
            info!("📤 Response: {}", truncate_response(&response, 300));
            Ok(response)
        }
        Err(e) => {
            error!("❌ Context understanding failed: {}", e);
            Err(e.to_string())
        }
    };

    // Test 3: Enhanced context generation (our use case)
    info!("\n🎯 Test 3: Enhanced Context Generation");
    info!("---------------------------------------");
    let start = Instant::now();
    let enhanced_result = match harness.test_enhanced_context_generation().await {
        Ok(response) => {
            let duration = start.elapsed();
            info!(
                "✅ Enhanced context generation successful ({:.2}s)",
                duration.as_secs_f32()
            );
            info!("📤 Enhanced context: {}", truncate_response(&response, 400));
            Ok(response)
        }
        Err(e) => {
            error!("❌ Enhanced context generation failed: {}", e);
            Err(e.to_string())
        }
    };

    TestResults {
        basic_inference: basic_result,
        context_understanding: context_result,
        enhanced_context_generation: enhanced_result,
        model_info: harness.model_info(),
    }
}

#[cfg(feature = "phi4-experimental")]
fn display_test_results(results: &TestResults) {
    info!("\n📊 Test Results Summary");
    info!("========================");

    // Basic inference
    match &results.basic_inference {
        Ok(_) => info!("✅ Basic Inference: PASSED"),
        Err(e) => error!("❌ Basic Inference: FAILED - {}", e),
    }

    // Context understanding
    match &results.context_understanding {
        Ok(_) => info!("✅ Context Understanding: PASSED"),
        Err(e) => error!("❌ Context Understanding: FAILED - {}", e),
    }

    // Enhanced context generation
    match &results.enhanced_context_generation {
        Ok(_) => info!("✅ Enhanced Context Generation: PASSED"),
        Err(e) => error!("❌ Enhanced Context Generation: FAILED - {}", e),
    }

    // Model status
    info!("\n🔧 Model Status:");
    info!("   Model Loaded: {}", results.model_info.model_loaded);
    info!("   Initialized: {}", results.model_info.initialized);
    info!("   Path: {:?}", results.model_info.model_path);
}

/// Truncate response for display
fn truncate_response(response: &str, max_length: usize) -> String {
    if response.len() <= max_length {
        response.to_string()
    } else {
        let truncated = &response[..max_length];
        format!("{}...", truncated)
    }
}

/// Performance testing helper
#[cfg(feature = "phi4-experimental")]
async fn run_performance_test(
    harness: &mut Phi4TestHarness,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("\n⚡ Performance Test");
    info!("-------------------");

    let test_prompts = vec![
        "Hello, how are you?",
        "Explain the concept of semantic embeddings",
        "What are the benefits of multimodal AI models?",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        let start = Instant::now();
        match harness.test_basic_inference().await {
            Ok(response) => {
                let duration = start.elapsed();
                info!(
                    "Test {}: {:.2}s - {} chars",
                    i + 1,
                    duration.as_secs_f32(),
                    response.len()
                );
            }
            Err(e) => {
                error!("Test {} failed: {}", i + 1, e);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "phi4-experimental")]
async fn demonstration_mode(
    harness: &mut Phi4TestHarness,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("\n🎭 Demonstration Mode");
    info!("=====================");
    info!("This shows how Phi-4 would enhance context for embeddings");

    // Example from our actual use case
    let context_scenarios = vec![
        (
            "Meeting note about authentication",
            r#"
Node: "Discussed OAuth2 implementation for user authentication"
Parent: "Q3 Security Improvements Project"
Siblings: ["Database encryption setup", "API security audit"]
Mentions: ["Authentication flow", "User session management"]
Related: ["Previous security vulnerabilities in login system"]
"#,
        ),
        (
            "Project planning note",
            r#"
Node: "Allocated $50k budget for React frontend development"
Parent: "Q3 Development Budget Planning"
Siblings: ["Backend development budget: $75k", "Testing and QA: $25k"]
Mentions: ["React components", "UI/UX design", "Development timeline"]
Related: ["Previous frontend had performance issues"]
"#,
        ),
    ];

    for (scenario_name, context) in context_scenarios {
        info!("\n🎬 Scenario: {}", scenario_name);
        info!("Context: {}", context.trim());

        let start = Instant::now();
        match harness.test_enhanced_context_generation().await {
            Ok(enhanced_context) => {
                let duration = start.elapsed();
                info!("✅ Enhanced context ({:.2}s):", duration.as_secs_f32());
                info!("📝 {}", enhanced_context.trim());
            }
            Err(e) => {
                error!("❌ Failed: {}", e);
            }
        }
    }

    Ok(())
}
