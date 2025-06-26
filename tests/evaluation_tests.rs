//! Comprehensive tests for the evaluation framework

use nodespace_nlp_engine::evaluation::*;
use nodespace_nlp_engine::*;

#[test]
fn test_evaluation_framework_creation() {
    let framework = EvaluationFramework::new();
    assert!(framework.is_ok(), "Failed to create evaluation framework");
}

#[test]
fn test_evaluation_framework_with_custom_config() {
    let rouge_config = ROUGEConfig {
        rouge_n_grams: vec![1, 2, 3],
        rouge_l_enabled: true,
        use_stemmer: true,
        alpha: 0.7,
        beta: 1.5,
    };

    let bleu_config = BLEUConfig {
        max_n_gram: 3,
        smoothing: false,
        weights: vec![0.33, 0.33, 0.34],
    };

    let framework = EvaluationFramework::with_config(rouge_config, bleu_config);
    assert!(
        framework.is_ok(),
        "Failed to create evaluation framework with custom config"
    );
}

#[test]
fn test_rouge_config_default_values() {
    let config = ROUGEConfig::default();
    assert_eq!(config.rouge_n_grams, vec![1, 2]);
    assert!(config.rouge_l_enabled);
    assert!(!config.use_stemmer);
    assert_eq!(config.alpha, 0.5);
    assert_eq!(config.beta, 1.2);
}

#[test]
fn test_bleu_config_default_values() {
    let config = BLEUConfig::default();
    assert_eq!(config.max_n_gram, 4);
    assert!(config.smoothing);
    assert_eq!(config.weights, vec![0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn test_rouge_evaluator_creation() {
    let config = ROUGEConfig::default();
    let evaluator = ROUGEEvaluator::new(config);
    assert!(evaluator.is_ok(), "Failed to create ROUGE evaluator");
}

#[test]
fn test_bleu_evaluator_creation() {
    let config = BLEUConfig::default();
    let evaluator = BLEUEvaluator::new(config);
    assert!(evaluator.is_ok(), "Failed to create BLEU evaluator");
}

#[test]
fn test_similarity_evaluator_creation() {
    let evaluator = SimilarityEvaluator::new();
    // Should not panic or fail
    let _result = evaluator.evaluate_similarity("test", "reference");
}

#[test]
fn test_rouge_scores_structure() {
    let scores = ROUGEScores {
        rouge_1: ROUGEScore {
            precision: 0.5,
            recall: 0.7,
            f1_score: 0.6,
        },
        rouge_2: ROUGEScore {
            precision: 0.4,
            recall: 0.6,
            f1_score: 0.5,
        },
        rouge_l: ROUGEScore {
            precision: 0.45,
            recall: 0.65,
            f1_score: 0.53,
        },
    };

    assert_eq!(scores.rouge_1.precision, 0.5);
    assert_eq!(scores.rouge_2.f1_score, 0.5);
    assert_eq!(scores.rouge_l.recall, 0.65);
}

#[test]
fn test_bleu_scores_structure() {
    let scores = BLEUScores {
        bleu_1_score: 0.8,
        bleu_2_score: 0.7,
        bleu_3_score: 0.6,
        bleu_4_score: 0.5,
        overall_bleu: 0.65,
        brevity_penalty: 0.95,
        precision_scores: vec![0.8, 0.7, 0.6, 0.5],
    };

    assert_eq!(scores.bleu_1_score, 0.8);
    assert_eq!(scores.overall_bleu, 0.65);
    assert_eq!(scores.precision_scores.len(), 4);
}

#[test]
fn test_similarity_scores_structure() {
    let scores = SimilarityScores {
        cosine_similarity: 0.85,
        jaccard_similarity: 0.45,
        levenshtein_similarity: 0.75,
        semantic_similarity: 0.90,
    };

    assert_eq!(scores.cosine_similarity, 0.85);
    assert_eq!(scores.semantic_similarity, 0.90);
}

#[tokio::test]
async fn test_rag_evaluation_result_stub() {
    let framework = EvaluationFramework::new().unwrap();

    let result = framework.evaluate_rag_response(
        "The capital of France is Paris, which is known for its beautiful architecture.",
        "Paris is the capital city of France.",
    );

    assert!(result.is_ok(), "RAG evaluation should succeed");

    let evaluation = result.unwrap();
    assert_eq!(evaluation.overall_quality_score, 0.0); // Stub implementation returns 0
}

#[test]
fn test_semantic_search_evaluation_stub() {
    let framework = EvaluationFramework::new().unwrap();

    let retrieved_docs = vec![
        "Document about Paris".to_string(),
        "Document about France".to_string(),
    ];

    let expected_docs = vec!["Expected document about Paris".to_string()];

    let result = framework.evaluate_semantic_search(
        "What is the capital of France?",
        &retrieved_docs,
        &expected_docs,
    );

    assert!(result.is_ok(), "Semantic search evaluation should succeed");

    let evaluation = result.unwrap();
    assert_eq!(evaluation.precision, 0.0); // Stub implementation
    assert_eq!(evaluation.recall, 0.0); // Stub implementation
}

#[test]
fn test_evaluation_report_structure() {
    let report = EvaluationReport {
        timestamp: "2025-06-26T21:00:00Z".to_string(),
        test_set_size: 100,
        rag_evaluations: vec![],
        semantic_search_evaluations: vec![],
        performance_benchmark: PerformanceBenchmark {
            evaluation_time_ms: 1500,
            throughput_evaluations_per_second: 66.67,
            memory_usage_mb: 256.0,
        },
        summary_statistics: SummaryStatistics {
            mean_rouge_f1: 0.75,
            mean_bleu_score: 0.68,
            mean_overall_quality: 0.71,
            standard_deviation: 0.12,
            percentiles: std::collections::HashMap::new(),
        },
    };

    assert_eq!(report.test_set_size, 100);
    assert_eq!(report.performance_benchmark.evaluation_time_ms, 1500);
    assert_eq!(report.summary_statistics.mean_rouge_f1, 0.75);
}

#[test]
fn test_evaluation_error_conversion() {
    let eval_error = EvaluationError::RougeError("Test error".to_string());
    let nlp_error: NLPError = eval_error.into();

    match nlp_error {
        NLPError::EvaluationError(msg) => {
            assert!(msg.contains("Test error"));
        }
        _ => panic!("Expected EvaluationError variant"),
    }
}

#[test]
fn test_serialization_rouge_config() {
    let config = ROUGEConfig::default();
    let serialized = serde_json::to_string(&config);
    assert!(serialized.is_ok(), "ROUGE config should be serializable");

    let deserialized: Result<ROUGEConfig, _> = serde_json::from_str(&serialized.unwrap());
    assert!(
        deserialized.is_ok(),
        "ROUGE config should be deserializable"
    );
}

#[test]
fn test_serialization_bleu_config() {
    let config = BLEUConfig::default();
    let serialized = serde_json::to_string(&config);
    assert!(serialized.is_ok(), "BLEU config should be serializable");

    let deserialized: Result<BLEUConfig, _> = serde_json::from_str(&serialized.unwrap());
    assert!(deserialized.is_ok(), "BLEU config should be deserializable");
}

#[test]
fn test_serialization_evaluation_results() {
    let rag_result = RAGEvaluationResult {
        rouge_scores: ROUGEScores {
            rouge_1: ROUGEScore {
                precision: 0.5,
                recall: 0.7,
                f1_score: 0.6,
            },
            rouge_2: ROUGEScore {
                precision: 0.4,
                recall: 0.6,
                f1_score: 0.5,
            },
            rouge_l: ROUGEScore {
                precision: 0.45,
                recall: 0.65,
                f1_score: 0.53,
            },
        },
        bleu_scores: BLEUScores {
            bleu_1_score: 0.8,
            bleu_2_score: 0.7,
            bleu_3_score: 0.6,
            bleu_4_score: 0.5,
            overall_bleu: 0.65,
            brevity_penalty: 0.95,
            precision_scores: vec![0.8, 0.7, 0.6, 0.5],
        },
        similarity_scores: SimilarityScores {
            cosine_similarity: 0.85,
            jaccard_similarity: 0.45,
            levenshtein_similarity: 0.75,
            semantic_similarity: 0.90,
        },
        overall_quality_score: 0.73,
    };

    let serialized = serde_json::to_string(&rag_result);
    assert!(
        serialized.is_ok(),
        "RAG evaluation result should be serializable"
    );

    let deserialized: Result<RAGEvaluationResult, _> = serde_json::from_str(&serialized.unwrap());
    assert!(
        deserialized.is_ok(),
        "RAG evaluation result should be deserializable"
    );
}

#[cfg(feature = "evaluation")]
mod evaluation_feature_tests {
    use super::*;

    #[test]
    fn test_evaluation_feature_enabled() {
        // This test only runs when the evaluation feature is enabled
        let framework = EvaluationFramework::new();
        assert!(
            framework.is_ok(),
            "Evaluation framework should be available with feature enabled"
        );
    }
}
