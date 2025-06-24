//! Unified text generation interface supporting multiple backends
//! Supports both Candle (current) and ONNX Runtime (new) backends

use crate::error::NLPError;
use crate::models::{DeviceType, TextGenerationModelConfig, TextGenerationBackend};
use crate::text_generation::{TextGenerator, EntityAnalysis, QueryIntent};
use crate::onnx_text_generation::OnnxTextGenerator;

/// Unified text generator that can use different backends
pub struct UnifiedTextGenerator {
    backend: TextGenerationBackend,
    candle_generator: Option<TextGenerator>,
    onnx_generator: Option<OnnxTextGenerator>,
    config: TextGenerationModelConfig,
    device_type: DeviceType,
    initialized: bool,
}

impl UnifiedTextGenerator {
    /// Create a new unified text generator with specified backend
    pub fn new(
        config: TextGenerationModelConfig,
        device_type: DeviceType,
        backend: TextGenerationBackend,
    ) -> Result<Self, NLPError> {
        let candle_generator = if matches!(backend, TextGenerationBackend::Candle) {
            Some(TextGenerator::new(config.clone(), device_type.clone())?)
        } else {
            None
        };

        let onnx_generator = if matches!(backend, TextGenerationBackend::Onnx) {
            Some(OnnxTextGenerator::new(config.clone(), device_type.clone())?)
        } else {
            None
        };

        Ok(Self {
            backend,
            candle_generator,
            onnx_generator,
            config,
            device_type,
            initialized: false,
        })
    }

    /// Create with automatic backend selection based on config or available features
    pub fn new_auto(
        config: TextGenerationModelConfig,
        device_type: DeviceType,
    ) -> Result<Self, NLPError> {
        let backend = match config.backend.as_ref().unwrap_or(&TextGenerationBackend::Auto) {
            TextGenerationBackend::Auto => {
                // Auto-select: Prefer ONNX if available, fallback to Candle
                if cfg!(feature = "onnx") {
                    TextGenerationBackend::Onnx
                } else {
                    TextGenerationBackend::Candle
                }
            },
            other => other.clone(),
        };

        Self::new(config, device_type, backend)
    }

    /// Initialize the text generation model
    pub async fn initialize(&mut self) -> Result<(), NLPError> {
        if self.initialized {
            return Ok(());
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &mut self.candle_generator {
                    generator.initialize().await?;
                } else {
                    return Err(NLPError::ModelLoading {
                        message: "Candle generator not initialized".to_string(),
                    });
                }
            }
            TextGenerationBackend::Onnx => {
                if let Some(generator) = &mut self.onnx_generator {
                    generator.initialize().await?;
                } else {
                    return Err(NLPError::ModelLoading {
                        message: "ONNX generator not initialized".to_string(),
                    });
                }
            }
            TextGenerationBackend::Auto => {
                return Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                });
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Generate text from a prompt
    pub async fn generate_text(&mut self, prompt: &str) -> Result<String, NLPError> {
        self.generate_text_with_params(
            prompt,
            self.config.default_max_tokens,
            self.config.default_temperature,
            self.config.default_top_p,
        )
        .await
    }

    /// Generate text with custom parameters
    pub async fn generate_text_with_params(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, NLPError> {
        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &mut self.candle_generator {
                    generator.generate_text_with_params(prompt, max_tokens, temperature, top_p).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "Candle generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Onnx => {
                if let Some(generator) = &mut self.onnx_generator {
                    generator.generate_text_with_params(prompt, max_tokens, temperature, top_p).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "ONNX generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Auto => {
                Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                })
            }
        }
    }

    /// Generate text with function calling capabilities
    pub async fn generate_with_function_calling(
        &mut self,
        prompt: &str,
        functions: Vec<serde_json::Value>,
    ) -> Result<String, NLPError> {
        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &self.candle_generator {
                    generator.generate_with_function_calling(prompt, functions).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "Candle generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Onnx => {
                // ONNX backend doesn't support function calling yet (MVP limitation)
                // Fallback to regular text generation with function context
                if let Some(generator) = &mut self.onnx_generator {
                    let function_context = if !functions.is_empty() {
                        format!("\n\nAvailable functions: {}", functions.len())
                    } else {
                        String::new()
                    };
                    let enhanced_prompt = format!("{}{}", prompt, function_context);
                    generator.generate_text(&enhanced_prompt).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "ONNX generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Auto => {
                Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                })
            }
        }
    }

    /// Generate a structured response for entity analysis
    pub async fn analyze_entity_creation(&mut self, text: &str) -> Result<EntityAnalysis, NLPError> {
        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &self.candle_generator {
                    generator.analyze_entity_creation(text).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "Candle generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Onnx => {
                // ONNX backend: Use the existing stub logic from Candle for now
                // In production, this would use the ONNX model for better analysis
                self.stub_analyze_entity_creation(text).await
            }
            TextGenerationBackend::Auto => {
                Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                })
            }
        }
    }

    /// Generate SurrealQL from natural language with schema context
    pub async fn generate_surrealql(
        &mut self,
        query: &str,
        schema_context: &str,
    ) -> Result<String, NLPError> {
        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &self.candle_generator {
                    generator.generate_surrealql(query, schema_context).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "Candle generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Onnx => {
                // ONNX backend: Use the existing stub logic from Candle for now
                self.stub_generate_surrealql(query, schema_context).await
            }
            TextGenerationBackend::Auto => {
                Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                })
            }
        }
    }

    /// Analyze query intent for natural language processing
    pub async fn analyze_query_intent(&mut self, query: &str) -> Result<QueryIntent, NLPError> {
        if !self.initialized {
            return Err(NLPError::ModelLoading {
                message: "Model not initialized".to_string(),
            });
        }

        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &self.candle_generator {
                    generator.analyze_query_intent(query).await
                } else {
                    Err(NLPError::ModelLoading {
                        message: "Candle generator not available".to_string(),
                    })
                }
            }
            TextGenerationBackend::Onnx => {
                // ONNX backend: Use the existing stub logic from Candle for now
                self.stub_analyze_query_intent(query).await
            }
            TextGenerationBackend::Auto => {
                Err(NLPError::ModelLoading {
                    message: "Auto backend should be resolved during construction".to_string(),
                })
            }
        }
    }

    /// Get model information
    pub fn model_info(&self) -> UnifiedModelInfo {
        match &self.backend {
            TextGenerationBackend::Candle => {
                if let Some(generator) = &self.candle_generator {
                    let info = generator.model_info();
                    UnifiedModelInfo {
                        model_name: info.model_name,
                        max_context_length: info.max_context_length,
                        device_type: info.device_type,
                        backend: "Candle".to_string(),
                        execution_providers: vec!["CPU/Metal".to_string()],
                    }
                } else {
                    UnifiedModelInfo::default()
                }
            }
            TextGenerationBackend::Onnx => {
                if let Some(generator) = &self.onnx_generator {
                    let info = generator.model_info();
                    UnifiedModelInfo {
                        model_name: info.model_name,
                        max_context_length: info.max_context_length,
                        device_type: self.device_type.clone(),
                        backend: info.backend,
                        execution_providers: info.execution_providers,
                    }
                } else {
                    UnifiedModelInfo::default()
                }
            }
            TextGenerationBackend::Auto => {
                UnifiedModelInfo::default()
            }
        }
    }

    /// Get current backend
    pub fn backend(&self) -> &TextGenerationBackend {
        &self.backend
    }

    // Stub implementations for ONNX backend (shared with Candle)
    async fn stub_analyze_entity_creation(&self, text: &str) -> Result<EntityAnalysis, NLPError> {
        let entity_type = if text.to_lowercase().contains("meeting") {
            "Meeting"
        } else if text.to_lowercase().contains("task") {
            "Task"
        } else if text.to_lowercase().contains("person") {
            "Person"
        } else {
            "Document"
        };

        let mut fields = std::collections::HashMap::new();
        if text.to_lowercase().contains("john") {
            fields.insert("participants".to_string(), serde_json::json!(["John"]));
        }
        if text.to_lowercase().contains("friday") {
            fields.insert("due_date".to_string(), serde_json::json!("Friday"));
        }

        Ok(EntityAnalysis {
            entity_type: entity_type.to_string(),
            title: format!("{} from ONNX text analysis", entity_type),
            fields,
            tags: vec!["auto-generated".to_string(), "onnx-backend".to_string()],
            confidence: 0.8,
        })
    }

    async fn stub_generate_surrealql(&self, query: &str, _schema_context: &str) -> Result<String, NLPError> {
        let surrealql = if query.to_lowercase().contains("find") || query.to_lowercase().contains("get") {
            if query.to_lowercase().contains("meeting") {
                "SELECT * FROM meeting WHERE date > time::now() - 1w ORDER BY date DESC LIMIT 10"
            } else if query.to_lowercase().contains("task") {
                "SELECT * FROM task WHERE status != 'completed' ORDER BY priority DESC LIMIT 10"
            } else {
                "SELECT * FROM entity ORDER BY created_at DESC LIMIT 10"
            }
        } else if query.to_lowercase().contains("create") {
            "CREATE meeting SET title = 'New Meeting (ONNX)', date = time::now(), status = 'active'"
        } else {
            "SELECT * FROM entity LIMIT 10"
        };

        Ok(surrealql.to_string())
    }

    async fn stub_analyze_query_intent(&self, query: &str) -> Result<QueryIntent, NLPError> {
        let intent_type = if query.to_lowercase().contains("create") {
            "CREATE_ENTITY"
        } else if query.to_lowercase().contains("find") || query.to_lowercase().contains("search") {
            "SEARCH_ENTITIES"
        } else if query.to_lowercase().contains("update") {
            "UPDATE_ENTITY"
        } else if query.to_lowercase().contains("delete") {
            "DELETE_ENTITY"
        } else {
            "SEARCH_ENTITIES"
        };

        let mut parameters = std::collections::HashMap::new();
        if query.to_lowercase().contains("meeting") {
            parameters.insert("entity_type".to_string(), serde_json::json!("meeting"));
        }

        Ok(QueryIntent {
            intent_type: intent_type.to_string(),
            parameters,
            confidence: 0.7,
            suggested_entity_types: Some(vec!["Meeting".to_string(), "Task".to_string()]),
            temporal_context: None,
        })
    }
}

/// Unified model information across backends
#[derive(Debug, Clone)]
pub struct UnifiedModelInfo {
    pub model_name: String,
    pub max_context_length: usize,
    pub device_type: DeviceType,
    pub backend: String,
    pub execution_providers: Vec<String>,
}

impl Default for UnifiedModelInfo {
    fn default() -> Self {
        Self {
            model_name: "unknown".to_string(),
            max_context_length: 2048,
            device_type: DeviceType::CPU,
            backend: "unknown".to_string(),
            execution_providers: vec!["CPU".to_string()],
        }
    }
}