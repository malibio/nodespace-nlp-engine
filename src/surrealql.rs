//! SurrealQL generation from natural language

use crate::error::NLPError;
use crate::text_generation::TextGenerator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SurrealQL generator that converts natural language to safe SurrealQL queries
pub struct SurrealQLGenerator {
    safety_rules: SafetyRules,
}

impl Default for SurrealQLGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SurrealQLGenerator {
    /// Create a new SurrealQL generator
    pub fn new() -> Self {
        Self {
            safety_rules: SafetyRules::default(),
        }
    }

    /// Generate SurrealQL from natural language with schema context
    pub async fn generate_surrealql(
        &self,
        text_generator: &mut TextGenerator,
        natural_query: &str,
        schema_context: &str,
        safety_checks: bool,
    ) -> Result<String, NLPError> {
        // First, analyze the query to understand intent
        let query_analysis = self.analyze_query_structure(natural_query).await?;

        // Generate the SurrealQL using the text generator
        let raw_surrealql = text_generator
            .generate_surrealql(natural_query, schema_context)
            .await?;

        // Apply safety checks if enabled
        let safe_surrealql = if safety_checks {
            self.apply_safety_checks(&raw_surrealql, &query_analysis)?
        } else {
            raw_surrealql
        };

        // Validate syntax
        self.validate_surrealql_syntax(&safe_surrealql)?;

        Ok(safe_surrealql)
    }

    /// Generate a CREATE statement for entity creation
    pub async fn generate_create_statement(
        &self,
        text_generator: &mut TextGenerator,
        entity_type: &str,
        fields: &HashMap<String, serde_json::Value>,
        schema_context: &str,
    ) -> Result<String, NLPError> {
        let fields_str = self.format_fields_for_create(fields)?;

        let prompt = format!(
            r#"Generate a SurrealQL CREATE statement for:
Entity Type: {}
Fields: {}

Schema Context: {}

Generate a valid CREATE statement with:
1. Proper table name (use lowercase with underscores)
2. All fields properly formatted for SurrealDB
3. Appropriate data types
4. UUID generation for ID if needed

SurrealQL:"#,
            entity_type, fields_str, schema_context
        );

        let response = text_generator.generate_text(&prompt).await?;

        // Extract the CREATE statement
        let create_statement = self.extract_sql_statement(&response, "CREATE")?;

        // Validate and sanitize
        self.validate_create_statement(&create_statement)?;

        Ok(create_statement)
    }

    /// Generate a SELECT statement for entity search
    pub async fn generate_select_statement(
        &self,
        text_generator: &mut TextGenerator,
        search_criteria: &SearchCriteria,
        schema_context: &str,
    ) -> Result<String, NLPError> {
        let prompt = format!(
            r#"Generate a SurrealQL SELECT statement for:
Table: {}
Filters: {}
Text Search: {}
Relationships: {}
Limit: {}

Schema Context: {}

Generate a valid SELECT statement with:
1. Proper table references
2. WHERE clauses for filters
3. Text search using CONTAINS or similar
4. JOIN or relationship traversal if needed
5. LIMIT clause

SurrealQL:"#,
            search_criteria.table.as_deref().unwrap_or("*"),
            serde_json::to_string(&search_criteria.filters).unwrap_or_default(),
            search_criteria.text_search.as_deref().unwrap_or(""),
            search_criteria.include_relationships,
            search_criteria.limit.unwrap_or(10),
            schema_context
        );

        let response = text_generator.generate_text(&prompt).await?;
        let select_statement = self.extract_sql_statement(&response, "SELECT")?;

        self.validate_select_statement(&select_statement)?;

        Ok(select_statement)
    }

    /// Generate a RELATE statement for relationship creation
    pub async fn generate_relate_statement(
        &self,
        text_generator: &mut TextGenerator,
        from_entity: &str,
        to_entity: &str,
        relationship_type: &str,
        properties: &HashMap<String, serde_json::Value>,
    ) -> Result<String, NLPError> {
        let properties_str = self.format_fields_for_create(properties)?;

        let prompt = format!(
            r#"Generate a SurrealQL RELATE statement for:
From Entity: {}
To Entity: {}
Relationship Type: {}
Properties: {}

Generate a valid RELATE statement with:
1. Proper entity references
2. Relationship table name (use lowercase with underscores)
3. Properties as SET clause
4. Proper SurrealDB syntax

SurrealQL:"#,
            from_entity, to_entity, relationship_type, properties_str
        );

        let response = text_generator.generate_text(&prompt).await?;
        let relate_statement = self.extract_sql_statement(&response, "RELATE")?;

        self.validate_relate_statement(&relate_statement)?;

        Ok(relate_statement)
    }

    /// Analyze the structure of a natural language query
    async fn analyze_query_structure(&self, query: &str) -> Result<QueryAnalysis, NLPError> {
        // STUB: Simple analysis for now
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

        let query_type = match intent_type {
            "CREATE_ENTITY" => QueryType::Create,
            "SEARCH_ENTITIES" => QueryType::Select,
            "UPDATE_ENTITY" => QueryType::Update,
            "DELETE_ENTITY" => QueryType::Delete,
            _ => QueryType::Select,
        };

        Ok(QueryAnalysis {
            query_type,
            confidence: 0.8,
            parameters: std::collections::HashMap::new(),
            tables_mentioned: self.extract_table_names(query),
            potential_injection: self.detect_potential_injection(query),
        })
    }

    /// Apply safety checks to generated SurrealQL
    fn apply_safety_checks(
        &self,
        surrealql: &str,
        analysis: &QueryAnalysis,
    ) -> Result<String, NLPError> {
        let mut safe_query = surrealql.to_string();

        // Remove potentially dangerous patterns
        for pattern in &self.safety_rules.forbidden_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                safe_query = regex.replace_all(&safe_query, "").to_string();
            }
        }

        // Ensure no SQL injection patterns
        if analysis.potential_injection {
            return Err(NLPError::SurrealQLGeneration {
                message: "Potential SQL injection detected".to_string(),
            });
        }

        // Validate table names against allowed patterns
        for table in &analysis.tables_mentioned {
            if !self.is_valid_table_name(table) {
                return Err(NLPError::SurrealQLGeneration {
                    message: format!("Invalid table name: {}", table),
                });
            }
        }

        // Add LIMIT clause if missing for SELECT statements
        if analysis.query_type == QueryType::Select && !safe_query.contains("LIMIT") {
            safe_query.push_str(" LIMIT 100"); // Default safety limit
        }

        Ok(safe_query)
    }

    /// Validate SurrealQL syntax
    fn validate_surrealql_syntax(&self, surrealql: &str) -> Result<(), NLPError> {
        // Basic syntax validation
        let trimmed = surrealql.trim();

        if trimmed.is_empty() {
            return Err(NLPError::SurrealQLGeneration {
                message: "Empty SurrealQL statement".to_string(),
            });
        }

        // Check for valid statement types
        let valid_starts = ["SELECT", "CREATE", "UPDATE", "DELETE", "RELATE", "INSERT"];
        let starts_with_valid = valid_starts
            .iter()
            .any(|&start| trimmed.to_uppercase().starts_with(start));

        if !starts_with_valid {
            return Err(NLPError::SurrealQLGeneration {
                message: "SurrealQL must start with a valid statement type".to_string(),
            });
        }

        // Check for balanced parentheses
        let open_count = trimmed.chars().filter(|&c| c == '(').count();
        let close_count = trimmed.chars().filter(|&c| c == ')').count();

        if open_count != close_count {
            return Err(NLPError::SurrealQLGeneration {
                message: "Unbalanced parentheses in SurrealQL".to_string(),
            });
        }

        Ok(())
    }

    /// Extract SQL statement from text response
    fn extract_sql_statement(
        &self,
        response: &str,
        statement_type: &str,
    ) -> Result<String, NLPError> {
        let lines: Vec<&str> = response.lines().collect();

        // Look for lines that start with the statement type
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.to_uppercase().starts_with(statement_type) {
                return Ok(trimmed.to_string());
            }
        }

        // If not found, try to extract from code blocks
        let code_block_regex = Regex::new(r"```(?:sql|surrealql)?\s*\n?([^`]+)```").unwrap();
        if let Some(captures) = code_block_regex.captures(response) {
            if let Some(code) = captures.get(1) {
                let code_lines: Vec<&str> = code.as_str().lines().collect();
                for line in code_lines {
                    let trimmed = line.trim();
                    if trimmed.to_uppercase().starts_with(statement_type) {
                        return Ok(trimmed.to_string());
                    }
                }
            }
        }

        // Fallback: return the first non-empty line
        for line in &lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with("```") {
                return Ok(trimmed.to_string());
            }
        }

        Err(NLPError::SurrealQLGeneration {
            message: format!(
                "Could not extract {} statement from response",
                statement_type
            ),
        })
    }

    /// Format fields for CREATE or SET statements
    fn format_fields_for_create(
        &self,
        fields: &HashMap<String, serde_json::Value>,
    ) -> Result<String, NLPError> {
        let mut formatted_fields = Vec::new();

        for (key, value) in fields {
            let formatted_value = match value {
                serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Array(arr) => {
                    serde_json::to_string(arr).map_err(|e| NLPError::SurrealQLGeneration {
                        message: format!("Failed to serialize array: {}", e),
                    })?
                }
                serde_json::Value::Object(obj) => {
                    serde_json::to_string(obj).map_err(|e| NLPError::SurrealQLGeneration {
                        message: format!("Failed to serialize object: {}", e),
                    })?
                }
                serde_json::Value::Null => "NULL".to_string(),
            };

            formatted_fields.push(format!("{}: {}", key, formatted_value));
        }

        Ok(formatted_fields.join(", "))
    }

    /// Extract table names mentioned in query
    fn extract_table_names(&self, query: &str) -> Vec<String> {
        // Simple extraction - in a real implementation, this would be more sophisticated
        let table_patterns = [
            r"\b(meeting|task|person|document|project|customer|order)s?\b",
            r"\b[a-z_]+\b",
        ];

        let mut tables = Vec::new();
        for pattern in &table_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for capture in regex.captures_iter(&query.to_lowercase()) {
                    if let Some(table) = capture.get(1).or_else(|| capture.get(0)) {
                        tables.push(table.as_str().to_string());
                    }
                }
            }
        }

        tables
    }

    /// Detect potential SQL injection patterns
    fn detect_potential_injection(&self, query: &str) -> bool {
        let injection_patterns = [
            r";\s*(drop|delete|update|insert|create|alter)\s+",
            r"--\s*",
            r"/\*.*\*/",
            r"\bunion\s+select\b",
            r"\bor\s+1\s*=\s*1\b",
            r"'\s*or\s*'",
        ];

        let lower_query = query.to_lowercase();
        for pattern in &injection_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(&lower_query) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if table name is valid
    fn is_valid_table_name(&self, table: &str) -> bool {
        let table_regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
        table_regex.is_match(table) && table.len() <= 64
    }

    /// Validate CREATE statement
    fn validate_create_statement(&self, statement: &str) -> Result<(), NLPError> {
        if !statement.to_uppercase().starts_with("CREATE") {
            return Err(NLPError::SurrealQLGeneration {
                message: "Statement must start with CREATE".to_string(),
            });
        }

        // Additional CREATE-specific validation
        Ok(())
    }

    /// Validate SELECT statement
    fn validate_select_statement(&self, statement: &str) -> Result<(), NLPError> {
        if !statement.to_uppercase().starts_with("SELECT") {
            return Err(NLPError::SurrealQLGeneration {
                message: "Statement must start with SELECT".to_string(),
            });
        }

        // Additional SELECT-specific validation
        Ok(())
    }

    /// Validate RELATE statement
    fn validate_relate_statement(&self, statement: &str) -> Result<(), NLPError> {
        if !statement.to_uppercase().starts_with("RELATE") {
            return Err(NLPError::SurrealQLGeneration {
                message: "Statement must start with RELATE".to_string(),
            });
        }

        // Additional RELATE-specific validation
        Ok(())
    }
}

/// Query analysis result
#[derive(Debug, Clone)]
struct QueryAnalysis {
    query_type: QueryType,
    #[allow(dead_code)]
    confidence: f32,
    #[allow(dead_code)]
    parameters: HashMap<String, serde_json::Value>,
    tables_mentioned: Vec<String>,
    potential_injection: bool,
}

/// Query type classification
#[derive(Debug, Clone, PartialEq)]
enum QueryType {
    Select,
    Create,
    Update,
    Delete,
    #[allow(dead_code)]
    Relate,
}

/// Search criteria for SELECT statement generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCriteria {
    pub table: Option<String>,
    pub filters: HashMap<String, serde_json::Value>,
    pub text_search: Option<String>,
    pub include_relationships: bool,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Safety rules for SurrealQL generation
#[derive(Debug, Clone)]
struct SafetyRules {
    forbidden_patterns: Vec<String>,
    #[allow(dead_code)]
    max_query_length: usize,
    #[allow(dead_code)]
    allowed_tables: Option<Vec<String>>,
}

impl Default for SafetyRules {
    fn default() -> Self {
        Self {
            forbidden_patterns: vec![
                r";\s*(drop|alter|create\s+database|create\s+user)".to_string(),
                r"--.*".to_string(),
                r"/\*.*\*/".to_string(),
                r"xp_cmdshell".to_string(),
                r"sp_executesql".to_string(),
            ],
            max_query_length: 2048,
            allowed_tables: None,
        }
    }
}
