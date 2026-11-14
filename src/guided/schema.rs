//! JSON Schema compilation and CPU masks, isolated from free text decoding.

use std::sync::{Arc, Mutex};

use llguidance::{
    api::{GrammarInit, ParserLimits, TopLevelGrammar},
    ParserFactory, TokenParser,
};
use serde_json::Value;
use toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie};

use super::{JsonTokenCatalog, JsonTokenConstraint, TokenConstraint};
use crate::{InferError, Result, RustTokenizer};

fn error(message: impl std::fmt::Display) -> InferError {
    InferError::Config(format!("json_schema: {message}"))
}

/// Validated, tokenizer-independent schema grammar.
#[derive(Clone)]
pub struct JsonSchema {
    grammar: GrammarInit,
}

impl std::fmt::Debug for JsonSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonSchema").finish_non_exhaustive()
    }
}

impl JsonSchema {
    /// Compile the supported JSON Schema subset, rejecting unknown keywords.
    ///
    /// # Errors
    /// Rejects invalid, unsupported, unsatisfiable or oversized schemas.
    pub fn compile(schema: &Value) -> Result<Self> {
        if schema.to_string().len() > 65_536 {
            return Err(error("schema exceeds 64 KiB"));
        }
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(error("root type must be object"));
        }
        validate_schema(schema, "$", 0, &mut 0)?;
        let grammar = GrammarInit::Serialized(TopLevelGrammar::from_json_schema(schema.clone()));
        // Contain the upstream compiler's panic boundary for untrusted schemas.
        let (grammar, lexer) =
            llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grammar.to_internal(None, ParserLimits::default())
            }))
            .map_err(error)?;
        let warnings = lexer.render_warnings();
        if !warnings.is_empty() {
            return Err(error(warnings.join("; ")));
        }
        Ok(Self {
            grammar: GrammarInit::Internal(grammar, lexer),
        })
    }
}

fn validate_schema(schema: &Value, path: &str, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    if depth > 32 || *nodes > 1024 {
        return Err(error(format!(
            "{path}: schema exceeds depth 32 or 1024 nodes"
        )));
    }
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| error(format!("{path}: expected schema object or boolean")))?;
    for (key, value) in object {
        let location = format!("{path}/{key}");
        let invalid = || error(format!("{location}: invalid keyword value"));
        match key.as_str() {
            "type" => {
                let valid = |v: &Value| {
                    matches!(
                        v.as_str(),
                        Some(
                            "object"
                                | "array"
                                | "string"
                                | "integer"
                                | "number"
                                | "boolean"
                                | "null"
                        )
                    )
                };
                if let Some(types) = value.as_array() {
                    if types.is_empty()
                        || !types.iter().all(valid)
                        || types
                            .iter()
                            .enumerate()
                            .any(|(i, v)| types[..i].contains(v))
                    {
                        return Err(invalid());
                    }
                } else if !valid(value) {
                    return Err(invalid());
                }
            }
            "properties" => {
                for (name, child) in value.as_object().ok_or_else(invalid)? {
                    validate_schema(child, &format!("{location}/{name}"), depth + 1, nodes)?;
                }
            }
            "required" => {
                let required = value.as_array().ok_or_else(invalid)?;
                for (i, name) in required.iter().enumerate() {
                    let name = name.as_str().ok_or_else(invalid)?;
                    if required[..i].iter().any(|v| v.as_str() == Some(name)) {
                        return Err(invalid());
                    }
                    if !object
                        .get("properties")
                        .and_then(Value::as_object)
                        .is_some_and(|p| p.contains_key(name))
                    {
                        return Err(error(format!(
                            "{location}: required key {name:?} must be declared in properties"
                        )));
                    }
                }
            }
            "items" | "additionalProperties" => {
                validate_schema(value, &location, depth + 1, nodes)?
            }
            "enum" => {
                let values = value.as_array().ok_or_else(invalid)?;
                if values.is_empty()
                    || values
                        .iter()
                        .enumerate()
                        .any(|(i, v)| v.is_object() || v.is_array() || values[..i].contains(v))
                {
                    return Err(invalid());
                }
            }
            "title" | "description" => {
                if !value.is_string() {
                    return Err(invalid());
                }
            }
            "default" => {} // Annotation only, never a generation default.
            _ => {
                return Err(error(format!(
                    "{location}: unsupported JSON Schema keyword"
                )))
            }
        }
    }
    Ok(())
}

/// Lazy, model-local trie and factory; never built for text/json_object.
pub struct SchemaTokenCatalog {
    factory: ParserFactory,
    catalog: Arc<JsonTokenCatalog>,
    eot_tokens: Vec<usize>,
    vocab_size: usize,
}

impl SchemaTokenCatalog {
    /// Build a trie from the HF tokenizer's raw byte surfaces.
    ///
    /// # Errors
    /// Rejects unsupported tokenizers and invalid end-of-turn token IDs.
    pub fn from_tokenizer(tokenizer: &RustTokenizer, eot_tokens: &[usize]) -> Result<Self> {
        Self::from_bytes(tokenizer.schema_token_bytes()?, eot_tokens)
    }

    fn from_bytes(mut bytes: Vec<Vec<u8>>, eot_tokens: &[usize]) -> Result<Self> {
        let vocab_size = bytes.len();
        let eos = eot_tokens
            .first()
            .copied()
            .ok_or_else(|| error("missing EOS token"))?;
        let vocab = u32::try_from(vocab_size).map_err(error)?;
        for &id in eot_tokens {
            bytes
                .get_mut(id)
                .ok_or_else(|| error("EOS outside vocabulary"))?
                .clear();
        }
        // Upstream special token surfaces start with 0xff, never JSON content.
        for token in &mut bytes {
            if token.first() == Some(&0xff) {
                token.clear();
            }
        }
        let trie = TokTrie::from(
            &TokRxInfo::new(vocab, u32::try_from(eos).map_err(error)?),
            &bytes,
        );
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        // No fast-forward/backtracking: the existing sampler commits one ID.
        let mut factory =
            ParserFactory::new(&env, InferenceCapabilities::default(), &[]).map_err(error)?;
        factory.quiet();
        Ok(Self {
            factory,
            catalog: Arc::new(JsonTokenCatalog::from_bytes(bytes)),
            eot_tokens: eot_tokens.to_vec(),
            vocab_size,
        })
    }
}

/// Intersection of schema semantics and the existing JSON syntax guard.
pub struct SchemaTokenConstraint {
    parser: Mutex<TokenParser>,
    syntax: JsonTokenConstraint,
    catalog: Arc<SchemaTokenCatalog>,
}

impl SchemaTokenConstraint {
    /// Create isolated parser state for one generation.
    ///
    /// # Errors
    /// Rejects grammars that cannot be instantiated for this vocabulary.
    pub fn new(catalog: Arc<SchemaTokenCatalog>, schema: &JsonSchema) -> Result<Self> {
        let mut parser = catalog
            .factory
            .create_parser_from_init_default(schema.grammar.clone())
            .map_err(error)?;
        parser.start_without_prompt();
        Ok(Self {
            syntax: JsonTokenConstraint::new(
                Arc::clone(&catalog.catalog),
                &catalog.eot_tokens,
                false,
            ),
            parser: Mutex::new(parser),
            catalog,
        })
    }
}

impl TokenConstraint for SchemaTokenConstraint {
    fn mask_logits(&self, logits: &mut [f32]) -> Result<()> {
        let mut parser = self.parser.lock().map_err(error)?;
        let finished = parser.is_accepting() && self.syntax.is_finished();
        let mask = if finished {
            None
        } else {
            Some(parser.compute_mask().map_err(error)?)
        };
        self.syntax.mask_logits(logits)?;
        for (id, logit) in logits.iter_mut().enumerate() {
            let allowed = if self.catalog.eot_tokens.contains(&id) {
                finished
            } else {
                !finished
                    && id < self.catalog.vocab_size
                    && mask.as_ref().is_some_and(|m| m.get(id))
            };
            if !allowed {
                *logit = f32::NEG_INFINITY;
            }
        }
        if !logits.iter().any(|v| v.is_finite()) {
            return Err(error("no admissible token to extend the prefix"));
        }
        Ok(())
    }

    fn accept_token(&self, token: usize) -> Result<()> {
        let mut parser = self.parser.lock().map_err(error)?;
        if self.catalog.eot_tokens.contains(&token) {
            if parser.is_accepting() && self.syntax.is_finished() {
                return Ok(());
            }
            return Err(error("EOS before complete schema output"));
        }
        if token >= self.catalog.vocab_size || self.syntax.is_finished() {
            return Err(error("token outside vocabulary or after completed output"));
        }
        let id = u32::try_from(token).map_err(error)?;
        if !parser.validate_token(id).map_err(error)? {
            return Err(error("token violates schema"));
        }
        self.syntax.accept_token(token)?;
        if parser.consume_token(id).map_err(error)? != 0 {
            return Err(error("unexpected parser backtracking"));
        }
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.parser
            .lock()
            .map(|mut parser| parser.is_accepting() && self.syntax.is_finished())
            .unwrap_or(false)
    }
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
