//! `pmetal tokenize` — tokenize a text corpus into binary shards for pretraining.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Tokenize", subcommand = "tokenize")]
#[serde(rename_all = "snake_case")]
pub struct TokenizeSpec {
    #[job(
        label = "Input JSONL",
        group = "Input",
        argv = "--input",
        kind = "path",
        required
    )]
    #[serde(default)]
    pub input: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        required
    )]
    #[serde(default)]
    pub output: String,

    #[job(
        label = "Tokenizer Model",
        group = "Tokenizer",
        argv = "--tokenizer",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub tokenizer: String,

    #[job(
        label = "Text Column",
        group = "Input",
        argv = "--text-column",
        default = "text"
    )]
    #[serde(default = "default_text_column")]
    pub text_column: String,

    #[job(
        label = "Docs Per Shard",
        group = "Output",
        argv = "--docs-per-shard",
        min = 1,
        max = 1_000_000,
        default_int = 10000
    )]
    #[serde(default = "default_docs_per_shard")]
    pub docs_per_shard: usize,
}

impl Default for TokenizeSpec {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: String::new(),
            tokenizer: String::new(),
            text_column: default_text_column(),
            docs_per_shard: default_docs_per_shard(),
        }
    }
}

impl TokenizeSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_text_column() -> String {
    "text".to_string()
}
fn default_docs_per_shard() -> usize {
    10000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = TokenizeSpec {
            input: "i".into(),
            output: "o".into(),
            tokenizer: "t".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--input".to_string()));
        assert!(argv.contains(&"--tokenizer".to_string()));
    }
}
