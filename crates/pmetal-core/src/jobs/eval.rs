//! `pmetal eval` — perplexity / accuracy evaluation.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Eval", subcommand = "eval")]
#[serde(rename_all = "snake_case")]
pub struct EvalSpec {
    #[job(label = "Model", group = "Model", argv = "--model", kind = "model_picker", required)]
    #[serde(default)]
    pub model: String,

    #[job(label = "Dataset", group = "Data", argv = "--dataset", kind = "dataset_picker", required)]
    #[serde(default)]
    pub dataset: String,

    #[job(label = "LoRA Adapter", group = "Model", argv = "--lora", kind = "path")]
    #[serde(default)]
    pub lora: Option<String>,

    #[job(label = "Max Seq Len", group = "Eval", argv = "--max-seq-len", default_int = 1024)]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[job(label = "Num Samples", group = "Eval", argv = "--num-samples", help = "0 = all", default_int = 0)]
    #[serde(default)]
    pub num_samples: usize,

    #[job(label = "JSON Output", group = "Output", argv = "--json", flag, default_bool = false)]
    #[serde(default)]
    pub json: bool,
}

impl Default for EvalSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            lora: None,
            max_seq_len: default_max_seq_len(),
            num_samples: 0,
            json: false,
        }
    }
}

impl EvalSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

fn default_max_seq_len() -> usize {
    1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let mut spec = EvalSpec::default();
        spec.model = "m".into();
        spec.dataset = "d".into();
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }
}
