//! `pmetal merge` — multi-model merging (SLERP / TIES / DARE / etc.).

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Merge", subcommand = "merge")]
#[serde(rename_all = "snake_case")]
pub struct MergeSpec {
    #[job(label = "Model A", group = "Models", argv = "--model-a", kind = "model_picker", required)]
    #[serde(default)]
    pub model_a: String,

    #[job(label = "Model B", group = "Models", argv = "--model-b", kind = "model_picker", required)]
    #[serde(default)]
    pub model_b: String,

    #[job(label = "Output Dir", group = "Output", argv = "--output", kind = "path", required)]
    #[serde(default)]
    pub output: String,

    #[job(label = "Method", group = "Method", argv = "--method", kind = "enum",
          enum_options = ["linear", "slerp", "ties", "dare_ties", "dare_linear",
                          "task_arithmetic", "della", "breadcrumbs", "model_stock",
                          "nearswap", "passthrough"],
          default = "slerp")]
    #[serde(default = "default_method")]
    pub method: String,

    #[job(label = "Base Model", group = "Models", argv = "--base", kind = "model_picker")]
    #[serde(default)]
    pub base: Option<String>,

    #[job(label = "SLERP t", group = "Method", argv = "--t", min = 0.0, max = 1.0, default_float = 0.5)]
    #[serde(default = "default_t")]
    pub t: f32,

    #[job(label = "Weight A", group = "Method", argv = "--weight-a", min = 0.0, max = 10.0, default_float = 0.5)]
    #[serde(default = "default_weight")]
    pub weight_a: f32,

    #[job(label = "Weight B", group = "Method", argv = "--weight-b", min = 0.0, max = 10.0, default_float = 0.5)]
    #[serde(default = "default_weight")]
    pub weight_b: f32,

    #[job(label = "Density", group = "Method", argv = "--density", min = 0.0, max = 1.0, default_float = 0.5)]
    #[serde(default = "default_density")]
    pub density: f32,

    #[job(label = "Output Dtype", group = "Output", argv = "--dtype", kind = "enum",
          enum_options = ["float32", "float16", "bfloat16"], default = "bfloat16")]
    #[serde(default = "default_dtype")]
    pub dtype: String,
}

impl Default for MergeSpec {
    fn default() -> Self {
        Self {
            model_a: String::new(),
            model_b: String::new(),
            output: String::new(),
            method: default_method(),
            base: None,
            t: default_t(),
            weight_a: default_weight(),
            weight_b: default_weight(),
            density: default_density(),
            dtype: default_dtype(),
        }
    }
}

impl MergeSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }
}

fn default_method() -> String {
    "slerp".to_string()
}
fn default_t() -> f32 {
    0.5
}
fn default_weight() -> f32 {
    0.5
}
fn default_density() -> f32 {
    0.5
}
fn default_dtype() -> String {
    "bfloat16".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let mut spec = MergeSpec::default();
        spec.model_a = "a".into();
        spec.model_b = "b".into();
        spec.output = "o".into();
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model-a".to_string()));
        assert!(argv.contains(&"--model-b".to_string()));
        assert!(argv.contains(&"--output".to_string()));
    }
}
