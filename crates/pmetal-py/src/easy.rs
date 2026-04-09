//! Top-level easy API functions for Python.
//!
//! These call core SDK crates directly rather than going through the `pmetal`
//! umbrella crate's former `easy` module.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::runtime_err;

// TrainableModel trait must be in scope for DynamicLoraModel methods
use pmetal_lora::TrainableModel as _;

/// Fine-tune a model with sensible defaults.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     dataset_path: Path to JSONL training dataset
///     lora_r: LoRA rank (default 16)
///     lora_alpha: LoRA alpha (default 32.0)
///     epochs: Number of training epochs (default 3)
///     learning_rate: Learning rate (default 2e-4)
///     batch_size: Batch size (default 4)
///     max_seq_len: Maximum sequence length (default 2048)
///     output: Output directory (default "./output")
///
/// Returns:
///     dict with keys: final_loss, total_steps, total_tokens, output_dir, lora_weights_path
#[pyfunction]
#[pyo3(signature = (
    model_id,
    dataset_path,
    lora_r=16,
    lora_alpha=32.0,
    epochs=3,
    learning_rate=2e-4,
    batch_size=4,
    max_seq_len=2048,
    output="./output",
))]
#[allow(clippy::too_many_arguments)]
pub fn finetune<'py>(
    py: Python<'py>,
    model_id: &str,
    dataset_path: &str,
    lora_r: usize,
    lora_alpha: f32,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    max_seq_len: usize,
    output: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let model_id = model_id.to_string();
    let dataset_path = dataset_path.to_string();
    let output = output.to_string();

    let result = py
        .detach(move || {
            crate::hub::shared_runtime().block_on(async {
                use pmetal_trainer::orchestrator;

                let job_config = orchestrator::TrainingJobConfig {
                    model_id: model_id.clone(),
                    dataset: dataset_path,
                    eval_dataset: None,
                    output_dir: output,
                    lora: pmetal_core::LoraConfig {
                        r: lora_r,
                        alpha: lora_alpha,
                        ..Default::default()
                    },
                    qlora: None,
                    training: pmetal_core::TrainingConfig {
                        learning_rate,
                        batch_size,
                        num_epochs: epochs,
                        max_seq_len,
                        ..Default::default()
                    },
                    columns: None,
                    dispatch: orchestrator::DispatchConfig::default(),
                    config_path: None,
                    log_metrics: None,
                    resume: false,
                    seed: 42,
                    emit_console_output: false,
                };

                let result = orchestrator::run_training(job_config, None, vec![])
                    .await
                    .map_err(|e| e.to_string())?;

                Ok::<_, String>((
                    result.final_loss,
                    result.total_steps,
                    result.total_tokens,
                    result.output_dir.to_string_lossy().to_string(),
                    result.lora_weights_path.to_string_lossy().to_string(),
                ))
            })
        })
        .map_err(runtime_err)?;

    let dict = PyDict::new(py);
    dict.set_item("final_loss", result.0)?;
    dict.set_item("total_steps", result.1)?;
    dict.set_item("total_tokens", result.2)?;
    dict.set_item("output_dir", result.3)?;
    dict.set_item("lora_weights_path", result.4)?;
    Ok(dict)
}

/// Run inference with a model.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     prompt: Text prompt
///     lora: Optional path to LoRA weights
///     max_tokens: Maximum tokens to generate (default 256)
///     temperature: Sampling temperature (default 0.7)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Generated text string
#[pyfunction]
#[pyo3(signature = (model_id, prompt, lora=None, max_tokens=256, temperature=0.7, seed=None))]
pub fn infer(
    py: Python<'_>,
    model_id: &str,
    prompt: &str,
    lora: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
) -> PyResult<String> {
    let model_id = model_id.to_string();
    let prompt = prompt.to_string();
    let lora = lora.map(String::from);

    py.detach(move || {
        crate::hub::shared_runtime().block_on(async {
            use pmetal_data::Tokenizer;
            use pmetal_data::chat_templates::{Message, detect_chat_template};
            use pmetal_hub::resolve_model_path;
            use pmetal_models::{DynamicModel, GenerationConfig, generate_cached_async};

            let model_path = resolve_model_path(&model_id)
                .await
                .map_err(|e| e.to_string())?;
            let tokenizer = Tokenizer::from_model_dir(&model_path)
                .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

            // Chat template
            let template = detect_chat_template(&model_path, &model_id);
            let msgs = vec![Message::user(&prompt)];
            let formatted = template.apply(&msgs).text;
            let input_ids = tokenizer
                .encode_with_special_tokens(&formatted)
                .map_err(|e| e.to_string())?;

            // Gen config — load model defaults, user params override
            let defaults = pmetal_data::inference_config::load_sampling_defaults(
                &model_path,
                None,
                pmetal_data::inference_config::SamplingMode::Auto,
                false,
            );
            let mut gen_config = if temperature < 1e-6 {
                GenerationConfig::greedy(max_tokens)
            } else {
                GenerationConfig::sampling(max_tokens, temperature)
                    .with_top_k(defaults.top_k)
                    .with_top_p(defaults.top_p)
                    .with_min_p(defaults.min_p)
                    .with_repetition_penalty(defaults.repetition_penalty)
            };
            // Collect ALL stop tokens (multi-EOS models like Qwen3)
            let stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
                &model_path,
                &tokenizer,
                Some(template.template_type),
            );
            gen_config = gen_config.with_stop_tokens(stop_tokens);
            if let Some(s) = seed {
                gen_config = gen_config.with_seed(s);
            }

            let max_seq_len = input_ids.len() + max_tokens + 64;

            if let Some(ref lora_path) = lora {
                use pmetal_core::LoraConfig;
                use pmetal_lora::DynamicLoraModel;

                let lora_dir = std::path::Path::new(lora_path.as_str());
                let adapter_dir = if lora_dir.is_dir() {
                    lora_dir
                } else {
                    lora_dir.parent().unwrap_or(lora_dir)
                };
                let lora_config = if let Ok(cfg_str) =
                    std::fs::read_to_string(adapter_dir.join("adapter_config.json"))
                {
                    if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) {
                        let r = cfg["r"].as_u64().unwrap_or(16) as usize;
                        let alpha = cfg["alpha"]
                            .as_f64()
                            .or_else(|| cfg["lora_alpha"].as_f64())
                            .unwrap_or(32.0) as f32;
                        LoraConfig {
                            r,
                            alpha,
                            ..LoraConfig::default()
                        }
                    } else {
                        LoraConfig::default()
                    }
                } else {
                    LoraConfig::default()
                };

                let mut model = DynamicLoraModel::from_pretrained(&model_path, lora_config)
                    .map_err(|e| e.to_string())?;
                model
                    .load_lora_weights(lora_path)
                    .map_err(|e| e.to_string())?;

                let mut cache = model
                    .create_cache(max_seq_len)
                    .ok_or_else(|| "Model does not support KV cache".to_string())?;

                let output = generate_cached_async(
                    |input, cache| {
                        model
                            .forward_with_cache(input, None, Some(cache))
                            .map_err(|e| pmetal_mlx::Exception::custom(e.to_string()))
                    },
                    &input_ids,
                    gen_config,
                    &mut cache,
                )
                .map_err(|e| e.to_string())?;

                let generated_tokens = &output.token_ids[input_ids.len()..];
                let text = tokenizer
                    .decode(generated_tokens)
                    .map_err(|e| e.to_string())?;
                Ok::<_, String>(text)
            } else {
                let mut model = DynamicModel::load(&model_path).map_err(|e| e.to_string())?;
                let mut cache = model.create_cache(max_seq_len);
                let mut mamba_cache = model.create_mamba_cache();

                let output = generate_cached_async(
                    |input, cache| {
                        model.forward_with_hybrid_cache(
                            input,
                            None,
                            Some(cache),
                            mamba_cache.as_mut(),
                        )
                    },
                    &input_ids,
                    gen_config,
                    &mut cache,
                )
                .map_err(|e| e.to_string())?;

                let generated_tokens = &output.token_ids[input_ids.len()..];
                let text = tokenizer
                    .decode(generated_tokens)
                    .map_err(|e| e.to_string())?;
                Ok(text)
            }
        })
    })
    .map_err(runtime_err)
}
