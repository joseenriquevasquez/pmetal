use std::path::{Path, PathBuf};

use pmetal_mlx::{Array, Dtype};

fn tensor_to_f32_vec(name: &str, tensor: &Array) -> anyhow::Result<Option<Vec<f32>>> {
    let materialized = tensor.clone();
    materialized.eval();

    match materialized.dtype() {
        Dtype::Float32 => Ok(Some(materialized.as_slice::<f32>().to_vec())),
        Dtype::Float16 | Dtype::Bfloat16 => {
            let float32 = materialized.as_dtype(Dtype::Float32.as_i32());
            float32.eval();
            Ok(Some(float32.as_slice::<f32>().to_vec()))
        }
        other => {
            tracing::debug!("Skipping non-float tensor {name} with dtype {:?}", other);
            Ok(None)
        }
    }
}

/// Run model quantization.
pub(crate) async fn run_quantization(
    model_path: &str,
    output_path: &str,
    imatrix_path: Option<&str>,
    method: crate::QuantizeMethod,
    kl_calibrate: bool,
    target_bpw: Option<f32>,
    kl_threshold: f64,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgufBuilder,
        dynamic::{
            CalibrationMap, DynamicQuantizationConfig, DynamicQuantizer, KlCalibrationConfig,
        },
        imatrix::IMatrix,
        quantize::quantize,
    };

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method.as_str());
    if let Some(imp) = imatrix_path {
        println!("IMatrix:  {}", imp);
    }
    if kl_calibrate {
        println!("KL Calib: enabled (threshold={:.4})", kl_threshold);
        if let Some(bpw) = target_bpw {
            println!("Target BPW: {:.2}", bpw);
        }
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    let resolved_model_path: PathBuf =
        if model_path.contains('/') && !PathBuf::from(model_path).exists() {
            tracing::info!("Resolving HuggingFace model: {}", model_path);
            pmetal_hub::download_model(model_path, None, None).await?
        } else {
            PathBuf::from(model_path)
        };

    // 1. Load IMatrix if provided
    let imatrix = if let Some(path) = imatrix_path {
        tracing::info!("Loading IMatrix from {}", path);
        Some(IMatrix::load(Path::new(path))?)
    } else {
        None
    };

    // 2. Initialize quantizer
    let quantizer = if let Some(base_type) = method.to_ggml_type() {
        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        DynamicQuantizer::new(config, None)
    } else {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    };

    // 3. Load Model Weights
    tracing::info!("Scanning model weights from {:?}...", resolved_model_path);
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    tracing::info!("Loaded {} tensors", weights.len());

    // 4. Detect Architecture
    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string();

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" => "llama".to_string(),
                            "MistralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" => "qwen2".to_string(),
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            _ => {
                                tracing::warn!(
                                    "Unknown architecture '{}', defaulting to 'llama'",
                                    arch_str
                                );
                                "llama".to_string()
                            }
                        };
                        tracing::info!(
                            "Detected architecture: {} (from {})",
                            architecture,
                            arch_str
                        );
                    }
                }
            }
        }
    } else {
        tracing::warn!("config.json not found, defaulting architecture to 'llama'");
    }

    // 5. KL-divergence calibration pass (optional)
    let mut float_cache: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)> =
        std::collections::HashMap::new();
    let calibration_map: CalibrationMap;

    if kl_calibrate {
        println!(
            "Running KL calibration pass over {} tensors...",
            weights.len()
        );

        let mut tensor_data: Vec<(String, Vec<f32>, Vec<i32>)> = Vec::new();
        let mut sorted_names: Vec<_> = weights.keys().cloned().collect();
        sorted_names.sort();

        for name in &sorted_names {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} missing after key listing", name))?;

            let data_f32 = match tensor_to_f32_vec(name, tensor)? {
                Some(data) => data,
                None => continue,
            };

            let shape_i32: Vec<i32> = tensor.shape().iter().map(|&d| d as i32).collect();
            float_cache.insert(name.clone(), (data_f32.clone(), shape_i32.clone()));
            tensor_data.push((name.clone(), data_f32, shape_i32));
        }

        let kl_config = KlCalibrationConfig {
            kl_threshold,
            target_bpw,
            ..Default::default()
        };

        calibration_map = quantizer.calibrate_all(&tensor_data, &kl_config);

        let tensor_sizes: Vec<(String, usize)> = tensor_data
            .iter()
            .map(|(n, d, _)| (n.clone(), d.len()))
            .collect();
        let summary = quantizer.summarize_calibration(&calibration_map, &tensor_sizes);
        println!(
            "Calibration complete: {} tensors, avg KL={:.6}, worst={} ({:.6}), est. BPW={:.2}",
            summary.total_tensors,
            summary.avg_kl_score,
            summary.worst_tensor,
            summary.max_kl_score,
            summary.estimated_bpw,
        );
        let mut type_vec: Vec<_> = summary.type_counts.iter().collect();
        type_vec.sort_by_key(|(t, _)| format!("{:?}", t));
        for (dtype, count) in type_vec {
            println!("  {:?}: {} tensors", dtype, count);
        }
        println!();
    } else {
        calibration_map = CalibrationMap::new();
    }

    // 6. Initialize GGUF Builder
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");

    // 7. Quantize and Write
    tracing::info!("Starting quantization...");

    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let shape_u64: Vec<u64>;
        let data_f32: Vec<f32>;

        if let Some((cached_data, cached_shape)) = float_cache.remove(name) {
            shape_u64 = cached_shape.iter().map(|&d| d as u64).collect();
            data_f32 = cached_data;
        } else {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in loaded weights", name))?;

            let shape = tensor.shape();
            shape_u64 = shape.iter().map(|&d| d as u64).collect();

            data_f32 = match tensor_to_f32_vec(name, tensor)? {
                Some(data) => data,
                None => continue,
            };
        }

        let target_type = if calibration_map.is_empty() {
            quantizer.get_tensor_type(name, &shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(name, &shape_u64, &calibration_map)
        };

        tracing::info!("Quantizing {} to {:?}", name, target_type);
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", name, e))?;

        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    // 8. Write GGUF output
    let validated_output = crate::validate_output_path(output_path, "quantization output")?;
    let mut file = std::fs::File::create(&validated_output)?;
    builder.write(&mut file)?;

    println!("Quantization complete!");
    Ok(())
}

// ── MLX safetensors path ──────────────────────────────────────────────────────

/// Run MLX-format safetensors quantization with per-tensor quality-based bit allocation.
///
/// `output_path` is treated as a directory.  The directory is created if it
/// does not exist.  Inside it the function writes:
/// - `model.safetensors`   — quantized weights in MLX affine format
/// - `config.json`         — source config + injected `quantization_config`
/// - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
///   `merges.txt`, `vocab.json`, `tokenizer.model` (copied if present)
pub(crate) async fn run_quantization_mlx(
    model_path: &str,
    output_path: &str,
    default_bits: i32,
    group_size: i32,
    target_bpw: Option<f32>,
) -> anyhow::Result<()> {
    use pmetal_bridge::mlx_quant;

    println!("========================================");
    println!("  PMetal MLX Safetensors Quantization");
    println!("========================================");
    println!("Model:      {}", model_path);
    println!("Output:     {}", output_path);
    println!("Bits:       {}", default_bits);
    println!("Group size: {}", group_size);
    if let Some(bpw) = target_bpw {
        println!("Target BPW: {:.2}", bpw);
    }
    println!("========================================\n");

    // 1. Resolve HuggingFace model ID to local path.
    let resolved_model_path: std::path::PathBuf =
        if model_path.contains('/') && !std::path::PathBuf::from(model_path).exists() {
            println!("Resolving HuggingFace model: {}", model_path);
            pmetal_hub::download_model(model_path, None, None).await?
        } else {
            std::path::PathBuf::from(model_path)
        };

    // 2. Load all weights as InlineArray (stays on GPU, avoids f32 copy).
    println!("Loading weights...");
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    println!("Loaded {} tensors", weights.len());

    // 3. Run the full pipeline: evaluate quality → allocate bits → quantize → save.
    let output_dir = std::path::PathBuf::from(output_path);
    let effective_bpw = target_bpw.unwrap_or(default_bits as f32);
    let source_config = resolved_model_path.join("config.json");

    println!(
        "Evaluating tensor quality and allocating bits (target BPW={:.2})...",
        effective_bpw
    );

    let assignments = mlx_quant::quantize_model(
        &weights,
        &source_config,
        &output_dir,
        effective_bpw,
        group_size,
        mlx_quant::DEFAULT_BITS_CANDIDATES,
        &[], // no extra critical tensor patterns
    )
    .map_err(|e| anyhow::anyhow!("Quantization failed: {}", e))?;

    // 4. Print allocation summary.
    let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut total_params: usize = 0;
    let mut total_weighted_bits: f64 = 0.0;
    for a in &assignments {
        *counts.entry(a.bits).or_insert(0) += 1;
        total_params += a.param_count;
        let bits = if a.bits == 0 { 16 } else { a.bits };
        total_weighted_bits += a.param_count as f64 * bits as f64;
    }
    let final_bpw = if total_params > 0 {
        total_weighted_bits / total_params as f64
    } else {
        0.0
    };

    println!("\nBit allocation summary:");
    let mut bit_keys: Vec<_> = counts.keys().collect();
    bit_keys.sort();
    for &bits in &bit_keys {
        let count = counts[&bits];
        if *bits == 0 {
            println!("  bf16 (passthrough): {} tensors", count);
        } else {
            println!("  Q{}: {} tensors", *bits, count);
        }
    }
    println!("Effective BPW: {:.3}", final_bpw);
    println!("Total tensors: {}", assignments.len());

    // 5. Copy tokenizer files from source to output.
    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
    ];
    for fname in &tokenizer_files {
        let src = resolved_model_path.join(fname);
        if src.exists() {
            let dst = output_dir.join(fname);
            std::fs::copy(&src, &dst).ok();
        }
    }

    println!("\nMLX quantization complete!");
    println!("Output: {}", output_dir.display());
    Ok(())
}
