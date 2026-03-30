//! Native inference — all models through pmetal-bridge, zero mlx-rs.
//!
//! Each architecture has its own `run_{arch}` function that owns the full
//! pipeline: load config → load weights → prefill → decode.  The module is
//! intentionally self-contained; the only external dependencies are
//! `pmetal_bridge` and `serde_json`.

use std::path::Path;

use pmetal_bridge::{InlineArray, turboquant::TurboQuantConfig};

// ============================================================================
// Architecture enum
// ============================================================================

/// Supported model architectures for native inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeArch {
    /// Dense Qwen3 (`model_type = "qwen3"`).
    Qwen3,
    /// Qwen3.5 dense or MoE (`model_type = "qwen3_5"` / `"qwen3_5_text"`).
    Qwen3_5,
    /// Llama 4 (`model_type = "llama4"` / `"llama4_text"`).
    Llama4,
    /// DeepSeek V3/R1 (`model_type = "deepseek_v3"`).
    DeepSeek,
    /// GPT-OSS (`model_type = "gpt_oss"`).
    GptOss,
}

// ============================================================================
// Architecture detection
// ============================================================================

/// Detect architecture from `config.json`.
///
/// Checks `text_config.model_type` first (multi-modal configs), then falls
/// back to the top-level `model_type` field.
pub fn detect_arch(model_path: &Path) -> Option<NativeArch> {
    let data = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;

    let mt = v
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .or_else(|| v.get("model_type"))
        .and_then(|mv| mv.as_str())?;

    match mt {
        "qwen3" | "qwen3dense" => Some(NativeArch::Qwen3),
        "qwen3_5" | "qwen3_5_text" => Some(NativeArch::Qwen3_5),
        "llama4" | "llama4_text" => Some(NativeArch::Llama4),
        "deepseek_v3" => Some(NativeArch::DeepSeek),
        "gpt_oss" => Some(NativeArch::GptOss),
        _ => None,
    }
}

// ============================================================================
// Output type
// ============================================================================

/// Output produced by a native generation run.
pub struct NativeGenerationOutput {
    /// All token IDs: prompt + generated.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excludes prompt).
    pub num_generated: usize,
    /// True when generation stopped because `on_token` returned `false`
    /// (i.e. an EOS or stop token was hit).
    pub stopped_by_token: bool,
    /// True when generation stopped because `max_tokens` was exhausted.
    pub stopped_by_length: bool,
}

#[derive(Debug, Clone)]
pub struct MlxLmBenchmarkTrial {
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub peak_memory_gb: f64,
}

// ============================================================================
// Top-level dispatch
// ============================================================================

/// Run native inference end-to-end: load → prefill → generate.
///
/// `on_token(id)` is called for every generated token; return `false` to stop
/// early (EOS, stop token, or user cancel).
///
/// Returns `Err` if the architecture is unsupported or if loading fails.
pub fn run_native_inference(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    turboquant: Option<TurboQuantConfig>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;

    match arch {
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => run_qwen3(
            model_path,
            input_ids,
            max_tokens,
            temperature,
            turboquant,
            &mut on_token,
        ),
        NativeArch::Llama4 => {
            if turboquant.is_some() {
                return Err(
                    "TurboQuant native cache is only supported for Qwen3/Qwen3.5".to_string(),
                );
            }
            run_llama4(
                model_path,
                input_ids,
                max_tokens,
                temperature,
                &mut on_token,
            )
        }
        NativeArch::DeepSeek => {
            if turboquant.is_some() {
                return Err(
                    "TurboQuant native cache is only supported for Qwen3/Qwen3.5".to_string(),
                );
            }
            run_deepseek(
                model_path,
                input_ids,
                max_tokens,
                temperature,
                &mut on_token,
            )
        }
        NativeArch::GptOss => {
            if turboquant.is_some() {
                return Err(
                    "TurboQuant native cache is only supported for Qwen3/Qwen3.5".to_string(),
                );
            }
            run_gpt_oss(
                model_path,
                input_ids,
                max_tokens,
                temperature,
                &mut on_token,
            )
        }
    }
}

// ============================================================================
// Shared helper
// ============================================================================

/// Convert a `&[u32]` prompt to an `InlineArray` of shape `[1, T]` (i32 dtype).
///
/// All `forward_step` implementations expect `[B, T]` int32 token IDs.
fn prompt_to_input(input_ids: &[u32]) -> InlineArray {
    let ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
    InlineArray::from_i32_slice(&ids_i32).reshape(&[1, ids_i32.len() as i32])
}

/// Extract last-token logits from a `[B, T, vocab]` logits tensor.
///
/// Returns a `[1, vocab]` slice, ready for `sample_token`.
fn last_token_logits(logits: &InlineArray) -> InlineArray {
    // logits: [B, T, vocab]
    let b = logits.dim(0);
    let t = logits.dim(1);
    let vocab = logits.dim(2);
    // Slice out the last sequence position: [B, T-1:T, vocab] → reshape [B, vocab].
    logits
        .slice(&[0, t - 1, 0], &[b, t, vocab])
        .reshape(&[b, vocab])
}

fn sample_first_token(
    last_logits: &InlineArray,
    temperature: f32,
    sample_token: impl Fn(&InlineArray, f32) -> InlineArray,
) -> u32 {
    let mut tok_arr = sample_token(last_logits, temperature);
    tok_arr.eval();
    tok_arr.item_u32()
}

fn finish_with_bridge_generate(
    prompt: &[u32],
    first_tok: u32,
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> bool,
    generate_tail: impl FnOnce(&mut dyn FnMut(u32) -> bool) -> Vec<u32>,
) -> NativeGenerationOutput {
    let prompt_len = prompt.len();
    let mut all_tokens = prompt.to_vec();
    all_tokens.push(first_tok);

    if !on_token(first_tok) {
        return NativeGenerationOutput {
            token_ids: all_tokens,
            num_generated: 1,
            stopped_by_token: true,
            stopped_by_length: false,
        };
    }

    let remaining = max_tokens.saturating_sub(1);
    let generated_tail = generate_tail(on_token);
    let stopped_by_token = generated_tail.len() < remaining;
    all_tokens.extend(generated_tail);
    let num_generated = all_tokens.len() - prompt_len;

    NativeGenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token,
        stopped_by_length: !stopped_by_token && num_generated >= max_tokens,
    }
}

// ============================================================================
// Qwen3 / Qwen3.5
// ============================================================================

fn run_qwen3(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    turboquant: Option<TurboQuantConfig>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::qwen3_native;

    let config = qwen3_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] Qwen3{}: {} layers, hidden={}{}",
        if config.is_moe() { " MoE" } else { "" },
        config.num_hidden_layers,
        config.hidden_size,
        if config.is_qwen3_dense() {
            " (Qwen3 dense)"
        } else {
            ""
        },
    );

    let t0 = std::time::Instant::now();
    let weights = qwen3_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = build_qwen3_cache(&weights, turboquant);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = qwen3_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]
    let first_tok = sample_first_token(&last_logits, temperature, qwen3_native::sample_token);

    Ok(finish_with_bridge_generate(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |on_token| {
            // Keep the Rust/bridge decode loop canonical until the monolithic
            // C++ path demonstrably outperforms it on real models.
            qwen3_native::generate(
                &weights,
                &mut cache,
                first_tok,
                max_tokens.saturating_sub(1),
                temperature,
                |tok| on_token(tok),
            )
        },
    ))
}

fn build_qwen3_cache(
    weights: &pmetal_bridge::qwen3_native::NativeWeights,
    turboquant: Option<TurboQuantConfig>,
) -> pmetal_bridge::qwen3_native::NativeCache {
    match turboquant {
        Some(config) => {
            pmetal_bridge::qwen3_native::NativeCache::new_with_turboquant(weights, Some(config))
        }
        None => pmetal_bridge::qwen3_native::NativeCache::new_empty(weights),
    }
}

/// Benchmark full prompt + generation throughput using the same workload shape
/// as `mlx_lm.benchmark`: fixed prompt token ids, one warmup, EOS disabled, and
/// repeated generations from a fresh cache.
pub fn benchmark_native_mlx_lm(
    model_path: &Path,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<TurboQuantConfig>,
    num_trials: usize,
) -> Result<Vec<MlxLmBenchmarkTrial>, String> {
    use pmetal_bridge::qwen3_native;

    if prompt_ids.is_empty() {
        return Err("MLX-LM parity benchmark requires prompt_tokens > 0".to_string());
    }
    if generation_tokens == 0 {
        return Err("MLX-LM parity benchmark requires generation_tokens > 0".to_string());
    }

    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;
    if !matches!(arch, NativeArch::Qwen3 | NativeArch::Qwen3_5) {
        return Err(
            "native MLX-LM parity benchmark is currently only implemented for Qwen3/Qwen3.5"
                .to_string(),
        );
    }

    let config = qwen3_native::load_config(model_path)?;
    let weights = qwen3_native::load_model(model_path, &config)?;
    if num_trials == 0 {
        return Ok(Vec::new());
    }

    let prompt = prompt_to_input(prompt_ids);
    let run_once = || {
        pmetal_bridge::inline_array::reset_peak_memory();
        let mut cache = build_qwen3_cache(&weights, turboquant);
        let prompt_tic = std::time::Instant::now();
        let logits = qwen3_native::forward_step(&weights, &prompt, &mut cache);
        let last_logits = last_token_logits(&logits);
        let first_tok = sample_first_token(&last_logits, 0.0, qwen3_native::sample_token);
        let prompt_time = prompt_tic.elapsed().as_secs_f64();

        let generation_tic = std::time::Instant::now();
        if generation_tokens > 1 {
            let current_y = qwen3_native::prime_generation_preserve_peak_silent(
                &weights,
                &mut cache,
                first_tok,
                0.0,
            );
            let generated_tail = qwen3_native::generate_from_primed_sample_silent(
                &weights,
                &mut cache,
                current_y,
                generation_tokens - 1,
                0.0,
                |_| true,
            );
            debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        } else {
            pmetal_bridge::inline_array::synchronize();
        }
        let generation_time = generation_tic.elapsed().as_secs_f64();

        MlxLmBenchmarkTrial {
            prompt_tps: prompt_ids.len() as f64 / prompt_time.max(f64::MIN_POSITIVE),
            generation_tps: generation_tokens as f64 / generation_time.max(f64::MIN_POSITIVE),
            peak_memory_gb: pmetal_bridge::inline_array::get_peak_memory() as f64 / 1e9,
        }
    };

    let _warmup = run_once();

    let mut trials = Vec::with_capacity(num_trials);
    for _ in 0..num_trials {
        trials.push(run_once());
    }

    Ok(trials)
}

// ============================================================================
// Llama 4
// ============================================================================

fn run_llama4(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::llama4_native;

    let config = llama4_native::load_config(model_path)?;
    let tc = config.text();
    eprintln!(
        "[NATIVE] Llama4 MoE: {} layers, hidden={}, experts={}/tok={}",
        tc.num_hidden_layers, tc.hidden_size, tc.num_local_experts, tc.num_experts_per_tok,
    );

    let t0 = std::time::Instant::now();
    let weights = llama4_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = llama4_native::NativeCache::new_empty(&weights);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = llama4_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]
    let first_tok = sample_first_token(&last_logits, temperature, llama4_native::sample_token);

    Ok(finish_with_bridge_generate(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |on_token| {
            llama4_native::generate(
                &weights,
                &mut cache,
                first_tok,
                max_tokens.saturating_sub(1),
                temperature,
                |tok| on_token(tok),
            )
        },
    ))
}

// ============================================================================
// DeepSeek V3/R1
// ============================================================================

fn run_deepseek(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::deepseek_native;

    let config = deepseek_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] DeepSeek V3: {} layers, hidden={}, experts={}/tok={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.n_routed_experts.unwrap_or(0),
        config.num_experts_per_tok,
    );

    let t0 = std::time::Instant::now();
    let weights = deepseek_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let num_layers = config.num_hidden_layers as usize;
    let mut cache = deepseek_native::NativeCache::new_empty(num_layers);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = deepseek_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]
    let first_tok = sample_first_token(&last_logits, temperature, deepseek_native::sample_token);

    Ok(finish_with_bridge_generate(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |on_token| {
            deepseek_native::generate(
                &weights,
                &mut cache,
                first_tok,
                max_tokens.saturating_sub(1),
                temperature,
                |tok| on_token(tok),
            )
        },
    ))
}

// ============================================================================
// GPT-OSS
// ============================================================================

fn run_gpt_oss(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::gpt_oss_native;

    let config = gpt_oss_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] GPT-OSS: {} layers, hidden={}, experts={}/tok={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_local_experts,
        config.experts_per_tok(),
    );

    let t0 = std::time::Instant::now();
    let weights = gpt_oss_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = gpt_oss_native::NativeCache::new_empty(&weights);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = gpt_oss_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]
    let first_tok = sample_first_token(&last_logits, temperature, gpt_oss_native::sample_token);

    Ok(finish_with_bridge_generate(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |on_token| {
            gpt_oss_native::generate(
                &weights,
                &mut cache,
                first_tok,
                max_tokens.saturating_sub(1),
                temperature,
                |tok| on_token(tok),
            )
        },
    ))
}
