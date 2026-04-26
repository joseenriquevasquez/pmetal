//! Benchmark ANE inference using the SDK directly.
//!
//! ```sh
//! cargo run -p pmetal --example inference_ane -- \
//!     --model Qwen/Qwen3-0.6B --prompt "Explain quantum physics."
//! ```

use std::env;

use pmetal::data::Tokenizer;
use pmetal::hub::resolve_model_path;
use pmetal::metal::ane::inference::{AneInferenceConfig, AneInferenceEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model_id = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("Qwen/Qwen3-0.6B");
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("Once upon a time,");
    let max_tokens: usize = 128;

    println!("========================================");
    println!("  PMetal ANE Inference Benchmark");
    println!("========================================");
    println!("Model:  {model_id}");
    println!("Device: ANE (Neural Engine)");
    println!("----------------------------------------\n");

    // Resolve model
    let model_path = resolve_model_path(model_id, None, None).await?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;
    let input_ids = tokenizer.encode(prompt)?;
    let prompt_len = input_ids.len();

    // Parse config.json for model dimensions
    let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)?;

    let dim = config["hidden_size"].as_u64().unwrap() as usize;
    let hidden_dim = config["intermediate_size"].as_u64().unwrap() as usize;
    let n_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let n_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let n_kv_heads = config["num_key_value_heads"]
        .as_u64()
        .unwrap_or(n_heads as u64) as usize;
    let rope_theta = config["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32;
    let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let head_dim = config["head_dim"].as_u64().map(|v| v as usize);

    // Configure ANE engine
    let ane_config = AneInferenceConfig {
        dim,
        hidden_dim,
        n_heads,
        n_kv_heads,
        n_layers,
        vocab_size,
        max_seq_len: prompt_len + max_tokens + 64,
        ane_seq_len: None,
        temperature: 0.0, // Greedy for benchmarking
        top_k: 1,
        max_tokens,
        eos_token_id: tokenizer.eos_token_id(),
        rope_theta,
        rms_norm_eps,
        head_dim,
        ..Default::default()
    };

    let mut engine = AneInferenceEngine::new(ane_config, prompt_len)?;

    // Load weights: SafeTensors (single or sharded) or raw f32 binary
    let safetensors_single = model_path.join("model.safetensors");
    let safetensors_multi = model_path.join("model-00001-of-00002.safetensors");
    let safetensors_index = model_path.join("model.safetensors.index.json");
    let weights_bin = model_path.join("model.bin");

    if safetensors_single.exists() {
        engine.load_weights_safetensors(&safetensors_single)?;
    } else if safetensors_index.exists() || safetensors_multi.exists() {
        engine.load_weights_safetensors(&model_path)?;
    } else if weights_bin.exists() {
        let weight_data = std::fs::read(&weights_bin)?;
        if weight_data.len() % 4 != 0 {
            return Err("model.bin size must be a multiple of 4 bytes".into());
        }
        #[allow(unsafe_code)]
        let (prefix, weights, suffix) = unsafe { weight_data.align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("model.bin data is not properly aligned for f32".into());
        }
        engine.load_weights_flat(weights);
    } else {
        return Err(format!(
            "No weight files found in {:?}. Expected model.safetensors or model.bin.",
            model_path
        )
        .into());
    }

    // Check for LoRA adapter
    if model_path.join("adapter_config.json").exists()
        && (model_path.join("lora_weights.safetensors").exists()
            || model_path.join("adapter_model.safetensors").exists())
    {
        engine.load_lora_adapter(&model_path)?;
    }

    // Compile ANE kernels
    engine.compile_kernels()?;

    // Generate
    let start = std::time::Instant::now();
    let output_ids = engine.generate_cached(&input_ids)?;
    let elapsed = start.elapsed();

    let generated: Vec<u32> = output_ids[prompt_len..].to_vec();
    let text = tokenizer.decode(&generated)?;

    println!("Response:\n{text}\n");
    println!("---");
    println!(
        "Throughput: {:.1} tok/s ({} tokens)",
        generated.len() as f64 / elapsed.as_secs_f64(),
        generated.len()
    );
    println!("========================================\n");

    Ok(())
}
