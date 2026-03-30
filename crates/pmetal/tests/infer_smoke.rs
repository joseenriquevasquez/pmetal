#![cfg(target_os = "macos")]

use pmetal::{
    data::{
        Tokenizer,
        chat_templates::{Message, detect_chat_template},
    },
    models::DynamicModel,
};
use pmetal_bridge::compat::{Array, ops};
use std::path::PathBuf;
use std::process::Command;

fn cached_qwen3_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca");
    path.is_dir().then_some(path)
}

fn cached_qwen35_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/cb9632e46f3232cffd569f81efa81dfceddb2c48");
    path.is_dir().then_some(path)
}

#[test]
#[ignore = "requires local Metal hardware, cached Qwen3 weights, and unsandboxed cargo test"]
fn qwen3_cached_infer_smoke_exits_successfully() {
    let Some(model_path) = cached_qwen3_path() else {
        eprintln!("Skipping infer smoke: cached Qwen3 snapshot not found");
        return;
    };

    let binary = env!("CARGO_BIN_EXE_pmetal");
    let output = Command::new(binary)
        .args([
            "infer",
            "--model",
            model_path.to_str().expect("utf-8 model path"),
            "--prompt",
            "test",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .env("RUST_LOG", "warn")
        .output()
        .expect("spawn pmetal infer");

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "pmetal infer smoke failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        );
    }
}

#[test]
#[ignore = "requires local Metal hardware, cached Qwen3.5 weights, and unsandboxed cargo test"]
fn qwen35_cached_first_token_matches_plain_and_cached_paths() {
    let Some(model_path) = cached_qwen35_path() else {
        eprintln!("Skipping infer smoke: cached Qwen3.5 snapshot not found");
        return;
    };

    let tokenizer = Tokenizer::from_model_dir(&model_path).expect("load tokenizer");
    let template = detect_chat_template(&model_path, "unsloth/Qwen3.5-0.8B");
    let formatted = template
        .apply_inference(
            &[Message::user("write a fizzbuzz program in python")],
            true,
            None,
        )
        .text;
    let token_ids = tokenizer
        .encode_with_special_tokens(&formatted)
        .expect("encode prompt");
    let input_ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let input = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

    let mut plain_model = DynamicModel::load(&model_path).expect("load plain model");
    let plain_logits = plain_model.forward(&input, None).expect("plain forward");
    let plain_last = ops::select_axis(&plain_logits, -1, 1);
    let mut plain_next = ops::argmax_axis(&plain_last, -1);
    plain_next.eval();
    let plain_next = plain_next.item::<u32>();

    let mut cached_model = DynamicModel::load(&model_path).expect("load cached model");
    let mut cache = cached_model.create_cache(token_ids.len() + 1);
    let mut mamba_cache = cached_model.create_mamba_cache();
    let cached_logits = cached_model
        .forward_with_hybrid_cache(&input, None, Some(&mut cache), mamba_cache.as_mut())
        .expect("cached forward");
    let cached_last = ops::select_axis(&cached_logits, -1, 1);
    let mut cached_next = ops::argmax_axis(&cached_last, -1);
    cached_next.eval();
    let cached_next = cached_next.item::<u32>();
    let decoded = tokenizer
        .decode(&[plain_next])
        .unwrap_or_else(|_| "<decode failed>".to_string());

    eprintln!(
        "qwen3.5 first token: plain={} cached={} decoded={:?}",
        plain_next, cached_next, decoded
    );

    assert_eq!(
        plain_next, cached_next,
        "first next-token diverged between plain and cached Qwen3.5 paths"
    );
    assert_eq!(plain_next, 8160, "unexpected Qwen3.5 first token");
    assert_eq!(decoded, "Here", "unexpected Qwen3.5 first token decode");
}
