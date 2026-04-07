use pmetal_bridge::InlineArray;
use std::time::Instant;

const MODEL_PATH: &str = "/Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/cb9632e46f3232cffd569f81efa81dfceddb2c48/model.safetensors-00001-of-00001.safetensors";

fn load_w(key: &str) -> InlineArray {
    eprintln!("  Loading: {key}");
    let w = InlineArray::load_safetensors(MODEL_PATH, key)
        .unwrap_or_else(|| panic!("Failed to load {key}"));
    eprintln!("  Loaded: shape ndim={}", w.ndim());
    w
}

#[test]
fn native_weight_forward() {
    let h = 1024i32;

    // Load a few weights NATIVELY (through pmetal-bridge's MLX, not mlx-rs)
    let embed_w = load_w("model.language_model.embed_tokens.weight");
    let ln_w = load_w("model.language_model.layers.0.input_layernorm.weight");
    let gate_w = load_w("model.language_model.layers.0.mlp.gate_proj.weight");
    let up_w = load_w("model.language_model.layers.0.mlp.up_proj.weight");
    let down_w = load_w("model.language_model.layers.0.mlp.down_proj.weight");

    // Transpose projection weights (matching the model's pattern)
    let gate_wt = gate_w.t();
    gate_wt.eval();
    let up_wt = up_w.t();
    up_wt.eval();
    let down_wt = down_w.t();
    down_wt.eval();

    // Eval all
    let e = embed_w.clone();
    e.eval();
    let l = ln_w.clone();
    l.eval();

    eprintln!(
        "Loaded weights natively. gate_wt: ({},{}) dtype={}",
        gate_wt.dim(0),
        gate_wt.dim(1),
        gate_wt.dtype_raw()
    );

    // Load ALL 24 layers of UNIQUE weights (matching real model exactly)
    struct LayerW {
        ln_w: InlineArray,
        ln2_w: InlineArray,
        gate_wt: InlineArray,
        up_wt: InlineArray,
        down_wt: InlineArray,
    }
    let mut all_layers: Vec<LayerW> = Vec::new();
    for i in 0..24 {
        let p = format!("model.language_model.layers.{i}");
        all_layers.push(LayerW {
            ln_w: load_w(&format!("{p}.input_layernorm.weight")),
            ln2_w: load_w(&format!("{p}.post_attention_layernorm.weight")),
            gate_wt: load_w(&format!("{p}.mlp.gate_proj.weight")).t(),
            up_wt: load_w(&format!("{p}.mlp.up_proj.weight")).t(),
            down_wt: load_w(&format!("{p}.mlp.down_proj.weight")).t(),
        });
    }
    // Eval all
    for lw in &mut all_layers {
        lw.ln_w.eval();
        lw.ln2_w.eval();
        lw.gate_wt.eval();
        lw.up_wt.eval();
        lw.down_wt.eval();
    }
    let final_ln = load_w("model.language_model.norm.weight");

    eprintln!(
        "Loaded ALL 24 layers of unique weights. Active: {:.0} MB",
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6
    );

    let run_step = |tok: i32| -> InlineArray {
        let t = InlineArray::from_i32(tok).reshape(&[1, 1]);
        let mut hidden = embed_w.take_axis(&t.squeeze(0), 0).reshape(&[1, 1, h]);
        for lw in &all_layers {
            let normed = hidden.rms_norm(Some(&lw.ln_w), 1e-6);
            let r = normed.matmul(&lw.gate_wt).slice(&[0, 0, 0], &[1, 1, h]);
            let h2 = hidden.add(&r);
            let mlp_in = h2.rms_norm(Some(&lw.ln2_w), 1e-6);
            let gate = mlp_in.matmul(&lw.gate_wt);
            let up = mlp_in.matmul(&lw.up_wt);
            let mlp_out = InlineArray::fused_swiglu(&gate, &up).matmul(&lw.down_wt);
            hidden = h2.add(&mlp_out);
        }
        hidden.rms_norm(Some(&final_ln), 1e-6).matmul(&embed_w.t())
    };

    // Warmup
    for i in 0..5 {
        let r = run_step(42 + i);
        r.eval();
    }

    let mut times = Vec::new();
    for i in 0..30 {
        let t0 = Instant::now();
        let r = run_step(100 + i);
        r.eval();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    eprintln!(
        "Native weights forward: avg={avg:.2}ms = {:.0} tok/s",
        1000.0 / avg
    );
}
