use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::Dtype;
use pmetal_bridge::decode::sdpa_causal_like_mlx;
use pmetal_bridge::inline_array::eval_and_detach_many;
use pmetal_bridge::turboquant::{QuantizedKvCache, TurboQuantConfig};
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    QueryPath,
    Dense,
    Q8,
}

#[derive(Clone, Copy)]
struct Config {
    mode: Mode,
    input_dtype: i32,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    hidden_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    prefill: i32,
    warmup: usize,
    iters: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Q8,
            input_dtype: Dtype::Bfloat16.as_i32(),
            batch: 1,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            hidden_dim: 4096,
            rope_dims: 64,
            rope_base: 1_000_000.0,
            rope_scale: 1.0,
            prefill: 2047,
            warmup: 5,
            iters: 20,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                cfg.mode = match args.next().unwrap().as_str() {
                    "query-path" => Mode::QueryPath,
                    "dense" => Mode::Dense,
                    "q8" => Mode::Q8,
                    other => panic!("unsupported mode: {other}"),
                }
            }
            "--dtype" => {
                cfg.input_dtype = match args.next().unwrap().as_str() {
                    "bf16" => Dtype::Bfloat16.as_i32(),
                    "f32" => Dtype::Float32.as_i32(),
                    other => panic!("unsupported dtype: {other}"),
                }
            }
            "--batch" => cfg.batch = args.next().unwrap().parse().unwrap(),
            "--q-heads" => cfg.q_heads = args.next().unwrap().parse().unwrap(),
            "--kv-heads" => cfg.kv_heads = args.next().unwrap().parse().unwrap(),
            "--head-dim" => cfg.head_dim = args.next().unwrap().parse().unwrap(),
            "--hidden-dim" => cfg.hidden_dim = args.next().unwrap().parse().unwrap(),
            "--rope-dims" => cfg.rope_dims = args.next().unwrap().parse().unwrap(),
            "--rope-base" => cfg.rope_base = args.next().unwrap().parse().unwrap(),
            "--rope-scale" => cfg.rope_scale = args.next().unwrap().parse().unwrap(),
            "--prefill" => cfg.prefill = args.next().unwrap().parse().unwrap(),
            "--warmup" => cfg.warmup = args.next().unwrap().parse().unwrap(),
            "--iters" => cfg.iters = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    cfg
}

fn make_data(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
        .collect()
}

fn make_weight(in_dim: i32, out_dim: i32, dtype: i32, seed: f32) -> InlineArray {
    InlineArray::from_f32_slice(
        &make_data((in_dim * out_dim) as usize, seed),
        &[in_dim, out_dim],
    )
    .as_dtype(dtype)
}

fn make_vector(dim: i32, dtype: i32, seed: f32) -> InlineArray {
    InlineArray::from_f32_slice(&make_data(dim as usize, seed), &[dim]).as_dtype(dtype)
}

fn bench_eval_ms(mut f: impl FnMut() -> InlineArray, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let out = f();
        out.eval();
    }
    let mut total_ms = 0.0;
    for _ in 0..iters {
        let out = f();
        let t0 = Instant::now();
        out.eval();
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    total_ms / iters as f64
}

fn make_dense_seed(
    prefill_keys: &InlineArray,
    prefill_values: &InlineArray,
    batch: i32,
    kv_heads: i32,
    prefill: i32,
    dim: i32,
    dtype: i32,
) -> (InlineArray, InlineArray) {
    let zeros_k = InlineArray::zeros(&[batch, kv_heads, prefill + 1, dim], dtype);
    let zeros_v = InlineArray::zeros(&[batch, kv_heads, prefill + 1, dim], dtype);
    let start = [0, 0, 0, 0];
    let stop = [batch, kv_heads, prefill, dim];
    let mut k_buf = zeros_k.slice_set(prefill_keys, &start, &stop);
    let mut v_buf = zeros_v.slice_set(prefill_values, &start, &stop);
    let mut to_eval = vec![&mut k_buf, &mut v_buf];
    eval_and_detach_many(&mut to_eval);
    (k_buf, v_buf)
}

struct ReplayWeights {
    q_w: InlineArray,
    k_w: InlineArray,
    v_w: InlineArray,
    q_norm_w: InlineArray,
    k_norm_w: InlineArray,
}

#[allow(clippy::too_many_arguments)]
fn build_qkv(
    normed: &InlineArray,
    weights: &ReplayWeights,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    rope_scale: f32,
    rope_offset: i32,
) -> (InlineArray, InlineArray, InlineArray) {
    let b = normed.dim(0);
    let s = normed.dim(1);

    let queries = normed
        .matmul(&weights.q_w)
        .reshape(&[b, s, q_heads, head_dim])
        .rms_norm(Some(&weights.q_norm_w), 1e-6)
        .transpose_axes(&[0, 2, 1, 3])
        .rope(rope_dims, false, rope_base, rope_scale, rope_offset);

    let keys = normed
        .matmul(&weights.k_w)
        .reshape(&[b, s, kv_heads, head_dim])
        .rms_norm(Some(&weights.k_norm_w), 1e-6)
        .transpose_axes(&[0, 2, 1, 3])
        .rope(rope_dims, false, rope_base, rope_scale, rope_offset);

    let values = normed
        .matmul(&weights.v_w)
        .reshape(&[b, s, kv_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);

    (queries, keys, values)
}

fn main() {
    let cfg = parse_args();
    assert!(
        cfg.q_heads % cfg.kv_heads == 0,
        "q_heads must be divisible by kv_heads"
    );
    assert!(cfg.prefill >= 1, "prefill must be >= 1");

    let b = cfg.batch;
    let qh = cfg.q_heads;
    let kvh = cfg.kv_heads;
    let d = cfg.head_dim;
    let hidden = cfg.hidden_dim;
    let prefill = cfg.prefill;
    let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

    let prefill_hidden_len = (b * prefill * hidden) as usize;
    let step_hidden_len = (b * hidden) as usize;

    let prefill_hidden =
        InlineArray::from_f32_slice(&make_data(prefill_hidden_len, 0.2), &[b, prefill, hidden])
            .as_dtype(cfg.input_dtype);
    let step_hidden =
        InlineArray::from_f32_slice(&make_data(step_hidden_len, 0.7), &[b, 1, hidden])
            .as_dtype(cfg.input_dtype);

    let weights = ReplayWeights {
        q_w: make_weight(hidden, qh * d, cfg.input_dtype, 1.3),
        k_w: make_weight(hidden, kvh * d, cfg.input_dtype, 1.9),
        v_w: make_weight(hidden, kvh * d, cfg.input_dtype, 2.5),
        q_norm_w: make_vector(d, cfg.input_dtype, 3.1),
        k_norm_w: make_vector(d, cfg.input_dtype, 3.7),
    };

    let (_prefill_queries, prefill_keys, prefill_values) = build_qkv(
        &prefill_hidden,
        &weights,
        qh,
        kvh,
        d,
        cfg.rope_dims,
        cfg.rope_base,
        cfg.rope_scale,
        0,
    );

    let (seed_keys, seed_values) = make_dense_seed(
        &prefill_keys,
        &prefill_values,
        b,
        kvh,
        prefill,
        d,
        cfg.input_dtype,
    );

    let mut seed_q8 = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
    seed_q8
        .append(&prefill_keys, &prefill_values)
        .expect("prefill append");
    seed_q8.eval_and_detach_gpu_state();

    let bench_ms = match cfg.mode {
        Mode::QueryPath => bench_eval_ms(
            || {
                let (queries, keys, values) = build_qkv(
                    &step_hidden,
                    &weights,
                    qh,
                    kvh,
                    d,
                    cfg.rope_dims,
                    cfg.rope_base,
                    cfg.rope_scale,
                    prefill,
                );
                let query_sum = queries.sum(Some(-1));
                let key_sum = keys.repeat(qh / kvh, 1).sum(Some(-1));
                let value_sum = values.repeat(qh / kvh, 1).sum(Some(-1));
                query_sum.add(&key_sum).add(&value_sum)
            },
            cfg.warmup,
            cfg.iters,
        ),
        Mode::Dense => bench_eval_ms(
            || {
                let (queries, step_keys, step_values) = build_qkv(
                    &step_hidden,
                    &weights,
                    qh,
                    kvh,
                    d,
                    cfg.rope_dims,
                    cfg.rope_base,
                    cfg.rope_scale,
                    prefill,
                );
                let k_buf = seed_keys.clone();
                let v_buf = seed_values.clone();
                let start = [0, 0, prefill, 0];
                let stop = [b, kvh, prefill + 1, d];
                let valid_start = [0, 0, 0, 0];
                let valid_stop = [b, kvh, prefill + 1, d];
                let cache_keys = k_buf.slice_set(&step_keys, &start, &stop);
                let cache_values = v_buf.slice_set(&step_values, &start, &stop);
                let valid_keys = cache_keys.slice(&valid_start, &valid_stop);
                let valid_values = cache_values.slice(&valid_start, &valid_stop);
                sdpa_causal_like_mlx(&queries, &valid_keys, &valid_values, scale, 1)
            },
            cfg.warmup,
            cfg.iters,
        ),
        Mode::Q8 => bench_eval_ms(
            || {
                let (queries, step_keys, step_values) = build_qkv(
                    &step_hidden,
                    &weights,
                    qh,
                    kvh,
                    d,
                    cfg.rope_dims,
                    cfg.rope_base,
                    cfg.rope_scale,
                    prefill,
                );
                let mut cache = seed_q8.clone();
                cache
                    .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
                    .expect("q8 attention")
            },
            cfg.warmup,
            cfg.iters,
        ),
    };

    let toks_per_s = 1000.0 / bench_ms;
    let mode = match cfg.mode {
        Mode::QueryPath => "query-path",
        Mode::Dense => "dense",
        Mode::Q8 => "q8",
    };
    let dtype = if cfg.input_dtype == Dtype::Bfloat16.as_i32() {
        "bf16"
    } else {
        "f32"
    };
    println!(
        "mode={} dtype={} batch={} q_heads={} kv_heads={} head_dim={} hidden_dim={} rope_dims={} prefill={} decode_ms={:.3} tok_s={:.3}",
        mode,
        dtype,
        b,
        qh,
        kvh,
        d,
        hidden,
        cfg.rope_dims,
        prefill + 1,
        bench_ms,
        toks_per_s
    );
}
