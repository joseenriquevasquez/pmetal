use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::Dtype;
use pmetal_bridge::decode::sdpa_causal_like_mlx;
use pmetal_bridge::inline_array::eval_and_detach_many;
use pmetal_bridge::turboquant::{QuantizedKvCache, TurboQuantConfig, UniformAttentionBenchMode};
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    Dense,
    DenseAppend,
    Q8,
    Q8Append,
    Q8CoreSplit,
    Q8CoreD256,
    Q8CoreD256FullbytePass1,
    Q8CoreD256FullbytePass2,
    Q8CoreD256FullbyteSplitDenseV,
    Q8CoreD256FullbyteLocalSoftmax,
    Q8Score,
    Q8ScoreFullbyte,
    Q8Softmax,
    Q8Decode,
    Q8Transforms,
}

#[derive(Clone, Copy)]
struct Config {
    mode: Mode,
    input_dtype: i32,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    dim: i32,
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
            dim: 256,
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
                    "dense" => Mode::Dense,
                    "dense-append" => Mode::DenseAppend,
                    "q8" => Mode::Q8,
                    "q8-append" => Mode::Q8Append,
                    "q8-core-split" => Mode::Q8CoreSplit,
                    "q8-core-d256" => Mode::Q8CoreD256,
                    "q8-core-d256-fullbyte-pass1" => Mode::Q8CoreD256FullbytePass1,
                    "q8-core-d256-fullbyte-pass2" => Mode::Q8CoreD256FullbytePass2,
                    "q8-core-d256-fullbyte-split-densev" => Mode::Q8CoreD256FullbyteSplitDenseV,
                    "q8-core-d256-fullbyte-localsoftmax" => Mode::Q8CoreD256FullbyteLocalSoftmax,
                    "q8-score" => Mode::Q8Score,
                    "q8-score-fullbyte" => Mode::Q8ScoreFullbyte,
                    "q8-softmax" => Mode::Q8Softmax,
                    "q8-decode" => Mode::Q8Decode,
                    "q8-transforms" => Mode::Q8Transforms,
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
            "--dim" => cfg.dim = args.next().unwrap().parse().unwrap(),
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

fn bench_dense_step_ms(
    queries: &InlineArray,
    step_keys: &InlineArray,
    step_values: &InlineArray,
    seed_keys: &InlineArray,
    seed_values: &InlineArray,
    scale: f32,
    prefill: i32,
    warmup: usize,
    iters: usize,
) -> f64 {
    bench_eval_ms(
        || {
            let k_buf = seed_keys.clone();
            let v_buf = seed_values.clone();
            let start = [0, 0, prefill, 0];
            let stop = [
                queries.dim(0),
                step_keys.dim(1),
                prefill + 1,
                step_keys.dim(3),
            ];
            let valid_stop = [
                queries.dim(0),
                step_keys.dim(1),
                prefill + 1,
                step_keys.dim(3),
            ];
            let valid_start = [0, 0, 0, 0];

            let cache_keys = k_buf.slice_set(step_keys, &start, &stop);
            let cache_values = v_buf.slice_set(step_values, &start, &stop);
            let valid_keys = cache_keys.slice(&valid_start, &valid_stop);
            let valid_values = cache_values.slice(&valid_start, &valid_stop);
            sdpa_causal_like_mlx(queries, &valid_keys, &valid_values, scale, 1)
        },
        warmup,
        iters,
    )
}

fn bench_dense_append_ms(
    step_keys: &InlineArray,
    step_values: &InlineArray,
    seed_keys: &InlineArray,
    seed_values: &InlineArray,
    prefill: i32,
    warmup: usize,
    iters: usize,
) -> f64 {
    bench_eval_ms(
        || {
            let k_buf = seed_keys.clone();
            let v_buf = seed_values.clone();
            let start = [0, 0, prefill, 0];
            let stop = [
                step_keys.dim(0),
                step_keys.dim(1),
                prefill + 1,
                step_keys.dim(3),
            ];
            let cache_keys = k_buf.slice_set(step_keys, &start, &stop);
            let cache_values = v_buf.slice_set(step_values, &start, &stop);
            cache_keys.add(&cache_values)
        },
        warmup,
        iters,
    )
}

fn bench_q8_append_ms(
    step_keys: &InlineArray,
    step_values: &InlineArray,
    seed_cache: &QuantizedKvCache,
    warmup: usize,
    iters: usize,
) -> f64 {
    bench_eval_ms(
        || {
            let mut cache = seed_cache.clone();
            cache.append(step_keys, step_values).expect("q8 append");
            cache.eval_and_detach_gpu_state();
            cache
                .dequantize_values()
                .expect("q8 append dequantize values")
                .sum(Some(-1))
        },
        warmup,
        iters,
    )
}

fn bench_q8_core_ms(
    seed_cache: &QuantizedKvCache,
    queries: &InlineArray,
    scale: f32,
    mode: UniformAttentionBenchMode,
    warmup: usize,
    iters: usize,
) -> f64 {
    let queries_f32 = if queries.dtype_raw() == Dtype::Float32.as_i32() {
        queries.clone()
    } else {
        queries.as_dtype(Dtype::Float32.as_i32())
    };
    let (query_rot, query_proj) = seed_cache
        .bench_gpu_uniform_query_transforms(&queries_f32)
        .expect("q8 core transforms");
    let q_heads = queries.dim(1);

    bench_eval_ms(
        || {
            seed_cache
                .bench_gpu_uniform_attention_core_precomputed(
                    &query_rot,
                    &query_proj,
                    q_heads,
                    scale,
                    mode,
                )
                .expect("q8 core")
        },
        warmup,
        iters,
    )
}

fn bench_q8_score_ms(
    seed_cache: &QuantizedKvCache,
    queries: &InlineArray,
    scale: f32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let queries_f32 = if queries.dtype_raw() == Dtype::Float32.as_i32() {
        queries.clone()
    } else {
        queries.as_dtype(Dtype::Float32.as_i32())
    };
    let (query_rot, query_proj) = seed_cache
        .bench_gpu_uniform_query_transforms(&queries_f32)
        .expect("q8 score transforms");
    let q_heads = queries.dim(1);
    bench_eval_ms(
        || {
            seed_cache
                .bench_gpu_uniform_scores_precomputed(&query_rot, &query_proj, q_heads, scale)
                .expect("q8 score")
        },
        warmup,
        iters,
    )
}

fn bench_q8_score_fullbyte_ms(
    seed_cache: &QuantizedKvCache,
    queries: &InlineArray,
    scale: f32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let queries_f32 = if queries.dtype_raw() == Dtype::Float32.as_i32() {
        queries.clone()
    } else {
        queries.as_dtype(Dtype::Float32.as_i32())
    };
    let (query_rot, _) = seed_cache
        .bench_gpu_uniform_query_transforms(&queries_f32)
        .expect("q8 fullbyte score transforms");
    let q_heads = queries.dim(1);
    bench_eval_ms(
        || {
            seed_cache
                .bench_gpu_uniform_scores_precomputed_fullbyte(&query_rot, q_heads, scale)
                .expect("q8 fullbyte score")
        },
        warmup,
        iters,
    )
}

fn bench_q8_softmax_ms(
    seed_cache: &QuantizedKvCache,
    queries: &InlineArray,
    scale: f32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let queries_f32 = if queries.dtype_raw() == Dtype::Float32.as_i32() {
        queries.clone()
    } else {
        queries.as_dtype(Dtype::Float32.as_i32())
    };
    let (query_rot, query_proj) = seed_cache
        .bench_gpu_uniform_query_transforms(&queries_f32)
        .expect("q8 softmax transforms");
    let q_heads = queries.dim(1);
    let scores = seed_cache
        .bench_gpu_uniform_scores_precomputed(&query_rot, &query_proj, q_heads, scale)
        .expect("q8 scores");
    bench_eval_ms(|| scores.softmax(-1), warmup, iters)
}

fn bench_q8_decode_ms(
    seed_cache: &QuantizedKvCache,
    queries: &InlineArray,
    scale: f32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let queries_f32 = if queries.dtype_raw() == Dtype::Float32.as_i32() {
        queries.clone()
    } else {
        queries.as_dtype(Dtype::Float32.as_i32())
    };
    let (query_rot, query_proj) = seed_cache
        .bench_gpu_uniform_query_transforms(&queries_f32)
        .expect("q8 decode transforms");
    let q_heads = queries.dim(1);
    let scores = seed_cache
        .bench_gpu_uniform_scores_precomputed(&query_rot, &query_proj, q_heads, scale)
        .expect("q8 scores");
    let weights = scores.softmax(-1);
    bench_eval_ms(
        || {
            seed_cache
                .bench_gpu_uniform_weighted_decode(&weights, q_heads)
                .expect("q8 decode")
        },
        warmup,
        iters,
    )
}

fn bench_q8_transforms_ms(
    queries: &InlineArray,
    dim: i32,
    input_dtype: i32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let query_rows = queries.reshape(&[queries.dim(0) * queries.dim(1), dim]);
    let decoded_rot = InlineArray::from_f32_slice(
        &make_data((query_rows.dim(0) * dim) as usize, 3.1),
        &[query_rows.dim(0), dim],
    );
    let rot1 = InlineArray::from_f32_slice(&make_data((dim * dim) as usize, 4.1), &[dim, dim]);
    let rot2 = InlineArray::from_f32_slice(&make_data((dim * dim) as usize, 5.1), &[dim, dim]);
    let rot3 = InlineArray::from_f32_slice(&make_data((dim * dim) as usize, 6.1), &[dim, dim]);

    bench_eval_ms(
        || {
            let q = if input_dtype == Dtype::Float32.as_i32() {
                query_rows.clone()
            } else {
                query_rows.as_dtype(Dtype::Float32.as_i32())
            };
            let q_rot = q.matmul(&rot1);
            let q_proj = q.matmul(&rot2);
            let out = decoded_rot.matmul(&rot3);
            q_rot.add(&q_proj).add(&out)
        },
        warmup,
        iters,
    )
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
    let d = cfg.dim;
    let prefill = cfg.prefill;
    let scale = 1.0f32 / (cfg.dim as f32).sqrt();

    let prefill_kv_len = (b * kvh * prefill * d) as usize;
    let step_kv_len = (b * kvh * d) as usize;
    let step_q_len = (b * qh * d) as usize;

    let prefill_keys =
        InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.2), &[b, kvh, prefill, d])
            .as_dtype(cfg.input_dtype);
    let prefill_values =
        InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.7), &[b, kvh, prefill, d])
            .as_dtype(cfg.input_dtype);
    let queries = InlineArray::from_f32_slice(&make_data(step_q_len, 1.3), &[b, qh, 1, d])
        .as_dtype(cfg.input_dtype);
    let step_keys = InlineArray::from_f32_slice(&make_data(step_kv_len, 1.9), &[b, kvh, 1, d])
        .as_dtype(cfg.input_dtype);
    let step_values = InlineArray::from_f32_slice(&make_data(step_kv_len, 2.4), &[b, kvh, 1, d])
        .as_dtype(cfg.input_dtype);

    let bench_ms = match cfg.mode {
        Mode::Q8 => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();

            bench_eval_ms(
                || {
                    let mut cache = seed_cache.clone();
                    cache
                        .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
                        .expect("direct attention")
                },
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8Append => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_append_ms(&step_keys, &step_values, &seed_cache, cfg.warmup, cfg.iters)
        }
        Mode::Q8CoreSplit => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::Split,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8CoreD256 => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::SpecializedQ8D256TwoPass,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8CoreD256FullbytePass1 => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::SpecializedQ8D256FullbytePass1,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8CoreD256FullbytePass2 => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::SpecializedQ8D256FullbytePass2,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8CoreD256FullbyteSplitDenseV => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::SpecializedQ8D256FullbyteSplitDenseV,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8CoreD256FullbyteLocalSoftmax => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_core_ms(
                &seed_cache,
                &queries,
                scale,
                UniformAttentionBenchMode::SpecializedQ8D256FullbyteLocalSoftmax,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::Q8Score => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_score_ms(&seed_cache, &queries, scale, cfg.warmup, cfg.iters)
        }
        Mode::Q8ScoreFullbyte => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_score_fullbyte_ms(&seed_cache, &queries, scale, cfg.warmup, cfg.iters)
        }
        Mode::Q8Softmax => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_softmax_ms(&seed_cache, &queries, scale, cfg.warmup, cfg.iters)
        }
        Mode::Q8Decode => {
            let mut seed_cache = QuantizedKvCache::new(TurboQuantConfig::uniform(8, 8));
            seed_cache
                .append(&prefill_keys, &prefill_values)
                .expect("prefill append");
            seed_cache.eval_and_detach_gpu_state();
            bench_q8_decode_ms(&seed_cache, &queries, scale, cfg.warmup, cfg.iters)
        }
        Mode::Q8Transforms => {
            bench_q8_transforms_ms(&queries, d, cfg.input_dtype, cfg.warmup, cfg.iters)
        }
        Mode::Dense => {
            let (seed_keys, seed_values) = make_dense_seed(
                &prefill_keys,
                &prefill_values,
                b,
                kvh,
                prefill,
                d,
                cfg.input_dtype,
            );
            bench_dense_step_ms(
                &queries,
                &step_keys,
                &step_values,
                &seed_keys,
                &seed_values,
                scale,
                prefill,
                cfg.warmup,
                cfg.iters,
            )
        }
        Mode::DenseAppend => {
            let (seed_keys, seed_values) = make_dense_seed(
                &prefill_keys,
                &prefill_values,
                b,
                kvh,
                prefill,
                d,
                cfg.input_dtype,
            );
            bench_dense_append_ms(
                &step_keys,
                &step_values,
                &seed_keys,
                &seed_values,
                prefill,
                cfg.warmup,
                cfg.iters,
            )
        }
    };

    let toks_per_s = 1000.0 / bench_ms;
    let mode = match cfg.mode {
        Mode::Dense => "dense",
        Mode::DenseAppend => "dense-append",
        Mode::Q8 => "q8",
        Mode::Q8Append => "q8-append",
        Mode::Q8CoreSplit => "q8-core-split",
        Mode::Q8CoreD256 => "q8-core-d256",
        Mode::Q8CoreD256FullbytePass1 => "q8-core-d256-fullbyte-pass1",
        Mode::Q8CoreD256FullbytePass2 => "q8-core-d256-fullbyte-pass2",
        Mode::Q8CoreD256FullbyteSplitDenseV => "q8-core-d256-fullbyte-split-densev",
        Mode::Q8CoreD256FullbyteLocalSoftmax => "q8-core-d256-fullbyte-localsoftmax",
        Mode::Q8Score => "q8-score",
        Mode::Q8ScoreFullbyte => "q8-score-fullbyte",
        Mode::Q8Softmax => "q8-softmax",
        Mode::Q8Decode => "q8-decode",
        Mode::Q8Transforms => "q8-transforms",
    };
    let dtype = match cfg.input_dtype {
        x if x == Dtype::Bfloat16.as_i32() => "bf16",
        x if x == Dtype::Float32.as_i32() => "f32",
        _ => "other",
    };
    println!(
        "mode={} dtype={} batch={} q_heads={} kv_heads={} dim={} prefill={} decode_ms={:.3} tok_s={:.3}",
        mode,
        dtype,
        cfg.batch,
        cfg.q_heads,
        cfg.kv_heads,
        cfg.dim,
        cfg.prefill + 1,
        bench_ms,
        toks_per_s,
    );
}
