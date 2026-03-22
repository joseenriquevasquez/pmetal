//! Generic FP8 weight quantization utilities for all model architectures.
//!
//! This module provides architecture-agnostic FP8 quantization by traversing
//! a model's flattened parameter map and replacing every `.weight` tensor with
//! its `to_fp8` equivalent.  It is intentionally separate from the NemotronH-
//! specific implementation in `architectures/nemotron_h.rs`, which operates on
//! concrete `nn::Linear` structs with custom FP8-aware forward passes.
//!
//! # How it works
//!
//! `ModuleParameters::parameters_mut().flatten()` yields a
//! `HashMap<Rc<str>, &mut Array>` whose keys are dot-separated parameter paths
//! (e.g. `"model.layers.0.self_attn.q_proj.weight"`).  Any key whose final
//! component is `"weight"` is a weight matrix eligible for FP8 quantization.
//! Bias vectors (suffix `"bias"`) and normalisation scales (suffix
//! `"weight"` inside a `LayerNorm`/`RMSNorm` are indistinguishable at this
//! level, but the FP8 range of ±240 safely covers normalisation scales in
//! practice).
//!
//! After quantisation the parameter arrays are stored as `uint8` (E4M3 format)
//! in-place.  Inference code that reads these parameters must call
//! `mlx_rs::ops::from_fp8` to dequantise before computation — this matches the
//! semantics already used by `pmetal_mlx::fp8_quantization`.

use mlx_rs::{
    error::Exception,
    module::ModuleParameters,
    ops::to_fp8,
};

/// Quantize every `.weight` parameter of `model` to FP8 (E4M3) in-place.
///
/// The function iterates the fully-flattened parameter map and replaces every
/// array whose key ends with `".weight"` (or equals `"weight"` for top-level
/// parameters) with its `to_fp8` representation.
///
/// Biases, embedding tables, and normalisation scale vectors whose keys end
/// with suffixes other than `".weight"` are left untouched.
///
/// # Errors
///
/// Returns the first `Exception` produced by `mlx_rs::ops::to_fp8` if any
/// quantisation call fails.
pub fn quantize_model_linears<M: ModuleParameters>(model: &mut M) -> Result<(), Exception> {
    // Collect the keys we need to quantize first (avoid borrow issues).
    // We need owned quantized arrays, then write them back through parameters_mut.
    let keys_to_quantize: Vec<std::rc::Rc<str>> = {
        let params = model.parameters();
        let flat = params.flatten();
        flat.into_iter()
            .filter_map(|(k, _)| {
                if k.ends_with(".weight") || k.as_ref() == "weight" {
                    Some(k)
                } else {
                    None
                }
            })
            .collect()
    };

    if keys_to_quantize.is_empty() {
        return Ok(());
    }

    // Quantize each eligible weight.  We do this in two passes to satisfy the
    // borrow checker: first read + compute FP8 tensors (immutable borrow),
    // then write them back (mutable borrow).
    let quantized: Vec<(std::rc::Rc<str>, mlx_rs::Array)> = {
        let params = model.parameters();
        let flat = params.flatten();
        keys_to_quantize
            .iter()
            .filter_map(|k| flat.get(k).map(|arr| (k.clone(), *arr)))
            .map(|(k, arr)| to_fp8(arr).map(|q| (k, q)))
            .collect::<Result<_, _>>()?
    };

    // Write the FP8 tensors back through the mutable parameter map.
    {
        let params_mut = model.parameters_mut();
        #[allow(unused_mut)]
        let mut flat_mut = params_mut.flatten();
        for (k, q) in quantized {
            if let Some(dest) = flat_mut.get_mut(&k) {
                **dest = q;
            }
        }
    }

    Ok(())
}
