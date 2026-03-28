//! `pmetal-bridge` — zero-allocation MLX C++ bridge.
//!
//! Provides [`InlineArray`], a stack-allocated wrapper around
//! `mlx::core::array` with no per-op heap allocation.  All C++ calls go
//! directly through `extern "C"` declarations; this crate has no dependency
//! on mlx-rs.
//!
//! During the transition period, [`InlineArray::from_raw_ctx`] lets callers
//! interop with existing mlx-rs `Array` values by passing the opaque context
//! pointer (`arr.as_ptr().ctx`).

pub mod inline_array;
pub use inline_array::InlineArray;

pub mod optimizer;
pub use optimizer::{AdamW, ParamClass, ParamSet};

pub mod turboquant;
pub mod training;

pub mod qwen3_native;
pub mod qwen3_train;
pub mod deepseek_native;
pub mod gpt_oss_native;
pub mod llama4_native;
