//! Evaluation and readback on [`InlineArray`].
//!
//! - `eval` / `async_eval` / `eval_2`: materialize the backing computation
//!   graph. `detach` severs the graph to free input references between
//!   decode steps.
//! - `item*` / `as_slice`: extract scalar / borrowed-slice views (require a
//!   prior eval).

use super::dtype::BridgeScalar;
use super::ffi::*;
use super::{EvalToken, InlineArray};

impl InlineArray {
    // ── Eval ─────────────────────────────────────────────────────────────

    pub fn eval(&self) -> EvalToken {
        // MLX array handles are internally mutable; eval materializes the backing
        // graph state but does not change the logical Rust ownership model.
        unsafe { mlx_inline_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken
    }
    pub fn async_eval(&self) -> EvalToken {
        unsafe { mlx_inline_async_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken
    }

    /// Eval two arrays in one call (avoids two FFI round-trips).
    #[inline]
    pub fn eval_2(a: &mut Self, b: &mut Self) {
        unsafe { mlx_inline_eval_2(&mut a.raw, &mut b.raw) }
    }

    /// Sever the computation graph, freeing all input references.
    /// CRITICAL for cache arrays: without this, cache updates chain across
    /// decode steps, keeping ALL previous steps' Metal buffers alive.
    /// Call on cache arrays after each eval to prevent memory accumulation.
    #[inline]
    pub fn detach(&mut self) {
        unsafe { mlx_inline_detach(&mut self.raw) }
    }

    // ── Async eval on borrowed ref ──────────────────────────────────────

    pub fn async_eval_ref(&self) {
        unsafe { mlx_inline_async_eval_arr(&self.raw) }
    }

    // ── Slice access (requires prior eval) ───────────────────────────────

    /// Return a borrowed slice of the array's f32 data.
    ///
    /// # Panics
    /// Panics if the array has not been evaluated (GPU→CPU sync), if the
    /// dtype is not Float32, or if the data pointer is null.
    pub fn as_slice<T: crate::inline_array::BridgeScalar>(&self) -> &[T] {
        let ptr = self.data_ptr() as *const T;
        assert!(
            !ptr.is_null(),
            "as_slice: array not evaluated (null data ptr)"
        );
        let n = self.size();
        // SAFETY: `data_ptr` returns a valid pointer into MLX's heap allocation
        // for the lifetime of `self`. The array must have been `eval()`d first
        // so the pointer is on the CPU (not on the GPU).
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }

    // ── Item extraction ──────────────────────────────────────────────────

    pub fn item_f32(&self) -> f32 {
        let mut owned = self.clone();
        owned.eval();
        unsafe { mlx_inline_item_f32(&mut owned.raw) }
    }
    pub fn item_u32(&self) -> u32 {
        let mut owned = self.clone();
        owned.eval();
        unsafe { mlx_inline_item_u32(&mut owned.raw) }
    }

    // ── Item extraction (generic) ────────────────────────────────────────

    /// Extract the scalar value from a 0-d array. Evaluates lazily if needed.
    /// `T` must be `f32` or `u32` (the only types exported by the bridge).
    pub fn item<T: BridgeScalar>(&self) -> T {
        let mut owned = self.clone();
        owned.eval();
        T::extract(&mut owned)
    }
}
