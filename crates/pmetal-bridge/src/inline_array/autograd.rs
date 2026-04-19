//! Autograd entry points: `value_and_grad` and `checkpoint_apply`.
//!
//! Both cross the FFI boundary via trampoline callbacks so the inner closure
//! can use normal `InlineArray` ops while C++ manages the autograd tape.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

// ── value_and_grad ───────────────────────────────────────────────────────

/// Compute loss + gradients via callback-based autograd.
///
/// `loss_fn` receives all arrays (params first, then inputs) and must return
/// a scalar loss. Gradients are computed w.r.t. the first `params.len()` arrays.
///
/// Returns `(loss, gradients)` where `gradients[i]` is `dloss/dparams[i]`.
pub fn value_and_grad<F>(
    mut loss_fn: F,
    params: &[InlineArray],
    inputs: &[InlineArray],
) -> (InlineArray, Vec<InlineArray>)
where
    F: FnMut(&[InlineArray]) -> InlineArray,
{
    // Trampoline: C++ calls this with InlineArray-sized buffers
    unsafe extern "C" fn trampoline<F: FnMut(&[InlineArray]) -> InlineArray>(
        all_arrays: *const *const RawBuf,
        n_total: i32,
        loss_out: *mut RawBuf,
        ctx: *mut std::ffi::c_void,
    ) {
        let f = unsafe { &mut *(ctx as *mut F) };
        // Wrap raw pointers as borrowed InlineArrays (no ownership transfer)
        let arrays: Vec<InlineArray> = (0..n_total as usize)
            .map(|i| {
                let ptr = unsafe { *all_arrays.add(i) };
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), ptr) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();
        let loss = f(&arrays);
        // Write loss into output buffer (placement-copy)
        unsafe { mlx_inline_init_copy(loss_out, &loss.raw) };
        // arrays and loss drop here (calling mlx_inline_destroy for each)
    }

    let n_params = params.len();
    let n_total = n_params + inputs.len();

    // Build flat pointer array: [param0, param1, ..., input0, input1, ...]
    let all_ptrs: Vec<*const RawBuf> = params
        .iter()
        .chain(inputs.iter())
        .map(|a| &a.raw as *const RawBuf)
        .collect();

    let mut loss = InlineArray::from_f32(0.0);
    let mut grads: Vec<InlineArray> = (0..n_params).map(|_| InlineArray::from_f32(0.0)).collect();
    let mut grad_ptrs: Vec<*mut RawBuf> = grads
        .iter_mut()
        .map(|g| &mut g.raw as *mut RawBuf)
        .collect();

    unsafe {
        mlx_inline_value_and_grad(
            trampoline::<F>,
            &mut loss_fn as *mut F as *mut std::ffi::c_void,
            all_ptrs.as_ptr(),
            n_params as i32,
            n_total as i32,
            &mut loss.raw,
            grad_ptrs.as_mut_ptr(),
        );
    }

    (loss, grads)
}

// ── Gradient checkpointing ───────────────────────────────────────────────

/// Apply gradient checkpointing to a forward function.
///
/// `inner_fn` receives the input arrays and must return a `Vec<InlineArray>`.
/// The returned arrays are computed through `mlx::core::checkpoint()`, which
/// discards all intermediate activations after the forward pass and recomputes
/// them during the backward pass.  This reduces peak activation memory from
/// O(layers × batch × seq × hidden) to O(1 layer) at the cost of one extra
/// forward pass per gradient step.
///
/// # Usage
///
/// ```ignore
/// let outputs = checkpoint_apply(&inputs, |arrays| {
///     // normal forward computation using InlineArray ops
///     let h = arrays[0].matmul(&weight);
///     vec![h.relu()]
/// });
/// ```
///
/// # Panics
///
/// Panics if the C++ checkpoint call throws (logged to stderr before panicking).
// TODO(gradient-checkpointing): wire into training loops (v0.4.0 roadmap).
// Memory win: O(layers) → O(1) activation memory at the cost of one recompute
// pass. Kept available for when the training loop opts in.
#[allow(dead_code)]
pub fn checkpoint_apply<F>(inputs: &[InlineArray], mut inner_fn: F) -> Vec<InlineArray>
where
    F: FnMut(&[InlineArray]) -> Vec<InlineArray>,
{
    // Trampoline: C++ calls this with InlineArray-sized bufs for both the
    // input arrays and the flat output buffer.
    unsafe extern "C" fn trampoline<F: FnMut(&[InlineArray]) -> Vec<InlineArray>>(
        all_arrays: *const *const RawBuf,
        n_total: i32,
        outputs_out: *mut RawBuf,
        n_outputs_out: *mut i32,
        ctx: *mut std::ffi::c_void,
    ) {
        let f = unsafe { &mut *(ctx as *mut F) };

        // Borrow-wrap each input pointer as an InlineArray (copy-construct).
        let arrays: Vec<InlineArray> = (0..n_total as usize)
            .map(|i| {
                let ptr = unsafe { *all_arrays.add(i) };
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), ptr) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();

        let results = f(&arrays);
        let n = results.len();

        // Write each output via placement-copy into the caller's flat buffer.
        for (i, r) in results.iter().enumerate() {
            unsafe { mlx_inline_init_copy(outputs_out.add(i), &r.raw) };
        }
        unsafe { *n_outputs_out = n as i32 };
        // `arrays` and `results` drop here, calling mlx_inline_destroy for each.
    }

    let n_total = inputs.len();
    // Build flat pointer array for the C++ side.
    let all_ptrs: Vec<*const RawBuf> = inputs.iter().map(|a| &a.raw as *const RawBuf).collect();

    // Pre-allocate output buffer.  We don't know n_outputs ahead of time so
    // we ask inner_fn once with a dry run — but that would break the graph.
    // Instead, we use a convention: inner_fn is called exactly once inside
    // checkpoint(); its return Vec length sets n_outputs_max.  We must know
    // this capacity before the C++ call.  The caller communicates this via a
    // sentinel: we call inner_fn on *copies* to count outputs, then replay
    // via the checkpointed path.
    //
    // In practice, callers always know how many tensors their forward pass
    // returns (it's statically determined).  The trampoline writes into
    // outputs_out using the count returned by the callback itself, so we
    // only need to allocate a buffer large enough.  We use a fixed max of
    // 64 outputs — sufficient for any realistic use (a 28-layer model running
    // one layer at a time produces at most ~4 outputs: hidden, kv_k, kv_v, state).
    //
    // The C++ side also guards against overflow via `i < n_outputs_max`.
    // Pre-initialize each slot with a valid scalar so the C++ side can safely
    // call placement-new over them (matching the established bridge convention
    // used in value_and_grad's grads_out pre-initialization).
    const MAX_OUTPUTS: usize = 64;
    let mut output_storage: Vec<InlineArray> = (0..MAX_OUTPUTS)
        .map(|_| InlineArray::from_f32(0.0))
        .collect();

    let mut n_written: i32 = 0;
    unsafe {
        mlx_inline_checkpoint(
            trampoline::<F>,
            &mut inner_fn as *mut F as *mut std::ffi::c_void,
            all_ptrs.as_ptr(),
            n_total as i32,
            MAX_OUTPUTS as i32,
            // Safety: InlineArray is a single-field struct whose only field is
            // RawBuf, so &mut output_storage[0].raw is the first byte of the
            // first element of a contiguous Vec<InlineArray> — valid for
            // pointer arithmetic up to MAX_OUTPUTS elements.
            &mut output_storage[0].raw,
            &mut n_written,
        );
    }

    // Truncate to the actual count written by the callback.  The remaining
    // slots (still holding their from_f32(0.0) arrays) are dropped by Vec.
    let n = n_written as usize;
    output_storage.truncate(n);
    output_storage
}
