//! Collective and point-to-point operations using MLX distributed.
//!
//! These wrap the `mlx_distributed_*` C API functions, providing safe Rust
//! interfaces that work with `pmetal_bridge::compat::Array`. All operations
//! are lazy — they build the MLX computation graph and execute when
//! `eval()` is called.
//!
//! # Collective Operations
//!
//! - [`all_sum`] — Element-wise sum across all ranks (primary for TP all-reduce)
//! - [`all_gather`] — Concatenate tensors from all ranks
//! - [`all_max`] / [`all_min`] — Element-wise max/min reduction
//! - [`sum_scatter`] — Reduce-scatter (sum + split across ranks)
//!
//! # Point-to-Point Operations
//!
//! - [`send`] — Send tensor to a specific rank
//! - [`recv`] — Receive tensor of known shape from a specific rank
//! - [`recv_like`] — Receive tensor matching another tensor's shape/dtype

use super::group::{DistributedGroup, MlxDistributedGroup};
use pmetal_bridge::compat::{Array, Dtype, Exception};
use std::ffi::c_void;

// ── Local FFI declarations for MLX distributed ops ───────────────────────────
//
// These mirror the symbols from mlx-sys / mlx-c but are declared directly here
// so pmetal-distributed does not need the mlx-sys crate.
//
// `mlx_array` and `mlx_stream` are opaque C-ABI handles — each is a struct
// containing a single heap-allocated `ctx` pointer.  `InlineArray::to_raw_ctx()`
// / `from_raw_ctx()` bridge between the bridge's inline representation and these
// heap-allocated handles.

/// Opaque handle for the MLX C array type (`mlx_array` in `mlx/c/array.h`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MlxArray {
    ctx: *mut c_void,
}

/// Opaque handle for the MLX C stream type (`mlx_stream` in `mlx/c/stream.h`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MlxStream {
    ctx: *mut c_void,
}

#[allow(unsafe_code)]
unsafe extern "C" {
    // Stream helpers (from mlx/c/stream.h)
    fn mlx_get_default_device(dev: *mut MlxStream) -> i32;
    fn mlx_stream_new() -> MlxStream;
    fn mlx_get_default_stream(stream: *mut MlxStream, dev: MlxStream) -> i32;
    fn mlx_stream_free(stream: MlxStream);
    fn mlx_device_new() -> MlxStream; // mlx_device and mlx_stream have the same ABI layout

    // Array lifecycle (from mlx/c/array.h)
    fn mlx_array_free(arr: MlxArray);

    // Distributed collectives (from mlx/c/distributed.h)
    fn mlx_distributed_all_sum(
        result: *mut MlxArray,
        input: MlxArray,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_all_gather(
        result: *mut MlxArray,
        input: MlxArray,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_all_max(
        result: *mut MlxArray,
        input: MlxArray,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_all_min(
        result: *mut MlxArray,
        input: MlxArray,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_sum_scatter(
        result: *mut MlxArray,
        input: MlxArray,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_send(
        result: *mut MlxArray,
        input: MlxArray,
        dst: i32,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_recv(
        result: *mut MlxArray,
        shape: *const i32,
        num_dims: usize,
        dtype: i32,
        src: i32,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
    fn mlx_distributed_recv_like(
        result: *mut MlxArray,
        input: MlxArray,
        src: i32,
        group: MlxDistributedGroup,
        stream: MlxStream,
    ) -> i32;
}

const SUCCESS: i32 = 0;

/// Acquire the default GPU stream.  The caller is responsible for freeing
/// via `mlx_stream_free` when done (use the RAII wrapper below).
#[allow(unsafe_code)]
fn default_stream() -> MlxStream {
    unsafe {
        // mlx_device and mlx_stream have identical ABI (both are `{ ctx }`).
        // We reuse the same type for both here; the casts are deliberate.
        let mut dev: MlxStream = mlx_device_new();
        // Ignore return value — the C API guarantees success for GPU device.
        mlx_get_default_device(&mut dev as *mut MlxStream);
        let mut stream: MlxStream = mlx_stream_new();
        mlx_get_default_stream(&mut stream as *mut MlxStream, dev);
        // Free the device handle (stream keeps its own reference).
        mlx_stream_free(dev);
        stream
    }
}

/// RAII guard that frees an `MlxStream` on drop.
struct StreamGuard(MlxStream);

impl Drop for StreamGuard {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_stream_free(self.0);
        }
    }
}

/// Convert an optional group reference to the raw C handle.
fn group_handle(group: Option<&DistributedGroup>) -> MlxDistributedGroup {
    match group {
        Some(g) => g.inner,
        None => DistributedGroup::null_handle(),
    }
}

/// Convert `Array` (InlineArray) to a heap-allocated `MlxArray` handle for FFI.
///
/// The returned `MlxArray` is owned by the caller and must be freed with
/// `mlx_array_free` after use.  We wrap it in an RAII guard.
struct ArrayHandle(MlxArray);

impl ArrayHandle {
    #[allow(unsafe_code)]
    fn from_array(arr: &Array) -> Self {
        // to_raw_ctx() heap-allocates and ref-counts the array — we own it.
        let ctx = arr.to_raw_ctx();
        Self(MlxArray { ctx })
    }
}

impl Drop for ArrayHandle {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_array_free(self.0);
        }
    }
}

/// Convert a raw `MlxArray` result from a distributed op into an `Array`.
///
/// Takes ownership of the heap-allocated handle and wraps it in an InlineArray.
#[allow(unsafe_code)]
fn array_from_handle(handle: MlxArray) -> Array {
    // from_raw_ctx copies (ref-counts) the C++ array into the inline buffer,
    // then we free the original heap handle.
    let arr = unsafe { Array::from_raw_ctx(handle.ctx) };
    // Free the outer heap allocation — the inline array now owns its copy.
    unsafe { mlx_array_free(handle) };
    arr
}

/// Element-wise sum across all ranks.
///
/// This is the primary collective for tensor parallelism: used in
/// `ShardedToAllLinear` to reduce partial matmul results.
///
/// Returns a new array where each element is the sum of corresponding
/// elements across all ranks in the group.
#[allow(unsafe_code)]
pub fn all_sum(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_all_sum(
            &mut result,
            x_handle.0,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_all_sum failed"));
    }
    Ok(array_from_handle(result))
}

/// Gather tensors from all ranks, concatenating along the first axis.
///
/// If each rank has a tensor of shape `[N, ...]`, the result has shape
/// `[N * world_size, ...]` with data from rank 0 first, then rank 1, etc.
#[allow(unsafe_code)]
pub fn all_gather(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_all_gather(
            &mut result,
            x_handle.0,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_all_gather failed"));
    }
    Ok(array_from_handle(result))
}

/// Element-wise maximum across all ranks.
#[allow(unsafe_code)]
pub fn all_max(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_all_max(
            &mut result,
            x_handle.0,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_all_max failed"));
    }
    Ok(array_from_handle(result))
}

/// Element-wise minimum across all ranks.
#[allow(unsafe_code)]
pub fn all_min(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_all_min(
            &mut result,
            x_handle.0,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_all_min failed"));
    }
    Ok(array_from_handle(result))
}

/// Reduce-scatter: sum across ranks, then split the result.
///
/// Each rank receives a different shard of the reduced result.
/// If each rank has a tensor of shape `[N, ...]`, each rank gets
/// a tensor of shape `[N / world_size, ...]` after reduction.
#[allow(unsafe_code)]
pub fn sum_scatter(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_sum_scatter(
            &mut result,
            x_handle.0,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_sum_scatter failed"));
    }
    Ok(array_from_handle(result))
}

/// Send a tensor to a destination rank.
///
/// Returns a sentinel array that must be evaluated to trigger the send.
/// The send is non-blocking in the MLX graph but synchronizes when evaluated.
#[allow(unsafe_code)]
pub fn send(x: &Array, dst: i32, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_send(
            &mut result,
            x_handle.0,
            dst,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_send failed"));
    }
    Ok(array_from_handle(result))
}

/// Receive a tensor of known shape and dtype from a source rank.
///
/// The shape and dtype must match what the sender is transmitting.
#[allow(unsafe_code)]
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    src: i32,
    group: Option<&DistributedGroup>,
) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let dtype_i32 = dtype.as_i32();
    let status = unsafe {
        mlx_distributed_recv(
            &mut result,
            shape.as_ptr(),
            shape.len(),
            dtype_i32,
            src,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_recv failed"));
    }
    Ok(array_from_handle(result))
}

/// Receive a tensor matching another tensor's shape and dtype.
///
/// Convenience wrapper around [`recv`] that infers shape and dtype
/// from a reference array.
#[allow(unsafe_code)]
pub fn recv_like(
    x: &Array,
    src: i32,
    group: Option<&DistributedGroup>,
) -> Result<Array, Exception> {
    let _stream = StreamGuard(default_stream());
    let x_handle = ArrayHandle::from_array(x);
    let mut result = MlxArray { ctx: std::ptr::null_mut() };
    let status = unsafe {
        mlx_distributed_recv_like(
            &mut result,
            x_handle.0,
            src,
            group_handle(group),
            _stream.0,
        )
    };
    if status != SUCCESS {
        return Err(Exception::custom("mlx_distributed_recv_like failed"));
    }
    Ok(array_from_handle(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_module_compiles() {
        // Verify the module compiles and all functions are callable.
        // Actual distributed tests require a multi-process environment.
        let _ = DistributedGroup::is_available();
    }
}
