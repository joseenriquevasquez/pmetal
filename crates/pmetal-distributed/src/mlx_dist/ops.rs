//! Collective and point-to-point operations using MLX distributed.
//!
//! These wrap the `mlx_distributed_*` C API functions, providing safe Rust
//! interfaces that work with `mlx_rs::Array`. All operations are lazy —
//! they build the MLX computation graph and execute when `eval()` is called.
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

use super::group::DistributedGroup;
use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype, Stream};

const SUCCESS: i32 = 0;

/// Convert an optional group reference to the raw C handle.
fn group_handle(group: Option<&DistributedGroup>) -> mlx_sys::mlx_distributed_group {
    match group {
        Some(g) => g.as_raw(),
        None => DistributedGroup::null_handle(),
    }
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
    // Keep the stream alive for the duration of the FFI call.
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_all_sum(
            &mut result,
            x.as_ptr(),
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_all_sum failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Gather tensors from all ranks, concatenating along the first axis.
///
/// If each rank has a tensor of shape `[N, ...]`, the result has shape
/// `[N * world_size, ...]` with data from rank 0 first, then rank 1, etc.
#[allow(unsafe_code)]
pub fn all_gather(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_all_gather(
            &mut result,
            x.as_ptr(),
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_all_gather failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Element-wise maximum across all ranks.
#[allow(unsafe_code)]
pub fn all_max(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_all_max(
            &mut result,
            x.as_ptr(),
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_all_max failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Element-wise minimum across all ranks.
#[allow(unsafe_code)]
pub fn all_min(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_all_min(
            &mut result,
            x.as_ptr(),
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_all_min failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Reduce-scatter: sum across ranks, then split the result.
///
/// Each rank receives a different shard of the reduced result.
/// If each rank has a tensor of shape `[N, ...]`, each rank gets
/// a tensor of shape `[N / world_size, ...]` after reduction.
#[allow(unsafe_code)]
pub fn sum_scatter(x: &Array, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_sum_scatter(
            &mut result,
            x.as_ptr(),
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_sum_scatter failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Send a tensor to a destination rank.
///
/// Returns a sentinel array that must be evaluated to trigger the send.
/// The send is non-blocking in the MLX graph but synchronizes when evaluated.
#[allow(unsafe_code)]
pub fn send(x: &Array, dst: i32, group: Option<&DistributedGroup>) -> Result<Array, Exception> {
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_send(
            &mut result,
            x.as_ptr(),
            dst,
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_send failed"));
        }
        Ok(Array::from_ptr(result))
    }
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
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let mlx_dtype = dtype.into();
        let status = mlx_sys::mlx_distributed_recv(
            &mut result,
            shape.as_ptr(),
            shape.len(),
            mlx_dtype,
            src,
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_recv failed"));
        }
        Ok(Array::from_ptr(result))
    }
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
    let stream = Stream::new();
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = mlx_sys::mlx_distributed_recv_like(
            &mut result,
            x.as_ptr(),
            src,
            group_handle(group),
            stream.as_ptr(),
        );
        if status != SUCCESS {
            return Err(Exception::custom("mlx_distributed_recv_like failed"));
        }
        Ok(Array::from_ptr(result))
    }
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
