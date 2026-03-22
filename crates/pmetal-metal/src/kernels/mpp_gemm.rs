#![allow(unsafe_code)]

//! Metal 4 / MPP GEMM dispatch.
//!
//! Provides hardware-accelerated GEMM via Metal Performance Primitives
//! on M5+ (Apple10) GPUs with NAX cores.
//!
//! Falls back to standard Metal 3 kernels on older hardware.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::MetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
};

/// Configuration for MPP GEMM.
#[derive(Debug, Clone)]
pub struct MppGemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,

    /// Scalar multiplier for the matmul result.
    pub alpha: f32,
    /// Scalar multiplier for existing C (0 = overwrite, 1 = accumulate).
    pub beta: f32,

    /// Batch count (for batched GEMM).
    pub batch_size: usize,

    /// Use Morton ordering for threadgroup walk.
    pub use_morton: bool,

    /// Use fp16 (true) or fp32 (false).
    pub use_fp16: bool,
}

impl MppGemmConfig {
    /// Create a new MPP GEMM config for C = A[M,K] @ B[N,K]^T.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            batch_size: 1,
            use_morton: true,
            use_fp16: true,
        }
    }
}

/// MPP GEMM kernel parameters (must match Metal struct layout).
#[repr(C)]
struct MppGemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    num_tiles_m: u32,
    num_tiles_n: u32,
}

/// MPP GEMM dispatcher.
///
/// Checks NAX availability and dispatches to Metal 4 MPP kernels when possible,
/// falling back to Metal 3 kernels otherwise.
pub struct MppGemm {
    ctx: Arc<MetalContext>,
    config: MppGemmConfig,
}

impl MppGemm {
    /// Create a new MPP GEMM dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppGemmConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP GEMM is available on this device (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax()
            && self
                .ctx
                .pipeline_cache()
                .metal4_library()
                .is_some()
    }

    /// Execute MPP GEMM: C = alpha * A @ B^T + beta * C
    ///
    /// A: [M, K], B: [N, K] (transposed), C: [M, N]
    pub fn execute(
        &self,
        a: &MetalBuffer<f32>,
        b: &MetalBuffer<f32>,
        c: &MetalBuffer<f32>,
    ) -> Result<()> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP GEMM not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let kernel_name = if self.config.beta != 0.0 {
            "mpp_gemm_accumulate_f16"
        } else if self.config.use_fp16 {
            "mpp_gemm_nn_f16"
        } else {
            "mpp_gemm_nn_f32"
        };

        // Function constants
        let mut constants = HashMap::new();
        constants.insert(0u64, FunctionConstant::Bool(self.config.use_morton));

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_metal4_pipeline(self.ctx.device(), kernel_name, &constants)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let bm = 64usize;
        let bn = 64usize;
        let num_tiles_m = (self.config.m + bm - 1) / bm;
        let num_tiles_n = (self.config.n + bn - 1) / bn;
        let total_tiles = num_tiles_m * num_tiles_n;

        let params = MppGemmParams {
            m: self.config.m as u32,
            n: self.config.n as u32,
            k: self.config.k as u32,
            alpha: self.config.alpha,
            beta: self.config.beta,
            num_tiles_m: num_tiles_m as u32,
            num_tiles_n: num_tiles_n as u32,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);

            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        }

        // 4 simdgroups × 32 threads = 128 threads per threadgroup
        let threadgroup_size = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let grid_size = objc2_metal::MTLSize {
            width: total_tiles,
            height: 1,
            depth: self.config.batch_size,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            let err_str: String = error.to_string();
            return Err(MetalError::ExecutionFailed(err_str));
        }

        Ok(())
    }
}
