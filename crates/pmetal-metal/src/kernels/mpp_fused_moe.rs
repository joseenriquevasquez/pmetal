#![allow(unsafe_code)]

//! Metal 4 / MPP Fused MoE expert forward dispatch.
//!
//! Provides hardware-accelerated MoE expert forward passes via Metal Performance
//! Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Replaces the Metal 3 quantized-expert kernels for dense (fp16) MoE models.
//! Uses `matmul2d` with single simdgroup (NAX) for the GEMM steps, and MPP
//! postfix fusion to keep SwiGLU in register space between gate/up and down
//! projections.
//!
//! Kernel families:
//! - `mpp_fused_moe_gate_up_f16` — gate + up GEMMs + SwiGLU postfix
//! - `mpp_fused_moe_down_f16` — down projection GEMM
//! - `mpp_moe_weighted_scatter_f16` — weighted residual accumulation (combine)
//!
//! Grid for gate/up: `[ceil(intermediate/32), ceil(tokens/32), 1]`
//! Grid for down:    `[ceil(hidden/32), ceil(tokens/32), 1]`

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::AsMetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
};

// =============================================================================
// Config
// =============================================================================

/// Configuration for one MPP MoE expert forward pass.
#[derive(Debug, Clone)]
pub struct MppFusedMoEConfig {
    /// Number of tokens dispatched to this expert.
    pub batch_size: usize,
    /// Hidden dimension (input and output of the expert).
    pub hidden_dim: usize,
    /// Expert intermediate dimension (gate/up output).
    pub intermediate_dim: usize,
}

impl MppFusedMoEConfig {
    /// Create a new expert config.
    pub fn new(batch_size: usize, hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self { batch_size, hidden_dim, intermediate_dim }
    }
}

/// Configuration for the weighted scatter (combine) kernel.
#[derive(Debug, Clone)]
pub struct MppMoEScatterConfig {
    /// Number of tokens being accumulated.
    pub num_tokens: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

// =============================================================================
// Metal-side parameter blocks
// =============================================================================

/// Must match `MppMoEParams` in mpp_fused_moe.metal.
#[repr(C)]
struct MppMoEParamsMetal {
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
}

/// Must match `MppMoEScatterParams` in mpp_fused_moe.metal.
#[repr(C)]
struct MppMoEScatterParamsMetal {
    num_tokens: u32,
    hidden_dim: u32,
}

// =============================================================================
// MoE Expert Forward Dispatcher
// =============================================================================

/// MPP Fused MoE expert forward dispatcher.
///
/// Dispatches a full expert forward pass (gate+up SwiGLU + down projection)
/// using MPP matmul2d on M5+ hardware.
pub struct MppFusedMoE {
    ctx: Arc<MetalContext>,
    config: MppFusedMoEConfig,
}

impl MppFusedMoE {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedMoEConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP MoE is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute the gate + up SwiGLU projection synchronously.
    ///
    /// - `input`: `[batch, hidden]`
    /// - `gate_weight`: `[intermediate, hidden]`
    /// - `up_weight`: `[intermediate, hidden]`
    /// - `act_out`: `[batch, intermediate]` — SwiGLU output
    pub fn execute_gate_up(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_gate_up_async(input, gate_weight, up_weight, act_out)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute gate + up SwiGLU asynchronously.
    pub fn execute_gate_up_async(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused MoE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let pipeline = self.get_pipeline("mpp_fused_moe_gate_up_f16")?;

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppMoEParamsMetal {
            batch_size: self.config.batch_size as u32,
            hidden_dim: self.config.hidden_dim as u32,
            intermediate_dim: self.config.intermediate_dim as u32,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate_weight.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up_weight.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(act_out.as_metal_buffer()), 0, 3);
            let p_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 4);
        }

        // Grid: [ceil(intermediate/32), ceil(batch/32), 1]
        let tiles_i = self.config.intermediate_dim.div_ceil(32);
        let tiles_b = self.config.batch_size.div_ceil(32);
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize { width: tiles_i, height: tiles_b, depth: 1 };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }

    /// Execute the down projection synchronously.
    ///
    /// - `act_in`: `[batch, intermediate]` — SwiGLU output from gate_up
    /// - `down_weight`: `[hidden, intermediate]`
    /// - `out`: `[batch, hidden]`
    pub fn execute_down(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_down_async(act_in, down_weight, out)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute down projection asynchronously.
    pub fn execute_down_async(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused MoE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let pipeline = self.get_pipeline("mpp_fused_moe_down_f16")?;

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppMoEParamsMetal {
            batch_size: self.config.batch_size as u32,
            hidden_dim: self.config.hidden_dim as u32,
            intermediate_dim: self.config.intermediate_dim as u32,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(act_in.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(down_weight.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.as_metal_buffer()), 0, 2);
            let p_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 3);
        }

        // Grid: [ceil(hidden/32), ceil(batch/32), 1]
        let tiles_h = self.config.hidden_dim.div_ceil(32);
        let tiles_b = self.config.batch_size.div_ceil(32);
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize { width: tiles_h, height: tiles_b, depth: 1 };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }

    fn get_pipeline(
        &self,
        kernel_name: &str,
    ) -> Result<objc2::rc::Retained<objc2_metal::MTLComputePipelineState>> {
        let constants: HashMap<u64, crate::pipeline::FunctionConstant> = HashMap::new();
        let mut cache = self.ctx.pipeline_cache_mut();
        cache.get_or_create_metal4_pipeline(self.ctx.device(), kernel_name, &constants)
    }
}

// =============================================================================
// Weighted Scatter Dispatcher (MoE combine)
// =============================================================================

/// MPP MoE weighted scatter dispatcher.
///
/// Accumulates expert contributions into an output buffer with router weighting:
/// `out[token] += weight[token] * expert_out[token]`.
pub struct MppMoEScatter {
    ctx: Arc<MetalContext>,
    config: MppMoEScatterConfig,
}

impl MppMoEScatter {
    /// Create a new scatter dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppMoEScatterConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP scatter is available.
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute weighted scatter synchronously.
    ///
    /// `expert_out`: `[num_tokens, hidden]`, `weights`: `[num_tokens]` (fp32),
    /// `accum`: `[num_tokens, hidden]` read-modify-write.
    pub fn execute(
        &self,
        expert_out: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        accum: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_async(expert_out, weights, accum)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously.
    pub fn execute_async(
        &self,
        expert_out: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        accum: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP MoE scatter not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let constants: HashMap<u64, crate::pipeline::FunctionConstant> = HashMap::new();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_metal4_pipeline(
                self.ctx.device(),
                "mpp_moe_weighted_scatter_f16",
                &constants,
            )?
        };

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppMoEScatterParamsMetal {
            num_tokens: self.config.num_tokens as u32,
            hidden_dim: self.config.hidden_dim as u32,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(expert_out.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(accum.as_metal_buffer()), 0, 2);
            let p_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 3);
        }

        // Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let cfg = MppFusedMoEConfig::new(4, 2048, 4096);
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.intermediate_dim, 4096);
    }

    #[test]
    fn test_gate_up_tiles() {
        let cfg = MppFusedMoEConfig::new(1, 2048, 4096);
        let tiles_i = cfg.intermediate_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_i, 128);
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_down_tiles() {
        let cfg = MppFusedMoEConfig::new(32, 2048, 4096);
        let tiles_h = cfg.hidden_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_h, 64);
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_scatter_config() {
        let cfg = MppMoEScatterConfig { num_tokens: 8, hidden_dim: 2048 };
        assert_eq!(cfg.num_tokens, 8);
    }
}
