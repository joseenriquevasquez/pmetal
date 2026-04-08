//! Metal 4 command buffer wrapper with MPP dispatch support.
//!
//! Metal 4 replaces `MTLCommandBuffer` with a new type that supports MPP
//! (Mesh Processing Pipeline) and NAX (Neural Accelerator) kernel dispatch.
//! This module wraps those Metal 4 command buffer types and exposes a Rust
//! interface for encoding MPP GEMM, MPP flash attention, and MPP quantized
//! GEMM operations.
//!
//! # Status
//!
//! Stub — Task 9 will implement the command buffer type using Metal 4's
//! `MTLCommandBuffer4` API and wire it into [`CommandAllocatorPool`] for
//! allocator-backed encoding.
//!
//! [`CommandAllocatorPool`]: super::allocator_pool::CommandAllocatorPool

/// A single Metal 4 command buffer ready for MPP kernel encoding.
///
/// Wraps the platform `MTLCommandBuffer4` object and provides type-safe
/// methods for encoding MPP GEMM, flash attention, and quantized GEMM
/// operations targeting NAX cores.
pub struct Metal4CommandBuffer;
