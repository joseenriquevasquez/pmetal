//! Command allocator pool for Metal 4 / MPP command buffers.
//!
//! On Metal 4, command buffers are allocated from a fixed-size pool to avoid
//! per-frame heap allocation pressure. Each command buffer owns its own
//! `MTLCommandAllocator` whose memory is recycled on completion rather than
//! freed and reallocated.
//!
//! # Status
//!
//! Stub — Task 7 will implement the allocator pool using Metal 4's
//! `MTLCommandAllocator` API and double-/triple-buffering recycle logic.

/// Pool of reusable Metal 4 command allocators.
///
/// Allocators are checked out before encoding a command buffer and returned
/// to the pool when the GPU signals completion. The pool is bounded to avoid
/// unbounded heap growth under backpressure.
pub struct CommandAllocatorPool;
