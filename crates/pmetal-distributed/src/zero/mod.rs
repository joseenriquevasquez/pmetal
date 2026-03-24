//! ZeRO (Zero Redundancy Optimizer) style memory optimization.
//!
//! Partitions optimizer states and gradients across ranks to reduce
//! per-node memory usage during distributed training.
//!
//! # Stages
//!
//! - **Stage 1**: Partition optimizer states (Adam momentum/variance).
//!   Each rank only stores optimizer state for its assigned parameters.
//!   Memory reduction: ~4x for Adam (fp32 m/v per parameter).
//!
//! - **Stage 2**: + Reduce-scatter gradients. Instead of all-reduce
//!   (where every rank gets the full gradient), each rank accumulates
//!   only its own shard via reduce-scatter.
//!
//! # MoE-Aware Partitioning
//!
//! For MoE models, parameters are naturally grouped by expert.
//! ZeRO partitioning respects this structure: each rank's optimizer
//! state aligns with the experts it owns in expert parallelism.
//!
//! # Reference
//!
//! - ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
//! - FSDP2 (PyTorch): Per-parameter sharding

pub mod gradient_shard;
pub mod state_partition;

pub use gradient_shard::{all_gather_params, reduce_scatter_gradients};
pub use state_partition::{ZeROPartitioner, ZeROStage};
