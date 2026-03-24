//! Expert parallelism for MoE (Mixture of Experts) models.
//!
//! Distributes routed experts across nodes so each node holds a subset
//! of all experts. Tokens are dispatched to expert-owning nodes via
//! point-to-point MLX `send`/`recv`, computed locally, and results
//! returned to the originating node.
//!
//! # Design
//!
//! - **Replicated routing**: All nodes compute the full routing scores
//!   (cheap — just a small gate projection). This avoids communicating
//!   routing decisions.
//!
//! - **Expert-sharded weights**: Each node loads only its assigned expert
//!   weights. For Qwen 3.5 with 512 experts on 4 nodes: 128 experts/node.
//!
//! - **All-to-all dispatch**: Tokens are sorted by destination rank,
//!   batched, and sent via MLX point-to-point ops. Each rank computes
//!   its local experts on received tokens and returns results.
//!
//! # Reference
//!
//! - GShard: Sharded expert placement with capacity factor
//! - DeepEP: RDMA-based expert dispatch (adapted for JACCL/Thunderbolt)
//! - FlashMoE: Fused dispatch+compute+combine (future optimization)

pub mod capacity;
pub mod dispatch;
pub mod placement;

pub use capacity::{apply_capacity, CapacityConfig, DropPolicy};
pub use dispatch::ExpertDispatcher;
pub use placement::ExpertPlacement;
