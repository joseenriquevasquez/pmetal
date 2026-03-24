//! Pipeline training schedules for distributed training.
//!
//! Coordinates forward and backward passes across pipeline stages
//! to maximize GPU utilization and minimize the pipeline bubble.
//!
//! # Schedules
//!
//! - **1F1B**: One Forward, One Backward. Classic schedule from PipeDream.
//!   Warmup phase with (num_stages - rank - 1) forwards, then alternating
//!   forward/backward in steady state.
//!
//! - **Zero Bubble V (ZBV)**: Splits backward into B (activation gradient)
//!   and W (weight gradient). W has no cross-stage dependency so it can fill
//!   bubble slots, achieving near-zero pipeline bubble.
//!
//! # Reference
//!
//! - PipeDream: Generalized Pipeline Parallelism (Narayanan et al., 2019)
//! - Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)

pub mod schedule_1f1b;
pub mod schedule_zbv;

pub use schedule_1f1b::{MicroBatchAction, schedule_1f1b};
pub use schedule_zbv::{ZBAction, schedule_zero_bubble};
