//! Context parallelism for long-sequence inference.
//!
//! Splits sequences across ranks along the sequence dimension, enabling
//! inference on sequences longer than a single node's memory can hold.
//! KV blocks are exchanged between ranks in a ring pattern.
//!
//! # Design
//!
//! - **Pass-KV** (prefill): Each rank holds a local Q chunk and passes
//!   KV blocks around the ring. Partial attention is computed at each
//!   step and accumulated using online softmax.
//!
//! - **Pass-Q** (decode): Each rank holds the full KV cache and passes
//!   Q blocks around the ring. Better for decode where KV is large.
//!
//! # Hybrid Model Consideration
//!
//! Only attention layers participate in context parallelism.
//! GDN layers process sequences recurrently and keep state local —
//! no cross-rank communication needed for GDN.
//!
//! # Reference
//!
//! - Ring Attention (Liu et al., 2023)
//! - Context Parallelism for scaling million-token inference (Meta, 2025)

pub mod ring_attention;
pub mod sequence_split;

pub use ring_attention::{CPMode, ring_attention_forward};
pub use sequence_split::{gather_sequence, split_sequence};
