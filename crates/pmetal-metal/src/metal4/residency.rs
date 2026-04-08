//! Residency manager for Metal 4 / MPP weight tensors.
//!
//! Metal 4 introduces explicit residency sets (`MTLResidencySet`) that let the
//! driver pre-page weight buffers into GPU-accessible memory before command
//! buffer execution begins, eliminating mid-execution page faults for large
//! models.
//!
//! # Status
//!
//! Stub — Task 8 will implement residency set creation, buffer registration,
//! and the commit/request/endResidency lifecycle tied to command buffer
//! submission.

/// Manages Metal 4 residency sets for model weight buffers.
///
/// Weight buffers are registered once at model load time and kept resident
/// across inference steps. Activation buffers are registered transiently
/// per-step and released after GPU completion.
pub struct ResidencyManager;
