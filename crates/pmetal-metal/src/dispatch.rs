//! Kernel dispatch router — selects Metal 3 or Metal 4 backend per operation.
//!
//! [`KernelDispatch`] is the single entry point for all kernel routing decisions.
//! It holds both backends and exposes methods that return the appropriate
//! [`KernelBackend`] reference for a given operation shape.
//!
//! # Routing policy
//!
//! - **`backend_for_gemm(m, n, k)`**: Returns Metal 4 when `#[cfg(has_metal4)]`
//!   and the Metal 4 backend is present and [`KernelBackend::should_handle_gemm`]
//!   returns `true`. Falls back to Metal 3 otherwise (decode M=1, unaligned K,
//!   or Metal 4 not compiled in).
//!
//! - **`preferred_backend()`**: Returns Metal 4 when available; Metal 3
//!   otherwise. Used for non-shape-dependent ops (e.g., MPP flash attention)
//!   where Metal 4 unconditionally wins.
//!
//! - **`metal3()`**: Explicit Metal 3 accessor for callers that need to bypass
//!   routing (e.g., fallback paths inside the Metal 4 backend itself).
//!
//! # Metal 4 placeholder
//!
//! The `metal4` field is `Option<()>` under `#[cfg(has_metal4)]` and will be
//! replaced with `Option<Metal4Backend>` in Task 10 once the Metal 4 backend
//! is implemented. The routing methods already contain the correct conditional
//! structure so Task 10 only needs to change the type and fill in the `Some`
//! branch.

use std::sync::Arc;

use crate::backend::KernelBackend;
use crate::context::MetalContext;
use crate::metal3_backend::Metal3Backend;

// ============================================================================
// KernelDispatch
// ============================================================================

/// Routes kernel operations to the appropriate backend (Metal 3 or Metal 4).
///
/// On M5+ hardware built with the Metal 4 SDK, operations are routed to the
/// Metal 4 / MPP backend for supported shapes. All other operations fall back
/// to the Metal 3 backend.
///
/// Construct once per inference or training session via [`KernelDispatch::new`]
/// and share freely — `KernelDispatch` is `Send + Sync` because both backends are.
pub struct KernelDispatch {
    metal3: Metal3Backend,
    /// Placeholder for the Metal 4 backend (Task 10).
    ///
    /// `Option<()>` will become `Option<Metal4Backend>` when Task 10 lands.
    /// Gated by `#[cfg(has_metal4)]` so the field does not exist at all on
    /// non-Metal-4 builds, keeping the struct layout identical to today.
    #[cfg(has_metal4)]
    metal4: Option<()>,
}

impl KernelDispatch {
    /// Create a new dispatch router backed by the given Metal context.
    ///
    /// Both the Metal 3 backend (and the Metal 4 placeholder when compiled in)
    /// are initialised from the shared `ctx`.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        let metal3 = Metal3Backend::new(ctx.clone());
        Self {
            metal3,
            #[cfg(has_metal4)]
            metal4: None, // Metal4Backend not implemented yet — Task 10
        }
    }

    // ---- Routing methods ----------------------------------------------------

    /// Select the backend for a GEMM of the given shape.
    ///
    /// Returns Metal 4 when:
    /// - `has_metal4` is compiled in, **and**
    /// - `self.metal4` is `Some`, **and**
    /// - [`KernelBackend::should_handle_gemm`] on the Metal 4 backend returns `true`.
    ///
    /// Falls back to Metal 3 for:
    /// - Decode / matvec (M = 1) — MPP tile constraints waste threads.
    /// - K not divisible by 32 — NAX alignment requirement.
    /// - All non-Metal-4 builds.
    pub fn backend_for_gemm(&self, _m: usize, _n: usize, _k: usize) -> &dyn KernelBackend {
        // Task 10 will replace this with:
        //
        // #[cfg(has_metal4)]
        // if let Some(ref m4) = self.metal4 {
        //     if m4.should_handle_gemm(_m, _n, _k) {
        //         return m4;
        //     }
        // }
        //
        // For now, always return Metal 3.
        &self.metal3
    }

    /// Select the preferred backend for non-shape-dependent operations.
    ///
    /// Returns Metal 4 when available (e.g., MPP flash attention); Metal 3
    /// otherwise. Operations that always require a specific backend (e.g.,
    /// training losses only on Metal 3) should call the backend directly via
    /// [`metal3`][KernelDispatch::metal3].
    pub fn preferred_backend(&self) -> &dyn KernelBackend {
        // Task 10 will add:
        //
        // #[cfg(has_metal4)]
        // if let Some(ref m4) = self.metal4 {
        //     return m4;
        // }
        &self.metal3
    }

    /// Return the Metal 3 backend directly.
    ///
    /// Used for explicit fallback paths and for operations that Metal 4 does
    /// not support (training losses, RoPE, etc.) as advertised by
    /// [`BackendCaps::metal4`][crate::backend::BackendCaps::metal4].
    pub fn metal3(&self) -> &Metal3Backend {
        &self.metal3
    }
}
