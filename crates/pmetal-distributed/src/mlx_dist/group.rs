//! Safe wrapper around `mlx_distributed_group`.
//!
//! The distributed group represents a communication group of processes.
//! It is initialized once at process startup via [`DistributedGroup::init`]
//! and lives for the process lifetime (the MLX C API does not expose a
//! public free function for groups — they are reference-counted internally).

// ── Local FFI declarations for MLX distributed group API ─────────────────────
//
// These mirror the `mlx_distributed_*` symbols from mlx-sys / mlx-c but are
// declared here so that pmetal-distributed does not need the mlx-sys crate.
// All types are plain C structs with a single `ctx: *mut c_void` field.

use std::ffi::c_void;

/// Opaque handle for an MLX distributed communication group.
///
/// Matches `mlx_distributed_group` in `mlx/c/distributed.h`.  The `ctx`
/// field is an opaque pointer to the C++ `Group` shared_ptr wrapper.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MlxDistributedGroup {
    pub(crate) ctx: *mut c_void,
}

#[allow(unsafe_code)]
unsafe impl Send for MlxDistributedGroup {}
#[allow(unsafe_code)]
unsafe impl Sync for MlxDistributedGroup {}

#[allow(unsafe_code)]
unsafe extern "C" {
    fn mlx_distributed_is_available() -> bool;
    fn mlx_distributed_init(strict: bool) -> MlxDistributedGroup;
    fn mlx_distributed_group_rank(group: MlxDistributedGroup) -> i32;
    fn mlx_distributed_group_size(group: MlxDistributedGroup) -> i32;
    fn mlx_distributed_group_split(
        group: MlxDistributedGroup,
        color: i32,
        key: i32,
    ) -> MlxDistributedGroup;
}

/// A communication group for distributed operations.
///
/// Wraps the opaque `mlx_distributed_group` handle from the MLX C API.
/// Groups are process-lifetime objects: initialized once via [`init`] and
/// valid until process exit.
///
/// [`init`]: DistributedGroup::init
pub struct DistributedGroup {
    pub(crate) inner: MlxDistributedGroup,
}

// SAFETY: The MLX distributed group is thread-safe — the underlying C++
// Group object uses shared_ptr and all collective operations synchronize
// via the communication backend (JACCL/Ring/MPI).
#[allow(unsafe_code)]
unsafe impl Send for DistributedGroup {}
#[allow(unsafe_code)]
unsafe impl Sync for DistributedGroup {}

impl DistributedGroup {
    /// Check if any distributed backend is available.
    ///
    /// Returns `true` if the process was launched with a distributed
    /// configuration (e.g., via `mlx.launch` or manual env vars).
    #[allow(unsafe_code)]
    pub fn is_available() -> bool {
        // SAFETY: FFI call with no side effects.
        unsafe { mlx_distributed_is_available() }
    }

    /// Initialize the distributed group.
    ///
    /// When `strict` is `true`, initialization fails if no backend is
    /// available. When `false`, returns `None` if unavailable.
    ///
    /// This should be called once at process startup. The returned group
    /// is valid for the process lifetime.
    #[allow(unsafe_code)]
    pub fn init(strict: bool) -> Option<Self> {
        // SAFETY: FFI call that initializes the distributed backend.
        // Returns a group with ctx=nullptr if unavailable and !strict.
        let group = unsafe { mlx_distributed_init(strict) };

        if group.ctx.is_null() {
            None
        } else {
            Some(Self { inner: group })
        }
    }

    /// Get this process's rank within the group (0-indexed).
    #[allow(unsafe_code)]
    pub fn rank(&self) -> i32 {
        // SAFETY: self.inner is a valid, non-null group handle.
        unsafe { mlx_distributed_group_rank(self.inner) }
    }

    /// Get the total number of processes in the group.
    #[allow(unsafe_code)]
    pub fn size(&self) -> i32 {
        // SAFETY: self.inner is a valid, non-null group handle.
        unsafe { mlx_distributed_group_size(self.inner) }
    }

    /// Split the group into sub-groups.
    ///
    /// Processes with the same `color` end up in the same sub-group.
    /// `key` controls the rank ordering within the new group (use -1
    /// to preserve the original ordering).
    ///
    /// Returns `None` if the split produces an empty group for this rank.
    #[allow(unsafe_code)]
    pub fn split(&self, color: i32, key: i32) -> Option<Self> {
        // SAFETY: self.inner is a valid group handle. Split returns a
        // new group handle (possibly with ctx=nullptr for empty groups).
        let group = unsafe { mlx_distributed_group_split(self.inner, color, key) };

        if group.ctx.is_null() {
            None
        } else {
            Some(Self { inner: group })
        }
    }

    /// Create a null group handle (represents "use default group").
    ///
    /// When passed to collective ops, the default global group is used.
    pub(crate) fn null_handle() -> MlxDistributedGroup {
        MlxDistributedGroup {
            ctx: std::ptr::null_mut(),
        }
    }
}

impl std::fmt::Debug for DistributedGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.inner.ctx.is_null() {
            write!(f, "DistributedGroup(null)")
        } else {
            write!(
                f,
                "DistributedGroup(rank={}, size={})",
                self.rank(),
                self.size()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handle_is_null() {
        let handle = DistributedGroup::null_handle();
        assert!(handle.ctx.is_null());
    }

    #[test]
    fn is_available_does_not_panic() {
        // This test just verifies the FFI call doesn't crash.
        // It will return false in most test environments.
        let _available = DistributedGroup::is_available();
    }

    #[test]
    fn init_non_strict_returns_none_when_unavailable() {
        // In a non-distributed test environment, init(false) should return None.
        if !DistributedGroup::is_available() {
            assert!(DistributedGroup::init(false).is_none());
        }
    }
}
