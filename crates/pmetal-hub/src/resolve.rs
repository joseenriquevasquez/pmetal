//! Model and dataset path resolution.
//!
//! Provides canonical `resolve_model_path()` used by all consumers (CLI, GUI,
//! Python bindings, orchestrator) to resolve a HuggingFace model ID or local
//! path to a directory on disk, downloading if necessary.

use std::path::PathBuf;

/// Check if a string looks like a HuggingFace model ID (e.g., `"org/model"`)
/// rather than a local filesystem path.
pub fn is_hf_id(s: &str) -> bool {
    !s.starts_with('/') && !s.starts_with('.') && s.contains('/')
}

/// Resolve a model identifier to a local directory path.
///
/// If `model_id` looks like a HuggingFace ID and the path does not already
/// exist on disk, the model is downloaded via [`crate::download_model`].
/// Otherwise the string is treated as a local path.
pub async fn resolve_model_path(model_id: &str) -> pmetal_core::Result<PathBuf> {
    if is_hf_id(model_id) && !PathBuf::from(model_id).exists() {
        crate::download_model(model_id, None, None).await
    } else {
        Ok(PathBuf::from(model_id))
    }
}
