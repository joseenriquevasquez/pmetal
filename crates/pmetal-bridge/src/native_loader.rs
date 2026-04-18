//! Shared loader primitives for the `*_native.rs` model modules.
//!
//! Each native-architecture file previously open-coded the same ~65 LOC
//! shard-discovery + bulk-load block (look for `model.safetensors`, fall
//! back to `model.safetensors.index.json`, read every shard into one
//! keyed map). Bug fixes to error handling had to land in four places.
//!
//! This module hosts the I/O-only portion of the pipeline: parsing
//! `config.json`, resolving shard paths, and merging shards into a
//! `HashMap<String, InlineArray>`. Model-specific weight sanitization,
//! key normalization, and per-layer slicing still live in each arch's
//! own `load_model`.
//!
//! Error-message shape is preserved byte-for-byte from the original
//! callers so tests asserting on `.to_string()` output keep working.

use crate::InlineArray;
use crate::inline_array as bridge;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Read `model_dir/config.json` as text.
///
/// Error format matches what the native loaders produced before
/// extraction: `"failed to read {path}: {io-error}"`.
pub fn read_config_json(model_dir: &Path) -> Result<String, String> {
    let path = model_dir.join("config.json");
    std::fs::read_to_string(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

/// Resolve the set of `.safetensors` shard paths under `model_dir`,
/// using the same fallback order as every pre-existing native loader:
///
/// 1. `model.safetensors` (single-file checkpoint).
/// 2. `model.safetensors.index.json` — parse `weight_map` and collect
///    unique shard filenames. Rejects entries containing `..` or a
///    leading `/` to prevent path traversal.
/// 3. Otherwise, returns an error that matches the legacy wording so
///    callers' user-facing messages do not regress.
pub fn discover_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single_path = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    if single_path.exists() {
        return Ok(vec![single_path]);
    }
    if !index_path.exists() {
        return Err(format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        ));
    }

    let content = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("failed to read index JSON: {e}"))?;
    let index: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("failed to parse index JSON: {e}"))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "index JSON missing weight_map".to_string())?;

    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for shard_file in weight_map.values() {
        let name = shard_file
            .as_str()
            .ok_or_else(|| "shard filename is not a string".to_string())?;
        if seen.insert(name.to_string()) {
            if name.contains("..") || name.starts_with('/') {
                return Err(format!("shard filename contains path traversal: {name}"));
            }
            paths.push(model_dir.join(name));
        }
    }
    Ok(paths)
}

/// Load every tensor from each shard in `shard_paths`, merging into one
/// keyed map. On key collisions across shards, later shards win — this
/// matches the behavior of the previous inline loaders, which all used
/// `HashMap::insert` in shard-iteration order.
///
/// `model_dir` is only used in the "no weights loaded" error message.
pub fn load_shards_into_map(
    shard_paths: &[PathBuf],
    model_dir: &Path,
) -> Result<HashMap<String, InlineArray>, String> {
    let mut raw: HashMap<String, InlineArray> = HashMap::new();
    for shard_path in shard_paths {
        let path_str = shard_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 shard path: {:?}", shard_path))?;
        let entries = bridge::load_safetensors_shard(path_str)
            .ok_or_else(|| format!("failed to load shard: {path_str}"))?;
        for (key, arr) in entries {
            raw.insert(key, arr);
        }
    }
    if raw.is_empty() {
        return Err(format!("no weights loaded from {}", model_dir.display()));
    }
    Ok(raw)
}
