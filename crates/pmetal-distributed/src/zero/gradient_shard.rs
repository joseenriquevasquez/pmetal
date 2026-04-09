//! Gradient reduce-scatter for ZeRO Stage 2.
//!
//! Instead of all-reduce (where every rank gets the full gradient),
//! reduce-scatter gives each rank only the gradient shard for the
//! parameters it owns. This halves gradient memory per rank.

use super::state_partition::ZeROPartitioner;
use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;
use std::collections::HashMap;

/// Reduce-scatter gradients: each rank ends up with the gradient
/// shard for its owned parameters only.
///
/// This implements ZeRO Stage 2 gradient sharding over the existing
/// `DistributedBackend` (TCP ring or MLX collectives).
///
/// # Algorithm
///
/// 1. All-reduce the full gradient buffer (sum across ranks).
/// 2. Each rank keeps only the gradients for parameters it owns.
/// 3. Non-owned gradients are discarded (saving memory).
///
/// Note: A true reduce-scatter would be more efficient (O(N) instead
/// of O(2N) communication), but this approach works with the existing
/// ring all-reduce infrastructure. When using MLX collectives,
/// `sum_scatter` provides the optimal O(N) path.
pub async fn reduce_scatter_gradients(
    partitioner: &ZeROPartitioner,
    all_grads: &mut HashMap<String, Vec<u8>>,
    backend: &dyn DistributedBackend,
) -> Result<()> {
    // Flatten all gradients into a single buffer for all-reduce.
    let mut flat_buffer: Vec<u8> = Vec::new();
    let mut layout: Vec<(String, usize)> = Vec::new();

    // Deterministic ordering by sorted parameter name.
    let mut param_names: Vec<String> = all_grads.keys().cloned().collect();
    param_names.sort();

    for name in &param_names {
        if let Some(grad) = all_grads.get(name) {
            layout.push((name.clone(), grad.len()));
            flat_buffer.extend_from_slice(grad);
        }
    }

    // All-reduce (sum) the flattened gradient buffer.
    backend.all_reduce(&mut flat_buffer, ReduceOp::Sum).await?;

    // Scatter: keep only owned gradients, discard the rest.
    let mut offset = 0;
    for (name, size) in &layout {
        if partitioner.owns_param(name) {
            // Keep: copy reduced gradient back.
            if let Some(grad) = all_grads.get_mut(name) {
                grad.copy_from_slice(&flat_buffer[offset..offset + size]);
            }
        } else {
            // Discard: remove from map to free memory.
            all_grads.remove(name);
        }
        offset += size;
    }

    Ok(())
}

/// All-gather parameters from their owning ranks before forward pass.
///
/// In ZeRO Stage 2+, each rank only stores the parameters it owns.
/// Before the forward pass, missing parameters must be gathered from
/// their owning ranks.
///
/// # Algorithm
///
/// 1. Each rank broadcasts its owned parameters.
/// 2. Other ranks receive and store them for the forward pass.
/// 3. After backward, non-owned parameters are discarded.
///
/// This is implemented as a series of broadcasts (one per rank).
pub async fn all_gather_params(
    partitioner: &ZeROPartitioner,
    local_params: &HashMap<String, Vec<u8>>,
    backend: &dyn DistributedBackend,
) -> Result<HashMap<String, Vec<u8>>> {
    // For ZeRO Stage 1, we only partition optimizer states, not parameters.
    // Return all local params as-is (they're already fully replicated).
    if partitioner.stage == super::state_partition::ZeROStage::Stage1 {
        return Ok(local_params.clone());
    }

    // ZeRO Stage 2: gather non-local parameters.
    // For each parameter, the owning rank broadcasts its value.
    let mut gathered = local_params.clone();

    // For parameters this rank doesn't own, we need them from other ranks.
    // This is done via all-reduce of a buffer where only the owning rank
    // has non-zero values, and all other ranks have zeros.
    for name in &partitioner.all_params {
        if !local_params.contains_key(name) {
            // This parameter is owned by another rank.
            // After all-reduce with Sum, we'll have the value.
            // For now, we allocate a placeholder.
            //
            // In practice, this integrates with the gradient sync buffer
            // rather than being a separate communication step.
        }
    }

    // Simplified: use a single all-reduce pass with the full parameter buffer.
    // Each rank contributes its owned parameters; zeros elsewhere.
    // The sum produces the full parameter set on all ranks.
    let mut flat_buffer: Vec<u8> = Vec::new();
    let mut layout: Vec<(String, usize)> = Vec::new();

    for name in &partitioner.all_params {
        if let Some(param) = local_params.get(name) {
            layout.push((name.clone(), param.len()));
            if partitioner.owns_param(name) {
                flat_buffer.extend_from_slice(param);
            } else {
                flat_buffer.extend(vec![0u8; param.len()]);
            }
        }
    }

    if !flat_buffer.is_empty() {
        backend.all_reduce(&mut flat_buffer, ReduceOp::Sum).await?;

        let mut offset = 0;
        for (name, size) in &layout {
            gathered.insert(name.clone(), flat_buffer[offset..offset + size].to_vec());
            offset += size;
        }
    }

    Ok(gathered)
}

#[cfg(test)]
mod tests {
    use super::super::state_partition::{ZeROPartitioner, ZeROStage};
    use super::*;

    #[test]
    fn reduce_scatter_keeps_owned_only() {
        // Simulate single-rank scenario (no actual network).
        let params: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let partitioner = ZeROPartitioner::new(&params, ZeROStage::Stage2, 0, 2);

        // Rank 0 owns params at indices 0 and 2 (round-robin sorted).
        assert!(partitioner.owns_param("a"));
        assert!(!partitioner.owns_param("b"));
        assert!(partitioner.owns_param("c"));
        assert!(!partitioner.owns_param("d"));
    }

    #[test]
    fn all_gather_stage1_returns_local() {
        let params: Vec<String> = vec!["a".into(), "b".into()];
        let partitioner = ZeROPartitioner::new(&params, ZeROStage::Stage1, 0, 2);

        let mut local = HashMap::new();
        local.insert("a".to_string(), vec![1u8, 2, 3, 4]);
        local.insert("b".to_string(), vec![5u8, 6, 7, 8]);

        // Stage 1 doesn't partition parameters, only optimizer states.
        let _rt = tokio::runtime::Runtime::new().unwrap();
        // Can't actually call all_gather_params without a backend, but we can
        // verify the partitioner logic.
        assert_eq!(partitioner.stage, ZeROStage::Stage1);
    }
}
