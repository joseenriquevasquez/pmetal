//! Layer assignment solvers for pipeline-parallel inference.
//!
//! Determines how to partition a model's decoder layers across multiple nodes.
//! Two strategies:
//! - **Proportional**: layers proportional to available RAM (good default)
//! - **Bandwidth-aware**: minimize bottleneck link cost
//!
//! With 2-4 nodes (typical Apple Silicon home cluster), exhaustive search
//! over contiguous splits is feasible — no MILP solver needed.

use std::ops::Range;

/// Divide `num_layers` across nodes proportionally to `available_ram`.
///
/// Returns contiguous, non-overlapping layer ranges that cover `0..num_layers`.
#[allow(clippy::single_range_in_vec_init)]
pub fn assign_layers_proportional(num_layers: usize, available_ram: &[u64]) -> Vec<Range<usize>> {
    let world_size = available_ram.len();
    assert!(world_size > 0, "need at least one node");
    assert!(num_layers > 0, "need at least one layer");
    assert!(
        num_layers >= world_size,
        "more nodes ({world_size}) than layers ({num_layers})"
    );

    if world_size == 1 {
        return vec![0..num_layers];
    }

    let total_ram: f64 = available_ram.iter().sum::<u64>() as f64;
    let mut assignments = Vec::with_capacity(world_size);
    let mut start = 0;

    for (i, &ram) in available_ram.iter().enumerate() {
        if i == world_size - 1 {
            // Last node gets all remaining layers
            assignments.push(start..num_layers);
        } else {
            let proportion = ram as f64 / total_ram;
            let remaining_nodes = world_size - i - 1;
            let remaining_layers = num_layers - start;
            let max_for_this = remaining_layers - remaining_nodes; // leave >=1 per remaining node
            let count = (proportion * num_layers as f64).round() as usize;
            let count = count.clamp(1, max_for_this);
            assignments.push(start..start + count);
            start += count;
        }
    }

    assignments
}

/// Divide layers to minimize bottleneck latency, accounting for per-node bandwidth.
///
/// `bandwidths[i]` is the link bandwidth (bytes/sec) from node i to node i+1.
/// Nodes with higher bandwidth can handle the activation transfer cost of more layers.
///
/// For 2 nodes: exhaustive search over all split points.
/// For 3+ nodes: heuristic weighted by bandwidth * ram.
pub fn assign_layers_bandwidth_aware(
    num_layers: usize,
    available_ram: &[u64],
    bandwidths: &[u64],
) -> Vec<Range<usize>> {
    let world_size = available_ram.len();
    assert_eq!(world_size, bandwidths.len());

    if world_size <= 1 {
        return assign_layers_proportional(num_layers, available_ram);
    }

    if world_size == 2 {
        return assign_two_nodes(num_layers, available_ram, bandwidths);
    }

    if world_size == 3 {
        return assign_three_nodes(num_layers, available_ram, bandwidths);
    }

    // 4+ nodes: weighted proportional
    let weights: Vec<u64> = available_ram
        .iter()
        .zip(bandwidths.iter())
        .map(|(&r, &b)| {
            let r_mb = (r / 1_000_000).max(1);
            let b_mb = (b / 1_000_000).max(1);
            r_mb * b_mb
        })
        .collect();
    assign_layers_proportional(num_layers, &weights)
}

/// Exhaustive search for 2-node split.
fn assign_two_nodes(
    num_layers: usize,
    _available_ram: &[u64],
    bandwidths: &[u64],
) -> Vec<Range<usize>> {
    let mut best_split = 1;
    let mut best_cost = f64::MAX;

    for split in 1..num_layers {
        let cost_0 = split as f64 / bandwidths[0].max(1) as f64;
        let cost_1 = (num_layers - split) as f64 / bandwidths[1].max(1) as f64;
        let max_cost = cost_0.max(cost_1);
        if max_cost < best_cost {
            best_cost = max_cost;
            best_split = split;
        }
    }

    vec![0..best_split, best_split..num_layers]
}

/// Exhaustive search for 3-node split.
fn assign_three_nodes(
    num_layers: usize,
    _available_ram: &[u64],
    bandwidths: &[u64],
) -> Vec<Range<usize>> {
    let mut best = (1usize, 2usize);
    let mut best_cost = f64::MAX;

    for s1 in 1..num_layers - 1 {
        for s2 in s1 + 1..num_layers {
            let cost_0 = s1 as f64 / bandwidths[0].max(1) as f64;
            let cost_1 = (s2 - s1) as f64 / bandwidths[1].max(1) as f64;
            let cost_2 = (num_layers - s2) as f64 / bandwidths[2].max(1) as f64;
            let max_cost = cost_0.max(cost_1).max(cost_2);
            if max_cost < best_cost {
                best_cost = max_cost;
                best = (s1, s2);
            }
        }
    }

    vec![0..best.0, best.0..best.1, best.1..num_layers]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_equal() {
        let r = assign_layers_proportional(32, &[16_000, 16_000]);
        assert_eq!(r, vec![0..16, 16..32]);
    }

    #[test]
    fn proportional_3x1() {
        let r = assign_layers_proportional(32, &[48_000, 16_000]);
        assert_eq!(r, vec![0..24, 24..32]);
    }

    #[test]
    fn proportional_three_equal() {
        let r = assign_layers_proportional(30, &[10_000, 10_000, 10_000]);
        assert_eq!(r, vec![0..10, 10..20, 20..30]);
    }

    #[test]
    fn bandwidth_two_nodes() {
        // Node 0 has 2x bandwidth → should get ~2x layers
        let r = assign_layers_bandwidth_aware(30, &[16_000, 16_000], &[200_000, 100_000]);
        assert_eq!(r.len(), 2);
        assert!(
            r[0].len() > r[1].len(),
            "faster node should get more layers"
        );
    }

    #[test]
    fn bandwidth_three_nodes() {
        let r = assign_layers_bandwidth_aware(
            30,
            &[16_000, 16_000, 16_000],
            &[100_000, 100_000, 100_000],
        );
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].start, 0);
        assert_eq!(r[2].end, 30);
        // All ranges should be contiguous
        assert_eq!(r[0].end, r[1].start);
        assert_eq!(r[1].end, r[2].start);
    }

    #[test]
    fn minimum_one_layer_per_node() {
        let r = assign_layers_proportional(4, &[100, 100, 100, 100]);
        assert_eq!(r.len(), 4);
        for range in &r {
            assert!(!range.is_empty(), "each node must get at least one layer");
        }
    }
}
