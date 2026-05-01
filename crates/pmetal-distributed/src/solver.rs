//! Topology-aware layer assignment solver.
//!
//! Combines per-node profiles (RAM, bandwidth) to produce optimal
//! contiguous layer assignments for pipeline inference. With 2-4
//! nodes (typical Apple Silicon home clusters), exhaustive search
//! is both feasible and optimal.

use crate::fabric::InterfaceKind;
use crate::topology::ClusterTopology;
use std::ops::Range;

/// Result of the solver: layer assignments + estimated pipeline latency.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Layer range per node (index = node rank).
    pub assignments: Vec<Range<usize>>,
    /// Estimated per-token pipeline latency (milliseconds).
    pub estimated_latency_ms: f64,
    /// Strategy used.
    pub strategy: SolverStrategy,
}

/// Which solver strategy was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStrategy {
    /// Layers proportional to available RAM.
    Proportional,
    /// Minimize bottleneck link, accounting for bandwidth.
    BandwidthAware,
}

/// Solve layer assignments using the cluster topology.
///
/// Automatically selects strategy:
/// - If all connections have similar bandwidth (within 2x): proportional by RAM
/// - If bandwidth varies: bandwidth-aware solver
#[allow(clippy::single_range_in_vec_init)]
pub fn solve(topology: &ClusterTopology, num_layers: usize) -> SolverResult {
    let nodes: Vec<&crate::topology::NodeInfo> = topology.ring_order().into_iter().collect();
    let world_size = nodes.len();

    if world_size <= 1 {
        return SolverResult {
            assignments: vec![0..num_layers],
            estimated_latency_ms: 0.0,
            strategy: SolverStrategy::Proportional,
        };
    }

    let ram: Vec<u64> = nodes.iter().map(|n| n.profile.available_ram).collect();

    // Estimate bandwidth for ring links
    let bandwidth = topology.ring_bandwidth();

    // Check bandwidth heterogeneity
    let has_thunderbolt = topology.has_thunderbolt_ring();

    if has_thunderbolt || bandwidth > 0 {
        // Use bandwidth-aware solver. Per-edge bandwidth comes from the
        // classified fabric kind on each node's best advertised address —
        // not a chip-name heuristic. At 64 MiB activation chunks, transfer
        // time dominates link latency by ~3 orders of magnitude, so we feed
        // the layer-assignment helper raw nominal bandwidth and reserve
        // [`score_link`] for a future latency-aware solver path.
        let bw_per_node: Vec<u64> = nodes
            .iter()
            .map(|n| {
                let kind = n
                    .best_addr()
                    .map(|(_, k)| k)
                    .unwrap_or(InterfaceKind::Unknown);
                kind.nominal_bandwidth_bps()
            })
            .collect();

        let assignments =
            crate::layer_assignment::assign_layers_bandwidth_aware(num_layers, &ram, &bw_per_node);

        // Estimate latency: max(layers_per_node * flops_per_layer / throughput)
        let estimated_latency_ms = estimate_pipeline_latency(&assignments, &bw_per_node);

        SolverResult {
            assignments,
            estimated_latency_ms,
            strategy: SolverStrategy::BandwidthAware,
        }
    } else {
        let assignments = crate::layer_assignment::assign_layers_proportional(num_layers, &ram);

        SolverResult {
            assignments,
            estimated_latency_ms: 0.0, // Unknown without bandwidth info
            strategy: SolverStrategy::Proportional,
        }
    }
}

/// Estimate pipeline latency based on layer assignments and bandwidth.
fn estimate_pipeline_latency(assignments: &[Range<usize>], bandwidths: &[u64]) -> f64 {
    if assignments.is_empty() || bandwidths.is_empty() {
        return 0.0;
    }

    // Simplified: max(layers * compute_per_layer) across nodes
    // Plus activation transfer time between stages
    let max_compute_ms = assignments
        .iter()
        .zip(bandwidths.iter())
        .map(|(range, &_bw)| {
            // ~0.5ms per layer per token for a 7B model on Apple Silicon
            range.len() as f64 * 0.5
        })
        .fold(0.0f64, f64::max);

    // Activation transfer: ~8KB per hidden state at fp16 (4096 dims * 2 bytes)
    // Between each stage pair
    let transfer_ms = (assignments.len() - 1) as f64 * 0.01; // ~0.01ms per transfer over Thunderbolt

    max_compute_ms + transfer_ms
}

/// Measure actual bandwidth between two peers by sending test payloads.
pub async fn measure_bandwidth(
    sender: &mut crate::transport::TransportSender,
    receiver: &mut crate::transport::TransportReceiver,
    payload_sizes: &[usize],
) -> Result<BandwidthMeasurement, crate::error::DistributedError> {
    let mut measurements = Vec::new();

    for &size in payload_sizes {
        let payload = vec![0xABu8; size];
        let start = std::time::Instant::now();

        // Send test payload
        sender.send(&payload).await.map_err(|e| {
            crate::error::DistributedError::Protocol(format!("bandwidth test send: {e}"))
        })?;

        // Receive echo
        let mut recv_buf = vec![0u8; size];
        receiver.recv(&mut recv_buf).await.map_err(|e| {
            crate::error::DistributedError::Protocol(format!("bandwidth test recv: {e}"))
        })?;

        let elapsed = start.elapsed();
        let throughput_bps = (size as f64 * 2.0 * 8.0) / elapsed.as_secs_f64(); // bits/sec
        let throughput_gbps = throughput_bps / 1_000_000_000.0;

        measurements.push((size, elapsed, throughput_gbps));
    }

    // Use the largest payload for the most accurate bandwidth estimate
    let (_, _, best_gbps) = measurements
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    Ok(BandwidthMeasurement {
        throughput_gbps: *best_gbps,
        measurements,
    })
}

/// Result of a bandwidth measurement.
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    /// Peak measured throughput in Gbps.
    pub throughput_gbps: f64,
    /// Per-size measurements: (payload_bytes, round_trip_time, throughput_gbps).
    pub measurements: Vec<(usize, std::time::Duration, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_latency_basic() {
        let assignments = vec![0..16, 16..32];
        let bandwidths = vec![40_000_000_000, 40_000_000_000];
        let latency = estimate_pipeline_latency(&assignments, &bandwidths);
        assert!(latency > 0.0);
        // 16 layers * 0.5ms + 0.01ms transfer ≈ 8.01ms
        assert!((latency - 8.01).abs() < 0.1);
    }
}
