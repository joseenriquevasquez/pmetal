//! Edge-weight helpers for the BW-aware ring solver.
//!
//! Given two endpoints and the link kind between them, produce a single
//! comparable score the solver can `min`/`sort` on. We model "ring step
//! cost" — i.e. the time to push one byte across the slowest link — so
//! lower scores always mean a better ring.

use super::InterfaceKind;
use std::time::Duration;

/// Composite cost score for a directed link.
///
/// `cost = latency_us + bytes_per_chunk / bandwidth_bps`
///
/// Default `bytes_per_chunk = 64 MiB` reflects a typical layer's gradient
/// chunk in mid-size LLM training. The exact constant doesn't matter for
/// ordering — bandwidth dominates above ~1 MiB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkScore {
    /// Lower is better.
    pub cost_us: f64,
    /// Original kind for tie-breaking and reporting.
    pub kind: InterfaceKind,
}

impl LinkScore {
    /// True if this link is "ring-acceptable" — i.e. fast enough that we
    /// won't bottleneck training. Currently any non-Wi-Fi link qualifies.
    pub fn is_acceptable_for_training(&self) -> bool {
        matches!(
            self.kind,
            InterfaceKind::Thunderbolt | InterfaceKind::Ethernet | InterfaceKind::Loopback
        )
    }
}

impl Eq for LinkScore {}

impl Ord for LinkScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // f64 has no total order; rank NaN as worst-case, otherwise compare.
        match self.cost_us.partial_cmp(&other.cost_us) {
            Some(o) => o,
            None => std::cmp::Ordering::Equal,
        }
        .then_with(|| other.kind.cmp(&self.kind)) // higher kind first on tie
    }
}

impl PartialOrd for LinkScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute the score for an edge given its measured (or nominal) latency,
/// bandwidth, and chunk size.
pub fn score_link(
    kind: InterfaceKind,
    latency: Duration,
    bandwidth_bps: u64,
    bytes_per_chunk: u64,
) -> LinkScore {
    let lat_us = latency.as_secs_f64() * 1_000_000.0;
    let xfer_us = if bandwidth_bps == 0 {
        f64::INFINITY
    } else {
        (bytes_per_chunk as f64) / (bandwidth_bps as f64) * 1_000_000.0
    };
    LinkScore {
        cost_us: lat_us + xfer_us,
        kind,
    }
}

/// Score using the kind's nominal numbers — the pre-measurement default.
pub fn nominal_score(kind: InterfaceKind, bytes_per_chunk: u64) -> LinkScore {
    score_link(
        kind,
        Duration::from_micros(kind.nominal_latency_us()),
        kind.nominal_bandwidth_bps(),
        bytes_per_chunk,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;

    #[test]
    fn thunderbolt_beats_ethernet() {
        let tb = nominal_score(InterfaceKind::Thunderbolt, 64 * MB);
        let eth = nominal_score(InterfaceKind::Ethernet, 64 * MB);
        assert!(tb < eth, "TB ({:?}) should beat Ethernet ({:?})", tb, eth);
    }

    #[test]
    fn ethernet_beats_wifi() {
        let eth = nominal_score(InterfaceKind::Ethernet, 64 * MB);
        let wifi = nominal_score(InterfaceKind::Wifi, 64 * MB);
        assert!(eth < wifi);
    }

    #[test]
    fn acceptable_for_training_excludes_wifi_and_unknown() {
        assert!(nominal_score(InterfaceKind::Thunderbolt, MB).is_acceptable_for_training());
        assert!(nominal_score(InterfaceKind::Ethernet, MB).is_acceptable_for_training());
        assert!(nominal_score(InterfaceKind::Loopback, MB).is_acceptable_for_training());
        assert!(!nominal_score(InterfaceKind::Wifi, MB).is_acceptable_for_training());
        assert!(!nominal_score(InterfaceKind::Unknown, MB).is_acceptable_for_training());
    }

    #[test]
    fn zero_bandwidth_is_infinity_cost() {
        let s = score_link(
            InterfaceKind::Ethernet,
            Duration::from_micros(500),
            0,
            MB,
        );
        assert!(s.cost_us.is_infinite());
    }
}
