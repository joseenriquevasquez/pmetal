//! Fabric profiling — what kind of network link connects two peers.
//!
//! On Apple Silicon home clusters the practical fabric hierarchy is:
//!
//! ```text
//! Thunderbolt-bridge   40–80 Gbps   ~10 µs    direct cable, link-local IP
//! Ethernet              1–10 Gbps   ~500 µs   switch / router
//! Wi-Fi                0.1–2 Gbps   ~5 ms    household AP
//! ```
//!
//! When two Macs are connected with a Thunderbolt cable, macOS exposes a
//! "Thunderbolt Bridge" `bridgeN` interface (or directly `enN` on newer
//! systems) using a 169.254/16 link-local subnet. We detect that locally,
//! advertise the resulting socket addresses with peer-discovery, and bias
//! ring formation to use those addresses whenever both endpoints can reach
//! each other on a Thunderbolt subnet.
//!
//! The [`LocalFabric`] snapshot is the single source of truth that downstream
//! modules (discovery, transport, topology, solver) read from.

pub mod probe;
mod score;

pub use probe::{InterfaceInfo, LocalFabric, probe_local_fabric};
pub use score::{LinkScore, nominal_score, score_link};

use serde::{Deserialize, Serialize};

/// Classification of a network interface.
///
/// Order is meaningful: variants are ranked low → high desirability, so
/// `InterfaceKind::Thunderbolt > InterfaceKind::Ethernet > ...`. Use
/// [`InterfaceKind::cmp`] / [`Ord`] to pick the better fabric for an edge.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum InterfaceKind {
    /// Could not classify (missing data, unrecognised driver name).
    #[default]
    Unknown,
    /// Loopback (`lo0`). Used for in-process / same-host testing.
    Loopback,
    /// 802.11 wireless (`enN` driven by Wi-Fi, `awdlN`, `llwN`).
    Wifi,
    /// Wired Ethernet, USB-Ethernet adapters, or a generic non-TB `enN`.
    Ethernet,
    /// Thunderbolt-bridge / direct TB IP-over-Thunderbolt link (`bridgeN`,
    /// or `enN` whose physical port is a Thunderbolt port).
    Thunderbolt,
}

impl InterfaceKind {
    /// Approximate raw link bandwidth in bytes/sec. Used by the solver as a
    /// starting weight before measured probes refine it.
    pub fn nominal_bandwidth_bps(self) -> u64 {
        match self {
            InterfaceKind::Loopback => 50_000_000_000,    // 50 GB/s memcpy-ish
            InterfaceKind::Thunderbolt => 5_000_000_000,  // 40 Gbps / 8
            InterfaceKind::Ethernet => 125_000_000,       // 1 Gbps / 8 (conservative)
            InterfaceKind::Wifi => 100_000_000,           // 800 Mbps / 8
            InterfaceKind::Unknown => 50_000_000,
        }
    }

    /// Approximate one-way link latency.
    pub fn nominal_latency_us(self) -> u64 {
        match self {
            InterfaceKind::Loopback => 1,
            InterfaceKind::Thunderbolt => 10,
            InterfaceKind::Ethernet => 500,
            InterfaceKind::Wifi => 5_000,
            InterfaceKind::Unknown => 5_000,
        }
    }

    /// Human-friendly tag for logs and `pmetal cluster status`.
    pub fn tag(self) -> &'static str {
        match self {
            InterfaceKind::Loopback => "loopback",
            InterfaceKind::Thunderbolt => "thunderbolt",
            InterfaceKind::Ethernet => "ethernet",
            InterfaceKind::Wifi => "wifi",
            InterfaceKind::Unknown => "unknown",
        }
    }
}

/// True if `ip` is on the IPv4 link-local subnet (169.254.0.0/16) macOS
/// auto-assigns to direct Thunderbolt and crossover-cable links.
pub fn is_link_local_ipv4(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == 169 && o[1] == 254
        }
        std::net::IpAddr::V6(_) => false,
    }
}
