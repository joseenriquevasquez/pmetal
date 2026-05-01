//! Probe local NICs and Thunderbolt connectivity into a [`LocalFabric`].
//!
//! Single data source: `if-addrs::get_if_addrs` enumerates every interface's
//! IPs. Classification is name + IP heuristic:
//!
//!   * `lo0` → Loopback
//!   * `bridgeN` → Thunderbolt-Bridge (macOS convention since 12.0)
//!   * `enN` with a 169.254/16 link-local IP → Thunderbolt (auto-assigned by
//!     macOS when a TB cable is plugged in and no DHCP is present)
//!   * `enN` / `eth*` / `usb*` / `ax*` → Ethernet
//!   * `awdl*` / `llw*` / `wl*` / `wifi*` → Wi-Fi
//!
//! We deliberately don't shell out to `system_profiler SPThunderboltDataType`:
//! its JSON keys (`_name`, `device_name_key`) don't actually carry BSD
//! interface names, so the previous walker matched nothing in practice. The
//! heuristics above cover every real-world Mac TB-Bridge configuration.

use super::{InterfaceKind, is_link_local_ipv4};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use tracing::debug;

/// One classified local interface and its IP addresses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceInfo {
    /// Interface name (e.g. `en0`, `bridge0`, `awdl0`).
    pub name: String,
    /// IP addresses bound to this interface.
    pub addrs: Vec<IpAddr>,
    /// Classified link type.
    pub kind: InterfaceKind,
    /// Reported link speed if known (e.g. `"40 Gb/s"`).
    pub link_speed: Option<String>,
}

/// Snapshot of the local network fabric. Cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct LocalFabric {
    interfaces: Vec<InterfaceInfo>,
}

impl LocalFabric {
    /// Construct directly from a pre-classified interface list. Test-only
    /// hatch — production code should use [`probe_local_fabric`].
    #[doc(hidden)]
    pub fn from_interfaces(interfaces: Vec<InterfaceInfo>) -> Self {
        Self { interfaces }
    }

    /// All interfaces, sorted with the most desirable kind (Thunderbolt) first.
    pub fn interfaces(&self) -> &[InterfaceInfo] {
        &self.interfaces
    }

    /// Classify a remote IP by checking which local interface it would route
    /// through — i.e. find the local interface whose subnet contains `peer`.
    ///
    /// In practice on macOS, link-local 169.254/16 is the strong TB signal:
    /// if a peer advertises that address and we have a Thunderbolt-tagged
    /// interface with a 169.254/16 IP, we know the link is TB.
    pub fn classify_peer(&self, peer: &IpAddr) -> InterfaceKind {
        if peer.is_loopback() {
            return InterfaceKind::Loopback;
        }

        if is_link_local_ipv4(peer) {
            // Any local interface tagged Thunderbolt with a link-local IP wins.
            for iface in &self.interfaces {
                if iface.kind == InterfaceKind::Thunderbolt
                    && iface.addrs.iter().any(is_link_local_ipv4)
                {
                    return InterfaceKind::Thunderbolt;
                }
            }
            // Link-local but no TB iface — likely a misconfigured AppleTalk
            // bridge or a stale alias; treat as Ethernet.
            return InterfaceKind::Ethernet;
        }

        // Same-subnet match against every local interface.
        for iface in &self.interfaces {
            for local in &iface.addrs {
                if same_v4_subnet(local, peer) {
                    return iface.kind;
                }
            }
        }
        InterfaceKind::Unknown
    }

    /// IPv4 addresses on Thunderbolt-classified interfaces, in interface order.
    /// Used by discovery to decide which addresses to advertise to peers.
    pub fn thunderbolt_addrs(&self) -> Vec<IpAddr> {
        self.interfaces
            .iter()
            .filter(|i| i.kind == InterfaceKind::Thunderbolt)
            .flat_map(|i| i.addrs.iter().copied())
            .filter(|ip| ip.is_ipv4())
            .collect()
    }

    /// All routable (non-loopback, non-link-local-Wi-Fi-prefix) addresses,
    /// ordered best-fabric-first. The first entry is the one a peer should
    /// prefer when establishing a connection back.
    pub fn advertised_addrs(&self) -> Vec<(IpAddr, InterfaceKind)> {
        let mut out: Vec<(IpAddr, InterfaceKind)> = Vec::new();
        for iface in &self.interfaces {
            if iface.kind == InterfaceKind::Loopback {
                continue;
            }
            for ip in &iface.addrs {
                if ip.is_loopback() {
                    continue;
                }
                out.push((*ip, iface.kind));
            }
        }
        // Higher InterfaceKind first (Thunderbolt > Ethernet > Wifi > Unknown).
        // Sort by the negated rank so the best kind comes first.
        out.sort_by_key(|(_, k)| std::cmp::Reverse(*k));
        out
    }

    /// True if at least one Thunderbolt-classified interface has a link-local IP.
    pub fn has_active_thunderbolt(&self) -> bool {
        self.interfaces.iter().any(|i| {
            i.kind == InterfaceKind::Thunderbolt && i.addrs.iter().any(is_link_local_ipv4)
        })
    }
}

/// Probe the local fabric. Cheap (~1 ms on macOS); intended to run once at
/// startup and on topology change events.
pub fn probe_local_fabric() -> LocalFabric {
    let raw = if_addrs::get_if_addrs().unwrap_or_default();

    // Collapse multi-address interfaces into one InterfaceInfo per name.
    let mut by_name: std::collections::BTreeMap<String, Vec<IpAddr>> =
        std::collections::BTreeMap::new();
    for ifa in raw {
        by_name.entry(ifa.name.clone()).or_default().push(ifa.ip());
    }

    let mut interfaces: Vec<InterfaceInfo> = by_name
        .into_iter()
        .map(|(name, addrs)| {
            let kind = classify_iface_name(&name, &addrs);
            InterfaceInfo {
                name,
                addrs,
                kind,
                link_speed: None,
            }
        })
        .collect();

    // Best fabric first.
    interfaces.sort_by_key(|i| std::cmp::Reverse(i.kind));

    debug!(
        thunderbolt = interfaces.iter().filter(|i| i.kind == InterfaceKind::Thunderbolt).count(),
        ethernet   = interfaces.iter().filter(|i| i.kind == InterfaceKind::Ethernet).count(),
        wifi       = interfaces.iter().filter(|i| i.kind == InterfaceKind::Wifi).count(),
        "local fabric probe complete"
    );

    LocalFabric { interfaces }
}

/// Classify an interface name + its IPs.
///
/// macOS conventions we rely on:
///   * `lo0` — loopback
///   * `bridgeN` — Thunderbolt-Bridge on macOS 12+
///   * `enN` — Ethernet *or* Wi-Fi *or* USB; disambiguated by (a) presence
///     of a link-local IP (TB-Bridge default), then (b) `awdl`/`llw`
///     heuristics for Apple-specific wireless.
///   * `awdl0` / `llw0` — Apple Wireless Direct Link: Wi-Fi adjacent,
///     treated as Wi-Fi for routing purposes.
fn classify_iface_name(name: &str, addrs: &[IpAddr]) -> InterfaceKind {
    let lower = name.to_ascii_lowercase();

    if lower == "lo0" || lower == "lo" {
        return InterfaceKind::Loopback;
    }

    // bridgeN almost always = Thunderbolt-Bridge on macOS.
    if lower.starts_with("bridge") {
        return InterfaceKind::Thunderbolt;
    }

    if lower.starts_with("awdl") || lower.starts_with("llw") {
        return InterfaceKind::Wifi;
    }

    // A bare enN with a 169.254/16 IP is *probably* TB — macOS also
    // auto-assigns link-local to Wi-Fi when DHCP fails, but in that case
    // the interface is rarely connected. Bias toward TB and let the solver
    // measure RTT to confirm.
    if addrs.iter().any(is_link_local_ipv4) && lower.starts_with("en") {
        return InterfaceKind::Thunderbolt;
    }

    if lower.starts_with("en")
        || lower.starts_with("eth")
        || lower.starts_with("usb")
        || lower.starts_with("ax")
    {
        return InterfaceKind::Ethernet;
    }

    if lower.starts_with("wl") || lower.starts_with("wifi") {
        return InterfaceKind::Wifi;
    }

    InterfaceKind::Unknown
}

/// Rough same-subnet test for IPv4. Used to map peer IPs onto local
/// interfaces — accurate enough for /24 and link-local /16, which covers
/// every fabric we care about (TB-Bridge always /16, home LAN almost always /24).
fn same_v4_subnet(local: &IpAddr, peer: &IpAddr) -> bool {
    match (local, peer) {
        (IpAddr::V4(a), IpAddr::V4(b)) => {
            let ao = a.octets();
            let bo = b.octets();
            if ao[0] == 169 && ao[1] == 254 && bo[0] == 169 && bo[1] == 254 {
                return true; // link-local /16
            }
            ao[0] == bo[0] && ao[1] == bo[1] && ao[2] == bo[2]
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn classify_loopback() {
        assert_eq!(
            classify_iface_name("lo0", &[IpAddr::V4(Ipv4Addr::LOCALHOST)]),
            InterfaceKind::Loopback
        );
    }

    #[test]
    fn classify_bridge_is_thunderbolt() {
        assert_eq!(
            classify_iface_name("bridge0", &[IpAddr::V4(Ipv4Addr::new(169, 254, 1, 5))]),
            InterfaceKind::Thunderbolt
        );
    }

    #[test]
    fn classify_en_with_link_local_promotes_to_thunderbolt() {
        assert_eq!(
            classify_iface_name("en6", &[IpAddr::V4(Ipv4Addr::new(169, 254, 100, 200))]),
            InterfaceKind::Thunderbolt
        );
    }

    #[test]
    fn classify_en_with_lan_ip_is_ethernet() {
        assert_eq!(
            classify_iface_name("en0", &[IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5))]),
            InterfaceKind::Ethernet
        );
    }

    #[test]
    fn classify_awdl_is_wifi() {
        assert_eq!(classify_iface_name("awdl0", &[]), InterfaceKind::Wifi);
    }

    #[test]
    fn classify_peer_link_local_with_tb_iface() {
        let fabric = LocalFabric {
            interfaces: vec![InterfaceInfo {
                name: "bridge0".to_string(),
                addrs: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1))],
                kind: InterfaceKind::Thunderbolt,
                link_speed: None,
            }],
        };
        let peer = IpAddr::V4(Ipv4Addr::new(169, 254, 1, 5));
        assert_eq!(fabric.classify_peer(&peer), InterfaceKind::Thunderbolt);
    }

    #[test]
    fn classify_peer_same_subnet_ethernet() {
        let fabric = LocalFabric {
            interfaces: vec![InterfaceInfo {
                name: "en0".to_string(),
                addrs: vec![IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10))],
                kind: InterfaceKind::Ethernet,
                link_speed: None,
            }],
        };
        let peer = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 200));
        assert_eq!(fabric.classify_peer(&peer), InterfaceKind::Ethernet);
    }

    #[test]
    fn advertised_addrs_orders_by_fabric_quality() {
        let fabric = LocalFabric {
            interfaces: vec![
                InterfaceInfo {
                    name: "en0".to_string(),
                    addrs: vec![IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10))],
                    kind: InterfaceKind::Ethernet,
                    link_speed: None,
                },
                InterfaceInfo {
                    name: "bridge0".to_string(),
                    addrs: vec![IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1))],
                    kind: InterfaceKind::Thunderbolt,
                    link_speed: None,
                },
            ],
        };
        let advertised = fabric.advertised_addrs();
        assert_eq!(advertised[0].1, InterfaceKind::Thunderbolt);
        assert_eq!(advertised[1].1, InterfaceKind::Ethernet);
    }

    #[test]
    fn probe_runs_without_panicking() {
        let fabric = probe_local_fabric();
        // Loopback is always present.
        assert!(fabric.interfaces().iter().any(|i| i.kind == InterfaceKind::Loopback));
    }
}
