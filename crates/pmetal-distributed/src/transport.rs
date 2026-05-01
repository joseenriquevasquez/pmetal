//! TCP transport layer for distributed training.
//!
//! Provides reliable, ordered delivery over TCP with optimizations
//! for gradient synchronization workloads.

use crate::config::DistributedConfig;
use crate::error::DistributedError;
use anyhow::Result;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Maximum backoff delay between connection retries.
const MAX_BACKOFF_MS: u64 = 5000;

/// Initial backoff delay between connection retries.
const INITIAL_BACKOFF_MS: u64 = 100;

/// Sender half of the transport.
pub struct TransportSender {
    inner: TransportSenderInner,
}

/// Receiver half of the transport.
pub struct TransportReceiver {
    inner: TransportReceiverInner,
}

enum TransportSenderInner {
    Tcp(OwnedWriteHalf),
    Memory(mpsc::Sender<Vec<u8>>),
}

enum TransportReceiverInner {
    Tcp(OwnedReadHalf),
    Memory(mpsc::Receiver<Vec<u8>>),
}

impl TransportSender {
    /// Send data with length prefix.
    pub async fn send(&mut self, data: &[u8]) -> Result<()> {
        match &mut self.inner {
            TransportSenderInner::Tcp(stream) => {
                let len = (data.len() as u32).to_le_bytes();
                stream.write_all(&len).await?;
                stream.write_all(data).await?;
            }
            TransportSenderInner::Memory(sender) => {
                sender.send(data.to_vec()).await.map_err(|e| {
                    DistributedError::Protocol(format!("in-memory send failed: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

impl TransportReceiver {
    /// Receive data into buffer (must match expected size).
    ///
    /// Enforces a maximum message size of 512 MiB to prevent resource
    /// exhaustion from malicious or corrupted length prefixes.
    ///
    /// Each `read_exact` call is wrapped in a 60-second timeout so that a
    /// peer crash or network partition does not cause indefinite blocking.
    pub async fn recv(&mut self, buffer: &mut [u8]) -> Result<()> {
        const MAX_MSG_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
        const READ_TIMEOUT: Duration = Duration::from_secs(60);

        match &mut self.inner {
            TransportReceiverInner::Tcp(stream) => {
                // Read the 4-byte length prefix with timeout.
                let mut len_buf = [0u8; 4];
                timeout(READ_TIMEOUT, stream.read_exact(&mut len_buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for message length prefix (60s) — peer may have crashed"
                                .to_string(),
                        )
                    })??;
                let len = u32::from_le_bytes(len_buf) as usize;

                if len > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        len, MAX_MSG_BYTES
                    ))
                    .into());
                }

                if len != buffer.len() {
                    return Err(DistributedError::Protocol(format!(
                        "Expected {} bytes, got {}",
                        buffer.len(),
                        len
                    ))
                    .into());
                }

                timeout(READ_TIMEOUT, stream.read_exact(buffer))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(format!(
                            "Timed out waiting for {} bytes of payload (60s) — peer may have crashed",
                            buffer.len()
                        ))
                    })??;
            }
            TransportReceiverInner::Memory(receiver) => {
                let data = timeout(READ_TIMEOUT, receiver.recv())
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for in-memory payload (60s)".to_string(),
                        )
                    })?
                    .ok_or_else(|| {
                        DistributedError::Protocol("in-memory channel closed".to_string())
                    })?;

                if data.len() > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        data.len(),
                        MAX_MSG_BYTES
                    ))
                    .into());
                }

                if data.len() != buffer.len() {
                    return Err(DistributedError::Protocol(format!(
                        "Expected {} bytes, got {}",
                        buffer.len(),
                        data.len()
                    ))
                    .into());
                }

                buffer.copy_from_slice(&data);
            }
        }

        Ok(())
    }

    /// Receive a dynamically-sized message, returning the data as a `Vec<u8>`.
    ///
    /// Reads the 4-byte length prefix, allocates a buffer, and reads the payload.
    /// This is useful when the caller does not know the message size in advance
    /// (e.g. activation messages with variable tensor shapes).
    pub async fn recv_vec(&mut self) -> Result<Vec<u8>> {
        const MAX_MSG_BYTES: usize = 512 * 1024 * 1024;
        const READ_TIMEOUT: Duration = Duration::from_secs(60);

        match &mut self.inner {
            TransportReceiverInner::Tcp(stream) => {
                let mut len_buf = [0u8; 4];
                timeout(READ_TIMEOUT, stream.read_exact(&mut len_buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for message length prefix (60s)".to_string(),
                        )
                    })??;
                let len = u32::from_le_bytes(len_buf) as usize;

                if len > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        len, MAX_MSG_BYTES
                    ))
                    .into());
                }

                let mut buf = vec![0u8; len];
                timeout(READ_TIMEOUT, stream.read_exact(&mut buf))
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(format!(
                            "Timed out waiting for {} bytes of payload (60s)",
                            len
                        ))
                    })??;

                Ok(buf)
            }
            TransportReceiverInner::Memory(receiver) => {
                let data = timeout(READ_TIMEOUT, receiver.recv())
                    .await
                    .map_err(|_| {
                        DistributedError::Protocol(
                            "Timed out waiting for in-memory payload (60s)".to_string(),
                        )
                    })?
                    .ok_or_else(|| {
                        DistributedError::Protocol("in-memory channel closed".to_string())
                    })?;

                if data.len() > MAX_MSG_BYTES {
                    return Err(DistributedError::Protocol(format!(
                        "Message size {} bytes exceeds maximum {} bytes",
                        data.len(),
                        MAX_MSG_BYTES
                    ))
                    .into());
                }

                Ok(data)
            }
        }
    }
}

impl TransportSender {
    fn from_tcp(stream: OwnedWriteHalf) -> Self {
        Self {
            inner: TransportSenderInner::Tcp(stream),
        }
    }

    fn from_memory(sender: mpsc::Sender<Vec<u8>>) -> Self {
        Self {
            inner: TransportSenderInner::Memory(sender),
        }
    }

    /// Crate-internal: wrap an already-split TCP write half. Used by the
    /// pipeline harness which builds extra TCP rings outside `TcpTransport`.
    pub(crate) fn from_owned_write(stream: OwnedWriteHalf) -> Self {
        Self::from_tcp(stream)
    }
}

impl TransportReceiver {
    fn from_tcp(stream: OwnedReadHalf) -> Self {
        Self {
            inner: TransportReceiverInner::Tcp(stream),
        }
    }

    fn from_memory(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            inner: TransportReceiverInner::Memory(receiver),
        }
    }

    pub(crate) fn from_owned_read(stream: OwnedReadHalf) -> Self {
        Self::from_tcp(stream)
    }
}

/// Create a unidirectional in-memory transport channel.
///
/// This is used for local same-process pipeline stages, such as UltraFusion
/// multi-die execution planning, where TCP framing would add unnecessary cost.
pub fn in_memory_channel(capacity: usize) -> (TransportSender, TransportReceiver) {
    let (tx, rx) = mpsc::channel(capacity.max(1));
    (
        TransportSender::from_memory(tx),
        TransportReceiver::from_memory(rx),
    )
}

/// TCP transport for ring communication.
pub struct TcpTransport;

impl TcpTransport {
    /// Connect to peers in a ring topology, with fabric-aware fallback.
    ///
    /// Each rank:
    ///   1. Binds the listener on `0.0.0.0:<my_port>`. Listening on the
    ///      unspecified address means peers can dial in over *any* fabric
    ///      we have an interface on — Thunderbolt or Ethernet — without
    ///      forcing us to pick one at bind time.
    ///   2. Connects to the next peer by trying each address in
    ///      `addrs_for(next_rank)` in priority order (best fabric first).
    ///      Each address gets `max_retries / addrs.len()` retry budget;
    ///      we exhaust one before falling back to the next.
    pub async fn connect(
        config: &DistributedConfig,
    ) -> Result<(TransportSender, TransportReceiver)> {
        let world_size = config.nodes.len();
        let rank = config.rank;
        let my_addr = config.nodes[rank];
        let connection_timeout = Duration::from_millis(config.connection_timeout_ms);

        // Bind on 0.0.0.0:<my_port> so we accept on every local interface.
        // Falling back to a specific address would mean a TB cable disconnect
        // breaks the listener even though Ethernet is still up.
        let bind_addr: std::net::SocketAddr = SocketAddr::new(
            "0.0.0.0".parse().expect("0.0.0.0 always parses"),
            my_addr.port(),
        );
        info!(
            "Node {} listening on {} (advertised as {})",
            rank, bind_addr, my_addr
        );
        let listener = TcpListener::bind(bind_addr).await?;

        let next_rank = (rank + 1) % world_size;
        let next_endpoints = config.addrs_for(next_rank);
        if next_endpoints.is_empty() {
            return Err(DistributedError::Config(format!(
                "no addresses configured for next peer rank {}",
                next_rank
            ))
            .into());
        }

        // Walk fabrics in priority order, retrying each before falling through.
        let connect_fut =
            Self::connect_with_fallback(next_rank, &next_endpoints, config.max_retries);

        // Accept connection from previous peer.
        let accept_fut = async {
            match timeout(connection_timeout, listener.accept()).await {
                Ok(Ok((stream, addr))) => {
                    if let Err(e) = stream.set_nodelay(true) {
                        warn!("Failed to set TCP_NODELAY on incoming: {}", e);
                    }
                    info!("Accepted connection from {}", addr);
                    Ok(stream)
                }
                Ok(Err(e)) => Err(DistributedError::Io(e)),
                Err(_) => Err(DistributedError::Protocol(format!(
                    "Timeout waiting for incoming connection ({}ms)",
                    config.connection_timeout_ms
                ))),
            }
        };

        let (next_peer, prev_peer) = tokio::try_join!(
            async {
                connect_fut
                    .await
                    .map_err(|e: DistributedError| anyhow::anyhow!(e))
            },
            async {
                accept_fut
                    .await
                    .map_err(|e: DistributedError| anyhow::anyhow!(e))
            }
        )?;

        let (_, write_next) = next_peer.into_split();
        let (read_prev, _) = prev_peer.into_split();

        Ok((
            TransportSender::from_tcp(write_next),
            TransportReceiver::from_tcp(read_prev),
        ))
    }

    /// Round-robin connect: try each address once per round with shared
    /// exponential backoff between rounds. A transient outage on the
    /// best-fabric (e.g. a Thunderbolt cable jiggle) falls back to the
    /// next fabric within a few seconds rather than after the entire
    /// per-fabric budget is exhausted.
    ///
    /// `max_retries` caps the total number of *connect attempts across all
    /// fabrics*, not per fabric. With 2 endpoints and `max_retries = 50`,
    /// each address is tried up to ~25 times.
    async fn connect_with_fallback(
        next_rank: usize,
        endpoints: &[SocketAddr],
        max_retries: u32,
    ) -> std::result::Result<TcpStream, DistributedError> {
        if endpoints.is_empty() {
            return Err(DistributedError::Config(format!(
                "no endpoints configured for next peer rank {next_rank}"
            )));
        }
        let mut last_err: Option<std::io::Error> = None;
        let mut attempts: u32 = 0;
        let mut round: u32 = 0;

        while attempts < max_retries {
            for (idx, addr) in endpoints.iter().enumerate() {
                if attempts >= max_retries {
                    break;
                }
                attempts += 1;

                match TcpStream::connect(addr).await {
                    Ok(stream) => {
                        if let Err(e) = stream.set_nodelay(true) {
                            warn!("Failed to set TCP_NODELAY: {}", e);
                        }
                        info!(
                            "Connected to next peer {} at {} (fabric {}/{}, round {})",
                            next_rank,
                            addr,
                            idx + 1,
                            endpoints.len(),
                            round + 1,
                        );
                        return Ok(stream);
                    }
                    Err(e) => {
                        if round == 0 {
                            debug!("Waiting for peer {} at {} ({})", next_rank, addr, e);
                        }
                        last_err = Some(e);
                    }
                }
            }

            // Finished one round across every fabric — sleep before retrying.
            round += 1;
            let backoff = (INITIAL_BACKOFF_MS * 2u64.pow(round.min(6))).min(MAX_BACKOFF_MS);
            tokio::time::sleep(Duration::from_millis(backoff)).await;
        }

        if let Some(ioe) = last_err {
            debug!("Final connect error to rank {}: {}", next_rank, ioe);
        }
        Err(DistributedError::MaxRetriesExceeded {
            addr: endpoints[0],
            max_retries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_transport_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"hello").await.expect("send");

        let mut buf = [0u8; 5];
        receiver.recv(&mut buf).await.expect("recv");
        assert_eq!(&buf, b"hello");
    }

    #[tokio::test]
    async fn in_memory_transport_recv_vec_roundtrip() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"payload").await.expect("send");

        let data = receiver.recv_vec().await.expect("recv_vec");
        assert_eq!(data, b"payload");
    }

    #[tokio::test]
    async fn in_memory_transport_len_mismatch_errors() {
        let (mut sender, mut receiver) = in_memory_channel(4);
        sender.send(b"toolong").await.expect("send");

        let mut buf = [0u8; 3];
        let err = receiver.recv(&mut buf).await.expect_err("length mismatch");
        assert!(err.to_string().contains("Expected 3 bytes, got 7"));
    }

    #[tokio::test]
    async fn connect_falls_back_when_first_fabric_unreachable() {
        // Bind a real listener on 127.0.0.1; this is the *fallback* address.
        // The "primary" address points at a black-hole port we know is closed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let working_addr = listener.local_addr().unwrap();

        let unreachable: SocketAddr = "127.0.0.1:1".parse().unwrap(); // port 1 always refuses

        let endpoints = vec![unreachable, working_addr];

        // Spawn the connect side in a task, accept in the test body.
        let connect_task = tokio::spawn(async move {
            // Tight retry budget so the unreachable fabric exhausts quickly.
            TcpTransport::connect_with_fallback(0, &endpoints, /*max_retries=*/ 4).await
        });

        let (_inbound, _from) = listener.accept().await.unwrap();

        let conn = connect_task.await.expect("task").expect("connect");
        let peer = conn.peer_addr().expect("peer_addr");
        assert_eq!(
            peer.port(),
            working_addr.port(),
            "should have connected to the working fallback fabric, got {}",
            peer
        );
    }
}
