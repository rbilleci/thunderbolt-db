use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    path::Path,
    sync::Arc,
    time::Duration,
};

use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term};
use rustls::{
    pki_types::ServerName, ClientConfig as TlsClientConfig, ClientConnection, RootCertStore,
    ServerConfig as TlsServerConfig, ServerConnection, StreamOwned,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryState {
    pub term: Term,
    pub snapshot: SnapshotMeta,
    pub committed_entries: Vec<LogEntry>,
    pub applied_index: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationProgress {
    pub role: Role,
    pub term: Term,
    pub commit_index: Index,
    pub applied_index: Index,
    pub next_index: Index,
    pub snapshot: SnapshotMeta,
    pub committed_but_unapplied_count: usize,
    pub has_committed_entries_pending_apply: bool,
    pub uncommitted_entry_count: usize,
    pub has_uncommitted_entries: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryProgressGap {
    pub commit_index_gap: usize,
    pub applied_index_gap: usize,
    pub next_index_gap: usize,
    pub uncommitted_entry_gap: usize,
}

impl RecoveryProgressGap {
    pub fn has_gap(&self) -> bool {
        self.commit_index_gap > 0
            || self.applied_index_gap > 0
            || self.next_index_gap > 0
            || self.uncommitted_entry_gap > 0
    }

    pub fn has_speculative_tail(&self) -> bool {
        self.next_index_gap > 0 || self.uncommitted_entry_gap > 0
    }

    pub fn is_restart_equivalent(&self) -> bool {
        !self.has_gap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationStatusSnapshot {
    pub live: ReplicationProgress,
    pub durable: ReplicationProgress,
    pub recovery_gap: RecoveryProgressGap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalClusterSmokeReport {
    pub promoted_leader_term: Term,
    pub promoted_leader_commit_index: Index,
    pub follower_commit_index: Index,
    pub follower_applied_index: Index,
    pub follower_caught_up: bool,
    pub follower_read_after_apply: Vec<String>,
    pub old_leader_rejected_after_failover: bool,
    pub promoted_node_role: Role,
}

impl OperationalClusterSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        self.follower_caught_up
            && self.old_leader_rejected_after_failover
            && self.promoted_node_role == Role::Leader
            && self.follower_commit_index == self.promoted_leader_commit_index
            && self.follower_applied_index == self.follower_commit_index
    }

    pub fn to_operator_lines(&self) -> Vec<String> {
        vec![
            format!(
                "operational_replication_smoke={}",
                if self.readiness_passed() {
                    "passed"
                } else {
                    "failed"
                }
            ),
            format!(
                "promoted_leader_term={} promoted_leader_commit={} follower_commit={} follower_applied={} follower_caught_up={}",
                self.promoted_leader_term,
                self.promoted_leader_commit_index,
                self.follower_commit_index,
                self.follower_applied_index,
                self.follower_caught_up
            ),
            format!(
                "follower_read_after_apply={}",
                self.follower_read_after_apply.join(" | ")
            ),
            format!(
                "failover_admission_gate=old_leader_{} promoted_node_role={:?}",
                if self.old_leader_rejected_after_failover {
                    "not_leader"
                } else {
                    "accepted_write"
                },
                self.promoted_node_role
            ),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalTransportSmokeReport {
    pub transport_scope: &'static str,
    pub append_batches_sent: usize,
    pub heartbeat_batches_sent: usize,
    pub follower_acks_recorded: usize,
}

impl OperationalTransportSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.transport_scope.is_empty()
            && self.append_batches_sent > 0
            && self.heartbeat_batches_sent > 0
            && self.follower_acks_recorded > 0
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_transport={} append_batches_sent={} heartbeat_batches_sent={} follower_acks_recorded={}",
            self.transport_scope,
            self.append_batches_sent,
            self.heartbeat_batches_sent,
            self.follower_acks_recorded
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalElectionSmokeReport {
    pub election_scope: &'static str,
    pub candidate_id: u64,
    pub elected_term: Term,
    pub votes_granted: usize,
    pub quorum: usize,
    pub elected: bool,
}

impl OperationalElectionSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.election_scope.is_empty()
            && self.candidate_id > 0
            && self.elected
            && self.votes_granted >= self.quorum
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_election={} candidate_id={} elected_term={} votes_granted={} quorum={} elected={}",
            self.election_scope,
            self.candidate_id,
            self.elected_term,
            self.votes_granted,
            self.quorum,
            self.elected
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalPackageSmokeReport {
    pub package_scope: &'static str,
    pub entrypoint: &'static str,
    pub smoke_script: &'static str,
    pub packaged_script: &'static str,
    pub reproducible: bool,
}

impl OperationalPackageSmokeReport {
    pub fn readiness_passed(&self) -> bool {
        !self.package_scope.is_empty()
            && !self.entrypoint.is_empty()
            && !self.smoke_script.is_empty()
            && !self.packaged_script.is_empty()
            && self.reproducible
    }

    pub fn to_operator_line(&self) -> String {
        format!(
            "deployment_package={} entrypoint={} smoke_script={} packaged_script={} reproducible={}",
            self.package_scope,
            self.entrypoint,
            self.smoke_script,
            self.packaged_script,
            self.reproducible
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalDeploymentPreflightReport {
    pub smoke: OperationalClusterSmokeReport,
    pub transport: OperationalTransportSmokeReport,
    pub election: OperationalElectionSmokeReport,
    pub package: OperationalPackageSmokeReport,
    pub network_transport_implemented: bool,
    pub automatic_election_implemented: bool,
    pub packaged_deployment_implemented: bool,
}

impl OperationalDeploymentPreflightReport {
    pub fn readiness_passed(&self) -> bool {
        self.smoke.readiness_passed()
            && self.transport.readiness_passed()
            && self.election.readiness_passed()
            && self.package.readiness_passed()
    }

    pub fn to_operator_lines(&self) -> Vec<String> {
        let mut lines = self.smoke.to_operator_lines();
        lines.push(self.transport.to_operator_line());
        lines.push(self.election.to_operator_line());
        lines.push(self.package.to_operator_line());
        lines.extend([
            format!(
                "operational_deployment_preflight={}",
                if self.readiness_passed() {
                    "passed"
                } else {
                    "failed"
                }
            ),
            "deployment_scope=packaged_local_three_node_raft_smoke".to_string(),
            format!(
                "deployment_gap_network_transport={}",
                if self.network_transport_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
            format!(
                "deployment_gap_automatic_election={}",
                if self.automatic_election_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
            format!(
                "deployment_gap_packaged_deployment={}",
                if self.packaged_deployment_implemented {
                    "implemented"
                } else {
                    "missing"
                }
            ),
        ]);
        lines
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteRequest {
    pub candidate_term: Term,
    pub candidate_id: u64,
    pub last_log_index: Index,
    pub last_log_term: Term,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteResponse {
    pub granted: bool,
    pub voter_term: Term,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesRequest {
    pub leader_term: Term,
    pub prev_log_index: Index,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesResponse {
    pub accepted: bool,
    pub follower_term: Term,
    pub follower_commit_index: Index,
    pub follower_applied_index: Index,
    pub error: Option<String>,
}

impl AppendEntriesRequest {
    pub fn apply_to(&self, follower: &mut RaftReplicator) -> AppendEntriesResponse {
        let result = follower.append_entries_from_leader(
            self.leader_term,
            self.prev_log_index,
            self.prev_log_term,
            self.entries.clone(),
            self.leader_commit,
        );
        AppendEntriesResponse {
            accepted: result.is_ok(),
            follower_term: follower.current_term(),
            follower_commit_index: follower.commit_index(),
            follower_applied_index: follower.applied_index(),
            error: result.err().map(|err| err.to_string()),
        }
    }

    pub fn encode_frame(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_u64(&mut out, self.leader_term);
        write_u64(&mut out, self.prev_log_index);
        write_u64(&mut out, self.prev_log_term);
        write_u64(&mut out, self.leader_commit);
        write_u64(&mut out, self.entries.len() as u64);
        for entry in &self.entries {
            write_u64(&mut out, entry.term);
            write_u64(&mut out, entry.index);
            write_bytes(&mut out, &entry.payload);
        }
        out
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = FrameCursor::new(frame);
        let leader_term = cursor.read_u64("leader_term")?;
        let prev_log_index = cursor.read_u64("prev_log_index")?;
        let prev_log_term = cursor.read_u64("prev_log_term")?;
        let leader_commit = cursor.read_u64("leader_commit")?;
        let entry_count = cursor.read_len("entry_count")?;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            entries.push(LogEntry {
                term: cursor.read_u64("entry.term")?,
                index: cursor.read_u64("entry.index")?,
                payload: cursor.read_bytes("entry.payload")?.to_vec().into(),
            });
        }
        cursor.finish()?;
        Ok(Self {
            leader_term,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        })
    }
}

impl AppendEntriesResponse {
    pub fn encode_frame(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(u8::from(self.accepted));
        write_u64(&mut out, self.follower_term);
        write_u64(&mut out, self.follower_commit_index);
        write_u64(&mut out, self.follower_applied_index);
        match &self.error {
            Some(error) => {
                out.push(1);
                write_bytes(&mut out, error.as_bytes());
            }
            None => out.push(0),
        }
        out
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = FrameCursor::new(frame);
        let accepted = cursor.read_bool("accepted")?;
        let follower_term = cursor.read_u64("follower_term")?;
        let follower_commit_index = cursor.read_u64("follower_commit_index")?;
        let follower_applied_index = cursor.read_u64("follower_applied_index")?;
        let has_error = cursor.read_bool("has_error")?;
        let error = if has_error {
            let raw = cursor.read_bytes("error")?;
            Some(String::from_utf8(raw.to_vec()).map_err(|err| {
                EngineError::ProposalFailed(format!("invalid response error utf8: {err}"))
            })?)
        } else {
            None
        };
        cursor.finish()?;
        Ok(Self {
            accepted,
            follower_term,
            follower_commit_index,
            follower_applied_index,
            error,
        })
    }
}

pub fn send_append_entries_once(
    addr: SocketAddr,
    request: &AppendEntriesRequest,
    timeout: Duration,
) -> Result<AppendEntriesResponse, EngineError> {
    let mut stream = TcpStream::connect(addr).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries connect failed: {err}"))
    })?;
    stream.set_read_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set read timeout failed: {err}"))
    })?;
    stream.set_write_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set write timeout failed: {err}"))
    })?;
    write_append_entries_request(&mut stream, request)?;
    stream
        .shutdown(Shutdown::Write)
        .map_err(|err| append_entries_error(format!("request shutdown failed: {err}")))?;

    let mut response_frame = Vec::new();
    stream
        .read_to_end(&mut response_frame)
        .map_err(|err| append_entries_error(format!("receive failed: {err}")))?;
    AppendEntriesResponse::decode_frame(&response_frame)
}

pub fn load_replication_mtls_server_config(
    ca_cert: impl AsRef<Path>,
    node_cert: impl AsRef<Path>,
    node_key: impl AsRef<Path>,
) -> Result<Arc<TlsServerConfig>, EngineError> {
    let trust_roots = load_root_store(ca_cert.as_ref(), "replication client CA")?;
    let client_verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(trust_roots))
        .build()
        .map_err(|err| append_entries_error(format!("invalid client CA verifier: {err}")))?;
    let certs = load_cert_chain(node_cert.as_ref(), "replication server certificate")?;
    let key = load_private_key(node_key.as_ref(), "replication server private key")?;
    let config = TlsServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(certs, key)
        .map_err(|err| {
            append_entries_error(format!("invalid replication server cert/key: {err}"))
        })?;
    Ok(Arc::new(config))
}

pub fn load_replication_mtls_client_config(
    ca_cert: impl AsRef<Path>,
    node_cert: impl AsRef<Path>,
    node_key: impl AsRef<Path>,
) -> Result<Arc<TlsClientConfig>, EngineError> {
    let trust_roots = load_root_store(ca_cert.as_ref(), "replication server CA")?;
    let certs = load_cert_chain(node_cert.as_ref(), "replication client certificate")?;
    let key = load_private_key(node_key.as_ref(), "replication client private key")?;
    let config = TlsClientConfig::builder()
        .with_root_certificates(trust_roots)
        .with_client_auth_cert(certs, key)
        .map_err(|err| {
            append_entries_error(format!("invalid replication client cert/key: {err}"))
        })?;
    Ok(Arc::new(config))
}

pub fn load_replication_tls_client_config_without_client_auth(
    ca_cert: impl AsRef<Path>,
) -> Result<Arc<TlsClientConfig>, EngineError> {
    let trust_roots = load_root_store(ca_cert.as_ref(), "replication server CA")?;
    Ok(Arc::new(
        TlsClientConfig::builder()
            .with_root_certificates(trust_roots)
            .with_no_client_auth(),
    ))
}

pub fn send_append_entries_mtls_once(
    addr: SocketAddr,
    server_name: &str,
    client_config: Arc<TlsClientConfig>,
    request: &AppendEntriesRequest,
    timeout: Duration,
) -> Result<AppendEntriesResponse, EngineError> {
    let stream = connect_append_entries_tcp(addr, timeout)?;
    let server_name = ServerName::try_from(server_name.to_string())
        .map_err(|err| append_entries_error(format!("invalid TLS server name: {err}")))?;
    let connection = ClientConnection::new(client_config, server_name)
        .map_err(|err| append_entries_error(format!("TLS client setup failed: {err}")))?;
    let mut stream = StreamOwned::new(connection, stream);
    write_append_entries_request(&mut stream, request)?;
    stream.conn.send_close_notify();
    let mut response_frame = Vec::new();
    stream
        .read_to_end(&mut response_frame)
        .map_err(|err| append_entries_error(format!("receive failed: {err}")))?;
    AppendEntriesResponse::decode_frame(&response_frame)
}

pub fn serve_append_entries_once(
    listener: &TcpListener,
    follower: &mut RaftReplicator,
    timeout: Duration,
) -> Result<AppendEntriesResponse, EngineError> {
    listener.set_nonblocking(false).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries listener setup failed: {err}"))
    })?;
    let (mut socket, _) = listener
        .accept()
        .map_err(|err| append_entries_error(format!("accept failed: {err}")))?;
    socket.set_read_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set read timeout failed: {err}"))
    })?;
    socket.set_write_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set write timeout failed: {err}"))
    })?;

    serve_append_entries_over_stream(&mut socket, follower)
}

pub fn serve_append_entries_mtls_once(
    listener: &TcpListener,
    follower: &mut RaftReplicator,
    server_config: Arc<TlsServerConfig>,
    timeout: Duration,
) -> Result<AppendEntriesResponse, EngineError> {
    listener.set_nonblocking(false).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries listener setup failed: {err}"))
    })?;
    let (socket, _) = listener
        .accept()
        .map_err(|err| append_entries_error(format!("accept failed: {err}")))?;
    socket.set_read_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set read timeout failed: {err}"))
    })?;
    socket.set_write_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set write timeout failed: {err}"))
    })?;
    let connection = ServerConnection::new(server_config)
        .map_err(|err| append_entries_error(format!("TLS server setup failed: {err}")))?;
    let mut stream = StreamOwned::new(connection, socket);
    let mut request_frame = Vec::new();
    stream
        .read_to_end(&mut request_frame)
        .map_err(|err| append_entries_error(format!("receive failed: {err}")))?;
    let request = AppendEntriesRequest::decode_frame(&request_frame)?;
    let response = request.apply_to(follower);
    stream
        .write_all(&response.encode_frame())
        .map_err(|err| append_entries_error(format!("response send failed: {err}")))?;
    stream.conn.send_close_notify();
    stream
        .flush()
        .map_err(|err| append_entries_error(format!("response flush failed: {err}")))?;
    Ok(response)
}

fn connect_append_entries_tcp(
    addr: SocketAddr,
    timeout: Duration,
) -> Result<TcpStream, EngineError> {
    let stream = TcpStream::connect(addr)
        .map_err(|err| append_entries_error(format!("connect failed: {err}")))?;
    stream.set_read_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set read timeout failed: {err}"))
    })?;
    stream.set_write_timeout(Some(timeout)).map_err(|err| {
        EngineError::ProposalFailed(format!("append entries set write timeout failed: {err}"))
    })?;
    Ok(stream)
}

fn serve_append_entries_over_stream(
    stream: &mut impl ReadWrite,
    follower: &mut RaftReplicator,
) -> Result<AppendEntriesResponse, EngineError> {
    let mut request_frame = Vec::new();
    stream
        .read_to_end(&mut request_frame)
        .map_err(|err| append_entries_error(format!("receive failed: {err}")))?;
    let request = AppendEntriesRequest::decode_frame(&request_frame)?;
    let response = request.apply_to(follower);
    stream
        .write_all(&response.encode_frame())
        .map_err(|err| append_entries_error(format!("response send failed: {err}")))?;
    Ok(response)
}

fn write_append_entries_request(
    stream: &mut impl Write,
    request: &AppendEntriesRequest,
) -> Result<(), EngineError> {
    stream
        .write_all(&request.encode_frame())
        .map_err(|err| append_entries_error(format!("send failed: {err}")))
}

trait ReadWrite: Read + Write {}

impl<T: Read + Write> ReadWrite for T {}

fn load_cert_chain(
    path: &Path,
    label: &'static str,
) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>, EngineError> {
    let mut reader = std::io::BufReader::new(
        File::open(path)
            .map_err(|err| append_entries_error(format!("{label} open failed: {err}")))?,
    );
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| append_entries_error(format!("{label} PEM parse failed: {err}")))?;
    if certs.is_empty() {
        return Err(append_entries_error(format!(
            "{label} contains no certificates"
        )));
    }
    Ok(certs)
}

fn load_private_key(
    path: &Path,
    label: &'static str,
) -> Result<rustls::pki_types::PrivateKeyDer<'static>, EngineError> {
    let mut reader = std::io::BufReader::new(
        File::open(path)
            .map_err(|err| append_entries_error(format!("{label} open failed: {err}")))?,
    );
    rustls_pemfile::private_key(&mut reader)
        .map_err(|err| append_entries_error(format!("{label} PEM parse failed: {err}")))?
        .ok_or_else(|| append_entries_error(format!("{label} contains no private key")))
}

fn load_root_store(path: &Path, label: &'static str) -> Result<RootCertStore, EngineError> {
    let mut roots = RootCertStore::empty();
    for cert in load_cert_chain(path, label)? {
        roots
            .add(cert)
            .map_err(|err| append_entries_error(format!("{label} trust root rejected: {err}")))?;
    }
    Ok(roots)
}

fn append_entries_error(message: String) -> EngineError {
    EngineError::ProposalFailed(format!("append entries {message}"))
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    write_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

struct FrameCursor<'a> {
    frame: &'a [u8],
    offset: usize,
}

impl<'a> FrameCursor<'a> {
    fn new(frame: &'a [u8]) -> Self {
        Self { frame, offset: 0 }
    }

    fn read_bool(&mut self, field: &'static str) -> Result<bool, EngineError> {
        let byte = *self
            .frame
            .get(self.offset)
            .ok_or_else(|| frame_error(format!("missing {field}")))?;
        self.offset += 1;
        match byte {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(frame_error(format!("invalid {field} bool byte {other}"))),
        }
    }

    fn read_u64(&mut self, field: &'static str) -> Result<u64, EngineError> {
        let bytes = self.read_exact(field, 8)?;
        Ok(u64::from_be_bytes(
            bytes
                .try_into()
                .expect("read_exact with len 8 should return 8 bytes"),
        ))
    }

    fn read_len(&mut self, field: &'static str) -> Result<usize, EngineError> {
        let raw = self.read_u64(field)?;
        usize::try_from(raw).map_err(|_| frame_error(format!("{field} length exceeds usize")))
    }

    fn read_bytes(&mut self, field: &'static str) -> Result<&'a [u8], EngineError> {
        let len = self.read_len(field)?;
        self.read_exact(field, len)
    }

    fn read_exact(&mut self, field: &'static str, len: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| frame_error(format!("{field} length overflows frame cursor")))?;
        if end > self.frame.len() {
            return Err(frame_error(format!(
                "{field} needs {len} bytes but frame has {} remaining",
                self.frame.len().saturating_sub(self.offset)
            )));
        }
        let bytes = &self.frame[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn finish(&self) -> Result<(), EngineError> {
        if self.offset == self.frame.len() {
            Ok(())
        } else {
            Err(frame_error(format!(
                "frame has {} trailing bytes",
                self.frame.len() - self.offset
            )))
        }
    }
}

fn frame_error(message: String) -> EngineError {
    EngineError::ProposalFailed(format!("append entries frame decode failed: {message}"))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicationStatusInvariantError {
    #[error("live progress is invalid: {0}")]
    LiveProgress(#[from] ReplicationProgressInvariantError),
    #[error("durable progress is invalid: {0}")]
    DurableProgress(ReplicationProgressInvariantError),
    #[error(
        "durable progress cannot be ahead of live progress for {field}: durable={durable}, live={live}"
    )]
    DurableAheadOfLive {
        field: &'static str,
        durable: u64,
        live: u64,
    },
    #[error(
        "snapshot identity drift at frontier index={last_included_index} term={last_included_term}: durable snapshot_id={durable_snapshot_id}, live snapshot_id={live_snapshot_id}"
    )]
    SnapshotIdentityDrift {
        last_included_index: Index,
        last_included_term: Term,
        durable_snapshot_id: u64,
        live_snapshot_id: u64,
    },
    #[error("durable term {durable} does not match live term {live}")]
    TermMismatch { durable: Term, live: Term },
    #[error("recovery gap {actual:?} does not match live-vs-durable delta {expected:?}")]
    RecoveryGapMismatch {
        expected: RecoveryProgressGap,
        actual: RecoveryProgressGap,
    },
}

impl ReplicationStatusSnapshot {
    pub fn new(
        live: ReplicationProgress,
        durable: ReplicationProgress,
        recovery_gap: RecoveryProgressGap,
    ) -> Result<Self, ReplicationStatusInvariantError> {
        let snapshot = Self {
            live,
            durable,
            recovery_gap,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), ReplicationStatusInvariantError> {
        self.live
            .validate()
            .map_err(ReplicationStatusInvariantError::LiveProgress)?;
        self.durable
            .validate()
            .map_err(ReplicationStatusInvariantError::DurableProgress)?;

        if self.durable.term != self.live.term {
            return Err(ReplicationStatusInvariantError::TermMismatch {
                durable: self.durable.term,
                live: self.live.term,
            });
        }

        for (field, durable, live) in [
            (
                "snapshot.last_included_index",
                self.durable.snapshot.last_included_index,
                self.live.snapshot.last_included_index,
            ),
            (
                "snapshot.last_included_term",
                self.durable.snapshot.last_included_term,
                self.live.snapshot.last_included_term,
            ),
            (
                "commit_index",
                self.durable.commit_index,
                self.live.commit_index,
            ),
            (
                "applied_index",
                self.durable.applied_index,
                self.live.applied_index,
            ),
            ("next_index", self.durable.next_index, self.live.next_index),
            (
                "uncommitted_entry_count",
                self.durable.uncommitted_entry_count as u64,
                self.live.uncommitted_entry_count as u64,
            ),
        ] {
            if durable > live {
                return Err(ReplicationStatusInvariantError::DurableAheadOfLive {
                    field,
                    durable,
                    live,
                });
            }
        }

        if self.durable.snapshot.last_included_index == self.live.snapshot.last_included_index
            && self.durable.snapshot.last_included_term == self.live.snapshot.last_included_term
            && self.durable.snapshot.snapshot_id != self.live.snapshot.snapshot_id
        {
            return Err(ReplicationStatusInvariantError::SnapshotIdentityDrift {
                last_included_index: self.live.snapshot.last_included_index,
                last_included_term: self.live.snapshot.last_included_term,
                durable_snapshot_id: self.durable.snapshot.snapshot_id,
                live_snapshot_id: self.live.snapshot.snapshot_id,
            });
        }

        let expected = Self::recovery_gap_between(&self.live, &self.durable);
        if self.recovery_gap != expected {
            return Err(ReplicationStatusInvariantError::RecoveryGapMismatch {
                expected,
                actual: self.recovery_gap.clone(),
            });
        }

        Ok(())
    }

    pub fn is_restart_equivalent(&self) -> bool {
        self.recovery_gap.is_restart_equivalent()
    }

    pub fn has_speculative_tail(&self) -> bool {
        self.recovery_gap.has_speculative_tail()
    }

    fn recovery_gap_between(
        live: &ReplicationProgress,
        durable: &ReplicationProgress,
    ) -> RecoveryProgressGap {
        RecoveryProgressGap {
            commit_index_gap: live.commit_index.saturating_sub(durable.commit_index) as usize,
            applied_index_gap: live.applied_index.saturating_sub(durable.applied_index) as usize,
            next_index_gap: live.next_index.saturating_sub(durable.next_index) as usize,
            uncommitted_entry_gap: live
                .uncommitted_entry_count
                .saturating_sub(durable.uncommitted_entry_count),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplicationProgressInvariantError {
    #[error("applied_index {applied_index} exceeds commit_index {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
    #[error("next_index {next_index} is behind commit boundary {commit_index}")]
    NextIndexBehindCommit {
        next_index: Index,
        commit_index: Index,
    },
    #[error(
        "committed_but_unapplied_count {committed_but_unapplied_count} does not match commit/apply gap {expected}"
    )]
    PendingApplyCountMismatch {
        committed_but_unapplied_count: usize,
        expected: usize,
    },
    #[error(
        "has_committed_entries_pending_apply {has_pending} does not match commit/apply gap {expected}"
    )]
    PendingApplyFlagMismatch { has_pending: bool, expected: bool },
    #[error(
        "has_uncommitted_entries {has_uncommitted} does not match uncommitted_entry_count {uncommitted_entry_count}"
    )]
    UncommittedFlagMismatch {
        has_uncommitted: bool,
        uncommitted_entry_count: usize,
    },
    #[error("snapshot last_included_index {snapshot_index} exceeds applied_index {applied_index}")]
    SnapshotAheadOfApplied {
        snapshot_index: Index,
        applied_index: Index,
    },
}

impl ReplicationProgress {
    pub fn apply_gap(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0 && !self.has_uncommitted_entries
    }

    pub fn validate(&self) -> Result<(), ReplicationProgressInvariantError> {
        if self.applied_index > self.commit_index {
            return Err(ReplicationProgressInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index: self.commit_index,
            });
        }
        if self.next_index < self.commit_index + 1 {
            return Err(ReplicationProgressInvariantError::NextIndexBehindCommit {
                next_index: self.next_index,
                commit_index: self.commit_index,
            });
        }

        let expected_pending = self.apply_gap();
        if self.committed_but_unapplied_count != expected_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyCountMismatch {
                    committed_but_unapplied_count: self.committed_but_unapplied_count,
                    expected: expected_pending,
                },
            );
        }

        let expected_has_pending = expected_pending > 0;
        if self.has_committed_entries_pending_apply != expected_has_pending {
            return Err(
                ReplicationProgressInvariantError::PendingApplyFlagMismatch {
                    has_pending: self.has_committed_entries_pending_apply,
                    expected: expected_has_pending,
                },
            );
        }

        let expected_has_uncommitted = self.uncommitted_entry_count > 0;
        if self.has_uncommitted_entries != expected_has_uncommitted {
            return Err(ReplicationProgressInvariantError::UncommittedFlagMismatch {
                has_uncommitted: self.has_uncommitted_entries,
                uncommitted_entry_count: self.uncommitted_entry_count,
            });
        }

        if self.snapshot.last_included_index > self.applied_index {
            return Err(ReplicationProgressInvariantError::SnapshotAheadOfApplied {
                snapshot_index: self.snapshot.last_included_index,
                applied_index: self.applied_index,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryInvariantError {
    #[error("applied_index {applied_index} is behind snapshot boundary {snapshot_index}")]
    AppliedBehindSnapshot {
        applied_index: Index,
        snapshot_index: Index,
    },
    #[error("recovery entries must be contiguous from {expected_index} but saw {actual_index}")]
    NonContiguousEntries {
        expected_index: Index,
        actual_index: Index,
    },
    #[error("recovery entry term {entry_term} exceeds local term {local_term} at index {index}")]
    EntryTermExceedsLocal {
        entry_term: Term,
        local_term: Term,
        index: Index,
    },
    #[error("applied_index {applied_index} exceeds commit boundary {commit_index}")]
    AppliedExceedsCommit {
        applied_index: Index,
        commit_index: Index,
    },
}

impl RecoveryState {
    pub fn commit_index(&self) -> Index {
        self.committed_entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.snapshot.last_included_index)
    }

    pub fn next_index(&self) -> Index {
        self.commit_index() + 1
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.applied_index < self.commit_index()
    }

    pub fn apply_gap(&self) -> usize {
        self.commit_index().saturating_sub(self.applied_index) as usize
    }

    pub fn is_caught_up(&self) -> bool {
        self.apply_gap() == 0
    }

    pub fn progress_as_follower(&self) -> Result<ReplicationProgress, RecoveryInvariantError> {
        self.validate()?;
        let progress = ReplicationProgress {
            role: Role::Follower,
            term: self.term.max(self.snapshot.last_included_term),
            commit_index: self.commit_index(),
            applied_index: self.applied_index,
            next_index: self.next_index(),
            snapshot: self.snapshot.clone(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        progress.validate().expect(
            "validated recovery state should always map to valid follower replication progress",
        );
        Ok(progress)
    }

    pub fn validate(&self) -> Result<(), RecoveryInvariantError> {
        if self.applied_index < self.snapshot.last_included_index {
            return Err(RecoveryInvariantError::AppliedBehindSnapshot {
                applied_index: self.applied_index,
                snapshot_index: self.snapshot.last_included_index,
            });
        }

        for (expected_index, entry) in
            (self.snapshot.last_included_index + 1..).zip(self.committed_entries.iter())
        {
            if entry.index != expected_index {
                return Err(RecoveryInvariantError::NonContiguousEntries {
                    expected_index,
                    actual_index: entry.index,
                });
            }
            if entry.term > self.term {
                return Err(RecoveryInvariantError::EntryTermExceedsLocal {
                    entry_term: entry.term,
                    local_term: self.term,
                    index: entry.index,
                });
            }
        }

        let commit_index = self.commit_index();
        if self.applied_index > commit_index {
            return Err(RecoveryInvariantError::AppliedExceedsCommit {
                applied_index: self.applied_index,
                commit_index,
            });
        }

        Ok(())
    }
}

pub trait LogReplicator {
    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError>;

    /// W1b — drop retained log entries that are both COMMITTED and APPLIED (never read again:
    /// `drain_committed_from` only yields entries past the applied index). Default no-op; the
    /// local single-node replicator compacts its in-memory Vec (the third unbounded per-commit
    /// structure alongside the WAL buffer and the timestamp map); Raft keeps its own log
    /// management (follower catch-up may still need applied entries).
    fn compact_applied_prefix(&mut self) {}
    fn wait_committed(
        &self,
        token: CommitToken,
        timeout: std::time::Duration,
    ) -> Result<Index, EngineError>;
    fn role(&self) -> Role;
    fn current_term(&self) -> Term;
    fn commit_index(&self) -> Index;
    fn applied_index(&self) -> Index;
    fn snapshot_meta(&self) -> SnapshotMeta;
}

pub trait ReplicatedStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError>;
}

#[derive(Debug)]
pub struct LocalReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    snapshot_id: u64,
}

impl LocalReplicator {
    pub fn leader() -> Self {
        Self {
            term: 1,
            next_index: 1,
            commit_index: 0,
            applied_index: 0,
            applied_term: 0,
            role: Role::Leader,
            entries: Vec::new(),
            snapshot_id: 0,
        }
    }

    pub fn become_follower(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Follower;
    }

    pub fn become_leader(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Leader;
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.term = self.term.max(term);
        self.role = Role::Candidate;
    }

    /// The `Index` the NEXT [`LogReplicator::propose`] will assign, WITHOUT consuming it. The
    /// concurrent commit path peeks this (under the single-proposer commit_mutex) to re-resolve a
    /// transaction's delta at its would-be `commit_seq` BEFORE proposing, so a re-validation failure
    /// can abort without ever consuming an index (no commit-seq hole, nothing durable).
    pub fn peek_next_index(&self) -> Index {
        self.next_index
    }

    /// O(1) lookup of the retained log entry at `index`. The log is contiguous in `index` (prefix-
    /// compacted by `install_snapshot`, suffix-trimmed by `rollback_unapplied_from`), so the position
    /// is `index - entries[0].index`. `None` if `index` was prefix-compacted away or is past the tail.
    fn entry_at(&self, index: Index) -> Option<&LogEntry> {
        let first = self.entries.first()?.index;
        let pos = index.checked_sub(first)? as usize;
        let entry = self.entries.get(pos)?;
        debug_assert_eq!(
            entry.index, index,
            "LocalReplicator log must stay contiguous in index"
        );
        Some(entry)
    }

    /// Committed entries with index in `(start_exclusive, commit_index]`, in order. O(k) in the
    /// number yielded, NOT O(entries): the contiguous log lets the window map directly to a slice
    /// range, so the commit hot path no longer scans the unbounded entries vec. Yields exactly the
    /// same set as the former `entries.iter().filter(index > start && index <= commit_index)`.
    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        let (start_pos, end_pos) = match self.entries.first().map(|e| e.index) {
            Some(first) if self.commit_index >= first => {
                let lo = start_exclusive.saturating_add(1).max(first);
                let start = (lo - first) as usize;
                let end = ((self.commit_index - first) as usize + 1).min(self.entries.len());
                (start.min(end), end)
            }
            _ => (0, 0),
        };
        self.entries[start_pos..end_pos].iter()
    }

    pub fn retained_entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.commit_index > self.applied_index
    }

    pub fn mark_applied(&mut self, idx: Index) {
        let bounded = idx.min(self.commit_index);
        if bounded <= self.applied_index {
            return;
        }

        self.applied_index = bounded;
        // O(1) via the contiguous-log index->position map (was an O(n) `entries.iter().find`).
        if let Some(term) = self.entry_at(bounded).map(|entry| entry.term) {
            self.applied_term = term;
        }
    }

    /// W1b: drop the applied prefix (see [`LogReplicator::compact_applied_prefix`]). Keeps the
    /// log contiguous-from-first (the `entry_at` invariant): only a PREFIX is removed.
    pub fn compact_applied_prefix_inner(&mut self) {
        let applied = self.applied_index;
        self.entries.retain(|e| e.index > applied);
    }

    pub fn rollback_unapplied_from(&mut self, index_inclusive: Index) {
        if index_inclusive <= self.applied_index {
            return;
        }

        self.entries.retain(|e| e.index < index_inclusive);
        self.commit_index = self
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(self.applied_index);
        self.next_index = self.commit_index + 1;
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
        self.snapshot_meta()
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let current = self.snapshot_meta();
        let advances_frontier = meta.last_included_index > current.last_included_index;
        let same_frontier_same_term = meta.last_included_index == current.last_included_index
            && meta.last_included_term == current.last_included_term;
        let regresses_term_on_advanced_frontier =
            advances_frontier && meta.last_included_term < current.last_included_term;
        if regresses_term_on_advanced_frontier || (!advances_frontier && !same_frontier_same_term) {
            return;
        }

        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = meta.snapshot_id;
        self.entries.retain(|e| e.index > meta.last_included_index);

        let tail_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    pub fn progress(&self) -> ReplicationProgress {
        let progress = ReplicationProgress {
            role: self.role,
            term: self.term,
            commit_index: self.commit_index,
            applied_index: self.applied_index,
            next_index: self.next_index,
            snapshot: self.snapshot_meta(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        progress
            .validate()
            .expect("local replication progress invariants should hold");
        progress
    }

    pub fn status_snapshot(&self) -> ReplicationStatusSnapshot {
        let live = self.progress();
        ReplicationStatusSnapshot::new(
            live.clone(),
            live,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .expect("local replication status snapshot invariants should hold")
    }
}

#[derive(Debug)]
pub struct RaftReplicator {
    term: Term,
    next_index: Index,
    commit_index: Index,
    applied_index: Index,
    applied_term: Term,
    role: Role,
    entries: Vec<LogEntry>,
    snapshot_id: u64,
    compacted_index: Index,
    compacted_term: Term,
    voters: usize,
    quorum: usize,
    voted_for: Option<u64>,
    ack_counts: BTreeMap<Index, BTreeSet<u64>>,
}

impl RaftReplicator {
    pub fn new(voters: usize) -> Self {
        assert!(voters >= 1, "raft requires at least one voter");
        let quorum = (voters / 2) + 1;
        Self {
            term: 1,
            next_index: 1,
            commit_index: 0,
            applied_index: 0,
            applied_term: 0,
            role: Role::Follower,
            entries: Vec::new(),
            snapshot_id: 0,
            compacted_index: 0,
            compacted_term: 0,
            voters,
            quorum,
            voted_for: None,
            ack_counts: BTreeMap::new(),
        }
    }

    pub fn single_node_leader() -> Self {
        let mut s = Self::new(1);
        s.role = Role::Leader;
        s
    }

    pub fn resume_as_follower(voters: usize, recovery: RecoveryState) -> Result<Self, EngineError> {
        recovery
            .validate()
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        let commit_index = recovery.commit_index();
        let next_index = recovery.next_index();

        let mut replicator = Self::new(voters);
        replicator.term = recovery.term.max(recovery.snapshot.last_included_term);
        replicator.role = Role::Follower;
        replicator.commit_index = commit_index;
        replicator.applied_index = recovery.applied_index;
        replicator.applied_term = if recovery.applied_index == recovery.snapshot.last_included_index
        {
            recovery.snapshot.last_included_term
        } else {
            recovery
                .committed_entries
                .iter()
                .find(|entry| entry.index == recovery.applied_index)
                .map(|entry| entry.term)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "missing applied entry {} in recovery state",
                        recovery.applied_index
                    ))
                })?
        };
        replicator.snapshot_id = recovery.snapshot.snapshot_id;
        replicator.compacted_index = recovery.snapshot.last_included_index;
        replicator.compacted_term = recovery.snapshot.last_included_term;
        replicator.entries = recovery.committed_entries;
        replicator.next_index = next_index;
        Ok(replicator)
    }

    pub fn become_follower(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Follower;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_leader(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Leader;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn become_candidate(&mut self, term: Term) {
        let previous_term = self.term;
        self.term = self.term.max(term);
        self.role = Role::Candidate;
        if self.term > previous_term {
            self.voted_for = None;
        }
        self.entries.retain(|e| e.index <= self.commit_index);
        self.next_index = self.commit_index + 1;
        self.ack_counts.clear();
    }

    pub fn start_candidate_election(&mut self, candidate_id: u64) -> RequestVoteRequest {
        assert!(candidate_id > 0, "candidate id must be non-zero");
        self.term += 1;
        self.role = Role::Candidate;
        self.voted_for = Some(candidate_id);
        self.ack_counts.clear();
        let (last_log_index, last_log_term) = self.last_log_position();
        RequestVoteRequest {
            candidate_term: self.term,
            candidate_id,
            last_log_index,
            last_log_term,
        }
    }

    pub fn request_vote_from_candidate(
        &mut self,
        request: &RequestVoteRequest,
    ) -> RequestVoteResponse {
        if request.candidate_term < self.term {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "stale candidate term {} (local term {})",
                    request.candidate_term, self.term
                )),
            };
        }

        if request.candidate_term > self.term {
            self.term = request.candidate_term;
            self.role = Role::Follower;
            self.voted_for = None;
            self.ack_counts.clear();
        }

        let already_voted_elsewhere = self
            .voted_for
            .is_some_and(|voted_for| voted_for != request.candidate_id);
        if already_voted_elsewhere {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "already voted for candidate {} in term {}",
                    self.voted_for.expect("checked is_some above"),
                    self.term
                )),
            };
        }

        if !self.candidate_log_is_up_to_date(request.last_log_index, request.last_log_term) {
            return RequestVoteResponse {
                granted: false,
                voter_term: self.term,
                error: Some(format!(
                    "candidate log is behind voter log at index {} term {}",
                    self.last_log_position().0,
                    self.last_log_position().1
                )),
            };
        }

        self.voted_for = Some(request.candidate_id);
        self.role = Role::Follower;
        RequestVoteResponse {
            granted: true,
            voter_term: self.term,
            error: None,
        }
    }

    pub fn voter_count(&self) -> usize {
        self.voters
    }

    pub fn quorum_size(&self) -> usize {
        self.quorum
    }

    pub fn drain_committed_from(&self, start_exclusive: Index) -> impl Iterator<Item = &LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.index > start_exclusive && e.index <= self.commit_index)
    }

    pub fn retained_entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn committed_but_unapplied_count(&self) -> usize {
        self.commit_index.saturating_sub(self.applied_index) as usize
    }

    pub fn has_committed_entries_pending_apply(&self) -> bool {
        self.commit_index > self.applied_index
    }

    pub fn uncommitted_entry_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.index > self.commit_index)
            .count()
    }

    pub fn has_uncommitted_entries(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.index > self.commit_index)
    }

    pub fn mark_applied(&mut self, idx: Index) {
        let bounded = idx.min(self.commit_index);
        if bounded <= self.applied_index {
            return;
        }

        self.applied_index = bounded;
        if let Some(entry) = self.entries.iter().find(|entry| entry.index == bounded) {
            self.applied_term = entry.term;
        }
    }

    pub fn truncate_uncommitted_from(&mut self, index_inclusive: Index) {
        if index_inclusive <= self.commit_index {
            return;
        }

        self.entries.retain(|e| e.index < index_inclusive);
        self.ack_counts.retain(|idx, _| *idx < index_inclusive);

        let tail_index = self
            .entries
            .last()
            .map(|e| e.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    pub fn register_follower_ack(&mut self, index: Index, follower_id: u64) {
        if self.role != Role::Leader || index == 0 || index >= self.next_index || follower_id == 0 {
            return;
        }

        let Some(acks) = self.ack_counts.get_mut(&index) else {
            return;
        };
        acks.insert(follower_id);

        while self.commit_index + 1 < self.next_index {
            let next = self.commit_index + 1;
            let Some(acks) = self.ack_counts.get(&next) else {
                break;
            };
            if acks.len() >= self.quorum {
                self.commit_index = next;
            } else {
                break;
            }
        }

        self.ack_counts.retain(|idx, _| *idx > self.commit_index);
    }

    pub fn export_snapshot_meta(&mut self) -> SnapshotMeta {
        self.snapshot_id += 1;
        self.snapshot_meta()
    }

    pub fn recovery_state(&self) -> RecoveryState {
        let snapshot = self.snapshot_meta();
        RecoveryState {
            term: self.term,
            snapshot: snapshot.clone(),
            committed_entries: self
                .entries
                .iter()
                .filter(|entry| {
                    entry.index > snapshot.last_included_index && entry.index <= self.commit_index
                })
                .cloned()
                .collect(),
            applied_index: self.applied_index,
        }
    }

    pub fn recovery_progress(&self) -> ReplicationProgress {
        self.recovery_state().progress_as_follower().expect(
            "live raft recovery state should always map to valid follower recovery progress",
        )
    }

    pub fn recovery_progress_gap(&self) -> RecoveryProgressGap {
        ReplicationStatusSnapshot::recovery_gap_between(&self.progress(), &self.recovery_progress())
    }

    pub fn status_snapshot(&self) -> ReplicationStatusSnapshot {
        ReplicationStatusSnapshot::new(
            self.progress(),
            self.recovery_progress(),
            self.recovery_progress_gap(),
        )
        .expect("raft replication status snapshot invariants should hold")
    }

    pub fn progress(&self) -> ReplicationProgress {
        let progress = ReplicationProgress {
            role: self.role,
            term: self.term,
            commit_index: self.commit_index,
            applied_index: self.applied_index,
            next_index: self.next_index,
            snapshot: self.snapshot_meta(),
            committed_but_unapplied_count: self.committed_but_unapplied_count(),
            has_committed_entries_pending_apply: self.has_committed_entries_pending_apply(),
            uncommitted_entry_count: self.uncommitted_entry_count(),
            has_uncommitted_entries: self.has_uncommitted_entries(),
        };
        progress
            .validate()
            .expect("raft replication progress invariants should hold");
        progress
    }

    pub fn append_entries_from_leader(
        &mut self,
        leader_term: Term,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> Result<(), EngineError> {
        if self.role == Role::Leader {
            return Err(EngineError::ProposalFailed(
                "leader cannot accept follower append path".to_string(),
            ));
        }

        if leader_term < self.term {
            return Err(EngineError::ProposalFailed(format!(
                "stale leader term {} (local term {})",
                leader_term, self.term
            )));
        }

        if self.role != Role::Follower || leader_term > self.term {
            self.become_follower(leader_term);
        } else {
            self.term = leader_term;
            self.role = Role::Follower;
        }

        if prev_log_index < self.compacted_index {
            return Err(EngineError::ProposalFailed(format!(
                "prev_log_index={} is behind compacted boundary {}",
                prev_log_index, self.compacted_index
            )));
        }

        if prev_log_index > 0 {
            let Some(local_prev_term) = self.term_at(prev_log_index) else {
                return Err(EngineError::ProposalFailed(format!(
                    "missing prev_log_index={} for append",
                    prev_log_index
                )));
            };

            if local_prev_term != prev_log_term {
                return Err(EngineError::ProposalFailed(format!(
                    "prev_log_term mismatch at index {}: local={}, remote={}",
                    prev_log_index, local_prev_term, prev_log_term
                )));
            }
        }

        for (expected_index, entry) in (prev_log_index + 1..).zip(entries.iter()) {
            if entry.term > leader_term {
                return Err(EngineError::ProposalFailed(format!(
                    "entry term {} exceeds leader term {} at index {}",
                    entry.term, leader_term, entry.index
                )));
            }

            if entry.index != expected_index {
                return Err(EngineError::ProposalFailed(format!(
                    "append entries must be contiguous from {} but saw {}",
                    prev_log_index + 1,
                    entry.index
                )));
            }
        }

        for incoming in entries {
            if let Some(existing) = self.entries.iter().find(|e| e.index == incoming.index) {
                if existing.term == incoming.term {
                    if existing.payload != incoming.payload {
                        return Err(EngineError::ProposalFailed(format!(
                            "payload mismatch at index {} term {}",
                            incoming.index, incoming.term
                        )));
                    }
                    continue;
                }

                if incoming.index <= self.commit_index {
                    return Err(EngineError::ProposalFailed(format!(
                        "refusing to overwrite committed index {}",
                        incoming.index
                    )));
                }

                self.truncate_uncommitted_from(incoming.index);
            }

            if self
                .entries
                .iter()
                .all(|entry| entry.index != incoming.index)
            {
                self.entries.push(incoming);
            }
        }

        self.entries.sort_by_key(|entry| entry.index);

        let last_local_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        let target_commit = leader_commit.min(last_local_index);
        if target_commit > self.commit_index {
            self.commit_index = target_commit;
        }
        self.next_index = last_local_index + 1;

        Ok(())
    }

    pub fn install_snapshot(&mut self, meta: SnapshotMeta) {
        let current = self.snapshot_meta();
        let advances_frontier = meta.last_included_index > current.last_included_index;
        let same_frontier_same_term = meta.last_included_index == current.last_included_index
            && meta.last_included_term == current.last_included_term;
        let retains_existing_suffix = same_frontier_same_term
            || self.term_at(meta.last_included_index) == Some(meta.last_included_term);
        let regresses_term_on_advanced_frontier =
            advances_frontier && meta.last_included_term < current.last_included_term;
        if regresses_term_on_advanced_frontier || (!advances_frontier && !same_frontier_same_term) {
            return;
        }

        self.term = self.term.max(meta.last_included_term);
        self.commit_index = self.commit_index.max(meta.last_included_index);
        if meta.last_included_index > self.compacted_index {
            self.compacted_index = meta.last_included_index;
            self.compacted_term = meta.last_included_term;
        }
        if meta.last_included_index > self.applied_index {
            self.applied_index = meta.last_included_index;
            self.applied_term = meta.last_included_term;
        }
        self.snapshot_id = meta.snapshot_id;
        self.entries.retain(|entry| {
            entry.index > meta.last_included_index
                && (retains_existing_suffix || entry.index <= self.commit_index)
        });
        if retains_existing_suffix {
            self.ack_counts
                .retain(|idx, _| *idx > meta.last_included_index);
        } else {
            self.ack_counts.clear();
        }

        let tail_index = self
            .entries
            .last()
            .map(|entry| entry.index)
            .unwrap_or(self.commit_index);
        self.next_index = tail_index + 1;
    }

    fn term_at(&self, index: Index) -> Option<Term> {
        if index == 0 {
            return Some(0);
        }

        if index == self.compacted_index {
            return Some(self.compacted_term);
        }

        if index < self.compacted_index {
            return None;
        }

        if let Some(entry) = self.entries.iter().find(|entry| entry.index == index) {
            return Some(entry.term);
        }

        if index == self.applied_index {
            return Some(self.applied_term);
        }

        None
    }

    fn last_log_position(&self) -> (Index, Term) {
        if let Some(entry) = self.entries.last() {
            (entry.index, entry.term)
        } else {
            (self.compacted_index, self.compacted_term)
        }
    }

    fn candidate_log_is_up_to_date(
        &self,
        candidate_last_index: Index,
        candidate_last_term: Term,
    ) -> bool {
        let (last_index, last_term) = self.last_log_position();
        candidate_last_term > last_term
            || (candidate_last_term == last_term && candidate_last_index >= last_index)
    }
}

impl LogReplicator for LocalReplicator {
    fn compact_applied_prefix(&mut self) {
        self.compact_applied_prefix_inner();
    }

    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let idx = self.next_index;
        self.next_index += 1;

        let entry = LogEntry {
            term: self.term,
            index: idx,
            payload,
        };

        self.entries.push(entry);
        self.commit_index = idx;

        Ok(CommitToken { index: idx })
    }

    fn wait_committed(
        &self,
        token: CommitToken,
        _timeout: std::time::Duration,
    ) -> Result<Index, EngineError> {
        if self.commit_index >= token.index {
            Ok(token.index)
        } else {
            Err(EngineError::ProposalFailed(format!(
                "token {} is not committed yet (commit_index={})",
                token.index, self.commit_index
            )))
        }
    }

    fn role(&self) -> Role {
        self.role
    }

    fn current_term(&self) -> Term {
        self.term
    }

    fn commit_index(&self) -> Index {
        self.commit_index
    }

    fn applied_index(&self) -> Index {
        self.applied_index
    }

    fn snapshot_meta(&self) -> SnapshotMeta {
        SnapshotMeta {
            last_included_index: self.applied_index,
            last_included_term: self.applied_term,
            snapshot_id: self.snapshot_id,
        }
    }
}

impl LogReplicator for RaftReplicator {
    fn propose(&mut self, payload: std::sync::Arc<[u8]>) -> Result<CommitToken, EngineError> {
        if self.role != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let idx = self.next_index;
        self.next_index += 1;

        self.entries.push(LogEntry {
            term: self.term,
            index: idx,
            payload,
        });

        // Leader has an implicit self-ack represented by voter id 0.
        let mut acks = BTreeSet::new();
        acks.insert(0);
        self.ack_counts.insert(idx, acks);
        if self.quorum == 1 {
            self.commit_index = idx;
        }

        Ok(CommitToken { index: idx })
    }

    fn wait_committed(
        &self,
        token: CommitToken,
        _timeout: std::time::Duration,
    ) -> Result<Index, EngineError> {
        if self.commit_index >= token.index {
            Ok(token.index)
        } else {
            Err(EngineError::ProposalFailed(format!(
                "token {} is not committed yet (commit_index={})",
                token.index, self.commit_index
            )))
        }
    }

    fn role(&self) -> Role {
        self.role
    }

    fn current_term(&self) -> Term {
        self.term
    }

    fn commit_index(&self) -> Index {
        self.commit_index
    }

    fn applied_index(&self) -> Index {
        self.applied_index
    }

    fn snapshot_meta(&self) -> SnapshotMeta {
        SnapshotMeta {
            last_included_index: self.applied_index,
            last_included_term: self.applied_term,
            snapshot_id: self.snapshot_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct AppliedLog {
        values: Vec<String>,
    }

    impl ReplicatedStateMachine for AppliedLog {
        fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
            let value = std::str::from_utf8(&entry.payload)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            self.values.push(value.to_string());
            Ok(())
        }
    }

    fn apply_committed_entries(
        node: &mut RaftReplicator,
        state: &mut AppliedLog,
    ) -> Result<(), EngineError> {
        let committed: Vec<LogEntry> = node
            .drain_committed_from(node.applied_index())
            .cloned()
            .collect();
        for entry in committed {
            state.apply(&entry)?;
            node.mark_applied(entry.index);
        }
        Ok(())
    }

    fn apply_append_request(
        follower: &mut RaftReplicator,
        leader_term: Term,
        prev_log_index: Index,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: Index,
    ) -> AppendEntriesResponse {
        AppendEntriesRequest {
            leader_term,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        }
        .apply_to(follower)
    }

    #[test]
    fn operational_replication_three_node_smoke_catches_up_reads_after_apply_and_gates_failover() {
        let mut leader = RaftReplicator::new(3);
        let mut follower_a = RaftReplicator::new(3);
        let mut follower_b = RaftReplicator::new(3);
        let mut state_a = AppliedLog::default();
        let mut state_b = AppliedLog::default();
        let mut append_batches_sent = 0;
        let mut heartbeat_batches_sent = 0;
        let mut follower_acks_recorded = 0;

        leader.become_leader(1);
        assert!(matches!(
            follower_a.propose(b"blocked follower write".to_vec().into()),
            Err(EngineError::NotLeader)
        ));

        let first = leader
            .propose(b"create table t(id int)".to_vec().into())
            .unwrap();
        let second = leader
            .propose(b"insert into t values (1)".to_vec().into())
            .unwrap();
        let term_one = leader.current_term();
        let first_batch = vec![
            LogEntry {
                term: term_one,
                index: first.index,
                payload: b"create table t(id int)".to_vec().into(),
            },
            LogEntry {
                term: term_one,
                index: second.index,
                payload: b"insert into t values (1)".to_vec().into(),
            },
        ];

        append_batches_sent += 1;
        assert!(
            apply_append_request(&mut follower_a, term_one, 0, 0, first_batch.clone(), 0).accepted
        );
        leader.register_follower_ack(first.index, 1);
        follower_acks_recorded += 1;
        leader.register_follower_ack(second.index, 1);
        follower_acks_recorded += 1;
        leader
            .wait_committed(second, std::time::Duration::from_millis(1))
            .unwrap();

        heartbeat_batches_sent += 1;
        assert!(
            apply_append_request(
                &mut follower_a,
                term_one,
                second.index,
                term_one,
                vec![],
                leader.commit_index(),
            )
            .accepted
        );
        apply_committed_entries(&mut follower_a, &mut state_a).unwrap();

        append_batches_sent += 1;
        assert!(
            apply_append_request(
                &mut follower_b,
                term_one,
                0,
                0,
                first_batch,
                leader.commit_index(),
            )
            .accepted
        );
        apply_committed_entries(&mut follower_b, &mut state_b).unwrap();

        assert_eq!(state_a.values, state_b.values);
        assert_eq!(
            state_b.values.last().map(String::as_str),
            Some("insert into t values (1)")
        );
        assert!(follower_b.progress().is_caught_up());
        assert_eq!(follower_b.status_snapshot().live.role, Role::Follower);

        let vote_request = follower_a.start_candidate_election(1);
        let mut votes_granted = 1;
        let old_leader_vote = leader.request_vote_from_candidate(&vote_request);
        if old_leader_vote.granted {
            votes_granted += 1;
        }
        let follower_b_vote = follower_b.request_vote_from_candidate(&vote_request);
        if follower_b_vote.granted {
            votes_granted += 1;
        }
        let election_quorum = follower_a.quorum_size();
        let election_passed = votes_granted >= election_quorum;
        if election_passed {
            follower_a.become_leader(vote_request.candidate_term);
        }
        assert!(election_passed);
        let old_leader_rejected_after_failover = matches!(
            leader.propose(b"blocked after failover".to_vec().into()),
            Err(EngineError::NotLeader)
        );
        assert!(old_leader_rejected_after_failover);

        let third = follower_a
            .propose(b"insert into t values (2)".to_vec().into())
            .unwrap();
        let term_two = follower_a.current_term();
        append_batches_sent += 1;
        assert!(
            apply_append_request(
                &mut follower_b,
                term_two,
                second.index,
                term_one,
                vec![LogEntry {
                    term: term_two,
                    index: third.index,
                    payload: b"insert into t values (2)".to_vec().into(),
                }],
                follower_a.commit_index(),
            )
            .accepted
        );
        follower_a.register_follower_ack(third.index, 2);
        follower_acks_recorded += 1;
        follower_a
            .wait_committed(third, std::time::Duration::from_millis(1))
            .unwrap();
        heartbeat_batches_sent += 1;
        assert!(
            apply_append_request(
                &mut follower_b,
                term_two,
                third.index,
                term_two,
                vec![],
                follower_a.commit_index(),
            )
            .accepted
        );
        apply_committed_entries(&mut follower_b, &mut state_b).unwrap();

        assert_eq!(
            state_b.values.last().map(String::as_str),
            Some("insert into t values (2)")
        );
        assert!(follower_b.progress().is_caught_up());
        assert_eq!(follower_a.status_snapshot().live.role, Role::Leader);

        let smoke = OperationalClusterSmokeReport {
            promoted_leader_term: follower_a.current_term(),
            promoted_leader_commit_index: follower_a.commit_index(),
            follower_commit_index: follower_b.commit_index(),
            follower_applied_index: follower_b.applied_index(),
            follower_caught_up: follower_b.progress().is_caught_up(),
            follower_read_after_apply: state_b.values,
            old_leader_rejected_after_failover,
            promoted_node_role: follower_a.role(),
        };
        assert!(smoke.readiness_passed());
        let report = OperationalDeploymentPreflightReport {
            smoke,
            transport: OperationalTransportSmokeReport {
                transport_scope: "single_request_tcp_append_entries",
                append_batches_sent,
                heartbeat_batches_sent,
                follower_acks_recorded,
            },
            election: OperationalElectionSmokeReport {
                election_scope: "deterministic_request_vote",
                candidate_id: vote_request.candidate_id,
                elected_term: vote_request.candidate_term,
                votes_granted,
                quorum: election_quorum,
                elected: election_passed,
            },
            package: OperationalPackageSmokeReport {
                package_scope: "local_cargo_example_binary",
                entrypoint: "crates/replication/examples/operational_cluster_smoke.rs",
                smoke_script: "scripts/run_replication_cluster_smoke.sh",
                packaged_script: "scripts/run_replication_packaged_smoke.sh",
                reproducible: true,
            },
            network_transport_implemented: true,
            automatic_election_implemented: true,
            packaged_deployment_implemented: true,
        };
        assert!(report.readiness_passed());
        assert_eq!(
            report.to_operator_lines(),
            vec![
                "operational_replication_smoke=passed".to_string(),
                "promoted_leader_term=2 promoted_leader_commit=3 follower_commit=3 follower_applied=3 follower_caught_up=true".to_string(),
                "follower_read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)".to_string(),
                "failover_admission_gate=old_leader_not_leader promoted_node_role=Leader".to_string(),
                "deployment_transport=single_request_tcp_append_entries append_batches_sent=3 heartbeat_batches_sent=2 follower_acks_recorded=3".to_string(),
                "deployment_election=deterministic_request_vote candidate_id=1 elected_term=2 votes_granted=3 quorum=2 elected=true".to_string(),
                "deployment_package=local_cargo_example_binary entrypoint=crates/replication/examples/operational_cluster_smoke.rs smoke_script=scripts/run_replication_cluster_smoke.sh packaged_script=scripts/run_replication_packaged_smoke.sh reproducible=true".to_string(),
                "operational_deployment_preflight=passed".to_string(),
                "deployment_scope=packaged_local_three_node_raft_smoke".to_string(),
                "deployment_gap_network_transport=implemented".to_string(),
                "deployment_gap_automatic_election=implemented".to_string(),
                "deployment_gap_packaged_deployment=implemented".to_string(),
            ]
        );
    }

    #[test]
    fn operational_replication_systemd_unit_contract_matches_follower_service() {
        let unit = include_str!(
            "../../../systemd/replication-follower/gpu-db-replication-follower@.service"
        );
        let follower_a =
            include_str!("../../../systemd/replication-follower/replication-follower@2.env");
        let follower_b =
            include_str!("../../../systemd/replication-follower/replication-follower@3.env");

        assert!(unit.contains("EnvironmentFile=-/etc/gpu-db/replication-follower@%i.env"));
        assert!(unit.contains(
            "ExecStart=/usr/local/bin/operational_service_smoke --follower-service --id ${GPU_DB_REPLICATION_FOLLOWER_ID} --expected-requests ${GPU_DB_REPLICATION_EXPECTED_REQUESTS} --listen ${GPU_DB_REPLICATION_LISTEN_ADDR}"
        ));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("NoNewPrivileges=true"));
        assert!(unit.contains("ProtectSystem=strict"));

        for env_file in [follower_a, follower_b] {
            assert!(env_file.contains("GPU_DB_REPLICATION_FOLLOWER_ID="));
            assert!(env_file.contains("GPU_DB_REPLICATION_EXPECTED_REQUESTS=4"));
            assert!(env_file.contains("GPU_DB_REPLICATION_LISTEN_ADDR=0.0.0.0:55432"));
        }
    }

    #[test]
    fn operational_replication_kubernetes_manifest_contract_matches_follower_service() {
        let manifest = include_str!("../../../k8s/replication-service/follower-services.yml");

        for follower_id in ["2", "3"] {
            assert!(manifest.contains(&format!("name: gpu-db-replication-follower-{follower_id}")));
            assert!(manifest.contains(&format!(
                "gpu-db.openclaw.dev/follower-id: \"{follower_id}\""
            )));
            assert!(manifest.contains(&format!(
                "- name: GPU_DB_REPLICATION_FOLLOWER_ID\n              value: \"{follower_id}\""
            )));
        }
        assert!(manifest.contains("kind: Deployment"));
        assert!(manifest.contains("kind: Service"));
        assert!(manifest.contains("image: gpu-db-replication-service:local"));
        assert!(manifest.contains("imagePullPolicy: IfNotPresent"));
        assert!(manifest.contains("- --follower-service"));
        assert!(manifest.contains("- --id"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_FOLLOWER_ID)"));
        assert!(manifest.contains("- --expected-requests"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_EXPECTED_REQUESTS)"));
        assert!(manifest.contains("- --listen"));
        assert!(manifest.contains("- $(GPU_DB_REPLICATION_LISTEN_ADDR)"));
        assert!(manifest
            .contains("- name: GPU_DB_REPLICATION_EXPECTED_REQUESTS\n              value: \"4\""));
        assert!(manifest.contains(
            "- name: GPU_DB_REPLICATION_LISTEN_ADDR\n              value: 0.0.0.0:55432"
        ));
        assert!(manifest.contains("containerPort: 55432"));
        assert!(manifest.contains("targetPort: append"));
    }

    #[test]
    fn request_vote_elects_up_to_date_candidate_and_rejects_stale_log() {
        let mut leader = RaftReplicator::new(3);
        let mut up_to_date = RaftReplicator::new(3);
        let mut stale = RaftReplicator::new(3);
        let mut stale_voter = RaftReplicator::new(3);
        leader.become_leader(1);
        let first = leader.propose(vec![1].into()).unwrap();
        let entry = LogEntry {
            term: leader.current_term(),
            index: first.index,
            payload: vec![1].into(),
        };
        assert!(
            AppendEntriesRequest {
                leader_term: leader.current_term(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![entry],
                leader_commit: leader.commit_index(),
            }
            .apply_to(&mut up_to_date)
            .accepted
        );

        let request = up_to_date.start_candidate_election(7);
        let vote = leader.request_vote_from_candidate(&request);
        assert!(vote.granted);
        assert_eq!(vote.voter_term, request.candidate_term);

        assert!(
            AppendEntriesRequest {
                leader_term: 1,
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    index: 1,
                    payload: vec![1].into(),
                }],
                leader_commit: 0,
            }
            .apply_to(&mut stale_voter)
            .accepted
        );
        let stale_request = stale.start_candidate_election(8);
        let stale_vote = stale_voter.request_vote_from_candidate(&stale_request);
        assert!(!stale_vote.granted);
        assert_eq!(
            stale_vote.error.as_deref(),
            Some("candidate log is behind voter log at index 1 term 1")
        );
    }

    #[test]
    fn append_entries_transport_request_reports_follower_response() {
        let mut follower = RaftReplicator::new(3);
        let accepted = AppendEntriesRequest {
            leader_term: 2,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 2,
                index: 1,
                payload: b"replicated".to_vec().into(),
            }],
            leader_commit: 1,
        }
        .apply_to(&mut follower);

        assert!(accepted.accepted);
        assert_eq!(accepted.follower_term, 2);
        assert_eq!(accepted.follower_commit_index, 1);
        assert_eq!(accepted.follower_applied_index, 0);
        assert_eq!(accepted.error, None);

        let rejected = AppendEntriesRequest {
            leader_term: 1,
            prev_log_index: 1,
            prev_log_term: 2,
            entries: vec![],
            leader_commit: 1,
        }
        .apply_to(&mut follower);

        assert!(!rejected.accepted);
        assert_eq!(rejected.follower_term, 2);
        assert_eq!(rejected.follower_commit_index, 1);
        assert_eq!(rejected.follower_applied_index, 0);
        assert_eq!(
            rejected.error.as_deref(),
            Some("proposal failed: stale leader term 1 (local term 2)")
        );
    }

    #[test]
    fn append_entries_transport_frame_round_trips_request_and_response() {
        let request = AppendEntriesRequest {
            leader_term: 7,
            prev_log_index: 3,
            prev_log_term: 6,
            entries: vec![
                LogEntry {
                    term: 7,
                    index: 4,
                    payload: b"set a=1".to_vec().into(),
                },
                LogEntry {
                    term: 7,
                    index: 5,
                    payload: b"set b=2".to_vec().into(),
                },
            ],
            leader_commit: 5,
        };
        let decoded = AppendEntriesRequest::decode_frame(&request.encode_frame()).unwrap();
        assert_eq!(decoded, request);

        let response = AppendEntriesResponse {
            accepted: false,
            follower_term: 8,
            follower_commit_index: 4,
            follower_applied_index: 3,
            error: Some("stale leader term".to_string()),
        };
        let decoded = AppendEntriesResponse::decode_frame(&response.encode_frame()).unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn append_entries_transport_frame_rejects_truncated_payload() {
        let request = AppendEntriesRequest {
            leader_term: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                payload: b"payload".to_vec().into(),
            }],
            leader_commit: 0,
        };
        let mut frame = request.encode_frame();
        frame.pop();

        let err = AppendEntriesRequest::decode_frame(&frame).unwrap_err();
        assert!(err
            .to_string()
            .contains("append entries frame decode failed"));
    }

    #[test]
    fn append_entries_transport_tcp_loopback_round_trips_frame() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let mut follower = RaftReplicator::new(3);
            let response = serve_append_entries_once(
                &listener,
                &mut follower,
                std::time::Duration::from_millis(250),
            )
            .unwrap();
            assert!(response.accepted);
            follower.commit_index()
        });

        let request = AppendEntriesRequest {
            leader_term: 3,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 3,
                index: 1,
                payload: b"replicated over loopback".to_vec().into(),
            }],
            leader_commit: 1,
        };
        let response =
            send_append_entries_once(addr, &request, std::time::Duration::from_millis(250))
                .unwrap();
        assert!(response.accepted);
        assert_eq!(response.follower_term, 3);
        assert_eq!(response.follower_commit_index, 1);
        assert_eq!(server.join().unwrap(), 1);
    }

    #[test]
    fn commit_index_monotonic() {
        let mut r = LocalReplicator::leader();
        let a = r.propose(vec![1].into()).unwrap();
        let b = r.propose(vec![2].into()).unwrap();

        assert!(b.index > a.index);
        assert_eq!(r.commit_index(), b.index);
        assert!(r.applied_index() <= r.commit_index());
    }

    #[test]
    fn follower_rejects_writes() {
        let mut r = LocalReplicator::leader();
        r.become_follower(2);
        let err = r.propose(vec![1].into()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(r.current_term(), 2);
    }

    #[test]
    fn leader_accepts_after_promotion() {
        let mut r = LocalReplicator::leader();
        r.become_follower(2);
        r.become_leader(3);
        let tok = r.propose(vec![42].into()).unwrap();
        assert_eq!(tok.index, 1);
        assert_eq!(r.current_term(), 3);
    }

    #[test]
    fn candidate_rejects_writes() {
        let mut r = LocalReplicator::leader();
        r.become_candidate(2);

        let err = r.propose(vec![1].into()).unwrap_err();

        assert!(matches!(err, EngineError::NotLeader));
        assert_eq!(r.current_term(), 2);
        assert_eq!(r.role(), Role::Candidate);
    }

    #[test]
    fn rollback_unapplied_removes_tail_and_resets_indices() {
        let mut r = LocalReplicator::leader();
        let _ = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();
        assert_eq!(r.commit_index(), t2.index);

        r.rollback_unapplied_from(t2.index);

        assert_eq!(r.commit_index(), 1);
        let t3 = r.propose(vec![3].into()).unwrap();
        assert_eq!(t3.index, 2);
    }

    #[test]
    fn local_progress_snapshot_tracks_rollback_of_unapplied_tail() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();
        r.mark_applied(t1.index);

        let before = r.progress();
        assert_eq!(before.commit_index, t2.index);
        assert_eq!(before.applied_index, t1.index);
        assert_eq!(before.next_index, t2.index + 1);
        assert_eq!(before.committed_but_unapplied_count, 1);
        assert_eq!(before.uncommitted_entry_count, 0);

        r.rollback_unapplied_from(t2.index);

        let after = r.progress();
        assert_eq!(after.commit_index, t1.index);
        assert_eq!(after.applied_index, t1.index);
        assert_eq!(after.next_index, t1.index + 1);
        assert_eq!(after.committed_but_unapplied_count, 0);
        assert!(!after.has_committed_entries_pending_apply);
        assert_eq!(after.uncommitted_entry_count, 0);
        assert!(!after.has_uncommitted_entries);
        assert_eq!(after.apply_gap(), 0);
        assert!(after.is_caught_up());
        after.validate().unwrap();
    }

    #[test]
    fn snapshot_meta_tracks_applied_index() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);

        let meta = r.export_snapshot_meta();

        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, r.current_term());
        assert_eq!(meta.snapshot_id, 1);
    }

    #[test]
    fn snapshot_meta_preserves_last_applied_term_across_term_bumps() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);

        r.become_follower(5);
        let meta = r.snapshot_meta();

        assert_eq!(r.current_term(), 5);
        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, 1);
    }

    /// Locks the contiguity assumption the O(1) `drain_committed_from`/`entry_at` rewrite relies on
    /// (audit follow-up): after prefix compaction via `install_snapshot`, `entries[0].index != 1`, so
    /// the relative position math `index - entries[0].index` is the ONLY thing that keeps drain/apply
    /// selecting the right WAL entries. A future mutator that broke contiguity would silently corrupt
    /// the applied set; this test would catch it.
    #[test]
    fn local_drain_and_entry_at_after_prefix_compaction() {
        let mut r = LocalReplicator::leader();
        for i in 1..=8u64 {
            assert_eq!(r.propose(vec![i as u8].into()).unwrap().index, i);
        }
        // Compact the prefix: apply through 5, snapshot, install -> retained entries become 6,7,8.
        r.mark_applied(5);
        let meta = r.export_snapshot_meta();
        assert_eq!(meta.last_included_index, 5);
        r.install_snapshot(meta);
        // Append more committed entries (indices 9, 10) on top of the compacted log.
        for i in 9..=10u64 {
            assert_eq!(r.propose(vec![i as u8].into()).unwrap().index, i);
        }
        // entries[0].index is now 6 (not 1). drain over EVERY start must equal the predicate it
        // replaced: `index > start && index <= commit_index`, over the retained set [6, commit_index].
        let commit_index = r.commit_index();
        assert_eq!(commit_index, 10);
        let retained_first = 6u64;
        for start in 0..=12u64 {
            let got: Vec<u64> = r.drain_committed_from(start).map(|e| e.index).collect();
            let expected: Vec<u64> = (retained_first..=commit_index)
                .filter(|&idx| idx > start && idx <= commit_index)
                .collect();
            assert_eq!(
                got, expected,
                "drain_committed_from({start}) diverged after prefix compaction"
            );
        }
        // entry_at maps via the relative position; compacted/out-of-range -> None (never a panic or
        // a wrong-position hit).
        assert_eq!(r.entry_at(6).map(|e| e.index), Some(6));
        assert_eq!(r.entry_at(10).map(|e| e.index), Some(10));
        assert_eq!(r.entry_at(5), None, "index 5 was prefix-compacted away");
        assert_eq!(r.entry_at(11), None, "index 11 is past the tail");
        assert_eq!(r.entry_at(0), None);
        // mark_applied past the compaction boundary reads the correct entry's term (no panic).
        r.mark_applied(9);
        assert_eq!(r.applied_index(), 9);
    }

    #[test]
    fn install_snapshot_advances_log_watermarks() {
        let mut r = LocalReplicator::leader();
        let _ = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 2,
            snapshot_id: 9,
        });

        assert_eq!(r.commit_index(), t2.index);
        assert_eq!(r.applied_index(), t2.index);
        assert_eq!(r.current_term(), 2);
        assert_eq!(r.snapshot_meta().snapshot_id, 9);

        let t3 = r.propose(vec![3].into()).unwrap();
        assert_eq!(t3.index, t2.index + 1);
    }

    #[test]
    fn install_older_snapshot_is_a_progress_no_op() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);
        let baseline = r.progress();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index.saturating_sub(1),
            last_included_term: 1,
            snapshot_id: baseline.snapshot.snapshot_id + 10,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.snapshot_meta(), baseline.snapshot);
    }

    #[test]
    fn local_install_snapshot_preserves_next_index_from_uncompacted_tail() {
        let mut r = LocalReplicator::leader();
        let _t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index - 1,
            last_included_term: 1,
            snapshot_id: 11,
        });

        let next = r.propose(vec![3].into()).unwrap();
        assert_eq!(next.index, t2.index + 1);
    }

    #[test]
    fn local_install_snapshot_updates_snapshot_id_for_same_frontier_same_term() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 1,
            snapshot_id: 11,
        });

        assert_eq!(r.snapshot_meta().snapshot_id, 11);
        assert_eq!(r.snapshot_meta().last_included_index, t1.index);
        assert_eq!(r.progress().snapshot.snapshot_id, 11);
    }

    #[test]
    fn local_install_snapshot_advancing_frontier_replaces_snapshot_identity_exactly() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 1,
            snapshot_id: 11,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index + 1,
            last_included_term: 2,
            snapshot_id: 4,
        });

        let snapshot = r.snapshot_meta();
        assert_eq!(snapshot.last_included_index, t1.index + 1);
        assert_eq!(snapshot.last_included_term, 2);
        assert_eq!(snapshot.snapshot_id, 4);
        assert_eq!(r.progress().snapshot, snapshot);
    }

    #[test]
    fn local_install_snapshot_with_higher_index_lower_term_is_a_progress_no_op() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        r.mark_applied(t1.index);
        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index + 1,
            last_included_term: 3,
            snapshot_id: 11,
        });
        let baseline = r.progress();

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index + 2,
            last_included_term: 2,
            snapshot_id: 19,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.snapshot_meta().snapshot_id, 11);
    }

    #[test]
    fn mark_applied_does_not_exceed_commit_index() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();

        r.mark_applied(t1.index + 10);

        assert_eq!(r.applied_index(), t1.index);
    }

    #[test]
    fn local_progress_snapshot_clamps_apply_frontier_to_commit_boundary() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();

        r.mark_applied(t1.index + 10);

        let progress = r.progress();
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, t1.index);
        assert_eq!(progress.apply_gap(), 0);
        assert!(progress.is_caught_up());
    }

    #[test]
    fn local_replicator_pending_apply_helpers_track_committed_tail() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        let _t2 = r.propose(vec![2].into()).unwrap();

        assert_eq!(r.retained_entry_count(), 2);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 2);

        r.mark_applied(t1.index);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);

        r.mark_applied(r.commit_index());
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);
    }

    #[test]
    fn local_progress_snapshot_matches_commit_and_apply_state() {
        let mut r = LocalReplicator::leader();
        let t1 = r.propose(vec![1].into()).unwrap();
        let _t2 = r.propose(vec![2].into()).unwrap();
        r.mark_applied(t1.index);

        let progress = r.progress();

        assert_eq!(progress.role, Role::Leader);
        assert_eq!(progress.commit_index, 2);
        assert_eq!(progress.applied_index, 1);
        assert_eq!(progress.next_index, 3);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(!progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_replicator_rejects_proposal_when_not_leader() {
        let mut r = RaftReplicator::new(3);
        let err = r.propose(vec![1].into()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));
    }

    #[test]
    fn raft_candidate_rejects_proposal_and_drops_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        let _uncommitted = r.propose(vec![2].into()).unwrap();

        r.become_candidate(2);
        let err = r.propose(vec![3].into()).unwrap_err();
        assert!(matches!(err, EngineError::NotLeader));

        r.become_leader(3);
        let tok = r.propose(vec![4].into()).unwrap();
        assert_eq!(tok.index, t1.index + 1);
    }

    #[test]
    fn raft_replicator_commits_after_quorum_acks() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let t1 = r.propose(vec![1].into()).unwrap();
        assert_eq!(r.commit_index(), 0, "self-ack is not quorum for 3 voters");

        r.register_follower_ack(t1.index, 1);

        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.quorum_size(), 2);
        assert_eq!(r.voter_count(), 3);
    }

    #[test]
    fn raft_replicator_commit_index_advances_in_order() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10].into()).unwrap();
        let t2 = r.propose(vec![20].into()).unwrap();

        r.register_follower_ack(t2.index, 2);
        assert_eq!(r.commit_index(), 0, "cannot skip index 1");

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t2.index);
    }

    #[test]
    fn raft_progress_snapshot_tracks_quorum_ack_commit_promotion() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10].into()).unwrap();
        let t2 = r.propose(vec![20].into()).unwrap();

        let before = r.progress();
        assert_eq!(before.commit_index, 0);
        assert_eq!(before.applied_index, 0);
        assert_eq!(before.uncommitted_entry_count, 2);
        assert!(before.has_uncommitted_entries);
        assert_eq!(before.committed_but_unapplied_count, 0);
        assert!(!before.has_committed_entries_pending_apply);

        r.register_follower_ack(t2.index, 2);
        let still_blocked = r.progress();
        assert_eq!(still_blocked.commit_index, 0);
        assert_eq!(still_blocked.uncommitted_entry_count, 2);
        assert!(still_blocked.has_uncommitted_entries);

        r.register_follower_ack(t1.index, 1);
        let after = r.progress();
        assert_eq!(after.commit_index, t2.index);
        assert_eq!(after.applied_index, 0);
        assert_eq!(after.uncommitted_entry_count, 0);
        assert!(!after.has_uncommitted_entries);
        assert_eq!(after.committed_but_unapplied_count, t2.index as usize);
        assert!(after.has_committed_entries_pending_apply);
        assert_eq!(after.apply_gap(), t2.index as usize);
        assert!(!after.is_caught_up());
        after.validate().unwrap();
    }

    #[test]
    fn raft_replicator_entry_state_helpers_track_committed_and_uncommitted_work() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10].into()).unwrap();
        let t2 = r.propose(vec![20].into()).unwrap();

        assert_eq!(r.retained_entry_count(), 2);
        assert!(r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 2);
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);

        r.register_follower_ack(t1.index, 1);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);
        assert!(r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 1);

        r.mark_applied(t1.index);
        assert!(!r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 0);

        r.register_follower_ack(t2.index, 1);
        assert!(r.has_committed_entries_pending_apply());
        assert_eq!(r.committed_but_unapplied_count(), 1);
        assert!(!r.has_uncommitted_entries());
        assert_eq!(r.uncommitted_entry_count(), 0);
    }

    #[test]
    fn raft_follower_lagging_apply_delay_exposes_pending_apply_until_catch_up() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(3);
        let t1 = leader.propose(vec![10].into()).unwrap();
        let t2 = leader.propose(vec![20].into()).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);

        let mut follower = RaftReplicator::new(3);
        follower.become_follower(3);
        follower
            .append_entries_from_leader(
                3,
                0,
                0,
                vec![
                    LogEntry {
                        term: 3,
                        index: t1.index,
                        payload: vec![10].into(),
                    },
                    LogEntry {
                        term: 3,
                        index: t2.index,
                        payload: vec![20].into(),
                    },
                ],
                t2.index,
            )
            .unwrap();

        assert_eq!(follower.commit_index(), t2.index);
        assert_eq!(follower.applied_index(), 0);
        assert!(follower.has_committed_entries_pending_apply());
        assert_eq!(follower.committed_but_unapplied_count(), 2);

        follower.mark_applied(t1.index);
        assert_eq!(follower.committed_but_unapplied_count(), 1);
        assert!(follower.has_committed_entries_pending_apply());

        follower.mark_applied(t2.index);
        assert_eq!(follower.applied_index(), t2.index);
        assert!(!follower.has_committed_entries_pending_apply());
        assert_eq!(follower.committed_but_unapplied_count(), 0);
    }

    #[test]
    fn raft_progress_snapshot_clamps_apply_frontier_to_commit_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);
        let t1 = r.propose(vec![10].into()).unwrap();
        let t2 = r.propose(vec![20].into()).unwrap();
        r.register_follower_ack(t1.index, 1);

        r.mark_applied(t2.index + 10);

        let progress = r.progress();
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, t1.index);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 0);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_recovery_state_resumes_pending_apply_without_rewinding_commit() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(4);
        let t1 = leader.propose(vec![1].into()).unwrap();
        let t2 = leader.propose(vec![2].into()).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);
        leader.mark_applied(t1.index);
        let snapshot = leader.export_snapshot_meta();

        let resumed = RaftReplicator::resume_as_follower(3, leader.recovery_state()).unwrap();

        assert_eq!(resumed.role(), Role::Follower);
        assert_eq!(resumed.current_term(), 4);
        assert_eq!(resumed.commit_index(), t2.index);
        assert_eq!(resumed.applied_index(), t1.index);
        assert_eq!(resumed.snapshot_meta(), snapshot);
        assert!(resumed.has_committed_entries_pending_apply());
        assert_eq!(resumed.committed_but_unapplied_count(), 1);
        assert_eq!(resumed.next_index, t2.index + 1);
    }

    #[test]
    fn raft_progress_snapshot_tracks_uncommitted_and_pending_apply_work() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![10].into()).unwrap();
        let _t2 = r.propose(vec![20].into()).unwrap();
        r.register_follower_ack(t1.index, 1);

        let progress = r.progress();

        assert_eq!(progress.role, Role::Leader);
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, 0);
        assert_eq!(progress.next_index, 3);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_progress_snapshot_stays_consistent_across_snapshot_boundary_append() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        r.append_entries_from_leader(
            4,
            5,
            3,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap();

        let progress = r.progress();

        assert_eq!(progress.snapshot.last_included_index, 5);
        assert_eq!(progress.commit_index, 6);
        assert_eq!(progress.applied_index, 5);
        assert_eq!(progress.next_index, 7);
        assert_eq!(progress.committed_but_unapplied_count, 1);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(!progress.has_uncommitted_entries);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn replication_progress_reports_caught_up_only_when_apply_and_tail_are_clear() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);

        let before_apply = r.progress();
        assert_eq!(before_apply.apply_gap(), 1);
        assert!(!before_apply.is_caught_up());

        r.mark_applied(t1.index);
        let after_apply = r.progress();
        assert_eq!(after_apply.apply_gap(), 0);
        assert!(after_apply.is_caught_up());
    }

    #[test]
    fn recovery_state_helpers_report_commit_apply_boundaries() {
        let state = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1].into(),
            }],
            applied_index: 3,
        };

        assert_eq!(state.commit_index(), 4);
        assert_eq!(state.next_index(), 5);
        assert!(state.has_committed_entries_pending_apply());
        assert_eq!(state.committed_but_unapplied_count(), 1);
        assert_eq!(state.apply_gap(), 1);
        assert!(!state.is_caught_up());
        state.validate().unwrap();

        let progress = state.progress_as_follower().unwrap();
        assert_eq!(progress.role, Role::Follower);
        assert_eq!(progress.term, 4);
        assert_eq!(progress.commit_index, 4);
        assert_eq!(progress.applied_index, 3);
        assert_eq!(progress.next_index, 5);
        assert_eq!(progress.apply_gap(), 1);
        assert!(!progress.is_caught_up());
    }

    #[test]
    fn recovery_state_reports_caught_up_when_apply_reaches_commit_boundary() {
        let state = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1].into(),
            }],
            applied_index: 4,
        };

        assert_eq!(state.apply_gap(), 0);
        assert!(state.is_caught_up());
        assert!(state.progress_as_follower().unwrap().is_caught_up());
    }

    #[test]
    fn snapshot_only_recovery_state_maps_to_caught_up_progress() {
        let state = RecoveryState {
            term: 6,
            snapshot: SnapshotMeta {
                last_included_index: 9,
                last_included_term: 5,
                snapshot_id: 11,
            },
            committed_entries: vec![],
            applied_index: 9,
        };

        assert_eq!(state.commit_index(), 9);
        assert_eq!(state.next_index(), 10);
        assert_eq!(state.committed_but_unapplied_count(), 0);
        assert_eq!(state.apply_gap(), 0);
        assert!(state.is_caught_up());

        let progress = state.progress_as_follower().unwrap();
        assert_eq!(progress.role, Role::Follower);
        assert_eq!(progress.term, 6);
        assert_eq!(progress.commit_index, 9);
        assert_eq!(progress.applied_index, 9);
        assert_eq!(progress.next_index, 10);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(progress.is_caught_up());
    }

    #[test]
    fn recovery_state_validation_rejects_applied_index_past_commit_boundary() {
        let err = RecoveryState {
            term: 4,
            snapshot: SnapshotMeta {
                last_included_index: 3,
                last_included_term: 4,
                snapshot_id: 7,
            },
            committed_entries: vec![LogEntry {
                term: 4,
                index: 4,
                payload: vec![1].into(),
            }],
            applied_index: 5,
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            err,
            RecoveryInvariantError::AppliedExceedsCommit {
                applied_index: 5,
                commit_index: 4,
            }
        );
    }

    #[test]
    fn resumed_follower_progress_matches_recovery_projection() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(4);
        let t1 = leader.propose(vec![1].into()).unwrap();
        let t2 = leader.propose(vec![2].into()).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.register_follower_ack(t2.index, 1);
        leader.mark_applied(t1.index);

        let recovery = leader.recovery_state();
        let projected = recovery.progress_as_follower().unwrap();
        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

        assert_eq!(resumed.progress(), projected);
    }

    #[test]
    fn resumed_follower_from_snapshot_only_recovery_matches_projection() {
        let recovery = RecoveryState {
            term: 6,
            snapshot: SnapshotMeta {
                last_included_index: 9,
                last_included_term: 5,
                snapshot_id: 11,
            },
            committed_entries: vec![],
            applied_index: 9,
        };

        let projected = recovery.progress_as_follower().unwrap();
        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

        assert_eq!(resumed.progress(), projected);
        assert!(resumed.progress().is_caught_up());
    }

    #[test]
    fn recovery_progress_handles_snapshot_boundary_committed_tail() {
        let recovery = RecoveryState {
            term: 7,
            snapshot: SnapshotMeta {
                last_included_index: 10,
                last_included_term: 6,
                snapshot_id: 21,
            },
            committed_entries: vec![
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![11].into(),
                },
                LogEntry {
                    term: 7,
                    index: 12,
                    payload: vec![12].into(),
                },
            ],
            applied_index: 10,
        };

        let progress = recovery.progress_as_follower().unwrap();

        assert_eq!(progress.commit_index, 12);
        assert_eq!(progress.applied_index, 10);
        assert_eq!(progress.next_index, 13);
        assert_eq!(progress.apply_gap(), 2);
        assert_eq!(progress.committed_but_unapplied_count, 2);
        assert!(progress.has_committed_entries_pending_apply);
        assert!(!progress.has_uncommitted_entries);
    }

    #[test]
    fn resumed_follower_from_snapshot_boundary_tail_matches_projection() {
        let recovery = RecoveryState {
            term: 7,
            snapshot: SnapshotMeta {
                last_included_index: 10,
                last_included_term: 6,
                snapshot_id: 21,
            },
            committed_entries: vec![
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![11].into(),
                },
                LogEntry {
                    term: 7,
                    index: 12,
                    payload: vec![12].into(),
                },
            ],
            applied_index: 10,
        };

        let projected = recovery.progress_as_follower().unwrap();
        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();

        assert_eq!(resumed.progress(), projected);
        assert_eq!(resumed.progress().apply_gap(), 2);
        let status = resumed.status_snapshot();
        assert_eq!(status.live, projected);
        assert_eq!(status.durable, projected);
        assert!(status.is_restart_equivalent());
    }

    #[test]
    fn resumed_follower_progress_stays_stable_through_catch_up_and_snapshot_stress() {
        let recovery = RecoveryState {
            term: 7,
            snapshot: SnapshotMeta {
                last_included_index: 10,
                last_included_term: 6,
                snapshot_id: 21,
            },
            committed_entries: vec![
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![11].into(),
                },
                LogEntry {
                    term: 7,
                    index: 12,
                    payload: vec![12].into(),
                },
            ],
            applied_index: 10,
        };

        let mut resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();
        let baseline = resumed.progress();
        let baseline_recovery = resumed.recovery_progress();
        let baseline_status = resumed.status_snapshot();
        assert_eq!(baseline.commit_index, 12);
        assert_eq!(baseline.applied_index, 10);
        assert_eq!(baseline.next_index, 13);
        assert_eq!(baseline.apply_gap(), 2);
        assert!(!baseline.is_caught_up());
        assert_eq!(baseline_recovery, baseline);
        assert_eq!(baseline_status.live, baseline);
        assert_eq!(baseline_status.durable, baseline_recovery);
        assert!(baseline_status.is_restart_equivalent());
        assert!(resumed.recovery_progress_gap().is_restart_equivalent());

        let err = resumed
            .append_entries_from_leader(
                7,
                12,
                7,
                vec![LogEntry {
                    term: 7,
                    index: 14,
                    payload: vec![14].into(),
                }],
                14,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(resumed.progress(), baseline);
        assert_eq!(resumed.recovery_progress(), baseline_recovery);
        assert_eq!(resumed.status_snapshot(), baseline_status);
        assert!(resumed.recovery_progress_gap().is_restart_equivalent());

        resumed
            .append_entries_from_leader(
                7,
                12,
                7,
                vec![LogEntry {
                    term: 7,
                    index: 13,
                    payload: vec![13].into(),
                }],
                12,
            )
            .unwrap();
        let after_append = resumed.progress();
        let after_append_recovery = resumed.recovery_progress();
        assert_eq!(after_append.commit_index, 12);
        assert_eq!(after_append.applied_index, 10);
        assert_eq!(after_append.next_index, 14);
        assert_eq!(after_append.uncommitted_entry_count, 1);
        assert!(after_append.has_uncommitted_entries);
        assert_eq!(after_append.apply_gap(), 2);
        assert!(!after_append.is_caught_up());
        after_append.validate().unwrap();
        assert_eq!(after_append_recovery.commit_index, 12);
        assert_eq!(after_append_recovery.applied_index, 10);
        assert_eq!(after_append_recovery.next_index, 13);
        assert_eq!(after_append_recovery.uncommitted_entry_count, 0);
        assert!(!after_append_recovery.has_uncommitted_entries);
        assert_eq!(after_append_recovery.apply_gap(), 2);
        assert!(!after_append_recovery.is_caught_up());
        let after_append_status = resumed.status_snapshot();
        assert_eq!(after_append_status.live, after_append);
        assert_eq!(after_append_status.durable, after_append_recovery);
        assert!(after_append_status.has_speculative_tail());
        assert_eq!(
            resumed.recovery_progress_gap(),
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );

        resumed
            .append_entries_from_leader(7, 13, 7, vec![], 13)
            .unwrap();
        let after_heartbeat = resumed.progress();
        let after_heartbeat_recovery = resumed.recovery_progress();
        assert_eq!(after_heartbeat.commit_index, 13);
        assert_eq!(after_heartbeat.applied_index, 10);
        assert_eq!(after_heartbeat.next_index, 14);
        assert_eq!(after_heartbeat.uncommitted_entry_count, 0);
        assert!(!after_heartbeat.has_uncommitted_entries);
        assert_eq!(after_heartbeat.apply_gap(), 3);
        assert!(!after_heartbeat.is_caught_up());
        after_heartbeat.validate().unwrap();
        assert_eq!(after_heartbeat_recovery, after_heartbeat);
        assert!(resumed.status_snapshot().is_restart_equivalent());
        assert!(resumed.recovery_progress_gap().is_restart_equivalent());

        resumed.install_snapshot(SnapshotMeta {
            last_included_index: 13,
            last_included_term: 7,
            snapshot_id: 22,
        });
        let after_snapshot = resumed.progress();
        let after_snapshot_recovery = resumed.recovery_progress();
        assert_eq!(after_snapshot.snapshot.snapshot_id, 22);
        assert_eq!(after_snapshot.snapshot.last_included_index, 13);
        assert_eq!(after_snapshot.commit_index, 13);
        assert_eq!(after_snapshot.applied_index, 13);
        assert_eq!(after_snapshot.next_index, 14);
        assert_eq!(after_snapshot.apply_gap(), 0);
        assert!(after_snapshot.is_caught_up());
        after_snapshot.validate().unwrap();
        assert_eq!(after_snapshot_recovery, after_snapshot);
        assert!(resumed.status_snapshot().is_restart_equivalent());
        assert!(resumed.recovery_progress_gap().is_restart_equivalent());

        let err = resumed
            .append_entries_from_leader(
                7,
                12,
                7,
                vec![LogEntry {
                    term: 7,
                    index: 13,
                    payload: vec![13].into(),
                }],
                13,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(resumed.progress(), after_snapshot);
        assert_eq!(resumed.recovery_progress(), after_snapshot_recovery);
        assert_eq!(resumed.status_snapshot().durable, after_snapshot_recovery);
        assert!(resumed.recovery_progress_gap().is_restart_equivalent());
    }

    #[test]
    fn replication_progress_validation_rejects_inconsistent_uncommitted_flag() {
        let err = ReplicationProgress {
            role: Role::Follower,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 6,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 1,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 1,
            has_uncommitted_entries: false,
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationProgressInvariantError::UncommittedFlagMismatch {
                has_uncommitted: false,
                uncommitted_entry_count: 1,
            }
        );
    }

    #[test]
    fn replication_status_snapshot_validation_rejects_durable_state_ahead_of_live() {
        let live = ReplicationProgress {
            role: Role::Leader,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 6,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 10,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        let durable = ReplicationProgress {
            next_index: 7,
            ..live.clone()
        };

        let err = ReplicationStatusSnapshot::new(
            live,
            durable,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationStatusInvariantError::DurableAheadOfLive {
                field: "next_index",
                durable: 7,
                live: 6,
            }
        );
    }

    #[test]
    fn replication_status_snapshot_validation_rejects_term_mismatch() {
        let live = ReplicationProgress {
            role: Role::Leader,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 6,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 10,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        let durable = ReplicationProgress {
            role: Role::Follower,
            term: 3,
            ..live.clone()
        };

        let err = ReplicationStatusSnapshot::new(
            live,
            durable,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationStatusInvariantError::TermMismatch {
                durable: 3,
                live: 4,
            }
        );
    }

    #[test]
    fn replication_status_snapshot_validation_rejects_same_frontier_snapshot_identity_drift() {
        let live = ReplicationProgress {
            role: Role::Follower,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 6,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 10,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
        };
        let durable = ReplicationProgress {
            snapshot: SnapshotMeta {
                snapshot_id: 9,
                ..live.snapshot.clone()
            },
            ..live.clone()
        };

        let err = ReplicationStatusSnapshot::new(
            live,
            durable,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationStatusInvariantError::SnapshotIdentityDrift {
                last_included_index: 5,
                last_included_term: 4,
                durable_snapshot_id: 9,
                live_snapshot_id: 10,
            }
        );
    }

    #[test]
    fn replication_status_snapshot_validation_rejects_recovery_gap_mismatch() {
        let live = ReplicationProgress {
            role: Role::Follower,
            term: 4,
            commit_index: 5,
            applied_index: 5,
            next_index: 7,
            snapshot: SnapshotMeta {
                last_included_index: 5,
                last_included_term: 4,
                snapshot_id: 10,
            },
            committed_but_unapplied_count: 0,
            has_committed_entries_pending_apply: false,
            uncommitted_entry_count: 1,
            has_uncommitted_entries: true,
        };
        let durable = ReplicationProgress {
            next_index: 6,
            uncommitted_entry_count: 0,
            has_uncommitted_entries: false,
            ..live.clone()
        };

        let err = ReplicationStatusSnapshot::new(
            live,
            durable,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            ReplicationStatusInvariantError::RecoveryGapMismatch {
                expected: RecoveryProgressGap {
                    commit_index_gap: 0,
                    applied_index_gap: 0,
                    next_index_gap: 1,
                    uncommitted_entry_gap: 1,
                },
                actual: RecoveryProgressGap {
                    commit_index_gap: 0,
                    applied_index_gap: 0,
                    next_index_gap: 0,
                    uncommitted_entry_gap: 0,
                },
            }
        );
    }

    #[test]
    fn raft_recovery_progress_projects_durable_follower_state_from_live_leader() {
        let mut leader = RaftReplicator::new(3);
        leader.become_leader(4);
        let t1 = leader.propose(vec![1].into()).unwrap();
        let _t2 = leader.propose(vec![2].into()).unwrap();
        leader.register_follower_ack(t1.index, 1);
        leader.mark_applied(t1.index);

        let live = leader.progress();
        assert_eq!(live.role, Role::Leader);
        assert_eq!(live.uncommitted_entry_count, 1);
        assert!(live.has_uncommitted_entries);

        let durable = leader.recovery_progress();
        assert_eq!(durable.role, Role::Follower);
        assert_eq!(durable.term, live.term);
        assert_eq!(durable.commit_index, t1.index);
        assert_eq!(durable.applied_index, t1.index);
        assert_eq!(durable.next_index, t1.index + 1);
        assert_eq!(durable.snapshot, live.snapshot);
        assert_eq!(durable.uncommitted_entry_count, 0);
        assert!(!durable.has_uncommitted_entries);
        assert!(durable.is_caught_up());

        let gap = leader.recovery_progress_gap();
        assert_eq!(
            gap,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );
        assert!(gap.has_gap());
        assert!(gap.has_speculative_tail());
        assert!(!gap.is_restart_equivalent());

        let status = leader.status_snapshot();
        assert_eq!(status.live, live);
        assert_eq!(status.durable, durable);
        assert_eq!(status.recovery_gap, gap);
        assert!(status.has_speculative_tail());
    }

    #[test]
    fn local_status_snapshot_is_always_restart_equivalent() {
        let mut local = LocalReplicator::leader();
        let token = local.propose(b"set a=1".to_vec().into()).unwrap();
        local.mark_applied(token.index);

        let status = local.status_snapshot();

        assert_eq!(status.live, local.progress());
        assert_eq!(status.durable, local.progress());
        assert!(status.is_restart_equivalent());
        assert!(!status.has_speculative_tail());
    }

    #[test]
    fn raft_resume_rejects_non_contiguous_recovery_entries() {
        let err = RaftReplicator::resume_as_follower(
            3,
            RecoveryState {
                term: 5,
                snapshot: SnapshotMeta {
                    last_included_index: 3,
                    last_included_term: 4,
                    snapshot_id: 8,
                },
                committed_entries: vec![LogEntry {
                    term: 5,
                    index: 5,
                    payload: vec![9].into(),
                }],
                applied_index: 3,
            },
        )
        .unwrap_err();

        assert!(matches!(err, EngineError::ApplyFailed(_)));
    }

    #[test]
    fn raft_single_node_leader_commits_immediately() {
        let mut r = RaftReplicator::single_node_leader();
        let tok = r.propose(vec![7].into()).unwrap();
        assert_eq!(r.commit_index(), tok.index);
    }

    #[test]
    fn raft_progress_snapshot_tracks_single_node_immediate_commit() {
        let mut r = RaftReplicator::single_node_leader();
        let tok = r.propose(vec![7].into()).unwrap();

        let progress = r.progress();
        assert_eq!(progress.role, Role::Leader);
        assert_eq!(progress.commit_index, tok.index);
        assert_eq!(progress.applied_index, 0);
        assert_eq!(progress.next_index, tok.index + 1);
        assert_eq!(progress.uncommitted_entry_count, 0);
        assert!(!progress.has_uncommitted_entries);
        assert_eq!(progress.committed_but_unapplied_count, tok.index as usize);
        assert!(progress.has_committed_entries_pending_apply);
        assert_eq!(progress.apply_gap(), tok.index as usize);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_follower_acks_are_ignored_when_not_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();

        r.become_follower(2);
        r.register_follower_ack(t1.index, 1);

        assert_eq!(r.commit_index(), 0);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_when_acks_arrive_off_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();

        r.become_follower(2);
        let before = r.progress();
        r.register_follower_ack(t1.index, 1);
        let after = r.progress();

        assert_eq!(after, before);
    }

    #[test]
    fn raft_rejects_ack_for_unknown_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let _ = r.propose(vec![1].into()).unwrap();

        r.register_follower_ack(2, 1);

        assert_eq!(r.commit_index(), 0);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_when_ack_targets_unknown_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let _ = r.propose(vec![1].into()).unwrap();

        let before = r.progress();
        r.register_follower_ack(2, 1);
        let after = r.progress();

        assert_eq!(after, before);
    }

    #[test]
    fn raft_ignores_reserved_self_ack_follower_id() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 0);

        assert_eq!(
            r.commit_index(),
            0,
            "follower id 0 is reserved for the leader self-ack"
        );
    }

    #[test]
    fn raft_progress_snapshot_is_stable_for_reserved_self_ack() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        let before = r.progress();
        r.register_follower_ack(t1.index, 0);
        let after = r.progress();

        assert_eq!(after, before);
    }

    #[test]
    fn raft_duplicate_follower_ack_does_not_count_twice() {
        let mut r = RaftReplicator::new(5);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t1.index, 2);
        assert_eq!(r.commit_index(), t1.index);

        let t2 = r.propose(vec![2].into()).unwrap();
        r.register_follower_ack(t2.index, 1);
        r.register_follower_ack(t2.index, 1);

        assert_eq!(
            r.commit_index(),
            t1.index,
            "same follower should not be able to satisfy quorum twice"
        );

        r.register_follower_ack(t2.index, 2);
        assert_eq!(r.commit_index(), t2.index);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_for_duplicate_follower_ack_until_quorum_changes() {
        let mut r = RaftReplicator::new(5);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t1.index, 2);
        assert_eq!(r.commit_index(), t1.index);

        let t2 = r.propose(vec![2].into()).unwrap();
        r.register_follower_ack(t2.index, 1);
        let before_duplicate = r.progress();

        r.register_follower_ack(t2.index, 1);
        let after_duplicate = r.progress();

        assert_eq!(after_duplicate, before_duplicate);

        r.register_follower_ack(t2.index, 2);
        let after_quorum = r.progress();
        assert_ne!(after_quorum, after_duplicate);
        assert_eq!(after_quorum.commit_index, t2.index);
    }

    #[test]
    fn raft_prunes_ack_tracking_for_committed_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        assert!(r.ack_counts.contains_key(&t1.index));
        assert!(r.ack_counts.contains_key(&t2.index));

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);
        assert!(!r.ack_counts.contains_key(&t1.index));
        assert!(r.ack_counts.contains_key(&t2.index));

        r.register_follower_ack(t2.index, 1);
        assert_eq!(r.commit_index(), t2.index);
        assert!(r.ack_counts.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_ack_tracking_pruning() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        let after_first_commit = r.progress();
        assert_eq!(after_first_commit.commit_index, t1.index);
        assert_eq!(after_first_commit.uncommitted_entry_count, 1);

        r.register_follower_ack(t2.index, 1);
        let after_second_commit = r.progress();
        assert_eq!(after_second_commit.commit_index, t2.index);
        assert_eq!(after_second_commit.uncommitted_entry_count, 0);
        assert!(r.ack_counts.is_empty());
        after_second_commit.validate().unwrap();
    }

    #[test]
    fn raft_role_change_discards_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let _t2_uncommitted = r.propose(vec![2].into()).unwrap();
        assert_eq!(r.commit_index(), t1.index);

        r.become_follower(2);
        r.become_leader(3);

        let t2_new_epoch = r.propose(vec![3].into()).unwrap();
        assert_eq!(t2_new_epoch.index, t1.index + 1);

        r.register_follower_ack(t2_new_epoch.index, 1);
        assert_eq!(r.commit_index(), t2_new_epoch.index);
    }

    #[test]
    fn raft_progress_snapshot_discards_uncommitted_tail_across_role_change() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        let _t2_uncommitted = r.propose(vec![2].into()).unwrap();

        let before = r.progress();
        assert_eq!(before.commit_index, t1.index);
        assert_eq!(before.uncommitted_entry_count, 1);
        assert!(before.has_uncommitted_entries);
        assert!(!before.is_caught_up());
        let before_gap = r.recovery_progress_gap();
        assert_eq!(before_gap.next_index_gap, 1);
        assert_eq!(before_gap.uncommitted_entry_gap, 1);
        assert!(before_gap.has_speculative_tail());

        r.become_follower(2);
        let after_follower = r.progress();
        assert_eq!(after_follower.role, Role::Follower);
        assert_eq!(after_follower.commit_index, t1.index);
        assert_eq!(after_follower.next_index, t1.index + 1);
        assert_eq!(after_follower.uncommitted_entry_count, 0);
        assert!(!after_follower.has_uncommitted_entries);
        assert_eq!(
            r.recovery_progress_gap(),
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 0,
                uncommitted_entry_gap: 0,
            }
        );

        r.become_leader(3);
        let after_leader = r.progress();
        assert_eq!(after_leader.role, Role::Leader);
        assert_eq!(after_leader.commit_index, t1.index);
        assert_eq!(after_leader.next_index, t1.index + 1);
        assert_eq!(after_leader.uncommitted_entry_count, 0);
        assert!(!after_leader.has_uncommitted_entries);
        after_leader.validate().unwrap();
        assert!(r.recovery_progress_gap().is_restart_equivalent());
    }

    #[test]
    fn raft_truncate_uncommitted_from_drops_tail_and_resets_next_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let t2 = r.propose(vec![2].into()).unwrap();
        let t3 = r.propose(vec![3].into()).unwrap();
        assert!(r.ack_counts.contains_key(&t2.index));
        assert!(r.ack_counts.contains_key(&t3.index));

        r.truncate_uncommitted_from(t2.index);

        assert_eq!(r.commit_index(), t1.index);
        assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
        assert!(r.ack_counts.is_empty());

        let replacement = r.propose(vec![9].into()).unwrap();
        assert_eq!(replacement.index, t1.index + 1);
    }

    #[test]
    fn raft_progress_snapshot_tracks_uncommitted_tail_truncation() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        let t2 = r.propose(vec![2].into()).unwrap();
        let t3 = r.propose(vec![3].into()).unwrap();

        let before = r.progress();
        assert_eq!(before.commit_index, t1.index);
        assert_eq!(before.next_index, t3.index + 1);
        assert_eq!(before.uncommitted_entry_count, 2);
        assert!(before.has_uncommitted_entries);
        assert!(!before.is_caught_up());

        r.truncate_uncommitted_from(t2.index);

        let after = r.progress();
        assert_eq!(after.commit_index, t1.index);
        assert_eq!(after.applied_index, 0);
        assert_eq!(after.next_index, t1.index + 1);
        assert_eq!(after.uncommitted_entry_count, 0);
        assert!(!after.has_uncommitted_entries);
        assert_eq!(after.committed_but_unapplied_count, t1.index as usize);
        assert!(after.has_committed_entries_pending_apply);
        assert_eq!(after.apply_gap(), t1.index as usize);
        assert!(!after.is_caught_up());
        after.validate().unwrap();
    }

    #[test]
    fn raft_truncate_uncommitted_from_ignores_committed_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.truncate_uncommitted_from(t1.index);

        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_when_truncation_targets_committed_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);

        let before = r.progress();
        r.truncate_uncommitted_from(t1.index);
        let after = r.progress();

        assert_eq!(after, before);
    }

    #[test]
    fn local_wait_committed_rejects_uncommitted_token() {
        let mut r = LocalReplicator::leader();
        let token = r.propose(vec![1].into()).unwrap();
        r.rollback_unapplied_from(token.index);

        let err = r
            .wait_committed(token, std::time::Duration::from_millis(1))
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));
    }

    #[test]
    fn raft_wait_committed_resolves_after_quorum_commit() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let token = r.propose(vec![1].into()).unwrap();
        let pending = r.wait_committed(token, std::time::Duration::from_millis(1));
        assert!(matches!(pending, Err(EngineError::ProposalFailed(_))));

        r.register_follower_ack(token.index, 1);
        let committed = r
            .wait_committed(token, std::time::Duration::from_millis(1))
            .unwrap();
        assert_eq!(committed, token.index);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_wait_committed_polling() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let token = r.propose(vec![1].into()).unwrap();
        let before_pending = r.progress();
        let pending = r.wait_committed(token, std::time::Duration::from_millis(1));
        assert!(matches!(pending, Err(EngineError::ProposalFailed(_))));
        let after_pending = r.progress();
        assert_eq!(after_pending, before_pending);

        r.register_follower_ack(token.index, 1);
        let before_resolved = r.progress();
        let committed = r
            .wait_committed(token, std::time::Duration::from_millis(1))
            .unwrap();
        assert_eq!(committed, token.index);
        let after_resolved = r.progress();
        assert_eq!(after_resolved, before_resolved);
    }

    #[test]
    fn raft_install_snapshot_prunes_ack_tracking_and_uncompacted_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(4);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();
        let t3 = r.propose(vec![3].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t2.index, 1);
        assert_eq!(r.commit_index(), t2.index);
        assert!(r.ack_counts.contains_key(&t3.index));

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 5,
            snapshot_id: 42,
        });

        assert_eq!(r.current_term(), 5);
        assert_eq!(r.commit_index(), t2.index);
        assert_eq!(r.applied_index(), t2.index);
        assert_eq!(r.snapshot_meta().snapshot_id, 42);
        assert!(!r.ack_counts.contains_key(&t1.index));
        assert!(!r.ack_counts.contains_key(&t2.index));
        assert!(!r.ack_counts.contains_key(&t3.index));
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_advances_consistently_after_snapshot_install() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(4);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t2.index, 1);
        r.mark_applied(t1.index);

        let before = r.progress();
        assert_eq!(before.commit_index, t2.index);
        assert_eq!(before.applied_index, t1.index);
        assert_eq!(before.apply_gap(), 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 5,
            snapshot_id: 42,
        });

        let after = r.progress();
        assert_eq!(after.commit_index, t2.index);
        assert_eq!(after.applied_index, t2.index);
        assert_eq!(after.snapshot.snapshot_id, 42);
        assert_eq!(after.snapshot.last_included_index, t2.index);
        assert_eq!(after.apply_gap(), 0);
        assert!(after.is_caught_up());
        after.validate().unwrap();
    }

    #[test]
    fn raft_install_snapshot_preserves_next_index_from_uncompacted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![1].into()).unwrap();
        let _t2 = r.propose(vec![2].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 3,
            snapshot_id: 7,
        });

        let replacement = r.propose(vec![9].into()).unwrap();
        assert_eq!(replacement.index, 3);
    }

    #[test]
    fn raft_progress_snapshot_preserves_uncommitted_tail_after_snapshot_install() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(3);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.install_snapshot(SnapshotMeta {
            last_included_index: t1.index,
            last_included_term: 3,
            snapshot_id: 7,
        });

        let progress = r.progress();
        assert_eq!(progress.commit_index, t1.index);
        assert_eq!(progress.applied_index, t1.index);
        assert_eq!(progress.next_index, t2.index + 1);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert!(!progress.is_caught_up());
        progress.validate().unwrap();
    }

    #[test]
    fn raft_snapshot_meta_preserves_last_applied_term_across_term_bumps() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(2);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.mark_applied(t1.index);

        r.become_follower(7);
        let meta = r.snapshot_meta();

        assert_eq!(r.current_term(), 7);
        assert_eq!(meta.last_included_index, t1.index);
        assert_eq!(meta.last_included_term, 2);
    }

    #[test]
    fn raft_follower_append_entries_truncates_conflicting_uncommitted_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        let _t2_old = r.propose(vec![2].into()).unwrap();
        r.become_follower(2);

        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t1.index + 1,
                payload: vec![9].into(),
            }],
            t1.index + 1,
        )
        .unwrap();

        assert_eq!(r.commit_index(), t1.index + 1);
        assert_eq!(r.next_index, t1.index + 2);
        assert!(r
            .entries
            .iter()
            .any(|entry| entry.index == t1.index + 1 && entry.term == 2));
    }

    #[test]
    fn raft_follower_append_entries_empty_heartbeat_can_advance_commit_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        assert_eq!(r.commit_index(), t1.index);

        r.become_follower(2);
        let t2_index = t1.index + 1;
        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t2_index,
                payload: vec![2].into(),
            }],
            t1.index,
        )
        .unwrap();
        assert_eq!(r.commit_index(), t1.index);

        r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
            .unwrap();

        assert_eq!(r.commit_index(), t2_index);
        assert_eq!(r.next_index, t2_index + 1);
    }

    #[test]
    fn raft_progress_snapshot_tracks_heartbeat_commit_advancement() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);

        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let t2_index = t1.index + 1;
        r.append_entries_from_leader(
            2,
            t1.index,
            1,
            vec![LogEntry {
                term: 2,
                index: t2_index,
                payload: vec![2].into(),
            }],
            t1.index,
        )
        .unwrap();

        let before_heartbeat = r.progress();
        assert_eq!(before_heartbeat.commit_index, t1.index);
        assert_eq!(before_heartbeat.applied_index, 0);
        assert_eq!(before_heartbeat.next_index, t2_index + 1);
        assert_eq!(before_heartbeat.uncommitted_entry_count, 1);
        assert!(before_heartbeat.has_uncommitted_entries);
        assert_eq!(before_heartbeat.apply_gap(), t1.index as usize);

        r.append_entries_from_leader(2, t2_index, 2, vec![], t2_index)
            .unwrap();

        let after_heartbeat = r.progress();
        assert_eq!(after_heartbeat.commit_index, t2_index);
        assert_eq!(after_heartbeat.applied_index, 0);
        assert_eq!(after_heartbeat.next_index, t2_index + 1);
        assert_eq!(after_heartbeat.uncommitted_entry_count, 0);
        assert!(!after_heartbeat.has_uncommitted_entries);
        assert_eq!(after_heartbeat.apply_gap(), t2_index as usize);
        assert!(!after_heartbeat.is_caught_up());
        after_heartbeat.validate().unwrap();
    }

    #[test]
    fn raft_follower_append_entries_bumps_local_term_from_leader_term() {
        let mut r = RaftReplicator::new(3);
        r.become_candidate(3);

        r.append_entries_from_leader(4, 0, 0, vec![], 0).unwrap();

        assert_eq!(r.current_term(), 4);
        assert_eq!(r.role(), Role::Follower);
    }

    #[test]
    fn raft_follower_append_entries_rejects_stale_leader_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let err = r
            .append_entries_from_leader(4, 0, 0, vec![], 0)
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.current_term(), 5);
    }

    #[test]
    fn raft_follower_append_entries_rejects_stale_term_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(5);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;
        let entries_before = r.entries.clone();

        let err = r
            .append_entries_from_leader(
                4,
                t1.index,
                5,
                vec![LogEntry {
                    term: 4,
                    index: t1.index + 1,
                    payload: vec![9].into(),
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert_eq!(r.entries, entries_before);
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_stale_append_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(5);

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                t1.index,
                5,
                vec![LogEntry {
                    term: 4,
                    index: t1.index + 1,
                    payload: vec![9].into(),
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_entries_with_term_ahead_of_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let err = r
            .append_entries_from_leader(
                5,
                0,
                0,
                vec![LogEntry {
                    term: 6,
                    index: 1,
                    payload: vec![1].into(),
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.current_term(), 5);
        assert_eq!(r.commit_index(), 0);
        assert_eq!(r.next_index, 1);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_leader_rejects_follower_append_path_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(6);

        let committed = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(committed.index, 1);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                7,
                committed.index,
                term_before,
                vec![LogEntry {
                    term: 7,
                    index: committed.index + 1,
                    payload: vec![9].into(),
                }],
                committed.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert!(r.entries.iter().all(|entry| entry.term != 7));
    }

    #[test]
    fn raft_follower_append_entries_rejects_prev_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                999,
                vec![LogEntry {
                    term: 2,
                    index: t1.index + 1,
                    payload: vec![2].into(),
                }],
                t1.index + 1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
    }

    #[test]
    fn raft_follower_append_entries_rejects_non_contiguous_batches() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![
                    LogEntry {
                        term: 2,
                        index: t1.index + 1,
                        payload: vec![2].into(),
                    },
                    LogEntry {
                        term: 2,
                        index: t1.index + 3,
                        payload: vec![3].into(),
                    },
                ],
                t1.index + 3,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
        assert!(r.entries.iter().all(|entry| entry.index <= t1.index));
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_non_contiguous_append_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![
                    LogEntry {
                        term: 2,
                        index: t1.index + 1,
                        payload: vec![2].into(),
                    },
                    LogEntry {
                        term: 2,
                        index: t1.index + 3,
                        payload: vec![3].into(),
                    },
                ],
                t1.index + 3,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_first_entry_that_skips_prev_index() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                t1.index,
                1,
                vec![LogEntry {
                    term: 2,
                    index: t1.index + 2,
                    payload: vec![2].into(),
                }],
                t1.index + 2,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), t1.index);
        assert_eq!(r.next_index, t1.index + 1);
    }

    #[test]
    fn raft_follower_append_entries_does_not_overwrite_committed_entries() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(1);
        let t1 = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(t1.index, 1);
        r.become_follower(2);

        let err = r
            .append_entries_from_leader(
                2,
                0,
                0,
                vec![LogEntry {
                    term: 2,
                    index: t1.index,
                    payload: vec![9].into(),
                }],
                t1.index,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        let committed = r
            .entries
            .iter()
            .find(|entry| entry.index == t1.index)
            .unwrap();
        assert_eq!(committed.term, 1);
        assert_eq!(&committed.payload[..], &vec![1][..]);
    }

    #[test]
    fn raft_follower_append_entries_rejects_payload_mismatch_for_same_index_and_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.append_entries_from_leader(
            3,
            0,
            0,
            vec![LogEntry {
                term: 3,
                index: 1,
                payload: vec![1].into(),
            }],
            1,
        )
        .unwrap();

        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                3,
                0,
                0,
                vec![LogEntry {
                    term: 3,
                    index: 1,
                    payload: vec![9].into(),
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        let preserved = r.entries.iter().find(|entry| entry.index == 1).unwrap();
        assert_eq!(preserved.term, 3);
        assert_eq!(&preserved.payload[..], &vec![1][..]);
    }

    #[test]
    fn raft_follower_append_entries_caps_commit_index_to_local_log_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(2);

        r.append_entries_from_leader(
            2,
            0,
            0,
            vec![LogEntry {
                term: 2,
                index: 1,
                payload: vec![1].into(),
            }],
            99,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 1);
        assert_eq!(r.next_index, 2);
    }

    #[test]
    fn raft_follower_append_entries_missing_prev_index_is_rejected_without_state_mutation() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        let role_before = r.role();
        let term_before = r.current_term();
        let commit_before = r.commit_index();
        let next_before = r.next_index;

        let err = r
            .append_entries_from_leader(
                5,
                10,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 11,
                    payload: vec![7].into(),
                }],
                11,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), role_before);
        assert_eq!(r.current_term(), term_before);
        assert_eq!(r.commit_index(), commit_before);
        assert_eq!(r.next_index, next_before);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_candidate_append_rejection_still_updates_term_and_role_for_newer_leader() {
        let mut r = RaftReplicator::new(3);
        r.become_candidate(5);

        let err = r
            .append_entries_from_leader(
                6,
                10,
                5,
                vec![LogEntry {
                    term: 6,
                    index: 11,
                    payload: vec![7].into(),
                }],
                11,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.role(), Role::Follower);
        assert_eq!(r.current_term(), 6);
        assert_eq!(r.commit_index(), 0);
        assert_eq!(r.next_index, 1);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_newer_leader_rejection_discards_candidate_speculative_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(4);

        let committed = r.propose(vec![1].into()).unwrap();
        r.register_follower_ack(committed.index, 1);
        let speculative = r.propose(vec![2].into()).unwrap();
        let speculative_status = r.status_snapshot();
        assert!(speculative_status.has_speculative_tail());
        assert_eq!(speculative_status.live.next_index, speculative.index + 1);
        assert_eq!(speculative_status.live.uncommitted_entry_count, 1);

        r.become_candidate(5);
        let err = r
            .append_entries_from_leader(
                6,
                committed.index + 9,
                6,
                vec![LogEntry {
                    term: 6,
                    index: committed.index + 10,
                    payload: vec![9].into(),
                }],
                committed.index + 10,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        let after = r.status_snapshot();
        assert!(after.is_restart_equivalent());
        assert_eq!(after.live.role, Role::Follower);
        assert_eq!(after.live.term, 6);
        assert_eq!(after.live.commit_index, committed.index);
        assert_eq!(after.live.applied_index, 0);
        assert_eq!(after.live.next_index, committed.index + 1);
        assert_eq!(after.live.uncommitted_entry_count, 0);
        assert!(!after.live.has_uncommitted_entries);
        assert_eq!(after.live, after.durable);
        assert_eq!(r.entries.len(), 1);
        assert_eq!(r.entries[0].index, committed.index);
    }

    #[test]
    fn raft_newer_leader_rejection_discards_follower_speculative_tail() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.uncommitted_entry_count, 1);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 2);

        let err = r
            .append_entries_from_leader(
                5,
                99,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        let after = r.status_snapshot();
        assert!(after.is_restart_equivalent());
        assert_eq!(after.live.role, Role::Follower);
        assert_eq!(after.live.term, 5);
        assert_eq!(after.live.commit_index, 5);
        assert_eq!(after.live.applied_index, 5);
        assert_eq!(after.live.next_index, 6);
        assert_eq!(after.live.snapshot.snapshot_id, 2);
        assert_eq!(after.live.uncommitted_entry_count, 0);
        assert!(!after.live.has_uncommitted_entries);
        assert_eq!(after.live, after.durable);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_newer_leader_acceptance_replaces_follower_speculative_tail_with_fresh_gap() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();

        let before = r.status_snapshot();
        assert!(before.has_speculative_tail());
        assert_eq!(before.live.term, 4);
        assert_eq!(before.live.commit_index, 5);
        assert_eq!(before.live.next_index, 7);
        assert_eq!(before.live.uncommitted_entry_count, 1);
        assert_eq!(before.durable.snapshot.snapshot_id, 2);

        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![60].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![70].into(),
                },
            ],
            6,
        )
        .unwrap();

        let after = r.status_snapshot();
        assert_eq!(after.live.role, Role::Follower);
        assert_eq!(after.live.term, 5);
        assert_eq!(after.live.commit_index, 6);
        assert_eq!(after.live.applied_index, 5);
        assert_eq!(after.live.next_index, 8);
        assert_eq!(after.live.uncommitted_entry_count, 1);
        assert!(after.live.has_committed_entries_pending_apply);
        assert_eq!(after.live.committed_but_unapplied_count, 1);
        assert_eq!(after.durable.term, 5);
        assert_eq!(after.durable.commit_index, 6);
        assert_eq!(after.durable.applied_index, 5);
        assert_eq!(after.durable.next_index, 7);
        assert_eq!(after.durable.uncommitted_entry_count, 0);
        assert_eq!(
            after.recovery_gap,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );
        assert!(after.has_speculative_tail());
        assert_eq!(after.live.snapshot.snapshot_id, 2);
        assert_eq!(after.durable.snapshot.snapshot_id, 2);
        assert_eq!(r.entries.len(), 2);
        assert_eq!(r.entries[0].index, 6);
        assert_eq!(r.entries[0].term, 5);
        assert_eq!(&r.entries[0].payload[..], &[60u8][..]);
        assert_eq!(r.entries[1].index, 7);
        assert_eq!(r.entries[1].term, 5);
        assert_eq!(&r.entries[1].payload[..], &[70u8][..]);
    }

    #[test]
    fn raft_newer_leader_catch_up_promotes_fresh_tail_without_reintroducing_stale_gap() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![60].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![70].into(),
                },
            ],
            6,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert_eq!(after_repair.live.commit_index, 6);
        assert_eq!(after_repair.live.applied_index, 5);
        assert_eq!(after_repair.live.next_index, 8);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.durable.commit_index, 6);
        assert_eq!(after_repair.durable.next_index, 7);
        assert!(after_repair.has_speculative_tail());

        r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();

        let after_commit = r.status_snapshot();
        assert_eq!(after_commit.live.commit_index, 7);
        assert_eq!(after_commit.live.applied_index, 5);
        assert_eq!(after_commit.live.next_index, 8);
        assert_eq!(after_commit.live.uncommitted_entry_count, 0);
        assert_eq!(after_commit.live.committed_but_unapplied_count, 2);
        assert_eq!(after_commit.durable.commit_index, 7);
        assert_eq!(after_commit.durable.applied_index, 5);
        assert_eq!(after_commit.durable.next_index, 8);
        assert_eq!(after_commit.durable.uncommitted_entry_count, 0);
        assert!(after_commit.is_restart_equivalent());
        assert!(!after_commit.has_speculative_tail());

        r.mark_applied(7);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert!(!after_apply.has_speculative_tail());
        assert_eq!(after_apply.live.commit_index, 7);
        assert_eq!(after_apply.live.applied_index, 7);
        assert_eq!(after_apply.live.next_index, 8);
        assert_eq!(after_apply.live.committed_but_unapplied_count, 0);
        assert_eq!(after_apply.live, after_apply.durable);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 2);
    }

    #[test]
    fn raft_follower_append_entries_accepts_prev_index_at_snapshot_boundary_after_apply_advances() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 1,
        });

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap();
        r.mark_applied(6);

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 6);
        assert_eq!(r.next_index, 7);
    }

    #[test]
    fn raft_progress_snapshot_remains_consistent_across_append_at_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(3);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 1,
        });

        let before = r.progress();
        assert_eq!(before.commit_index, 5);
        assert_eq!(before.applied_index, 5);
        assert_eq!(before.next_index, 6);
        assert!(before.is_caught_up());

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap();

        let after_append = r.progress();
        assert_eq!(after_append.commit_index, 6);
        assert_eq!(after_append.applied_index, 5);
        assert_eq!(after_append.next_index, 7);
        assert_eq!(after_append.apply_gap(), 1);
        assert!(!after_append.is_caught_up());

        r.mark_applied(6);
        let after_apply = r.progress();
        assert_eq!(after_apply.commit_index, 6);
        assert_eq!(after_apply.applied_index, 6);
        assert_eq!(after_apply.next_index, 7);
        assert_eq!(after_apply.apply_gap(), 0);
        assert!(after_apply.is_caught_up());

        r.append_entries_from_leader(
            3,
            5,
            2,
            vec![LogEntry {
                term: 3,
                index: 6,
                payload: vec![6].into(),
            }],
            6,
        )
        .unwrap();

        let after_repeat = r.progress();
        assert_eq!(after_repeat, after_apply);
    }

    #[test]
    fn raft_follower_append_entries_rejects_snapshot_boundary_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let err = r
            .append_entries_from_leader(
                4,
                5,
                2,
                vec![LogEntry {
                    term: 4,
                    index: 6,
                    payload: vec![6].into(),
                }],
                6,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), 5);
        assert_eq!(r.applied_index(), 5);
        assert_eq!(r.next_index, 6);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_snapshot_boundary_term_mismatch() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                5,
                2,
                vec![LogEntry {
                    term: 4,
                    index: 6,
                    payload: vec![6].into(),
                }],
                6,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_follower_append_entries_rejects_prev_index_behind_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let err = r
            .append_entries_from_leader(
                4,
                0,
                0,
                vec![LogEntry {
                    term: 4,
                    index: 1,
                    payload: vec![1].into(),
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.commit_index(), 5);
        assert_eq!(r.applied_index(), 5);
        assert_eq!(r.next_index, 6);
        assert!(r.entries.is_empty());
    }

    #[test]
    fn raft_progress_snapshot_is_stable_across_prev_index_behind_snapshot_boundary() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 3,
            snapshot_id: 1,
        });

        let before = r.progress();

        let err = r
            .append_entries_from_leader(
                4,
                0,
                0,
                vec![LogEntry {
                    term: 4,
                    index: 1,
                    payload: vec![1].into(),
                }],
                1,
            )
            .unwrap_err();

        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert_eq!(r.progress(), before);
    }

    #[test]
    fn raft_install_snapshot_with_same_index_wrong_term_is_a_progress_no_op() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        let baseline = r.progress();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 3,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.snapshot_meta().snapshot_id, 2);

        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9].into(),
            }],
            6,
        )
        .unwrap();

        assert_eq!(r.commit_index(), 6);
        assert_eq!(r.next_index, 7);
    }

    #[test]
    fn raft_install_snapshot_with_higher_index_lower_term_is_a_progress_no_op() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        let baseline = r.progress();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 3,
            snapshot_id: 8,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.snapshot_meta().snapshot_id, 2);
    }

    #[test]
    fn raft_install_snapshot_gap_is_stable_for_same_frontier_wrong_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9].into(),
            }],
            5,
        )
        .unwrap();

        let baseline = r.progress();
        let baseline_recovery = r.recovery_progress();
        let baseline_gap = r.recovery_progress_gap();
        assert_eq!(
            baseline_gap,
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 2,
            snapshot_id: 3,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.recovery_progress(), baseline_recovery);
        assert_eq!(r.recovery_progress_gap(), baseline_gap);
        assert_eq!(r.snapshot_meta().snapshot_id, 2);
    }

    #[test]
    fn raft_install_snapshot_gap_is_stable_for_higher_frontier_lower_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9].into(),
            }],
            5,
        )
        .unwrap();

        let baseline = r.progress();
        let baseline_recovery = r.recovery_progress();
        let baseline_gap = r.recovery_progress_gap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 3,
            snapshot_id: 8,
        });

        assert_eq!(r.progress(), baseline);
        assert_eq!(r.recovery_progress(), baseline_recovery);
        assert_eq!(r.recovery_progress_gap(), baseline_gap);
        assert_eq!(r.snapshot_meta().snapshot_id, 2);
    }

    #[test]
    fn raft_install_snapshot_updates_snapshot_id_for_same_frontier_same_term() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        assert_eq!(r.snapshot_meta().snapshot_id, 7);
        assert_eq!(r.progress().snapshot.snapshot_id, 7);
    }

    #[test]
    fn raft_install_snapshot_advancing_frontier_replaces_snapshot_identity_exactly() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 4,
        });

        let snapshot = r.snapshot_meta();
        assert_eq!(snapshot.last_included_index, 8);
        assert_eq!(snapshot.last_included_term, 5);
        assert_eq!(snapshot.snapshot_id, 4);
        assert_eq!(r.progress().snapshot, snapshot);
        assert_eq!(r.status_snapshot().live.snapshot, snapshot);
        assert_eq!(r.status_snapshot().durable.snapshot, snapshot);
    }

    #[test]
    fn raft_install_snapshot_advancing_frontier_preserves_compatible_suffix() {
        let mut r = RaftReplicator::new(3);
        r.become_leader(5);

        let t1 = r.propose(vec![1].into()).unwrap();
        let t2 = r.propose(vec![2].into()).unwrap();
        let t3 = r.propose(vec![3].into()).unwrap();

        r.register_follower_ack(t1.index, 1);
        r.register_follower_ack(t2.index, 1);
        assert_eq!(r.commit_index(), t2.index);
        assert!(r.ack_counts.contains_key(&t3.index));

        r.install_snapshot(SnapshotMeta {
            last_included_index: t2.index,
            last_included_term: 5,
            snapshot_id: 29,
        });

        let progress = r.progress();
        assert_eq!(progress.snapshot.snapshot_id, 29);
        assert_eq!(progress.snapshot.last_included_index, t2.index);
        assert_eq!(progress.snapshot.last_included_term, 5);
        assert_eq!(progress.commit_index, t2.index);
        assert_eq!(progress.applied_index, t2.index);
        assert_eq!(progress.next_index, t3.index + 1);
        assert_eq!(progress.uncommitted_entry_count, 1);
        assert!(progress.has_uncommitted_entries);
        assert_eq!(r.entries.len(), 1);
        assert_eq!(r.entries[0].index, t3.index);
        assert!(r.ack_counts.contains_key(&t3.index));

        let status = r.status_snapshot();
        assert!(status.has_speculative_tail());
        assert_eq!(status.live.snapshot.snapshot_id, 29);
        assert_eq!(status.durable.snapshot.snapshot_id, 29);
        assert_eq!(status.recovery_gap.next_index_gap, 1);
        assert_eq!(status.recovery_gap.uncommitted_entry_gap, 1);
    }

    #[test]
    fn same_frontier_same_term_snapshot_refresh_preserves_speculative_gap_semantics() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9].into(),
            }],
            5,
        )
        .unwrap();

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 2);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 2);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        let refreshed = r.status_snapshot();
        assert!(refreshed.has_speculative_tail());
        assert_eq!(refreshed.recovery_gap, baseline.recovery_gap);
        assert_eq!(refreshed.live.commit_index, baseline.live.commit_index);
        assert_eq!(refreshed.live.applied_index, baseline.live.applied_index);
        assert_eq!(refreshed.live.next_index, baseline.live.next_index);
        assert_eq!(
            refreshed.live.uncommitted_entry_count,
            baseline.live.uncommitted_entry_count
        );
        assert_eq!(
            refreshed.durable.commit_index,
            baseline.durable.commit_index
        );
        assert_eq!(
            refreshed.durable.applied_index,
            baseline.durable.applied_index
        );
        assert_eq!(refreshed.durable.next_index, baseline.durable.next_index);
        assert_eq!(
            refreshed.durable.uncommitted_entry_count,
            baseline.durable.uncommitted_entry_count
        );
        assert_eq!(refreshed.live.snapshot.last_included_index, 5);
        assert_eq!(refreshed.live.snapshot.last_included_term, 4);
        assert_eq!(refreshed.durable.snapshot.last_included_index, 5);
        assert_eq!(refreshed.durable.snapshot.last_included_term, 4);
        assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
        assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);
    }

    #[test]
    fn same_frontier_snapshot_refresh_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![9].into(),
            }],
            5,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        let refreshed = r.status_snapshot();
        assert!(refreshed.has_speculative_tail());
        assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
        assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);

        r.become_candidate(5);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.commit_index, 5);
        assert_eq!(after_role_change.live.applied_index, 5);
        assert_eq!(after_role_change.live.next_index, 6);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 7);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, after_role_change.live.term);
        assert_eq!(
            after_role_change.durable.commit_index,
            after_role_change.live.commit_index
        );
        assert_eq!(
            after_role_change.durable.applied_index,
            after_role_change.live.applied_index
        );
        assert_eq!(
            after_role_change.durable.next_index,
            after_role_change.live.next_index
        );
        assert_eq!(
            after_role_change.durable.snapshot,
            after_role_change.live.snapshot
        );
    }

    #[test]
    fn same_frontier_snapshot_refresh_survives_newer_leader_rejection_and_catch_up() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        let refreshed = r.status_snapshot();
        assert!(refreshed.has_speculative_tail());
        assert_eq!(refreshed.live.snapshot.snapshot_id, 7);
        assert_eq!(refreshed.durable.snapshot.snapshot_id, 7);

        let err = r
            .append_entries_from_leader(
                5,
                99,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 5);
        assert_eq!(after_reject.live.commit_index, 5);
        assert_eq!(after_reject.live.applied_index, 5);
        assert_eq!(after_reject.live.next_index, 6);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 7);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 7);

        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![60].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![70].into(),
                },
            ],
            6,
        )
        .unwrap();

        let after_append = r.status_snapshot();
        assert!(after_append.has_speculative_tail());
        assert_eq!(after_append.live.term, 5);
        assert_eq!(after_append.live.commit_index, 6);
        assert_eq!(after_append.live.applied_index, 5);
        assert_eq!(after_append.live.next_index, 8);
        assert_eq!(after_append.live.snapshot.snapshot_id, 7);
        assert_eq!(after_append.durable.commit_index, 6);
        assert_eq!(after_append.durable.applied_index, 5);
        assert_eq!(after_append.durable.next_index, 7);
        assert_eq!(after_append.durable.snapshot.snapshot_id, 7);

        r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();

        let after_commit = r.status_snapshot();
        assert!(after_commit.is_restart_equivalent());
        assert_eq!(after_commit.live.commit_index, 7);
        assert_eq!(after_commit.live.applied_index, 5);
        assert_eq!(after_commit.live.next_index, 8);
        assert_eq!(after_commit.live.committed_but_unapplied_count, 2);
        assert_eq!(after_commit.live.snapshot.snapshot_id, 7);
        assert_eq!(after_commit.durable.snapshot.snapshot_id, 7);

        r.mark_applied(7);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live, after_apply.durable);
        assert_eq!(after_apply.live.applied_index, 7);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 7);
    }

    #[test]
    fn same_frontier_snapshot_refresh_keeps_recovery_bundle_aligned_through_newer_leader_handoff() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        let err = r
            .append_entries_from_leader(
                5,
                99,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![60].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![70].into(),
                },
            ],
            6,
        )
        .unwrap();

        let after_append = r.status_snapshot();
        assert!(after_append.has_speculative_tail());
        assert_eq!(after_append.durable.snapshot.snapshot_id, 7);

        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 7);
        let recovery_progress = recovery.progress_as_follower().unwrap();
        assert_eq!(recovery_progress, after_append.durable);

        let resumed = RaftReplicator::resume_as_follower(3, recovery).unwrap();
        let resumed_status = resumed.status_snapshot();
        assert_eq!(resumed_status.live, after_append.durable);
        assert_eq!(resumed_status.durable, after_append.durable);
        assert!(resumed_status.is_restart_equivalent());

        r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();
        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 7);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(
            resumed.status_snapshot().live,
            recovery.progress_as_follower().unwrap()
        );
    }

    #[test]
    fn same_frontier_snapshot_refresh_keeps_recovery_gap_explicit_through_newer_leader_handoff() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 2,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![LogEntry {
                term: 4,
                index: 6,
                payload: vec![6].into(),
            }],
            5,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 7,
        });

        assert_eq!(
            r.recovery_progress_gap(),
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );

        let err = r
            .append_entries_from_leader(
                5,
                99,
                5,
                vec![LogEntry {
                    term: 5,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![60].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![70].into(),
                },
            ],
            6,
        )
        .unwrap();
        assert_eq!(
            r.recovery_progress_gap(),
            RecoveryProgressGap {
                commit_index_gap: 0,
                applied_index_gap: 0,
                next_index_gap: 1,
                uncommitted_entry_gap: 1,
            }
        );

        r.append_entries_from_leader(5, 7, 5, vec![], 7).unwrap();
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        r.mark_applied(7);
        assert!(r.recovery_progress_gap().is_restart_equivalent());
        assert_eq!(r.recovery_state().snapshot.snapshot_id, 7);
    }

    #[test]
    fn advanced_frontier_snapshot_discards_incompatible_speculative_tail_before_epoch_handoffs() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(4);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            4,
            5,
            4,
            vec![
                LogEntry {
                    term: 4,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 4,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 4,
                    index: 8,
                    payload: vec![8].into(),
                },
                LogEntry {
                    term: 4,
                    index: 9,
                    payload: vec![9].into(),
                },
            ],
            5,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 4,
        });

        let refreshed = r.status_snapshot();
        assert!(refreshed.is_restart_equivalent());
        assert_eq!(refreshed.live.snapshot.snapshot_id, 4);
        assert_eq!(refreshed.durable.snapshot.snapshot_id, 4);
        assert_eq!(refreshed.live.snapshot.last_included_index, 8);
        assert_eq!(refreshed.live.snapshot.last_included_term, 5);
        assert_eq!(refreshed.live.commit_index, 8);
        assert_eq!(refreshed.live.applied_index, 8);
        assert_eq!(refreshed.live.next_index, 9);
        assert_eq!(refreshed.recovery_gap.next_index_gap, 0);
        assert_eq!(refreshed.recovery_gap.uncommitted_entry_gap, 0);

        let err = r
            .append_entries_from_leader(
                6,
                99,
                6,
                vec![LogEntry {
                    term: 6,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 6);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 4);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 4);
        assert_eq!(after_reject.live.commit_index, 8);
        assert_eq!(after_reject.live.applied_index, 8);
        assert_eq!(after_reject.live.next_index, 9);

        r.append_entries_from_leader(
            6,
            8,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            9,
        )
        .unwrap();

        let after_append = r.status_snapshot();
        assert!(after_append.has_speculative_tail());
        assert_eq!(after_append.live.snapshot.snapshot_id, 4);
        assert_eq!(after_append.durable.snapshot.snapshot_id, 4);
        assert_eq!(after_append.live.commit_index, 9);
        assert_eq!(after_append.live.applied_index, 8);
        assert_eq!(after_append.live.next_index, 11);
        assert_eq!(after_append.durable.commit_index, 9);
        assert_eq!(after_append.durable.applied_index, 8);
        assert_eq!(after_append.durable.next_index, 10);

        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 4);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_append.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_append.durable
        );

        r.append_entries_from_leader(6, 10, 6, vec![], 10).unwrap();
        assert!(r.recovery_progress_gap().is_restart_equivalent());
        assert_eq!(r.recovery_state().snapshot.snapshot_id, 4);

        r.mark_applied(10);
        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live, after_apply.durable);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 4);
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_is_still_discarded_on_newer_leader_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        assert!(r.status_snapshot().has_speculative_tail());

        let err = r
            .append_entries_from_leader(
                6,
                99,
                6,
                vec![LogEntry {
                    term: 6,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let status = r.status_snapshot();
        assert!(status.is_restart_equivalent());
        assert_eq!(status.live.term, 6);
        assert_eq!(status.live.snapshot.snapshot_id, 29);
        assert_eq!(status.durable.snapshot.snapshot_id, 29);
        assert_eq!(status.live.commit_index, 7);
        assert_eq!(status.live.applied_index, 7);
        assert_eq!(status.live.next_index, 8);
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_keeps_exact_identity_across_resume_and_repair() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });

        let with_compatible_suffix = r.status_snapshot();
        assert!(with_compatible_suffix.has_speculative_tail());
        assert_eq!(with_compatible_suffix.live.snapshot.snapshot_id, 29);
        assert_eq!(with_compatible_suffix.durable.snapshot.snapshot_id, 29);
        assert_eq!(with_compatible_suffix.live.commit_index, 7);
        assert_eq!(with_compatible_suffix.live.applied_index, 7);
        assert_eq!(with_compatible_suffix.live.next_index, 9);
        assert_eq!(with_compatible_suffix.durable.commit_index, 7);
        assert_eq!(with_compatible_suffix.durable.applied_index, 7);
        assert_eq!(with_compatible_suffix.durable.next_index, 8);

        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 29);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        let resumed_status = resumed.status_snapshot();
        assert!(resumed_status.is_restart_equivalent());
        assert_eq!(resumed_status.live, with_compatible_suffix.durable);
        assert_eq!(resumed_status.durable, with_compatible_suffix.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            with_compatible_suffix.durable
        );

        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.term, 6);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 29);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 29);
        assert_eq!(after_repair.live.commit_index, 8);
        assert_eq!(after_repair.live.applied_index, 7);
        assert_eq!(after_repair.live.next_index, 10);
        assert_eq!(after_repair.durable.commit_index, 8);
        assert_eq!(after_repair.durable.applied_index, 7);
        assert_eq!(after_repair.durable.next_index, 9);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

        let recovery_after_repair = r.recovery_state();
        assert_eq!(recovery_after_repair.snapshot.snapshot_id, 29);
        let resumed_after_repair =
            RaftReplicator::resume_as_follower(3, recovery_after_repair.clone()).unwrap();
        assert_eq!(
            resumed_after_repair.status_snapshot().live,
            after_repair.durable
        );
        assert_eq!(
            recovery_after_repair.progress_as_follower().unwrap(),
            after_repair.durable
        );
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 29);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 29);

        r.become_candidate(6);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 6);
        assert_eq!(after_role_change.live.commit_index, 7);
        assert_eq!(after_role_change.live.applied_index, 7);
        assert_eq!(after_role_change.live.next_index, 8);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 29);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 6);
        assert_eq!(after_role_change.durable.commit_index, 7);
        assert_eq!(after_role_change.durable.applied_index, 7);
        assert_eq!(after_role_change.durable.next_index, 8);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 29);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 6);
        assert_eq!(recovery.snapshot.snapshot_id, 29);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 29);
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_refresh_keeps_exact_identity_across_resume_repair_and_apply_completion(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 29);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 29);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });

        let refreshed = r.status_snapshot();
        assert!(refreshed.has_speculative_tail());
        assert_eq!(refreshed.recovery_gap, baseline.recovery_gap);
        assert_eq!(refreshed.live.commit_index, baseline.live.commit_index);
        assert_eq!(refreshed.live.applied_index, baseline.live.applied_index);
        assert_eq!(refreshed.live.next_index, baseline.live.next_index);
        assert_eq!(
            refreshed.durable.commit_index,
            baseline.durable.commit_index
        );
        assert_eq!(
            refreshed.durable.applied_index,
            baseline.durable.applied_index
        );
        assert_eq!(refreshed.durable.next_index, baseline.durable.next_index);
        assert_eq!(refreshed.live.snapshot.snapshot_id, 31);
        assert_eq!(refreshed.durable.snapshot.snapshot_id, 31);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 31);
        let resumed = RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, refreshed.durable);
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            refreshed.durable
        );

        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.term, 6);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 31);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 31);
        assert_eq!(after_repair.live.commit_index, 8);
        assert_eq!(after_repair.live.applied_index, 7);
        assert_eq!(after_repair.live.next_index, 10);
        assert_eq!(after_repair.durable.commit_index, 8);
        assert_eq!(after_repair.durable.applied_index, 7);
        assert_eq!(after_repair.durable.next_index, 9);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.snapshot.snapshot_id, 31);
        let resumed_after_repair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_repair.status_snapshot().live,
            after_repair.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            after_repair.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let after_commit = r.status_snapshot();
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert!(!after_commit.has_speculative_tail());
        assert_eq!(after_commit.live.snapshot.snapshot_id, 31);
        assert_eq!(after_commit.durable.snapshot.snapshot_id, 31);
        assert_eq!(after_commit.live.commit_index, 9);
        assert_eq!(after_commit.live.applied_index, 7);
        assert_eq!(after_commit.live.next_index, 10);
        assert_eq!(after_commit.durable.commit_index, 9);
        assert_eq!(after_commit.durable.applied_index, 7);
        assert_eq!(after_commit.durable.next_index, 10);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.snapshot.snapshot_id, 31);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            after_commit.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            after_commit.durable
        );

        r.mark_applied(9);
        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.snapshot.snapshot_id, 31);
        assert_eq!(after_apply.durable.snapshot.snapshot_id, 31);
        assert_eq!(after_apply.live.commit_index, 9);
        assert_eq!(after_apply.live.applied_index, 9);
        assert_eq!(after_apply.live.next_index, 10);
        assert_eq!(after_apply.durable, after_apply.live);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.snapshot.snapshot_id, 31);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(resumed_after_apply.status_snapshot().live, after_apply.live);
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            after_apply.live
        );
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_refresh_ignores_stale_snapshots_without_perturbing_gap()
    {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });

        let baseline_status = r.status_snapshot();
        let baseline_recovery = r.recovery_state();
        let baseline_gap = r.recovery_progress_gap();
        assert!(baseline_status.has_speculative_tail());
        assert_eq!(baseline_status.live.snapshot.snapshot_id, 31);
        assert_eq!(baseline_status.durable.snapshot.snapshot_id, 31);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 101,
        });

        assert_eq!(r.status_snapshot(), baseline_status);
        assert_eq!(r.recovery_state(), baseline_recovery);
        assert_eq!(r.recovery_progress_gap(), baseline_gap);
        assert_eq!(r.snapshot_meta().snapshot_id, 31);
        assert_eq!(
            baseline_recovery.progress_as_follower().unwrap(),
            baseline_status.durable
        );
    }

    #[test]
    fn compatible_advanced_snapshot_suffix_refresh_survives_newer_leader_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        assert!(r.status_snapshot().has_speculative_tail());

        let err = r
            .append_entries_from_leader(
                6,
                99,
                6,
                vec![LogEntry {
                    term: 6,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 6);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 31);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 31);
        assert_eq!(after_reject.live.commit_index, 7);
        assert_eq!(after_reject.live.applied_index, 7);
        assert_eq!(after_reject.live.next_index, 8);
        assert!(r.recovery_progress_gap().is_restart_equivalent());
        assert_eq!(r.recovery_state().snapshot.snapshot_id, 31);
    }

    #[test]
    fn refreshed_compatible_suffix_repair_phase_snapshot_refresh_updates_durable_identity_without_perturbing_gap(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 31);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 31);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 37,
        });

        let refreshed_repair_status = r.status_snapshot();
        assert!(refreshed_repair_status.has_speculative_tail());
        assert_eq!(
            refreshed_repair_status.recovery_gap,
            repair_status.recovery_gap
        );
        assert_eq!(refreshed_repair_status.live.term, repair_status.live.term);
        assert_eq!(
            refreshed_repair_status.live.commit_index,
            repair_status.live.commit_index
        );
        assert_eq!(
            refreshed_repair_status.live.applied_index,
            repair_status.live.applied_index
        );
        assert_eq!(
            refreshed_repair_status.live.next_index,
            repair_status.live.next_index
        );
        assert_eq!(
            refreshed_repair_status.durable.commit_index,
            repair_status.durable.commit_index
        );
        assert_eq!(
            refreshed_repair_status.durable.applied_index,
            repair_status.durable.applied_index
        );
        assert_eq!(
            refreshed_repair_status.durable.next_index,
            repair_status.durable.next_index
        );
        assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 37);
        assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 37);

        let refreshed_repair_recovery = r.recovery_state();
        assert_eq!(refreshed_repair_recovery.snapshot.snapshot_id, 37);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_repair_status.durable
        );
        assert_eq!(
            refreshed_repair_recovery.progress_as_follower().unwrap(),
            refreshed_repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let after_commit = r.status_snapshot();
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert!(!after_commit.has_speculative_tail());
        assert_eq!(after_commit.live.snapshot.snapshot_id, 37);
        assert_eq!(after_commit.durable.snapshot.snapshot_id, 37);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.snapshot.snapshot_id, 37);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            after_commit.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            after_commit.durable
        );

        r.mark_applied(9);
        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.snapshot.snapshot_id, 37);
        assert_eq!(after_apply.durable.snapshot.snapshot_id, 37);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.snapshot.snapshot_id, 37);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(resumed_after_apply.status_snapshot().live, after_apply.live);
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            after_apply.live
        );
    }

    #[test]
    fn refreshed_compatible_suffix_repair_phase_second_refresh_still_collapses_cleanly_on_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 31);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 31);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 37,
        });

        let refreshed_repair = r.status_snapshot();
        assert!(refreshed_repair.has_speculative_tail());
        assert_eq!(refreshed_repair.recovery_gap, repair_status.recovery_gap);
        assert_eq!(refreshed_repair.live.snapshot.snapshot_id, 37);
        assert_eq!(refreshed_repair.durable.snapshot.snapshot_id, 37);

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 7);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 37);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 37);
        assert_eq!(after_reject.live.commit_index, 8);
        assert_eq!(after_reject.live.applied_index, 7);
        assert_eq!(after_reject.live.next_index, 9);
        assert_eq!(after_reject.durable.commit_index, 8);
        assert_eq!(after_reject.durable.applied_index, 7);
        assert_eq!(after_reject.durable.next_index, 9);
        assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
        assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
        assert!(after_reject.live.has_committed_entries_pending_apply);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let after_reject_recovery = r.recovery_state();
        assert_eq!(after_reject_recovery.term, 7);
        assert_eq!(after_reject_recovery.snapshot.snapshot_id, 37);
        let resumed_after_reject =
            RaftReplicator::resume_as_follower(3, after_reject_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_reject.status_snapshot().live,
            after_reject.durable
        );
        assert_eq!(
            after_reject_recovery.progress_as_follower().unwrap(),
            after_reject.durable
        );
    }

    #[test]
    fn refreshed_compatible_suffix_stale_snapshots_remain_noops_during_repair_commit_and_apply() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        assert!(repair_status.has_speculative_tail());

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 31);
    }

    #[test]
    fn refreshed_compatible_suffix_second_repair_refresh_keeps_stale_snapshots_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 37,
        });

        let refreshed_repair_status = r.status_snapshot();
        let refreshed_repair_recovery = r.recovery_state();
        let refreshed_repair_gap = r.recovery_progress_gap();
        assert!(refreshed_repair_status.has_speculative_tail());
        assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 37);
        assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 37);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), refreshed_repair_status);
        assert_eq!(r.recovery_state(), refreshed_repair_recovery);
        assert_eq!(r.recovery_progress_gap(), refreshed_repair_gap);
        let resumed_during_refreshed_repair_after_stale =
            RaftReplicator::resume_as_follower(3, refreshed_repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair_after_stale
                .status_snapshot()
                .live,
            refreshed_repair_status.durable
        );
        assert_eq!(
            refreshed_repair_recovery.progress_as_follower().unwrap(),
            refreshed_repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());
        assert_eq!(commit_status.live.snapshot.snapshot_id, 37);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 37);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 37);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 37);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 6,
            last_included_term: 5,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 4,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 4,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 37);
    }

    #[test]
    fn refreshed_compatible_suffix_second_repair_refresh_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 31,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 37,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 37);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 37);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 8);
        assert_eq!(after_role_change.live.applied_index, 7);
        assert_eq!(after_role_change.live.next_index, 9);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 37);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 8);
        assert_eq!(after_role_change.durable.applied_index, 7);
        assert_eq!(after_role_change.durable.next_index, 9);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 37);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 37);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 37);
    }

    #[test]
    fn repair_phase_advanced_snapshot_replaces_durable_identity_and_preserves_fresh_suffix() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 29);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 29);
        assert_eq!(repair_status.live.commit_index, 8);
        assert_eq!(repair_status.live.applied_index, 7);
        assert_eq!(repair_status.live.next_index, 10);
        assert_eq!(repair_status.durable.commit_index, 8);
        assert_eq!(repair_status.durable.applied_index, 7);
        assert_eq!(repair_status.durable.next_index, 9);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });

        let advanced_status = r.status_snapshot();
        assert!(advanced_status.has_speculative_tail());
        assert_eq!(advanced_status.live.term, 6);
        assert_eq!(advanced_status.live.snapshot.snapshot_id, 41);
        assert_eq!(advanced_status.durable.snapshot.snapshot_id, 41);
        assert_eq!(advanced_status.live.commit_index, 8);
        assert_eq!(advanced_status.live.applied_index, 8);
        assert_eq!(advanced_status.live.next_index, 10);
        assert_eq!(advanced_status.live.uncommitted_entry_count, 1);
        assert_eq!(advanced_status.durable.commit_index, 8);
        assert_eq!(advanced_status.durable.applied_index, 8);
        assert_eq!(advanced_status.durable.next_index, 9);
        assert_eq!(advanced_status.durable.uncommitted_entry_count, 0);
        assert_eq!(advanced_status.recovery_gap.next_index_gap, 1);
        assert_eq!(advanced_status.recovery_gap.uncommitted_entry_gap, 1);

        let advanced_recovery = r.recovery_state();
        assert_eq!(advanced_recovery.snapshot.snapshot_id, 41);
        let resumed_during_advanced_repair =
            RaftReplicator::resume_as_follower(3, advanced_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_advanced_repair.status_snapshot().live,
            advanced_status.durable
        );
        assert_eq!(
            advanced_recovery.progress_as_follower().unwrap(),
            advanced_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 41);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 41);

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 41);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 41);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 41);
    }

    #[test]
    fn repair_phase_advanced_snapshot_same_frontier_refresh_updates_identity_without_perturbing_gap(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });

        let advanced_status = r.status_snapshot();
        assert!(advanced_status.has_speculative_tail());
        let advanced_gap = advanced_status.recovery_gap.clone();
        assert_eq!(advanced_status.live.snapshot.snapshot_id, 41);
        assert_eq!(advanced_status.durable.snapshot.snapshot_id, 41);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.term, 6);
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 17);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 17);
        assert_eq!(
            refreshed_status.live.commit_index,
            advanced_status.live.commit_index
        );
        assert_eq!(
            refreshed_status.live.applied_index,
            advanced_status.live.applied_index
        );
        assert_eq!(
            refreshed_status.live.next_index,
            advanced_status.live.next_index
        );
        assert_eq!(
            refreshed_status.live.uncommitted_entry_count,
            advanced_status.live.uncommitted_entry_count
        );
        assert_eq!(
            refreshed_status.durable.commit_index,
            advanced_status.durable.commit_index
        );
        assert_eq!(
            refreshed_status.durable.applied_index,
            advanced_status.durable.applied_index
        );
        assert_eq!(
            refreshed_status.durable.next_index,
            advanced_status.durable.next_index
        );
        assert_eq!(
            refreshed_status.durable.uncommitted_entry_count,
            advanced_status.durable.uncommitted_entry_count
        );
        assert_eq!(refreshed_status.recovery_gap, advanced_gap);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 17);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_status.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            refreshed_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 17);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 17);

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 17);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 17);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 17);
    }

    #[test]
    fn repair_phase_advanced_snapshot_refresh_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 17);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 17);
        assert_eq!(before_role_change.live.commit_index, 8);
        assert_eq!(before_role_change.live.applied_index, 8);
        assert_eq!(before_role_change.live.next_index, 10);
        assert_eq!(before_role_change.durable.next_index, 9);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 8);
        assert_eq!(after_role_change.live.applied_index, 8);
        assert_eq!(after_role_change.live.next_index, 9);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 17);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 8);
        assert_eq!(after_role_change.durable.applied_index, 8);
        assert_eq!(after_role_change.durable.next_index, 9);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 17);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 17);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 17);
    }

    #[test]
    fn repair_phase_advanced_snapshot_refresh_collapses_cleanly_on_newer_leader_rejection() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 17);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 17);
        assert_eq!(refreshed_status.live.commit_index, 8);
        assert_eq!(refreshed_status.live.applied_index, 8);
        assert_eq!(refreshed_status.live.next_index, 10);
        assert_eq!(refreshed_status.durable.commit_index, 8);
        assert_eq!(refreshed_status.durable.applied_index, 8);
        assert_eq!(refreshed_status.durable.next_index, 9);

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 7);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 17);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 17);
        assert_eq!(after_reject.live.commit_index, 8);
        assert_eq!(after_reject.live.applied_index, 8);
        assert_eq!(after_reject.live.next_index, 9);
        assert_eq!(after_reject.live.uncommitted_entry_count, 0);
        assert_eq!(after_reject.durable.commit_index, 8);
        assert_eq!(after_reject.durable.applied_index, 8);
        assert_eq!(after_reject.durable.next_index, 9);
        assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
        assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
        assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let after_reject_recovery = r.recovery_state();
        assert_eq!(after_reject_recovery.term, 7);
        assert_eq!(after_reject_recovery.snapshot.snapshot_id, 17);
        let resumed_after_reject =
            RaftReplicator::resume_as_follower(3, after_reject_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_reject.status_snapshot().live,
            after_reject.durable
        );
        assert_eq!(
            after_reject_recovery.progress_as_follower().unwrap(),
            after_reject.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 17);
    }

    #[test]
    fn repair_phase_advanced_snapshot_second_refresh_still_collapses_cleanly_on_newer_leader_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.live.commit_index, 8);
        assert_eq!(refreshed_status.live.applied_index, 8);
        assert_eq!(refreshed_status.live.next_index, 10);
        assert_eq!(refreshed_status.durable.commit_index, 8);
        assert_eq!(refreshed_status.durable.applied_index, 8);
        assert_eq!(refreshed_status.durable.next_index, 9);

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 7);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 13);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 13);
        assert_eq!(after_reject.live.commit_index, 8);
        assert_eq!(after_reject.live.applied_index, 8);
        assert_eq!(after_reject.live.next_index, 9);
        assert_eq!(after_reject.live.uncommitted_entry_count, 0);
        assert_eq!(after_reject.durable.commit_index, 8);
        assert_eq!(after_reject.durable.applied_index, 8);
        assert_eq!(after_reject.durable.next_index, 9);
        assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
        assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
        assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let after_reject_recovery = r.recovery_state();
        assert_eq!(after_reject_recovery.term, 7);
        assert_eq!(after_reject_recovery.snapshot.snapshot_id, 13);
        let resumed_after_reject =
            RaftReplicator::resume_as_follower(3, after_reject_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_reject.status_snapshot().live,
            after_reject.durable
        );
        assert_eq!(
            after_reject_recovery.progress_as_follower().unwrap(),
            after_reject.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 13);
    }

    #[test]
    fn repair_phase_advanced_snapshot_role_change_discards_only_fresh_suffix() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 41);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 41);
        assert_eq!(before_role_change.live.commit_index, 8);
        assert_eq!(before_role_change.live.applied_index, 8);
        assert_eq!(before_role_change.live.next_index, 10);
        assert_eq!(before_role_change.durable.next_index, 9);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 8);
        assert_eq!(after_role_change.live.applied_index, 8);
        assert_eq!(after_role_change.live.next_index, 9);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 41);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 8);
        assert_eq!(after_role_change.durable.applied_index, 8);
        assert_eq!(after_role_change.durable.next_index, 9);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 41);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 41);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 41);
    }

    #[test]
    fn repair_phase_advanced_snapshot_second_refresh_preserves_identity_through_commit_and_apply() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.live.commit_index, 8);
        assert_eq!(refreshed_status.live.applied_index, 8);
        assert_eq!(refreshed_status.live.next_index, 10);
        assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
        assert_eq!(refreshed_status.durable.commit_index, 8);
        assert_eq!(refreshed_status.durable.applied_index, 8);
        assert_eq!(refreshed_status.durable.next_index, 9);
        assert_eq!(refreshed_status.durable.uncommitted_entry_count, 0);
        assert_eq!(refreshed_status.recovery_gap.next_index_gap, 1);
        assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 13);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_status.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            refreshed_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 13);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 13);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.snapshot.snapshot_id, 13);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 13);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 13);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.snapshot.snapshot_id, 13);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 13);
    }

    #[test]
    fn repair_phase_second_refresh_still_allows_later_advanced_snapshot_replacement() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 13);
        assert_eq!(refreshed_status.live.commit_index, 8);
        assert_eq!(refreshed_status.live.applied_index, 8);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.live.uncommitted_entry_count, 2);
        assert_eq!(refreshed_status.durable.next_index, 9);
        assert_eq!(refreshed_status.recovery_gap.next_index_gap, 2);
        assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 2);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });

        let replaced_status = r.status_snapshot();
        assert!(replaced_status.has_speculative_tail());
        assert_eq!(replaced_status.live.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.durable.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.live.commit_index, 9);
        assert_eq!(replaced_status.live.applied_index, 9);
        assert_eq!(replaced_status.live.next_index, 11);
        assert_eq!(replaced_status.live.uncommitted_entry_count, 1);
        assert_eq!(replaced_status.durable.commit_index, 9);
        assert_eq!(replaced_status.durable.applied_index, 9);
        assert_eq!(replaced_status.durable.next_index, 10);
        assert_eq!(replaced_status.durable.uncommitted_entry_count, 0);
        assert_eq!(replaced_status.recovery_gap.next_index_gap, 1);
        assert_eq!(replaced_status.recovery_gap.uncommitted_entry_gap, 1);

        let replaced_recovery = r.recovery_state();
        assert_eq!(replaced_recovery.snapshot.snapshot_id, 53);
        let resumed_during_replaced_repair =
            RaftReplicator::resume_as_follower(3, replaced_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_replaced_repair.status_snapshot().live,
            replaced_status.durable
        );
        assert_eq!(
            replaced_recovery.progress_as_follower().unwrap(),
            replaced_status.durable
        );

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 53);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 53);

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 53);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 53);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 53);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });

        let replaced_status = r.status_snapshot();
        assert!(replaced_status.has_speculative_tail());
        assert_eq!(replaced_status.live.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.durable.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.live.commit_index, 9);
        assert_eq!(replaced_status.live.applied_index, 9);
        assert_eq!(replaced_status.live.next_index, 11);
        assert_eq!(replaced_status.live.uncommitted_entry_count, 1);
        assert_eq!(replaced_status.durable.commit_index, 9);
        assert_eq!(replaced_status.durable.applied_index, 9);
        assert_eq!(replaced_status.durable.next_index, 10);
        assert_eq!(replaced_status.durable.uncommitted_entry_count, 0);
        assert_eq!(replaced_status.recovery_gap.next_index_gap, 1);
        assert_eq!(replaced_status.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 9);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 10);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 53);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 9);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 10);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 53);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 53);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 53);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_updates_identity_without_perturbing_gap(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });

        let replaced_status = r.status_snapshot();
        assert!(replaced_status.has_speculative_tail());
        let replaced_gap = replaced_status.recovery_gap.clone();
        assert_eq!(replaced_status.live.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.durable.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.live.commit_index, 9);
        assert_eq!(replaced_status.live.applied_index, 9);
        assert_eq!(replaced_status.live.next_index, 11);
        assert_eq!(replaced_status.live.uncommitted_entry_count, 1);
        assert_eq!(replaced_status.durable.commit_index, 9);
        assert_eq!(replaced_status.durable.applied_index, 9);
        assert_eq!(replaced_status.durable.next_index, 10);
        assert_eq!(replaced_status.durable.uncommitted_entry_count, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.term, replaced_status.live.term);
        assert_eq!(
            refreshed_status.live.commit_index,
            replaced_status.live.commit_index
        );
        assert_eq!(
            refreshed_status.live.applied_index,
            replaced_status.live.applied_index
        );
        assert_eq!(
            refreshed_status.live.next_index,
            replaced_status.live.next_index
        );
        assert_eq!(
            refreshed_status.live.uncommitted_entry_count,
            replaced_status.live.uncommitted_entry_count
        );
        assert_eq!(
            refreshed_status.durable.commit_index,
            replaced_status.durable.commit_index
        );
        assert_eq!(
            refreshed_status.durable.applied_index,
            replaced_status.durable.applied_index
        );
        assert_eq!(
            refreshed_status.durable.next_index,
            replaced_status.durable.next_index
        );
        assert_eq!(
            refreshed_status.durable.uncommitted_entry_count,
            replaced_status.durable.uncommitted_entry_count
        );
        assert_eq!(refreshed_status.recovery_gap, replaced_gap);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 47);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_status.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            refreshed_status.durable
        );

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_survives_role_change_tail_discard()
    {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.commit_index, 9);
        assert_eq!(refreshed_status.live.applied_index, 9);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.durable.next_index, 10);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 9);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 10);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 47);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 9);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 10);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 47);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_preserves_identity_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.commit_index, 9);
        assert_eq!(refreshed_status.live.applied_index, 9);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
        assert_eq!(refreshed_status.durable.commit_index, 9);
        assert_eq!(refreshed_status.durable.applied_index, 9);
        assert_eq!(refreshed_status.durable.next_index, 10);
        assert_eq!(refreshed_status.durable.uncommitted_entry_count, 0);
        assert_eq!(refreshed_status.recovery_gap.next_index_gap, 1);
        assert_eq!(refreshed_status.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 47);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_status.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            refreshed_status.durable
        );

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.snapshot.snapshot_id, 47);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.snapshot.snapshot_id, 47);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_collapses_cleanly_on_newer_leader_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.durable.next_index, 10);

        r.become_follower(7);

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.role, Role::Follower);
        assert_eq!(after_rejection.live.term, 7);
        assert_eq!(after_rejection.live.commit_index, 9);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 10);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 47);
        assert_eq!(after_rejection.durable.role, Role::Follower);
        assert_eq!(after_rejection.durable.term, 7);
        assert_eq!(after_rejection.durable.commit_index, 9);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 10);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 47);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_rejection.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_survives_newer_leader_rejection_and_later_repair(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.durable.next_index, 10);

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.term, 7);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 47);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_rejection.live.commit_index, 9);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 10);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.commit_index, 9);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 10);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.term, 7);
        assert_eq!(after_repair.live.commit_index, 10);
        assert_eq!(after_repair.live.applied_index, 9);
        assert_eq!(after_repair.live.next_index, 12);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 47);
        assert_eq!(after_repair.durable.commit_index, 10);
        assert_eq!(after_repair.durable.applied_index, 9);
        assert_eq!(after_repair.durable.next_index, 11);
        assert_eq!(after_repair.durable.uncommitted_entry_count, 0);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 7);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 47);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            after_repair.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            after_repair.durable
        );

        r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.live.commit_index, 11);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 12);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);

        r.mark_applied(11);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 7);
        assert_eq!(applied_status.live.applied_index, 11);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 7);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 47);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_refresh_updates_identity_without_perturbing_gap(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 7);
        assert_eq!(repair_status.live.commit_index, 10);
        assert_eq!(repair_status.live.applied_index, 9);
        assert_eq!(repair_status.live.next_index, 12);
        assert_eq!(repair_status.live.uncommitted_entry_count, 1);
        assert_eq!(repair_status.durable.commit_index, 10);
        assert_eq!(repair_status.durable.applied_index, 9);
        assert_eq!(repair_status.durable.next_index, 11);
        assert_eq!(repair_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);
        let repair_gap = r.recovery_progress_gap();

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let refreshed_repair_status = r.status_snapshot();
        assert!(refreshed_repair_status.has_speculative_tail());
        assert_eq!(refreshed_repair_status.recovery_gap, repair_gap);
        assert_eq!(refreshed_repair_status.live.term, repair_status.live.term);
        assert_eq!(
            refreshed_repair_status.live.commit_index,
            repair_status.live.commit_index
        );
        assert_eq!(
            refreshed_repair_status.live.applied_index,
            repair_status.live.applied_index
        );
        assert_eq!(
            refreshed_repair_status.live.next_index,
            repair_status.live.next_index
        );
        assert_eq!(
            refreshed_repair_status.live.uncommitted_entry_count,
            repair_status.live.uncommitted_entry_count
        );
        assert_eq!(
            refreshed_repair_status.durable.commit_index,
            repair_status.durable.commit_index
        );
        assert_eq!(
            refreshed_repair_status.durable.applied_index,
            repair_status.durable.applied_index
        );
        assert_eq!(
            refreshed_repair_status.durable.next_index,
            repair_status.durable.next_index
        );
        assert_eq!(
            refreshed_repair_status.durable.uncommitted_entry_count,
            repair_status.durable.uncommitted_entry_count
        );
        assert_eq!(refreshed_repair_status.live.snapshot.snapshot_id, 59);
        assert_eq!(refreshed_repair_status.durable.snapshot.snapshot_id, 59);

        let refreshed_repair_recovery = r.recovery_state();
        assert_eq!(refreshed_repair_recovery.term, 7);
        assert_eq!(refreshed_repair_recovery.snapshot.snapshot_id, 59);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            refreshed_repair_status.durable
        );
        assert_eq!(
            refreshed_repair_recovery.progress_as_follower().unwrap(),
            refreshed_repair_status.durable
        );

        r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.live.commit_index, 11);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 12);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 7);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 59);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(11);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 7);
        assert_eq!(applied_status.live.applied_index, 11);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 7);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.term, 7);
        assert_eq!(before_role_change.live.commit_index, 10);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 12);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 47);
        assert_eq!(before_role_change.durable.commit_index, 10);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 11);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 47);

        r.become_candidate(8);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 8);
        assert_eq!(after_role_change.live.commit_index, 10);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 11);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 47);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 8);
        assert_eq!(after_role_change.durable.commit_index, 10);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 11);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 47);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 8);
        assert_eq!(recovery.snapshot.snapshot_id, 47);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_collapses_cleanly(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.term, 7);
        assert_eq!(refreshed_status.live.commit_index, 10);
        assert_eq!(refreshed_status.live.applied_index, 9);
        assert_eq!(refreshed_status.live.next_index, 12);
        assert_eq!(refreshed_status.live.uncommitted_entry_count, 1);
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 59);

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.term, 8);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.live.commit_index, 10);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 11);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.commit_index, 10);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 11);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 8);
        assert_eq!(recovery.snapshot.snapshot_id, 59);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_rejection.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.term, 8);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_rejection.live.commit_index, 10);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 11);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.durable.commit_index, 10);
        assert_eq!(after_rejection.durable.applied_index, 9);
        assert_eq!(after_rejection.durable.next_index, 11);
        assert_eq!(after_rejection.durable.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.term, 8);
        assert_eq!(after_repair.live.commit_index, 11);
        assert_eq!(after_repair.live.applied_index, 9);
        assert_eq!(after_repair.live.next_index, 13);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 59);
        assert_eq!(after_repair.durable.commit_index, 11);
        assert_eq!(after_repair.durable.applied_index, 9);
        assert_eq!(after_repair.durable.next_index, 12);
        assert_eq!(after_repair.durable.uncommitted_entry_count, 0);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 8);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            after_repair.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            after_repair.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 8);
        assert_eq!(committed_status.live.commit_index, 12);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 13);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 8);
        assert_eq!(before_role_change.live.commit_index, 11);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 13);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 59);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 8);
        assert_eq!(before_role_change.durable.commit_index, 11);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 12);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(9);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 9);
        assert_eq!(after_role_change.live.commit_index, 11);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 12);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 59);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 9);
        assert_eq!(after_role_change.durable.commit_index, 11);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 12);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 59);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 9);
        assert_eq!(recovery.snapshot.snapshot_id, 59);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_retires_cleanly_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_commit = r.status_snapshot();
        assert!(before_commit.has_speculative_tail());
        assert_eq!(before_commit.live.term, 8);
        assert_eq!(before_commit.live.commit_index, 11);
        assert_eq!(before_commit.live.applied_index, 9);
        assert_eq!(before_commit.live.next_index, 13);
        assert_eq!(before_commit.live.uncommitted_entry_count, 1);
        assert_eq!(before_commit.live.snapshot.snapshot_id, 59);
        assert_eq!(before_commit.durable.term, 8);
        assert_eq!(before_commit.durable.commit_index, 11);
        assert_eq!(before_commit.durable.applied_index, 9);
        assert_eq!(before_commit.durable.next_index, 12);
        assert_eq!(before_commit.durable.uncommitted_entry_count, 0);
        assert_eq!(before_commit.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_commit.recovery_gap.next_index_gap, 1);
        assert_eq!(before_commit.recovery_gap.uncommitted_entry_gap, 1);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 8);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 59);
        let resumed_during_rerepair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_rerepair.status_snapshot().live,
            before_commit.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            before_commit.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 8);
        assert_eq!(committed_status.live.commit_index, 12);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 13);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.term, 8);
        assert_eq!(committed_status.durable.commit_index, 12);
        assert_eq!(committed_status.durable.applied_index, 9);
        assert_eq!(committed_status.durable.next_index, 13);
        assert_eq!(committed_status.durable.uncommitted_entry_count, 0);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.commit_index, 12);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.next_index, 13);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 59);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 8);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 59);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 59);

        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 58,
        });
        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_rerepair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_rerepair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 59);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 59);

        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 57,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 59);

        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 56,
        });
        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 59);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 59);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 59);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 8);
        assert_eq!(after_refresh.live.commit_index, 11);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 13);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 61);
        assert_eq!(after_refresh.durable.term, 8);
        assert_eq!(after_refresh.durable.commit_index, 11);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 12);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refresh_recovery = r.recovery_state();
        assert_eq!(refresh_recovery.term, 8);
        assert_eq!(refresh_recovery.snapshot.snapshot_id, 61);
        let resumed_during_refresh_rerepair =
            RaftReplicator::resume_as_follower(3, refresh_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refresh_rerepair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refresh_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(8, 12, 8, Vec::new(), 12)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

        r.mark_applied(12);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 8);
        assert_eq!(applied_status.live.applied_index, 12);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 8);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 8);
        assert_eq!(before_role_change.live.commit_index, 11);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 13);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 8);
        assert_eq!(before_role_change.durable.commit_index, 11);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 12);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(9);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 9);
        assert_eq!(after_role_change.live.commit_index, 11);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 12);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 61);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 9);
        assert_eq!(after_role_change.durable.commit_index, 11);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 12);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 9);
        assert_eq!(recovery.snapshot.snapshot_id, 61);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_still_collapses_cleanly_on_newer_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let before_rejection = r.status_snapshot();
        assert!(before_rejection.has_speculative_tail());
        assert_eq!(before_rejection.live.snapshot.snapshot_id, 61);
        assert_eq!(before_rejection.durable.snapshot.snapshot_id, 61);

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.role, Role::Follower);
        assert_eq!(after_rejection.live.term, 9);
        assert_eq!(after_rejection.live.commit_index, 11);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 12);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 61);
        assert!(after_rejection.live.has_committed_entries_pending_apply);
        assert_eq!(after_rejection.durable, after_rejection.live);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let rejection_recovery = r.recovery_state();
        assert_eq!(rejection_recovery.term, 9);
        assert_eq!(rejection_recovery.snapshot.snapshot_id, 61);
        let resumed_after_rejection =
            RaftReplicator::resume_as_follower(3, rejection_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_rejection.status_snapshot().live,
            after_rejection.live
        );
        assert_eq!(
            rejection_recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let collapsed_status = r.status_snapshot();
        assert!(collapsed_status.is_restart_equivalent());
        assert_eq!(collapsed_status.live.term, 9);
        assert_eq!(collapsed_status.live.commit_index, 11);
        assert_eq!(collapsed_status.live.applied_index, 9);
        assert_eq!(collapsed_status.live.next_index, 12);
        assert_eq!(collapsed_status.live.uncommitted_entry_count, 0);
        assert_eq!(collapsed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(collapsed_status.durable, collapsed_status.live);

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.commit_index, 12);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 14);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.term, 9);
        assert_eq!(repaired_status.durable.commit_index, 12);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 13);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 9);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 61);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 61);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 61);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 61);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_and_retires_cleanly(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let collapsed_status = r.status_snapshot();
        assert!(collapsed_status.is_restart_equivalent());
        assert_eq!(collapsed_status.live.term, 10);
        assert_eq!(collapsed_status.live.commit_index, 12);
        assert_eq!(collapsed_status.live.applied_index, 9);
        assert_eq!(collapsed_status.live.next_index, 13);
        assert_eq!(collapsed_status.live.uncommitted_entry_count, 0);
        assert_eq!(collapsed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(collapsed_status.durable, collapsed_status.live);

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.commit_index, 13);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 15);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.term, 10);
        assert_eq!(repaired_status.durable.commit_index, 13);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 14);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 10);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 67);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 67);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 67);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 67);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 67);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 67);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 67);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 67);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_retires_cleanly_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.commit_index, 13);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 15);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.term, 10);
        assert_eq!(repaired_status.durable.commit_index, 13);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 14);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        let repaired_recovery = r.recovery_state();
        assert_eq!(repaired_recovery.term, 10);
        assert_eq!(repaired_recovery.snapshot.snapshot_id, 67);
        let resumed_during_repair =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 67);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 67);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 67);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 67);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 10);
        assert_eq!(after_refresh.live.commit_index, 13);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 15);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 71);
        assert_eq!(after_refresh.durable.term, 10);
        assert_eq!(after_refresh.durable.commit_index, 13);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 14);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.term, 10);
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 71);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.commit_index, 14);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 15);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 10);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 71);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.commit_index, 14);
        assert_eq!(applied_status.live.applied_index, 14);
        assert_eq!(applied_status.live.next_index, 15);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 71);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 10);
        assert_eq!(before_role_change.live.commit_index, 13);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 15);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 10);
        assert_eq!(before_role_change.durable.commit_index, 13);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 14);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(11);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 11);
        assert_eq!(after_role_change.live.commit_index, 13);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 14);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 71);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 11);
        assert_eq!(after_role_change.durable.commit_index, 13);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 14);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 11);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 71);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_still_collapses_cleanly_on_newer_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let before_rejection = r.status_snapshot();
        assert!(before_rejection.has_speculative_tail());
        assert_eq!(before_rejection.live.snapshot.snapshot_id, 71);
        assert_eq!(before_rejection.durable.snapshot.snapshot_id, 71);

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.role, Role::Follower);
        assert_eq!(after_rejection.live.term, 11);
        assert_eq!(after_rejection.live.commit_index, 13);
        assert_eq!(after_rejection.live.applied_index, 9);
        assert_eq!(after_rejection.live.next_index, 14);
        assert_eq!(after_rejection.live.uncommitted_entry_count, 0);
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 71);
        assert!(after_rejection.live.has_committed_entries_pending_apply);
        assert_eq!(after_rejection.durable, after_rejection.live);
        assert_eq!(after_rejection.recovery_gap.next_index_gap, 0);
        assert_eq!(after_rejection.recovery_gap.uncommitted_entry_gap, 0);

        let rejection_recovery = r.recovery_state();
        assert_eq!(rejection_recovery.term, 11);
        assert_eq!(rejection_recovery.snapshot.snapshot_id, 71);
        let resumed_after_rejection =
            RaftReplicator::resume_as_follower(3, rejection_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_rejection.status_snapshot().live,
            after_rejection.live
        );
        assert_eq!(
            rejection_recovery.progress_as_follower().unwrap(),
            after_rejection.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 71);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 71);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 12,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 14,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 14,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 10);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 71);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_rejection = r.status_snapshot();
        assert!(after_rejection.is_restart_equivalent());
        assert_eq!(after_rejection.live.snapshot.snapshot_id, 71);
        assert_eq!(after_rejection.live.term, 11);
        assert_eq!(after_rejection.live.commit_index, 13);
        assert_eq!(after_rejection.live.next_index, 14);

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let after_repair = r.status_snapshot();
        assert!(!after_repair.is_restart_equivalent());
        assert!(after_repair.has_speculative_tail());
        assert_eq!(after_repair.live.role, Role::Follower);
        assert_eq!(after_repair.live.term, 11);
        assert_eq!(after_repair.live.commit_index, 14);
        assert_eq!(after_repair.live.applied_index, 9);
        assert_eq!(after_repair.live.next_index, 16);
        assert_eq!(after_repair.live.uncommitted_entry_count, 1);
        assert_eq!(after_repair.live.snapshot.snapshot_id, 71);
        assert_eq!(after_repair.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_repair.durable.commit_index, 14);
        assert_eq!(after_repair.durable.next_index, 15);
        assert_eq!(after_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(after_repair.recovery_gap.uncommitted_entry_gap, 1);
        assert!(after_repair.live.has_committed_entries_pending_apply);

        let repair_recovery = r.recovery_state();
        assert_eq!(repair_recovery.term, 11);
        assert_eq!(repair_recovery.snapshot.snapshot_id, 71);
        assert_eq!(repair_recovery.commit_index(), 14);
        assert_eq!(repair_recovery.next_index(), 15);
        let resumed_after_repair =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_repair.status_snapshot().live,
            after_repair.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            after_repair.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 11);
        assert_eq!(before_role_change.live.commit_index, 14);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 16);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 11);
        assert_eq!(before_role_change.durable.commit_index, 14);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 15);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(12);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 12);
        assert_eq!(after_role_change.live.commit_index, 14);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 15);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 71);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 12);
        assert_eq!(after_role_change.durable.commit_index, 14);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 15);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 71);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 12);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 71);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_and_retires_cleanly(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let during_repair = r.status_snapshot();
        assert!(during_repair.has_speculative_tail());
        assert_eq!(during_repair.live.snapshot.snapshot_id, 71);
        assert_eq!(during_repair.durable.snapshot.snapshot_id, 71);
        assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let after_commit = r.status_snapshot();
        assert!(after_commit.is_restart_equivalent());
        assert_eq!(after_commit.live.role, Role::Follower);
        assert_eq!(after_commit.live.term, 11);
        assert_eq!(after_commit.live.commit_index, 15);
        assert_eq!(after_commit.live.applied_index, 9);
        assert_eq!(after_commit.live.next_index, 16);
        assert_eq!(after_commit.live.uncommitted_entry_count, 0);
        assert_eq!(after_commit.live.snapshot.snapshot_id, 71);
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert_eq!(after_commit.durable, after_commit.live);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        r.mark_applied(15);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.role, Role::Follower);
        assert_eq!(after_apply.live.term, 11);
        assert_eq!(after_apply.live.commit_index, 15);
        assert_eq!(after_apply.live.applied_index, 15);
        assert_eq!(after_apply.live.next_index, 16);
        assert_eq!(after_apply.live.uncommitted_entry_count, 0);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 71);
        assert!(!after_apply.live.has_committed_entries_pending_apply);
        assert_eq!(after_apply.durable, after_apply.live);
        assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
        assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 71);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_apply.live);
        assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 71);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 71);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 71);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 71);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 71);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 71);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_gap_shape_and_retires_on_new_identity(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();

        let before_refresh = r.status_snapshot();
        assert!(before_refresh.has_speculative_tail());
        assert_eq!(before_refresh.live.snapshot.snapshot_id, 71);
        assert_eq!(before_refresh.durable.snapshot.snapshot_id, 71);
        assert_eq!(before_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(before_refresh.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let after_refresh = r.status_snapshot();
        assert!(after_refresh.has_speculative_tail());
        assert_eq!(after_refresh.live.term, 11);
        assert_eq!(after_refresh.live.commit_index, 14);
        assert_eq!(after_refresh.live.applied_index, 9);
        assert_eq!(after_refresh.live.next_index, 16);
        assert_eq!(after_refresh.live.uncommitted_entry_count, 1);
        assert_eq!(after_refresh.live.snapshot.snapshot_id, 73);
        assert_eq!(after_refresh.durable.term, 11);
        assert_eq!(after_refresh.durable.commit_index, 14);
        assert_eq!(after_refresh.durable.applied_index, 9);
        assert_eq!(after_refresh.durable.next_index, 15);
        assert_eq!(after_refresh.durable.uncommitted_entry_count, 0);
        assert_eq!(after_refresh.durable.snapshot.snapshot_id, 73);
        assert_eq!(after_refresh.recovery_gap.next_index_gap, 1);
        assert_eq!(after_refresh.recovery_gap.uncommitted_entry_gap, 1);

        let refreshed_recovery = r.recovery_state();
        assert_eq!(refreshed_recovery.term, 11);
        assert_eq!(refreshed_recovery.snapshot.snapshot_id, 73);
        let resumed_during_refreshed_repair =
            RaftReplicator::resume_as_follower(3, refreshed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_refreshed_repair.status_snapshot().live,
            after_refresh.durable
        );
        assert_eq!(
            refreshed_recovery.progress_as_follower().unwrap(),
            after_refresh.durable
        );

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 11);
        assert_eq!(committed_status.live.commit_index, 15);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 16);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 11);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 73);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 11);
        assert_eq!(applied_status.live.commit_index, 15);
        assert_eq!(applied_status.live.applied_index, 15);
        assert_eq!(applied_status.live.next_index, 16);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 11);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 73);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_role_handoff_discards_only_fresh_tail(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.role, Role::Follower);
        assert_eq!(before_role_change.live.term, 11);
        assert_eq!(before_role_change.live.commit_index, 14);
        assert_eq!(before_role_change.live.applied_index, 9);
        assert_eq!(before_role_change.live.next_index, 16);
        assert_eq!(before_role_change.live.uncommitted_entry_count, 1);
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 73);
        assert_eq!(before_role_change.durable.role, Role::Follower);
        assert_eq!(before_role_change.durable.term, 11);
        assert_eq!(before_role_change.durable.commit_index, 14);
        assert_eq!(before_role_change.durable.applied_index, 9);
        assert_eq!(before_role_change.durable.next_index, 15);
        assert_eq!(before_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 73);
        assert_eq!(before_role_change.recovery_gap.next_index_gap, 1);
        assert_eq!(before_role_change.recovery_gap.uncommitted_entry_gap, 1);

        r.become_candidate(12);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 12);
        assert_eq!(after_role_change.live.commit_index, 14);
        assert_eq!(after_role_change.live.applied_index, 9);
        assert_eq!(after_role_change.live.next_index, 15);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 73);
        assert!(after_role_change.live.has_committed_entries_pending_apply);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 12);
        assert_eq!(after_role_change.durable.commit_index, 14);
        assert_eq!(after_role_change.durable.applied_index, 9);
        assert_eq!(after_role_change.durable.next_index, 15);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 73);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let role_change_recovery = r.recovery_state();
        assert_eq!(role_change_recovery.term, 12);
        assert_eq!(role_change_recovery.snapshot.snapshot_id, 73);
        let resumed_after_role_change =
            RaftReplicator::resume_as_follower(3, role_change_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_role_change.status_snapshot().live,
            after_role_change.durable
        );
        assert_eq!(
            role_change_recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_and_retires_cleanly(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let during_repair = r.status_snapshot();
        assert!(during_repair.has_speculative_tail());
        assert_eq!(during_repair.live.snapshot.snapshot_id, 73);
        assert_eq!(during_repair.durable.snapshot.snapshot_id, 73);
        assert_eq!(during_repair.recovery_gap.next_index_gap, 1);
        assert_eq!(during_repair.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let after_commit = r.status_snapshot();
        assert!(after_commit.is_restart_equivalent());
        assert_eq!(after_commit.live.role, Role::Follower);
        assert_eq!(after_commit.live.term, 11);
        assert_eq!(after_commit.live.commit_index, 15);
        assert_eq!(after_commit.live.applied_index, 9);
        assert_eq!(after_commit.live.next_index, 16);
        assert_eq!(after_commit.live.uncommitted_entry_count, 0);
        assert_eq!(after_commit.live.snapshot.snapshot_id, 73);
        assert!(after_commit.live.has_committed_entries_pending_apply);
        assert_eq!(after_commit.durable, after_commit.live);
        assert_eq!(after_commit.recovery_gap.next_index_gap, 0);
        assert_eq!(after_commit.recovery_gap.uncommitted_entry_gap, 0);

        r.mark_applied(15);

        let after_apply = r.status_snapshot();
        assert!(after_apply.is_restart_equivalent());
        assert_eq!(after_apply.live.role, Role::Follower);
        assert_eq!(after_apply.live.term, 11);
        assert_eq!(after_apply.live.commit_index, 15);
        assert_eq!(after_apply.live.applied_index, 15);
        assert_eq!(after_apply.live.next_index, 16);
        assert_eq!(after_apply.live.uncommitted_entry_count, 0);
        assert_eq!(after_apply.live.snapshot.snapshot_id, 73);
        assert!(!after_apply.live.has_committed_entries_pending_apply);
        assert_eq!(after_apply.durable, after_apply.live);
        assert_eq!(after_apply.recovery_gap.next_index_gap, 0);
        assert_eq!(after_apply.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 73);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_apply.live);
        assert_eq!(recovery.progress_as_follower().unwrap(), after_apply.live);
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_refresh_rejection_survives_later_repair_refresh_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 71,
        });

        let err = r
            .append_entries_from_leader(
                11,
                99,
                10,
                vec![LogEntry {
                    term: 11,
                    index: 100,
                    payload: vec![150].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            11,
            13,
            10,
            vec![
                LogEntry {
                    term: 11,
                    index: 14,
                    payload: vec![151].into(),
                },
                LogEntry {
                    term: 11,
                    index: 15,
                    payload: vec![152].into(),
                },
            ],
            14,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 73,
        });

        let baseline = r.status_snapshot();
        assert!(baseline.has_speculative_tail());
        assert_eq!(baseline.live.snapshot.snapshot_id, 73);
        assert_eq!(baseline.durable.snapshot.snapshot_id, 73);
        assert_eq!(baseline.recovery_gap.next_index_gap, 1);
        assert_eq!(baseline.recovery_gap.uncommitted_entry_gap, 1);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1000,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1001,
        });

        let during_repair = r.status_snapshot();
        assert_eq!(during_repair, baseline);

        r.append_entries_from_leader(11, 15, 11, Vec::new(), 15)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(committed_status.is_restart_equivalent());
        assert_eq!(committed_status.live.snapshot.snapshot_id, 73);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1002,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1003,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1004,
        });

        let after_commit_stale = r.status_snapshot();
        assert_eq!(after_commit_stale, committed_status);

        r.mark_applied(15);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 73);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 1005,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 1006,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 15,
            last_included_term: 5,
            snapshot_id: 1007,
        });

        let after_apply_stale = r.status_snapshot();
        assert_eq!(after_apply_stale, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 11);
        assert_eq!(recovery.snapshot.snapshot_id, 73);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 73);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 991,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 990,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 989,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 988,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 987,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 986,
        });
        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_repair_refresh_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 67,
        });

        let err = r
            .append_entries_from_leader(
                10,
                99,
                9,
                vec![LogEntry {
                    term: 10,
                    index: 100,
                    payload: vec![140].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            10,
            12,
            9,
            vec![
                LogEntry {
                    term: 10,
                    index: 13,
                    payload: vec![141].into(),
                },
                LogEntry {
                    term: 10,
                    index: 14,
                    payload: vec![142].into(),
                },
            ],
            13,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 10);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 67);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 991,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 990,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(10, 14, 10, Vec::new(), 14)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 10);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 67);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 989,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 988,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(14);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 10);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 67);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 987,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 986,
        });
        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 67);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_and_retires_cleanly(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.commit_index, 12);
        assert_eq!(repaired_status.live.applied_index, 9);
        assert_eq!(repaired_status.live.next_index, 14);
        assert_eq!(repaired_status.live.uncommitted_entry_count, 1);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.term, 9);
        assert_eq!(repaired_status.durable.commit_index, 12);
        assert_eq!(repaired_status.durable.applied_index, 9);
        assert_eq!(repaired_status.durable.next_index, 13);
        assert_eq!(repaired_status.durable.uncommitted_entry_count, 0);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.recovery_gap.next_index_gap, 1);
        assert_eq!(repaired_status.recovery_gap.uncommitted_entry_gap, 1);

        r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
            .unwrap();

        let committed_status = r.status_snapshot();
        assert!(!committed_status.has_speculative_tail());
        assert_eq!(committed_status.live.term, 9);
        assert_eq!(committed_status.live.commit_index, 13);
        assert_eq!(committed_status.live.applied_index, 9);
        assert_eq!(committed_status.live.next_index, 14);
        assert_eq!(committed_status.live.uncommitted_entry_count, 0);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable, committed_status.live);
        assert_eq!(committed_status.recovery_gap.next_index_gap, 0);
        assert_eq!(committed_status.recovery_gap.uncommitted_entry_gap, 0);

        let committed_recovery = r.recovery_state();
        assert_eq!(committed_recovery.term, 9);
        assert_eq!(committed_recovery.snapshot.snapshot_id, 61);
        let resumed_after_commit =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit.status_snapshot().live,
            committed_status.live
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.live
        );

        r.mark_applied(13);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.term, 9);
        assert_eq!(applied_status.live.commit_index, 13);
        assert_eq!(applied_status.live.applied_index, 13);
        assert_eq!(applied_status.live.next_index, 14);
        assert_eq!(applied_status.live.uncommitted_entry_count, 0);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);
        assert_eq!(applied_status.durable, applied_status.live);
        assert_eq!(applied_status.recovery_gap.next_index_gap, 0);
        assert_eq!(applied_status.recovery_gap.uncommitted_entry_gap, 0);

        let applied_recovery = r.recovery_state();
        assert_eq!(applied_recovery.term, 9);
        assert_eq!(applied_recovery.snapshot.snapshot_id, 61);
        let resumed_after_apply =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply.status_snapshot().live,
            applied_status.live
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.live
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_rejection_survives_later_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 59,
        });

        let err = r
            .append_entries_from_leader(
                8,
                99,
                7,
                vec![LogEntry {
                    term: 8,
                    index: 100,
                    payload: vec![120].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            8,
            10,
            7,
            vec![
                LogEntry {
                    term: 8,
                    index: 11,
                    payload: vec![121].into(),
                },
                LogEntry {
                    term: 8,
                    index: 12,
                    payload: vec![122].into(),
                },
            ],
            11,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 61,
        });

        let err = r
            .append_entries_from_leader(
                9,
                99,
                8,
                vec![LogEntry {
                    term: 9,
                    index: 100,
                    payload: vec![130].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            9,
            11,
            8,
            vec![
                LogEntry {
                    term: 9,
                    index: 12,
                    payload: vec![131].into(),
                },
                LogEntry {
                    term: 9,
                    index: 13,
                    payload: vec![132].into(),
                },
            ],
            12,
        )
        .unwrap();

        let repaired_status = r.status_snapshot();
        let repaired_recovery = r.recovery_state();
        let repaired_gap = r.recovery_progress_gap();
        assert!(repaired_status.has_speculative_tail());
        assert_eq!(repaired_status.live.term, 9);
        assert_eq!(repaired_status.live.snapshot.snapshot_id, 61);
        assert_eq!(repaired_status.durable.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 998,
        });
        assert_eq!(r.status_snapshot(), repaired_status);
        assert_eq!(r.recovery_state(), repaired_recovery);
        assert_eq!(r.recovery_progress_gap(), repaired_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repaired_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repaired_status.durable
        );
        assert_eq!(
            repaired_recovery.progress_as_follower().unwrap(),
            repaired_status.durable
        );

        r.append_entries_from_leader(9, 13, 9, Vec::new(), 13)
            .unwrap();

        let committed_status = r.status_snapshot();
        let committed_recovery = r.recovery_state();
        let committed_gap = r.recovery_progress_gap();
        assert!(committed_status.is_restart_equivalent());
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.term, 9);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 61);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 996,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 995,
        });
        assert_eq!(r.status_snapshot(), committed_status);
        assert_eq!(r.recovery_state(), committed_recovery);
        assert_eq!(r.recovery_progress_gap(), committed_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, committed_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            committed_status.durable
        );
        assert_eq!(
            committed_recovery.progress_as_follower().unwrap(),
            committed_status.durable
        );

        r.mark_applied(13);

        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 9);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 61);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 993,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 992,
        });
        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 61);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_post_rejection_repair_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        r.append_entries_from_leader(
            7,
            9,
            6,
            vec![
                LogEntry {
                    term: 7,
                    index: 10,
                    payload: vec![110].into(),
                },
                LogEntry {
                    term: 7,
                    index: 11,
                    payload: vec![111].into(),
                },
            ],
            10,
        )
        .unwrap();

        let repair_status = r.status_snapshot();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.term, 7);
        assert_eq!(repair_status.live.snapshot.snapshot_id, 47);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 47);

        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(7, 11, 7, Vec::new(), 11)
            .unwrap();

        let commit_status = r.status_snapshot();
        assert!(commit_status.is_restart_equivalent());
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert_eq!(commit_status.live.snapshot.snapshot_id, 47);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 47);

        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 46,
        });
        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(11);

        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live, applied_status.durable);
        assert_eq!(applied_status.live.term, 7);
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);

        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 45,
        });
        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_refresh_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 47,
        });

        let refreshed_status = r.status_snapshot();
        assert!(refreshed_status.has_speculative_tail());
        assert_eq!(refreshed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.durable.snapshot.snapshot_id, 47);
        assert_eq!(refreshed_status.live.next_index, 11);
        assert_eq!(refreshed_status.durable.next_index, 10);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 999,
        });
        let after_stale_repair = r.status_snapshot();
        assert_eq!(after_stale_repair, refreshed_status);

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let committed_status = r.status_snapshot();
        assert!(committed_status.live.has_committed_entries_pending_apply);
        assert_eq!(committed_status.live.snapshot.snapshot_id, 47);
        assert_eq!(committed_status.durable.snapshot.snapshot_id, 47);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 46,
        });
        let after_stale_commit = r.status_snapshot();
        assert_eq!(after_stale_commit, committed_status);

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 47);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 47);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 45,
        });
        let after_stale_apply = r.status_snapshot();
        assert_eq!(after_stale_apply, applied_status);

        let recovery = r.recovery_state();
        assert_eq!(recovery.snapshot.snapshot_id, 47);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, applied_status.live);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 47);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_collapses_cleanly_on_newer_leader_rejection(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });

        let replaced_status = r.status_snapshot();
        assert!(replaced_status.has_speculative_tail());
        assert_eq!(replaced_status.live.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.durable.snapshot.snapshot_id, 53);
        assert_eq!(replaced_status.live.commit_index, 9);
        assert_eq!(replaced_status.live.applied_index, 9);
        assert_eq!(replaced_status.live.next_index, 11);
        assert_eq!(replaced_status.live.uncommitted_entry_count, 1);
        assert_eq!(replaced_status.durable.commit_index, 9);
        assert_eq!(replaced_status.durable.applied_index, 9);
        assert_eq!(replaced_status.durable.next_index, 10);
        assert_eq!(replaced_status.durable.uncommitted_entry_count, 0);
        assert_eq!(replaced_status.recovery_gap.next_index_gap, 1);
        assert_eq!(replaced_status.recovery_gap.uncommitted_entry_gap, 1);

        let err = r
            .append_entries_from_leader(
                7,
                99,
                6,
                vec![LogEntry {
                    term: 7,
                    index: 100,
                    payload: vec![100].into(),
                }],
                100,
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::ProposalFailed(_)));

        let after_reject = r.status_snapshot();
        assert!(after_reject.is_restart_equivalent());
        assert_eq!(after_reject.live.term, 7);
        assert_eq!(after_reject.live.snapshot.snapshot_id, 53);
        assert_eq!(after_reject.durable.snapshot.snapshot_id, 53);
        assert_eq!(after_reject.live.commit_index, 9);
        assert_eq!(after_reject.live.applied_index, 9);
        assert_eq!(after_reject.live.next_index, 10);
        assert_eq!(after_reject.live.uncommitted_entry_count, 0);
        assert_eq!(after_reject.durable.commit_index, 9);
        assert_eq!(after_reject.durable.applied_index, 9);
        assert_eq!(after_reject.durable.next_index, 10);
        assert_eq!(after_reject.durable.uncommitted_entry_count, 0);
        assert_eq!(after_reject.recovery_gap.next_index_gap, 0);
        assert_eq!(after_reject.recovery_gap.uncommitted_entry_gap, 0);
        assert!(r.recovery_progress_gap().is_restart_equivalent());

        let after_reject_recovery = r.recovery_state();
        assert_eq!(after_reject_recovery.term, 7);
        assert_eq!(after_reject_recovery.snapshot.snapshot_id, 53);
        let resumed_after_reject =
            RaftReplicator::resume_as_follower(3, after_reject_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_reject.status_snapshot().live,
            after_reject.durable
        );
        assert_eq!(
            after_reject_recovery.progress_as_follower().unwrap(),
            after_reject.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 53);
    }

    #[test]
    fn repair_phase_second_refresh_advanced_replacement_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
                LogEntry {
                    term: 6,
                    index: 10,
                    payload: vec![100].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 6,
            snapshot_id: 53,
        });

        let repair_status = r.status_snapshot();
        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 53);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 53);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 10,
            last_included_term: 5,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(6, 10, 6, Vec::new(), 10)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());
        assert_eq!(commit_status.live.snapshot.snapshot_id, 53);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 53);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 10,
            last_included_term: 5,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(10);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 53);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 53);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 10,
            last_included_term: 5,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 53);
    }

    #[test]
    fn repair_phase_advanced_snapshot_second_refresh_keeps_stale_installs_inert_through_commit_and_apply(
    ) {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });

        let repair_status = r.status_snapshot();
        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 13);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 13);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());
        assert_eq!(commit_status.live.snapshot.snapshot_id, 13);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 13);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 13);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 13);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 13);
    }

    #[test]
    fn repair_phase_advanced_snapshot_second_refresh_survives_role_change_tail_discard() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 13,
        });

        let before_role_change = r.status_snapshot();
        assert!(before_role_change.has_speculative_tail());
        assert_eq!(before_role_change.live.snapshot.snapshot_id, 13);
        assert_eq!(before_role_change.durable.snapshot.snapshot_id, 13);
        assert_eq!(before_role_change.live.commit_index, 8);
        assert_eq!(before_role_change.live.applied_index, 8);
        assert_eq!(before_role_change.live.next_index, 10);
        assert_eq!(before_role_change.durable.next_index, 9);

        r.become_candidate(7);

        let after_role_change = r.status_snapshot();
        assert!(after_role_change.is_restart_equivalent());
        assert_eq!(after_role_change.live.role, Role::Candidate);
        assert_eq!(after_role_change.live.term, 7);
        assert_eq!(after_role_change.live.commit_index, 8);
        assert_eq!(after_role_change.live.applied_index, 8);
        assert_eq!(after_role_change.live.next_index, 9);
        assert_eq!(after_role_change.live.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.live.snapshot.snapshot_id, 13);
        assert_eq!(after_role_change.durable.role, Role::Follower);
        assert_eq!(after_role_change.durable.term, 7);
        assert_eq!(after_role_change.durable.commit_index, 8);
        assert_eq!(after_role_change.durable.applied_index, 8);
        assert_eq!(after_role_change.durable.next_index, 9);
        assert_eq!(after_role_change.durable.uncommitted_entry_count, 0);
        assert_eq!(after_role_change.durable.snapshot.snapshot_id, 13);
        assert_eq!(after_role_change.recovery_gap.next_index_gap, 0);
        assert_eq!(after_role_change.recovery_gap.uncommitted_entry_gap, 0);

        let recovery = r.recovery_state();
        assert_eq!(recovery.term, 7);
        assert_eq!(recovery.snapshot.snapshot_id, 13);
        let resumed = RaftReplicator::resume_as_follower(3, recovery.clone()).unwrap();
        assert_eq!(resumed.status_snapshot().live, after_role_change.durable);
        assert_eq!(
            recovery.progress_as_follower().unwrap(),
            after_role_change.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 13);
    }

    #[test]
    fn repair_phase_advanced_snapshot_refresh_keeps_stale_installs_inert_through_commit_and_apply()
    {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 17,
        });

        let repair_status = r.status_snapshot();
        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 17);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 17);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());
        assert_eq!(commit_status.live.snapshot.snapshot_id, 17);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 17);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 17);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 17);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 17);
    }

    #[test]
    fn repair_phase_advanced_snapshot_stale_installs_remain_noops_during_repair_commit_and_apply() {
        let mut r = RaftReplicator::new(3);
        r.become_follower(5);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 5,
            last_included_term: 4,
            snapshot_id: 11,
        });
        r.append_entries_from_leader(
            5,
            5,
            4,
            vec![
                LogEntry {
                    term: 5,
                    index: 6,
                    payload: vec![6].into(),
                },
                LogEntry {
                    term: 5,
                    index: 7,
                    payload: vec![7].into(),
                },
                LogEntry {
                    term: 5,
                    index: 8,
                    payload: vec![8].into(),
                },
            ],
            7,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 29,
        });
        r.append_entries_from_leader(
            6,
            7,
            5,
            vec![
                LogEntry {
                    term: 6,
                    index: 8,
                    payload: vec![80].into(),
                },
                LogEntry {
                    term: 6,
                    index: 9,
                    payload: vec![90].into(),
                },
            ],
            8,
        )
        .unwrap();
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 6,
            snapshot_id: 41,
        });

        let repair_status = r.status_snapshot();
        let repair_recovery = r.recovery_state();
        let repair_gap = r.recovery_progress_gap();
        assert!(repair_status.has_speculative_tail());
        assert_eq!(repair_status.live.snapshot.snapshot_id, 41);
        assert_eq!(repair_status.durable.snapshot.snapshot_id, 41);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 97,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 98,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 99,
        });

        assert_eq!(r.status_snapshot(), repair_status);
        assert_eq!(r.recovery_state(), repair_recovery);
        assert_eq!(r.recovery_progress_gap(), repair_gap);
        let resumed_during_repair_after_stale =
            RaftReplicator::resume_as_follower(3, repair_recovery.clone()).unwrap();
        assert_eq!(
            resumed_during_repair_after_stale.status_snapshot().live,
            repair_status.durable
        );
        assert_eq!(
            repair_recovery.progress_as_follower().unwrap(),
            repair_status.durable
        );

        r.append_entries_from_leader(6, 9, 6, Vec::new(), 9)
            .unwrap();
        let commit_status = r.status_snapshot();
        let commit_recovery = r.recovery_state();
        let commit_gap = r.recovery_progress_gap();
        assert!(commit_status.live.has_committed_entries_pending_apply);
        assert!(!commit_status.has_speculative_tail());
        assert_eq!(commit_status.live.snapshot.snapshot_id, 41);
        assert_eq!(commit_status.durable.snapshot.snapshot_id, 41);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 107,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 108,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 109,
        });

        assert_eq!(r.status_snapshot(), commit_status);
        assert_eq!(r.recovery_state(), commit_recovery);
        assert_eq!(r.recovery_progress_gap(), commit_gap);
        let resumed_after_commit_stale =
            RaftReplicator::resume_as_follower(3, commit_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_commit_stale.status_snapshot().live,
            commit_status.durable
        );
        assert_eq!(
            commit_recovery.progress_as_follower().unwrap(),
            commit_status.durable
        );

        r.mark_applied(9);
        let applied_status = r.status_snapshot();
        let applied_recovery = r.recovery_state();
        let applied_gap = r.recovery_progress_gap();
        assert!(applied_status.is_restart_equivalent());
        assert_eq!(applied_status.live.snapshot.snapshot_id, 41);
        assert_eq!(applied_status.durable.snapshot.snapshot_id, 41);

        r.install_snapshot(SnapshotMeta {
            last_included_index: 7,
            last_included_term: 5,
            snapshot_id: 117,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 8,
            last_included_term: 5,
            snapshot_id: 118,
        });
        r.install_snapshot(SnapshotMeta {
            last_included_index: 9,
            last_included_term: 5,
            snapshot_id: 119,
        });

        assert_eq!(r.status_snapshot(), applied_status);
        assert_eq!(r.recovery_state(), applied_recovery);
        assert_eq!(r.recovery_progress_gap(), applied_gap);
        let resumed_after_apply_stale =
            RaftReplicator::resume_as_follower(3, applied_recovery.clone()).unwrap();
        assert_eq!(
            resumed_after_apply_stale.status_snapshot().live,
            applied_status.durable
        );
        assert_eq!(
            applied_recovery.progress_as_follower().unwrap(),
            applied_status.durable
        );
        assert_eq!(r.snapshot_meta().snapshot_id, 41);
    }
}
