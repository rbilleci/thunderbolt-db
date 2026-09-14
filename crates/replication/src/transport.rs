use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    path::Path,
    sync::Arc,
    time::Duration,
};

use rustls::{
    pki_types::{pem::PemObject as _, CertificateDer, PrivateKeyDer, ServerName},
    ClientConfig as TlsClientConfig, ClientConnection, RootCertStore,
    ServerConfig as TlsServerConfig, ServerConnection, StreamOwned,
};

use super::{AppendEntriesRequest, AppendEntriesResponse, EngineError, RaftReplicator};

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
) -> Result<Vec<CertificateDer<'static>>, EngineError> {
    let certs = CertificateDer::pem_file_iter(path)
        .map_err(|err| append_entries_error(format!("{label} open failed: {err}")))?
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
) -> Result<PrivateKeyDer<'static>, EngineError> {
    PrivateKeyDer::pem_file_iter(path)
        .map_err(|err| append_entries_error(format!("{label} open failed: {err}")))?
        .next()
        .transpose()
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
