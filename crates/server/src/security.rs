use std::env;
use std::io::{self, ErrorKind};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use gpu_db_protocol::backend::{BackendError, BackendWriter};
use gpu_db_protocol::{
    parse_frontend_message, parse_startup_packet, FrontendMessage, StartupPacket,
};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use rand::{rngs::OsRng, RngCore};
use rustls::{
    pki_types::{pem::PemObject as _, CertificateDer, PrivateKeyDer},
    ServerConfig as TlsServerConfig, ServerConnection, StreamOwned,
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::cancellation::{CancellationRegistry, ConnectionCancellation};
use crate::transport::{read_auth_frame, read_startup_frame, ReadWrite};

const DEFAULT_LISTEN: &str = "127.0.0.1:5432";
const SCRAM_MIN_ITERATIONS: u32 = 4096;
const SCRAM_MAX_ITERATIONS: u32 = 10_000_000;
const SCRAM_MAX_MESSAGE_BYTES: usize = 4096;
const SCRAM_MAX_NONCE_BYTES: usize = 1024;
const SCRAM_KEY_BYTES: usize = 32;
const CANCEL_REQUEST_CODE: u32 = 80_877_102;

/// PostgreSQL cancellation is a one-shot, no-response startup packet. If its identifying code is
/// present but its frame is not exactly 16 bytes, close silently rather than routing the malformed
/// packet through ordinary startup error reporting.
pub(crate) fn is_malformed_cancel_request(frame: &[u8]) -> bool {
    frame
        .get(4..8)
        .and_then(|code| <[u8; 4]>::try_from(code).ok())
        .is_some_and(|code| u32::from_be_bytes(code) == CANCEL_REQUEST_CODE)
        && frame.len() != 16
}

/// Complete process configuration for the canonical product server. Secret-bearing fields do not
/// implement `Debug`, so ordinary configuration errors and panic reports cannot print credentials.
pub struct ServerConfig {
    listen: String,
    security: SecurityConfig,
}

impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let profile = match &self.security {
            SecurityConfig::LocalDev => "local-dev",
            SecurityConfig::Production(_) => "production",
        };
        formatter
            .debug_struct("ServerConfig")
            .field("listen", &self.listen)
            .field("security_profile", &profile)
            .finish()
    }
}

enum SecurityConfig {
    LocalDev,
    Production(ProductionSecurityConfig),
}

struct ProductionSecurityConfig {
    tls_cert: PathBuf,
    tls_key: PathBuf,
    auth_user: String,
    auth_credential: ScramCredential,
}

enum ScramCredential {
    PasswordBootstrap(String),
    Verifier(ScramVerifier),
}

#[derive(Clone)]
struct ScramVerifier {
    iterations: u32,
    salt: Vec<u8>,
    stored_key: [u8; SCRAM_KEY_BYTES],
    server_key: [u8; SCRAM_KEY_BYTES],
}

#[derive(Default)]
struct SecurityInputs {
    profile: Option<String>,
    tls_cert: Option<PathBuf>,
    tls_key: Option<PathBuf>,
    auth_user: Option<String>,
    auth_password: Option<String>,
    auth_scram_verifier: Option<String>,
    auth_scram_verifier_file: Option<PathBuf>,
}

impl SecurityInputs {
    fn from_env() -> Self {
        Self {
            profile: env::var("GPU_DB_SECURITY_PROFILE").ok(),
            tls_cert: env::var("GPU_DB_TLS_CERT").ok().map(PathBuf::from),
            tls_key: env::var("GPU_DB_TLS_KEY").ok().map(PathBuf::from),
            auth_user: env::var("GPU_DB_AUTH_USER").ok(),
            auth_password: env::var("GPU_DB_AUTH_PASSWORD").ok(),
            auth_scram_verifier: env::var("GPU_DB_AUTH_SCRAM_VERIFIER").ok(),
            auth_scram_verifier_file: env::var("GPU_DB_AUTH_SCRAM_VERIFIER_FILE")
                .ok()
                .map(PathBuf::from),
        }
    }

    fn has_material(&self) -> bool {
        self.tls_cert.is_some()
            || self.tls_key.is_some()
            || self.auth_user.is_some()
            || self.auth_password.is_some()
            || self.auth_scram_verifier.is_some()
            || self.auth_scram_verifier_file.is_some()
    }
}

impl ServerConfig {
    pub fn from_env_args<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse(args, SecurityInputs::from_env())
    }

    fn parse<I>(args: I, mut inputs: SecurityInputs) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = args.into_iter();
        let mut listen = DEFAULT_LISTEN.to_string();
        let mut listen_was_set = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--listen" => {
                    if listen_was_set {
                        return Err(String::from("listen address was specified more than once"));
                    }
                    listen = next_value(&mut args, "--listen")?;
                    listen_was_set = true;
                }
                "--security-profile" => {
                    inputs.profile = Some(next_value(&mut args, "--security-profile")?);
                }
                "--tls-cert" => {
                    inputs.tls_cert = Some(PathBuf::from(next_value(&mut args, "--tls-cert")?));
                }
                "--tls-key" => {
                    inputs.tls_key = Some(PathBuf::from(next_value(&mut args, "--tls-key")?));
                }
                "--auth-user" => {
                    inputs.auth_user = Some(next_value(&mut args, "--auth-user")?);
                }
                "--auth-password" => {
                    inputs.auth_password = Some(next_value(&mut args, "--auth-password")?);
                }
                "--auth-scram-verifier" => {
                    inputs.auth_scram_verifier =
                        Some(next_value(&mut args, "--auth-scram-verifier")?);
                }
                "--auth-scram-verifier-file" => {
                    inputs.auth_scram_verifier_file = Some(PathBuf::from(next_value(
                        &mut args,
                        "--auth-scram-verifier-file",
                    )?));
                }
                "-h" | "--help" => return Err(usage()),
                option if option.starts_with('-') => {
                    return Err(format!("unsupported argument: {option}\n{}", usage()));
                }
                positional => {
                    if listen_was_set {
                        return Err(format!(
                            "listen address was specified more than once: {positional}"
                        ));
                    }
                    listen = positional.to_string();
                    listen_was_set = true;
                }
            }
        }
        if listen.is_empty() {
            return Err(String::from("listen address must not be empty"));
        }

        let profile = inputs.profile.as_deref().unwrap_or("local-dev");
        let security = match profile {
            "local-dev" | "development" => {
                if inputs.has_material() {
                    return Err(String::from(
                        "local-dev security profile does not accept TLS or authentication material; select --security-profile production",
                    ));
                }
                SecurityConfig::LocalDev
            }
            "production" => SecurityConfig::Production(production_config(inputs)?),
            other => return Err(format!("unsupported security profile: {other}")),
        };
        Ok(Self { listen, security })
    }

    pub fn listen_address(&self) -> &str {
        &self.listen
    }

    pub(crate) fn into_runtime(self) -> io::Result<(String, RuntimeSecurity)> {
        let runtime = RuntimeSecurity::from_config(self.security)?;
        Ok((self.listen, runtime))
    }
}

fn usage() -> String {
    String::from(
        "usage: gpu-db-engine-server [LISTEN_ADDR | --listen HOST:PORT] [--security-profile local-dev|production --tls-cert PATH --tls-key PATH --auth-user USER (--auth-scram-verifier VERIFIER | --auth-scram-verifier-file PATH | --auth-password PASSWORD)]",
    )
}

fn next_value<I>(args: &mut I, option: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn production_config(mut inputs: SecurityInputs) -> Result<ProductionSecurityConfig, String> {
    Ok(ProductionSecurityConfig {
        tls_cert: inputs.tls_cert.take().ok_or_else(|| {
            String::from("production security profile requires --tls-cert or GPU_DB_TLS_CERT")
        })?,
        tls_key: inputs.tls_key.take().ok_or_else(|| {
            String::from("production security profile requires --tls-key or GPU_DB_TLS_KEY")
        })?,
        auth_user: non_empty_config(inputs.auth_user.take(), "--auth-user or GPU_DB_AUTH_USER")?,
        auth_credential: production_auth_credential(
            inputs.auth_password.take(),
            inputs.auth_scram_verifier.take(),
            inputs.auth_scram_verifier_file.take(),
        )?,
    })
}

fn non_empty_config(value: Option<String>, name: &str) -> Result<String, String> {
    let value = value.ok_or_else(|| format!("production security profile requires {name}"))?;
    if value.is_empty() {
        return Err(format!(
            "production security profile requires non-empty {name}"
        ));
    }
    Ok(value)
}

fn production_auth_credential(
    auth_password: Option<String>,
    auth_scram_verifier: Option<String>,
    auth_scram_verifier_file: Option<PathBuf>,
) -> Result<ScramCredential, String> {
    let configured = [
        auth_password.is_some(),
        auth_scram_verifier.is_some(),
        auth_scram_verifier_file.is_some(),
    ]
    .into_iter()
    .filter(|configured| *configured)
    .count();
    if configured == 0 {
        return Err(String::from(
            "production security profile requires exactly one of --auth-scram-verifier, --auth-scram-verifier-file, or local/test --auth-password bootstrap",
        ));
    }
    if configured > 1 {
        return Err(String::from(
            "production security profile requires only one credential source; do not combine plaintext password and SCRAM verifier inputs",
        ));
    }
    if let Some(password) = auth_password {
        return Ok(ScramCredential::PasswordBootstrap(non_empty_config(
            Some(password),
            "local/test --auth-password or GPU_DB_AUTH_PASSWORD bootstrap",
        )?));
    }
    if let Some(verifier) = auth_scram_verifier {
        return parse_scram_verifier(&non_empty_config(
            Some(verifier),
            "--auth-scram-verifier or GPU_DB_AUTH_SCRAM_VERIFIER",
        )?)
        .map(ScramCredential::Verifier);
    }
    let path = auth_scram_verifier_file.expect("configured verifier file exists");
    let verifier = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read production SCRAM verifier file: {error}"))?;
    parse_scram_verifier(verifier.trim()).map(ScramCredential::Verifier)
}

pub(crate) struct RuntimeSecurity {
    profile: RuntimeProfile,
}

enum RuntimeProfile {
    LocalDev,
    Production {
        tls: Arc<TlsServerConfig>,
        auth_user: String,
        verifier: ScramVerifier,
    },
}

enum PlainStartupAction {
    Ready(ConnectionCancellation),
    UpgradeTls,
    Close,
}

impl RuntimeSecurity {
    fn from_config(config: SecurityConfig) -> io::Result<Self> {
        let profile = match config {
            SecurityConfig::LocalDev => RuntimeProfile::LocalDev,
            SecurityConfig::Production(production) => {
                let tls = Arc::new(load_tls_config(&production)?);
                let verifier = match production.auth_credential {
                    ScramCredential::PasswordBootstrap(password) => {
                        let mut salt = [0_u8; 16];
                        OsRng.fill_bytes(&mut salt);
                        ScramVerifier::from_password_bootstrap(
                            &password,
                            &salt,
                            SCRAM_MIN_ITERATIONS,
                        )
                    }
                    ScramCredential::Verifier(verifier) => verifier,
                };
                RuntimeProfile::Production {
                    tls,
                    auth_user: production.auth_user,
                    verifier,
                }
            }
        };
        Ok(Self { profile })
    }

    /// Negotiate startup on an owned product connection. The returned stream is authenticated and
    /// ready for the one canonical query loop, whether it is plain local-dev TCP or rustls.
    pub(crate) fn accept_blocking(
        &self,
        mut stream: TcpStream,
        cancellations: &Arc<CancellationRegistry>,
    ) -> io::Result<Option<(Box<dyn ReadWrite + Send>, ConnectionCancellation)>> {
        let production = matches!(self.profile, RuntimeProfile::Production { .. });
        match negotiate_plain_startup(&mut stream, production, cancellations)? {
            PlainStartupAction::Ready(cancellation) => Ok(Some((Box::new(stream), cancellation))),
            PlainStartupAction::Close => Ok(None),
            PlainStartupAction::UpgradeTls => {
                let RuntimeProfile::Production {
                    tls,
                    auth_user,
                    verifier,
                } = &self.profile
                else {
                    return Err(io::Error::new(
                        ErrorKind::InvalidInput,
                        "local-dev startup cannot request a TLS upgrade",
                    ));
                };
                let connection = ServerConnection::new(Arc::clone(tls))
                    .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
                let mut tls_stream = StreamOwned::new(connection, stream);
                finish_tls_startup(&mut tls_stream, auth_user, verifier, cancellations).map(
                    |cancellation| {
                        cancellation.map(|cancellation| {
                            (
                                Box::new(tls_stream) as Box<dyn ReadWrite + Send>,
                                cancellation,
                            )
                        })
                    },
                )
            }
        }
    }
}

fn negotiate_plain_startup(
    stream: &mut dyn ReadWrite,
    production: bool,
    cancellations: &Arc<CancellationRegistry>,
) -> io::Result<PlainStartupAction> {
    loop {
        let Some(frame) = read_startup_frame(stream)? else {
            return Ok(PlainStartupAction::Close);
        };
        if is_malformed_cancel_request(&frame) {
            return Ok(PlainStartupAction::Close);
        }
        match parse_startup_or_error(stream, &frame)? {
            StartupPacket::SslRequest if production => {
                stream.write_all(b"S")?;
                stream.flush()?;
                return Ok(PlainStartupAction::UpgradeTls);
            }
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => {
                stream.write_all(b"N")?;
                stream.flush()?;
            }
            StartupPacket::CancelRequest {
                process_id,
                secret_key,
            } => {
                cancellations.cancel(process_id, &secret_key);
                return Ok(PlainStartupAction::Close);
            }
            StartupPacket::Startup { .. } if production => return tls_required(stream),
            StartupPacket::Startup { .. } => {
                let cancellation = cancellations.register();
                write_authentication_ok(stream)?;
                write_startup_ready(stream, cancellation.backend_key())?;
                return Ok(PlainStartupAction::Ready(cancellation));
            }
        }
    }
}

fn finish_tls_startup(
    stream: &mut dyn ReadWrite,
    auth_user: &str,
    verifier: &ScramVerifier,
    cancellations: &Arc<CancellationRegistry>,
) -> io::Result<Option<ConnectionCancellation>> {
    loop {
        let Some(frame) = read_startup_frame(stream)? else {
            return Ok(None);
        };
        if is_malformed_cancel_request(&frame) {
            return Ok(None);
        }
        match parse_startup_or_error(stream, &frame)? {
            StartupPacket::SslRequest => {
                return protocol_failure(stream, "nested SSLRequest is not supported");
            }
            StartupPacket::GssEncRequest => {
                stream.write_all(b"N")?;
                stream.flush()?;
            }
            StartupPacket::CancelRequest {
                process_id,
                secret_key,
            } => {
                cancellations.cancel(process_id, &secret_key);
                return Ok(None);
            }
            StartupPacket::Startup { params, .. } => {
                authenticate_scram_sha256(stream, auth_user, verifier, &params)?;
                let cancellation = cancellations.register();
                write_startup_ready(stream, cancellation.backend_key())?;
                return Ok(Some(cancellation));
            }
        }
    }
}

pub(crate) fn complete_local_startup(
    stream: &mut dyn ReadWrite,
    cancellations: &Arc<CancellationRegistry>,
) -> Result<Option<ConnectionCancellation>, String> {
    match negotiate_plain_startup(stream, false, cancellations)
        .map_err(|error| error.to_string())?
    {
        PlainStartupAction::Ready(cancellation) => Ok(Some(cancellation)),
        PlainStartupAction::Close => Ok(None),
        PlainStartupAction::UpgradeTls => Err(String::from(
            "local-dev startup unexpectedly requested a TLS upgrade",
        )),
    }
}

fn load_tls_config(production: &ProductionSecurityConfig) -> io::Result<TlsServerConfig> {
    let certs = CertificateDer::pem_file_iter(&production.tls_cert)
        .map_err(pem_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(pem_error)?;
    if certs.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "production TLS certificate file contains no certificates",
        ));
    }
    let key = PrivateKeyDer::pem_file_iter(&production.tls_key)
        .map_err(pem_error)?
        .next()
        .transpose()
        .map_err(pem_error)?
        .ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                "production TLS key file contains no private key",
            )
        })?;
    TlsServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))
}

fn pem_error(error: rustls::pki_types::pem::Error) -> io::Error {
    match error {
        rustls::pki_types::pem::Error::Io(error) => error,
        error => io::Error::new(ErrorKind::InvalidData, error.to_string()),
    }
}

fn parse_startup_or_error(stream: &mut dyn ReadWrite, frame: &[u8]) -> io::Result<StartupPacket> {
    match parse_startup_packet(frame) {
        Ok(packet) => Ok(packet),
        Err(error) => protocol_failure(stream, &error.to_string()),
    }
}

fn tls_required<T>(stream: &mut dyn ReadWrite) -> io::Result<T> {
    write_backend_error(stream, "28000", "production security profile requires TLS")?;
    Err(io::Error::new(
        ErrorKind::PermissionDenied,
        "production security profile requires TLS",
    ))
}

fn protocol_failure<T>(stream: &mut dyn ReadWrite, message: &str) -> io::Result<T> {
    write_backend_error(stream, "08P01", message)?;
    Err(io::Error::new(ErrorKind::InvalidData, message.to_string()))
}

fn authentication_failure<T>(stream: &mut dyn ReadWrite) -> io::Result<T> {
    const MESSAGE: &str = "password authentication failed";
    write_backend_error(stream, "28P01", MESSAGE)?;
    Err(io::Error::new(ErrorKind::PermissionDenied, MESSAGE))
}

fn write_backend_error(stream: &mut dyn ReadWrite, code: &str, message: &str) -> io::Result<()> {
    BackendWriter::new(&mut *stream).error_response(&BackendError::new(code, message))?;
    stream.flush()
}

fn write_authentication_ok(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(&mut *stream).authentication_ok()
}

fn write_startup_ready(
    stream: &mut dyn ReadWrite,
    backend_key: &crate::cancellation::BackendKey,
) -> io::Result<()> {
    stream.write_all(&crate::encode_startup_statuses_and_ready(backend_key)?)?;
    stream.flush()
}

fn authenticate_scram_sha256(
    stream: &mut dyn ReadWrite,
    auth_user: &str,
    verifier: &ScramVerifier,
    startup_params: &[(String, String)],
) -> io::Result<()> {
    let mut startup_users = startup_params
        .iter()
        .filter_map(|(key, value)| (key == "user").then_some(value.as_str()));
    let startup_user = startup_users.next().unwrap_or("");
    let authentication_allowed = startup_user == auth_user && startup_users.next().is_none();

    BackendWriter::new(&mut *stream).authentication_sasl(&["SCRAM-SHA-256"])?;
    stream.flush()?;
    let initial_frame = read_auth_frame(stream)?.ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            "connection closed during SCRAM authentication",
        )
    })?;
    let initial = match parse_frontend_message(&initial_frame) {
        Ok(FrontendMessage::SaslInitialResponse {
            mechanism,
            initial_response,
        }) => {
            if mechanism != "SCRAM-SHA-256" {
                return protocol_failure(stream, "unsupported SASL mechanism");
            }
            let Some(initial_response) = initial_response else {
                return protocol_failure(stream, "SCRAM-SHA-256 requires an initial response");
            };
            initial_response
        }
        Ok(_) => return protocol_failure(stream, "expected SASL initial response"),
        Err(error) => return protocol_failure(stream, &error.to_string()),
    };
    let mut nonce_bytes = [0_u8; 18];
    OsRng.fill_bytes(&mut nonce_bytes);
    let server_nonce = BASE64_STANDARD.encode(nonce_bytes);
    let exchange = match ScramExchange::start(&initial, startup_user, &server_nonce, verifier) {
        Ok(exchange) => exchange,
        Err(error) => return write_scram_failure(stream, error),
    };
    BackendWriter::new(&mut *stream)
        .authentication_sasl_continue(exchange.server_first.as_bytes())?;
    stream.flush()?;

    let final_frame = read_auth_frame(stream)?.ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            "connection closed during SCRAM authentication",
        )
    })?;
    let final_message = match parse_frontend_message(&final_frame) {
        Ok(FrontendMessage::SaslResponse(message)) => message,
        Ok(_) => return protocol_failure(stream, "expected SASL final response"),
        Err(error) => return protocol_failure(stream, &error.to_string()),
    };
    let server_signature = match exchange.finish(&final_message, verifier) {
        Ok(signature) => signature,
        Err(error) => return write_scram_failure(stream, error),
    };
    // A nonexistent/malformed startup identity follows the same verifier-backed SCRAM exchange as
    // the configured identity and is doomed only after proof verification. This prevents the
    // first backend message or work profile from becoming a configured-user enumeration oracle.
    if !authentication_allowed {
        return authentication_failure(stream);
    }
    let server_final = format!("v={}", BASE64_STANDARD.encode(server_signature));
    let mut writer = BackendWriter::new(&mut *stream);
    writer.authentication_sasl_final(server_final.as_bytes())?;
    writer.authentication_ok()?;
    stream.flush()
}

#[derive(Clone, Copy, Debug)]
enum ScramFailure {
    Authentication,
    Protocol(&'static str),
}

fn write_scram_failure<T>(stream: &mut dyn ReadWrite, failure: ScramFailure) -> io::Result<T> {
    match failure {
        ScramFailure::Authentication => authentication_failure(stream),
        ScramFailure::Protocol(message) => protocol_failure(stream, message),
    }
}

struct ScramExchange {
    gs2_header: String,
    client_first_bare: String,
    combined_nonce: String,
    server_first: String,
}

impl ScramExchange {
    fn start(
        client_first: &[u8],
        auth_user: &str,
        server_nonce: &str,
        verifier: &ScramVerifier,
    ) -> Result<Self, ScramFailure> {
        if client_first.len() > SCRAM_MAX_MESSAGE_BYTES {
            return Err(ScramFailure::Protocol(
                "SCRAM client-first message is too large",
            ));
        }
        let client_first = std::str::from_utf8(client_first)
            .map_err(|_| ScramFailure::Protocol("invalid SCRAM client-first UTF-8"))?;
        let first_comma = client_first
            .find(',')
            .ok_or(ScramFailure::Protocol("malformed SCRAM GS2 header"))?;
        let second_comma = client_first[first_comma + 1..]
            .find(',')
            .map(|offset| first_comma + 1 + offset)
            .ok_or(ScramFailure::Protocol("malformed SCRAM GS2 header"))?;
        let channel_binding = &client_first[..first_comma];
        if !matches!(channel_binding, "n" | "y") {
            return Err(ScramFailure::Protocol(
                "unsupported SCRAM channel-binding flag",
            ));
        }
        if second_comma != first_comma + 1 {
            return Err(ScramFailure::Protocol(
                "SCRAM authorization identity is not supported",
            ));
        }
        let gs2_header = &client_first[..=second_comma];
        let client_first_bare = &client_first[second_comma + 1..];
        let attributes = parse_scram_attributes(client_first_bare)?;
        if attributes.len() < 2 || attributes[0].0 != 'n' || attributes[1].0 != 'r' {
            return Err(ScramFailure::Protocol(
                "SCRAM client-first requires ordered n and r attributes before extensions",
            ));
        }
        let encoded_user = attributes[0].1;
        let user = decode_scram_name(encoded_user)?;
        // PostgreSQL clients commonly leave the SASL username empty and rely on the startup user.
        if !user.is_empty() && user != auth_user {
            return Err(ScramFailure::Authentication);
        }
        let client_nonce = attributes[1].1;
        validate_scram_nonce(client_nonce)?;
        if server_nonce.is_empty() || server_nonce.contains(',') {
            return Err(ScramFailure::Protocol("invalid SCRAM server nonce"));
        }
        let combined_nonce = format!("{client_nonce}{server_nonce}");
        let server_first = format!(
            "r={combined_nonce},s={},i={}",
            BASE64_STANDARD.encode(&verifier.salt),
            verifier.iterations
        );
        Ok(Self {
            gs2_header: gs2_header.to_string(),
            client_first_bare: client_first_bare.to_string(),
            combined_nonce,
            server_first,
        })
    }

    fn finish(
        &self,
        client_final: &[u8],
        verifier: &ScramVerifier,
    ) -> Result<[u8; SCRAM_KEY_BYTES], ScramFailure> {
        if client_final.len() > SCRAM_MAX_MESSAGE_BYTES {
            return Err(ScramFailure::Protocol(
                "SCRAM client-final message is too large",
            ));
        }
        let client_final = std::str::from_utf8(client_final)
            .map_err(|_| ScramFailure::Protocol("invalid SCRAM client-final UTF-8"))?;
        let attributes = parse_scram_attributes(client_final)?;
        if attributes.len() < 3
            || attributes[0].0 != 'c'
            || attributes[1].0 != 'r'
            || attributes.last().map(|attribute| attribute.0) != Some('p')
        {
            return Err(ScramFailure::Protocol(
                "SCRAM client-final requires ordered c, r, and terminal p attributes",
            ));
        }
        let channel_binding = BASE64_STANDARD
            .decode(attributes[0].1)
            .map_err(|_| ScramFailure::Protocol("invalid SCRAM channel-binding base64"))?;
        if channel_binding.as_slice() != self.gs2_header.as_bytes() {
            return Err(ScramFailure::Protocol(
                "SCRAM channel binding does not match the GS2 header",
            ));
        }
        if attributes[1].1 != self.combined_nonce {
            return Err(ScramFailure::Protocol("SCRAM nonce does not match"));
        }
        let proof = BASE64_STANDARD
            .decode(attributes.last().expect("terminal proof exists").1)
            .map_err(|_| ScramFailure::Protocol("invalid SCRAM proof base64"))?;
        if proof.len() != SCRAM_KEY_BYTES {
            return Err(ScramFailure::Authentication);
        }
        let proof_attribute = attributes.last().expect("terminal proof exists").1;
        let proof_suffix = format!(",p={proof_attribute}");
        let client_final_without_proof = client_final
            .strip_suffix(&proof_suffix)
            .ok_or(ScramFailure::Protocol("malformed SCRAM proof attribute"))?;
        let auth_message = format!(
            "{},{},{}",
            self.client_first_bare, self.server_first, client_final_without_proof
        );
        let client_signature = hmac_sha256(&verifier.stored_key, auth_message.as_bytes());
        let mut recovered_client_key = [0_u8; SCRAM_KEY_BYTES];
        for (target, (proof, signature)) in recovered_client_key
            .iter_mut()
            .zip(proof.iter().zip(client_signature))
        {
            *target = proof ^ signature;
        }
        let recovered_stored_key = sha256(&recovered_client_key);
        if recovered_stored_key.ct_eq(&verifier.stored_key).unwrap_u8() != 1 {
            return Err(ScramFailure::Authentication);
        }
        Ok(hmac_sha256(&verifier.server_key, auth_message.as_bytes()))
    }
}

fn parse_scram_attributes(message: &str) -> Result<Vec<(char, &str)>, ScramFailure> {
    if message.is_empty() {
        return Err(ScramFailure::Protocol("SCRAM attribute list is empty"));
    }
    let mut attributes = Vec::new();
    for part in message.split(',') {
        let bytes = part.as_bytes();
        if bytes.len() < 2 || bytes[1] != b'=' || !bytes[0].is_ascii_alphabetic() {
            return Err(ScramFailure::Protocol("malformed SCRAM attribute"));
        }
        let name = char::from(bytes[0]);
        if name == 'm' {
            return Err(ScramFailure::Protocol(
                "unsupported mandatory SCRAM extension",
            ));
        }
        if attributes.iter().any(|(existing, _)| *existing == name) {
            return Err(ScramFailure::Protocol("duplicate SCRAM attribute"));
        }
        attributes.push((name, &part[2..]));
    }
    Ok(attributes)
}

fn decode_scram_name(encoded: &str) -> Result<String, ScramFailure> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'=' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let escape = bytes
            .get(index + 1..index + 3)
            .ok_or(ScramFailure::Protocol("invalid SCRAM username escape"))?;
        match escape {
            b"2C" => decoded.push(b','),
            b"3D" => decoded.push(b'='),
            _ => return Err(ScramFailure::Protocol("invalid SCRAM username escape")),
        }
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| ScramFailure::Protocol("invalid SCRAM username UTF-8"))
}

fn validate_scram_nonce(nonce: &str) -> Result<(), ScramFailure> {
    if nonce.is_empty()
        || nonce.len() > SCRAM_MAX_NONCE_BYTES
        || nonce
            .bytes()
            .any(|byte| !(0x21..=0x7e).contains(&byte) || byte == b',')
    {
        return Err(ScramFailure::Protocol("invalid SCRAM client nonce"));
    }
    Ok(())
}

impl ScramVerifier {
    /// PostgreSQL applies SASLprep when building a SCRAM secret and falls back to the original
    /// password when the profile rejects a character. Match that behavior for the explicitly
    /// local/test plaintext bootstrap; externally supplied verifiers are already derived.
    fn from_password_bootstrap(password: &str, salt: &[u8], iterations: u32) -> Self {
        let prepared =
            stringprep::saslprep(password).unwrap_or(std::borrow::Cow::Borrowed(password));
        Self::from_password(prepared.as_bytes(), salt, iterations)
    }

    fn from_password(password: &[u8], salt: &[u8], iterations: u32) -> Self {
        let mut salted_password = [0_u8; SCRAM_KEY_BYTES];
        pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut salted_password);
        let client_key = hmac_sha256(&salted_password, b"Client Key");
        Self {
            iterations,
            salt: salt.to_vec(),
            stored_key: sha256(&client_key),
            server_key: hmac_sha256(&salted_password, b"Server Key"),
        }
    }
}

fn parse_scram_verifier(verifier: &str) -> Result<ScramVerifier, String> {
    let Some(rest) = verifier.trim().strip_prefix("SCRAM-SHA-256$") else {
        return Err(String::from(
            "production SCRAM verifier must start with SCRAM-SHA-256$",
        ));
    };
    let mut fields = rest.split('$');
    let iterations_and_salt = fields
        .next()
        .ok_or_else(|| String::from("production SCRAM verifier missing iterations and salt"))?;
    let keys = fields
        .next()
        .ok_or_else(|| String::from("production SCRAM verifier missing stored/server keys"))?;
    if fields.next().is_some() {
        return Err(String::from(
            "production SCRAM verifier has too many fields",
        ));
    }
    let (iterations, salt) = iterations_and_salt
        .split_once(':')
        .ok_or_else(|| String::from("production SCRAM verifier missing salt separator"))?;
    let iterations = iterations
        .parse::<u32>()
        .map_err(|_| String::from("production SCRAM verifier has invalid iteration count"))?;
    if !(SCRAM_MIN_ITERATIONS..=SCRAM_MAX_ITERATIONS).contains(&iterations) {
        return Err(format!(
            "production SCRAM verifier iteration count must be between {SCRAM_MIN_ITERATIONS} and {SCRAM_MAX_ITERATIONS}"
        ));
    }
    let salt = BASE64_STANDARD
        .decode(salt)
        .map_err(|_| String::from("production SCRAM verifier has invalid salt base64"))?;
    if salt.is_empty() || salt.len() > 1024 {
        return Err(String::from(
            "production SCRAM verifier requires a non-empty salt of at most 1024 bytes",
        ));
    }
    let (stored_key, server_key) = keys
        .split_once(':')
        .ok_or_else(|| String::from("production SCRAM verifier missing key separator"))?;
    let stored_key = decode_scram_key(stored_key, "stored")?;
    let server_key = decode_scram_key(server_key, "server")?;
    Ok(ScramVerifier {
        iterations,
        salt,
        stored_key,
        server_key,
    })
}

fn decode_scram_key(encoded: &str, name: &str) -> Result<[u8; SCRAM_KEY_BYTES], String> {
    let bytes = BASE64_STANDARD
        .decode(encoded)
        .map_err(|_| format!("production SCRAM verifier has invalid {name}-key base64"))?;
    bytes.try_into().map_err(|_| {
        String::from("production SCRAM verifier stored and server keys must be 32 bytes")
    })
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; SCRAM_KEY_BYTES] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

fn sha256(data: &[u8]) -> [u8; SCRAM_KEY_BYTES] {
    Sha256::digest(data).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    struct ScriptedIo {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }

    impl ScriptedIo {
        fn new(input: Vec<u8>) -> Self {
            Self {
                input: Cursor::new(input),
                output: Vec::new(),
            }
        }
    }

    impl io::Read for ScriptedIo {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.input.read(buffer)
        }
    }

    impl io::Write for ScriptedIo {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn no_env_parse(args: &[&str]) -> Result<ServerConfig, String> {
        ServerConfig::parse(
            args.iter().map(|arg| (*arg).to_string()),
            SecurityInputs::default(),
        )
    }

    fn startup_code(code: u32, suffix: &[u8]) -> Vec<u8> {
        let mut frame = u32::try_from(8 + suffix.len())
            .unwrap()
            .to_be_bytes()
            .to_vec();
        frame.extend_from_slice(&code.to_be_bytes());
        frame.extend_from_slice(suffix);
        frame
    }

    fn startup_message(user: &str) -> Vec<u8> {
        let mut suffix = Vec::new();
        suffix.extend_from_slice(b"user\0");
        suffix.extend_from_slice(user.as_bytes());
        suffix.extend_from_slice(b"\0database\0postgres\0\0");
        startup_code(196_608, &suffix)
    }

    fn frontend_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![tag];
        frame.extend_from_slice(&u32::try_from(payload.len() + 4).unwrap().to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    fn read_backend_frame(stream: &mut UnixStream) -> (u8, Vec<u8>) {
        let mut tag = [0_u8; 1];
        stream.read_exact(&mut tag).unwrap();
        let mut length = [0_u8; 4];
        stream.read_exact(&mut length).unwrap();
        let length = u32::from_be_bytes(length) as usize;
        let mut payload = vec![0_u8; length - 4];
        stream.read_exact(&mut payload).unwrap();
        (tag[0], payload)
    }

    fn send_client_first(stream: &mut UnixStream, client_first: &str) {
        let mut payload = b"SCRAM-SHA-256\0".to_vec();
        payload.extend_from_slice(&i32::try_from(client_first.len()).unwrap().to_be_bytes());
        payload.extend_from_slice(client_first.as_bytes());
        stream.write_all(&frontend_frame(b'p', &payload)).unwrap();
    }

    fn client_final_for_password(
        password: &[u8],
        client_first_bare: &str,
        server_first: &str,
    ) -> String {
        let attributes: Vec<(&str, &str)> = server_first
            .split(',')
            .map(|attribute| attribute.split_once('=').unwrap())
            .collect();
        let nonce = attributes
            .iter()
            .find_map(|(name, value)| (*name == "r").then_some(*value))
            .unwrap();
        let salt = BASE64_STANDARD
            .decode(
                attributes
                    .iter()
                    .find_map(|(name, value)| (*name == "s").then_some(*value))
                    .unwrap(),
            )
            .unwrap();
        let iterations = attributes
            .iter()
            .find_map(|(name, value)| (*name == "i").then_some(*value))
            .unwrap()
            .parse::<u32>()
            .unwrap();
        let final_without_proof = format!("c={},r={nonce}", BASE64_STANDARD.encode(b"n,,"));
        let auth_message = format!("{client_first_bare},{server_first},{final_without_proof}");
        let mut salted = [0_u8; SCRAM_KEY_BYTES];
        pbkdf2_hmac::<Sha256>(password, &salt, iterations, &mut salted);
        let client_key = hmac_sha256(&salted, b"Client Key");
        let signature = hmac_sha256(&sha256(&client_key), auth_message.as_bytes());
        let proof: Vec<u8> = client_key
            .iter()
            .zip(signature)
            .map(|(key, signature)| key ^ signature)
            .collect();
        format!("{final_without_proof},p={}", BASE64_STANDARD.encode(proof))
    }

    #[test]
    fn config_preserves_positional_listen_and_fails_closed_on_partial_security() {
        let default = no_env_parse(&[]).unwrap();
        assert_eq!(default.listen_address(), DEFAULT_LISTEN);
        assert!(matches!(default.security, SecurityConfig::LocalDev));

        let positional = no_env_parse(&["127.0.0.1:6543"]).unwrap();
        assert_eq!(positional.listen_address(), "127.0.0.1:6543");
        assert!(no_env_parse(&["--listen", "a", "b"])
            .unwrap_err()
            .contains("more than once"));
        assert!(no_env_parse(&["--tls-cert", "server.crt"])
            .unwrap_err()
            .contains("does not accept"));
        assert!(no_env_parse(&["--security-profile", "production"])
            .unwrap_err()
            .contains("--tls-cert"));
    }

    #[test]
    fn production_config_requires_one_credential_and_valid_verifier() {
        let base = [
            "--security-profile",
            "production",
            "--tls-cert",
            "server.crt",
            "--tls-key",
            "server.key",
            "--auth-user",
            "gpudb",
        ];
        let mut conflict = base.to_vec();
        conflict.extend([
            "--auth-password",
            "secret",
            "--auth-scram-verifier",
            "not-a-verifier",
        ]);
        let error = no_env_parse(&conflict).unwrap_err();
        assert!(error.contains("only one credential source"));
        assert!(!error.contains("secret"));

        let mut malformed = base.to_vec();
        malformed.extend(["--auth-scram-verifier", "not-a-verifier"]);
        assert!(no_env_parse(&malformed)
            .unwrap_err()
            .contains("SCRAM-SHA-256$"));

        let weak = format!(
            "SCRAM-SHA-256$1:{}${}:{}",
            BASE64_STANDARD.encode(b"salt"),
            BASE64_STANDARD.encode([0_u8; 32]),
            BASE64_STANDARD.encode([1_u8; 32])
        );
        let mut weak_args = base.to_vec();
        weak_args.extend(["--auth-scram-verifier", &weak]);
        assert!(no_env_parse(&weak_args)
            .unwrap_err()
            .contains("between 4096"));

        let mut password_args = base.to_vec();
        password_args.extend(["--auth-password", "do-not-print-this-secret"]);
        let debug = format!("{:?}", no_env_parse(&password_args).unwrap());
        assert!(debug.contains("production"));
        assert!(!debug.contains("do-not-print-this-secret"));
    }

    #[test]
    fn password_bootstrap_matches_postgresql_saslprep_and_raw_fallback() {
        let normalized = ScramVerifier::from_password_bootstrap("I\u{00ad}X", b"salt", 4096);
        let expected = ScramVerifier::from_password(b"IX", b"salt", 4096);
        assert_eq!(normalized.stored_key, expected.stored_key);
        assert_eq!(normalized.server_key, expected.server_key);

        let prohibited = "raw\u{7f}password";
        let fallback = ScramVerifier::from_password_bootstrap(prohibited, b"salt", 4096);
        let expected = ScramVerifier::from_password(prohibited.as_bytes(), b"salt", 4096);
        assert_eq!(fallback.stored_key, expected.stored_key);
        assert_eq!(fallback.server_key, expected.server_key);
    }

    #[test]
    fn scram_first_message_validates_gs2_username_nonce_and_duplicates() {
        let verifier = ScramVerifier::from_password(b"secret", b"fixed salt", 4096);
        assert!(ScramExchange::start(b"n,,n=,r=client", "gpudb", "server", &verifier).is_ok());
        assert!(ScramExchange::start(b"n,,n=gpudb,r=client", "gpudb", "server", &verifier).is_ok());
        assert!(matches!(
            ScramExchange::start(b"p=tls-server-end-point,,n=,r=x", "gpudb", "s", &verifier),
            Err(ScramFailure::Protocol(_))
        ));
        assert!(matches!(
            ScramExchange::start(b"n,a=other,n=,r=x", "gpudb", "s", &verifier),
            Err(ScramFailure::Protocol(_))
        ));
        assert!(matches!(
            ScramExchange::start(b"n,,n=,r=x,r=y", "gpudb", "s", &verifier),
            Err(ScramFailure::Protocol(_))
        ));
        for reordered in [
            b"n,,r=x,n=".as_slice(),
            b"n,,x=extension,n=,r=x".as_slice(),
            b"n,,n=,x=extension,r=x".as_slice(),
        ] {
            assert!(matches!(
                ScramExchange::start(reordered, "gpudb", "s", &verifier),
                Err(ScramFailure::Protocol(_))
            ));
        }
        assert!(ScramExchange::start(b"n,,n=,r=x,x=extension", "gpudb", "s", &verifier).is_ok());
        assert!(matches!(
            ScramExchange::start(b"n,,n=other,r=x", "gpudb", "s", &verifier),
            Err(ScramFailure::Authentication)
        ));
    }

    #[test]
    fn scram_final_binds_nonce_gs2_and_proof() {
        let password = b"secret";
        let verifier = ScramVerifier::from_password(password, b"fixed salt", 4096);
        let exchange =
            ScramExchange::start(b"n,,n=,r=client", "gpudb", "server", &verifier).unwrap();
        let final_without_proof = format!(
            "c={},r={}",
            BASE64_STANDARD.encode(b"n,,"),
            exchange.combined_nonce
        );
        let auth_message = format!(
            "{},{},{}",
            exchange.client_first_bare, exchange.server_first, final_without_proof
        );
        let mut salted = [0_u8; 32];
        pbkdf2_hmac::<Sha256>(password, b"fixed salt", 4096, &mut salted);
        let client_key = hmac_sha256(&salted, b"Client Key");
        let client_signature = hmac_sha256(&sha256(&client_key), auth_message.as_bytes());
        let proof: Vec<u8> = client_key
            .iter()
            .zip(client_signature)
            .map(|(key, signature)| key ^ signature)
            .collect();
        let valid = format!("{final_without_proof},p={}", BASE64_STANDARD.encode(&proof));
        assert!(exchange.finish(valid.as_bytes(), &verifier).is_ok());

        let wrong_nonce = format!(
            "c={},r=client-other,p={}",
            BASE64_STANDARD.encode(b"n,,"),
            BASE64_STANDARD.encode(&proof)
        );
        assert!(matches!(
            exchange.finish(wrong_nonce.as_bytes(), &verifier),
            Err(ScramFailure::Protocol(_))
        ));
        let wrong_binding = format!(
            "c={},r={},p={}",
            BASE64_STANDARD.encode(b"y,,"),
            exchange.combined_nonce,
            BASE64_STANDARD.encode(&proof)
        );
        assert!(matches!(
            exchange.finish(wrong_binding.as_bytes(), &verifier),
            Err(ScramFailure::Protocol(_))
        ));
        let bad_proof = format!(
            "{final_without_proof},p={}",
            BASE64_STANDARD.encode([0_u8; 32])
        );
        assert!(matches!(
            exchange.finish(bad_proof.as_bytes(), &verifier),
            Err(ScramFailure::Authentication)
        ));
    }

    #[test]
    fn raw_startup_state_machine_covers_gss_ssl_cancel_nested_and_malformed() {
        const SSL_REQUEST: u32 = 80_877_103;
        const GSS_REQUEST: u32 = 80_877_104;
        const CANCEL_REQUEST: u32 = 80_877_102;
        let cancellations = Arc::new(CancellationRegistry::new());

        let mut gss_then_ssl = ScriptedIo::new(
            [
                startup_code(GSS_REQUEST, &[]),
                startup_code(SSL_REQUEST, &[]),
            ]
            .concat(),
        );
        assert!(matches!(
            negotiate_plain_startup(&mut gss_then_ssl, true, &cancellations).unwrap(),
            PlainStartupAction::UpgradeTls
        ));
        assert_eq!(gss_then_ssl.output, b"NS");

        let mut direct_plaintext = ScriptedIo::new(startup_message("gpudb"));
        let Err(error) = negotiate_plain_startup(&mut direct_plaintext, true, &cancellations)
        else {
            panic!("production plaintext startup unexpectedly succeeded");
        };
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert_eq!(direct_plaintext.output.first(), Some(&b'E'));
        assert!(direct_plaintext
            .output
            .windows(b"C28000\0".len())
            .any(|window| window == b"C28000\0"));

        let registered = cancellations.register();
        let direct_active = registered.begin_request().unwrap();
        let cancel_suffix = [
            registered.backend_key().process_id().to_be_bytes().to_vec(),
            registered.backend_key().secret_key_bytes().to_vec(),
        ]
        .concat();
        let mut direct_cancel = ScriptedIo::new(startup_code(CANCEL_REQUEST, &cancel_suffix));
        assert!(matches!(
            negotiate_plain_startup(&mut direct_cancel, true, &cancellations).unwrap(),
            PlainStartupAction::Close
        ));
        assert!(direct_cancel.output.is_empty());
        assert!(direct_active.is_cancelled());
        drop(direct_active);

        let verifier = ScramVerifier::from_password(b"secret", b"salt", 4096);
        let mut nested_ssl = ScriptedIo::new(startup_code(SSL_REQUEST, &[]));
        let Err(error) = finish_tls_startup(&mut nested_ssl, "gpudb", &verifier, &cancellations)
        else {
            panic!("nested TLS startup unexpectedly succeeded");
        };
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(nested_ssl.output.first(), Some(&b'E'));
        assert!(nested_ssl
            .output
            .windows(b"C08P01\0".len())
            .any(|window| window == b"C08P01\0"));

        let tls_active = registered.begin_request().unwrap();
        let mut tls_cancel = ScriptedIo::new(startup_code(CANCEL_REQUEST, &cancel_suffix));
        assert!(
            finish_tls_startup(&mut tls_cancel, "gpudb", &verifier, &cancellations)
                .unwrap()
                .is_none()
        );
        assert!(tls_cancel.output.is_empty());
        assert!(tls_active.is_cancelled());
        drop(tls_active);

        let malformed_target = cancellations.register();
        let malformed_active = malformed_target.begin_request().unwrap();
        let process_id = malformed_target.backend_key().process_id().to_be_bytes();
        let secret = malformed_target.backend_key().secret_key_bytes();
        let mut short_suffix = process_id.to_vec();
        short_suffix.extend_from_slice(&secret[..3]);
        let mut short_direct = ScriptedIo::new(startup_code(CANCEL_REQUEST, &short_suffix));
        assert!(matches!(
            negotiate_plain_startup(&mut short_direct, true, &cancellations).unwrap(),
            PlainStartupAction::Close
        ));
        assert!(short_direct.output.is_empty());
        assert!(!malformed_active.is_cancelled());

        let mut oversized_suffix = process_id.to_vec();
        oversized_suffix.extend_from_slice(&secret);
        oversized_suffix.push(0);
        let mut oversized_direct = ScriptedIo::new(startup_code(CANCEL_REQUEST, &oversized_suffix));
        assert!(matches!(
            negotiate_plain_startup(&mut oversized_direct, true, &cancellations).unwrap(),
            PlainStartupAction::Close
        ));
        assert!(oversized_direct.output.is_empty());
        assert!(!malformed_active.is_cancelled());

        let mut short_tls = ScriptedIo::new(startup_code(CANCEL_REQUEST, &short_suffix));
        assert!(
            finish_tls_startup(&mut short_tls, "gpudb", &verifier, &cancellations)
                .unwrap()
                .is_none()
        );
        assert!(short_tls.output.is_empty());
        assert!(!malformed_active.is_cancelled());

        let mut oversized_tls = ScriptedIo::new(startup_code(CANCEL_REQUEST, &oversized_suffix));
        assert!(
            finish_tls_startup(&mut oversized_tls, "gpudb", &verifier, &cancellations,)
                .unwrap()
                .is_none()
        );
        assert!(oversized_tls.output.is_empty());
        assert!(!malformed_active.is_cancelled());

        let mut malformed = ScriptedIo::new(startup_code(42, &[]));
        let Err(error) = negotiate_plain_startup(&mut malformed, true, &cancellations) else {
            panic!("malformed startup unexpectedly succeeded");
        };
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(malformed.output.first(), Some(&b'E'));
        assert!(malformed
            .output
            .windows(b"C08P01\0".len())
            .any(|window| window == b"C08P01\0"));
    }

    #[test]
    fn raw_scram_wrong_user_is_doomed_only_after_full_proof_exchange() {
        let (mut server, mut client) = UnixStream::pair().unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let verifier = ScramVerifier::from_password(b"secret", b"fixed salt", 4096);
        let server_thread = std::thread::spawn(move || {
            authenticate_scram_sha256(
                &mut server,
                "gpudb",
                &verifier,
                &[("user".to_string(), "other".to_string())],
            )
        });

        let (tag, sasl) = read_backend_frame(&mut client);
        assert_eq!(tag, b'R');
        assert_eq!(i32::from_be_bytes(sasl[..4].try_into().unwrap()), 10);
        let client_first = "n,,n=,r=client";
        send_client_first(&mut client, client_first);
        let (tag, server_first) = read_backend_frame(&mut client);
        assert_eq!(tag, b'R');
        assert_eq!(
            i32::from_be_bytes(server_first[..4].try_into().unwrap()),
            11
        );
        let server_first = std::str::from_utf8(&server_first[4..]).unwrap();
        let final_message = client_final_for_password(b"secret", "n=,r=client", server_first);
        client
            .write_all(&frontend_frame(b'p', final_message.as_bytes()))
            .unwrap();
        let (tag, error) = read_backend_frame(&mut client);
        assert_eq!(tag, b'E');
        assert!(error
            .windows(b"C28P01\0".len())
            .any(|window| window == b"C28P01\0"));
        assert_eq!(
            server_thread.join().unwrap().unwrap_err().kind(),
            ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn raw_scram_transcript_nonce_tampering_is_protocol_failure() {
        let (mut server, mut client) = UnixStream::pair().unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let verifier = ScramVerifier::from_password(b"secret", b"fixed salt", 4096);
        let server_thread = std::thread::spawn(move || {
            authenticate_scram_sha256(
                &mut server,
                "gpudb",
                &verifier,
                &[("user".to_string(), "gpudb".to_string())],
            )
        });

        assert_eq!(read_backend_frame(&mut client).0, b'R');
        send_client_first(&mut client, "n,,n=,r=client");
        let (tag, continuation) = read_backend_frame(&mut client);
        assert_eq!(tag, b'R');
        assert_eq!(
            i32::from_be_bytes(continuation[..4].try_into().unwrap()),
            11
        );
        let tampered = format!(
            "c={},r=client-tampered,p={}",
            BASE64_STANDARD.encode(b"n,,"),
            BASE64_STANDARD.encode([0_u8; 32])
        );
        client
            .write_all(&frontend_frame(b'p', tampered.as_bytes()))
            .unwrap();
        let (tag, error) = read_backend_frame(&mut client);
        assert_eq!(tag, b'E');
        assert!(error
            .windows(b"C08P01\0".len())
            .any(|window| window == b"C08P01\0"));
        assert_eq!(
            server_thread.join().unwrap().unwrap_err().kind(),
            ErrorKind::InvalidData
        );
    }
}
