use super::frontend_transport::{
    read_tagged_frame, validate_declared_frame_len, ReadWrite, MAX_STARTUP_FRAME_BYTES,
};
use super::{
    handle_ready_client, write_authentication_ok, write_authentication_sasl,
    write_authentication_sasl_continue, write_authentication_sasl_final, write_backend_key_data,
    write_error, write_parameter_status, write_ready_for_query, ErrorField,
};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use gpu_db_protocol::{
    parse_frontend_message, parse_startup_packet, FrontendMessage, StartupPacket,
};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use rustls::{ServerConfig as TlsServerConfig, ServerConnection, StreamOwned};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServerConfig {
    listen: String,
    shared_catalog: bool,
    security: SecurityConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SecurityConfig {
    LocalDev,
    Production(ProductionSecurityConfig),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductionSecurityConfig {
    tls_cert: PathBuf,
    tls_key: PathBuf,
    auth_user: String,
    auth_credential: ScramCredential,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ScramCredential {
    PasswordBootstrap(String),
    Verifier(ScramVerifier),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScramVerifier {
    iterations: u32,
    salt: Vec<u8>,
    stored_key: Vec<u8>,
    server_key: Vec<u8>,
}

struct RuntimeSecurity {
    config: SecurityConfig,
    tls: Option<Arc<TlsServerConfig>>,
}

pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let security = Arc::new(RuntimeSecurity::from_config(config.security.clone())?);
    let listener = TcpListener::bind(&config.listen)?;
    eprintln!("gpu-db-server listening on {}", config.listen);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let shared_catalog = config.shared_catalog;
                let security = Arc::clone(&security);
                thread::spawn(move || {
                    if let Err(error) = handle_client(stream, shared_catalog, &security) {
                        eprintln!("client error: {error}");
                    }
                });
            }
            Err(error) => eprintln!("accept error: {error}"),
        }
    }

    Ok(())
}

fn parse_args<I>(mut args: I) -> Result<ServerConfig, String>
where
    I: Iterator<Item = String>,
{
    let mut listen = String::from("127.0.0.1:5432");
    let mut shared_catalog = false;
    let mut security_profile = env::var("GPU_DB_SECURITY_PROFILE").ok();
    let mut tls_cert = env::var("GPU_DB_TLS_CERT").ok().map(PathBuf::from);
    let mut tls_key = env::var("GPU_DB_TLS_KEY").ok().map(PathBuf::from);
    let mut auth_user = env::var("GPU_DB_AUTH_USER").ok();
    let mut auth_password = env::var("GPU_DB_AUTH_PASSWORD").ok();
    let mut auth_scram_verifier = env::var("GPU_DB_AUTH_SCRAM_VERIFIER").ok();
    let mut auth_scram_verifier_file = env::var("GPU_DB_AUTH_SCRAM_VERIFIER_FILE")
        .ok()
        .map(PathBuf::from);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --listen"))?;
                listen = value;
            }
            "--shared-catalog" => {
                shared_catalog = true;
            }
            "--security-profile" => {
                security_profile = Some(
                    args.next()
                        .ok_or_else(|| String::from("missing value for --security-profile"))?,
                );
            }
            "--tls-cert" => {
                tls_cert =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        String::from("missing value for --tls-cert")
                    })?));
            }
            "--tls-key" => {
                tls_key =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        String::from("missing value for --tls-key")
                    })?));
            }
            "--auth-user" => {
                auth_user = Some(
                    args.next()
                        .ok_or_else(|| String::from("missing value for --auth-user"))?,
                );
            }
            "--auth-password" => {
                auth_password = Some(
                    args.next()
                        .ok_or_else(|| String::from("missing value for --auth-password"))?,
                );
            }
            "--auth-scram-verifier" => {
                auth_scram_verifier = Some(
                    args.next()
                        .ok_or_else(|| String::from("missing value for --auth-scram-verifier"))?,
                );
            }
            "--auth-scram-verifier-file" => {
                auth_scram_verifier_file = Some(PathBuf::from(args.next().ok_or_else(|| {
                    String::from("missing value for --auth-scram-verifier-file")
                })?));
            }
            "-h" | "--help" => {
                return Err(String::from(
                    "usage: gpu-db-server [--listen HOST:PORT] [--shared-catalog] [--security-profile local-dev|production --tls-cert PATH --tls-key PATH --auth-user USER (--auth-scram-verifier VERIFIER | --auth-scram-verifier-file PATH | --auth-password PASSWORD)]",
                ));
            }
            other => return Err(format!("unsupported argument: {other}")),
        }
    }
    let security = match security_profile.as_deref().unwrap_or("local-dev") {
        "local-dev" | "development" => SecurityConfig::LocalDev,
        "production" => SecurityConfig::Production(ProductionSecurityConfig {
            tls_cert: tls_cert.ok_or_else(|| {
                String::from("production security profile requires --tls-cert or GPU_DB_TLS_CERT")
            })?,
            tls_key: tls_key.ok_or_else(|| {
                String::from("production security profile requires --tls-key or GPU_DB_TLS_KEY")
            })?,
            auth_user: non_empty_config(auth_user, "--auth-user or GPU_DB_AUTH_USER")?,
            auth_credential: production_auth_credential(
                auth_password,
                auth_scram_verifier,
                auth_scram_verifier_file,
            )?,
        }),
        other => return Err(format!("unsupported security profile: {other}")),
    };
    Ok(ServerConfig {
        listen,
        shared_catalog,
        security,
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
    let verifier = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read production SCRAM verifier file: {error}"))?;
    parse_scram_verifier(verifier.trim()).map(ScramCredential::Verifier)
}

impl RuntimeSecurity {
    fn from_config(config: SecurityConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let tls = match &config {
            SecurityConfig::LocalDev => None,
            SecurityConfig::Production(production) => Some(Arc::new(load_tls_config(production)?)),
        };
        Ok(Self { config, tls })
    }
}

fn load_tls_config(
    production: &ProductionSecurityConfig,
) -> Result<TlsServerConfig, Box<dyn std::error::Error>> {
    let mut cert_reader = io::BufReader::new(File::open(&production.tls_cert)?);
    let certs = rustls_pemfile::certs(&mut cert_reader).collect::<Result<Vec<_>, _>>()?;
    if certs.is_empty() {
        return Err("production TLS certificate file contains no certificates".into());
    }

    let mut key_reader = io::BufReader::new(File::open(&production.tls_key)?);
    let key = rustls_pemfile::private_key(&mut key_reader)?
        .ok_or("production TLS key file contains no private key")?;

    Ok(TlsServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?)
}

fn handle_client(
    mut stream: TcpStream,
    shared_catalog: bool,
    security: &RuntimeSecurity,
) -> io::Result<()> {
    if startup_handshake(&mut stream, security, false)? {
        let tls_config = security.tls.as_ref().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidInput,
                "production profile missing TLS runtime config",
            )
        })?;
        let connection = ServerConnection::new(Arc::clone(tls_config))
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
        let mut tls_stream = StreamOwned::new(connection, stream);
        if startup_handshake(&mut tls_stream, security, true)? {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "nested SSLRequest is not supported",
            ));
        }
        return handle_ready_client(&mut tls_stream, shared_catalog);
    }
    handle_ready_client(&mut stream, shared_catalog)
}

fn startup_handshake(
    stream: &mut dyn ReadWrite,
    security: &RuntimeSecurity,
    tls_established: bool,
) -> io::Result<bool> {
    loop {
        let frame = read_startup_frame(stream)?;
        match parse_startup_packet(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?
        {
            StartupPacket::SslRequest => match &security.config {
                SecurityConfig::LocalDev => stream.write_all(b"N")?,
                SecurityConfig::Production(_) => {
                    stream.write_all(b"S")?;
                    stream.flush()?;
                    return Ok(true);
                }
            },
            StartupPacket::GssEncRequest => stream.write_all(b"N")?,
            StartupPacket::CancelRequest { .. } => return Ok(false),
            StartupPacket::Startup { params, .. } => {
                match &security.config {
                    SecurityConfig::LocalDev => write_authentication_ok(stream)?,
                    SecurityConfig::Production(production) => {
                        if !tls_established {
                            write_error(
                                stream,
                                &ErrorField {
                                    code: "28000",
                                    message: "production security profile requires TLS",
                                    position: None,
                                },
                            )?;
                            return Err(io::Error::new(
                                ErrorKind::PermissionDenied,
                                "production security profile requires TLS",
                            ));
                        }
                        authenticate_scram_sha256(stream, production, &params)?
                    }
                }
                write_startup_ready(stream)?;
                return Ok(false);
            }
        }
    }
}

fn write_startup_ready(stream: &mut dyn ReadWrite) -> io::Result<()> {
    write_parameter_status(stream, "client_encoding", "UTF8")?;
    write_parameter_status(stream, "server_version", "16.0")?;
    write_parameter_status(stream, "server_version_num", "160000")?;
    write_parameter_status(stream, "standard_conforming_strings", "on")?;
    write_backend_key_data(stream, 1, 1)?;
    write_ready_for_query(stream, false)
}

fn authenticate_scram_sha256(
    stream: &mut dyn ReadWrite,
    production: &ProductionSecurityConfig,
    startup_params: &[(String, String)],
) -> io::Result<()> {
    let startup_user = startup_params
        .iter()
        .find_map(|(key, value)| (key == "user").then_some(value.as_str()))
        .unwrap_or("");
    if startup_user != production.auth_user {
        write_error(
            stream,
            &ErrorField {
                code: "28P01",
                message: "password authentication failed",
                position: None,
            },
        )?;
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "password authentication failed",
        ));
    }

    write_authentication_sasl(stream, &["SCRAM-SHA-256"])?;
    let initial = read_tagged_frame(stream)?.ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            "connection closed during SCRAM authentication",
        )
    })?;
    let initial = parse_frontend_message(&initial)
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
    let FrontendMessage::SaslInitialResponse {
        mechanism,
        initial_response,
    } = initial
    else {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "expected SASL initial response",
        ));
    };
    if mechanism != "SCRAM-SHA-256" {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "unsupported SASL mechanism",
        ));
    }
    let client_first = String::from_utf8(initial_response.unwrap_or_default())
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid SCRAM client-first UTF-8"))?;
    let client_first_bare = client_first
        .find(",,")
        .map_or(client_first.as_str(), |idx| &client_first[idx + 2..]);
    let client_nonce = scram_attr(client_first_bare, "r").ok_or_else(|| {
        io::Error::new(ErrorKind::InvalidData, "SCRAM client-first missing nonce")
    })?;

    let mut nonce_bytes = [0_u8; 18];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let server_nonce = BASE64_STANDARD.encode(nonce_bytes);
    let combined_nonce = format!("{client_nonce}{server_nonce}");
    let verifier = match &production.auth_credential {
        ScramCredential::PasswordBootstrap(password) => {
            ScramVerifier::from_password_bootstrap(password.as_bytes())
        }
        ScramCredential::Verifier(verifier) => verifier.clone(),
    };
    let server_first = format!(
        "r={combined_nonce},s={},i={}",
        BASE64_STANDARD.encode(&verifier.salt),
        verifier.iterations
    );
    write_authentication_sasl_continue(stream, server_first.as_bytes())?;

    let final_frame = read_tagged_frame(stream)?.ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            "connection closed during SCRAM final response",
        )
    })?;
    let final_message = parse_frontend_message(&final_frame)
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
    let FrontendMessage::SaslResponse(client_final_bytes) = final_message else {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "expected SASL final response",
        ));
    };
    let client_final = String::from_utf8(client_final_bytes)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid SCRAM final UTF-8"))?;
    let proof = scram_attr(&client_final, "p")
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "SCRAM final missing proof"))?;
    let proof_marker = format!(",p={proof}");
    let client_final_without_proof = client_final
        .strip_suffix(&proof_marker)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "malformed SCRAM proof"))?;
    let auth_message = format!("{client_first_bare},{server_first},{client_final_without_proof}");
    let scram = verifier.authentication_secrets(&auth_message);
    let client_proof = BASE64_STANDARD
        .decode(proof)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "invalid SCRAM proof base64"))?;
    if client_proof.len() != scram.client_signature.len() {
        return scram_auth_failed(stream);
    }
    let recovered_client_key: Vec<u8> = client_proof
        .iter()
        .zip(scram.client_signature.iter())
        .map(|(proof, signature)| proof ^ signature)
        .collect();
    let recovered_stored_key = Sha256::digest(&recovered_client_key);
    if recovered_stored_key.as_slice() != scram.stored_key {
        return scram_auth_failed(stream);
    }
    let server_final = format!("v={}", BASE64_STANDARD.encode(scram.server_signature));
    write_authentication_sasl_final(stream, server_final.as_bytes())?;
    write_authentication_ok(stream)
}

fn scram_auth_failed(stream: &mut dyn ReadWrite) -> io::Result<()> {
    write_error(
        stream,
        &ErrorField {
            code: "28P01",
            message: "password authentication failed",
            position: None,
        },
    )?;
    Err(io::Error::new(
        ErrorKind::PermissionDenied,
        "password authentication failed",
    ))
}

struct ScramSecrets {
    stored_key: Vec<u8>,
    client_signature: Vec<u8>,
    server_signature: Vec<u8>,
}

impl ScramVerifier {
    fn from_password_bootstrap(password: &[u8]) -> Self {
        let salt = b"gpu-db-local-test-bootstrap-v1";
        let iterations = 4096_u32;
        let mut salted_password = [0_u8; 32];
        pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut salted_password);
        let client_key = hmac_sha256(&salted_password, b"Client Key");
        let stored_key = Sha256::digest(&client_key).to_vec();
        let server_key = hmac_sha256(&salted_password, b"Server Key");
        Self {
            iterations,
            salt: salt.to_vec(),
            stored_key,
            server_key,
        }
    }

    fn authentication_secrets(&self, auth_message: &str) -> ScramSecrets {
        ScramSecrets {
            stored_key: self.stored_key.clone(),
            client_signature: hmac_sha256(&self.stored_key, auth_message.as_bytes()),
            server_signature: hmac_sha256(&self.server_key, auth_message.as_bytes()),
        }
    }
}

fn parse_scram_verifier(verifier: &str) -> Result<ScramVerifier, String> {
    let verifier = verifier.trim();
    let Some(rest) = verifier.strip_prefix("SCRAM-SHA-256$") else {
        return Err(String::from(
            "production SCRAM verifier must start with SCRAM-SHA-256$",
        ));
    };
    let mut parts = rest.split('$');
    let iterations_and_salt = parts
        .next()
        .ok_or_else(|| String::from("production SCRAM verifier missing iterations and salt"))?;
    let keys = parts
        .next()
        .ok_or_else(|| String::from("production SCRAM verifier missing stored/server keys"))?;
    if parts.next().is_some() {
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
    if iterations == 0 {
        return Err(String::from(
            "production SCRAM verifier requires a positive iteration count",
        ));
    }
    let salt = BASE64_STANDARD
        .decode(salt)
        .map_err(|_| String::from("production SCRAM verifier has invalid salt base64"))?;
    if salt.is_empty() {
        return Err(String::from(
            "production SCRAM verifier requires a non-empty salt",
        ));
    }

    let (stored_key, server_key) = keys
        .split_once(':')
        .ok_or_else(|| String::from("production SCRAM verifier missing key separator"))?;
    let stored_key = BASE64_STANDARD
        .decode(stored_key)
        .map_err(|_| String::from("production SCRAM verifier has invalid stored-key base64"))?;
    let server_key = BASE64_STANDARD
        .decode(server_key)
        .map_err(|_| String::from("production SCRAM verifier has invalid server-key base64"))?;
    if stored_key.len() != 32 || server_key.len() != 32 {
        return Err(String::from(
            "production SCRAM verifier stored and server keys must be 32 bytes",
        ));
    }

    Ok(ScramVerifier {
        iterations,
        salt,
        stored_key,
        server_key,
    })
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn scram_attr<'a>(message: &'a str, name: &str) -> Option<&'a str> {
    message
        .split(',')
        .find_map(|part| part.strip_prefix(&format!("{name}=")))
}

fn read_startup_frame(stream: &mut dyn ReadWrite) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0_u8; 4];
    if let Err(error) = stream.read_exact(&mut len_bytes) {
        return if error.kind() == ErrorKind::UnexpectedEof {
            Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "connection closed during startup",
            ))
        } else {
            Err(error)
        };
    }

    let frame_len = validate_declared_frame_len(
        u32::from_be_bytes(len_bytes),
        8,
        MAX_STARTUP_FRAME_BYTES,
        "startup",
    )?;
    let mut frame = vec![0_u8; frame_len];
    frame[..4].copy_from_slice(&len_bytes);
    stream.read_exact(&mut frame[4..])?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn startup_frame_bounds_precede_allocation_and_partial_payloads_fail() {
        assert_eq!(
            validate_declared_frame_len(8, 8, MAX_STARTUP_FRAME_BYTES, "startup").unwrap(),
            8
        );
        assert_eq!(
            validate_declared_frame_len(
                MAX_STARTUP_FRAME_BYTES as u32,
                8,
                MAX_STARTUP_FRAME_BYTES,
                "startup",
            )
            .unwrap(),
            MAX_STARTUP_FRAME_BYTES
        );
        assert!(validate_declared_frame_len(7, 8, MAX_STARTUP_FRAME_BYTES, "startup").is_err());
        let over = (MAX_STARTUP_FRAME_BYTES as u32 + 1).to_be_bytes();
        assert_eq!(
            read_startup_frame(&mut Cursor::new(over))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            read_startup_frame(&mut Cursor::new(vec![0, 0]))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        let partial = [8_u32.to_be_bytes().to_vec(), vec![0, 3, 0]].concat();
        assert_eq!(
            read_startup_frame(&mut Cursor::new(partial))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        let valid = [8_u32.to_be_bytes().to_vec(), vec![0, 3, 0, 0]].concat();
        assert_eq!(
            read_startup_frame(&mut Cursor::new(valid.clone())).unwrap(),
            valid
        );
    }

    #[test]
    fn args_default_listen_and_shared_catalog_opt_in() {
        assert_eq!(
            parse_args(std::iter::empty()).unwrap(),
            ServerConfig {
                listen: "127.0.0.1:5432".to_string(),
                shared_catalog: false,
                security: SecurityConfig::LocalDev,
            }
        );
        assert_eq!(
            parse_args(
                vec![
                    String::from("--shared-catalog"),
                    String::from("--listen"),
                    String::from("0.0.0.0:9999"),
                ]
                .into_iter(),
            )
            .unwrap(),
            ServerConfig {
                listen: "0.0.0.0:9999".to_string(),
                shared_catalog: true,
                security: SecurityConfig::LocalDev,
            }
        );
    }

    #[test]
    fn args_production_security_profile_requires_explicit_material() {
        let missing = parse_args(
            vec![
                String::from("--security-profile"),
                String::from("production"),
            ]
            .into_iter(),
        )
        .unwrap_err();
        assert!(missing.contains("--tls-cert"));

        assert_eq!(
            parse_args(
                vec![
                    String::from("--security-profile"),
                    String::from("production"),
                    String::from("--tls-cert"),
                    String::from("server.crt"),
                    String::from("--tls-key"),
                    String::from("server.key"),
                    String::from("--auth-user"),
                    String::from("gpudb"),
                    String::from("--auth-password"),
                    String::from("secret"),
                ]
                .into_iter(),
            )
            .unwrap()
            .security,
            SecurityConfig::Production(ProductionSecurityConfig {
                tls_cert: PathBuf::from("server.crt"),
                tls_key: PathBuf::from("server.key"),
                auth_user: "gpudb".to_string(),
                auth_credential: ScramCredential::PasswordBootstrap("secret".to_string()),
            })
        );

        let verifier = "SCRAM-SHA-256$4096:Z3B1LWRiLWxvY2FsLXRlc3QtYm9vdHN0cmFwLXYx$4erxom0WBaSLt+HFK3YYcj24A2x3/3bDc8Q34O70v2I=:wFNhWvIT2jq0yJnneM+uNQ9EeJEjvt0W+LpRQtuHVhA=";
        assert!(matches!(
            parse_args(
                vec![
                    String::from("--security-profile"),
                    String::from("production"),
                    String::from("--tls-cert"),
                    String::from("server.crt"),
                    String::from("--tls-key"),
                    String::from("server.key"),
                    String::from("--auth-user"),
                    String::from("gpudb"),
                    String::from("--auth-scram-verifier"),
                    verifier.to_string(),
                ]
                .into_iter(),
            )
            .unwrap()
            .security,
            SecurityConfig::Production(ProductionSecurityConfig {
                auth_credential: ScramCredential::Verifier(_),
                ..
            })
        ));

        let conflict = parse_args(
            vec![
                String::from("--security-profile"),
                String::from("production"),
                String::from("--tls-cert"),
                String::from("server.crt"),
                String::from("--tls-key"),
                String::from("server.key"),
                String::from("--auth-user"),
                String::from("gpudb"),
                String::from("--auth-password"),
                String::from("secret"),
                String::from("--auth-scram-verifier"),
                verifier.to_string(),
            ]
            .into_iter(),
        )
        .unwrap_err();
        assert!(conflict.contains("only one credential source"));

        let malformed = parse_args(
            vec![
                String::from("--security-profile"),
                String::from("production"),
                String::from("--tls-cert"),
                String::from("server.crt"),
                String::from("--tls-key"),
                String::from("server.key"),
                String::from("--auth-user"),
                String::from("gpudb"),
                String::from("--auth-scram-verifier"),
                String::from("not-a-verifier"),
            ]
            .into_iter(),
        )
        .unwrap_err();
        assert!(malformed.contains("SCRAM-SHA-256"));
    }
}
