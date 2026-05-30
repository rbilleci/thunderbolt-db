use std::{env, error::Error, net::TcpListener, path::PathBuf, sync::Arc, thread, time::Duration};

use gpu_db_replication::{
    load_replication_mtls_client_config, load_replication_mtls_server_config,
    load_replication_tls_client_config_without_client_auth, send_append_entries_mtls_once,
    serve_append_entries_mtls_once, AppendEntriesRequest, RaftReplicator,
};
use gpu_db_types::LogEntry;

const TIMEOUT: Duration = Duration::from_secs(2);

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args().skip(1))?;

    let server_config =
        load_replication_mtls_server_config(&args.ca_cert, &args.server_cert, &args.server_key)?;
    let client_config =
        load_replication_mtls_client_config(&args.ca_cert, &args.client_cert, &args.client_key)?;
    println!("replication_channel_security_config_validation=passed");
    let missing_cert = args.server_cert.with_extension("missing");
    if load_replication_mtls_server_config(&args.ca_cert, &missing_cert, &args.server_key).is_ok() {
        return Err("mTLS server config accepted missing certificate material".into());
    }
    if load_replication_mtls_client_config(&args.ca_cert, &args.client_cert, &missing_cert).is_ok()
    {
        return Err("mTLS client config accepted missing private-key material".into());
    }
    println!("replication_channel_security_missing_material_rejection=passed");

    let missing_client_auth_config =
        load_replication_tls_client_config_without_client_auth(&args.ca_cert)?;
    let missing_client_auth_rejected = run_missing_client_auth_probe(
        Arc::clone(&server_config),
        missing_client_auth_config,
        args.server_name.as_str(),
    )?;
    if !missing_client_auth_rejected {
        return Err("mTLS server accepted a client without certificate material".into());
    }
    println!("replication_channel_security_missing_client_cert_rejection=passed");

    let response = run_mtls_append_probe(server_config, client_config, args.server_name.as_str())?;
    if !response.accepted || response.follower_term != 1 {
        return Err(format!("unexpected mTLS append response: {response:?}").into());
    }

    println!("operational_replication_channel_security_smoke=passed");
    println!("replication_channel_security_transport=mtls_append_entries");
    println!("replication_channel_security_scope=local_generated_ca_server_client_certs");
    println!(
        "replication_channel_security_identity=server_dns_name_plus_client_certificate_required"
    );
    println!("replication_channel_security_plain_transport_profile=dev_test_only");
    println!("replication_channel_security_gap_certificate_lifecycle=missing");
    println!("replication_channel_security_gap_production_trust_distribution=missing");
    Ok(())
}

fn run_mtls_append_probe(
    server_config: Arc<rustls::ServerConfig>,
    client_config: Arc<rustls::ClientConfig>,
    server_name: &str,
) -> Result<gpu_db_replication::AppendEntriesResponse, Box<dyn Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let follower = thread::spawn(move || {
        let mut follower = RaftReplicator::new(3);
        serve_append_entries_mtls_once(&listener, &mut follower, server_config, TIMEOUT)
    });

    let response = send_append_entries_mtls_once(
        addr,
        server_name,
        client_config,
        &AppendEntriesRequest {
            leader_term: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                payload: b"secure append".to_vec(),
            }],
            leader_commit: 0,
        },
        TIMEOUT,
    )?;
    let server_response = follower
        .join()
        .map_err(|_| "mTLS follower thread panicked")??;
    if response != server_response {
        return Err(
            format!("client/server response drift: {response:?} != {server_response:?}").into(),
        );
    }
    Ok(response)
}

fn run_missing_client_auth_probe(
    server_config: Arc<rustls::ServerConfig>,
    client_config: Arc<rustls::ClientConfig>,
    server_name: &str,
) -> Result<bool, Box<dyn Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let follower = thread::spawn(move || {
        let mut follower = RaftReplicator::new(3);
        serve_append_entries_mtls_once(&listener, &mut follower, server_config, TIMEOUT)
    });

    let client_result = send_append_entries_mtls_once(
        addr,
        server_name,
        client_config,
        &AppendEntriesRequest {
            leader_term: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        },
        TIMEOUT,
    );
    let server_result = follower
        .join()
        .map_err(|_| "mTLS missing-client-auth follower thread panicked")?;
    Ok(client_result.is_err() && server_result.is_err())
}

#[derive(Debug)]
struct Args {
    ca_cert: PathBuf,
    server_cert: PathBuf,
    server_key: PathBuf,
    client_cert: PathBuf,
    client_key: PathBuf,
    server_name: String,
}

fn parse_args<I>(args: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut ca_cert = None;
    let mut server_cert = None;
    let mut server_key = None;
    let mut client_cert = None;
    let mut client_key = None;
    let mut server_name = None;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--ca-cert" => ca_cert = iter.next().map(PathBuf::from),
            "--server-cert" => server_cert = iter.next().map(PathBuf::from),
            "--server-key" => server_key = iter.next().map(PathBuf::from),
            "--client-cert" => client_cert = iter.next().map(PathBuf::from),
            "--client-key" => client_key = iter.next().map(PathBuf::from),
            "--server-name" => server_name = iter.next(),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(Args {
        ca_cert: ca_cert.ok_or("missing --ca-cert")?,
        server_cert: server_cert.ok_or("missing --server-cert")?,
        server_key: server_key.ok_or("missing --server-key")?,
        client_cert: client_cert.ok_or("missing --client-cert")?,
        client_key: client_key.ok_or("missing --client-key")?,
        server_name: server_name.ok_or("missing --server-name")?,
    })
}
