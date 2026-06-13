//! Engine-backed pgwire server binary (P0-M3).
//!
//! Usage: `gpu-db-engine-server [LISTEN_ADDR]` (default `127.0.0.1:5432`).

use std::net::TcpListener;

fn main() -> std::io::Result<()> {
    let listen = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5432".to_string());
    let listener = TcpListener::bind(&listen)?;
    eprintln!("gpu-db-engine-server (facade-backed) listening on {listen}");
    gpu_db_server::serve(listener)
}
