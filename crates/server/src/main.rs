//! Engine-backed pgwire server binary (P0-M3).
//!
//! Usage: `gpu-db-engine-server [LISTEN_ADDR]` (default `127.0.0.1:5432`).

use std::net::TcpListener;

// Production global allocator. Under the engine's concurrent read+write load the
// per-commit allocations otherwise serialize readers on glibc's global malloc arena;
// jemalloc's per-thread caches remove most of that contention (measured ~1.4-1.7x
// throughput across the board, ~1.7x on reads under a heavy concurrent-writer load).
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> std::io::Result<()> {
    let listen = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5432".to_string());
    let listener = TcpListener::bind(&listen)?;
    eprintln!("gpu-db-engine-server (facade-backed) listening on {listen}");
    gpu_db_server::serve(listener)
}
