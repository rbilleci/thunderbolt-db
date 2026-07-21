//! Engine-backed pgwire server binary (P0-M3).
//!
//! Usage: `gpu-db-engine-server [LISTEN_ADDR | --listen HOST:PORT]` (default
//! `127.0.0.1:5432`). The default local-development profile is trust/no-TLS; production must
//! explicitly provide TLS and SCRAM-SHA-256 material.

// Production global allocator. Under the engine's concurrent read+write load the
// per-commit allocations otherwise serialize readers on glibc's global malloc arena;
// jemalloc's per-thread caches remove most of that contention (measured ~1.4-1.7x
// throughput across the board, ~1.7x on reads under a heavy concurrent-writer load).
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = gpu_db_server::ServerConfig::from_env_args(std::env::args().skip(1))?;
    gpu_db_server::serve_configured(config)?;
    Ok(())
}
