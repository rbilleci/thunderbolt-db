use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

use gpu_db_execution::CudaI32BatchProjectionRow;
use gpu_db_execution::DeviceTarget;
use gpu_db_metrics::GpuParityIssue;
use gpu_db_observability::InMemoryTelemetrySink;

// Shared test fixtures, helpers, and tests-wide state (re-exported so every
// feature submodule reaches them via `use super::*`).
mod commit_timestamps;
mod common;
pub(crate) use common::*;

static NEXT_TEST_WAL_PATH_ID: AtomicU64 = AtomicU64::new(1);

// Feature test modules (P0 §9.6), grouped by what they exercise — read order
// roughly follows the engine: read path, write path, then catalog/durability.
mod column_default;
mod concurrency; // &self read path, stage-3 reader/writer + value-index
mod intent_fast_path; // E2.1 covered-INSERT intent route + FUA recovery parity
mod literal_projection; // no-FROM typed scalar GPU route + fail-closed sabotage
mod mvcc_bundles; // provenance frames + bundle/occurrence filters
mod mvcc_joins; // labeled-branch + join-side projection/ordering
mod mvcc_provenance; // provenance + source-composition + value-chain queries
mod mvcc_query; // MVCC query basics: scan/first-cuda-slice/cuda-native/composition
mod null_representation; // M3 NULL slice 1: SqlValue::Null value-model foundation
mod recovery; // durable-WAL recovery + archive retention + checkpoint vacuum
mod replication_backlog; // replication watermarks + backlog blockers
mod resident_expr; // general GPU executor: Expr IR + device interpreter (Charter rule 2)
mod resident_probe; // resident-snapshot GPU probe execution
mod resident_route; // p8 resident-route planning + partitioned probes
mod snapshot_residency; // snapshot meta + residency invalidation
mod sql_catalog; // numeric coercion, pg_catalog, GPU bridge, constraints
mod sql_dml; // relational SQL CRUD, COPY, ALTER COLUMN, sequences, matviews
mod sql_pg; // SQL -> ResidentExpr binding via libpg_query (general GPU executor, Charter rule 2)
mod streaming_exec; // STRATA S-E.1: out-of-core streaming scalar reductions (ADR-012)
mod table_reset_catalog; // PRODUCT-001 typed root reset catalog/recovery behavior
mod table_reset_guards; // PRODUCT-001 stable-OID ownership across deferred/raw mutation surfaces
mod text_batching; // execute_text/read, batching, transactions, replication-role gating
mod text_point_read; // compact GPU int4-filter/text projection semantics
mod write_half; // SI ledger, active snapshots, concurrent DML, stage-0 replay
mod write_set; // prepare_* write-set + apply_delta round-trips
