//! End-to-end proof (P0-M3): a real PostgreSQL client (`tokio-postgres`) drives
//! the engine-backed façade server over a TCP socket through the simple query
//! protocol, and a CREATE/INSERT/SELECT lifecycle round-trips correctly.

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use bytes::{Bytes, BytesMut};
use futures_util::{stream, SinkExt, TryStreamExt};
use gpu_db_engine::Engine;
use gpu_db_execution::{CudaDriverRuntime, DeviceTarget};
use gpu_db_facade::SharedEngine;
use tokio_postgres::{types::Type, NoTls, SimpleQueryMessage};

/// Removes the test-owned durable-WAL directory even when the diagnostic convergence assertion
/// intentionally fails. Each gate invocation owns a unique directory below the system temporary
/// directory, never a user-selected or shared location.
struct TestWalDirectory(std::path::PathBuf);

impl Drop for TestWalDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Read the exact durable record sequence from the backend that the served engine selected.
/// The production open path is disk-shape authoritative: an FUA frame log has no plain segment
/// file, while a serial log does. The convergence gate follows that same distinction instead of
/// forcing a test-only durability backend or inspecting the live engine's private WAL buffer.
fn read_test_durable_wal_records(segment_path: &std::path::Path) -> Vec<gpu_db_wal::WalRecord> {
    #[cfg(unix)]
    if gpu_db_wal::fua_wal_segments_exist(segment_path) {
        return gpu_db_wal::recover_fua_wal_records(segment_path)
            .expect("read the FUA durable records emitted by the served engine");
    }
    gpu_db_wal::read_wal_segment(segment_path)
        .expect("read the serial durable records emitted by the served engine")
}

/// WRITE-000's production convergence gate crosses the real wire loop and must remain one
/// durable authority all the way through fresh-context reopen. A literal SimpleQuery and a
/// parameter-bound Parse/Bind/Execute cover both fixed-width and dense BOOL/TEXT vectors,
/// reordering, NULLs, DEFAULT/sequence effects, RETURNING, explicit commit/rollback, and
/// already-published named indexes. It is intentionally broad: a narrow codec-5 canary passing
/// beside legacy broad-shape routes is a failure, not partial completion.
#[tokio::test]
#[ignore = "requires a local NVIDIA driver and GPU"]
#[cfg(feature = "probe-timing")]
async fn write000_unified_insert_pgwire_reopen_uses_one_authority() {
    let runtime = CudaDriverRuntime::probe()
        .expect("the explicitly requested typed INSERT GPU proof requires a CUDA driver");
    let runtime = runtime.snapshot();
    assert!(
        runtime.driver_available && runtime.device_count > 0,
        "the explicitly requested typed INSERT GPU proof requires an NVIDIA device"
    );

    let wal_directory = std::env::temp_dir().join(format!(
        "gpu-db-write000-pgwire-convergence-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&wal_directory).expect("create the test-owned durable WAL directory");
    let _wal_directory_cleanup = TestWalDirectory(wal_directory.clone());
    let segment_path = wal_directory.join("pgwire-convergence.wal");

    let engine = Engine::with_durable_wal_segment(&segment_path);
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    // Seed the first relation through the engine API so the production wire fixture can begin
    // with INSERT in both protocol modes. Subsequent DDL also traverses pgwire and must enroll
    // independently authenticated roots in the same GPU-native holder.
    engine
        .execute_text(
            900_000,
            "CREATE TABLE pgwire_typed_spine (id int4, value int4)",
        )
        .unwrap();
    let shared = Arc::new(SharedEngine::from_engine(engine));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = Arc::clone(&shared);
    let server_task = tokio::spawn(async move {
        let _ = gpu_db_server::serve_async_with_engine_batching(listener, served, 64, true).await;
    });
    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .unwrap();
    let connection_task = tokio::spawn(async move {
        let _ = connection.await;
    });

    // Begin with both wire forms against the same typed table. The durable route assertions
    // below apply to this complete fixture; no statement is allowed to select a legacy/direct
    // authority merely because of protocol mode or transaction shape.
    let wire_wal_records_before = read_test_durable_wal_records(&segment_path).len();
    let insert_probe_before = shared.insert_probe_snapshot();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (1, 10), (2, 20)")
        .await
        .unwrap();
    let statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_spine VALUES ($1, $2), ($3, $4)",
            &[Type::INT4, Type::INT4, Type::INT4, Type::INT4],
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .execute(&statement, &[&3_i32, &30_i32, &4_i32, &40_i32])
            .await
            .unwrap(),
        2
    );

    // Two independently parsed INSERT contributions in one explicit transaction must compose
    // into one typed transaction-final lifecycle. This is intentionally not expressed as a
    // single multi-row statement: doing so would leave the old single-statement selector green.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (7, 70)")
        .await
        .unwrap();
    let multi_contribution_statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_spine VALUES ($1, $2)",
            &[Type::INT4, Type::INT4],
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .execute(&multi_contribution_statement, &[&8_i32, &80_i32])
            .await
            .unwrap(),
        1
    );
    client.simple_query("COMMIT").await.unwrap();
    // INSERT, UPDATE, and DELETE can coexist in one explicit transaction, but the presence of
    // either non-INSERT operation must not send the typed INSERT through the resolved binary
    // transaction body. This deliberately includes both kinds of final-image composition:
    // private rows that are rewritten/cancelled before publication, and rows that already exist
    // in the published GPU generation. The latter is WRITE-000's convergence assertion: the
    // one codec-5 transition must update and tombstone those pre-existing identities while it
    // appends the typed INSERT contribution, then fresh reopen must reproduce the same state.
    let mixed_preexisting_wal_start = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (14, 140)")
        .await
        .unwrap();
    client
        .simple_query("UPDATE pgwire_typed_spine SET value = 141 WHERE id = 14")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (15, 150)")
        .await
        .unwrap();
    client
        .simple_query("DELETE FROM pgwire_typed_spine WHERE id = 15")
        .await
        .unwrap();
    client
        .simple_query("UPDATE pgwire_typed_spine SET value = 11 WHERE id = 1")
        .await
        .unwrap();
    client
        .simple_query("DELETE FROM pgwire_typed_spine WHERE id = 2")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let mixed_preexisting_records = read_test_durable_wal_records(&segment_path);
    let mixed_preexisting_records = &mixed_preexisting_records[mixed_preexisting_wal_start..];
    assert_eq!(
        mixed_preexisting_records.len(),
        1,
        "the mixed INSERT/UPDATE/DELETE transaction must retain one canonical durable authority",
    );
    let mixed_preexisting_envelope =
        gpu_db_wal::decode_canonical_record_payload(&mixed_preexisting_records[0].payload)
            .expect("mixed pgwire transaction WAL must decode")
            .expect("mixed pgwire transaction must use a canonical codec-5 envelope");
    const OUTER_CONTENT_CATALOG: u32 = 1 << 1;
    const OUTER_CONTENT_OPERATION_COMPOSITION: u32 = 1 << 7;
    assert!(
        mixed_preexisting_envelope.header.flags & OUTER_CONTENT_OPERATION_COMPOSITION != 0,
        "mixed pre-existing-row UPDATE/DELETE must occupy codec-5 S3 rather than a legacy record",
    );
    assert_eq!(
        mixed_preexisting_envelope.header.flags & OUTER_CONTENT_CATALOG,
        0,
        "row-only S3 composition must preserve the catalog boundary",
    );
    assert!(
        mixed_preexisting_envelope.fragments.iter().any(|fragment| {
            fragment.kind == gpu_db_wal::CanonicalFragmentKind::RowMutation
                && fragment.body.get(..8) == Some(&b"GPUDBOP1"[..])
                && fragment.body.get(76..92) == Some(&b"GPUDBTXNAGG1\0\0\0\0"[..])
                && fragment.body.get(94..96).is_some_and(|bytes| {
                    u16::from_le_bytes(bytes.try_into().expect("two-byte semantics version")) == 2
                })
        }),
        "mixed pre-existing-row transaction must retain the codec-5 semantics-v2 aggregate",
    );

    // COMMIT AND CHAIN changes only the transaction context that follows the durable terminal;
    // it must not select a different INSERT record or apply authority. Exercise both sides of
    // the chained boundary so the durable route scan and fresh reopen below catch either the
    // parent falling back or the successor losing the canonical typed overlay.
    client
        .simple_query("CREATE TABLE pgwire_typed_chain_spine (id int4, value int4)")
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_chain_spine VALUES (1, 130)")
        .await
        .unwrap();
    client.simple_query("COMMIT AND CHAIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_chain_spine VALUES (2, 140)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // Catalog lifecycle operations around a typed INSERT must compose into the same canonical
    // transaction envelope. This is the shortest production-reachable mixed-operation shape:
    // the target table already has a GPU predecessor, so failure cannot be blamed on CREATE-table
    // bootstrap geometry. The final view identity and row must both survive fresh reopen below.
    client
        .simple_query("CREATE TABLE pgwire_typed_catalog_mix_spine (id int4, value int4)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE VIEW pgwire_typed_catalog_mix_view AS \
             SELECT id, value FROM pgwire_typed_catalog_mix_spine",
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "ALTER VIEW pgwire_typed_catalog_mix_view \
             RENAME TO pgwire_typed_catalog_mix_renamed",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_catalog_mix_spine VALUES (1, 150)")
        .await
        .unwrap();
    client
        .simple_query("DROP VIEW pgwire_typed_catalog_mix_renamed")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE VIEW pgwire_typed_catalog_mix_renamed AS \
             SELECT value FROM pgwire_typed_catalog_mix_spine",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    client
        .simple_query(
            "CREATE TABLE pgwire_typed_nonunique_index_spine (tenant_id int4, status int4)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_nonunique_index_by_status \
             ON pgwire_typed_nonunique_index_spine (tenant_id, status)",
        )
        .await
        .unwrap();
    // Multiple maintained indexes are one table-generation concern beneath DeviceInsertPlan;
    // they must not select the resolved INSERT encoder merely because S7 has more than one
    // owned index descriptor. This second index turns the old one-index canary into a mandatory
    // production assertion covered by the durable route scan and fresh reopen below.
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_nonunique_index_by_single_status \
             ON pgwire_typed_nonunique_index_spine (status)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_variable_index_spine (id int4, enabled bool, body text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_variable_index_by_body \
             ON pgwire_typed_variable_index_spine (body)",
        )
        .await
        .unwrap();
    // Both seed rows traverse the same real pgwire → SharedEngine mutation boundary as the
    // workload. The generic terminal itself must publish complete named-index coverage; there is
    // no qualification-only second publisher between the write and this assertion.
    client
        .simple_query("INSERT INTO pgwire_typed_nonunique_index_spine VALUES (0, 0)")
        .await
        .expect("seed the non-unique named-index fixture through real pgwire");
    client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_index_spine VALUES (0, false, 'seed') \
             RETURNING id, enabled, body",
        )
        .await
        .expect("seed the BOOL/TEXT named-index fixture through real pgwire");
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_variable_index_peer (id int4, enabled bool, body text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_variable_index_peer_by_body \
             ON pgwire_typed_variable_index_peer (body)",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_variable_index_peer VALUES (0, true, 'peer-seed')")
        .await
        .expect("seed the peer BOOL/TEXT named-index fixture through real pgwire");
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_spine"),
        Some(1),
        "the production BOOL/TEXT terminal covers its one pgwire seed row"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_peer"),
        Some(1),
        "the peer BOOL/TEXT terminal covers its one pgwire seed row"
    );
    client
        .simple_query("CREATE TABLE pgwire_typed_fixed_spine (id int4, tally int8)")
        .await
        .unwrap();
    // A transaction overlay is database-scoped, not table-scoped. Two typed contributions for
    // different tables must close into one codec-5 transaction and one immutable table-map
    // successor instead of selecting the resolved transaction body at COMMIT.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (9, 90)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_fixed_spine VALUES (9, 9000000000)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_returning_spine (id int4, tally int8)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_fixed_types_spine \
             (small int2, day date, observed_at timestamp, amount numeric(12,2), ident uuid)",
        )
        .await
        .unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_nullable_spine (id int4, tally int8)")
        .await
        .unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_variable_spine (id int4, enabled bool, body text)")
        .await
        .unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_copy_spine (id int4, enabled bool, body text)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_scalar_default_spine \
             (id int4 DEFAULT 41, tally int8 DEFAULT 7000000000)",
        )
        .await
        .unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_sequence_spine")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_sequence_owner \
             (id int4 DEFAULT nextval('pgwire_typed_sequence_spine'::regclass), value int4)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_sequence_owner_by_generated_id \
             ON pgwire_typed_sequence_owner (id)",
        )
        .await
        .unwrap();

    // COPY is compatibility ingress into this same typed INSERT authority, not a COPY-local WAL
    // or publication owner. A successful real pgwire stream must therefore be caught by the
    // codec-5 route-identity loop below and reproduce its BOOL/TEXT validity on fresh reopen. It
    // must also retain typed ingress: reconstructing a synthetic INSERT string solely to derive
    // request identity would reintroduce SQL text as a semantic carrier below the facade.
    let copy_probe_before = shared.insert_probe_snapshot();
    let mut copy_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"1,true,copy\n2,,\n",
    ))]);
    let copy_sink = client
        .copy_in("COPY pgwire_typed_copy_spine FROM STDIN WITH CSV")
        .await
        .expect("begin the WRITE-001 compatibility COPY through real pgwire");
    futures_util::pin_mut!(copy_sink);
    copy_sink
        .send_all(&mut copy_data)
        .await
        .expect("stream the WRITE-001 compatibility COPY rows");
    assert_eq!(
        copy_sink
            .finish()
            .await
            .expect("finish the WRITE-001 compatibility COPY"),
        2
    );
    let copy_probe = shared
        .insert_probe_snapshot()
        .delta_since(copy_probe_before);
    assert_eq!(
        copy_probe.raw_request_digest_derivations, 0,
        "COPY compatibility ingress reconstructed and hashed a synthetic SQL INSERT: {copy_probe:?}"
    );
    // The SimpleQuery result must be projected from the explicit transaction's private GPU
    // shard before COMMIT, rather than reconstructed from a host write delta.
    let returning_statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_returning_spine VALUES ($1, $2), ($3, $4) \
             RETURNING tally, id, tally",
            &[Type::INT4, Type::INT8, Type::INT4, Type::INT8],
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    let simple_returning = client
        .simple_query(
            "INSERT INTO pgwire_typed_returning_spine VALUES (1, 10), (2, 20) \
             RETURNING tally, id, tally",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        simple_returning,
        vec![
            (
                Some("10".to_string()),
                Some("1".to_string()),
                Some("10".to_string())
            ),
            (
                Some("20".to_string()),
                Some("2".to_string()),
                Some("20".to_string())
            ),
        ]
    );
    // A separately parsed extended statement in the same explicit transaction keeps its own
    // result projection/digest while sharing the transaction-final typed image and publication.
    let extended_returning = client
        .query(&returning_statement, &[&3_i32, &30_i64, &4_i32, &40_i64])
        .await
        .unwrap();
    assert_eq!(
        extended_returning
            .iter()
            .map(|row| {
                (
                    row.get::<_, i64>(0),
                    row.get::<_, i32>(1),
                    row.get::<_, i64>(2),
                )
            })
            .collect::<Vec<_>>(),
        vec![(30, 3, 30), (40, 4, 40)]
    );
    client.simple_query("COMMIT").await.unwrap();

    // Outside an explicit block, the same result-bearing INSERT is one typed private overlay
    // with an immediate canonical terminal. It must not drop back to concurrent legacy result
    // preparation merely because the session is in autocommit mode.
    let variable_autocommit_returning = client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_spine VALUES (5, false, 'autocommit') \
             RETURNING body, enabled, id, body",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
                row.get(3).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        variable_autocommit_returning,
        vec![(
            Some("autocommit".to_string()),
            Some("f".to_string()),
            Some("5".to_string()),
            Some("autocommit".to_string()),
        )]
    );

    // The no-RETURNING form must use that identical implicit transaction lifecycle.  Keeping
    // this on SimpleQuery catches a regression where only result-bearing autocommit INSERTs
    // enter the private typed overlay while ordinary wire INSERTs retain the concurrent-wave
    // authority.
    let no_returning_completion = client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_spine VALUES \
             (6, true, 'autocommit-no-returning')",
        )
        .await
        .expect(
            "WRITE-001 convergence: a real pgwire autocommit BOOL/TEXT INSERT without RETURNING \
             must use the same durable authority as its result-bearing sibling",
        );
    assert!(matches!(
        no_returning_completion.as_slice(),
        [SimpleQueryMessage::CommandComplete(1)]
    ));

    // The dense private descriptor has to cross both wire forms too: packed BOOL bits and TEXT
    // offsets/blob remain device-backed through simple-query RETURNING before the transaction
    // becomes durable.
    client.simple_query("BEGIN").await.unwrap();
    let variable_simple_returning = client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_spine VALUES \
             (1, true, 'simple'), (2, NULL, NULL) RETURNING body, enabled, id, body",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
                row.get(3).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        variable_simple_returning,
        vec![
            (
                Some("simple".to_string()),
                Some("t".to_string()),
                Some("1".to_string()),
                Some("simple".to_string()),
            ),
            (None, None, Some("2".to_string()), None),
        ]
    );
    client.simple_query("COMMIT").await.unwrap();

    // Bound extended execution must use that same dense typed carrier rather than recovering a
    // row matrix at Bind/Execute.  Include NULL separately from false to exercise the validity
    // bitmap over the actual pgwire codec boundary.
    let variable_statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_variable_spine VALUES \
             ($1, $2, $3), ($4, $5, $6) RETURNING enabled, body, id, enabled",
            &[
                Type::INT4,
                Type::BOOL,
                Type::TEXT,
                Type::INT4,
                Type::BOOL,
                Type::TEXT,
            ],
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    let variable_extended_returning = client
        .query(
            &variable_statement,
            &[
                &3_i32,
                &false,
                &"extended",
                &4_i32,
                &Option::<bool>::None,
                &Option::<String>::None,
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        variable_extended_returning
            .iter()
            .map(|row| {
                (
                    row.get::<_, Option<bool>>(0),
                    row.get::<_, Option<String>>(1),
                    row.get::<_, i32>(2),
                    row.get::<_, Option<bool>>(3),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (Some(false), Some("extended".to_string()), 3, Some(false)),
            (None, None, 4, None),
        ]
    );
    client.simple_query("COMMIT").await.unwrap();

    // Published `nextval` transitions remain independently durable, but their materialized
    // values from two independently parsed statements bind to the same typed transaction-final
    // artifact instead of making the explicit transaction fall back to a row delta.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_sequence_owner (value) VALUES (70)")
        .await
        .unwrap();
    let sequence_contribution = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_sequence_owner (value) VALUES ($1)",
            &[Type::INT4],
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .execute(&sequence_contribution, &[&80_i32])
            .await
            .unwrap(),
        1
    );
    client.simple_query("COMMIT").await.unwrap();

    // A published sequence receipt is transaction-scoped, not table-scoped. Compose its default
    // value and device RETURNING projection with a second table contribution, then require the
    // same one-record codec-5 identity and fresh-replay sequence/table state as every other
    // accepted shape in this gate.
    client.simple_query("BEGIN").await.unwrap();
    let plural_sequence_returning = client
        .simple_query(
            "INSERT INTO pgwire_typed_sequence_owner (value) VALUES (90) \
             RETURNING id, value",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        plural_sequence_returning,
        vec![(Some("3".to_string()), Some("90".to_string()))]
    );
    client
        .simple_query("INSERT INTO pgwire_typed_fixed_spine VALUES (11, 11000000000)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // A sequence restart remains private until this transaction commits. Its default value and
    // final sequence state must be children of the same codec-5 transaction authority.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("ALTER SEQUENCE pgwire_typed_sequence_spine RESTART WITH 40")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_sequence_owner (value) VALUES (400)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_sequence_owner (value) VALUES (410)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_sequence_owner"),
        Some(5),
        "cross-statement private sequence values and restart effects must maintain the named index through codec-5"
    );

    // A transaction-created relation has no public catalog entry or resident predecessor while
    // its first rowset is planned. CREATE SEQUENCE + CREATE TABLE + private FK + INSERT + rename
    // + INSERT + COPY compatibility ingress must nevertheless stay one S3/codec-5 device
    // lifecycle, then appear atomically through this real pgwire connection and a later fresh
    // reopen.
    client
        .simple_query("CREATE TABLE pgwire_typed_private_fk_parent (id int4 PRIMARY KEY)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_private_fk_parent VALUES (7)")
        .await
        .unwrap();
    client
        .simple_query("CREATE DOMAIN pgwire_typed_private_tally AS BIGINT")
        .await
        .unwrap();

    // An immediate private FK must fail when the COPY statement ends, not when the later
    // transaction terminal happens to fold its device plan. The failed statement must poison the
    // same private catalog/sequence overlay, and ROLLBACK must leave no durable catalog or row
    // authority behind.
    let private_fk_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_failure_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_failure_owner \
             (id int4 DEFAULT nextval('pgwire_typed_private_failure_sequence'::regclass), \
              row_key int4 PRIMARY KEY, parent_id int4)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_private_failure_owner \
             ADD CONSTRAINT pgwire_typed_private_failure_owner_parent \
             FOREIGN KEY (parent_id) REFERENCES pgwire_typed_private_fk_parent(id)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_failure_owner (row_key, parent_id) VALUES (1, 7)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_failure_sequence \
             RENAME TO pgwire_typed_private_failure_sequence_final",
        )
        .await
        .unwrap();
    client
        .simple_query("ALTER SEQUENCE pgwire_typed_private_failure_sequence_final RESTART WITH 40")
        .await
        .unwrap();
    let private_failure_copy_probe_before = shared.insert_probe_snapshot();
    let mut private_failure_copy_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(
        Bytes::from_static(b"2,999\n"),
    )]);
    let private_failure_copy_sink = client
        .copy_in(
            "COPY pgwire_typed_private_failure_owner (row_key, parent_id) \
             FROM STDIN WITH CSV",
        )
        .await
        .expect("begin private FK-failure COPY through real pgwire");
    futures_util::pin_mut!(private_failure_copy_sink);
    private_failure_copy_sink
        .send_all(&mut private_failure_copy_data)
        .await
        .expect("stream the private FK-failure COPY row");
    let private_foreign_key_error = private_failure_copy_sink
        .finish()
        .await
        .expect_err("the private FK COPY statement must reject before codec-5 durability");
    assert_eq!(
        private_foreign_key_error.code().map(|code| code.code()),
        Some("23503"),
        "private FK rejection must preserve PostgreSQL SQLSTATE"
    );
    let private_failure_copy_probe = shared
        .insert_probe_snapshot()
        .delta_since(private_failure_copy_probe_before);
    assert_eq!(
        private_failure_copy_probe.raw_request_digest_derivations, 0,
        "private FK-failure COPY reconstructed and hashed a synthetic SQL INSERT: {private_failure_copy_probe:?}"
    );
    let private_foreign_key_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a private FK failure must abort its explicit transaction");
    assert_eq!(
        private_foreign_key_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        private_fk_failure_wal_records,
        "a rolled-back private FK failure must append no durable mutation"
    );
    let missing_private_foreign_key_table = client
        .simple_query("SELECT * FROM pgwire_typed_private_failure_owner")
        .await
        .expect_err("rollback must discard the private FK table");
    assert_eq!(
        missing_private_foreign_key_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );

    // A statement-local CHECK failure must abort the same transaction-private catalog/sequence
    // overlay before WAL, even after the first row, stable-OID rename, and restart have all
    // staged successfully. This is deliberately distinct from the immediate FK COPY verdict
    // above: it proves the server's failed-transaction state cannot leak any private catalog
    // predecessor or select a CHECK-specific INSERT authority.
    let private_check_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_check_failure_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_check_failure_owner \
             (id int4 DEFAULT nextval('pgwire_typed_private_check_failure_sequence'::regclass), \
              row_key int4 PRIMARY KEY, parent_id int4, tally pgwire_typed_private_tally, \
              CONSTRAINT pgwire_typed_private_check_failure_positive CHECK (tally > 0))",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_private_check_failure_owner \
             ADD CONSTRAINT pgwire_typed_private_check_failure_owner_parent \
             FOREIGN KEY (parent_id) REFERENCES pgwire_typed_private_fk_parent(id)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_check_failure_owner \
             (row_key, parent_id, tally) VALUES (1, 7, 1)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_check_failure_sequence \
             RENAME TO pgwire_typed_private_check_failure_sequence_final",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_check_failure_sequence_final RESTART WITH 40",
        )
        .await
        .unwrap();
    let private_check_error = client
        .simple_query(
            "INSERT INTO pgwire_typed_private_check_failure_owner \
             (row_key, parent_id, tally) VALUES (2, 7, 0)",
        )
        .await
        .expect_err("a private domain/CHECK row must reject before codec-5 durability");
    assert_eq!(
        private_check_error.code().map(|code| code.code()),
        Some("23514"),
        "private CHECK rejection must preserve PostgreSQL SQLSTATE"
    );
    let private_check_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a private CHECK failure must abort its explicit transaction");
    assert_eq!(
        private_check_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        private_check_failure_wal_records,
        "a rolled-back private CHECK failure must append no durable mutation"
    );
    let missing_private_check_table = client
        .simple_query("SELECT * FROM pgwire_typed_private_check_failure_owner")
        .await
        .expect_err("rollback must discard the private CHECK table");
    assert_eq!(
        missing_private_check_table.code().map(|code| code.code()),
        Some("42P01"),
        "rollback must classify the discarded private relation as undefined: {missing_private_check_table}"
    );

    // PostgreSQL reports an existing primary-key conflict at the second INSERT, not only when
    // COMMIT later unifies the private GPU shards.  The first statement, sequence rename, and
    // restart make this a transaction-private catalog shape; the duplicate must nevertheless be
    // rejected at statement admission by the same device proof that protects the codec-5 final
    // image, without producing a provisional row, sequence publication, or WAL record.
    let private_unique_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_unique_failure_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_unique_failure_owner \
             (id int4 DEFAULT nextval('pgwire_typed_private_unique_failure_sequence'::regclass), \
              row_key int4 PRIMARY KEY, note text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_unique_failure_owner (row_key, note) \
             VALUES (1, 'first')",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_unique_failure_sequence \
             RENAME TO pgwire_typed_private_unique_failure_sequence_final",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_unique_failure_sequence_final RESTART WITH 40",
        )
        .await
        .unwrap();
    let private_unique_error = client
        .simple_query(
            "INSERT INTO pgwire_typed_private_unique_failure_owner (row_key, note) \
             VALUES (1, 'duplicate')",
        )
        .await
        .expect_err("the second private primary-key INSERT must reject before COMMIT");
    assert_eq!(
        private_unique_error.code().map(|code| code.code()),
        Some("23505"),
        "private primary-key rejection must preserve PostgreSQL SQLSTATE"
    );
    let private_unique_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a private primary-key failure must abort its explicit transaction");
    assert_eq!(
        private_unique_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        private_unique_failure_wal_records,
        "a rolled-back private primary-key failure must append no durable mutation"
    );
    let missing_private_unique_table = client
        .simple_query("SELECT * FROM pgwire_typed_private_unique_failure_owner")
        .await
        .expect_err("rollback must discard the private primary-key table");
    assert_eq!(
        missing_private_unique_table.code().map(|code| code.code()),
        Some("42P01")
    );

    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_create_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_create_owner \
             (id int4 DEFAULT nextval('pgwire_typed_private_create_sequence'::regclass), \
              row_key int4 PRIMARY KEY, parent_id int4, tally pgwire_typed_private_tally, note text, \
              CONSTRAINT pgwire_typed_private_create_tally_positive CHECK (tally > 0))",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_private_create_owner \
             ADD CONSTRAINT pgwire_typed_private_create_owner_parent \
             FOREIGN KEY (parent_id) REFERENCES pgwire_typed_private_fk_parent(id)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_create_owner (row_key, parent_id, tally, note) \
             VALUES (1, 7, 11, 'before-rename')",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_create_sequence \
             RENAME TO pgwire_typed_private_create_sequence_final",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_create_owner (row_key, parent_id, tally, note) \
             VALUES (2, 7, 22, 'after-rename')",
        )
        .await
        .unwrap();
    // The S3 catalog program addresses INSERTs by its global operation cursor while S5 retains
    // the dense typed-statement receipt. Interleave a restart after the stable-OID rename and
    // require the next defaulted INSERT to bind both representations to the same sequence
    // transition inside this transaction-created catalog.
    client
        .simple_query("ALTER SEQUENCE pgwire_typed_private_create_sequence_final RESTART WITH 40")
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_create_owner (row_key, parent_id, tally, note) \
             VALUES (3, 7, 33, 'after-restart')",
        )
        .await
        .unwrap();
    // The private target and renamed default binding are both transaction-local at COPY start.
    // COPY must retain typed values and stage its row in the same overlay, never synthesize a
    // SQL INSERT or publish the first index generation before the enclosing COMMIT.
    let private_copy_probe_before = shared.insert_probe_snapshot();
    let mut private_copy_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(
        Bytes::from_static(b"4,7,44,copy-after-restart\n"),
    )]);
    let private_copy_sink = client
        .copy_in(
            "COPY pgwire_typed_private_create_owner (row_key, parent_id, tally, note) \
             FROM STDIN WITH CSV",
        )
        .await
        .expect("begin transaction-private WRITE-001 COPY through real pgwire");
    futures_util::pin_mut!(private_copy_sink);
    private_copy_sink
        .send_all(&mut private_copy_data)
        .await
        .expect("stream the transaction-private WRITE-001 COPY row");
    assert_eq!(
        private_copy_sink
            .finish()
            .await
            .expect("finish the transaction-private WRITE-001 COPY"),
        1
    );
    let private_copy_probe = shared
        .insert_probe_snapshot()
        .delta_since(private_copy_probe_before);
    assert_eq!(
        private_copy_probe.raw_request_digest_derivations, 0,
        "transaction-private COPY reconstructed and hashed a synthetic SQL INSERT: {private_copy_probe:?}"
    );
    client.simple_query("COMMIT").await.unwrap();
    let private_created_rows = client
        .simple_query(
            "SELECT id, parent_id, tally, note FROM pgwire_typed_private_create_owner ORDER BY id",
        )
        .await
        .expect("read transaction-created codec-5 table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
                row.get(3).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_created_rows,
        vec![
            (
                Some("1".to_string()),
                Some("7".to_string()),
                Some("11".to_string()),
                Some("before-rename".to_string()),
            ),
            (
                Some("2".to_string()),
                Some("7".to_string()),
                Some("22".to_string()),
                Some("after-rename".to_string()),
            ),
            (
                Some("40".to_string()),
                Some("7".to_string()),
                Some("33".to_string()),
                Some("after-restart".to_string()),
            ),
            (
                Some("41".to_string()),
                Some("7".to_string()),
                Some("44".to_string()),
                Some("copy-after-restart".to_string()),
            ),
        ]
    );
    let private_created_sequence_state = client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_create_sequence_final",
        )
        .await
        .expect("read transaction-created private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_created_sequence_state,
        vec![(Some("41".to_string()), Some("t".to_string()))]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_create_owner"),
        Some(4),
        "the transaction-created primary index must publish both INSERT and COPY rows through the codec-5 lifecycle"
    );

    // This is deliberately the terminal S3 rename form: no later S2 record can merely happen
    // to witness the final default name. The sole codec-5 aggregate must instead bind the final
    // table digest to the exact private stable-OID sequence transition, then replay both through
    // the ordinary catalog composition and GPU publication authority.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_terminal_rename_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_terminal_rename_owner \
             (id int4 DEFAULT nextval('pgwire_typed_private_terminal_rename_sequence'::regclass), \
              row_key int4 PRIMARY KEY, note text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_terminal_rename_owner (row_key, note) \
             VALUES (1, 'before-terminal-rename')",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_terminal_rename_sequence \
             RENAME TO pgwire_typed_private_terminal_rename_sequence_final",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let private_terminal_rename_rows = client
        .simple_query(
            "SELECT id, row_key, note FROM pgwire_typed_private_terminal_rename_owner ORDER BY id",
        )
        .await
        .expect("read terminal private-sequence rename through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_terminal_rename_rows,
        vec![(
            Some("1".to_string()),
            Some("1".to_string()),
            Some("before-terminal-rename".to_string()),
        ),]
    );
    let private_terminal_rename_sequence_state = client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_terminal_rename_sequence_final",
        )
        .await
        .expect("read terminal renamed private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_terminal_rename_sequence_state,
        vec![(Some("1".to_string()), Some("t".to_string()))]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_terminal_rename_owner"),
        Some(1),
        "the terminal-rename private primary index must publish through the same codec-5 lifecycle"
    );

    // `serial` creates its sequence as part of CREATE TABLE, rather than through a separate
    // CREATE SEQUENCE command.  Rename that private stable OID before its first typed INSERT:
    // the sole codec-5 lifecycle must bind the original generated default to the final name.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_serial_owner \
             (id serial PRIMARY KEY, note text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_serial_owner_id_seq \
             RENAME TO pgwire_typed_private_serial_owner_id_seq_final",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_serial_owner (note) \
             VALUES ('first'), ('second')",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let private_serial_rows = client
        .simple_query("SELECT id, note FROM pgwire_typed_private_serial_owner ORDER BY id")
        .await
        .expect("read renamed implicit-private-sequence rows through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_serial_rows,
        vec![
            (Some("1".to_string()), Some("first".to_string())),
            (Some("2".to_string()), Some("second".to_string())),
        ]
    );
    let private_serial_sequence_state = client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_serial_owner_id_seq_final",
        )
        .await
        .expect("read renamed implicit private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_serial_sequence_state,
        vec![(Some("2".to_string()), Some("t".to_string()))]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_serial_owner"),
        Some(2),
        "the implicit serial primary index must publish through the same codec-5 lifecycle"
    );

    // A private FK parent is a materially different transaction shape from the published parent
    // above: both newly-created primary-index roots and the child FK verdict must compose in one
    // catalog overlay before the only codec-5 terminal exists.  The child is deliberately
    // defaulted across a stable-OID rename and the final row crosses COPY, so a parent/child
    // shortcut, a second sequence owner, or a COPY-specific lifecycle cannot satisfy this gate.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_private_graph_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_graph_parent (parent_id int4 PRIMARY KEY, note text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_graph_child \
             (id int4 DEFAULT nextval('pgwire_typed_private_graph_sequence'::regclass), \
              row_key int4 PRIMARY KEY, parent_id int4, note text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_private_graph_child \
             ADD CONSTRAINT pgwire_typed_private_graph_child_parent \
             FOREIGN KEY (parent_id) REFERENCES pgwire_typed_private_graph_parent(parent_id)",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_private_graph_parent VALUES (7, 'private-parent')")
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_graph_child (row_key, parent_id, note) \
             VALUES (1, 7, 'before-rename')",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER SEQUENCE pgwire_typed_private_graph_sequence \
             RENAME TO pgwire_typed_private_graph_sequence_final",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_graph_child (row_key, parent_id, note) \
             VALUES (2, 7, 'after-rename')",
        )
        .await
        .unwrap();
    client
        .simple_query("ALTER SEQUENCE pgwire_typed_private_graph_sequence_final RESTART WITH 40")
        .await
        .unwrap();
    let private_graph_copy_probe_before = shared.insert_probe_snapshot();
    let mut private_graph_copy_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(
        Bytes::from_static(b"3,7,copy-after-restart\n"),
    )]);
    let private_graph_copy_sink = client
        .copy_in(
            "COPY pgwire_typed_private_graph_child (row_key, parent_id, note) \
             FROM STDIN WITH CSV",
        )
        .await
        .expect("begin private parent/child COPY through real pgwire");
    futures_util::pin_mut!(private_graph_copy_sink);
    private_graph_copy_sink
        .send_all(&mut private_graph_copy_data)
        .await
        .expect("stream the private parent/child COPY row");
    assert_eq!(
        private_graph_copy_sink
            .finish()
            .await
            .expect("finish the private parent/child COPY"),
        1
    );
    let private_graph_copy_probe = shared
        .insert_probe_snapshot()
        .delta_since(private_graph_copy_probe_before);
    assert_eq!(
        private_graph_copy_probe.raw_request_digest_derivations, 0,
        "private parent/child COPY reconstructed a synthetic INSERT: {private_graph_copy_probe:?}"
    );
    client.simple_query("COMMIT").await.unwrap();
    let private_graph_parent_rows = client
        .simple_query(
            "SELECT parent_id, note FROM pgwire_typed_private_graph_parent ORDER BY parent_id",
        )
        .await
        .expect("read the transaction-created private FK parent through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_graph_parent_rows,
        vec![(Some("7".to_string()), Some("private-parent".to_string()),)]
    );
    let private_graph_child_rows = client
        .simple_query(
            "SELECT id, parent_id, note FROM pgwire_typed_private_graph_child ORDER BY id",
        )
        .await
        .expect("read the transaction-created private FK child through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_graph_child_rows,
        vec![
            (
                Some("1".to_string()),
                Some("7".to_string()),
                Some("before-rename".to_string()),
            ),
            (
                Some("2".to_string()),
                Some("7".to_string()),
                Some("after-rename".to_string()),
            ),
            (
                Some("40".to_string()),
                Some("7".to_string()),
                Some("copy-after-restart".to_string()),
            ),
        ]
    );
    let private_graph_sequence_state = client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_graph_sequence_final",
        )
        .await
        .expect("read the renamed private graph sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_graph_sequence_state,
        vec![(Some("40".to_string()), Some("t".to_string()))]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_graph_parent"),
        Some(1),
        "the private FK parent must publish its first primary-index generation through codec-5"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_graph_child"),
        Some(3),
        "the private FK child must publish every defaulted/COPY key through codec-5"
    );

    // The domain itself must be transaction-private too. Its S3 creation, the table's first
    // primary-index root, CHECK validation, and typed rows must still form one codec-5 lifecycle;
    // a published-domain prelude would not prove this catalog-composition ingress.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE DOMAIN pgwire_typed_private_inline_amount AS int4")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_inline_domain_owner \
             (id int4 PRIMARY KEY, amount pgwire_typed_private_inline_amount, \
              CONSTRAINT pgwire_typed_private_inline_amount_positive CHECK (amount > 0))",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_private_inline_domain_owner VALUES (1, 11), (2, 22)",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let private_inline_domain_rows = client
        .simple_query("SELECT id, amount FROM pgwire_typed_private_inline_domain_owner ORDER BY id")
        .await
        .expect("read the transaction-private domain table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_inline_domain_rows,
        vec![
            (Some("1".to_string()), Some("11".to_string())),
            (Some("2".to_string()), Some("22".to_string())),
        ]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_inline_domain_owner"),
        Some(2),
        "the transaction-private domain table must publish its first primary-index generation through codec-5"
    );

    // A CHECK failure after a valid typed row must poison and discard the entire private-domain
    // catalog program. In particular, the staged domain/table cannot be made visible through a
    // non-codec-5 error or rollback path.
    let private_inline_domain_failure_wal_records =
        read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE DOMAIN pgwire_typed_private_inline_failure_amount AS int4")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_private_inline_failure_owner \
             (id int4 PRIMARY KEY, amount pgwire_typed_private_inline_failure_amount, \
              CONSTRAINT pgwire_typed_private_inline_failure_amount_positive CHECK (amount > 0))",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_private_inline_failure_owner VALUES (1, 11)")
        .await
        .unwrap();
    let private_inline_domain_check = client
        .simple_query("INSERT INTO pgwire_typed_private_inline_failure_owner VALUES (2, -1)")
        .await
        .expect_err("the private-domain CHECK must reject the invalid row before COMMIT");
    assert_eq!(
        private_inline_domain_check.code().map(|code| code.code()),
        Some("23514"),
        "private-domain CHECK rejection must preserve PostgreSQL SQLSTATE"
    );
    let private_inline_domain_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("the failed private-domain transaction must remain aborted");
    assert_eq!(
        private_inline_domain_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        private_inline_domain_failure_wal_records,
        "a failed private-domain transaction must append no durable mutation"
    );
    let missing_private_inline_domain_table = client
        .simple_query("SELECT * FROM pgwire_typed_private_inline_failure_owner")
        .await
        .expect_err("rollback must discard the private-domain table");
    assert_eq!(
        missing_private_inline_domain_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );

    // A separately declared transaction-private index is distinct from the inline primary-key
    // roots above. Its CREATE INDEX catalog command, typed rows, S3 identity, and first GPU root
    // must share the same one codec-5 terminal and fresh-reopen owner.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_private_index_spine (id int4, value int4)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE UNIQUE INDEX pgwire_typed_private_index_spine_id \
             ON pgwire_typed_private_index_spine (id)",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_private_index_spine VALUES (1, 10), (2, 20)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let private_index_rows = client
        .simple_query("SELECT id, value FROM pgwire_typed_private_index_spine ORDER BY id")
        .await
        .expect("read the transaction-private indexed table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_index_rows,
        vec![
            (Some("1".to_string()), Some("10".to_string())),
            (Some("2".to_string()), Some("20".to_string())),
        ]
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_private_index_spine"),
        Some(2),
        "the transaction-private CREATE INDEX must publish its first named GPU generation through codec-5"
    );

    // PRODUCT-001 already accepts ordered private DML on both sides of CREATE INDEX. This is
    // deliberately a populated, GPU-published predecessor rather than another initial-table
    // canary: the index root must cover the public prefix and the typed suffix through the one
    // pre-WAL DeviceInsertPlan/S3/S7 lifecycle, then survive fresh replay.
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_populated_index_spine \
             (id int4, code int4, region text)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_populated_index_spine VALUES \
             (1, NULL, 'north'), (2, NULL, 'north'), (3, 7, 'west')",
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_populated_index_spine VALUES (4, NULL, 'west')")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE UNIQUE INDEX pgwire_typed_populated_index_spine_code_region \
             ON pgwire_typed_populated_index_spine (code, region)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_populated_index_spine VALUES \
             (5, 8, 'south'), (6, NULL, 'south')",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    let private_populated_index_rows = client
        .simple_query("SELECT id, code, region FROM pgwire_typed_populated_index_spine ORDER BY id")
        .await
        .expect("read the populated-table private index through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_populated_index_rows,
        vec![
            (Some("1".to_string()), None, Some("north".to_string())),
            (Some("2".to_string()), None, Some("north".to_string())),
            (Some("3".to_string()), Some("7".to_string()), Some("west".to_string())),
            (Some("4".to_string()), None, Some("west".to_string())),
            (Some("5".to_string()), Some("8".to_string()), Some("south".to_string())),
            (Some("6".to_string()), None, Some("south".to_string())),
        ],
        "the S3-created index must cover the complete final relation, including its published prefix"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_populated_index_spine"),
        Some(6),
        "the populated-table private index must publish one complete GPU generation through codec-5"
    );

    // The same populated-table S3 authority must reject a private replacement index's duplicate
    // at statement admission. ROLLBACK must retain the preexisting index and every public row;
    // this is the production pgwire counterpart to the crash/retry seam below the engine layer.
    let populated_index_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("DROP INDEX pgwire_typed_populated_index_spine_code_region")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE UNIQUE INDEX pgwire_typed_populated_index_spine_code_region_rebuilt \
             ON pgwire_typed_populated_index_spine (code, region)",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_populated_index_spine VALUES (7, 9, 'east')")
        .await
        .unwrap();
    let populated_index_duplicate = client
        .simple_query("INSERT INTO pgwire_typed_populated_index_spine VALUES (8, 9, 'east')")
        .await
        .expect_err("the S3 replacement index must reject a private duplicate before WAL");
    assert_eq!(
        populated_index_duplicate.code().map(|code| code.code()),
        Some("23505"),
        "the populated-table S3 replacement index must preserve PostgreSQL SQLSTATE"
    );
    let populated_index_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a populated-table S3 unique failure must abort its explicit transaction");
    assert_eq!(
        populated_index_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        populated_index_failure_wal_records,
        "a rolled-back populated-table S3 replacement must append no durable mutation"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_populated_index_spine"),
        Some(6),
        "rollback must retain coverage for the original populated-table index"
    );
    let rebuilt_index_after_rollback = client
        .simple_query(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'pgwire_typed_populated_index_spine_code_region_rebuilt'",
        )
        .await
        .expect("query the rolled-back S3 replacement index catalog entry")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        rebuilt_index_after_rollback.is_empty(),
        "rollback must discard the replacement S3 catalog identity: {rebuilt_index_after_rollback:?}"
    );

    // The production route must also publish the successful DROP/CREATE transition, not merely
    // reject its failed sibling. This is the pgwire/fresh-reopen counterpart to the durable
    // crash-prefix suite: the original root retires and its replacement covers the complete
    // resident prefix plus one new typed row through the same codec-5 DeviceInsertPlan.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("DROP INDEX pgwire_typed_populated_index_spine_code_region")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE UNIQUE INDEX pgwire_typed_populated_index_spine_code_region_replacement \
             ON pgwire_typed_populated_index_spine (code, region)",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_populated_index_spine VALUES (7, 9, 'east')")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_populated_index_spine"),
        Some(7),
        "the successful S3 replacement must publish one complete GPU index generation"
    );
    let private_populated_index_rows = client
        .simple_query("SELECT id, code, region FROM pgwire_typed_populated_index_spine ORDER BY id")
        .await
        .expect("read the successfully replaced populated-table index through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        private_populated_index_rows,
        vec![
            (Some("1".to_string()), None, Some("north".to_string())),
            (Some("2".to_string()), None, Some("north".to_string())),
            (
                Some("3".to_string()),
                Some("7".to_string()),
                Some("west".to_string())
            ),
            (Some("4".to_string()), None, Some("west".to_string())),
            (
                Some("5".to_string()),
                Some("8".to_string()),
                Some("south".to_string())
            ),
            (Some("6".to_string()), None, Some("south".to_string())),
            (
                Some("7".to_string()),
                Some("9".to_string()),
                Some("east".to_string())
            ),
        ],
        "the successful replacement must retain every prefix row and its typed suffix"
    );
    let replacement_index = client
        .simple_query(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'pgwire_typed_populated_index_spine_code_region_replacement'",
        )
        .await
        .expect("query the successful S3 replacement index catalog entry")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        replacement_index,
        vec!["pgwire_typed_populated_index_spine_code_region_replacement".to_string()],
        "the S3 replacement catalog identity must publish exactly once"
    );

    // The separately declared named index must reject a duplicate at statement admission, not
    // defer it to the transaction terminal.  This preserves the same SQLSTATE/aborted-session
    // contract as the inline-primary-key case without allowing a failed private catalog shape to
    // materialize through the generic resolved INSERT path.
    let private_named_index_failure_wal_records =
        read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_private_named_index_failure (id int4, value int4)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE UNIQUE INDEX pgwire_typed_private_named_index_failure_id \
             ON pgwire_typed_private_named_index_failure (id)",
        )
        .await
        .unwrap();
    let private_named_index_duplicate = client
        .simple_query(
            "INSERT INTO pgwire_typed_private_named_index_failure VALUES (1, 10), (1, 20)",
        )
        .await
        .expect_err("the private named index must reject an in-statement duplicate before codec-5 admission");
    assert_eq!(
        private_named_index_duplicate.code().map(|code| code.code()),
        Some("23505"),
        "private named-index rejection must preserve PostgreSQL SQLSTATE"
    );
    let private_named_index_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a private named-index failure must abort its explicit transaction");
    assert_eq!(
        private_named_index_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        private_named_index_failure_wal_records,
        "a rolled-back private named-index failure must append no durable mutation"
    );
    let missing_private_named_index_table = client
        .simple_query("SELECT * FROM pgwire_typed_private_named_index_failure")
        .await
        .expect_err("rollback must discard the private named-index table");
    assert_eq!(
        missing_private_named_index_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );

    // Two independently parsed contributions must compose through the same named-index
    // generation. A single multi-row statement would leave the bounded one-statement indexed
    // selector green without proving transaction-level index composition.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_nonunique_index_spine VALUES (7, 2), (7, 2)")
        .await
        .unwrap();
    let indexed_contribution = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_nonunique_index_spine VALUES ($1, $2)",
            &[Type::INT4, Type::INT4],
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .execute(&indexed_contribution, &[&8_i32, &3_i32])
            .await
            .unwrap(),
        1
    );
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_nonunique_index_spine"),
        Some(8),
        "the pgwire typed terminal must extend both seeded named GPU indexes by all three rows"
    );

    // Table cardinality must not turn a maintained GPU index into a second transaction
    // authority. This combines an indexed target and a differently typed unindexed target in
    // one explicit transaction; the returned row is produced before the one terminal closes.
    client.simple_query("BEGIN").await.unwrap();
    let plural_index_returning = client
        .simple_query(
            "INSERT INTO pgwire_typed_nonunique_index_spine VALUES (9, 4) \
             RETURNING status, tenant_id",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        plural_index_returning,
        vec![(Some("4".to_string()), Some("9".to_string()))]
    );
    client
        .simple_query("INSERT INTO pgwire_typed_fixed_spine VALUES (10, 10000000000)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_nonunique_index_spine"),
        Some(10),
        "one plural terminal must publish both indexed table successors"
    );

    // Dense BOOL/TEXT staging must meet the same already-published named-index eligibility as
    // fixed vectors. The terminal still owns the final canonical index publication; the private
    // shard itself must not create a second index authority.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_index_spine VALUES \
             (7, true, 'indexed'), (8, false, NULL)",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_spine"),
        Some(3),
        "the BOOL/TEXT typed terminal must extend the seeded named GPU index exactly once"
    );

    // The named-index lifecycle is transaction-scoped, not restricted to one indexed target.
    // Two independently maintained indexed tables must hand the same guard through both physical
    // plans and emit one codec-5 transaction instead of reviving the resolved INSERT record.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_nonunique_index_spine VALUES (10, 5)")
        .await
        .unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_index_spine VALUES (9, true, 'plural-indexed')",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_nonunique_index_spine"),
        Some(12),
        "both non-unique indexes must cover the sixth committed row"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_spine"),
        Some(4),
        "the second indexed table must publish under the same transaction lifecycle"
    );

    // Two dense indexed rollovers must chain their database-root substitutions inside the same
    // transaction candidate. One rollover plus one in-place append is insufficient evidence:
    // both plans here replace TEXT-bearing shard generations before the single visibility cut.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_variable_index_spine VALUES (10, false, 'rollover-a')",
        )
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_variable_index_peer VALUES (1, true, 'rollover-b')")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_spine"),
        Some(5),
        "the first dense rollover must remain covered after the second root substitution"
    );
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_variable_index_peer"),
        Some(2),
        "the second dense rollover must publish under the shared named-index lifecycle"
    );

    // Deterministic DEFAULT and omitted cells resolve into the same catalog-order private
    // vector before staging. This specifically excludes sequence effects, which still keep
    // their dedicated receipt owner until that carrier is wired through the terminal.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_scalar_default_spine (tally) \
             VALUES (DEFAULT), (8000000001)",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // Exercise every remaining non-NULL fixed-width resident vector family in an explicit
    // transaction: i32 (smallint/date), i64 (timestamp), and b128 (numeric/UUID).
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_fixed_types_spine VALUES \
             (-7, '2024-01-02', '2024-01-02 03:04:05.678901', 12.34, \
              '550e8400-e29b-41d4-a716-446655440000')",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // NULL values keep their typed vector payload plus a sparse validity bitmap. They must not
    // be treated as an omitted/default input or disappear at the transaction terminal.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query(
            "INSERT INTO pgwire_typed_nullable_spine VALUES (9, NULL), (NULL, 9000000000)",
        )
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // This crosses the same transaction carrier with two physical vector sections: int4 and
    // int8. The assertion rejects a route that only handles the original all-int4 layout.
    let mixed_statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_fixed_spine VALUES ($1, $2), ($3, $4)",
            &[Type::INT4, Type::INT8, Type::INT4, Type::INT8],
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    assert_eq!(
        client
            .execute(
                &mixed_statement,
                &[&7_i32, &7_000_000_000_i64, &8_i32, &8_000_000_000_i64],
            )
            .await
            .unwrap(),
        2
    );
    client.simple_query("COMMIT").await.unwrap();

    // Column-list input still lowers once into catalog order before it reaches the transaction
    // carrier. This catches a protocol route that merely happens to support table-order VALUES.
    let reordered_statement = client
        .prepare_typed(
            "INSERT INTO pgwire_typed_spine (value, id) VALUES ($1, $2), ($3, $4)",
            &[Type::INT4, Type::INT4, Type::INT4, Type::INT4],
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    assert_eq!(
        client
            .execute(&reordered_statement, &[&50_i32, &5_i32, &60_i32, &6_i32],)
            .await
            .unwrap(),
        2
    );
    client.simple_query("COMMIT").await.unwrap();

    // A second bound execution must remain private until Sync's transaction terminal.  ROLLBACK
    // therefore drops both its staged GPU generation and its potential typed-commit accounting.
    client.simple_query("BEGIN").await.unwrap();
    assert_eq!(
        client
            .execute(&statement, &[&5_i32, &50_i32, &6_i32, &60_i32])
            .await
            .unwrap(),
        2
    );
    client.simple_query("ROLLBACK").await.unwrap();

    // The existing transaction-final GPU FK verdict must remain the only semantic decision.
    // Its exact parent table/index dependencies belong in the same codec-5 closure rather than
    // selecting the resolved INSERT body after the statement has already staged typed vectors.
    client
        .simple_query("CREATE TABLE pgwire_typed_fk_parent (id int4 PRIMARY KEY)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_fk_parent VALUES (101), (102)")
        .await
        .expect("seed the FK provider through the canonical pgwire INSERT route");
    client
        .simple_query("CREATE TABLE pgwire_typed_fk_child (id int4, parent_id int4)")
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_fk_child \
             ADD CONSTRAINT pgwire_typed_fk_child_parent_fk FOREIGN KEY (parent_id) \
             REFERENCES pgwire_typed_fk_parent(id)",
        )
        .await
        .unwrap();
    let fk_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    let foreign_key_error = client
        .simple_query("INSERT INTO pgwire_typed_fk_child VALUES (2, 999)")
        .await
        .expect_err("an absent FK provider must fail before durable INSERT publication");
    assert_eq!(
        foreign_key_error.code().map(|code| code.code()),
        Some("23503"),
        "the production pgwire FK verdict must preserve PostgreSQL SQLSTATE"
    );
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        fk_failure_wal_records,
        "a failed autocommit FK INSERT must append no durable row mutation"
    );
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_fk_child VALUES (1, 101)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // FK validation and local named-index maintenance are orthogonal inputs to the same terminal.
    // Their conjunction must not select the resolved INSERT body merely because the child table
    // needs both a sealed parent dependency and an indexed physical publication.
    client
        .simple_query("CREATE TABLE pgwire_typed_indexed_fk_child (id int4, parent_id int4)")
        .await
        .unwrap();
    client
        .simple_query(
            "ALTER TABLE ONLY pgwire_typed_indexed_fk_child \
             ADD CONSTRAINT pgwire_typed_indexed_fk_child_parent_fk FOREIGN KEY (parent_id) \
             REFERENCES pgwire_typed_fk_parent(id)",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_indexed_fk_child_by_parent \
             ON pgwire_typed_indexed_fk_child (parent_id)",
        )
        .await
        .unwrap();
    let indexed_fk_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    let indexed_foreign_key_error = client
        .simple_query("INSERT INTO pgwire_typed_indexed_fk_child VALUES (2, 999)")
        .await
        .expect_err("an indexed FK child must reject an absent provider before WAL");
    assert_eq!(
        indexed_foreign_key_error.code().map(|code| code.code()),
        Some("23503")
    );
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        indexed_fk_failure_wal_records,
        "a failed indexed-FK INSERT must append no durable mutation"
    );
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_indexed_fk_child VALUES (1, 101)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // The FK guard and the child's maintained index must bind the final image, not the original
    // INSERT statement image. A later UPDATE of a provisional row is still one codec-5 INSERT
    // lifecycle; fresh replay must authenticate and publish parent_id=102 at both boundaries.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_indexed_fk_child VALUES (3, 101)")
        .await
        .unwrap();
    client
        .simple_query("UPDATE pgwire_typed_indexed_fk_child SET parent_id = 102 WHERE id = 3")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_indexed_fk_child"),
        Some(2),
        "the indexed FK child must publish both final-image keys under the codec-5 lifecycle"
    );

    // The FK verdict must also remain statement-local when the violating value is introduced by
    // an UPDATE over a previously staged typed row.  This is distinct from an invalid INSERT:
    // the overlay already owns an INSERT final image, but the rejected rewritten image must not
    // escape through an UPDATE-specific durable or publication tail.
    let mixed_fk_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_indexed_fk_child VALUES (4, 101)")
        .await
        .unwrap();
    let mixed_foreign_key_error = client
        .simple_query("UPDATE pgwire_typed_indexed_fk_child SET parent_id = 999 WHERE id = 4")
        .await
        .expect_err("an UPDATE over a private typed INSERT must reject an absent FK provider");
    assert_eq!(
        mixed_foreign_key_error.code().map(|code| code.code()),
        Some("23503"),
        "the mixed INSERT/UPDATE FK verdict must preserve PostgreSQL SQLSTATE"
    );
    let mixed_foreign_key_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a failed FK UPDATE after typed INSERT must abort the transaction");
    assert_eq!(
        mixed_foreign_key_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        mixed_fk_failure_wal_records,
        "a rolled-back mixed INSERT/UPDATE FK failure must append no durable mutation"
    );

    // Row-local CHECK validation and local named-index maintenance are likewise independent
    // inputs to the one terminal. The empty table must enroll its index before the first valid
    // append, while an invalid append must fail before WAL without selecting a CHECK-specialized
    // or resolved INSERT authority.
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_indexed_check_spine (\
                 id int4, marker int4, \
                 CONSTRAINT pgwire_typed_indexed_check_positive CHECK (marker > 0))",
        )
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_indexed_check_by_marker \
             ON pgwire_typed_indexed_check_spine (marker)",
        )
        .await
        .unwrap();
    let indexed_check_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    let indexed_check_error = client
        .simple_query("INSERT INTO pgwire_typed_indexed_check_spine VALUES (2, -1)")
        .await
        .expect_err("an indexed CHECK table must reject an invalid row before WAL");
    assert_eq!(
        indexed_check_error.code().map(|code| code.code()),
        Some("23514")
    );
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        indexed_check_failure_wal_records,
        "a failed indexed-CHECK INSERT must append no durable mutation"
    );
    client
        .simple_query("INSERT INTO pgwire_typed_indexed_check_spine VALUES (1, 7)")
        .await
        .unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_indexed_check_spine"),
        Some(1),
        "the indexed CHECK table must publish its local index through the codec-5 lifecycle"
    );

    // FK validation must compose with another target under one transaction-final authority.
    // This deliberately crosses the former plural-selector restriction.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_fk_child VALUES (3, 101)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (11, 110)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // The existing row-local device CHECK verdict is semantic input to the same transaction
    // terminal. A successful guarded row must retain that catalog dependency in S7 without
    // selecting a resolved INSERT body or inventing a CHECK-specific apply path.
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_check_spine (\
                 id int4, CONSTRAINT pgwire_typed_check_positive CHECK (id > 0))",
        )
        .await
        .unwrap();
    let check_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    let check_error = client
        .simple_query("INSERT INTO pgwire_typed_check_spine VALUES (-1)")
        .await
        .expect_err("the device CHECK verdict must reject the invalid row");
    assert_eq!(
        check_error.code().map(|code| code.code()),
        Some("23514"),
        "the production pgwire CHECK verdict must preserve PostgreSQL SQLSTATE"
    );
    let aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a failed explicit INSERT must abort its transaction");
    assert_eq!(aborted.code().map(|code| code.code()), Some("25P02"));
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        check_failure_wal_records,
        "a rolled-back CHECK failure must append no durable row mutation"
    );
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_check_spine VALUES (7)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // CHECK must not be a single-target specialization. Combine its device verdict with an
    // ordinary table in one transaction and require the same codec-5 terminal/replay identity.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_check_spine VALUES (8)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (10, 100)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // A successful typed INSERT followed by a failing UPDATE exercises a different rollback
    // boundary than a rejected INSERT: the overlay already owns a private typed image when the
    // CHECK verdict aborts the transaction.  Neither that provisional row nor an UPDATE-specific
    // publication/recovery branch may survive to WAL, and the wire session must retain ordinary
    // PostgreSQL failed-transaction state.
    let mixed_check_failure_wal_records = read_test_durable_wal_records(&segment_path).len();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_check_spine VALUES (9)")
        .await
        .unwrap();
    let mixed_check_error = client
        .simple_query("UPDATE pgwire_typed_check_spine SET id = 0 WHERE id = 9")
        .await
        .expect_err("an UPDATE over a private typed INSERT must apply the CHECK before COMMIT");
    assert_eq!(
        mixed_check_error.code().map(|code| code.code()),
        Some("23514"),
        "the mixed INSERT/UPDATE CHECK verdict must preserve PostgreSQL SQLSTATE"
    );
    let mixed_check_aborted = client
        .simple_query("SELECT 1")
        .await
        .expect_err("a failed UPDATE after typed INSERT must abort the transaction");
    assert_eq!(
        mixed_check_aborted.code().map(|code| code.code()),
        Some("25P02")
    );
    client.simple_query("ROLLBACK").await.unwrap();
    assert_eq!(
        read_test_durable_wal_records(&segment_path).len(),
        mixed_check_failure_wal_records,
        "a rolled-back mixed INSERT/UPDATE CHECK failure must append no durable mutation"
    );

    // A catalog domain remains metadata over the same base-type device vector. Its exact domain
    // identity must survive S2/S7 catalog closure and fresh replay without selecting the resolved
    // row-image carrier merely because the declared type OID differs from BIGINT.
    client
        .simple_query("CREATE DOMAIN pgwire_typed_tally AS BIGINT")
        .await
        .unwrap();
    client
        .simple_query("CREATE TABLE pgwire_typed_domain_spine (id int4, tally pgwire_typed_tally)")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE INDEX pgwire_typed_domain_by_tally \
             ON pgwire_typed_domain_spine (tally)",
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_domain_spine VALUES (9, 9000000001)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    // Domain identity is catalog closure, not a reason to specialize by target count.
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_domain_spine VALUES (10, 10000000001)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (12, 120)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();
    assert_eq!(
        shared.relational_named_index_covered_rows("pgwire_typed_domain_spine"),
        Some(2),
        "domain-typed values must maintain their named GPU index through codec-5"
    );

    // A later DELETE may cancel one provisional INSERT without canceling the transaction or
    // rewinding its private sequence. Compose that neutral table with a surviving table so live
    // publication and fresh replay must distinguish logical affected/allocator rows from GPU
    // final-row transitions inside the same codec-5 authority.
    client
        .simple_query("CREATE SEQUENCE pgwire_typed_canceled_sequence")
        .await
        .unwrap();
    client
        .simple_query(
            "CREATE TABLE pgwire_typed_canceled_sequence_owner (\
                 id int4 DEFAULT nextval('pgwire_typed_canceled_sequence'::regclass), value int4)",
        )
        .await
        .unwrap();
    client.simple_query("BEGIN").await.unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_canceled_sequence_owner (value) VALUES (500)")
        .await
        .unwrap();
    client
        .simple_query("DELETE FROM pgwire_typed_canceled_sequence_owner WHERE value = 500")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO pgwire_typed_spine VALUES (13, 130)")
        .await
        .unwrap();
    client.simple_query("COMMIT").await.unwrap();

    let rows = client
        .simple_query("SELECT id, value FROM pgwire_typed_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (Some("1".to_string()), Some("11".to_string())),
            (Some("3".to_string()), Some("30".to_string())),
            (Some("4".to_string()), Some("40".to_string())),
            (Some("5".to_string()), Some("50".to_string())),
            (Some("6".to_string()), Some("60".to_string())),
            (Some("7".to_string()), Some("70".to_string())),
            (Some("8".to_string()), Some("80".to_string())),
            (Some("9".to_string()), Some("90".to_string())),
            (Some("10".to_string()), Some("100".to_string())),
            (Some("11".to_string()), Some("110".to_string())),
            (Some("12".to_string()), Some("120".to_string())),
            (Some("13".to_string()), Some("130".to_string())),
            (Some("14".to_string()), Some("141".to_string())),
        ]
    );
    let chain_rows = client
        .simple_query("SELECT id, value FROM pgwire_typed_chain_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chain_rows,
        vec![
            (Some("1".to_string()), Some("130".to_string())),
            (Some("2".to_string()), Some("140".to_string())),
        ]
    );
    let mixed_rows = client
        .simple_query("SELECT id, tally FROM pgwire_typed_fixed_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        mixed_rows,
        vec![
            (Some("7".to_string()), Some("7000000000".to_string())),
            (Some("8".to_string()), Some("8000000000".to_string())),
            (Some("9".to_string()), Some("9000000000".to_string())),
            (Some("10".to_string()), Some("10000000000".to_string())),
            (Some("11".to_string()), Some("11000000000".to_string())),
        ]
    );
    let fixed_type_rows = client
        .simple_query(
            "SELECT small, day, observed_at, amount, ident \
             FROM pgwire_typed_fixed_types_spine",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some(
                (0..5)
                    .map(|column| row.get(column).map(str::to_owned))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fixed_type_rows,
        vec![vec![
            Some("-7".to_string()),
            Some("2024-01-02".to_string()),
            Some("2024-01-02 03:04:05.678901".to_string()),
            Some("12.34".to_string()),
            Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
        ]]
    );
    let nullable_rows = client
        .simple_query(
            "SELECT id, tally FROM pgwire_typed_nullable_spine ORDER BY tally NULLS FIRST",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nullable_rows,
        vec![
            (Some("9".to_string()), None),
            (None, Some("9000000000".to_string())),
        ]
    );
    let scalar_default_rows = client
        .simple_query("SELECT id, tally FROM pgwire_typed_scalar_default_spine ORDER BY tally")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scalar_default_rows,
        vec![
            (Some("41".to_string()), Some("7000000000".to_string())),
            (Some("41".to_string()), Some("8000000001".to_string())),
        ]
    );
    let sequence_default_rows = client
        .simple_query("SELECT id, value FROM pgwire_typed_sequence_owner ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequence_default_rows,
        vec![
            (Some("1".to_string()), Some("70".to_string())),
            (Some("2".to_string()), Some("80".to_string())),
            (Some("3".to_string()), Some("90".to_string())),
            (Some("40".to_string()), Some("400".to_string())),
            (Some("41".to_string()), Some("410".to_string())),
        ]
    );
    let sequence_state = client
        .simple_query("SELECT last_value, is_called FROM public.pgwire_typed_sequence_spine")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequence_state,
        vec![(Some("41".to_string()), Some("t".to_string()))]
    );
    let canceled_sequence_rows = client
        .simple_query("SELECT id, value FROM pgwire_typed_canceled_sequence_owner ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        canceled_sequence_rows.is_empty(),
        "the later DELETE must cancel the provisional INSERT row"
    );
    let canceled_sequence_state = client
        .simple_query("SELECT last_value, is_called FROM public.pgwire_typed_canceled_sequence")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        canceled_sequence_state,
        vec![(Some("1".to_string()), Some("t".to_string()))],
        "the committed private sequence effect must survive cancellation of its row"
    );
    let copy_rows = client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_copy_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        copy_rows,
        vec![
            (
                Some("1".to_string()),
                Some("t".to_string()),
                Some("copy".to_string()),
            ),
            (Some("2".to_string()), None, None),
        ]
    );
    let foreign_key_rows = client
        .simple_query("SELECT id, parent_id FROM pgwire_typed_fk_child ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        foreign_key_rows,
        vec![
            (Some("1".to_string()), Some("101".to_string())),
            (Some("3".to_string()), Some("101".to_string())),
        ]
    );
    let indexed_foreign_key_rows = client
        .simple_query("SELECT id, parent_id FROM pgwire_typed_indexed_fk_child ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        indexed_foreign_key_rows,
        vec![
            (Some("1".to_string()), Some("101".to_string())),
            (Some("3".to_string()), Some("102".to_string())),
        ]
    );
    let indexed_check_rows = client
        .simple_query("SELECT id, marker FROM pgwire_typed_indexed_check_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        indexed_check_rows,
        vec![(Some("1".to_string()), Some("7".to_string()))]
    );
    let check_rows = client
        .simple_query("SELECT id FROM pgwire_typed_check_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(check_rows, vec!["7".to_string(), "8".to_string()]);
    let domain_rows = client
        .simple_query("SELECT id, tally FROM pgwire_typed_domain_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        domain_rows,
        vec![
            (Some("9".to_string()), Some("9000000001".to_string())),
            (Some("10".to_string()), Some("10000000001".to_string())),
        ]
    );
    let nonunique_index_rows = client
        .simple_query(
            "SELECT tenant_id, status FROM pgwire_typed_nonunique_index_spine \
             ORDER BY tenant_id, status",
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nonunique_index_rows,
        vec![
            (Some("0".to_string()), Some("0".to_string())),
            (Some("7".to_string()), Some("2".to_string())),
            (Some("7".to_string()), Some("2".to_string())),
            (Some("8".to_string()), Some("3".to_string())),
            (Some("9".to_string()), Some("4".to_string())),
            (Some("10".to_string()), Some("5".to_string())),
        ]
    );
    let variable_index_rows = client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_variable_index_spine ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        variable_index_rows,
        vec![
            (
                Some("0".to_string()),
                Some("f".to_string()),
                Some("seed".to_string()),
            ),
            (
                Some("7".to_string()),
                Some("t".to_string()),
                Some("indexed".to_string()),
            ),
            (Some("8".to_string()), Some("f".to_string()), None),
            (
                Some("9".to_string()),
                Some("t".to_string()),
                Some("plural-indexed".to_string()),
            ),
            (
                Some("10".to_string()),
                Some("f".to_string()),
                Some("rollover-a".to_string()),
            ),
        ]
    );
    let variable_index_peer_rows = client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_variable_index_peer ORDER BY id")
        .await
        .unwrap()
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        variable_index_peer_rows,
        vec![
            (
                Some("0".to_string()),
                Some("t".to_string()),
                Some("peer-seed".to_string()),
            ),
            (
                Some("1".to_string()),
                Some("t".to_string()),
                Some("rollover-b".to_string()),
            ),
        ]
    );
    let catalog_mix_rows = client
        .simple_query("SELECT * FROM pgwire_typed_catalog_mix_renamed")
        .await
        .expect("read the mixed catalog/INSERT view through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(catalog_mix_rows, vec!["150".to_string()]);

    // Route identity is established from the durable record emitted *after* a real pgwire
    // statement, rather than from a private engine object or an incidental timing counter. Every
    // user row mutation in this workload must be the codec-5 aggregate stream at semantics-v2;
    // a canonical envelope carrying a resolved binary INSERT is still a second live write path.
    let wire_records = read_test_durable_wal_records(&segment_path);
    let mut codec5_row_mutations = 0usize;
    for record in wire_records.iter().skip(wire_wal_records_before) {
        let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
            .expect("pgwire mutation WAL payload must decode")
            .unwrap_or_else(|| {
                panic!(
                    "WRITE-001 pgwire transaction {} bypassed the canonical codec-5 envelope",
                    record.txn_id
                )
            });
        for fragment in envelope
            .fragments
            .iter()
            .filter(|fragment| fragment.kind == gpu_db_wal::CanonicalFragmentKind::RowMutation)
        {
            codec5_row_mutations += 1;
            assert_eq!(
                fragment.body.get(..8),
                Some(&b"GPUDBOP1"[..]),
                "WRITE-001 pgwire transaction {} row mutation is not a codec-5 aggregate chunk",
                record.txn_id
            );
            assert_eq!(
                fragment.body.get(76..92),
                Some(&b"GPUDBTXNAGG1\0\0\0\0"[..]),
                "WRITE-001 pgwire transaction {} selected a legacy/resolved INSERT body instead of the codec-5 aggregate stream",
                record.txn_id
            );
            assert_eq!(
                fragment.body.get(94..96).map(|bytes| {
                    u16::from_le_bytes(bytes.try_into().expect("two-byte semantics version"))
                }),
                Some(2),
                "WRITE-001 pgwire transaction {} did not emit codec-5 semantics-v2",
                record.txn_id
            );
        }
    }
    assert!(
        codec5_row_mutations > 0,
        "the pgwire fixture must emit at least one durable INSERT row-mutation envelope"
    );
    let insert_probe = shared
        .insert_probe_snapshot()
        .delta_since(insert_probe_before);
    assert_eq!(
        insert_probe.successful_insert_statements, 73,
        "every successful pgwire INSERT, including COPY compatibility ingress, must close through the one codec-5 terminal: {insert_probe:?}"
    );
    assert_eq!(
        insert_probe.successful_insert_rows, 93,
        "the codec-5 success counter must account for every committed pgwire row without a compatibility/direct bypass: {insert_probe:?}"
    );

    // Every durable INSERT record above was inspected as codec-5. This direct WAL identity check
    // is the migration guard; the displaced legacy-template timing surface no longer exists.
    // Destroy the served engine before reopening. The second listener receives a separately
    // recovered `SharedEngine`, so equality below is a process-style WAL recovery assertion at
    // the same real pgwire boundary rather than a second query over the live in-memory engine.
    drop(client);
    let _ = connection_task.await;
    server_task.abort();
    let _ = server_task.await;
    drop(shared);

    let recovered = Arc::new(
        SharedEngine::new_durable(&segment_path)
            .expect("fresh SharedEngine must recover the durable pgwire WAL segment"),
    );
    assert!(recovered.is_durable());
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_private_create_owner"),
        Some(4),
        "fresh recovery must reconstruct the transaction-created primary index from the same codec-5 record"
    );
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_private_terminal_rename_owner"),
        Some(1),
        "fresh recovery must reconstruct the terminal-rename private primary index from the same codec-5 record"
    );
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_private_serial_owner"),
        Some(2),
        "fresh recovery must reconstruct the renamed implicit serial primary index from the same codec-5 record"
    );
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_private_index_spine"),
        Some(2),
        "fresh recovery must reconstruct the transaction-private explicit index from the same codec-5 record"
    );
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_populated_index_spine"),
        Some(7),
        "fresh recovery must reconstruct the successful populated-table index replacement through codec-5"
    );
    assert_eq!(
        recovered.relational_named_index_covered_rows("pgwire_typed_private_inline_domain_owner"),
        Some(2),
        "fresh recovery must reconstruct the transaction-private domain table's primary index from the same codec-5 record"
    );
    let recovered_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the recovered pgwire listener");
    let recovered_port = recovered_listener.local_addr().unwrap().port();
    let recovered_server_engine = Arc::clone(&recovered);
    let recovered_server_task = tokio::spawn(async move {
        let _ = gpu_db_server::serve_async_with_engine_batching(
            recovered_listener,
            recovered_server_engine,
            64,
            true,
        )
        .await;
    });
    let (recovered_client, recovered_connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={recovered_port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .expect("connect a real pgwire client to the recovered SharedEngine");
    let recovered_connection_task = tokio::spawn(async move {
        let _ = recovered_connection.await;
    });
    let reopened_rebuilt_populated_index = recovered_client
        .simple_query(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'pgwire_typed_populated_index_spine_code_region_rebuilt'",
        )
        .await
        .expect("query the recovered S3 replacement index catalog entry")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        reopened_rebuilt_populated_index.is_empty(),
        "fresh recovery must not materialize the rolled-back S3 replacement index: {reopened_rebuilt_populated_index:?}"
    );
    let reopened_replacement_populated_index = recovered_client
        .simple_query(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'pgwire_typed_populated_index_spine_code_region_replacement'",
        )
        .await
        .expect("query the recovered successful S3 replacement index catalog entry")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reopened_replacement_populated_index,
        vec!["pgwire_typed_populated_index_spine_code_region_replacement".to_string()],
        "fresh recovery must retain exactly the successful S3 replacement index"
    );
    let reopened_missing_private_foreign_key_table = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_private_failure_owner")
        .await
        .expect_err("fresh recovery must not materialize the rolled-back private FK table");
    assert_eq!(
        reopened_missing_private_foreign_key_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );
    let reopened_missing_private_check_table = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_private_check_failure_owner")
        .await
        .expect_err("fresh recovery must not materialize the rolled-back private CHECK table");
    assert_eq!(
        reopened_missing_private_check_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );
    let reopened_missing_private_unique_table = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_private_unique_failure_owner")
        .await
        .expect_err(
            "fresh recovery must not materialize the rolled-back private primary-key table",
        );
    assert_eq!(
        reopened_missing_private_unique_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );
    let reopened_missing_private_named_index_table = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_private_named_index_failure")
        .await
        .expect_err(
            "fresh recovery must not materialize the rolled-back private named-index table",
        );
    assert_eq!(
        reopened_missing_private_named_index_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );
    let reopened_missing_private_inline_domain_table = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_private_inline_failure_owner")
        .await
        .expect_err("fresh recovery must not materialize the rolled-back private-domain table");
    assert_eq!(
        reopened_missing_private_inline_domain_table
            .code()
            .map(|code| code.code()),
        Some("42P01")
    );
    let reopened_rows = recovered_client
        .simple_query("SELECT id, value FROM pgwire_typed_spine ORDER BY id")
        .await
        .expect("read the recovered reordered/bound table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_index_rows = recovered_client
        .simple_query(
            "SELECT tenant_id, status FROM pgwire_typed_nonunique_index_spine \
             ORDER BY tenant_id, status",
        )
        .await
        .expect("read the recovered indexed table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_variable_index_rows = recovered_client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_variable_index_spine ORDER BY id")
        .await
        .expect("read the second recovered indexed table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_variable_index_peer_rows = recovered_client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_variable_index_peer ORDER BY id")
        .await
        .expect("read the recovered peer indexed table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_chain_rows = recovered_client
        .simple_query("SELECT id, value FROM pgwire_typed_chain_spine ORDER BY id")
        .await
        .expect("read the recovered COMMIT AND CHAIN target through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_catalog_mix_rows = recovered_client
        .simple_query("SELECT * FROM pgwire_typed_catalog_mix_renamed")
        .await
        .expect("read the recovered mixed catalog/INSERT view through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_fixed_rows = recovered_client
        .simple_query("SELECT id, tally FROM pgwire_typed_fixed_spine ORDER BY id")
        .await
        .expect("read the recovered cross-table fixed-width target through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_sequence_rows = recovered_client
        .simple_query("SELECT id, value FROM pgwire_typed_sequence_owner ORDER BY id")
        .await
        .expect("read the recovered sequence-default target through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_sequence_state = recovered_client
        .simple_query("SELECT last_value, is_called FROM public.pgwire_typed_sequence_spine")
        .await
        .expect("read the recovered published sequence state through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_created_rows = recovered_client
        .simple_query(
            "SELECT id, parent_id, tally, note FROM pgwire_typed_private_create_owner ORDER BY id",
        )
        .await
        .expect("read the recovered transaction-created table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
                row.get(3).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_created_sequence_state = recovered_client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_create_sequence_final",
        )
        .await
        .expect("read the recovered transaction-created private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_terminal_rename_rows = recovered_client
        .simple_query(
            "SELECT id, row_key, note FROM pgwire_typed_private_terminal_rename_owner ORDER BY id",
        )
        .await
        .expect("read recovered terminal private-sequence rename table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_terminal_rename_sequence_state = recovered_client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_terminal_rename_sequence_final",
        )
        .await
        .expect("read recovered terminal renamed private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_serial_rows = recovered_client
        .simple_query("SELECT id, note FROM pgwire_typed_private_serial_owner ORDER BY id")
        .await
        .expect("read recovered renamed implicit-private-sequence rows through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_serial_sequence_state = recovered_client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_serial_owner_id_seq_final",
        )
        .await
        .expect("read recovered renamed implicit private sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_graph_parent_rows = recovered_client
        .simple_query(
            "SELECT parent_id, note FROM pgwire_typed_private_graph_parent ORDER BY parent_id",
        )
        .await
        .expect("read the recovered transaction-created private FK parent through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_graph_child_rows = recovered_client
        .simple_query(
            "SELECT id, parent_id, note FROM pgwire_typed_private_graph_child ORDER BY id",
        )
        .await
        .expect("read the recovered transaction-created private FK child through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_graph_sequence_state = recovered_client
        .simple_query(
            "SELECT last_value, is_called \
             FROM public.pgwire_typed_private_graph_sequence_final",
        )
        .await
        .expect("read the recovered renamed private graph sequence through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_inline_domain_rows = recovered_client
        .simple_query("SELECT id, amount FROM pgwire_typed_private_inline_domain_owner ORDER BY id")
        .await
        .expect("read the recovered transaction-private domain table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_index_rows = recovered_client
        .simple_query("SELECT id, value FROM pgwire_typed_private_index_spine ORDER BY id")
        .await
        .expect("read the recovered transaction-private explicit-index table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_private_populated_index_rows = recovered_client
        .simple_query("SELECT id, code, region FROM pgwire_typed_populated_index_spine ORDER BY id")
        .await
        .expect("read the recovered populated-table private index through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_canceled_sequence_rows = recovered_client
        .simple_query("SELECT id, value FROM pgwire_typed_canceled_sequence_owner ORDER BY id")
        .await
        .expect("read the recovered canceled private-sequence target through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_canceled_sequence_state = recovered_client
        .simple_query("SELECT last_value, is_called FROM public.pgwire_typed_canceled_sequence")
        .await
        .expect("read the recovered canceled private-sequence state through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_copy_rows = recovered_client
        .simple_query("SELECT id, enabled, body FROM pgwire_typed_copy_spine ORDER BY id")
        .await
        .expect("read the recovered COPY compatibility target through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((
                row.get(0).map(str::to_owned),
                row.get(1).map(str::to_owned),
                row.get(2).map(str::to_owned),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_foreign_key_rows = recovered_client
        .simple_query("SELECT id, parent_id FROM pgwire_typed_fk_child ORDER BY id")
        .await
        .expect("read the recovered FK child table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_indexed_foreign_key_rows = recovered_client
        .simple_query("SELECT id, parent_id FROM pgwire_typed_indexed_fk_child ORDER BY id")
        .await
        .expect("read the recovered indexed FK child table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_indexed_check_rows = recovered_client
        .simple_query("SELECT id, marker FROM pgwire_typed_indexed_check_spine ORDER BY id")
        .await
        .expect("read the recovered indexed CHECK table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_check_rows = recovered_client
        .simple_query("SELECT id FROM pgwire_typed_check_spine ORDER BY id")
        .await
        .expect("read the recovered CHECK-constrained table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    let reopened_domain_rows = recovered_client
        .simple_query("SELECT id, tally FROM pgwire_typed_domain_spine ORDER BY id")
        .await
        .expect("read the recovered domain-typed table through pgwire")
        .into_iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(0).map(str::to_owned), row.get(1).map(str::to_owned)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reopened_rows, rows,
        "fresh-context reopen must reproduce the pgwire simple/extended/reordered table"
    );
    assert_eq!(
        reopened_index_rows, nonunique_index_rows,
        "fresh-context reopen must reproduce the indexed table and its committed rows"
    );
    assert_eq!(
        reopened_variable_index_rows, variable_index_rows,
        "fresh-context reopen must reproduce both indexed targets from one transaction"
    );
    assert_eq!(
        reopened_variable_index_peer_rows, variable_index_peer_rows,
        "fresh-context reopen must reproduce both dense indexed rollovers"
    );
    assert_eq!(
        reopened_chain_rows, chain_rows,
        "fresh-context reopen must reproduce both sides of COMMIT AND CHAIN"
    );
    assert_eq!(
        reopened_catalog_mix_rows, catalog_mix_rows,
        "fresh-context reopen must reproduce the mixed catalog/INSERT transaction"
    );
    assert_eq!(
        reopened_fixed_rows, mixed_rows,
        "fresh-context reopen must reproduce the other table from the indexed plural transaction"
    );
    assert_eq!(
        reopened_sequence_rows, sequence_default_rows,
        "fresh-context reopen must reproduce the plural sequence-default target"
    );
    assert_eq!(
        reopened_sequence_state, sequence_state,
        "fresh-context reopen must reproduce the plural published sequence state"
    );
    assert_eq!(
        reopened_private_created_rows, private_created_rows,
        "fresh-context reopen must reproduce the transaction-created codec-5 table"
    );
    assert_eq!(
        reopened_private_created_sequence_state, private_created_sequence_state,
        "fresh-context reopen must reproduce the transaction-created private sequence rename and final state"
    );
    assert_eq!(
        reopened_private_terminal_rename_rows, private_terminal_rename_rows,
        "fresh-context reopen must reproduce the terminal private sequence rename table"
    );
    assert_eq!(
        reopened_private_terminal_rename_sequence_state, private_terminal_rename_sequence_state,
        "fresh-context reopen must reproduce the terminal private sequence rename state"
    );
    assert_eq!(
        reopened_private_serial_rows, private_serial_rows,
        "fresh-context reopen must reproduce the renamed implicit-private-sequence rows"
    );
    assert_eq!(
        reopened_private_serial_sequence_state, private_serial_sequence_state,
        "fresh-context reopen must reproduce the renamed implicit private sequence state"
    );
    assert_eq!(
        reopened_private_graph_parent_rows, private_graph_parent_rows,
        "fresh-context reopen must reproduce the transaction-created FK parent"
    );
    assert_eq!(
        reopened_private_graph_child_rows, private_graph_child_rows,
        "fresh-context reopen must reproduce the transaction-created FK child"
    );
    assert_eq!(
        reopened_private_graph_sequence_state, private_graph_sequence_state,
        "fresh-context reopen must reproduce the private parent/child sequence rename and final state"
    );
    assert_eq!(
        reopened_private_inline_domain_rows, private_inline_domain_rows,
        "fresh-context reopen must reproduce the transaction-private domain/table/INSERT composition"
    );
    assert_eq!(
        reopened_private_index_rows, private_index_rows,
        "fresh-context reopen must reproduce the transaction-private explicit named index rows"
    );
    assert_eq!(
        reopened_private_populated_index_rows, private_populated_index_rows,
        "fresh-context reopen must reproduce the complete populated-table private index"
    );
    assert_eq!(
        reopened_canceled_sequence_rows, canceled_sequence_rows,
        "fresh-context reopen must preserve cancellation of the provisional sequence row"
    );
    assert_eq!(
        reopened_canceled_sequence_state, canceled_sequence_state,
        "fresh-context reopen must reproduce the committed private sequence effect"
    );
    assert_eq!(
        reopened_copy_rows, copy_rows,
        "fresh-context reopen must reproduce the COPY compatibility rows"
    );
    assert_eq!(
        reopened_foreign_key_rows, foreign_key_rows,
        "fresh-context reopen must reproduce the FK-validated child row"
    );
    assert_eq!(
        reopened_indexed_foreign_key_rows, indexed_foreign_key_rows,
        "fresh-context reopen must reproduce the indexed FK child row"
    );
    assert_eq!(
        reopened_indexed_check_rows, indexed_check_rows,
        "fresh-context reopen must reproduce the indexed CHECK-constrained row"
    );
    assert_eq!(
        reopened_check_rows, check_rows,
        "fresh-context reopen must reproduce the CHECK-validated row"
    );
    assert_eq!(
        reopened_domain_rows, domain_rows,
        "fresh-context reopen must reproduce the domain-typed row"
    );
    drop(recovered_client);
    let _ = recovered_connection_task.await;
    recovered_server_task.abort();
    let _ = recovered_server_task.await;
    drop(recovered);
}

/// STRATA golden gate: a real pgwire client carries alternating NULL/non-NULL values for every
/// exposed logical type, then reaches the dense GPU point route for coarse- and fine-sharded
/// generations. The all-type wire assertion is deliberately separate from the route counter:
/// that counter proves the specialized keyed resident `accounts` projection, not a claim that
/// the current point kernel covers one wide heterogeneous projection.
#[tokio::test]
#[ignore = "requires a local NVIDIA driver and GPU"]
async fn pgwire_gpu_point_route_is_non_vacuous_across_coarse_and_fine_shards() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    let runtime = runtime.snapshot();
    if !runtime.driver_available || runtime.device_count == 0 {
        return;
    }
    async fn run_arm(
        shard_target: usize,
    ) -> (
        Vec<String>,
        Vec<Option<String>>,
        Vec<Vec<Option<String>>>,
        u64,
        u64,
        usize,
    ) {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(shard_target);
        engine.set_shard_index_probe_enabled(true);
        engine.set_shard_batched_point_read_enabled(true);
        engine.set_auto_admit_on_commit(true);
        let mut txn_id = 1u64;
        engine
            .execute_text(
                txn_id,
                "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT, nullable_note INT)",
            )
            .unwrap();
        let row_values = (0..130)
            .map(|id| {
                let note = if id == 1 { "NULL" } else { "9" };
                format!("({id}, {}, {note})", id * 7)
            })
            .collect::<Vec<_>>();
        if shard_target == 32 {
            for value in &row_values {
                txn_id += 1;
                engine
                    .execute_text(
                        txn_id,
                        &format!(
                            "INSERT INTO accounts (id, balance, nullable_note) VALUES {value}"
                        ),
                    )
                    .unwrap();
            }
        } else {
            let values = row_values.join(",");
            txn_id += 1;
            engine
                .execute_text(
                    txn_id,
                    &format!("INSERT INTO accounts (id, balance, nullable_note) VALUES {values}"),
                )
                .unwrap();
        }
        let keyed_device_authoritative_publications = engine.device_authoritative_commits();
        let direct_payload = engine
            .execute_relational_select_text("SELECT balance FROM accounts WHERE id = 100")
            .unwrap();
        assert_eq!(direct_payload.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(direct_payload.fallback_reason, None);
        let direct_null = engine
            .execute_relational_select_text("SELECT nullable_note FROM accounts WHERE id = 1")
            .unwrap();
        assert_eq!(direct_null.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(direct_null.fallback_reason, None);

        // Every logical type crosses pgwire below with a non-NULL value in exactly one row and
        // NULL in the other. The expected text is fixed fixture algebra, not a CPU-executor
        // oracle. The resident point-route proof remains the keyed `accounts` projection above.
        txn_id += 1;
        engine
            .execute_text(
                txn_id,
                "CREATE TABLE wire_all_types (\
                    row_id INT PRIMARY KEY, i2 SMALLINT, i4 INT, i8 BIGINT, amount NUMERIC(12,4),\
                    flag BOOL, note TEXT, day DATE, created_at TIMESTAMP, ident UUID\
                 )",
            )
            .unwrap();
        txn_id += 1;
        engine
            .execute_text(
                txn_id,
                "INSERT INTO wire_all_types VALUES \
                    (1, -7, NULL, -9000000000, 12.3400, NULL, 'left', NULL, \
                     '2000-01-01 00:00:01.234567', NULL),\
                    (2, NULL, 42, NULL, NULL, true, NULL, '1999-12-31', NULL, \
                     '550e8400-e29b-41d4-a716-446655440000')",
            )
            .unwrap();

        let shared = Arc::new(SharedEngine::from_engine(engine));
        let before = shared.gpu_native_activity_snapshot("accounts");
        if before.resident_shards == 0 {
            return (Vec::new(), Vec::new(), Vec::new(), 0, 0, 0);
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let served = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ =
                gpu_db_server::serve_async_with_engine_batching(listener, served, 64, true).await;
        });
        let (client, connection) = tokio_postgres::connect(
            &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
            NoTls,
        )
        .await
        .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let payload_messages = client
            .simple_query("SELECT balance FROM accounts WHERE id = 100")
            .await
            .unwrap();
        let payloads = payload_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => row.get(0).map(str::to_owned),
                _ => None,
            })
            .collect();
        let after = shared.gpu_native_activity_snapshot("accounts");
        let null_messages = client
            .simple_query("SELECT nullable_note FROM accounts WHERE id = 1")
            .await
            .unwrap();
        let nulls = null_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(row.get(0).map(str::to_owned)),
                _ => None,
            })
            .collect();
        let typed_messages = client
            .simple_query(
                "SELECT i2, i4, i8, amount, flag, note, day, created_at, ident \
                 FROM wire_all_types ORDER BY row_id",
            )
            .await
            .unwrap();
        let typed_rows = typed_messages
            .into_iter()
            .filter_map(|message| match message {
                SimpleQueryMessage::Row(row) => Some(
                    (0..9)
                        .map(|column| row.get(column).map(str::to_owned))
                        .collect(),
                ),
                _ => None,
            })
            .collect();
        (
            payloads,
            nulls,
            typed_rows,
            after.sharded_gpu_probe_batches - before.sharded_gpu_probe_batches,
            keyed_device_authoritative_publications,
            before.resident_shards,
        )
    }

    let coarse = run_arm(1024).await;
    let fine = run_arm(32).await;
    assert_eq!(coarse.0, vec!["700"]);
    assert_eq!(coarse.1, vec![None]);
    assert_eq!(
        coarse.2,
        vec![
            vec![
                Some("-7".to_string()),
                None,
                Some("-9000000000".to_string()),
                Some("12.3400".to_string()),
                None,
                Some("left".to_string()),
                None,
                Some("2000-01-01 00:00:01.234567".to_string()),
                None,
            ],
            vec![
                None,
                Some("42".to_string()),
                None,
                None,
                Some("t".to_string()),
                None,
                Some("1999-12-31".to_string()),
                None,
                Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            ],
        ],
        "all-type pgwire rows preserve both typed values and per-column NULLs"
    );
    assert_eq!(
        (fine.0.clone(), fine.1.clone(), fine.2.clone()),
        (coarse.0, coarse.1, coarse.2),
        "coarse and fine resident generations agree on the same closed-form wire fixture"
    );
    assert!(coarse.5 > 0, "coarse-sharded arm has resident shards");
    assert!(
        fine.5 > coarse.5,
        "fine-sharded arm has more shards than coarse arm"
    );
    assert!(
        coarse.3 > 0 && fine.3 > 0,
        "wire reads fired the dense GPU route"
    );
    assert!(
        coarse.4 > 0 && fine.4 > 0,
        "the keyed resident accounts fixture published device-authoritative state before pgwire reads"
    );
}

#[tokio::test]
async fn engine_backed_facade_server_round_trips_over_pgwire() {
    // Bind first so we know the port; the server owns its (non-Send) engine on
    // its own thread, created inside the closure.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let _ = gpu_db_server::serve(listener);
    });

    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .expect("connect to engine-backed facade server");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE accounts (id INT, name TEXT)")
        .await
        .expect("create table");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (1, 'alice')")
        .await
        .expect("insert alice");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (2, 'bob')")
        .await
        .expect("insert bob");

    let messages = client
        .simple_query("SELECT id, name FROM accounts WHERE id = 1")
        .await
        .expect("select");

    let mut rows = 0;
    for message in messages {
        if let SimpleQueryMessage::Row(row) = message {
            assert_eq!(row.get("id"), Some("1"));
            assert_eq!(row.get("name"), Some("alice"));
            rows += 1;
        }
    }
    assert_eq!(rows, 1, "expected exactly one row for id = 1");
}

/// P1-M5: the async-ingress server round-trips a real client over the wire (proves the
/// async pgwire framing + the async→blocking-engine bridge via spawn_blocking).
#[tokio::test]
async fn async_ingress_server_round_trips_over_pgwire() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async(listener).await;
    });

    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .expect("connect to async-ingress server");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .simple_query("CREATE TABLE accounts (id INT, name TEXT)")
        .await
        .expect("create table");
    client
        .simple_query("INSERT INTO accounts (id, name) VALUES (1, 'alice')")
        .await
        .expect("insert alice");

    let messages = client
        .simple_query("SELECT id, name FROM accounts WHERE id = 1")
        .await
        .expect("select");
    let mut rows = 0;
    for message in messages {
        if let SimpleQueryMessage::Row(row) = message {
            assert_eq!(row.get("id"), Some("1"));
            assert_eq!(row.get("name"), Some("alice"));
            rows += 1;
        }
    }
    assert_eq!(rows, 1);
}

/// PRODUCT-001 COPY slice: async ingress uses the same facade-owned typed admission as blocking
/// ingress. Explicit rollback, CSV NULL-vs-empty, COPY TO, parse failure, and client abort all prove
/// that no buffered host rows become a second publication owner.
#[tokio::test]
async fn async_ingress_copy_is_transactional_typed_and_recovers() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async(listener).await;
    });
    let (client, connection) = tokio_postgres::connect(
        &format!("host=127.0.0.1 port={port} user=postgres dbname=postgres"),
        NoTls,
    )
    .await
    .unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute("CREATE TABLE copy_t (id INT, name TEXT)")
        .await
        .unwrap();

    client.batch_execute("BEGIN").await.unwrap();
    let mut rollback_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"9,rolled back\n",
    ))]);
    let rollback_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(rollback_sink);
    rollback_sink.send_all(&mut rollback_data).await.unwrap();
    assert_eq!(rollback_sink.finish().await.unwrap(), 1);
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    client.batch_execute("ROLLBACK").await.unwrap();
    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        0
    );

    let mut typed_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"1,\n2,\"\"\n",
    ))]);
    let typed_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(typed_sink);
    typed_sink.send_all(&mut typed_data).await.unwrap();
    assert_eq!(typed_sink.finish().await.unwrap(), 2);
    let typed_rows = client
        .query("SELECT id, name FROM copy_t ORDER BY id", &[])
        .await
        .unwrap();
    assert_eq!(typed_rows[0].get::<_, Option<String>>(1), None);
    assert_eq!(
        typed_rows[1].get::<_, Option<String>>(1),
        Some(String::new())
    );

    let copy_out_statement = client
        .prepare("COPY copy_t TO STDOUT WITH CSV")
        .await
        .unwrap();
    let copy_out = client
        .copy_out(&copy_out_statement)
        .await
        .unwrap()
        .try_fold(BytesMut::new(), |mut output, chunk| async move {
            output.extend_from_slice(&chunk);
            Ok(output)
        })
        .await
        .unwrap();
    assert_eq!(&copy_out[..], b"1,\n2,\"\"\n");

    let mut invalid_data = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from_static(
        b"not-an-int,bad\n",
    ))]);
    let invalid_sink = client
        .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(invalid_sink);
    invalid_sink.send_all(&mut invalid_data).await.unwrap();
    let invalid = invalid_sink.finish().await.unwrap_err();
    assert_eq!(invalid.code().map(|code| code.code()), Some("22P02"));

    let mut abort_sink = Box::pin(
        client
            .copy_in("COPY copy_t (id, name) FROM STDIN WITH CSV")
            .await
            .unwrap(),
    );
    abort_sink
        .send(Bytes::from_static(b"3,must-not-publish\n"))
        .await
        .unwrap();
    abort_sink.flush().await.unwrap();
    drop(abort_sink);

    assert_eq!(
        client
            .query_one("SELECT COUNT(*) FROM copy_t", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        2
    );
}

/// Device-lifetime hazard gate for canonical COPY: three sequential wire copies followed by two
/// truly overlapping copies, then concurrent NULL-vs-zero point reads from the retained device
/// generation. Counters make both mutation publication and GPU read routing non-vacuous.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a local NVIDIA driver and GPU"]
async fn canonical_copy_survives_sequential_and_concurrent_gpu_lifetime_hazard() {
    let Ok(runtime) = CudaDriverRuntime::probe() else {
        return;
    };
    let runtime = runtime.snapshot();
    if !runtime.driver_available || runtime.device_count == 0 {
        return;
    }

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine.set_shard_index_probe_enabled(true);
    engine.set_shard_batched_point_read_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(1, "CREATE TABLE copy_hazard (id INT PRIMARY KEY, note INT)")
        .unwrap();
    let seed = (0..128)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            2,
            &format!("INSERT INTO copy_hazard (id, note) VALUES {seed}"),
        )
        .unwrap();
    let warm = engine
        .execute_relational_select_text("SELECT note FROM copy_hazard WHERE id = 64")
        .unwrap();
    assert_eq!(warm.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(warm.fallback_reason, None);

    let shared = Arc::new(SharedEngine::from_engine(engine));
    let before = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        before.resident_shards > 0,
        "fixture must be device resident"
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = Arc::clone(&shared);
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async_with_engine_batching(listener, served, 64, true).await;
    });
    let conn = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    let (pre_client, pre_connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = pre_connection.await;
    });
    pre_client
        .simple_query("SELECT id FROM copy_hazard WHERE id = 64")
        .await
        .unwrap();
    let pre_copy_route = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        pre_copy_route.sharded_gpu_probe_batches > before.sharded_gpu_probe_batches,
        "fixture must reach the GPU point route before COPY: before={before:?} pre={pre_copy_route:?}"
    );

    copy_one_hazard_row(&conn, 1_001, None).await;
    assert_eq!(
        pre_client
            .query_one("SELECT COUNT(*) FROM copy_hazard", &[])
            .await
            .expect("first COPY-published generation remains countable")
            .get::<_, i64>(0),
        129,
        "the first acknowledged COPY commit must retain its row"
    );
    let after_first_copy_probe = shared.gpu_native_activity_snapshot("copy_hazard");
    pre_client
        .simple_query("SELECT id FROM copy_hazard WHERE id = 64")
        .await
        .expect("unreferenced NULL COPY data must retain the GPU point route");
    assert!(
        shared
            .gpu_native_activity_snapshot("copy_hazard")
            .sharded_gpu_probe_batches
            > after_first_copy_probe.sharded_gpu_probe_batches,
        "a NULL in unreferenced COPY data must not evict the GPU point route"
    );
    copy_one_hazard_row(&conn, 1_002, Some(0)).await;
    assert_eq!(
        pre_client
            .query_one("SELECT COUNT(*) FROM copy_hazard", &[])
            .await
            .expect("second COPY-published generation remains countable")
            .get::<_, i64>(0),
        130,
        "the second acknowledged COPY commit must retain its row"
    );
    copy_one_hazard_row(&conn, 1_003, Some(7)).await;
    assert_eq!(
        pre_client
            .query_one("SELECT COUNT(*) FROM copy_hazard", &[])
            .await
            .expect("third COPY-published generation remains countable")
            .get::<_, i64>(0),
        131,
        "the third acknowledged COPY commit must retain its row"
    );
    let left = copy_one_hazard_row(&conn, 1_004, None);
    let right = copy_one_hazard_row(&conn, 1_005, Some(0));
    tokio::join!(left, right);

    // Prove that the generation published by the five COPY commits remains usable by the retained
    // GPU point route. Newly appended keys intentionally live outside the dense index's prepared
    // key range, so their NULL-vs-zero checks below prove content while this original key makes
    // execution-target non-vacuity explicit.
    let (probe_client, probe_connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = probe_connection.await;
    });
    let probe = probe_client
        .simple_query("SELECT id FROM copy_hazard WHERE id = 64")
        .await
        .unwrap();
    assert!(probe.iter().any(|message| {
        matches!(message, SimpleQueryMessage::Row(row) if row.get(0) == Some("64"))
    }));
    let copied_count = probe_client
        .query_one("SELECT COUNT(*) FROM copy_hazard", &[])
        .await
        .expect("COPY-published generation remains countable")
        .get::<_, i64>(0);
    assert_eq!(
        copied_count, 133,
        "five acknowledged COPY commits must retain every row before point reads"
    );

    let mut readers = Vec::new();
    for (id, expected) in [
        (1_001, None),
        (1_002, Some(0)),
        (1_003, Some(7)),
        (1_004, None),
        (1_005, Some(0)),
    ] {
        let conn = conn.clone();
        readers.push(tokio::spawn(async move {
            let (client, connection) = tokio_postgres::connect(&conn, NoTls).await.unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            let sql = format!("SELECT note FROM copy_hazard WHERE id = {id}");
            let messages = client.simple_query(&sql).await.unwrap();
            let actual = messages
                .into_iter()
                .find_map(|message| match message {
                    SimpleQueryMessage::Row(row) => {
                        Some(row.get(0).map(|value| value.parse::<i32>().unwrap()))
                    }
                    _ => None,
                })
                .expect("point query returned one row");
            assert_eq!(actual, expected);
        }));
    }
    for reader in readers {
        reader.await.unwrap();
    }

    let after = shared.gpu_native_activity_snapshot("copy_hazard");
    assert!(
        after.open_shard_append_commits > before.open_shard_append_commits
            || after.device_authoritative_commits > before.device_authoritative_commits,
        "COPY must publish a device-maintained generation: before={before:?} after={after:?}"
    );
    assert!(
        after.sharded_gpu_probe_batches > pre_copy_route.sharded_gpu_probe_batches,
        "the COPY-published generation must remain on the GPU point route: pre={pre_copy_route:?} after={after:?}"
    );
}

async fn copy_one_hazard_row(connection: &str, id: i32, note: Option<i32>) {
    let (client, driver) = tokio_postgres::connect(connection, NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = driver.await;
    });
    let note = note.map_or_else(String::new, |value| value.to_string());
    let mut input = stream::iter(vec![Ok::<_, tokio_postgres::Error>(Bytes::from(format!(
        "{id},{note}\n"
    )))]);
    let sink = client
        .copy_in("COPY copy_hazard (id, note) FROM STDIN WITH CSV")
        .await
        .unwrap();
    futures_util::pin_mut!(sink);
    sink.send_all(&mut input).await.unwrap();
    assert_eq!(sink.finish().await.unwrap(), 1);
}

/// P1-M5: many connections are served concurrently as lightweight tasks (no thread per
/// connection), each round-tripping correctly against the shared engine.
#[tokio::test(flavor = "multi_thread")]
async fn async_ingress_serves_many_concurrent_connections() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = gpu_db_server::serve_async(listener).await;
    });
    let conn_str = format!("host=127.0.0.1 port={port} user=postgres dbname=postgres");

    // Seed on one connection.
    {
        let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client.simple_query("CREATE TABLE t (a INT)").await.unwrap();
        for i in 0..25 {
            client
                .simple_query(&format!("INSERT INTO t (a) VALUES ({i})"))
                .await
                .unwrap();
        }
    }

    // 64 concurrent connections each read the table and must see 25 rows.
    let mut handles = Vec::new();
    for _ in 0..64 {
        let conn_str = conn_str.clone();
        handles.push(tokio::spawn(async move {
            let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await.unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            for _ in 0..10 {
                let messages = client.simple_query("SELECT COUNT(*) FROM t").await.unwrap();
                let mut got = None;
                for message in messages {
                    if let SimpleQueryMessage::Row(row) = message {
                        got = row.get(0).map(|s| s.to_string());
                    }
                }
                assert_eq!(got.as_deref(), Some("25"));
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
}
