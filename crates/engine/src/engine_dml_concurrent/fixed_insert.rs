//! INSERT-001 typed fixed-width candidate preparation for the existing serial wave.
//!
//! This owner stops at a sealed pre-WAL candidate. In particular, it has no replication, WAL,
//! status, ledger, or publication authority; [`super::canonical`] remains the sole serial-wave
//! canonical chain.

use super::canonical::WaveCanonicalOperation;
#[cfg(test)]
use super::CanonicalRequest;
#[cfg(test)]
use super::CommitWaveDone;
#[cfg(test)]
use super::OfflockPreparedDml;
use super::{CatalogSnapshot, CommitWaveItem, Engine, ExecuteError, Index};

pub(super) struct FixedInsertPreWal<'a> {
    plan: crate::engine_residency::PreparedI32OpenShardAppendPlan<'a>,
    proposal_payload: std::sync::Arc<[u8]>,
    bound: crate::wal_binary::BoundBinaryInsert,
    request_digest: gpu_db_wal::CanonicalDigest,
    row_count: u64,
}

pub(super) struct FixedInsertApply<'a> {
    plan: crate::engine_residency::PreparedI32OpenShardAppendPlan<'a>,
    request_digest: gpu_db_wal::CanonicalDigest,
    row_count: u64,
}

// Keeping the sealed plan inline avoids a successful-route heap allocation between preflight and
// the canonical cut; this short-lived enum never crosses a queue or public boundary.
#[allow(clippy::large_enum_variant)]
pub(super) enum FixedInsertPreflightResult<'a> {
    /// A direct off-lock carrier lost its catalog/request binding. Discard it and run the
    /// established full `prepare_dml` once from `CommitWaveItem::cmd` before WAL.
    FullReprepare,
    Ready(FixedInsertPreWal<'a>),
    /// An identity proof failed before the canonical cut. It must never re-enter legacy apply,
    /// because that path would derive a fresh row-id range after the sealed proof was rejected.
    PreWalFailure(ExecuteError),
    /// A device-authoritative resident table cannot safely fall through after its sealed plan
    /// declined: legacy would reach the same geometry after sequence/WAL and could only panic.
    /// Refuse retryably before any durable or allocator effect instead.
    RetryableDecline,
}

impl<'a> FixedInsertPreWal<'a> {
    pub(super) fn into_canonical_operation_and_apply(
        self,
    ) -> (WaveCanonicalOperation, FixedInsertApply<'a>) {
        let FixedInsertPreWal {
            plan,
            proposal_payload,
            bound,
            request_digest,
            row_count,
        } = self;
        (
            WaveCanonicalOperation::FixedInsert {
                proposal_payload,
                bound,
            },
            FixedInsertApply {
                plan,
                request_digest,
                row_count,
            },
        )
    }
}

impl FixedInsertApply<'_> {
    #[cfg(test)]
    pub(super) fn fail_next_post_wal_apply(engine: &Engine) {
        engine
            .fail_next_fixed_insert_post_wal_apply
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(super) fn request_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.request_digest
    }

    pub(super) fn row_count(&self) -> u64 {
        self.row_count
    }

    pub(super) fn apply(
        self,
        engine: &Engine,
        commit_seq: Index,
        proposed_range: crate::wal_binary::ProposedRowIdRange,
    ) -> String {
        let table = self.plan.table_name().to_string();
        #[cfg(test)]
        if engine
            .fail_next_fixed_insert_post_wal_apply
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            panic!(
                "injected fixed INSERT post-WAL device apply failure: restart recovery required"
            );
        }
        engine
            .apply_prepared_i32_open_shard_append(
                self.plan,
                crate::engine_residency::AppendCreatedBy::InsertUniform(commit_seq),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "commit-path invariant violation: fixed INSERT device apply at commit_seq \
                     {commit_seq} failed after canonical WAL/status/ledger: {error:?}"
                )
            });
        engine
            .read_state
            .mvcc
            .consume_proposed_row_id_range(proposed_range)
            .unwrap_or_else(|error| {
                panic!(
                    "commit-path invariant violation: fixed INSERT allocator consumption at \
                     commit_seq {commit_seq} drifted after device apply: {error}"
                )
            });
        table
    }
}

impl Engine {
    /// Consume a fixed carrier only if every off-lock proof is still live. A mismatch discards
    /// the delta-free carrier and forces the established full prepare before WAL; accepted work
    /// retains only typed source/template metadata through the canonical apply cut.
    pub(super) fn prepare_fixed_insert_pre_wal<'a>(
        &'a self,
        item: &mut CommitWaveItem,
        wave_catalog: &CatalogSnapshot,
        wave_catalog_seq: Index,
        next_row_id: u64,
        reuse_eligible: bool,
    ) -> FixedInsertPreflightResult<'a> {
        // The typed route's WAL record is binary. A gate change invalidates this direct carrier;
        // discard it and use the ordinary full preparation before any canonical mutation.
        if !self.binary_wal_records_enabled() {
            return FixedInsertPreflightResult::FullReprepare;
        }
        let Some(prepared) = item.offlock_prepared.take() else {
            return FixedInsertPreflightResult::FullReprepare;
        };
        let preflight = match prepared.into_fixed_insert_preflight(
            &item.request,
            &item.write_set,
            wave_catalog,
            wave_catalog_seq,
            item.read_snapshot,
            reuse_eligible,
        ) {
            Ok(preflight) => preflight,
            Err(_) => return FixedInsertPreflightResult::FullReprepare,
        };
        let (source, template, request_digest, row_count) = preflight.into_parts();
        let proposed_range =
            match crate::wal_binary::ProposedRowIdRange::new(next_row_id, row_count) {
                Ok(proposed_range) => proposed_range,
                Err(error) => {
                    return FixedInsertPreflightResult::PreWalFailure(ExecuteError::Engine(error));
                }
            };
        let row_ids = match proposed_range.exact_row_ids() {
            Ok(row_ids) => row_ids,
            Err(error) => {
                return FixedInsertPreflightResult::PreWalFailure(ExecuteError::Engine(error));
            }
        };
        let resident_authoritative = self.table_device_authoritative(source.table_name())
            || self
                .table_chunk_authoritative(source.table_name())
                .is_some();
        match self.prepare_prepared_i32_open_shard_append(
            source,
            crate::engine_residency::PreparedI32AppendRowIds::exact(row_ids),
        ) {
            Ok(plan) => {
                // Bind before accepting the typed route. A malformed template is still a
                // pre-WAL decline, so dropping the plan releases its reservations and restores
                // the unchanged general delta rather than leaving a fallible bind after delta
                // ownership has been discarded. A bound CREATE sentinel is the exception: it
                // cannot restore legacy after its device-native first-write geometry was sealed.
                let bound_bootstrap = plan.is_bound_bootstrap_sentinel();
                match template.bind(proposed_range) {
                    Ok(bound) => {
                        let proposal_payload = bound.proposal_payload();
                        FixedInsertPreflightResult::Ready(FixedInsertPreWal {
                            plan,
                            proposal_payload,
                            bound,
                            request_digest,
                            row_count: u64::from(row_count),
                        })
                    }
                    Err(_) => {
                        drop(plan);
                        if bound_bootstrap {
                            FixedInsertPreflightResult::RetryableDecline
                        } else {
                            FixedInsertPreflightResult::FullReprepare
                        }
                    }
                }
            }
            Err(
                crate::engine_residency::PreparedI32AppendPrepareError::RetryableBoundBootstrapResource
                | crate::engine_residency::PreparedI32AppendPrepareError::RetryableBoundBootstrapState,
            ) => {
                // A live CREATE sentinel has already bound this sealed source to the only
                // device-native first-write path. It must not restore legacy and let that path
                // claim canonical WAL/row identities before encountering the same resource or
                // descriptor failure in publication.
                FixedInsertPreflightResult::RetryableDecline
            }
            Err(crate::engine_residency::PreparedI32AppendPrepareError::UnsupportedShape) => {
                if resident_authoritative {
                    FixedInsertPreflightResult::RetryableDecline
                } else {
                    FixedInsertPreflightResult::FullReprepare
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_command, Command, MutationRequest, SqlValue};

    fn read_device_u64(memory: &crate::CudaResidentDeviceMemory, slot: usize) -> u64 {
        let words = memory
            .read_resident_i32_column((slot * std::mem::size_of::<u64>()) as u64, 2)
            .expect("read fixed INSERT row-id sidecar");
        (words[0] as u32 as u64) | ((words[1] as u32 as u64) << 32)
    }

    fn fixed_values(rows: usize) -> String {
        (0..rows)
            .map(|row| format!("({}, {})", row + 1, -((row as i32) + 1)))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn fixed_insert_sql(rows: usize) -> String {
        format!(
            "INSERT INTO accounts (id, balance) VALUES {}",
            fixed_values(rows)
        )
    }

    #[test]
    fn fixed_preflight_decline_restores_legacy_without_duplicate_wal_or_row_ids() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let sql = fixed_insert_sql(1_000);
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let wal_before = engine.durable_wal_records().len();

        // No resident open shard exists, so fixed preflight must decline and restore the one
        // ordinary delta path. The observable commit remains exactly one statement/record/range.
        engine.execute_dml_concurrent(2, &sql).unwrap();
        assert_eq!(
            engine.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        engine.execute_dml_concurrent(2, &sql).unwrap();
        assert_eq!(
            engine.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        let Command::Select(select) =
            parse_command("SELECT id, balance FROM accounts ORDER BY id").unwrap()
        else {
            unreachable!("test SELECT must parse");
        };
        let rows = recovered.execute_relational_select(&select).unwrap().rows;
        assert_eq!(rows.len(), 1_000);
        assert_eq!(rows.row(0), &[SqlValue::Int4(1), SqlValue::Int4(-1)]);
        assert_eq!(
            rows.row(999),
            &[SqlValue::Int4(1_000), SqlValue::Int4(-1_000)]
        );
    }

    #[test]
    fn fixed_i32_catalog_drift_discards_the_direct_carrier_before_wal() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let sql = fixed_insert_sql(2);
        let command = parse_command(&sql).unwrap();
        let prepared_catalog = engine.catalog_snapshot();
        let batch = crate::prepared_insert_batch::try_prepare_direct_fixed_insert_batch(
            &command,
            &prepared_catalog,
            prepared_catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("exact fixed INSERT has a direct carrier before catalog drift");
        let request = CanonicalRequest::from_text(&engine, &sql);
        let prepared =
            OfflockPreparedDml::fixed_insert(batch, &request, engine.committed_seq()).unwrap();
        let mut item = CommitWaveItem {
            txn_id: 2,
            cmd: command,
            request,
            write_set: crate::write_path::WriteSet {
                tables: std::collections::BTreeSet::from(["accounts".to_string()]),
                rows: Vec::new(),
                unique_slots: Vec::new(),
                unique_slots_i32: Vec::new(),
            },
            read_snapshot: engine.committed_seq(),
            prepared_catalog_seq: prepared_catalog.commit_seq,
            expected_catalog_version: None,
            offlock_prepared: Some(prepared),
            #[cfg(feature = "probe-timing")]
            fixed_insert_typed: false,
            binary_wal_template: None,
            table_access: None,
            outcome: std::sync::Arc::new(CommitWaveDone::default()),
        };
        engine
            .execute_text(3, "CREATE TABLE catalog_drift (value int4)")
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let wave_catalog = engine.catalog_snapshot();
        let result = engine.prepare_fixed_insert_pre_wal(
            &mut item,
            &wave_catalog,
            wave_catalog.commit_seq,
            row_id_before,
            false,
        );
        assert!(matches!(result, FixedInsertPreflightResult::FullReprepare));
        assert!(item.offlock_prepared.is_none());
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    }

    #[cfg(feature = "probe-timing")]
    #[test]
    fn static_ineligible_insert_builds_one_legacy_delta_and_no_direct_carrier() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE notes (id int4, body text)")
            .unwrap();
        let before = engine.insert_probe_snapshot();
        engine
            .execute_dml_concurrent(2, "INSERT INTO notes VALUES (7, 'legacy')")
            .unwrap();
        let probe = engine.insert_probe_snapshot().delta_since(before);
        assert_eq!(probe.direct_fixed_insert_carriers, 0);
        assert_eq!(probe.legacy_insert_delta_builds, 1);
        assert_eq!(probe.predicted_row_keys_materialized, 1);
    }

    #[test]
    fn fixed_insert_cutover_has_one_serial_canonical_owner_and_one_allocator_consumer() {
        let fixed = include_str!("fixed_insert.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production fixed INSERT owner precedes its tests");
        let canonical = include_str!("canonical.rs");
        let wave = include_str!("wave.rs");
        assert_eq!(fixed.matches(".append_canonical(").count(), 0);
        assert_eq!(fixed.matches(".propose(").count(), 0);
        assert_eq!(fixed.matches("record_transaction_status").count(), 0);
        assert_eq!(fixed.matches("consume_proposed_row_id_range(").count(), 1);
        assert_eq!(canonical.matches(".append_canonical(").count(), 1);
        assert_eq!(canonical.matches(".propose(").count(), 1);
        assert_eq!(
            canonical
                .matches("record_transaction_status_digest_outcome(")
                .count(),
            1
        );
        assert_eq!(wave.matches("fn sequence_commit_wave_inner").count(), 1);
        assert!(wave.contains("prepare_fixed_insert_pre_wal"));
        assert!(wave.contains("append_canonical_wave_operation"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn fixed_i32_cutover_commits_exact_wal_digest_rows_and_recovery() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(2_048);
        engine.set_fused_apply_enabled(true);
        assert!(
            engine.binary_wal_records_enabled(),
            "the product local constructor must activate the typed fixed-INSERT cutover"
        );
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let bootstrap = engine
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .filter(|shards| shards.len() == 1)
            .map(|shards| shards[0].clone())
            .expect("CREATE auto-admission publishes the sole bootstrap shard");
        assert_eq!(bootstrap.shard_id, 0);
        assert_eq!(bootstrap.row_start, 0);
        assert_eq!(bootstrap.row_count, 0);
        assert_eq!(bootstrap.capacity, 0);
        assert!(bootstrap.device_memory_proof.is_some());
        assert!(bootstrap.created_by_region.is_none());
        assert!(bootstrap.row_id_region.is_none());
        assert!(engine
            .read_state
            .residency
            .shard_created_by_memory
            .get(&("accounts".to_string(), 0))
            .is_none());
        assert!(engine
            .read_state
            .residency
            .shard_row_id_memory
            .get(&("accounts".to_string(), 0))
            .is_none());

        let sql = fixed_insert_sql(1_000);
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let wal_before = engine.durable_wal_records().len();
        let append_hits_before = engine.open_shard_append_hits();
        let fused_hits_before = engine.fused_apply_hits();
        let device_commits_before = engine
            .read_state
            .residency
            .device_authoritative_commits
            .load(std::sync::atomic::Ordering::Acquire);
        #[cfg(feature = "probe-timing")]
        let probe_before = engine.insert_probe_snapshot();
        let decode_scope = crate::engine_canonical_operation::LiveBinaryDecodeScope::begin();
        engine.execute_dml_concurrent(3, &sql).unwrap();
        assert_eq!(
            decode_scope.count(),
            0,
            "the sealed bound canonical builder must not decode/re-encode the live binary INSERT"
        );
        #[cfg(feature = "probe-timing")]
        {
            let probe = engine.insert_probe_snapshot().delta_since(probe_before);
            assert_eq!(probe.direct_fixed_insert_carriers, 1);
            assert_eq!(probe.legacy_insert_delta_builds, 0);
            assert_eq!(probe.predicted_row_keys_materialized, 0);
        }

        assert_eq!(
            engine.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert_eq!(
            engine
                .read_state
                .residency
                .device_authoritative_commits
                .load(std::sync::atomic::Ordering::Acquire),
            device_commits_before + 1,
            "one typed statement advances authority exactly once"
        );
        assert_eq!(engine.open_shard_append_hits(), append_hits_before + 1);
        assert_eq!(
            engine.fused_apply_hits(),
            fused_hits_before,
            "the zero-capacity sentinel must roll over, never take the in-place fused branch"
        );
        let after_shards = engine
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .cloned()
            .expect("typed apply retains the bootstrap and publishes a rollover shard");
        assert_eq!(after_shards.len(), 2, "bootstrap never appends in place");
        assert_eq!(after_shards[0], bootstrap);
        let after_shard = &after_shards[1];
        assert_eq!(after_shard.shard_id, 1);
        assert_eq!(after_shard.row_start, 0);
        assert_eq!(after_shard.row_count, 1_000);
        let row_ids = after_shard
            .row_id_region
            .as_ref()
            .expect("typed append retains exact row identities");
        assert_eq!(read_device_u64(row_ids, 0), row_id_before);
        assert_eq!(read_device_u64(row_ids, 999), row_id_before + 999);
        let created_by = after_shard
            .created_by_region
            .as_ref()
            .expect("rollover publishes exact creation stamps with the first identities");
        let committed_seq = engine.committed_seq();
        assert_eq!(read_device_u64(created_by, 0), committed_seq);
        assert_eq!(read_device_u64(created_by, 999), committed_seq);
        let envelope = gpu_db_wal::decode_canonical_record_payload(
            &engine.durable_wal_records().last().unwrap().payload,
        )
        .unwrap()
        .expect("fixed INSERT commits a canonical record");
        assert_eq!(
            envelope.header.request_digest,
            gpu_db_wal::canonical_request_digest(sql.as_bytes())
        );
        assert_eq!(envelope.outcome.affected_rows, 1_000);
        assert_eq!(envelope.header.allocator_high_water, row_id_before + 1_000);

        let Command::Select(live_select) =
            parse_command("SELECT id, balance FROM accounts WHERE id = 1000").unwrap()
        else {
            unreachable!("test SELECT must parse");
        };
        assert_eq!(
            engine.execute_relational_select(&live_select).unwrap().rows,
            vec![vec![SqlValue::Int4(1_000), SqlValue::Int4(-1_000)]],
            "the live device-authoritative generation exposes the typed row before recovery"
        );

        // Exact retry is idempotent through the normal terminal-status index.
        engine.execute_dml_concurrent(3, &sql).unwrap();
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert_eq!(
            engine.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );
        let Command::Select(select) =
            parse_command("SELECT id, balance FROM accounts WHERE id = 1000").unwrap()
        else {
            unreachable!("test SELECT must parse");
        };
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            vec![vec![SqlValue::Int4(1_000), SqlValue::Int4(-1_000)]]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn fixed_i32_create_bootstrap_budget_decline_is_retryable_before_wal() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(2_048);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        assert!(
            !engine.table_device_authoritative("accounts"),
            "CREATE admission has a live device descriptor before the first typed append"
        );
        let bootstrap = engine
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .filter(|shards| shards.len() == 1)
            .map(|shards| shards[0].clone())
            .expect("CREATE auto-admission publishes the sole bootstrap sentinel");
        assert_eq!(bootstrap.shard_id, 0);
        assert_eq!(bootstrap.row_start, 0);
        assert_eq!(bootstrap.row_count, 0);
        assert_eq!(bootstrap.capacity, 0);
        assert!(bootstrap.created_by_region.is_none());
        assert!(bootstrap.row_id_region.is_none());
        assert!(bootstrap.deleted_by_region.is_none());
        engine.set_relational_residency_budget_bytes(0, 0);
        let durable_wal_before = engine.durable_wal_records().len();
        let buffered_wal_before = engine.wal_buffered_count();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let append_before = engine.open_shard_append_hits();
        let authority_before = engine.device_authoritative_commits();
        let budget_declines_before = engine.rollover_budget_declines();
        #[cfg(feature = "probe-timing")]
        let probe_before = engine.insert_probe_snapshot();

        let error = engine
            .execute_dml_concurrent(2, &fixed_insert_sql(1_000))
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), durable_wal_before);
        assert_eq!(engine.wal_buffered_count(), buffered_wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.open_shard_append_hits(), append_before);
        assert_eq!(engine.device_authoritative_commits(), authority_before);
        assert!(engine.rollover_budget_declines() > budget_declines_before);
        assert_eq!(
            engine
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .cloned(),
            Some(vec![bootstrap]),
            "retryable bootstrap resource decline must not allocate or publish a new shard"
        );
        for sidecar in [
            &engine.read_state.residency.shard_created_by_memory,
            &engine.read_state.residency.shard_row_id_memory,
            &engine.read_state.residency.shard_deleted_by_memory,
        ] {
            assert!(sidecar.get(&("accounts".to_string(), 0)).is_none());
        }
        #[cfg(feature = "probe-timing")]
        {
            let probe = engine.insert_probe_snapshot().delta_since(probe_before);
            assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
            assert_eq!(probe.fixed_insert_legacy_commit_validation_reresolves, 0);
            assert_eq!(probe.fixed_insert_retryable_declines, 1);
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn fixed_i32_default_product_durable_reopen_preserves_bound_rows_and_retry() {
        let path = crate::tests::test_wal_path("fixed-insert-default-product-durable");
        let sql = fixed_insert_sql(1_000);
        let row_id_before;
        let durable_records;
        {
            let engine = Engine::with_durable_wal_segment(&path);
            engine.set_shard_residency_enabled(true);
            engine.set_shard_size_target(2_048);
            engine.set_fused_apply_enabled(true);
            assert!(engine.wal_is_durable());
            assert!(
                engine.binary_wal_records_enabled(),
                "the durable product constructor must activate fixed INSERT binary WAL"
            );
            engine
                .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
                .unwrap();
            assert!(engine
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .is_some_and(|shards| {
                    shards.len() == 1
                        && shards[0].shard_id == 0
                        && shards[0].row_start == 0
                        && shards[0].row_count == 0
                        && shards[0].capacity == 0
                        && shards[0].row_id_region.is_none()
                }));
            row_id_before = engine.read_state.mvcc.current_row_id();
            let wal_before = engine.durable_wal_records().len();
            let decode_scope = crate::engine_canonical_operation::LiveBinaryDecodeScope::begin();
            engine.execute_dml_concurrent(3, &sql).unwrap();
            assert_eq!(decode_scope.count(), 0);
            assert_eq!(
                engine.read_state.mvcc.current_row_id(),
                row_id_before + 1_000
            );
            assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
            // The terminal status remains idempotent on the physical durable segment too.
            engine.execute_dml_concurrent(3, &sql).unwrap();
            assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
            durable_records = engine.durable_wal_records().len();
        }

        let reopened = Engine::open_durable_wal_segment_auto(&path).unwrap();
        assert!(reopened.wal_is_durable());
        assert!(reopened.binary_wal_records_enabled());
        assert_eq!(
            reopened.read_state.mvcc.current_row_id(),
            row_id_before + 1_000
        );
        assert_eq!(reopened.durable_wal_records().len(), durable_records);
        reopened.execute_dml_concurrent(3, &sql).unwrap();
        assert_eq!(reopened.durable_wal_records().len(), durable_records);
        assert_eq!(
            reopened.read_state.mvcc.current_row_id(),
            row_id_before + 1_000,
            "exact retry remains terminal after process-equivalent reopen"
        );
        let Command::Select(select) =
            parse_command("SELECT id, balance FROM accounts WHERE id = 1000").unwrap()
        else {
            unreachable!("test SELECT must parse");
        };
        assert_eq!(
            reopened.execute_relational_select(&select).unwrap().rows,
            vec![vec![SqlValue::Int4(1_000), SqlValue::Int4(-1_000)]],
            "physical reopen replays the bound multi-row product route exactly once"
        );
        let _ = std::fs::remove_file(gpu_db_wal::wal_tail_offset_path(&path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn fixed_i32_post_wal_apply_fault_wedges_without_legacy_fallback() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        engine
            .execute_dml_concurrent(2, "INSERT INTO accounts VALUES (0, 0)")
            .unwrap();
        let snapshot = engine
            .populate_relational_residency_snapshot("accounts")
            .unwrap();
        assert!(snapshot.device_memory_proof.is_some());
        engine.set_table_device_authoritative("accounts", true);
        let wal_before = engine.durable_wal_records().len();
        let buffered_wal_before = engine.wal_buffered_count();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let appends_before = engine.open_shard_append_hits();
        let authority_before = engine
            .read_state
            .residency
            .device_authoritative_commits
            .load(std::sync::atomic::Ordering::Acquire);
        FixedInsertApply::fail_next_post_wal_apply(&engine);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine
                .execute_dml_concurrent(3, &fixed_insert_sql(2))
                .unwrap();
        }));
        assert!(
            result.is_err(),
            "post-WAL typed apply is fatal, never legacy fallback"
        );
        assert!(engine.is_commit_path_poisoned());
        // The wave panics before its group-fsync tail, but the canonical record is already
        // buffered. It must never be rolled back or converted into a legacy retry.
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.wal_buffered_count(), buffered_wal_before + 1);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.open_shard_append_hits(), appends_before);
        assert_eq!(
            engine
                .read_state
                .residency
                .device_authoritative_commits
                .load(std::sync::atomic::Ordering::Acquire),
            authority_before,
            "post-WAL failure must not fall through to legacy publication"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn fixed_i32_authoritative_plan_decline_is_retryable_before_wal() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(2);
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        engine
            .execute_dml_concurrent(2, "INSERT INTO accounts VALUES (0, 0)")
            .unwrap();
        let snapshot = engine
            .populate_relational_residency_snapshot("accounts")
            .unwrap();
        assert!(snapshot.device_memory_proof.is_some());
        engine.set_table_device_authoritative("accounts", true);
        // The two-row candidate cannot fit the one-row open shard and rollover has no budget.
        engine.set_relational_residency_budget_bytes(0, 0);
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let append_before = engine.open_shard_append_hits();
        let authority_before = engine
            .read_state
            .residency
            .device_authoritative_commits
            .load(std::sync::atomic::Ordering::Acquire);
        let error = engine
            .execute_dml_concurrent(3, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
            .unwrap_err();
        assert!(matches!(error, crate::ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.open_shard_append_hits(), append_before);
        assert_eq!(
            engine
                .read_state
                .residency
                .device_authoritative_commits
                .load(std::sync::atomic::Ordering::Acquire),
            authority_before
        );
    }

    #[test]
    fn fixed_i32_identity_exhaustion_fails_closed_before_wal() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let sql = fixed_insert_sql(2);
        let command = parse_command(&sql).unwrap();
        let catalog = engine.catalog_snapshot();
        let batch = crate::prepared_insert_batch::try_prepare_direct_fixed_insert_batch(
            &command,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("exact int4 INSERT remains directly eligible");
        let write_set = crate::write_path::WriteSet {
            tables: std::collections::BTreeSet::from(["accounts".to_string()]),
            rows: Vec::new(),
            unique_slots: Vec::new(),
            unique_slots_i32: Vec::new(),
        };
        let request = CanonicalRequest::from_text(&engine, &sql);
        let prepared =
            OfflockPreparedDml::fixed_insert(batch, &request, engine.committed_seq()).unwrap();
        let mut item = CommitWaveItem {
            txn_id: 2,
            cmd: command,
            request,
            write_set,
            read_snapshot: engine.committed_seq(),
            prepared_catalog_seq: catalog.commit_seq,
            expected_catalog_version: None,
            offlock_prepared: Some(prepared),
            #[cfg(feature = "probe-timing")]
            fixed_insert_typed: false,
            binary_wal_template: None,
            table_access: None,
            outcome: std::sync::Arc::new(CommitWaveDone::default()),
        };
        engine
            .read_state
            .mvcc
            .next_row_id
            .store(u64::MAX - 1, std::sync::atomic::Ordering::Release);
        let wal_before = engine.durable_wal_records().len();
        let append_before = engine.open_shard_append_hits();
        let authority_before = engine
            .read_state
            .residency
            .device_authoritative_commits
            .load(std::sync::atomic::Ordering::Acquire);

        let result = engine.prepare_fixed_insert_pre_wal(
            &mut item,
            &catalog,
            catalog.commit_seq,
            u64::MAX - 1,
            true,
        );
        assert!(matches!(
            result,
            FixedInsertPreflightResult::PreWalFailure(crate::ExecuteError::Engine(_))
        ));
        assert!(
            item.offlock_prepared.is_none(),
            "no legacy carrier may remain"
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), u64::MAX - 1);
        assert_eq!(engine.open_shard_append_hits(), append_before);
        assert_eq!(
            engine
                .read_state
                .residency
                .device_authoritative_commits
                .load(std::sync::atomic::Ordering::Acquire),
            authority_before
        );
    }

    #[test]
    fn stale_catalog_expectation_rejects_named_fixed_shape_before_wal() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let expected_catalog_version = engine.catalog_snapshot().commit_seq;
        engine
            .execute_text(2, "CREATE TABLE catalog_expectation_bump (id int4)")
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();

        let error = engine
            .submit_transaction(
                3,
                MutationRequest::new(
                    gpu_db_sql::ParsedCommand::parse(
                        "INSERT INTO accounts (id, balance) VALUES (1, 10)",
                    )
                    .unwrap(),
                )
                .with_expected_catalog_version(expected_catalog_version),
            )
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Unsupported(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    }
}
