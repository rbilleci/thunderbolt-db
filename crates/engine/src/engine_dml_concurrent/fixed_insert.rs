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

pub(super) struct TypedInsertPreWal<'a> {
    bound_plan: crate::engine_insert_plan::BoundDeviceInsertPlan<'a>,
    request_digest: gpu_db_wal::CanonicalDigest,
    row_count: u64,
}

pub(super) struct TypedInsertApply<'a> {
    plan: crate::engine_residency::DeviceInsertPlan<'a>,
    request_digest: gpu_db_wal::CanonicalDigest,
    row_count: u64,
}

// Keeping the sealed plan inline avoids a successful-route heap allocation between preflight and
// the canonical cut; this short-lived enum never crosses a queue or public boundary.
#[allow(clippy::large_enum_variant)]
pub(super) enum TypedInsertPreflightResult<'a> {
    /// A direct off-lock carrier lost its catalog/request binding. Discard it and run the
    /// established full `prepare_dml` once from `CommitWaveItem::cmd` before WAL.
    FullReprepare,
    Ready(TypedInsertPreWal<'a>),
    /// An identity proof failed before the canonical cut. It must never re-enter legacy apply,
    /// because that path would derive a fresh row-id range after the sealed proof was rejected.
    PreWalFailure(ExecuteError),
    /// A device-authoritative resident table cannot safely fall through after its sealed plan
    /// declined: legacy would reach the same geometry after sequence/WAL and could only panic.
    /// Refuse retryably before any durable or allocator effect instead.
    RetryableDecline,
}

impl<'a> TypedInsertPreWal<'a> {
    pub(super) fn into_canonical_operation_and_apply(
        self,
    ) -> (WaveCanonicalOperation, TypedInsertApply<'a>) {
        let TypedInsertPreWal {
            bound_plan,
            request_digest,
            row_count,
        } = self;
        let (plan, bound) = bound_plan.into_parts();
        let proposal_payload = bound.proposal_payload();
        (
            WaveCanonicalOperation::TypedInsert {
                proposal_payload,
                bound,
            },
            TypedInsertApply {
                plan,
                request_digest,
                row_count,
            },
        )
    }
}

impl TypedInsertApply<'_> {
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
        self.plan
            .apply(
                engine,
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
    pub(super) fn prepare_typed_insert_pre_wal<'a>(
        &'a self,
        item: &mut CommitWaveItem,
        wave_catalog: &CatalogSnapshot,
        wave_catalog_seq: Index,
        next_row_id: u64,
        reuse_eligible: bool,
    ) -> TypedInsertPreflightResult<'a> {
        // The typed route's WAL record is binary. A gate change invalidates this direct carrier;
        // discard it and use the ordinary full preparation before any canonical mutation.
        if !self.binary_wal_records_enabled() {
            return TypedInsertPreflightResult::FullReprepare;
        }
        let Some(prepared) = item.offlock_prepared.take() else {
            return TypedInsertPreflightResult::FullReprepare;
        };
        let preflight = match prepared.into_typed_insert_preflight(
            &item.request,
            &item.write_set,
            wave_catalog,
            wave_catalog_seq,
            item.read_snapshot,
            reuse_eligible,
        ) {
            Ok(preflight) => preflight,
            Err(_) => return TypedInsertPreflightResult::FullReprepare,
        };
        let (prepared_plan, request_digest) = preflight.into_parts();
        let row_count = prepared_plan.row_count();
        let row_id_proposal = match prepared_plan.prepare_row_id_proposal(next_row_id) {
            Ok(row_ids) => row_ids,
            Err(error) => {
                return TypedInsertPreflightResult::PreWalFailure(ExecuteError::Engine(error));
            }
        };
        let resident_authoritative = self.table_device_authoritative(prepared_plan.table_name())
            || self
                .table_chunk_authoritative(prepared_plan.table_name())
                .is_some();
        match prepared_plan.bind_for_current_commit(self, row_id_proposal) {
            Ok(bound_plan) => TypedInsertPreflightResult::Ready(TypedInsertPreWal {
                bound_plan,
                request_digest,
                row_count: u64::from(row_count),
            }),
            Err(crate::engine_insert_plan::BindDeviceInsertPlanError::Template {
                bound_bootstrap: true,
            }) => TypedInsertPreflightResult::RetryableDecline,
            Err(crate::engine_insert_plan::BindDeviceInsertPlanError::Template {
                bound_bootstrap: false,
            }) => TypedInsertPreflightResult::FullReprepare,
            Err(crate::engine_insert_plan::BindDeviceInsertPlanError::Device(
                crate::engine_residency::DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource
                | crate::engine_residency::DeviceInsertPlanPrepareError::RetryableBoundBootstrapState,
            )) => {
                // A live CREATE sentinel has already bound this sealed source to the only
                // device-native first-write path. It must not restore legacy and let that path
                // claim canonical WAL/row identities before encountering the same resource or
                // descriptor failure in publication.
                TypedInsertPreflightResult::RetryableDecline
            }
            Err(crate::engine_insert_plan::BindDeviceInsertPlanError::Device(
                crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape,
            )) => {
                if resident_authoritative {
                    TypedInsertPreflightResult::RetryableDecline
                } else {
                    TypedInsertPreflightResult::FullReprepare
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
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn production_wave_routes_reordered_all_fixed_scalars_without_a_legacy_delta() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(8);
        engine.set_fused_apply_enabled(true);
        engine
            .execute_text(
                1,
                "CREATE TABLE scalar_wave (small int2, integer int4, day date, big int8, moment timestamp, amount numeric(10,2), token uuid, flag bool)",
            )
            .unwrap();
        let many = "INSERT INTO scalar_wave (flag, token, amount, moment, big, day, integer, small) VALUES \
            (true, '00112233-4455-6677-8899-aabbccddeeff', 12.34, '2026-07-27 12:34:56', 99, '2026-07-27', -44, -7), \
            (false, 'ffeeddcc-bbaa-9988-7766-554433221100', -0.25, '2026-07-28 00:00:00', -9, '2026-07-28', 44, 7)";
        let one = "INSERT INTO scalar_wave (flag, token, amount, moment, big, day, integer, small) VALUES \
            (true, '10213243-5465-7687-98a9-bacbdcedfe0f', 0.01, '2026-07-29 01:02:03', 0, '2026-07-29', 0, 1)";
        let append_before = engine.open_shard_append_hits();
        #[cfg(feature = "probe-timing")]
        let probe_before = engine.insert_probe_snapshot();
        engine.execute_dml_concurrent(2, many).unwrap();
        engine.execute_dml_concurrent(3, one).unwrap();
        #[cfg(feature = "probe-timing")]
        {
            let probe = engine.insert_probe_snapshot().delta_since(probe_before);
            assert_eq!(probe.direct_fixed_insert_carriers, 2);
            assert_eq!(probe.fixed_insert_typed_commits, 2);
            assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
            assert_eq!(probe.legacy_insert_delta_builds, 0);
            assert_eq!(probe.predicted_row_keys_materialized, 0);
        }
        assert_eq!(engine.open_shard_append_hits(), append_before + 2);
        let result = engine
            .execute_relational_select_text(
                "SELECT small, integer, day, big, moment, amount, token, flag FROM scalar_wave ORDER BY integer",
            )
            .unwrap();
        assert_eq!(result.executed_target, crate::DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        let date = |text: &str| {
            crate::rel_exec_helpers::coerce_insert_value(
                SqlValue::Text(text.to_string()),
                crate::SqlType::Date,
                "day",
            )
            .unwrap()
        };
        let timestamp = |text: &str| {
            crate::rel_exec_helpers::coerce_insert_value(
                SqlValue::Text(text.to_string()),
                crate::SqlType::Timestamp,
                "moment",
            )
            .unwrap()
        };
        let uuid = |text: &str| {
            crate::rel_exec_helpers::coerce_insert_value(
                SqlValue::Text(text.to_string()),
                crate::SqlType::Uuid,
                "token",
            )
            .unwrap()
        };
        assert_eq!(
            result.rows.into_boxed(),
            vec![
                vec![
                    SqlValue::Int2(-7),
                    SqlValue::Int4(-44),
                    date("2026-07-27"),
                    SqlValue::Int8(99),
                    timestamp("2026-07-27 12:34:56"),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(1_234, 2)),
                    uuid("00112233-4455-6677-8899-aabbccddeeff"),
                    SqlValue::Bool(true),
                ],
                vec![
                    SqlValue::Int2(1),
                    SqlValue::Int4(0),
                    date("2026-07-29"),
                    SqlValue::Int8(0),
                    timestamp("2026-07-29 01:02:03"),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(1, 2)),
                    uuid("10213243-5465-7687-98a9-bacbdcedfe0f"),
                    SqlValue::Bool(true),
                ],
                vec![
                    SqlValue::Int2(7),
                    SqlValue::Int4(44),
                    date("2026-07-28"),
                    SqlValue::Int8(-9),
                    timestamp("2026-07-28 00:00:00"),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(-25, 2)),
                    uuid("ffeeddcc-bbaa-9988-7766-554433221100"),
                    SqlValue::Bool(false),
                ],
            ]
        );
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
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
            &command,
            &prepared_catalog,
            prepared_catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("exact fixed INSERT has a direct carrier before catalog drift");
        let request = CanonicalRequest::from_text(&engine, &sql);
        let prepared = OfflockPreparedDml::typed_insert(
            batch,
            &engine,
            &prepared_catalog,
            &request,
            engine.committed_seq(),
        )
        .unwrap();
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
        let result = engine.prepare_typed_insert_pre_wal(
            &mut item,
            &wave_catalog,
            wave_catalog.commit_seq,
            row_id_before,
            false,
        );
        assert!(matches!(result, TypedInsertPreflightResult::FullReprepare));
        assert!(item.offlock_prepared.is_none());
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_check_binding_drift_discards_prepared_plan_before_wal_or_row_id() {
        let engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine
            .execute_text(
                1,
                "CREATE TABLE checked_drift (id int4, \
                 CONSTRAINT checked_drift_positive CHECK (id > 0))",
            )
            .unwrap();
        let sql = "INSERT INTO checked_drift VALUES (1)";
        let command = parse_command(sql).unwrap();
        let prepared_catalog = engine.catalog_snapshot();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
            &command,
            &prepared_catalog,
            prepared_catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("CHECK-only typed INSERT is an off-lock candidate");
        let request = CanonicalRequest::from_text(&engine, sql);
        let prepared = OfflockPreparedDml::typed_insert(
            batch,
            &engine,
            &prepared_catalog,
            &request,
            engine.committed_seq(),
        )
        .expect("device CHECK proof succeeds before queueing");
        let mut item = CommitWaveItem {
            txn_id: 2,
            cmd: command,
            request,
            write_set: crate::write_path::WriteSet {
                tables: std::collections::BTreeSet::from(["checked_drift".to_string()]),
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
        let mut drift = (*prepared_catalog).clone();
        drift
            .relational_catalog
            .get_mut("checked_drift")
            .unwrap()
            .check_constraints[0]
            .value = SqlValue::Int4(2);
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let result = engine.prepare_typed_insert_pre_wal(
            &mut item,
            &drift,
            drift.commit_seq,
            row_id_before,
            false,
        );
        assert!(matches!(result, TypedInsertPreflightResult::FullReprepare));
        assert!(item.offlock_prepared.is_none());
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    }

    #[test]
    fn metadata_only_domain_bindings_revalidate_name_oid_base_type_and_column_before_wal() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE DOMAIN account_code AS int4")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE TABLE domain_accounts (id account_code, balance int4)",
            )
            .unwrap();
        let sql = "INSERT INTO domain_accounts VALUES (7, 70)";
        let build_item = || {
            let command = parse_command(sql).unwrap();
            let prepared_catalog = engine.catalog_snapshot();
            let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
                &command,
                &prepared_catalog,
                prepared_catalog.commit_seq,
                None,
            )
            .unwrap()
            .expect("metadata-only domain columns are direct candidates");
            let request = CanonicalRequest::from_text(&engine, sql);
            let prepared = OfflockPreparedDml::typed_insert(
                batch,
                &engine,
                &prepared_catalog,
                &request,
                engine.committed_seq(),
            )
            .unwrap();
            CommitWaveItem {
                txn_id: 3,
                cmd: command,
                request,
                write_set: crate::write_path::WriteSet {
                    tables: std::collections::BTreeSet::from(["domain_accounts".to_string()]),
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
            }
        };

        for sabotage in ["name", "oid", "base", "column"] {
            let mut item = build_item();
            let mut wave_catalog = (*engine.catalog_snapshot()).clone();
            match sabotage {
                "name" => {
                    wave_catalog.relational_domains.remove("account_code");
                }
                "oid" => {
                    wave_catalog
                        .relational_domains
                        .get_mut("account_code")
                        .unwrap()
                        .oid += 1;
                }
                "base" => {
                    wave_catalog
                        .relational_domains
                        .get_mut("account_code")
                        .unwrap()
                        .base_type = crate::SqlType::Int8;
                }
                "column" => {
                    wave_catalog
                        .relational_catalog
                        .get_mut("domain_accounts")
                        .unwrap()
                        .columns[0]
                        .domain = None;
                }
                _ => unreachable!(),
            }
            let wal_before = engine.durable_wal_records().len();
            let row_id_before = engine.read_state.mvcc.current_row_id();
            let result = engine.prepare_typed_insert_pre_wal(
                &mut item,
                &wave_catalog,
                wave_catalog.commit_seq,
                row_id_before,
                false,
            );
            assert!(
                matches!(result, TypedInsertPreflightResult::FullReprepare),
                "{sabotage}"
            );
            assert!(item.offlock_prepared.is_none(), "{sabotage}");
            assert_eq!(engine.durable_wal_records().len(), wal_before, "{sabotage}");
            assert_eq!(
                engine.read_state.mvcc.current_row_id(),
                row_id_before,
                "{sabotage}"
            );
        }
    }

    #[cfg(feature = "probe-timing")]
    #[test]
    fn defaulted_table_with_all_cells_supplied_uses_direct_plan_without_legacy_state() {
        let engine = Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(1, "CREATE TABLE notes (id int4 DEFAULT 1, body text)")
            .unwrap();
        let before = engine.insert_probe_snapshot();
        engine
            .execute_dml_concurrent(2, "INSERT INTO notes VALUES (7, 'legacy')")
            .unwrap();
        let probe = engine.insert_probe_snapshot().delta_since(before);
        assert_eq!(probe.direct_fixed_insert_carriers, 1);
        assert_eq!(probe.fixed_insert_typed_commits, 1);
        assert_eq!(probe.legacy_insert_delta_builds, 0);
        assert_eq!(probe.predicted_row_keys_materialized, 0);
        assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
    }

    #[cfg(feature = "probe-timing")]
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn production_wave_routes_reordered_nullable_text_without_legacy_state() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(8);
        engine
            .execute_text(1, "CREATE TABLE notes (id int4, body text)")
            .unwrap();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let wal_before = engine.durable_wal_records().len();
        let commits_before = engine.device_authoritative_commits();
        let probe_before = engine.insert_probe_snapshot();
        let sql = "INSERT INTO notes (body, id) VALUES ('', 7), (NULL, 8), ('naïve', 9)";
        engine.execute_dml_concurrent(2, sql).unwrap();
        let probe = engine.insert_probe_snapshot().delta_since(probe_before);
        assert_eq!(probe.direct_fixed_insert_carriers, 1);
        assert_eq!(probe.fixed_insert_typed_commits, 1);
        assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
        assert_eq!(probe.legacy_insert_delta_builds, 0);
        assert_eq!(probe.predicted_row_keys_materialized, 0);
        assert_eq!(probe.rollover_count, 1);
        assert_eq!(probe.rollover_capacity_rows_total, 3);
        assert_eq!(probe.rollover_capacity_rows_max, 3);
        assert_eq!(probe.persistent_allocation_count, 3);
        assert_eq!(probe.budget_scan_entries, 2);
        assert_eq!(probe.sidecar_fill_bytes, 24);
        assert_eq!(probe.live_h2d_bytes, 118);
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before + 3);
        assert_eq!(engine.device_authoritative_commits(), commits_before + 1);
        assert!(engine.table_device_authoritative("notes"));
        let shards = engine
            .read_state
            .residency
            .shards
            .load()
            .get("notes")
            .cloned()
            .expect("typed dense plan publishes exactly one successor shard");
        assert_eq!(shards.len(), 2);
        let dense = &shards[1];
        assert_eq!(dense.row_count, 3);
        assert_eq!(dense.resident_device_text_columns.len(), 1);
        assert_eq!(dense.resident_device_null_columns.len(), 1);
        let ids = dense.row_id_region.as_ref().unwrap();
        assert_eq!(read_device_u64(ids, 0), row_id_before);
        assert_eq!(read_device_u64(ids, 2), row_id_before + 2);
        let rows = engine
            .execute_relational_select_text("SELECT id, body FROM notes ORDER BY id")
            .unwrap();
        assert_eq!(rows.executed_target, crate::DeviceTarget::Gpu(0));
        assert_eq!(rows.fallback_reason, None);
        assert_eq!(
            rows.rows,
            vec![
                vec![SqlValue::Int4(7), SqlValue::Text(String::new())],
                vec![SqlValue::Int4(8), SqlValue::Null],
                vec![SqlValue::Int4(9), SqlValue::Text("naïve".to_string())],
            ],
        );
        let nulls = engine
            .execute_relational_select_text("SELECT id FROM notes WHERE body IS NULL")
            .unwrap();
        assert_eq!(nulls.executed_target, crate::DeviceTarget::Gpu(0));
        assert_eq!(nulls.fallback_reason, None);
        assert_eq!(nulls.rows, vec![vec![SqlValue::Int4(8)]]);
    }

    #[cfg(feature = "probe-timing")]
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn production_wave_routes_literal_absent_defaults_and_metadata_only_domain_without_legacy_state(
    ) {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(8);
        engine
            .execute_text(1, "CREATE DOMAIN wave_code AS int4")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE TABLE default_wave (id wave_code, body text DEFAULT '', enabled bool DEFAULT true, optional int4)",
            )
            .unwrap();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let wal_before = engine.durable_wal_records().len();
        let probe_before = engine.insert_probe_snapshot();
        engine
            .execute_dml_concurrent(
                3,
                "INSERT INTO default_wave (enabled, id, body) VALUES \
                 (DEFAULT, 7, DEFAULT), (DEFAULT, 8, NULL), (false, 9, 'naïve')",
            )
            .unwrap();
        let probe = engine.insert_probe_snapshot().delta_since(probe_before);
        assert_eq!(probe.direct_fixed_insert_carriers, 1);
        assert_eq!(probe.fixed_insert_typed_commits, 1);
        assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
        assert_eq!(probe.legacy_insert_delta_builds, 0);
        assert_eq!(probe.predicted_row_keys_materialized, 0);
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before + 3);
        let result = engine
            .execute_relational_select_text(
                "SELECT id, body, enabled, optional FROM default_wave ORDER BY id",
            )
            .unwrap();
        assert_eq!(result.executed_target, crate::DeviceTarget::Gpu(0));
        assert_eq!(result.fallback_reason, None);
        assert_eq!(
            result.rows,
            vec![
                vec![
                    SqlValue::Int4(7),
                    SqlValue::Text(String::new()),
                    SqlValue::Bool(true),
                    SqlValue::Null,
                ],
                vec![
                    SqlValue::Int4(8),
                    SqlValue::Null,
                    SqlValue::Bool(true),
                    SqlValue::Null,
                ],
                vec![
                    SqlValue::Int4(9),
                    SqlValue::Text("naïve".to_string()),
                    SqlValue::Bool(false),
                    SqlValue::Null,
                ],
            ]
        );
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
        assert!(wave.contains("prepare_typed_insert_pre_wal"));
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
        TypedInsertApply::fail_next_post_wal_apply(&engine);

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
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
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
        let prepared = OfflockPreparedDml::typed_insert(
            batch,
            &engine,
            &catalog,
            &request,
            engine.committed_seq(),
        )
        .unwrap();
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

        let result = engine.prepare_typed_insert_pre_wal(
            &mut item,
            &catalog,
            catalog.commit_seq,
            u64::MAX - 1,
            true,
        );
        assert!(matches!(
            result,
            TypedInsertPreflightResult::PreWalFailure(crate::ExecuteError::Engine(_))
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
