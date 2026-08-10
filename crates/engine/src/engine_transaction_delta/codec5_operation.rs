//! Ephemeral codec-5 operation preparation for the generic transaction finalizer.
//!
//! This leaf has no proposal, WAL, status, device-apply, or publication capability. It closes
//! the one-record aggregate from already-validated generic transaction facts, then returns the
//! move-only ingredients to the sole generic lifecycle owner.

use super::*;

pub(super) struct SelectedGenericCodec5 {
    staged: Vec<Arc<StagedTypedInsert>>,
    /// Transaction-local rows are keyed by immutable catalog identity, never by a duplicated
    /// relation-name allocation. The staged source still owns names for diagnostics/S3, while
    /// the terminal needs only the exact stable row lookup for rewrite disposition.
    final_rows: BTreeMap<(u64, u64), SelectedCodec5FinalRow>,
    final_table_resets: BTreeMap<String, Arc<StagedTableReset>>,
    write_set: WriteSet,
    has_rewritten_rows: bool,
    has_operation_composition: bool,
    terminal_sequence_restarts: Vec<crate::typed_insert_aggregate::LiveTypedInsertSequenceRestart>,
}

struct SelectedCodec5FinalRow {
    statement_ordinal: u32,
    source_operation_ordinal: u32,
    source_row_ordinal: u32,
    final_writer_statement_ordinal: u32,
    final_writer_statement_digest: gpu_db_wal::CanonicalDigest,
    survives: bool,
    /// Only mixed UPDATE/DELETE chains, sequence-final-value binding, and the admission-only
    /// geometry reader consume row values. A plain typed INSERT already owns exact columnar S2
    /// and final-image sources, so retaining another `Vec<SqlValue>` per row would recreate a
    /// displaced semantic carrier at the terminal.
    values: Option<Vec<SqlValue>>,
}

impl std::ops::Deref for SelectedGenericCodec5 {
    type Target = [Arc<StagedTypedInsert>];

    fn deref(&self) -> &Self::Target {
        &self.staged
    }
}

impl SelectedGenericCodec5 {
    pub(super) fn has_operation_composition(&self) -> bool {
        self.has_operation_composition
    }

    fn rows_for_table(&self, table: &str) -> Vec<(&u64, &SelectedCodec5FinalRow)> {
        let Some(stable_table_id) = self
            .staged
            .iter()
            .find(|staged| staged.table == table)
            .map(|staged| staged.stable_table_id)
        else {
            return Vec::new();
        };
        self.final_rows
            .range((stable_table_id, u64::MIN)..=(stable_table_id, u64::MAX))
            .map(|((_, row_id), row)| (row_id, row))
            .collect()
    }

    fn table_name_for_stable(&self, stable_table_id: u64) -> Option<&str> {
        self.staged
            .iter()
            .find(|staged| staged.stable_table_id == stable_table_id)
            .map(|staged| staged.table.as_str())
    }

    fn resets_table(&self, table: &str) -> bool {
        self.final_table_resets.contains_key(table)
    }

    pub(super) fn final_writer_statement_digests(&self) -> Vec<(u32, gpu_db_wal::CanonicalDigest)> {
        self.staged
            .iter()
            .flat_map(|staged| {
                staged.provisional_row_ids.iter().filter_map(|row_id| {
                    let row = self.final_rows.get(&(staged.stable_table_id, *row_id))?;
                    (row.final_writer_statement_digest != [0; 32]).then_some((
                        row.final_writer_statement_ordinal,
                        row.final_writer_statement_digest,
                    ))
                })
            })
            .collect()
    }

    pub(super) fn has_surviving_indexed_table(&self, catalog: &CatalogSnapshot) -> bool {
        self.staged.iter().any(|staged| {
            catalog
                .relational_catalog
                .get(&staged.table)
                .is_some_and(|table| {
                    !table.indexes.is_empty()
                        && self
                            .final_rows
                            .range(
                                (staged.stable_table_id, u64::MIN)
                                    ..=(staged.stable_table_id, u64::MAX),
                            )
                            .any(|(_, row)| row.survives)
                })
        })
    }

    pub(super) fn bind_published_sequence_final_values(
        &self,
        references: &mut [BinarySequenceValueReference],
        catalog: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        for reference in references
            .iter_mut()
            .filter(|reference| reference.default_expression)
        {
            let table = catalog
                .relational_catalog
                .values()
                .find(|table| table.oid == reference.table_oid)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "codec-5 sequence default {} lost its table identity",
                        reference.transition_txn_id
                    )))
                })?;
            let column = table
                .columns
                .iter()
                .position(|column| column.id == reference.column_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "codec-5 sequence default {} lost its column identity",
                        reference.transition_txn_id
                    )))
                })?;
            let expected = i32::try_from(reference.returned_value).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "codec-5 sequence default {} is outside its int4 domain",
                    reference.transition_txn_id
                )))
            })?;
            let value_is_retained = self
                .final_rows
                .get(&(table.stable_table_id, reference.row_id))
                .filter(|row| row.survives)
                .and_then(|row| row.values.as_ref().and_then(|values| values.get(column)))
                == Some(&SqlValue::Int4(expected));
            reference.final_value_overwritten = !value_is_retained;
        }
        Ok(())
    }
}

pub(super) struct Codec5CatalogComposition {
    pub(super) record: BinaryTransactionRecord,
    pub(super) operation_body: Arc<[u8]>,
    pub(super) changes_catalog: bool,
}

/// Compose the existing ordered catalog and pre-existing-row UPDATE/DELETE facts into codec-5
/// without creating a row, allocator, terminal, or recovery authority. The encoded GPUDBOP1 body
/// is carried verbatim by S3; typed INSERT images remain exclusively in S2/S4/S7.
#[allow(clippy::too_many_arguments)] // one codec-5 composition boundary with independently sealed inputs
pub(super) fn prepare_catalog_composition(
    engine: &Engine,
    txn_id: TxnId,
    selected: &SelectedGenericCodec5,
    operations: &[TransactionOperation],
    final_staged_rows: &[FinalTransactionRowOperation],
    catalog_commands: &[StagedCatalogCommand],
    table_resets: &[StagedTableReset],
    transaction_catalog: &CatalogSnapshot,
    provisional_inserts: &BTreeSet<(String, u64)>,
    final_base: u64,
    sequence_value_references: &[BinarySequenceValueReference],
) -> Result<Option<Codec5CatalogComposition>, ExecuteError> {
    if !selected.has_operation_composition() {
        return Ok(None);
    }
    if !table_resets.is_empty()
        || catalog_commands.iter().any(|staged| {
            !crate::wal_binary::command_is_codec5_catalog_composition(&staged.command)
        })
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 ordered composition lost its admitted catalog or reset shape".to_string(),
        )));
    }
    // Typed INSERT images are deliberately skipped by the resolver. A private INSERT
    // UPDATE/DELETE is already folded into its S2 final image; only mutations of a row that
    // predates this transaction remain in the S3 operation body.
    let mut record = Engine::resolved_transaction_record(
        final_staged_rows,
        provisional_inserts,
        final_base,
        0,
        &[],
    )?;
    // S5 owns typed-row sequence final values. An S3 catalog operation needs only the existing
    // ordered reference closure, rebound from dense typed statement ordinals to transaction
    // operation ordinals; a row-only S3 composition must not duplicate those typed references.
    if !catalog_commands.is_empty() {
        record.sequence_value_references =
            catalog_composition_sequence_references(sequence_value_references, operations)?;
    }
    Engine::bind_transaction_record_catalog_envelope(
        &mut record,
        operations,
        catalog_commands,
        table_resets,
        transaction_catalog,
    )?;
    if record.allocator_high_water != 0 || !record.table_resets.is_empty() {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 operation composition acquired reset or allocator authority".to_string(),
        )));
    }
    let changes_catalog = !record.catalog_commands.is_empty();
    if changes_catalog {
        engine.validate_transaction_catalog_before_wal(
            record.catalog_epoch,
            &record.catalog_commands,
            TransactionCatalogEnvelopeSlices {
                view_operations: &record.view_operations,
                view_lifecycle_operations: &record.view_lifecycle_operations,
                index_lifecycle_operations: &record.index_lifecycle_operations,
                sequence_lifecycle_operations: &record.sequence_lifecycle_operations,
                sequence_reset_operations: &record.sequence_reset_operations,
                operation_order: &record.operation_order,
                catalog_output: record.catalog_output.as_ref(),
                sequence_input_oids: &record.sequence_input_oids,
                sequence_value_references: &record.sequence_value_references,
            },
            transaction_catalog,
            true,
        )?;
    }
    if !changes_catalog && record.mutations.is_empty() {
        return Ok(None);
    }
    let payload = try_encode_binary_transaction(&record).ok_or_else(|| {
        ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 ordered catalog envelope exceeds or violates binary framing".to_string(),
        ))
    })?;
    let sealed =
        crate::engine_canonical_operation::SealedCanonicalOperation::from_live_binary_transaction(
            &payload, &record, txn_id,
        )?;
    if sealed.kind()
        != if changes_catalog {
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation
        } else {
            gpu_db_wal::CanonicalFragmentKind::RowMutation
        }
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 did not seal as the selected ordered operation".to_string(),
        )));
    }
    Ok(Some(Codec5CatalogComposition {
        record,
        operation_body: sealed.into_fragment().body.into(),
        changes_catalog,
    }))
}

fn catalog_composition_sequence_references(
    references: &[BinarySequenceValueReference],
    operations: &[TransactionOperation],
) -> Result<Vec<BinarySequenceValueReference>, ExecuteError> {
    let typed_operation_ordinals = operations
        .iter()
        .filter_map(|operation| match operation {
            TransactionOperation::TypedInsert(staged) => Some(staged.operation_ordinal),
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TableReset(_) => None,
        })
        .collect::<Vec<_>>();
    let mut rebound = references.to_vec();
    for reference in rebound
        .iter_mut()
        .filter(|reference| reference.default_expression)
    {
        reference.statement_ordinal = typed_operation_ordinals
            .get(reference.statement_ordinal as usize)
            .copied()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "codec-5 S3 sequence default reference {} lost its typed INSERT operation owner",
                    reference.transition_txn_id
                )))
            })?;
    }
    if rebound
        .windows(2)
        .any(|pair| pair[0].statement_ordinal > pair[1].statement_ordinal)
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 sequence references do not follow transaction operation order".to_string(),
        )));
    }
    Ok(rebound)
}

impl Engine {
    /// Measure a predeclared typed INSERT from the already-selected codec-5 overlay. This keeps
    /// the established logical resource-contract scale while avoiding construction of the
    /// displaced resolved INSERT record. Exact codec-5 framing and physical WAL capacity remain
    /// owned by the terminal's pre-WAL reservation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predeclared_codec5_insert_resource_geometry(
        &self,
        operations: &[TransactionOperation],
        row_delta_count: usize,
        table_reset_count: usize,
        final_rows: &[FinalTransactionRowOperation],
        catalog_commands: &[StagedCatalogCommand],
        transaction_catalog: &CatalogSnapshot,
        sequence_value_references: &[BinarySequenceValueReference],
    ) -> Result<Option<(u64, u64, u32)>, ExecuteError> {
        #[cfg(feature = "probe-timing")]
        let probe_select_started = std::time::Instant::now();
        if !operations
            .iter()
            .any(|operation| matches!(operation, TransactionOperation::TypedInsert(_)))
        {
            return Ok(None);
        }
        let selected = compile_insert_bearing_codec5_staged(
            operations,
            row_delta_count,
            table_reset_count,
            final_rows,
            catalog_commands,
            transaction_catalog,
            sequence_value_references,
            &self.read_state.typed_generation_roots.load_full(),
            true,
        )?;
        #[cfg(feature = "probe-timing")]
        let probe_select_nanos = probe_select_started.elapsed().as_nanos() as u64;
        #[cfg(feature = "probe-timing")]
        let probe_geometry_started = std::time::Instant::now();

        const LOGICAL_TRANSACTION_HEADER_BYTES: u64 = 3 + 8 + 4 + 4;
        const CANONICAL_LOGICAL_OVERHEAD_BYTES: u64 = 216;
        let mut post_image_bytes = 0_u64;
        let mut logical_mutation_bytes = LOGICAL_TRANSACTION_HEADER_BYTES;
        let mut maintained_index_fanout = 0_u32;
        let mut surviving_rows = 0_u64;
        for ((stable_table_id, _), row) in &selected.final_rows {
            if !row.survives {
                continue;
            }
            let values = row.values.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 admission geometry lost its explicitly materialized row image"
                        .to_string(),
                ))
            })?;
            let row_encoded = encode_relational_row(values);
            let row_bytes = u64::try_from(row_encoded.len()).map_err(|_| {
                ExecuteError::Unsupported(
                    "predeclared INSERT row image length exceeds u64 framing".to_string(),
                )
            })?;
            let table_name = selected
                .table_name_for_stable(*stable_table_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 admission geometry lost a typed table identity".to_string(),
                    ))
                })?;
            let table_bytes = u64::try_from(table_name.len()).map_err(|_| {
                ExecuteError::Unsupported(
                    "predeclared INSERT table name length exceeds u64 framing".to_string(),
                )
            })?;
            post_image_bytes = post_image_bytes.checked_add(row_bytes).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction post-image byte count overflow".to_string(),
                )
            })?;
            logical_mutation_bytes = logical_mutation_bytes
                .checked_add(1 + 2)
                .and_then(|bytes| bytes.checked_add(table_bytes))
                .and_then(|bytes| bytes.checked_add(8 + 4))
                .and_then(|bytes| bytes.checked_add(row_bytes))
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction logical-WAL byte count overflow".to_string(),
                    )
                })?;
            let indexes = transaction_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| ExecuteError::UndefinedRelation(table_name.to_string()))?
                .indexes
                .len();
            maintained_index_fanout = maintained_index_fanout
                .checked_add(u32::try_from(indexes).map_err(|_| {
                    ExecuteError::Unsupported(
                        "predeclared transaction index count exceeds u32 framing".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction index-fanout count overflow".to_string(),
                    )
                })?;
            surviving_rows = surviving_rows.checked_add(1).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction surviving INSERT count overflow".to_string(),
                )
            })?;
        }
        let logical_wal_bytes = if surviving_rows == 0 {
            0
        } else {
            logical_mutation_bytes
                .checked_add(CANONICAL_LOGICAL_OVERHEAD_BYTES)
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction logical-WAL byte count overflow".to_string(),
                    )
                })?
        };
        let geometry = Some((post_image_bytes, logical_wal_bytes, maintained_index_fanout));
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_codec5_materialization_nanos([
            probe_select_nanos,
            probe_geometry_started.elapsed().as_nanos() as u64,
            0,
            0,
        ]);
        Ok(geometry)
    }
}

/// Compile the already-staged typed transaction for the single codec-5 terminal. This is only a
/// shape proof over existing transaction artifacts: it creates no payload, record, allocator, or
/// publication authority. Every caller has already established that the transaction bears a
/// typed INSERT, so an unsupported shape rejects before a legacy resolved-record state exists.
#[allow(clippy::too_many_arguments)] // selection verifies distinct staged transaction authorities
pub(super) fn compile_insert_bearing_codec5_staged(
    operations: &[TransactionOperation],
    row_delta_count: usize,
    table_reset_count: usize,
    _final_staged_rows: &[FinalTransactionRowOperation],
    catalog_commands: &[StagedCatalogCommand],
    transaction_catalog: &CatalogSnapshot,
    sequence_value_references: &[BinarySequenceValueReference],
    predecessor_roots: &crate::engine_state::TypedGenerationRootSnapshot,
    materialize_all_row_values: bool,
) -> Result<SelectedGenericCodec5, ExecuteError> {
    let decline = |reason: &str| {
        Err(ExecuteError::Unsupported(format!(
            "typed INSERT transaction cannot compile codec-5 ({reason}); resolved INSERT WAL fallback is removed"
        )))
    };
    // A row operation may be entirely private to a typed INSERT overlay, in which case S3 will
    // prove empty after final-image folding. It may also touch a pre-existing row, in which case
    // the same S3 slot carries its resolved UPDATE/DELETE. The selector deliberately admits both
    // before any legacy record is constructed.
    let has_operation_composition = !catalog_commands.is_empty()
        || operations
            .iter()
            .any(|operation| matches!(operation, TransactionOperation::Row(_)));
    if operations.is_empty()
        || !operations.iter().all(|operation| match operation {
            TransactionOperation::TypedInsert(_) => true,
            TransactionOperation::Catalog(staged) => {
                matches!(
                    staged.command,
                    Command::SequenceRestart(_) | Command::RenameSequence(_)
                ) || crate::wal_binary::command_is_codec5_catalog_composition(&staged.command)
            }
            TransactionOperation::Row(staged) => {
                matches!(
                    staged.mutation,
                    PreparedMutation::Update {
                        class_epoch: None,
                        ..
                    } | PreparedMutation::Delete {
                        class_epoch: None,
                        ..
                    }
                )
            }
            TransactionOperation::TableReset(_) => true,
        })
    {
        return decline("operation, catalog, or table-reset composition is not encoded");
    }
    let typed = operations
        .iter()
        .filter_map(|operation| match operation {
            TransactionOperation::TypedInsert(staged) => Some(Arc::clone(staged)),
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TableReset(_) => None,
        })
        .collect::<Vec<_>>();
    let first = typed.first().ok_or_else(|| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "mandatory codec-5 compiler received no typed INSERT contribution".to_string(),
        ))
    })?;
    if row_delta_count
        != operations
            .iter()
            .filter(|operation| matches!(operation, TransactionOperation::Row(_)))
            .count()
    {
        return decline("row-delta cardinality differs from the ordered operation stream");
    }
    // The existing final image is already the owned value source for a typed-only INSERT. Keep
    // transient row values only for consumers that must compare/rewrite them. The admission
    // geometry reader explicitly asks for values because it still encodes legacy-equivalent
    // bounds; the terminal does not when its transaction has neither row mutations nor defaults.
    let materialize_row_values = materialize_all_row_values
        || !sequence_value_references.is_empty()
        || operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::Row(_) | TransactionOperation::TableReset(_)
            )
        });
    let mut final_rows = BTreeMap::new();
    let mut write_set = WriteSet::default();
    for staged in &typed {
        let table = transaction_catalog
            .relational_catalog
            .get(&staged.table)
            .ok_or_else(|| ExecuteError::UndefinedRelation(staged.table.clone()))?;
        for (source_row_ordinal, row_id) in staged.provisional_row_ids.iter().copied().enumerate() {
            let row = SelectedCodec5FinalRow {
                statement_ordinal: staged.statement_ordinal,
                source_operation_ordinal: staged.operation_ordinal,
                source_row_ordinal: u32::try_from(source_row_ordinal).map_err(|_| {
                    ExecuteError::Unsupported("codec-5 source row ordinal exceeds u32".to_string())
                })?,
                final_writer_statement_ordinal: staged.statement_ordinal,
                final_writer_statement_digest: [0; 32],
                survives: true,
                values: materialize_row_values
                    .then(|| staged.private_row_values(table, source_row_ordinal))
                    .transpose()?,
            };
            if final_rows
                .insert((staged.stable_table_id, row_id), row)
                .is_some()
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "codec-5 typed overlay reused a provisional row identity".to_string(),
                )));
            }
        }
        write_set.extend_deduplicated(&staged.write_set);
    }
    let mut has_rewritten_rows = false;
    let mut final_table_resets = BTreeMap::<String, Arc<StagedTableReset>>::new();
    for (operation_ordinal, operation) in operations.iter().enumerate() {
        let final_writer_statement_ordinal = u32::try_from(operation_ordinal).map_err(|_| {
            ExecuteError::Unsupported("codec-5 writer operation ordinal exceeds u32".to_string())
        })?;
        if let TransactionOperation::TableReset(reset) = operation {
            let table = transaction_catalog
                .relational_catalog
                .get(&reset.table)
                .ok_or_else(|| ExecuteError::UndefinedRelation(reset.table.clone()))?;
            if reset.ordinal != final_writer_statement_ordinal
                || reset.table_oid != table.oid
                || reset.schema_digest != table_schema_digest(table).unwrap_or([0; 32])
                || reset.dependency_identities.get(&reset.table) != Some(&table.oid)
            {
                return decline("a table reset drifted from its ordered catalog identity");
            }
            let mut shadowed = 0_usize;
            for ((row_stable_table_id, _), row) in &mut final_rows {
                if *row_stable_table_id == table.stable_table_id
                    && row.source_operation_ordinal < final_writer_statement_ordinal
                    && row.survives
                {
                    row.survives = false;
                    row.final_writer_statement_ordinal = final_writer_statement_ordinal;
                    row.final_writer_statement_digest = reset.statement_digest;
                    shadowed += 1;
                }
            }
            if shadowed != 0 {
                has_rewritten_rows = true;
            }
            write_set.tables.insert(reset.table.clone());
            write_set
                .tables
                .extend(reset.foreign_key_dependencies.iter().cloned());
            write_set.table_oids.push(reset.table_oid);
            write_set
                .table_oids
                .extend(reset.dependency_identities.values().copied());
            final_table_resets.insert(reset.table.clone(), Arc::clone(reset));
            continue;
        }
        let TransactionOperation::Row(staged) = operation else {
            continue;
        };
        match &staged.mutation {
            PreparedMutation::Update {
                table,
                installs,
                updated_old_rows,
                class_epoch: None,
            } => {
                if installs.len() != updated_old_rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "codec-5 private UPDATE lost its old/new row bijection".to_string(),
                    )));
                }
                if installs.is_empty() {
                    return decline("an UPDATE contribution has no selected provisional row");
                }
                let stable_table_id = transaction_catalog
                    .relational_catalog
                    .get(table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(table.clone()))?
                    .stable_table_id;
                for ((row_id, _, new_values), old_values) in installs.iter().zip(updated_old_rows) {
                    let Some(row) = final_rows.get_mut(&(stable_table_id, *row_id)) else {
                        // This is a pre-existing row. Its final typed INSERT image is absent by
                        // construction, so the shared S3 operation composition retains the
                        // already-resolved UPDATE for the common device maintainer.
                        continue;
                    };
                    if !row.survives || row.final_writer_statement_digest != [0; 32] {
                        return decline("an UPDATE targets an already rewritten provisional row");
                    }
                    if row.values.as_ref() != Some(old_values) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "codec-5 private UPDATE chain for entity {row_id} in relation \"{table}\" is discontinuous"
                        ))));
                    }
                    row.values = Some(new_values.clone());
                    row.final_writer_statement_ordinal = final_writer_statement_ordinal;
                    row.final_writer_statement_digest = staged.statement_digest;
                    has_rewritten_rows = true;
                }
            }
            PreparedMutation::Delete {
                table,
                tuple_ids,
                deleted_rows,
                class_epoch: None,
            } => {
                if tuple_ids.is_empty() || tuple_ids.len() != deleted_rows.len() {
                    return decline("a DELETE contribution lacks an exact selected row image");
                }
                let stable_table_id = transaction_catalog
                    .relational_catalog
                    .get(table)
                    .ok_or_else(|| ExecuteError::UndefinedRelation(table.clone()))?
                    .stable_table_id;
                for (row_id, deleted) in tuple_ids.iter().zip(deleted_rows) {
                    let Some(row) = final_rows.get_mut(&(stable_table_id, *row_id)) else {
                        // Like UPDATE above, a pre-existing delete belongs in S3 and reaches the
                        // same atomic transaction device maintainer after durable codec-5 apply.
                        continue;
                    };
                    if !row.survives
                        || row.final_writer_statement_digest != [0; 32]
                        || row.values.as_ref() != Some(deleted)
                    {
                        return decline(
                            "a DELETE targets an already rewritten or drifted provisional row",
                        );
                    }
                    row.survives = false;
                    row.final_writer_statement_ordinal = final_writer_statement_ordinal;
                    row.final_writer_statement_digest = staged.statement_digest;
                    has_rewritten_rows = true;
                }
            }
            PreparedMutation::Insert { .. }
            | PreparedMutation::Update {
                class_epoch: Some(_),
                ..
            }
            | PreparedMutation::Delete {
                class_epoch: Some(_),
                ..
            } => return decline("a row contribution uses a non-overlay mutation shape"),
        }
        write_set.extend_deduplicated(&staged.write_set);
    }
    let target_names = typed
        .iter()
        .map(|staged| staged.table.as_str())
        .collect::<BTreeSet<_>>();
    if final_table_resets.len() != table_reset_count
        || final_table_resets
            .keys()
            .any(|table| !target_names.contains(table.as_str()))
    {
        return decline("the final table-reset set is not owned by typed INSERT targets");
    }
    let index_shape = target_names.iter().all(|name| {
        transaction_catalog
            .relational_catalog
            .get(*name)
            .is_some_and(|table| {
                table.indexes.is_empty()
                    || table.indexes.iter().all(|index| {
                        (!index.primary_key || index.unique)
                            && (!index.unique_constraint || index.unique)
                            && !index.key_columns.is_empty()
                            && crate::engine_residency::index_all_key_columns_foldable(table, index)
                    })
            })
    });
    let contributions_are_exact = typed.iter().enumerate().all(|(ordinal, staged)| {
        let Some(table) = transaction_catalog.relational_catalog.get(&staged.table) else {
            return false;
        };
        let Some(statement_table) = staged.catalog_dependencies.get(&staged.table) else {
            return false;
        };
        let same_table_first = typed
            .iter()
            .find(|candidate| candidate.table == staged.table)
            .expect("current contribution is a same-table member");
        staged.statement_ordinal as usize == ordinal
            && operations
                .get(staged.operation_ordinal as usize)
                .is_some_and(|operation| {
                    matches!(operation, TransactionOperation::TypedInsert(owner) if Arc::ptr_eq(owner, staged))
                })
            && staged.table_oid == table.oid
            && staged.stable_table_id == table.stable_table_id
            && staged.table_schema_digest
                == table_schema_digest(statement_table).unwrap_or([0; 32])
            && staged.read_snapshot == first.read_snapshot
            && staged.foreign_key_dependencies == same_table_first.foreign_key_dependencies
            // DML may legitimately straddle an ordered CREATE/DROP/RENAME INDEX operation.
            // The transaction-owned index lifecycle has its own exact before/after identity, so
            // require the same dependency *names* and validate every statement-time dependency
            // against the final overlay modulo that index-list transition.  Requiring bytewise
            // equality here would reject the first typed contribution merely because a later
            // contribution correctly sees the transaction-private index.
            && staged.catalog_dependencies.len() == same_table_first.catalog_dependencies.len()
            && staged
                .catalog_dependencies
                .keys()
                .eq(same_table_first.catalog_dependencies.keys())
            && staged.catalog_dependencies.iter().all(|(name, statement_dependency)| {
                transaction_catalog
                    .relational_catalog
                    .get(name)
                    .is_some_and(|final_dependency| {
                        same_transaction_row_catalog_dependency(
                            statement_dependency,
                            final_dependency,
                            &BTreeMap::new(),
                            transaction_catalog,
                            catalog_commands,
                        )
                    })
            })
            && same_transaction_row_catalog_dependency(
                statement_table,
                table,
                &BTreeMap::new(),
                transaction_catalog,
                catalog_commands,
            )
                || catalog_commands.iter().any(|catalog| {
                    matches!(&catalog.command, Command::CreateTable(create) if create.table == staged.table)
                        && statement_table.oid == table.oid
                        && statement_table.stable_table_id == table.stable_table_id
                        && table_schema_digest(statement_table).ok()
                            == table_schema_digest(table).ok()
                })
    });
    let sequences_are_exact = sequence_value_references.iter().all(|reference| {
        reference.default_expression
            && (reference.statement_ordinal as usize) < typed.len()
            && reference.table_oid == typed[reference.statement_ordinal as usize].table_oid
    });
    if !index_shape {
        return decline("a target index is outside the admitted device index shape");
    }
    if !contributions_are_exact {
        return decline("a typed contribution drifted from its ordered catalog overlay");
    }
    let terminal_sequence_restarts = select_terminal_sequence_restarts(
        catalog_commands,
        operations,
        &typed,
        sequence_value_references,
        transaction_catalog,
    )?
    .ok_or_else(|| {
        ExecuteError::Unsupported(
            "typed INSERT transaction cannot compile codec-5 (private sequence catalog operations are not exact); resolved INSERT WAL fallback is removed"
                .to_string(),
        )
    })?;
    if !sequences_are_exact {
        return decline("a published sequence reference is not bound to its typed statement");
    }
    let created_targets = catalog_commands
        .iter()
        .filter_map(|staged| match &staged.command {
            Command::CreateTable(create) => Some(create.table.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if predecessor_roots.database_root.is_none()
        && !target_names
            .iter()
            .all(|table| created_targets.contains(table))
    {
        return decline("the typed database predecessor root is absent");
    }
    if !typed.iter().all(|staged| {
        predecessor_roots.table(staged.stable_table_id).is_some()
            || created_targets.contains(staged.table.as_str())
    }) {
        return decline("a typed target predecessor root is absent");
    }
    Ok(SelectedGenericCodec5 {
        staged: typed,
        final_rows,
        final_table_resets,
        write_set,
        has_rewritten_rows,
        has_operation_composition,
        terminal_sequence_restarts,
    })
}

/// Bind every admitted RESTART either to the private S2 chain it created or, when it follows the
/// final default allocation, to the existing last S5 effect as one terminal-state suffix. A
/// sequence rename adds no scalar terminal: its stable-OID transition must be witnessed by a
/// prior private creation or typed effect, then by a final private S2 effect carrying the renamed
/// catalog binding.
fn select_terminal_sequence_restarts(
    catalog_commands: &[StagedCatalogCommand],
    operations: &[TransactionOperation],
    typed: &[Arc<StagedTypedInsert>],
    published: &[BinarySequenceValueReference],
    transaction_catalog: &CatalogSnapshot,
) -> Result<Option<Vec<crate::typed_insert_aggregate::LiveTypedInsertSequenceRestart>>, ExecuteError>
{
    let has_sequence_reset = operations.iter().any(|operation| {
        matches!(operation, TransactionOperation::TableReset(reset) if reset.sequence_reset_identity.is_some())
    });
    if catalog_commands.is_empty() && !has_sequence_reset {
        return Ok(typed
            .iter()
            .all(|staged| staged.private_sequence_advances.is_empty())
            .then(Vec::new));
    }
    let mut commands = BTreeMap::<u32, (&StagedCatalogCommand, &crate::SequenceRestart)>::new();
    let mut creates = BTreeMap::<u32, (&StagedCatalogCommand, &crate::CreateSequence)>::new();
    let mut renames = Vec::<(u32, u32, &str, &str)>::new();
    let mut renamed_oids = BTreeSet::new();
    for staged in catalog_commands {
        match &staged.command {
            Command::CreateSequence(create) => {
                if creates.insert(staged.ordinal, (staged, create)).is_some() {
                    return Ok(None);
                }
            }
            Command::SequenceRestart(restart) => {
                if commands.insert(staged.ordinal, (staged, restart)).is_some() {
                    return Ok(None);
                }
            }
            Command::RenameSequence(rename) => {
                let Some(identity) = staged.sequence_identity.as_ref() else {
                    return Ok(None);
                };
                let [target] = identity.targets.as_slice() else {
                    return Ok(None);
                };
                let (Some(before), Some(after)) =
                    (target.target_before.as_ref(), target.target_after.as_ref())
                else {
                    return Ok(None);
                };
                if identity.ordinal != staged.ordinal
                    || target.before_name != rename.old_name
                    || target.after_name.as_deref() != Some(rename.new_name.as_str())
                    || before.oid == 0
                    || before.oid != after.oid
                    || !renamed_oids.insert(before.oid)
                    || transaction_catalog
                        .relational_sequences
                        .get(&rename.new_name)
                        .is_none_or(|sequence| sequence.oid != before.oid)
                    || transaction_catalog
                        .relational_sequences
                        .contains_key(&rename.old_name)
                {
                    return Ok(None);
                }
                renames.push((
                    staged.ordinal,
                    before.oid,
                    rename.old_name.as_str(),
                    rename.new_name.as_str(),
                ));
            }
            // Domain, table, and FK metadata are carried by the existing S3 ordered catalog
            // body. They own no sequence state themselves, so they cannot add a terminal scalar
            // effect or alter the private sequence proof below.
            Command::CreateDomain(_)
            | Command::CreateTable(_)
            | Command::CreateIndex(_)
            | Command::DropIndex(_)
            | Command::AddForeignKey(_) => {}
            command if command_is_view_lifecycle(command) => {}
            _ => return Ok(None),
        }
    }
    let mut last_effect_operation = BTreeMap::<u32, u32>::new();
    for reference in published {
        let Some(staged) = typed.get(reference.statement_ordinal as usize) else {
            return Ok(None);
        };
        last_effect_operation
            .entry(reference.sequence_oid)
            .and_modify(|ordinal| *ordinal = (*ordinal).max(staged.operation_ordinal))
            .or_insert(staged.operation_ordinal);
    }
    let mut prior_outcome = BTreeMap::<(u32, u32), gpu_db_wal::CanonicalDigest>::new();
    let mut matched = BTreeSet::<u32>::new();
    for staged in typed {
        for advance in staged.private_sequence_advances.iter() {
            if !matches!(advance.lifetime_origin, 1 | 2)
                || advance.owner_statement_ordinal >= staged.operation_ordinal
            {
                return Ok(None);
            }
            let (owner_digest, first_state) = match advance.owner_kind {
                1 => {
                    if let Some(creator_column_ordinal) =
                        advance.owner_creator_catalog_column_ordinal
                    {
                        let mut owners = catalog_commands
                            .iter()
                            .filter(|owner| owner.ordinal == advance.owner_statement_ordinal);
                        let (Some(owner), None) = (owners.next(), owners.next()) else {
                            return Ok(None);
                        };
                        let Command::CreateTable(create) = &owner.command else {
                            return Ok(None);
                        };
                        let Some(column) = create.columns.get(creator_column_ordinal as usize)
                        else {
                            return Ok(None);
                        };
                        let Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) = &column.default
                        else {
                            return Ok(None);
                        };
                        if advance.lifetime_origin != 2
                            || owner.sequence_input_oids.get(sequence)
                                != Some(&advance.sequence_oid)
                            || !transaction_catalog
                                .relational_sequences
                                .values()
                                .any(|sequence| sequence.oid == advance.sequence_oid)
                        {
                            return Ok(None);
                        }
                        // The S3 CREATE TABLE command fixes the original default name and its
                        // statement-local OID.  The S2 effect intentionally carries the final
                        // effective name, so a later private RENAME remains a stable-OID
                        // continuation rather than a second sequence owner.
                        (owner.statement_digest, (1, true))
                    } else {
                        let Some((owner, create)) = creates.get(&advance.owner_statement_ordinal)
                        else {
                            return Ok(None);
                        };
                        let Some(identity) = owner.sequence_identity.as_ref() else {
                            return Ok(None);
                        };
                        let [target] = identity.targets.as_slice() else {
                            return Ok(None);
                        };
                        let stable = target
                            .target_after
                            .as_ref()
                            .or(target.target_before.as_ref());
                        if advance.lifetime_origin != 2
                            || identity.ordinal != owner.ordinal
                            || create.name != target.before_name
                            || stable.is_none_or(|stable| stable.oid != advance.sequence_oid)
                        {
                            return Ok(None);
                        }
                        // CREATE's catalog state begins at (1,false); this first typed default
                        // allocation is the lifecycle's first visible effect and must therefore
                        // close at (1,true), exactly as the S2 private receipt records.
                        (owner.statement_digest, (1, true))
                    }
                }
                2 => {
                    let Some((owner, restart)) = commands.get(&advance.owner_statement_ordinal)
                    else {
                        return Ok(None);
                    };
                    if advance.sequence_name != restart.name {
                        return Ok(None);
                    }
                    matched.insert(advance.owner_statement_ordinal);
                    (owner.statement_digest, (restart.value, true))
                }
                3 => {
                    let Some(TransactionOperation::TableReset(reset)) =
                        operations.get(advance.owner_statement_ordinal as usize)
                    else {
                        return Ok(None);
                    };
                    let Some(identity) = reset.sequence_reset_identity.as_ref() else {
                        return Ok(None);
                    };
                    let target_matches = identity.targets.iter().any(|target| {
                        target.before_name == advance.sequence_name
                            && target
                                .target_before
                                .as_ref()
                                .is_some_and(|sequence| sequence.oid == advance.sequence_oid)
                            && target
                                .target_after
                                .as_ref()
                                .is_some_and(|sequence| sequence.oid == advance.sequence_oid)
                    });
                    if identity.ordinal != reset.ordinal
                        || identity.table != reset.table
                        || !target_matches
                    {
                        return Ok(None);
                    }
                    (reset.statement_digest, (1, true))
                }
                _ => return Ok(None),
            };
            if advance.owner_statement_digest != owner_digest {
                return Ok(None);
            }
            match prior_outcome.insert(
                (advance.owner_statement_ordinal, advance.sequence_oid),
                advance.outcome_digest,
            ) {
                None => {
                    if advance.predecessor_tag != 1
                        || advance.predecessor_digest != owner_digest
                        || advance.next_state != first_state
                    {
                        return Ok(None);
                    }
                }
                Some(prior) => {
                    if advance.predecessor_tag != 2 || advance.predecessor_digest != prior {
                        return Ok(None);
                    }
                }
            }
            last_effect_operation
                .entry(advance.sequence_oid)
                .and_modify(|ordinal| *ordinal = (*ordinal).max(staged.operation_ordinal))
                .or_insert(staged.operation_ordinal);
        }
    }
    for (rename_ordinal, sequence_oid, old_name, new_name) in renames {
        let creator_before = creates.iter().any(|(ordinal, (owner, create))| {
            *ordinal < rename_ordinal
                && create.name == old_name
                && owner.sequence_identity.as_ref().is_some_and(|identity| {
                    identity.ordinal == owner.ordinal
                        && identity.targets.as_slice().iter().any(|target| {
                            target.before_name == old_name
                                && target
                                    .target_after
                                    .as_ref()
                                    .or(target.target_before.as_ref())
                                    .is_some_and(|sequence| sequence.oid == sequence_oid)
                        })
                })
        }) || catalog_commands.iter().any(|owner| {
            owner.ordinal < rename_ordinal
                && matches!(&owner.command, Command::CreateTable(create) if create.columns.iter().any(|column| {
                    matches!(
                        &column.default,
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) if sequence == old_name
                    ) && owner.sequence_input_oids.get(old_name) == Some(&sequence_oid)
                }))
        });
        let mut published_operations = published
            .iter()
            .filter(|reference| reference.sequence_oid == sequence_oid)
            .filter_map(|reference| typed.get(reference.statement_ordinal as usize))
            .map(|staged| staged.operation_ordinal);
        let mut private_effects = typed.iter().flat_map(|staged| {
            staged
                .private_sequence_advances
                .iter()
                .filter(move |advance| advance.sequence_oid == sequence_oid)
                .map(move |advance| (staged.operation_ordinal, advance.sequence_name.as_str()))
        });
        let has_before = published_operations
            .clone()
            .any(|operation| operation < rename_ordinal)
            || private_effects
                .clone()
                .any(|(operation, _)| operation < rename_ordinal);
        let has_after = published_operations.any(|operation| operation > rename_ordinal)
            || private_effects
                .clone()
                .any(|(operation, _)| operation > rename_ordinal);
        let final_private_name = private_effects
            .rfind(|(operation, _)| *operation > rename_ordinal)
            .map(|(_, name)| name);
        // A final S3 rename may follow the last private default effect.  In that shape no later
        // S2 record can carry the new display name, but the already-retained old-name S2 effect
        // and this exact stable-OID S3 transition together still close one sequence state.  The
        // retained replay artifact normalizes that final publication name from the same S3
        // identity; it does not create a rename terminal or a second sequence owner.
        let terminal_s3_rename = !has_after
            && last_effect_operation
                .get(&sequence_oid)
                .is_some_and(|operation| *operation < rename_ordinal);
        if old_name == new_name
            || !(has_before || creator_before)
            || !(has_after && final_private_name == Some(new_name) || terminal_s3_rename)
        {
            return Ok(None);
        }
    }
    let mut terminal_sequences = BTreeSet::new();
    let mut terminals = Vec::new();
    for (ordinal, (owner, restart)) in commands {
        if matched.contains(&ordinal) {
            continue;
        }
        let Some(sequence) = transaction_catalog.relational_sequences.get(&restart.name) else {
            return Ok(None);
        };
        let Some(source_operation) = last_effect_operation.get(&sequence.oid).copied() else {
            return Ok(None);
        };
        if source_operation >= ordinal
            || sequence.oid == 0
            || (sequence.last_value, sequence.is_called) != (restart.value, false)
            || !terminal_sequences.insert(sequence.oid)
        {
            return Ok(None);
        }
        terminals.push(
            crate::typed_insert_aggregate::LiveTypedInsertSequenceRestart {
                owner_operation_ordinal: ordinal,
                owner_statement_digest: owner.statement_digest,
                sequence_oid: sequence.oid,
                sequence_name: sequence.name.clone(),
                last_value: restart.value,
            },
        );
    }
    terminals.sort_unstable_by_key(|restart| restart.owner_operation_ordinal);
    Ok(Some(terminals))
}

pub(crate) fn indexed_generation_inputs(
    table: &RelationalTable,
    predecessor_index_roots: &[crate::engine_state::TypedIndexGenerationRoot],
    row_count: u32,
    resets_existing_rows: bool,
    initial_table_absent: bool,
    created_index_ids: &[u64],
    retired_index_ids: &[u64],
) -> Result<Vec<TypedInsertRuntimeGenerationIndexInput>, ExecuteError> {
    if table.indexes.is_empty() {
        if predecessor_index_roots.is_empty() || resets_existing_rows || initial_table_absent {
            return Ok(Vec::new());
        }
        return Err(ExecuteError::Engine(EngineError::Durability(
            "index-neutral codec-5 table retained named index roots".to_string(),
        )));
    }
    let created_are_ordered = created_index_ids.iter().enumerate().all(|(ordinal, id)| {
        *id != 0 && *id != u64::MAX && (ordinal == 0 || created_index_ids[ordinal - 1] < *id)
    });
    let retired_are_ordered = retired_index_ids.iter().enumerate().all(|(ordinal, id)| {
        *id != 0 && *id != u64::MAX && (ordinal == 0 || retired_index_ids[ordinal - 1] < *id)
    });
    if !created_are_ordered
        || !retired_are_ordered
        || created_index_ids
            .iter()
            .any(|id| retired_index_ids.binary_search(id).is_ok())
        || (resets_existing_rows
            && (!created_index_ids.is_empty() || !retired_index_ids.is_empty()))
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 S3 index transition identities are not an exact append-only proof".to_string(),
        )));
    }
    if !resets_existing_rows && !initial_table_absent {
        let exact_created = table
            .indexes
            .iter()
            .filter(|index| {
                predecessor_index_roots
                    .iter()
                    .all(|root| root.stable_index_id != u64::from(index.oid))
            })
            .map(|index| u64::from(index.oid))
            .eq(created_index_ids.iter().copied());
        let exact_retired = predecessor_index_roots
            .iter()
            .filter(|root| {
                table
                    .indexes
                    .iter()
                    .all(|index| u64::from(index.oid) != root.stable_index_id)
            })
            .map(|root| root.stable_index_id)
            .eq(retired_index_ids.iter().copied());
        if table.indexes.len()
            != predecessor_index_roots
                .len()
                .saturating_sub(retired_index_ids.len())
                .saturating_add(created_index_ids.len())
            || !exact_created
            || !exact_retired
        {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 catalog indexes and typed predecessor roots differ outside the S3 index-transition proof".to_string(),
            )));
        }
    }
    let mut inputs = Vec::with_capacity(table.indexes.len());
    let mut key_start = 0_u32;
    let mut effect_start = 0_u32;
    for (raw_catalog_index_ordinal, index) in table.indexes.iter().enumerate() {
        let stable_index_id = u64::from(index.oid);
        let predecessor = predecessor_index_roots
            .iter()
            .find(|root| root.stable_index_id == stable_index_id);
        let (base_generation, base_root) = match predecessor {
            Some(predecessor) => (predecessor.index_generation, predecessor.index_root),
            // ResetThenRowSetInsert and the first rowset of a transaction-created table derive
            // every successor index from their exact replacement image.  Their paired zero
            // values are an explicit absence sentinel, never a host-derived root.
            None if resets_existing_rows
                || initial_table_absent
                || created_index_ids.binary_search(&stable_index_id).is_ok() =>
            {
                (0, [0; 32])
            }
            None => {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "codec-5 named index lacks its GPU-authenticated predecessor root".to_string(),
                )))
            }
        };
        let mut key_columns = Vec::with_capacity(index.key_columns.len());
        for (key_ordinal, key_name) in index.key_columns.iter().enumerate() {
            let (catalog_column_ordinal, column) = table
                .columns
                .iter()
                .enumerate()
                .find(|(_, column)| column.name == *key_name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 named index key is absent from its catalog table".to_string(),
                    ))
                })?;
            key_columns.push(
                gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                    key_ordinal: u32::try_from(key_ordinal).map_err(|_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 index key ordinal exceeds u32".to_string(),
                        ))
                    })?,
                    catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(
                        |_| {
                            ExecuteError::Engine(EngineError::Durability(
                                "codec-5 index key catalog ordinal exceeds u32".to_string(),
                            ))
                        },
                    )?,
                    stable_column_id: column.id,
                    attnum: column.attnum,
                    storage: crate::typed_insert_batch::typed_image_sql_storage(column.ty),
                    declared_type_oid: column.type_oid,
                    signed_type_size: column.type_size,
                    column_name_digest: crate::typed_insert_aggregate::write001_identifier_digest(
                        &column.name,
                    )
                    .map_err(ExecuteError::Engine)?,
                },
            );
        }
        let key_count = u32::try_from(key_columns.len()).map_err(|_| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 named index key count exceeds u32".to_string(),
            ))
        })?;
        if stable_index_id == 0
            || key_count == 0
            || ((base_generation == 0) != (base_root == [0; 32]))
            || (!resets_existing_rows
                && !initial_table_absent
                && base_generation == 0
                && created_index_ids.binary_search(&stable_index_id).is_err())
        {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 named index predecessor shape is invalid".to_string(),
            )));
        }
        let index_flags = u32::from(index.unique)
            | (u32::from(index.primary_key) << 1)
            | (u32::from(index.unique_constraint) << 2)
            | (1 << 3);
        inputs.push(TypedInsertRuntimeGenerationIndexInput {
            descriptor: gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                stable_index_id,
                raw_catalog_index_ordinal: u32::try_from(raw_catalog_index_ordinal).map_err(
                    |_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 raw catalog index ordinal exceeds u32".to_string(),
                        ))
                    },
                )?,
                index_flags,
                // Canonical S7 descriptors encode the ordinary PostgreSQL distinct-NULL
                // policy as one. The device revalidates this byte before deriving any root.
                null_equality_policy: 1,
                base_generation,
                base_root,
                key_start,
                key_count,
                effect_start,
                effect_count: row_count,
            },
            key_columns: key_columns.into_boxed_slice(),
        });
        key_start = key_start.checked_add(key_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 named index key range overflows".to_string(),
            ))
        })?;
        effect_start = effect_start.checked_add(row_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 named index effect range overflows".to_string(),
            ))
        })?;
    }
    if !resets_existing_rows
        && !initial_table_absent
        && predecessor_index_roots.iter().any(|root| {
            retired_index_ids
                .binary_search(&root.stable_index_id)
                .is_err()
                && !inputs
                    .iter()
                    .any(|input| input.descriptor.stable_index_id == root.stable_index_id)
        })
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 named index predecessor roots contain an unbound identity".to_string(),
        )));
    }
    Ok(inputs)
}

/// An existing table may gain an index only through the S3 ordered lifecycle identity already
/// sealed for this codec-5 transaction.  This extracts that exact absent-to-present identity
/// from the final catalog postimage; it does not inspect a selector, infer a name, or mint a
/// recovery/catalog path.
#[derive(Clone, Default)]
struct S3ExistingTableIndexChanges {
    created: Vec<crate::typed_insert_aggregate::LiveTypedInsertCreatedIndex>,
    retired_index_ids: Vec<u64>,
}

/// Select the S3-proven named-index delta for an already-published table. The common ordered
/// catalog applier remains the authority for each command; this only tells the one GPU
/// generation which predecessor roots are absent or retired in its final catalog image.
fn s3_index_changes_on_existing_tables(
    catalog_composition: Option<&Codec5CatalogComposition>,
    published_catalog: &CatalogSnapshot,
    transaction_catalog: &BTreeMap<String, RelationalTable>,
) -> Result<BTreeMap<String, S3ExistingTableIndexChanges>, ExecuteError> {
    let mut by_table = BTreeMap::new();
    let Some(composition) = catalog_composition else {
        return Ok(by_table);
    };
    for operation in &composition.record.index_lifecycle_operations {
        for target in &operation.targets {
            if let (
                Some(table_before),
                Some(table_after),
                None,
                Some(index_after),
                Some(after_name),
            ) = (
                target.table_before.as_ref(),
                target.table_after.as_ref(),
                target.index_before.as_ref(),
                target.index_after.as_ref(),
                target.after_name.as_ref(),
            ) {
                if table_before.oid != table_after.oid {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "codec-5 S3 CREATE INDEX changed its owning table identity".to_string(),
                    )));
                }
                // A legal ordered CREATE followed by DROP has no final catalog postimage and
                // therefore contributes no paired-zero root; S3 still validates that history.
                let Some((table_name, final_table)) =
                    transaction_catalog.iter().find(|(_, table)| {
                        table.oid == table_after.oid
                            && table.indexes.iter().any(|index| {
                                index.oid == index_after.oid
                                    && index.name == *after_name
                                    && index.table == table.name
                            })
                    })
                else {
                    continue;
                };
                if index_after.table_oid != final_table.oid
                    || published_catalog
                        .relational_catalog
                        .get(table_name)
                        .is_none_or(|published| {
                            published.oid != final_table.oid
                                || published
                                    .indexes
                                    .iter()
                                    .any(|index| index.oid == index_after.oid)
                        })
                {
                    continue;
                }
                let entries = by_table.entry(table_name.clone()).or_default();
                if entries
                    .created
                    .iter()
                    .any(|entry| entry.stable_index_id == u64::from(index_after.oid))
                {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "codec-5 S3 repeats a transaction-created index identity".to_string(),
                    )));
                }
                entries
                    .created
                    .push(crate::typed_insert_aggregate::LiveTypedInsertCreatedIndex {
                        stable_index_id: u64::from(index_after.oid),
                        operation_ordinal: operation.ordinal,
                    });
                continue;
            }
            let (Some(table_before), Some(table_after), Some(index_before), None) = (
                target.table_before.as_ref(),
                target.table_after.as_ref(),
                target.index_before.as_ref(),
                target.index_after.as_ref(),
            ) else {
                continue;
            };
            if table_before.oid != table_after.oid || index_before.table_oid != table_before.oid {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "codec-5 S3 DROP INDEX changed its owning table identity".to_string(),
                )));
            }
            let Some((table_name, published_table)) = published_catalog
                .relational_catalog
                .iter()
                .find(|(_, table)| {
                    table.oid == table_before.oid
                        && table
                            .indexes
                            .iter()
                            .any(|index| index.oid == index_before.oid)
                })
            else {
                // A transaction-created index can be dropped before commit; it never had a
                // published GPU root and cannot be retired from this generation.
                continue;
            };
            let Some(final_table) = transaction_catalog.get(table_name) else {
                continue;
            };
            if final_table.oid != table_after.oid
                || final_table
                    .indexes
                    .iter()
                    .any(|index| index.oid == index_before.oid)
            {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "codec-5 S3 DROP INDEX does not close its final catalog absence".to_string(),
                )));
            }
            let entries = by_table.entry(published_table.name.clone()).or_default();
            let retired = u64::from(index_before.oid);
            if entries.retired_index_ids.contains(&retired) {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "codec-5 S3 repeats a retired published index identity".to_string(),
                )));
            }
            entries.retired_index_ids.push(retired);
        }
    }
    for entries in by_table.values_mut() {
        entries
            .created
            .sort_unstable_by_key(|entry| entry.stable_index_id);
        entries.retired_index_ids.sort_unstable();
        if entries.created.iter().any(|created| {
            entries
                .retired_index_ids
                .binary_search(&created.stable_index_id)
                .is_ok()
        }) {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 S3 index identity cannot be both created and retired".to_string(),
            )));
        }
    }
    Ok(by_table)
}

fn foreign_index_generations<'a>(
    table: &RelationalTable,
    dependencies: &'a BTreeMap<String, RelationalTable>,
    roots: &crate::engine_state::TypedGenerationRootSnapshot,
) -> Result<
    Vec<crate::typed_insert_aggregate::LiveTypedInsertForeignIndexGeneration<'a>>,
    ExecuteError,
> {
    let mut generations = Vec::new();
    for foreign_key in &table.foreign_keys {
        let parent = dependencies
            .get(&foreign_key.referenced_table)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 FK parent is absent from the sealed transaction catalog".to_string(),
                ))
            })?;
        let index = parent
            .indexes
            .iter()
            .find(|index| {
                index.table == parent.name
                    && index.column == foreign_key.referenced_column
                    && index.unique
                    && (index.primary_key || index.unique_constraint)
                    && index.key_columns.as_slice() == [foreign_key.referenced_column.as_str()]
            })
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 FK parent has no exact supporting index".to_string(),
                ))
            })?;
        if generations.iter().any(
            |generation: &crate::typed_insert_aggregate::LiveTypedInsertForeignIndexGeneration<
                '_,
            >| {
                generation.parent.stable_table_id == parent.stable_table_id
                    && generation.catalog.oid == index.oid
            },
        ) {
            continue;
        }
        let parent_root = roots.table(parent.stable_table_id).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 FK parent has no GPU-authenticated table generation".to_string(),
            ))
        })?;
        let index_root = roots
            .table_index_root(parent.stable_table_id, u64::from(index.oid))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 FK parent has no GPU-authenticated supporting-index generation"
                        .to_string(),
                ))
            })?;
        generations.push(
            crate::typed_insert_aggregate::LiveTypedInsertForeignIndexGeneration {
                parent,
                parent_schema_digest: table_schema_digest(parent)?,
                parent_generation: parent_root.data_generation,
                parent_root: parent_root.table_root,
                catalog: index,
                index_generation: index_root.index_generation,
                index_root: index_root.index_root,
            },
        );
    }
    generations.sort_unstable_by_key(|generation| {
        (
            generation.parent.stable_table_id,
            u64::from(generation.catalog.oid),
        )
    });
    Ok(generations)
}

/// Prepared one-record codec-5 ingredients. The generic transaction finalizer alone consumes
/// these through proposal, exact WAL/status/timestamp ownership, device apply, and publication.
#[must_use = "the generic transaction finalizer must either consume or drop the prepared codec-5 operation"]
pub(super) struct PreparedCodec5Operation<'a> {
    /// Moved exactly once into the generic typed control-plane reservation. This leaf cannot
    /// propose or append it itself.
    pub(super) record: Option<gpu_db_wal::PreparedCanonicalWalRecord>,
    pub(super) payload_authority: Arc<[u8]>,
    pub(super) plans: Vec<crate::engine_residency::DeviceInsertPlan<'a>>,
    pub(super) root_publication: Option<crate::engine_commit::LiveTypedGenerationRootPublication>,
    pub(super) tables: Box<[String]>,
    pub(super) proposed_range: crate::wal_binary::ProposedRowIdRange,
    pub(super) allocator_high_water: u64,
    pub(super) affected_rows: u64,
    pub(super) write_set: WriteSet,
    pub(super) private_sequence_publications:
        Box<[crate::engine_commit::LiveTypedPrivateSequencePublication]>,
    pub(super) catalog_composition: Option<BinaryTransactionRecord>,
    /// Retains the CUDA-completed generation authority until the generic terminal consumes the
    /// accompanying plan. It is intentionally not a publication handle.
    pub(super) generations: Box<[TypedInsertRuntimeGenerationOutput]>,
}

struct PreparedCodec5Table<'catalog> {
    table: &'catalog RelationalTable,
    table_schema_digest: gpu_db_wal::CanonicalDigest,
    staged: Vec<Arc<StagedTypedInsert>>,
    source: Option<crate::typed_insert_batch::PreparedResidentAppendSource>,
    final_image: Arc<[u8]>,
    /// One immutable digest of `final_image`, derived before the same value is handed to CUDA.
    /// The codec-5 S7 closure reuses this fact rather than rescanning the sealed image twice.
    image_content_digest: gpu_db_wal::CanonicalDigest,
    final_writers: Vec<crate::typed_insert_aggregate::LiveTypedInsertFinalWriter>,
    survivor_row_ids: Box<[u64]>,
    resets_existing_rows: bool,
    /// The table identity was created by this catalog composition and has no published device
    /// root yet.  Its first row set is therefore one GPU CREATE-with-rows generation, not a
    /// host-created empty predecessor followed by a second mutation path.
    initial_table_absent: bool,
    /// Exact S3 absent-to-present index identities for this already-published table.  This is
    /// retained solely through the one GPU root/action and the existing S7 writer closure.
    created_indexes_on_existing_table:
        Vec<crate::typed_insert_aggregate::LiveTypedInsertCreatedIndex>,
    /// Exact S3 DROP INDEX identities that had an already-published typed root. They are
    /// removed by the same final GPU manifest/root swap as the surviving and created indexes.
    retired_indexes_on_existing_table: Vec<u64>,
    generation: Option<TypedInsertRuntimeGenerationOutput>,
    predecessor: crate::engine_state::TypedTableGenerationRoot,
    predecessor_index_roots: Vec<crate::engine_state::TypedIndexGenerationRoot>,
    successor_index_roots: Vec<crate::engine_state::TypedIndexGenerationRoot>,
    data_generation_after: u64,
    final_table_root: gpu_db_wal::CanonicalDigest,
    final_database_root: gpu_db_wal::CanonicalDigest,
    first_row_id: u64,
    row_allocator_high_water: u64,
    final_logical_row_count: u64,
}

/// Resolves the transaction-private row identity into its one committed allocator identity.
///
/// Ordinary single-statement typed INSERTs allocate one already-validated dense provisional
/// interval.  Retaining a `(String, u64)` map for every one of those rows only repeats that
/// interval proof and reconstructs table-name owners at the terminal.  Mixed table, rewrite,
/// and sequence shapes retain the existing exact map; this scalar form is permitted only after
/// the caller has checked the same interval and overlay membership facts.
enum Codec5FinalIdentityResolution {
    DenseSingleTable {
        table: String,
        first_provisional: u64,
        final_base: u64,
        row_count: u64,
    },
    Mapped(BTreeMap<(String, u64), u64>),
}

impl Codec5FinalIdentityResolution {
    fn resolve(&self, table: &str, provisional: u64) -> Option<u64> {
        match self {
            Self::DenseSingleTable {
                table: expected_table,
                first_provisional,
                final_base,
                row_count,
            } => (table == expected_table)
                .then(|| provisional.checked_sub(*first_provisional))
                .flatten()
                .filter(|offset| offset < row_count)
                .and_then(|offset| final_base.checked_add(offset)),
            Self::Mapped(ids) => ids.get(&(table.to_string(), provisional)).copied(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_generic_codec5_operation<'a>(
    engine: &'a Engine,
    txn_id: TxnId,
    staged: &SelectedGenericCodec5,
    published_sequence_references: &[crate::BinarySequenceValueReference],
    provisional_inserts: &BTreeSet<(String, u64)>,
    transaction_catalog: &'a BTreeMap<String, RelationalTable>,
    transaction_catalog_snapshot: &CatalogSnapshot,
    catalog_commands: &[StagedCatalogCommand],
    canonical_identity: gpu_db_wal::CanonicalIdentity,
    leader_epoch: u64,
    expected_commit_seq: Index,
    catalog_epoch: u64,
    catalog_digest: gpu_db_wal::CanonicalDigest,
    mode: crate::typed_insert_aggregate::LiveTypedInsertMode,
    request_digest: gpu_db_wal::CanonicalDigest,
    catalog_composition: Option<Codec5CatalogComposition>,
    isolation: gpu_db_wal::CanonicalIsolation,
    final_base: u64,
    allocator_high_water: u64,
    mut indexed_lifecycle: Option<crate::engine_state::TransactionNamedIndexPublicationGuard<'a>>,
) -> Result<PreparedCodec5Operation<'a>, ExecuteError> {
    #[cfg(feature = "probe-timing")]
    let probe_operation_started = std::time::Instant::now();
    let first = staged.first().ok_or_else(|| {
        ExecuteError::Engine(EngineError::Durability(
            "codec-5 generic operation has no typed INSERT contribution".to_string(),
        ))
    })?;
    let row_count = staged.iter().try_fold(0_u32, |count, contribution| {
        count
            .checked_add(u32::try_from(contribution.rows_consumed()).map_err(|_| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 generic row count exceeds aggregate framing".to_string(),
                ))
            })?)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 generic row count overflows aggregate framing".to_string(),
                ))
            })
    })?;
    let contributions_are_exact = staged.iter().enumerate().all(|(ordinal, contribution)| {
        let statement_table = contribution.catalog_dependencies.get(&contribution.table);
        contribution.statement_ordinal as usize == ordinal
            && contribution.read_snapshot == first.read_snapshot
            && contribution.read_snapshot < expected_commit_seq
            && statement_table.is_some_and(|table| {
                contribution.table_schema_digest == table_schema_digest(table).unwrap_or([0; 32])
            })
            && transaction_catalog
                .get(&contribution.table)
                .is_some_and(|table| {
                    contribution.table_oid == table.oid
                        && contribution.stable_table_id == table.stable_table_id
                        && statement_table.is_some_and(|statement_table| {
                            same_transaction_row_catalog_dependency(
                                statement_table,
                                table,
                                &BTreeMap::new(),
                                transaction_catalog_snapshot,
                                catalog_commands,
                            )
                        })
                })
    });
    let mut grouped = BTreeMap::<u64, Vec<Arc<StagedTypedInsert>>>::new();
    for contribution in staged.iter() {
        grouped
            .entry(contribution.stable_table_id)
            .or_default()
            .push(Arc::clone(contribution));
    }
    #[cfg(feature = "probe-timing")]
    let multi_table = grouped.len() > 1;
    #[cfg(feature = "probe-timing")]
    if multi_table {
        eprintln!(
            "[probe] codec5_plural_prepare stage=selected tables={} statements={} rows={row_count}",
            grouped.len(),
            staged.len(),
        );
    }
    let table_shapes_are_admitted = grouped.iter().all(|(stable_table_id, contributions)| {
        let Some(contribution) = contributions.first() else {
            return false;
        };
        transaction_catalog
            .get(&contribution.table)
            .is_some_and(|table| {
                table.stable_table_id == *stable_table_id
                    && (table.indexes.is_empty()
                        || table.indexes.iter().all(|index| {
                            (!index.primary_key || index.unique)
                                && (!index.unique_constraint || index.unique)
                                && !index.key_columns.is_empty()
                                && crate::engine_residency::index_all_key_columns_foldable(
                                    table, index,
                                )
                        }))
            })
    });
    let has_surviving_indexed_table = grouped.values().any(|contributions| {
        transaction_catalog
            .get(&contributions[0].table)
            .is_some_and(|table| {
                !table.indexes.is_empty()
                    && staged
                        .rows_for_table(&table.name)
                        .iter()
                        .any(|(_, row)| row.survives)
            })
    });
    // RETURNING is a presentation effect already completed from the transaction's private GPU
    // shard before this commit owner is built. It does not alter the durable row image, device
    // apply, or publication lifecycle and therefore cannot select another WAL body.
    if row_count == 0
        || !contributions_are_exact
        || !table_shapes_are_admitted
        || has_surviving_indexed_table != indexed_lifecycle.is_some()
        || allocator_high_water
            != final_base
                .checked_add(u64::from(row_count))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 generic allocator range overflows".to_string(),
                    ))
                })?
    {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "generic codec-5 operation lost its validated transaction shape".to_string(),
        )));
    }
    if engine.read_state.mvcc.current_row_id() != final_base {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "generic codec-5 global allocator frontier differs from canonical row-id base"
                .to_string(),
        )));
    }
    let mut final_row_ids = BTreeMap::<String, Vec<u64>>::new();
    let mut published_sequence_references = published_sequence_references.to_vec();
    let dense_single_table = (staged.len() == 1
        && grouped.len() == 1
        && !staged.has_rewritten_rows
        && staged.final_table_resets.is_empty()
        && published_sequence_references.is_empty())
    .then(|| staged.first().expect("one checked typed contribution"));
    let final_identity_by_provisional = if let Some(contribution) = dense_single_table {
        let row_count = u64::try_from(contribution.provisional_row_ids.len()).map_err(|_| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 dense provisional row count exceeds u64".to_string(),
            ))
        })?;
        let first_provisional = contribution
            .provisional_row_ids
            .first()
            .copied()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 dense typed contribution has no provisional rows".to_string(),
                ))
            })?;
        let interval_high_water = first_provisional.checked_add(row_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 dense provisional interval overflows".to_string(),
            ))
        })?;
        let exact_dense_interval = contribution
            .provisional_row_ids
            .iter()
            .copied()
            .eq(first_provisional..interval_high_water)
            && provisional_inserts.len() == contribution.provisional_row_ids.len()
            && provisional_inserts
                .iter()
                .map(|(table, row_id)| (table.as_str(), *row_id))
                .eq(contribution
                    .provisional_row_ids
                    .iter()
                    .copied()
                    .map(|row_id| (contribution.table.as_str(), row_id)))
            && staged.final_rows.len() == contribution.provisional_row_ids.len()
            && staged
                .final_rows
                .keys()
                .map(|(stable_table_id, row_id)| (*stable_table_id, *row_id))
                .eq(contribution
                    .provisional_row_ids
                    .iter()
                    .copied()
                    .map(|row_id| (contribution.stable_table_id, row_id)));
        if !exact_dense_interval {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 dense identity interval differs from its selected typed overlay"
                    .to_string(),
            )));
        }
        let final_high_water = final_base.checked_add(row_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 dense final allocator interval overflows".to_string(),
            ))
        })?;
        final_row_ids.insert(
            contribution.table.clone(),
            (final_base..final_high_water).collect(),
        );
        Codec5FinalIdentityResolution::DenseSingleTable {
            table: contribution.table.clone(),
            first_provisional,
            final_base,
            row_count,
        }
    } else {
        let original_final_ids =
            Engine::resolved_insert_identities(provisional_inserts, final_base)?;
        if original_final_ids.len() != staged.final_rows.len()
            || original_final_ids.keys().any(|(table, row_id)| {
                transaction_catalog.get(table).is_none_or(|table| {
                    !staged
                        .final_rows
                        .contains_key(&(table.stable_table_id, *row_id))
                })
            })
        {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "selected codec-5 rows differ from the allocator identity set".to_string(),
            )));
        }
        let mut identities = BTreeMap::<(String, u64), u64>::new();
        for stable_table_id in staged
            .final_rows
            .keys()
            .map(|(stable_table_id, _)| *stable_table_id)
            .collect::<BTreeSet<_>>()
        {
            let contribution = staged
                .iter()
                .find(|contribution| contribution.stable_table_id == stable_table_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 selected stable table identity has no typed source".to_string(),
                    ))
                })?;
            let table = &contribution.table;
            let selected = staged.rows_for_table(table);
            let first = selected
                .iter()
                .filter_map(|(provisional, _)| {
                    original_final_ids
                        .get(&(table.clone(), **provisional))
                        .copied()
                })
                .min()
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 table has no allocator interval".to_string(),
                    ))
                })?;
            let ordered = selected
                .iter()
                .filter(|(_, row)| row.survives)
                .chain(selected.iter().filter(|(_, row)| !row.survives));
            for (offset, (provisional, _)) in ordered.enumerate() {
                let final_id = first
                    .checked_add(u64::try_from(offset).map_err(|_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 table allocator offset exceeds u64".to_string(),
                        ))
                    })?)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 table allocator interval overflows".to_string(),
                        ))
                    })?;
                identities.insert((table.clone(), **provisional), final_id);
                final_row_ids
                    .entry(table.clone())
                    .or_default()
                    .push(final_id);
            }
        }
        let old_final_to_new = original_final_ids
            .iter()
            .map(|(identity, old)| {
                identities
                    .get(identity)
                    .copied()
                    .map(|new| (*old, new))
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 row identity remap is incomplete".to_string(),
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, ExecuteError>>()?;
        for reference in published_sequence_references
            .iter_mut()
            .filter(|reference| reference.default_expression)
        {
            reference.row_id = old_final_to_new
                .get(&reference.row_id)
                .copied()
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(format!(
                        "codec-5 sequence receipt {} lost its compact final row identity",
                        reference.transition_txn_id
                    )))
                })?;
        }
        if provisional_inserts.len() != row_count as usize {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "selected codec-5 transaction lost its exact provisional INSERT identity set"
                    .to_string(),
            )));
        }
        for ids in final_row_ids.values_mut() {
            ids.sort_unstable();
        }
        Codec5FinalIdentityResolution::Mapped(identities)
    };
    #[cfg(feature = "probe-timing")]
    let probe_input_nanos = probe_operation_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_table_prepare_started = std::time::Instant::now();
    let original_roots = engine.read_state.typed_generation_roots.load_full();
    let published_catalog = engine.catalog_snapshot();
    let index_changes_on_existing_tables = s3_index_changes_on_existing_tables(
        catalog_composition.as_ref(),
        &published_catalog,
        transaction_catalog,
    )?;
    let mut evolving_roots = Arc::clone(&original_roots);
    let mut root_publication: Option<crate::engine_commit::LiveTypedGenerationRootPublication> =
        None;
    let mut prepared_tables = Vec::with_capacity(grouped.len());
    for (table_ref, (_, contributions)) in grouped.into_iter().enumerate() {
        let table_ref = u32::try_from(table_ref).map_err(|_| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 table reference exceeds u32".to_string(),
            ))
        })?;
        let contribution = contributions
            .first()
            .expect("nonempty grouped codec-5 contribution");
        let table = transaction_catalog
            .get(&contribution.table)
            .ok_or_else(|| {
                ExecuteError::Serialization(
                    "generic codec-5 INSERT target left the transaction catalog".to_string(),
                )
            })?;
        let private_created_table = catalog_composition.as_ref().is_some_and(|composition| {
            composition
                .record
                .created_table_identities
                .get(&contribution.table)
                .is_some_and(|identity| identity.table_oid == table.oid)
        });
        let physical_table = published_catalog
            .relational_catalog
            .get(&contribution.table)
            .filter(|physical| {
                physical.oid == table.oid
                    && physical.stable_table_id == table.stable_table_id
                    && same_transaction_row_catalog_dependency(
                        physical,
                        table,
                        &BTreeMap::new(),
                        transaction_catalog_snapshot,
                        catalog_commands,
                    )
            })
            .or_else(|| private_created_table.then_some(table))
            .ok_or_else(|| {
                ExecuteError::Serialization(
                    "generic codec-5 INSERT physical predecessor changed outside its admitted catalog lifecycle"
                        .to_string(),
                )
            })?;
        let physical_schema_digest = table_schema_digest(physical_table)?;
        let final_table_schema_digest = table_schema_digest(table)?;
        let index_changes_on_existing_table = index_changes_on_existing_tables
            .get(&table.name)
            .cloned()
            .unwrap_or_default();
        let created_indexes_on_existing_table = index_changes_on_existing_table.created;
        let retired_indexes_on_existing_table = index_changes_on_existing_table.retired_index_ids;
        let created_index_ids = created_indexes_on_existing_table
            .iter()
            .map(|index| index.stable_index_id)
            .collect::<Vec<_>>();
        let retired_index_ids = retired_indexes_on_existing_table.as_slice();
        #[cfg(feature = "probe-timing")]
        if multi_table {
            eprintln!(
                "[probe] codec5_plural_prepare stage=table_generation_begin table={} statements={}",
                table.name,
                contributions.len(),
            );
        }
        let table_row_count = contributions
            .iter()
            .try_fold(0_u32, |count, contribution| {
                count
                    .checked_add(u32::try_from(contribution.rows_consumed()).map_err(|_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 table row count exceeds u32".to_string(),
                        ))
                    })?)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 table row count overflows".to_string(),
                        ))
                    })
            })?;
        let ids = final_row_ids.remove(&table.name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 table has no resolved row-ID interval".to_string(),
            ))
        })?;
        let first_row_id = *ids.first().ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 table row-ID interval is empty".to_string(),
            ))
        })?;
        let table_high_water = first_row_id
            .checked_add(u64::from(table_row_count))
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "codec-5 table allocator interval overflows".to_string(),
                ))
            })?;
        if ids.len() != table_row_count as usize
            || ids.iter().copied().ne(first_row_id..table_high_water)
        {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 table rows do not occupy one exact allocator interval".to_string(),
            )));
        }
        let selected_rows = staged.rows_for_table(&table.name);
        if selected_rows.len() != table_row_count as usize {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 final private-overlay rows differ from their typed source cardinality"
                    .to_string(),
            )));
        }
        let surviving_rows = selected_rows
            .iter()
            .filter(|(_, row)| row.survives)
            .copied()
            .collect::<Vec<_>>();
        let survivor_row_ids = surviving_rows
            .iter()
            .map(|(provisional, _)| {
                final_identity_by_provisional
                    .resolve(&table.name, **provisional)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 survivor lost its final row identity".to_string(),
                        ))
                    })
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?
            .into_boxed_slice();
        let surviving_row_count = u32::try_from(surviving_rows.len()).map_err(|_| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 table survivor count exceeds u32".to_string(),
            ))
        })?;
        #[cfg(feature = "probe-timing")]
        let probe_final_image_source_started = std::time::Instant::now();
        let (source, final_image) = if staged.has_rewritten_rows {
            let resolved_rows = surviving_rows
                .iter()
                .map(|(_, row)| {
                    row.values.clone().ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 rewritten final image lost its transient row values"
                                .to_string(),
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (decoded, encoded) =
                crate::typed_insert_batch::encode_final_table_image_from_resolved_rows(
                    table_ref,
                    table,
                    &resolved_rows,
                )
                .map_err(ExecuteError::Engine)?;
            let source = if surviving_rows.is_empty() {
                None
            } else {
                Some(
                    crate::typed_insert_batch::PreparedResidentAppendSource::
                        from_decoded_final_table_image(
                            decoded,
                            physical_table,
                            physical_schema_digest,
                            contribution.prepared_catalog_seq,
                        )
                        .map_err(ExecuteError::Engine)?,
                )
            };
            (source, encoded)
        } else {
            let decoded = contributions
                .iter()
                .map(|contribution| {
                    crate::typed_insert_batch::decode_typed_image(
                        contribution.codec5_sources.final_image(),
                    )
                    .map(|image| (image, contribution.codec5_sources.final_image_authority()))
                    .map_err(ExecuteError::Engine)
                })
                .collect::<Result<Vec<_>, ExecuteError>>()?;
            let (source, image) = crate::typed_insert_batch::PreparedResidentAppendSource::
                from_decoded_final_table_images_for_table_ref(
                    decoded,
                    table_ref,
                    physical_table,
                    physical_schema_digest,
                    contribution.prepared_catalog_seq,
                )
                .map_err(ExecuteError::Engine)?;
            (Some(source), image)
        };
        #[cfg(feature = "probe-timing")]
        engine.record_insert_probe_codec5_materialization_nanos([
            0,
            0,
            0,
            probe_final_image_source_started.elapsed().as_nanos() as u64,
        ]);
        let image_layout_digest: [u8; 32] = final_image
            .get(64..96)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "combined codec-5 final image lacks its layout digest".to_string(),
                ))
            })?;
        let image_content_digest = {
            use sha2::{Digest, Sha256};
            let domain = b"gpu-db/write001/s7-image-content/v2";
            let mut digest = Sha256::new();
            digest.update((domain.len() as u64).to_le_bytes());
            digest.update(domain);
            digest.update((final_image.len() as u64).to_le_bytes());
            digest.update(final_image.as_ref());
            digest.finalize().into()
        };
        let initial_table_absent =
            private_created_table && evolving_roots.table(table.stable_table_id).is_none();
        let predecessor = match evolving_roots.table(table.stable_table_id) {
            Some(predecessor) => predecessor,
            None if initial_table_absent => crate::engine_state::TypedTableGenerationRoot {
                data_generation: 0,
                table_root: [0; 32],
                logical_row_count: 0,
            },
            None => {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "generic codec-5 operation has no GPU-authenticated INSERT predecessor"
                        .to_string(),
                )));
            }
        };
        let predecessor_database_root = evolving_roots.database_root;
        let predecessor_index_roots = evolving_roots
            .table_index_roots(table.stable_table_id)
            .collect::<Vec<_>>();
        // An S3 DROP can retire a published device directory that was enrolled after the last
        // typed row-generation root.  Retire every such directory from the manifest, but evolve
        // only the predecessor roots that actually exist in this generation witness.
        let retired_predecessor_index_ids = retired_index_ids
            .iter()
            .filter(|retired| {
                predecessor_index_roots
                    .iter()
                    .any(|root| root.stable_index_id == **retired)
            })
            .copied()
            .collect::<Vec<_>>();
        let resets_existing_rows = staged.resets_table(&table.name);
        if initial_table_absent && resets_existing_rows {
            return Err(ExecuteError::Engine(EngineError::Durability(
                "codec-5 transaction-created table cannot reset an absent GPU generation"
                    .to_string(),
            )));
        }
        let runtime_indexes = indexed_generation_inputs(
            table,
            &predecessor_index_roots,
            surviving_row_count,
            resets_existing_rows,
            initial_table_absent,
            &created_index_ids,
            &retired_predecessor_index_ids,
        )?;
        let table_map_predecessor = match predecessor_database_root {
            Some(database_root) => {
                evolving_roots
                    .retained_table_map_predecessor(table.stable_table_id, database_root)
                    .map_err(ExecuteError::Engine)?
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "generic codec-5 operation has no retained table-map witness"
                                .to_string(),
                        ))
                    })?
            }
            None if initial_table_absent => {
                gpu_db_execution::RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase
            }
            None => {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "generic codec-5 operation has no GPU-authenticated database predecessor"
                        .to_string(),
                )));
            }
        };
        let final_logical_row_count = if resets_existing_rows {
            u64::from(surviving_row_count)
        } else {
            predecessor
                .logical_row_count
                .checked_add(u64::from(surviving_row_count))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "generic codec-5 logical row count overflows".to_string(),
                    ))
                })?
        };
        let row_sources = surviving_rows
            .iter()
            .zip(survivor_row_ids.iter().copied())
            .map(
                |((_, row), stable_row_id)| TypedInsertRuntimeGenerationRowSource {
                    stable_row_id,
                    statement_ordinal: row.statement_ordinal,
                    source_row_ordinal: row.source_row_ordinal,
                },
            )
            .collect::<Vec<_>>();
        let final_writers = selected_rows
            .iter()
            .map(|(provisional, row)| {
                Ok(crate::typed_insert_aggregate::LiveTypedInsertFinalWriter {
                    stable_row_id: final_identity_by_provisional
                        .resolve(&table.name, **provisional)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::Durability(
                                "codec-5 disposition lost its final row identity".to_string(),
                            ))
                        })?,
                    source_statement_ordinal: row.statement_ordinal,
                    source_row_ordinal: row.source_row_ordinal,
                    final_writer_statement_ordinal: row.final_writer_statement_ordinal,
                    final_writer_statement_digest: row.final_writer_statement_digest,
                    survives: row.survives,
                })
            })
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let generation = if let Some(source) = source.as_ref() {
            Some(engine
            .run_typed_insert_runtime_generation(TypedInsertRuntimeGenerationInput {
                source,
                row_allocator_before: first_row_id,
                first_row_id,
                row_sources: &row_sources,
                database_id: canonical_identity.database_id,
                catalog_epoch,
                catalog_digest,
                stable_transaction_id: txn_id,
                commit_sequence: expected_commit_seq,
                typed_statement_digest: if staged.len() == 1 {
                    contribution.typed_statement_digest
                } else {
                    [0; 32]
                },
                action: if initial_table_absent {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateWithRowSet
                } else if resets_existing_rows {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert
                } else if !created_index_ids.is_empty() {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateIndexThenRowSetInsert
                } else {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::RowSetInsert
                },
                table_map_predecessor,
                stable_table_id: table.stable_table_id,
                write001_final_image_ref: table_ref,
                // The GPU derives CREATE roots at this commit sequence while the durable S7
                // closure still carries the true absent predecessor (generation/root zero).
                base_data_generation: if initial_table_absent {
                    expected_commit_seq
                } else {
                    predecessor.data_generation
                },
                base_table_root: predecessor.table_root,
                row_allocator_high_water: table_high_water,
                initial_logical_row_count: predecessor.logical_row_count,
                final_logical_row_count,
                image_layout_digest,
                image_content_digest,
                indexes: &runtime_indexes,
            })
            .map_err(ExecuteError::Engine)?)
        } else {
            None
        };
        let (successor_index_roots, data_generation_after, final_table_root, final_database_root) =
            if let Some(generation) = generation.as_ref() {
                if generation.initial_table_root != predecessor.table_root
                    || predecessor_database_root
                        .is_some_and(|root| generation.initial_database_root != root)
                {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "generic codec-5 CUDA generation predecessor commitment drifted"
                            .to_string(),
                    )));
                }
                let mut runtime_index_roots = vec![
            gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: 0,
                initial_generation: 0,
                initial_root: [0; 32],
                final_generation: 0,
                final_root: [0; 32],
            };
            runtime_indexes.len()
        ];
                generation
                    .logical_completion
                    .copy_index_generation_roots_into(&mut runtime_index_roots)
                    .map_err(|_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "codec-5 CUDA index-root cardinality drifted".to_string(),
                        ))
                    })?;
                let successor_index_roots = runtime_index_roots
                    .iter()
                    .zip(&runtime_indexes)
                    .map(|(runtime_root, runtime_index)| {
                        let descriptor = &runtime_index.descriptor;
                        if runtime_root.stable_index_id != descriptor.stable_index_id
                            || runtime_root.initial_generation != descriptor.base_generation
                            || runtime_root.initial_root != descriptor.base_root
                            || runtime_root.final_generation != expected_commit_seq
                            || runtime_root.final_root == [0; 32]
                            || runtime_root.final_root == runtime_root.initial_root
                        {
                            return Err(ExecuteError::Engine(EngineError::Durability(
                                "codec-5 CUDA named index-root commitment drifted".to_string(),
                            )));
                        }
                        Ok(crate::engine_state::TypedIndexGenerationRoot {
                            stable_index_id: runtime_root.stable_index_id,
                            index_generation: runtime_root.final_generation,
                            index_root: runtime_root.final_root,
                        })
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                let mut shape_roots = vec![[0; 32]; table.columns.len()];
                let mut column_roots = vec![[0; 32]; table.columns.len()];
                generation
                    .logical_completion
                    .copy_column_roots_into(&mut shape_roots, &mut column_roots)
                    .map_err(|_| {
                        ExecuteError::Engine(EngineError::Durability(
                            "generic codec-5 GPU column-root cardinality drifted".to_string(),
                        ))
                    })?;
                let successor_columns = table
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(ordinal, column)| {
                        Ok(crate::engine_state::TypedColumnGenerationRoot {
                            catalog_column_ordinal: u32::try_from(ordinal).map_err(|_| {
                                ExecuteError::Engine(EngineError::Durability(
                                    "generic codec-5 column ordinal exceeds u32".to_string(),
                                ))
                            })?,
                            stable_column_id: column.id,
                            attnum: column.attnum,
                            column_shape_root: shape_roots[ordinal],
                            column_root: column_roots[ordinal],
                        })
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                root_publication = Some(match root_publication {
            None => crate::engine_commit::LiveTypedGenerationRootPublication::
                from_exact_gpu_completed_table_map_predecessor_with_created_indexes(
                    Arc::clone(&original_roots),
                    table.stable_table_id,
                    (!initial_table_absent).then_some(predecessor),
                    resets_existing_rows,
                    predecessor_database_root,
                    crate::engine_state::TypedTableGenerationRoot {
                        data_generation: expected_commit_seq,
                        table_root: generation.final_table_root,
                        logical_row_count: final_logical_row_count,
                    },
                    &successor_columns,
                    &predecessor_index_roots,
                    &successor_index_roots,
                    generation.final_database_root,
                    &generation.table_map_completion,
                    &created_index_ids,
                    &retired_predecessor_index_ids,
                )
                .map_err(ExecuteError::Engine)?,
            Some(publication) => publication
                .then_exact_gpu_completed_table_map_predecessor_with_created_indexes(
                    table.stable_table_id,
                    (!initial_table_absent).then_some(predecessor),
                    resets_existing_rows,
                    crate::engine_state::TypedTableGenerationRoot {
                        data_generation: expected_commit_seq,
                        table_root: generation.final_table_root,
                        logical_row_count: final_logical_row_count,
                    },
                    &successor_columns,
                    &predecessor_index_roots,
                    &successor_index_roots,
                    generation.final_database_root,
                    &generation.table_map_completion,
                    &created_index_ids,
                    &retired_predecessor_index_ids,
                )
                .map_err(ExecuteError::Engine)?,
                });
                evolving_roots = root_publication
                    .as_ref()
                    .expect("root publication was just built")
                    .private_candidate();
                (
                    successor_index_roots,
                    expected_commit_seq,
                    generation.final_table_root,
                    generation.final_database_root,
                )
            } else {
                (
                    predecessor_index_roots.clone(),
                    predecessor.data_generation,
                    predecessor.table_root,
                    predecessor_database_root.expect("generation requires a derived database root"),
                )
            };
        prepared_tables.push(PreparedCodec5Table {
            table,
            table_schema_digest: final_table_schema_digest,
            staged: contributions,
            source,
            final_image,
            image_content_digest,
            final_writers,
            survivor_row_ids,
            resets_existing_rows,
            initial_table_absent,
            created_indexes_on_existing_table,
            retired_indexes_on_existing_table,
            generation,
            predecessor,
            predecessor_index_roots,
            successor_index_roots,
            data_generation_after,
            final_table_root,
            final_database_root,
            first_row_id,
            row_allocator_high_water: table_high_water,
            final_logical_row_count,
        });
        #[cfg(feature = "probe-timing")]
        if multi_table {
            eprintln!(
                "[probe] codec5_plural_prepare stage=table_generation_complete table={}",
                table.name,
            );
        }
    }
    if !final_row_ids.is_empty() {
        return Err(ExecuteError::Engine(EngineError::Durability(
            "codec-5 provisional identity set contains an unbound target table".to_string(),
        )));
    }
    #[cfg(feature = "probe-timing")]
    let probe_table_prepare_nanos = probe_table_prepare_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_started = std::time::Instant::now();
    let writer_index_generations = prepared_tables
        .iter()
        .map(|prepared| {
            prepared
                .table
                .indexes
                .iter()
                .map(|catalog| {
                    let stable_index_id = u64::from(catalog.oid);
                    let predecessor = prepared
                        .predecessor_index_roots
                        .iter()
                        .find(|root| root.stable_index_id == stable_index_id)
                        .copied()
                        .or_else(|| {
                            (prepared.initial_table_absent
                                || prepared
                                    .created_indexes_on_existing_table
                                    .binary_search_by_key(&stable_index_id, |index| {
                                        index.stable_index_id
                                    })
                                    .is_ok())
                            .then_some(crate::engine_state::TypedIndexGenerationRoot {
                                stable_index_id,
                                index_generation: 0,
                                index_root: [0; 32],
                            })
                        })
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::Durability(
                                "codec-5 writer index lacks its predecessor root".to_string(),
                            ))
                        })?;
                    let successor = prepared
                        .successor_index_roots
                        .iter()
                        .find(|root| root.stable_index_id == stable_index_id)
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::Durability(
                                "codec-5 writer index lacks its successor root".to_string(),
                            ))
                        })?;
                    Ok(
                        crate::typed_insert_aggregate::LiveTypedInsertIndexGeneration {
                            catalog,
                            base_generation: predecessor.index_generation,
                            base_root: predecessor.index_root,
                            final_generation: successor.index_generation,
                            final_root: successor.index_root,
                        },
                    )
                })
                .collect::<Result<Vec<_>, ExecuteError>>()
        })
        .collect::<Result<Vec<_>, ExecuteError>>()?;
    let foreign_index_generations = prepared_tables
        .iter()
        .map(|prepared| {
            foreign_index_generations(
                prepared.table,
                &prepared.staged[0].catalog_dependencies,
                // The FK proof binds the parent generation that the same transaction will
                // publish, not merely the BEGIN snapshot. `evolving_roots` is the one
                // immutable GPU candidate assembled by the normal table-map publication
                // path; it contains an untouched parent's predecessor and every mutated or
                // transaction-created parent's authenticated successor. Using original roots
                // here made a private parent/child transaction unrepresentable despite both
                // tables already having one DeviceInsertPlan and one final publication.
                &evolving_roots,
            )
        })
        .collect::<Result<Vec<_>, ExecuteError>>()?;
    let statement_views = prepared_tables
        .iter()
        .enumerate()
        .map(|(table_ref, prepared)| {
            prepared
                .staged
                .iter()
                .map(
                    |contribution| crate::typed_insert_aggregate::LiveTypedInsertStatementView {
                        statement_ordinal: contribution.statement_ordinal,
                        operation_ordinal: contribution.operation_ordinal,
                        table_ref: table_ref as u32,
                        table_schema_digest: contribution.table_schema_digest,
                        typed_statement_digest: contribution.typed_statement_digest,
                        record: contribution.codec5_sources.record(),
                        sealed_source: Some(&contribution.codec5_sources),
                    },
                )
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let catalog_operation_body = catalog_composition
        .as_ref()
        .map(|composition| composition.operation_body.as_ref());
    let composition_changes_catalog = catalog_composition
        .as_ref()
        .is_some_and(|composition| composition.changes_catalog);
    let catalog_after_epoch = if composition_changes_catalog {
        catalog_epoch.checked_add(1).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(
                "codec-5 catalog epoch overflow".to_string(),
            ))
        })?
    } else {
        catalog_epoch
    };
    let catalog_after_digest = if composition_changes_catalog {
        let body = catalog_operation_body.expect("catalog-changing composition owns S3 bytes");
        Engine::canonical_catalog_transition(
            catalog_digest,
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            body,
        )
    } else {
        catalog_digest
    };
    let identity = crate::typed_insert_aggregate::LiveTypedInsertIdentity {
        physical: gpu_db_wal::CanonicalPhysicalRange {
            log_epoch: 1,
            lane_id: 0,
            segment_id: expected_commit_seq,
            first_frame_ordinal: 0,
        },
        canonical: canonical_identity,
        leader_epoch,
        commit_sequence: expected_commit_seq,
        stable_transaction_id: txn_id,
        mode,
        request_digest,
        isolation,
        catalog_epoch,
        catalog_digest,
        catalog_after_epoch,
        catalog_after_digest,
        dependency_validation_floor: first.read_snapshot,
    };
    let writer_sequence_references = prepared_tables
        .iter()
        .map(|prepared| {
            published_sequence_references
                .iter()
                .filter(|reference| reference.table_oid == prepared.table.oid)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let writer_inputs = prepared_tables
        .iter()
        .enumerate()
        .map(
            |(table_ref, prepared)| crate::typed_insert_aggregate::LiveTypedInsertView {
                identity,
                table: crate::typed_insert_aggregate::LiveTypedInsertTableGeneration {
                    schema: &prepared.table.schema,
                    name: &prepared.table.name,
                    stable_table_id: prepared.table.stable_table_id,
                    display_oid: prepared.table.oid,
                    // S7's table block names the final composed catalog relation.  Individual
                    // S2 records retain their statement-time default-expression bindings, so a
                    // sequence rename may legitimately change those record digests while the
                    // ordered stable-OID witness binds them to this final table identity.
                    schema_digest: prepared.table_schema_digest,
                    image_content_digest: prepared.image_content_digest,
                    data_generation_before: prepared.predecessor.data_generation,
                    data_generation_after: prepared.data_generation_after,
                    row_allocator_before: prepared.first_row_id,
                    row_allocator_high_water: prepared.row_allocator_high_water,
                    initial_logical_row_count: prepared.predecessor.logical_row_count,
                    final_logical_row_count: prepared.final_logical_row_count,
                    initial_table_root: prepared.predecessor.table_root,
                    final_table_root: prepared.final_table_root,
                    initial_database_root: prepared
                        .generation
                        .as_ref()
                        .map_or(prepared.final_database_root, |generation| {
                            generation.initial_database_root
                        }),
                    final_database_root: prepared.final_database_root,
                    resets_existing_rows: prepared.resets_existing_rows,
                    initial_table_absent: prepared.initial_table_absent,
                    indexes: &writer_index_generations[table_ref],
                    created_indexes_on_existing_table: &prepared.created_indexes_on_existing_table,
                },
                statements: &statement_views[table_ref],
                final_image: &prepared.final_image,
                foreign_indexes: &foreign_index_generations[table_ref],
                published_sequence_references: &writer_sequence_references[table_ref],
                final_row_digests: prepared.generation.as_ref().map(|generation| {
                    &generation.logical_completion
                        as &dyn crate::typed_insert_aggregate::LiveFinalRowDigestSource
                }),
                final_writers: &prepared.final_writers,
                source_geometry: prepared
                    .generation
                    .as_ref()
                    .map(|generation| generation.source_geometry),
            },
        )
        .collect::<Vec<_>>();
    let closed = crate::typed_insert_aggregate::encode_live_typed_insert_transaction(
        &writer_inputs,
        &staged.terminal_sequence_restarts,
        catalog_operation_body,
        composition_changes_catalog,
    )
    .map_err(ExecuteError::Engine)?;
    #[cfg(feature = "probe-timing")]
    engine.record_insert_probe_codec5_live_encoder_nanos(closed.probe_timing_nanos());
    #[cfg(feature = "probe-timing")]
    if multi_table {
        eprintln!("[probe] codec5_plural_prepare stage=aggregate_closed");
    }
    let payload_authority = closed.packed_payload_authority();
    let record = closed.into_prepared_record();
    let proposed_range = crate::wal_binary::ProposedRowIdRange::new(final_base, row_count)
        .map_err(ExecuteError::Engine)?;
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_nanos = probe_aggregate_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_plan_compile_started = std::time::Instant::now();
    let mut compile_order = (0..prepared_tables.len())
        .filter(|index| prepared_tables[*index].source.is_some())
        .collect::<Vec<_>>();
    compile_order.sort_by_key(|index| {
        let prepared = &prepared_tables[*index];
        let requires_rollover = engine.transaction_terminal_typed_insert_requires_rollover(
            prepared
                .source
                .as_ref()
                .expect("source is retained through plan compilation"),
            prepared.resets_existing_rows,
        );
        (
            prepared.table.indexes.is_empty() || !requires_rollover,
            prepared.table.indexes.is_empty(),
        )
    });
    #[cfg(feature = "probe-timing")]
    let rollover_count = compile_order
        .iter()
        .filter(|index| {
            !prepared_tables[**index].initial_table_absent
                && engine.transaction_terminal_typed_insert_requires_rollover(
                    prepared_tables[**index]
                        .source
                        .as_ref()
                        .expect("source is retained through plan compilation"),
                    prepared_tables[**index].resets_existing_rows,
                )
        })
        .count();
    let mut remaining_indexed_rollovers = compile_order
        .iter()
        .filter(|index| {
            let prepared = &prepared_tables[**index];
            !prepared.initial_table_absent
                && !prepared.table.indexes.is_empty()
                && engine.transaction_terminal_typed_insert_requires_rollover(
                    prepared
                        .source
                        .as_ref()
                        .expect("source is retained through plan compilation"),
                    prepared.resets_existing_rows,
                )
        })
        .count();
    #[cfg(feature = "probe-timing")]
    if multi_table {
        eprintln!(
            "[probe] codec5_plural_prepare stage=plan_order rollover_count={rollover_count} order={compile_order:?}"
        );
    }
    let plan_count = compile_order.len();
    let mut plans = Vec::with_capacity(plan_count);
    let mut shared_gate = None;
    let mut shared_budget_guard = None;
    let mut manifest_predecessor = None;
    let mut prior_reserved_bytes = 0_u64;
    let mut remaining_indexed_tables = prepared_tables
        .iter()
        .filter(|prepared| prepared.source.is_some() && !prepared.table.indexes.is_empty())
        .count();
    for (position, table_index) in compile_order.into_iter().enumerate() {
        let prepared = &mut prepared_tables[table_index];
        let current_table_is_indexed = !prepared.table.indexes.is_empty();
        let current_is_indexed_rollover = !prepared.initial_table_absent
            && current_table_is_indexed
            && engine.transaction_terminal_typed_insert_requires_rollover(
                prepared
                    .source
                    .as_ref()
                    .expect("source is retained through plan compilation"),
                prepared.resets_existing_rows,
            );
        if current_table_is_indexed {
            remaining_indexed_tables = remaining_indexed_tables.saturating_sub(1);
        }
        let source = prepared
            .source
            .take()
            .expect("codec-5 source is consumed by exactly one device plan");
        let ids = std::mem::take(&mut prepared.survivor_row_ids);
        let mut plan = if prepared.initial_table_absent {
            let named_index_lifecycle = current_table_is_indexed.then(|| {
                indexed_lifecycle
                    .take()
                    .expect("validated transaction-created indexed codec-5 shape retains its lifecycle")
            });
            if let Some(gate) = shared_gate.take() {
                engine.compile_transaction_created_table_typed_insert_device_plan_with_gate(
                    prepared.table,
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    expected_commit_seq,
                    named_index_lifecycle,
                    gate,
                    shared_budget_guard.take(),
                    prior_reserved_bytes,
                )
            } else {
                if shared_budget_guard.is_some() || prior_reserved_bytes != 0 {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "plural codec-5 lost the shared reservation before a transaction-created table"
                            .to_string(),
                    )));
                }
                engine.compile_transaction_created_table_typed_insert_device_plan(
                    prepared.table,
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    expected_commit_seq,
                    named_index_lifecycle,
                )
            }
        } else if !prepared.table.indexes.is_empty() {
            let named_index_lifecycle = indexed_lifecycle
                .take()
                .expect("validated indexed codec-5 shape retains its lifecycle");
            let created_index_ids = prepared
                .created_indexes_on_existing_table
                .iter()
                .map(|created| created.stable_index_id)
                .collect::<Vec<_>>();
            let retired_index_ids = prepared.retired_indexes_on_existing_table.clone();
            let has_s3_index_transition =
                !created_index_ids.is_empty() || !retired_index_ids.is_empty();
            let public_table = if !has_s3_index_transition {
                None
            } else {
                Some(engine.catalog_snapshot().relational_catalog.get(&prepared.table.name).cloned().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 S3-created index lost its public predecessor table".to_string(),
                    ))
                })?)
            };
            if let Some(gate) = shared_gate.take() {
                if let Some(public_table) = public_table.as_ref() {
                    engine.compile_transaction_terminal_s3_created_index_typed_insert_device_plan_with_gate(
                        prepared.table,
                        public_table,
                        &created_index_ids,
                        &retired_index_ids,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        named_index_lifecycle,
                        expected_commit_seq,
                        gate,
                        shared_budget_guard.take(),
                        prior_reserved_bytes,
                        manifest_predecessor.take(),
                    )
                } else {
                engine.compile_transaction_terminal_indexed_typed_insert_device_plan_with_gate(
                    prepared.table,
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    named_index_lifecycle,
                    expected_commit_seq,
                    prepared.resets_existing_rows,
                    gate,
                    shared_budget_guard.take(),
                    prior_reserved_bytes,
                    manifest_predecessor.take(),
                )
                }
            } else {
                if let Some(public_table) = public_table.as_ref() {
                    engine.compile_transaction_terminal_s3_created_index_typed_insert_device_plan(
                        prepared.table,
                        public_table,
                        &created_index_ids,
                        &retired_index_ids,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        named_index_lifecycle,
                        expected_commit_seq,
                    )
                } else {
                engine.compile_transaction_terminal_indexed_typed_insert_device_plan(
                    prepared.table,
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    named_index_lifecycle,
                    expected_commit_seq,
                    prepared.resets_existing_rows,
                )
                }
            }
        } else if let Some(gate) = shared_gate.take() {
            engine.compile_transaction_terminal_typed_insert_device_plan_with_gate(
                source,
                crate::engine_residency::DeviceInsertRowIds::exact(ids),
                expected_commit_seq,
                prepared.resets_existing_rows,
                gate,
                shared_budget_guard.take(),
                prior_reserved_bytes,
            )
        } else {
            engine.compile_transaction_terminal_typed_insert_device_plan(
                source,
                crate::engine_residency::DeviceInsertRowIds::exact(ids),
                expected_commit_seq,
                prepared.resets_existing_rows,
            )
        }
        .map_err(|error| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "generic codec-5 device plan compilation failed: {error:?}"
            )))
        })?;
        if current_is_indexed_rollover {
            remaining_indexed_rollovers = remaining_indexed_rollovers.saturating_sub(1);
            if remaining_indexed_rollovers != 0 {
                manifest_predecessor = plan.indexed_rollover_manifest_successor_predecessor();
                if manifest_predecessor.is_none() {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "plural codec-5 rollover plan lost its prepared manifest successor"
                            .to_string(),
                    )));
                }
            }
        }
        if current_table_is_indexed && remaining_indexed_tables != 0 {
            indexed_lifecycle = plan.take_named_index_publication_guard();
            if indexed_lifecycle.is_none() {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "plural indexed codec-5 plan lost the shared named-index lifecycle".to_string(),
                )));
            }
        }
        #[cfg(feature = "probe-timing")]
        if multi_table {
            eprintln!(
                "[probe] codec5_plural_prepare stage=plan_compiled position={position} table={} budget_guard={}",
                prepared.table.name,
                plan.probe_retains_exclusive_budget_guard(),
            );
        }
        if plan_count > 1 && position + 1 != plan_count {
            shared_gate = plan.take_transaction_terminal_unindexed_device_apply_guard();
            if shared_gate.is_none() {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "plural codec-5 plan lost the shared mutation gate".to_string(),
                )));
            }
            if let Some((guard, reserved_bytes)) =
                plan.take_transaction_terminal_unindexed_budget_guard()
            {
                prior_reserved_bytes = prior_reserved_bytes
                    .checked_add(reserved_bytes)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "plural codec-5 reserved-byte charge overflows".to_string(),
                        ))
                    })?;
                shared_budget_guard = Some(guard);
            }
        }
        plans.push(plan);
    }
    #[cfg(feature = "probe-timing")]
    let probe_plan_compile_nanos = probe_plan_compile_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_finalize_started = std::time::Instant::now();
    let tables = prepared_tables
        .iter()
        .map(|prepared| prepared.table.name.clone())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    // Stable OID, rather than the statement-local spelling, owns a private sequence's final
    // state.  A validated S3 rename intentionally gives earlier and later defaults different
    // names while preserving this one identity; publish the final private-catalog name below.
    let mut private_sequences = BTreeMap::<u32, (i64, bool)>::new();
    for advance in staged
        .iter()
        .flat_map(|contribution| contribution.private_sequence_advances.iter())
    {
        private_sequences.insert(advance.sequence_oid, advance.next_state);
    }
    for restart in &staged.terminal_sequence_restarts {
        private_sequences.insert(restart.sequence_oid, (restart.last_value, false));
    }
    let private_sequence_publications = private_sequences
        .into_iter()
        .map(|(sequence_oid, (last_value, is_called))| -> Result<_, ExecuteError> {
            let name = transaction_catalog_snapshot
                .relational_sequences
                .iter()
                .find_map(|(name, sequence)| (sequence.oid == sequence_oid).then_some(name))
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "codec-5 private sequence publication lost its final stable-OID catalog binding"
                            .to_string(),
                    ))
                })?;
            Ok(crate::engine_commit::LiveTypedPrivateSequencePublication {
                name: name.clone().into_boxed_str(),
                sequence_oid,
                last_value,
                is_called,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        ?
        .into_boxed_slice();
    let generations = prepared_tables
        .into_iter()
        .filter_map(|prepared| prepared.generation)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    #[cfg(feature = "probe-timing")]
    engine.record_insert_probe_codec5_operation_nanos([
        probe_input_nanos,
        probe_table_prepare_nanos,
        probe_aggregate_nanos,
        probe_plan_compile_nanos,
        probe_finalize_started.elapsed().as_nanos() as u64,
    ]);
    Ok(PreparedCodec5Operation {
        record: Some(record),
        payload_authority,
        plans,
        root_publication,
        tables,
        proposed_range,
        allocator_high_water,
        affected_rows: u64::from(row_count),
        write_set: staged.write_set.clone(),
        private_sequence_publications,
        catalog_composition: catalog_composition.map(|composition| composition.record),
        generations,
    })
}
