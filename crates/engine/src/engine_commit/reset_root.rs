//! GPU typed-root replacement for an already-durable zero-row table replacement.
//!
//! TRUNCATE and catalog shape rewrites keep their historical records and apply behavior. This
//! module only closes the corresponding immutable typed-generation root at the existing commit
//! visibility cut; it owns no WAL encoder, terminal, allocator, row apply, recovery selector, or
//! publication tail.

use super::*;

impl Engine {
    pub(super) fn prepare_empty_typed_reset_root_publication(
        &self,
        commit: &mut CommitState,
        entry: &LogEntry,
        cat: &DdlCatalogState,
    ) -> Result<Option<LiveTypedGenerationRootPublication>, EngineError> {
        let Some(canonical) = commit
            .wal
            .last_record()
            .and_then(|record| {
                gpu_db_wal::decode_canonical_record_payload(&record.payload).transpose()
            })
            .transpose()?
        else {
            return Ok(None);
        };
        let carries_reset = canonical
            .fragments
            .iter()
            .any(|fragment| fragment.kind == gpu_db_wal::CanonicalFragmentKind::TableReset);
        let reset = if carries_reset {
            let crate::wal_binary::BinaryWalRecord::Transaction(record) =
                crate::wal_binary::decode_binary_record(&entry.payload)?
            else {
                return Ok(None);
            };
            if record.table_resets.len() != 1 || !record.mutations.is_empty() {
                return Ok(None);
            }
            Some(
                record
                    .table_resets
                    .into_iter()
                    .next()
                    .expect("one checked reset"),
            )
        } else {
            None
        };
        let catalog_rewrite_table = if reset.is_none() {
            match Self::decode_engine_command(&entry.payload)? {
                Some(Command::AddColumn(add)) => Some(add.table),
                Some(Command::DropColumn(drop)) => Some(drop.table),
                Some(Command::DropConstraint(drop)) => Some(drop.table),
                Some(Command::CreateIndex(create)) => Some(create.table),
                Some(Command::AddPrimaryKey(add)) => Some(add.table),
                Some(Command::AddUniqueConstraint(add)) => Some(add.table),
                _ => None,
            }
        } else {
            None
        };
        let Some(table_name) = reset
            .as_ref()
            .map(|reset| reset.table.as_str())
            .or(catalog_rewrite_table.as_deref())
        else {
            return Ok(None);
        };
        let table = cat.relational_catalog.get(table_name).ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed table replacement lost its catalog table before root publication"
                    .to_string(),
            )
        })?;
        if table.columns.is_empty()
            || reset
                .as_ref()
                .is_some_and(|reset| table.oid != reset.table_oid)
        {
            return Err(EngineError::ApplyFailed(
                "typed table replacement catalog identity differs from its durable operation"
                    .to_string(),
            ));
        }
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        if reset
            .as_ref()
            .is_some_and(|reset| schema_digest != reset.schema_digest)
            || table.indexes.iter().any(|index| {
                (index.primary_key && !index.unique)
                    || (index.unique_constraint && !index.unique)
                    || !crate::engine_residency::index_all_key_columns_foldable(table, index)
            })
        {
            return Err(EngineError::ApplyFailed(
                "typed table reset post-catalog shape differs from its durable identity"
                    .to_string(),
            ));
        }
        if catalog_rewrite_table.is_some() {
            let rows = self.visible_relational_rows(
                table,
                StorageVisibility {
                    read_txn_id: entry.index,
                },
            )?;
            if !rows.is_empty() {
                return Ok(None);
            }
        }
        if canonical.header.commit_seq != entry.index
            || canonical.header.catalog_after_epoch == 0
            || canonical.header.catalog_after_digest == [0; 32]
            || canonical.header.identity != commit.canonical_identity
        {
            return Err(EngineError::ApplyFailed(
                "typed table reset canonical identity differs from its committed entry".to_string(),
            ));
        }

        let predecessor = self.read_state.typed_generation_roots.load_full();
        let Some(expected_table) = predecessor.table(table.stable_table_id) else {
            // A pre-typed historical CREATE/INSERT prefix has no root to replace. Do not invent
            // one from its host shadow during recovery.
            return Ok(None);
        };
        let expected_database_root = predecessor.database_root.ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed table reset has no GPU-authenticated database predecessor".to_string(),
            )
        })?;
        let expected_index_roots = predecessor
            .table_index_roots(table.stable_table_id)
            .collect::<Vec<_>>();
        let table_map_predecessor = predecessor
            .retained_table_map_predecessor(table.stable_table_id, expected_database_root)?
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed table reset database root has no retained table-map witness".to_string(),
                )
            })?;

        let mut index_keys = Vec::with_capacity(table.indexes.len());
        for index in &table.indexes {
            let mut keys = Vec::with_capacity(index.key_columns.len());
            for (key_ordinal, key_name) in index.key_columns.iter().enumerate() {
                let (catalog_column_ordinal, column) = table
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, column)| column.name == *key_name)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "typed table reset index key is absent from its catalog table"
                                .to_string(),
                        )
                    })?;
                keys.push(
                    gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                        key_ordinal: u32::try_from(key_ordinal).map_err(|_| {
                            EngineError::ApplyFailed(
                                "typed table reset index key ordinal exceeds u32".to_string(),
                            )
                        })?,
                        catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(
                            |_| {
                                EngineError::ApplyFailed(
                                    "typed table reset index catalog ordinal exceeds u32"
                                        .to_string(),
                                )
                            },
                        )?,
                        stable_column_id: column.id,
                        attnum: column.attnum,
                        storage: crate::typed_insert_batch::typed_image_sql_storage(column.ty),
                        declared_type_oid: column.type_oid,
                        signed_type_size: column.type_size,
                        column_name_digest:
                            crate::typed_insert_aggregate::write001_identifier_digest(
                                &column.name,
                            )?,
                    },
                );
            }
            index_keys.push(keys);
        }

        let target = self
            .cuda_driver_probe_runtime()
            .runtime_typed_insert_generation_target(0)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed table reset CUDA target declined: {error:?}"
                ))
            })?;
        let attempt = gpu_db_execution::RuntimeTypedInsertGenerationAttempt::new(entry.index)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed table reset generation attempt is invalid: {error:?}"
                ))
            })?;
        let prepared = gpu_db_execution::PreparedRuntimeTypedInsertGeneration::reserve(
            target,
            attempt,
            gpu_db_execution::RuntimeTypedInsertGenerationGeometry {
                rows: 0,
                cells: table.columns.len(),
                value_bytes: 0,
                indexes: table.indexes.len(),
                index_keys: index_keys.iter().map(Vec::len).sum(),
                index_effects: 0,
                index_effect_components: 0,
            },
        )
        .map_err(|error| {
            EngineError::ApplyFailed(format!(
                "typed table reset generation reservation declined: {error:?}"
            ))
        })?;
        let allocator_frontier = self.read_state.mvcc.current_row_id();
        let runtime_table = gpu_db_execution::RuntimeTypedInsertGenerationTable {
            action:
                gpu_db_execution::RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert,
            table_map_predecessor,
            stable_table_id: table.stable_table_id,
            write001_final_image_ref: 0,
            base_data_generation: expected_table.data_generation,
            base_table_root: expected_table.table_root,
            row_allocator_before: allocator_frontier,
            row_allocator_high_water: allocator_frontier,
            initial_logical_row_count: expected_table.logical_row_count,
            final_logical_row_count: 0,
            image_layout_digest: schema_digest,
            image_content_digest: [0; 32],
        };
        let identity = gpu_db_execution::RuntimeTypedInsertGenerationIdentity {
            database_id: canonical.header.identity.database_id,
            catalog_epoch: canonical.header.catalog_after_epoch,
            catalog_digest: canonical.header.catalog_after_digest,
            stable_transaction_id: canonical.header.stable_transaction_id,
            commit_sequence: entry.index,
            write001_typed_statement_digest: [0; 32],
        };
        let completion = prepared
            .launch(|encoder| {
                encoder.write_identity(identity);
                encoder.write_table(runtime_table);
                for (ordinal, column) in table.columns.iter().enumerate() {
                    encoder.write_cell(gpu_db_execution::RuntimeTypedInsertGenerationCell {
                        catalog_column_ordinal: u32::try_from(ordinal)
                            .expect("catalog column ordinal fits u32"),
                        stable_column_id: column.id,
                        attnum: column.attnum,
                        storage: crate::typed_insert_batch::typed_image_sql_storage(column.ty),
                        declared_type_oid: column.type_oid,
                        signed_type_size: column.type_size,
                        is_null: true,
                        value: &[],
                    });
                }
                let mut key_start = 0_u32;
                for (raw_ordinal, (index, keys)) in
                    table.indexes.iter().zip(&index_keys).enumerate()
                {
                    let predecessor = expected_index_roots
                        .iter()
                        .find(|root| root.stable_index_id == u64::from(index.oid));
                    let key_count = u32::try_from(keys.len())
                        .expect("validated reset index key count fits u32");
                    encoder.write_index(gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                        stable_index_id: u64::from(index.oid),
                        raw_catalog_index_ordinal: u32::try_from(raw_ordinal)
                            .expect("validated reset index ordinal fits u32"),
                        index_flags: u32::from(index.unique)
                            | (u32::from(index.primary_key) << 1)
                            | (u32::from(index.unique_constraint) << 2)
                            | (1 << 3),
                        null_equality_policy: 1,
                        base_generation: predecessor.map_or(0, |root| root.index_generation),
                        base_root: predecessor.map_or([0; 32], |root| root.index_root),
                        key_start,
                        key_count,
                        effect_start: 0,
                        effect_count: 0,
                    });
                    for key in keys.iter().copied() {
                        encoder.write_index_key(key);
                    }
                    key_start = key_start
                        .checked_add(key_count)
                        .expect("validated reset index key range fits u32");
                }
            })
            .complete();
        let proof = match completion {
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
                return Err(EngineError::ApplyFailed(format!(
                    "typed table reset generation failed: {error}"
                )));
            }
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(
                unknown,
            ) => {
                let error = unknown.error().to_string();
                drop(unknown);
                return Err(EngineError::ApplyFailed(format!(
                    "typed table reset generation quiescence is unproven: {error}"
                )));
            }
        };
        let mut initial_table_root = [0; 32];
        let mut final_table_root = [0; 32];
        let mut initial_database_root = [0; 32];
        let mut final_database_root = [0; 32];
        let (commitments, logical) = proof.consume(|_attempt, commitments, logical| {
            commitments.copy_initial_table_root_into(&mut initial_table_root);
            commitments.copy_final_table_root_into(&mut final_table_root);
            commitments.copy_initial_database_root_into(&mut initial_database_root);
            commitments.copy_final_database_root_into(&mut final_database_root);
            (commitments, logical)
        });
        if initial_table_root != expected_table.table_root
            || final_table_root == [0; 32]
            || final_table_root == initial_table_root
            || initial_database_root != expected_database_root
            || final_database_root == [0; 32]
            || final_database_root == initial_database_root
        {
            return Err(EngineError::ApplyFailed(
                "typed table reset returned inconsistent GPU roots".to_string(),
            ));
        }
        let mut shapes = vec![[0; 32]; table.columns.len()];
        let mut column_roots = vec![[0; 32]; table.columns.len()];
        logical
            .copy_column_roots_into(&mut shapes, &mut column_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed table reset GPU column-root cardinality drifted".to_string(),
                )
            })?;
        let columns = table
            .columns
            .iter()
            .enumerate()
            .map(|(ordinal, column)| TypedColumnGenerationRoot {
                catalog_column_ordinal: u32::try_from(ordinal)
                    .expect("catalog column ordinal fits u32"),
                stable_column_id: column.id,
                attnum: column.attnum,
                column_shape_root: shapes[ordinal],
                column_root: column_roots[ordinal],
            })
            .collect::<Vec<_>>();
        let mut runtime_roots = vec![
            gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: 0,
                initial_generation: 0,
                initial_root: [0; 32],
                final_generation: 0,
                final_root: [0; 32],
            };
            table.indexes.len()
        ];
        logical
            .copy_index_generation_roots_into(&mut runtime_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed table reset GPU index-root cardinality drifted".to_string(),
                )
            })?;
        let successor_index_roots = table
            .indexes
            .iter()
            .zip(runtime_roots)
            .map(|(index, root)| {
                let predecessor = expected_index_roots
                    .iter()
                    .find(|before| before.stable_index_id == u64::from(index.oid));
                if root.stable_index_id != u64::from(index.oid)
                    || root.initial_generation != predecessor.map_or(0, |r| r.index_generation)
                    || root.initial_root != predecessor.map_or([0; 32], |r| r.index_root)
                    || root.final_generation != entry.index
                    || root.final_root == [0; 32]
                {
                    return Err(EngineError::ApplyFailed(
                        "typed table reset GPU index root is invalid".to_string(),
                    ));
                }
                Ok(TypedIndexGenerationRoot {
                    stable_index_id: root.stable_index_id,
                    index_generation: root.final_generation,
                    index_root: root.final_root,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let table_map_completion = TypedTableMapGpuCompletion::from_logical_completion(&logical)?;
        let _generation_authority = commitments;
        LiveTypedGenerationRootPublication::from_exact_gpu_completed_table_map_predecessor(
            predecessor,
            table.stable_table_id,
            Some(expected_table),
            true,
            Some(expected_database_root),
            TypedTableGenerationRoot {
                data_generation: entry.index,
                table_root: final_table_root,
                logical_row_count: 0,
            },
            &columns,
            &expected_index_roots,
            &successor_index_roots,
            final_database_root,
            &table_map_completion,
        )
        .map(Some)
    }
}

impl Engine {
    /// Re-derive the one typed table leaf after an already-durable catalog operation rewrote a
    /// populated table's physical column/index shape. The repair scan is the existing DDL
    /// bootstrap boundary; values move immediately into the shared typed image and generic CUDA
    /// generation, and the ordinary root holder performs the sole publication.
    pub(super) fn prepare_populated_typed_catalog_rewrite_root_publication(
        &self,
        commit: &mut CommitState,
        entry: &LogEntry,
        cat: &DdlCatalogState,
    ) -> Result<Option<LiveTypedGenerationRootPublication>, EngineError> {
        let table_name = match Self::decode_engine_command(&entry.payload)? {
            Some(Command::AddColumn(add)) => add.table,
            Some(Command::DropColumn(drop)) => drop.table,
            Some(Command::DropConstraint(drop)) => drop.table,
            Some(Command::CreateIndex(create)) => create.table,
            Some(Command::AddPrimaryKey(add)) => add.table,
            Some(Command::AddUniqueConstraint(add)) => add.table,
            _ => return Ok(None),
        };
        let table = cat.relational_catalog.get(&table_name).ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed catalog rewrite lost its table before root publication".to_string(),
            )
        })?;
        if table.columns.is_empty()
            || table.indexes.iter().any(|index| {
                (index.primary_key && !index.unique)
                    || (index.unique_constraint && !index.unique)
                    || !crate::engine_residency::index_all_key_columns_foldable(table, index)
            })
        {
            return Ok(None);
        }
        let Some(canonical) = commit
            .wal
            .last_record()
            .and_then(|record| {
                gpu_db_wal::decode_canonical_record_payload(&record.payload).transpose()
            })
            .transpose()?
        else {
            return Ok(None);
        };
        if canonical.header.commit_seq != entry.index
            || canonical.header.catalog_after_epoch == 0
            || canonical.header.catalog_after_digest == [0; 32]
            || canonical.header.identity != commit.canonical_identity
        {
            return Err(EngineError::ApplyFailed(
                "typed catalog rewrite canonical identity differs from its committed entry"
                    .to_string(),
            ));
        }

        let predecessor = self.read_state.typed_generation_roots.load_full();
        let Some(expected_table) = predecessor.table(table.stable_table_id) else {
            return Ok(None);
        };
        let expected_database_root = predecessor.database_root.ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed catalog rewrite has no GPU-authenticated database predecessor".to_string(),
            )
        })?;
        let expected_columns = predecessor
            .table_columns(table.stable_table_id)
            .collect::<Vec<_>>();
        let expected_index_roots = predecessor
            .table_index_roots(table.stable_table_id)
            .collect::<Vec<_>>();
        let column_identity_changed = expected_columns.len() != table.columns.len()
            || expected_columns
                .iter()
                .zip(&table.columns)
                .any(|(root, column)| {
                    root.stable_column_id != column.id || root.attnum != column.attnum
                });
        let index_identity_changed = expected_index_roots.len() != table.indexes.len()
            || expected_index_roots
                .iter()
                .zip(&table.indexes)
                .any(|(root, index)| root.stable_index_id != u64::from(index.oid));
        if !column_identity_changed && !index_identity_changed {
            return Ok(None);
        }

        let prefix = relational_key_prefix(&table.name);
        let visibility = StorageVisibility {
            read_txn_id: entry.index,
        };
        let table_rows = self.read_state.mvcc.table_rows(&table.name);
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let mut resolved = Vec::<(u64, Vec<SqlValue>)>::new();
        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let row_id = crate::engine_residency::parse_relational_row_id(&tuple.key, &prefix)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "typed catalog rewrite row has an invalid stable identity".to_string(),
                    )
                })?;
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
            resolved.push((row_id, row));
        }
        resolved.sort_unstable_by_key(|(row_id, _)| *row_id);
        if resolved.is_empty() {
            return Ok(None);
        }
        if resolved.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(EngineError::ApplyFailed(
                "typed catalog rewrite row identities differ from the published predecessor"
                    .to_string(),
            ));
        }
        let row_ids = resolved
            .iter()
            .map(|(row_id, _)| *row_id)
            .collect::<Vec<_>>();
        let rows = resolved.into_iter().map(|(_, row)| row).collect::<Vec<_>>();
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let (decoded, final_image) =
            crate::typed_insert_batch::encode_final_table_image_from_resolved_rows(
                0, table, &rows,
            )?;
        let image_layout_digest = final_image
            .get(64..96)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed catalog rewrite final image lacks its layout digest".to_string(),
                )
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
        let source =
            crate::typed_insert_batch::PreparedResidentAppendSource::from_decoded_final_table_image(
                decoded,
                table,
                schema_digest,
                entry.index,
            )?;
        let row_sources = row_ids
            .iter()
            .enumerate()
            .map(|(ordinal, row_id)| {
                Ok(
                    crate::engine_transaction_delta::TypedInsertRuntimeGenerationRowSource {
                        stable_row_id: *row_id,
                        statement_ordinal: 0,
                        source_row_ordinal: u32::try_from(ordinal).map_err(|_| {
                            EngineError::ApplyFailed(
                                "typed catalog rewrite row ordinal exceeds u32".to_string(),
                            )
                        })?,
                    },
                )
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let row_count = u32::try_from(row_ids.len()).map_err(|_| {
            EngineError::ApplyFailed(
                "typed catalog rewrite row count exceeds runtime geometry".to_string(),
            )
        })?;
        let runtime_indexes = crate::engine_transaction_delta::indexed_generation_inputs(
            table,
            &expected_index_roots,
            row_count,
            true,
            false,
            &[],
            &[],
        )
        .map_err(|error| match error {
            ExecuteError::Engine(error) => error,
            other => EngineError::ApplyFailed(other.to_string()),
        })?;
        let table_map_predecessor = predecessor
            .retained_table_map_predecessor(table.stable_table_id, expected_database_root)?
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed catalog rewrite database root has no retained table-map witness"
                        .to_string(),
                )
            })?;
        let allocator_frontier = self.read_state.mvcc.current_row_id();
        let generation = self.run_typed_insert_runtime_generation(
            crate::engine_transaction_delta::TypedInsertRuntimeGenerationInput {
                source: &source,
                row_allocator_before: allocator_frontier,
                first_row_id: row_ids[0],
                row_sources: &row_sources,
                database_id: canonical.header.identity.database_id,
                catalog_epoch: canonical.header.catalog_after_epoch,
                catalog_digest: canonical.header.catalog_after_digest,
                stable_transaction_id: canonical.header.stable_transaction_id,
                commit_sequence: entry.index,
                typed_statement_digest: [0; 32],
                action:
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert,
                table_map_predecessor,
                stable_table_id: table.stable_table_id,
                write001_final_image_ref: 0,
                base_data_generation: expected_table.data_generation,
                base_table_root: expected_table.table_root,
                row_allocator_high_water: allocator_frontier,
                initial_logical_row_count: expected_table.logical_row_count,
                final_logical_row_count: row_ids.len() as u64,
                image_layout_digest,
                image_content_digest,
                indexes: &runtime_indexes,
            },
        )?;
        if generation.initial_table_root != expected_table.table_root
            || generation.final_table_root == [0; 32]
            || generation.final_table_root == generation.initial_table_root
            || generation.initial_database_root != expected_database_root
            || generation.final_database_root == [0; 32]
            || generation.final_database_root == generation.initial_database_root
        {
            return Err(EngineError::ApplyFailed(
                "typed catalog rewrite returned inconsistent GPU roots".to_string(),
            ));
        }
        let mut shape_roots = vec![[0; 32]; table.columns.len()];
        let mut column_roots = vec![[0; 32]; table.columns.len()];
        generation
            .logical_completion
            .copy_column_roots_into(&mut shape_roots, &mut column_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed catalog rewrite GPU column-root cardinality drifted".to_string(),
                )
            })?;
        let successor_columns = table
            .columns
            .iter()
            .enumerate()
            .map(|(ordinal, column)| TypedColumnGenerationRoot {
                catalog_column_ordinal: u32::try_from(ordinal)
                    .expect("catalog column ordinal fits u32"),
                stable_column_id: column.id,
                attnum: column.attnum,
                column_shape_root: shape_roots[ordinal],
                column_root: column_roots[ordinal],
            })
            .collect::<Vec<_>>();
        let mut generated_index_roots = vec![
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
            .copy_index_generation_roots_into(&mut generated_index_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed catalog rewrite GPU index-root cardinality drifted".to_string(),
                )
            })?;
        let successor_index_roots = generated_index_roots
            .into_iter()
            .zip(&runtime_indexes)
            .map(|(root, input)| {
                if root.stable_index_id != input.descriptor.stable_index_id
                    || root.initial_generation != input.descriptor.base_generation
                    || root.initial_root != input.descriptor.base_root
                    || root.final_generation != entry.index
                    || root.final_root == [0; 32]
                {
                    return Err(EngineError::ApplyFailed(
                        "typed catalog rewrite GPU index root is invalid".to_string(),
                    ));
                }
                Ok(TypedIndexGenerationRoot {
                    stable_index_id: root.stable_index_id,
                    index_generation: root.final_generation,
                    index_root: root.final_root,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let table_map_completion = generation.table_map_completion;
        let final_table_root = generation.final_table_root;
        let final_database_root = generation.final_database_root;
        let _generation_authority = (generation.commitments, generation.logical_completion);
        LiveTypedGenerationRootPublication::from_exact_gpu_completed_table_map_predecessor(
            predecessor,
            table.stable_table_id,
            Some(expected_table),
            true,
            Some(expected_database_root),
            TypedTableGenerationRoot {
                data_generation: entry.index,
                table_root: final_table_root,
                logical_row_count: row_ids.len() as u64,
            },
            &successor_columns,
            &expected_index_roots,
            &successor_index_roots,
            final_database_root,
            &table_map_completion,
        )
        .map(Some)
    }
}
