//! GPU-root enrollment for an empty foldable `CREATE INDEX` predecessor.
//!
//! This module owns no WAL, catalog mutation, allocator, physical append, or publication tail.
//! It prepares the one immutable typed-root successor after ordinary catalog application has
//! made the index identity visible, and the existing commit loop installs that candidate at its
//! normal single visibility cut.

use super::*;

impl Engine {
    pub(super) fn prepare_empty_typed_index_enrollment_root_publication(
        &self,
        commit: &mut CommitState,
        entry: &LogEntry,
        cat: &DdlCatalogState,
    ) -> Result<Option<LiveTypedGenerationRootPublication>, EngineError> {
        let (table_name, index_name) = match Self::decode_engine_command(&entry.payload)? {
            Some(Command::CreateIndex(create)) => (create.table, create.name),
            Some(Command::AddPrimaryKey(add)) => (add.table, add.name),
            Some(Command::AddUniqueConstraint(add)) => (add.table, add.name),
            _ => return Ok(None),
        };
        let table = cat.relational_catalog.get(&table_name).ok_or_else(|| {
            EngineError::ApplyFailed(
                "index-creating DDL did not publish its owner table before typed root enrollment"
                    .to_string(),
            )
        })?;
        let Some(index) = table.indexes.iter().find(|index| index.name == index_name) else {
            return Err(EngineError::ApplyFailed(
                "index-creating DDL catalog result lacks its declared index".to_string(),
            ));
        };
        // Empty-table CREATE INDEX appends one GPU-authenticated index root to the existing table
        // leaf. Every already-enrolled index remains an exact predecessor; index cardinality is
        // not a reason to replace the typed root authority with catalog-only publication.
        if index.oid == 0
            || table.columns.is_empty()
            || table.indexes.is_empty()
            || table.indexes.iter().any(|candidate| {
                (candidate.primary_key && !candidate.unique)
                    || (candidate.unique_constraint && !candidate.unique)
                    || !crate::engine_residency::index_all_key_columns_foldable(table, candidate)
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
            // Historical index DDL retains its frozen decoder/recovery policy. Establish that
            // this entry belongs to the current canonical envelope before consulting the new
            // typed-root map: an old replay prefix is not required to have that later authority.
            return Ok(None);
        };
        if canonical.header.commit_seq != entry.index
            || canonical.header.catalog_after_epoch == 0
            || canonical.header.catalog_after_digest == [0; 32]
            || canonical.header.identity != commit.canonical_identity
        {
            return Err(EngineError::ApplyFailed(
                "typed CREATE INDEX canonical identity differs from its committed entry"
                    .to_string(),
            ));
        }
        let mut keys = Vec::with_capacity(index.key_columns.len());
        for (ordinal, name) in index.key_columns.iter().enumerate() {
            let Some((catalog_column_ordinal, column)) = table
                .columns
                .iter()
                .enumerate()
                .find(|(_, column)| column.name == *name)
            else {
                return Err(EngineError::ApplyFailed(
                    "CREATE INDEX key is absent from the catalog owner table".to_string(),
                ));
            };
            keys.push(
                gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                    key_ordinal: u32::try_from(ordinal).map_err(|_| {
                        EngineError::ApplyFailed("CREATE INDEX key ordinal exceeds u32".to_string())
                    })?,
                    catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(
                        |_| {
                            EngineError::ApplyFailed(
                                "CREATE INDEX key catalog ordinal exceeds u32".to_string(),
                            )
                        },
                    )?,
                    stable_column_id: column.id,
                    attnum: column.attnum,
                    storage: crate::typed_insert_batch::typed_image_sql_storage(column.ty),
                    declared_type_oid: column.type_oid,
                    signed_type_size: column.type_size,
                    column_name_digest: crate::typed_insert_aggregate::write001_identifier_digest(
                        &column.name,
                    )?,
                },
            );
        }

        let predecessor = self.read_state.typed_generation_roots.load_full();
        let Some(expected_table) = predecessor.table(table.stable_table_id) else {
            // A historical WAL prefix predates typed generation-root publication.  Retain that
            // decoder/DDL compatibility boundary, but do not synthesize a codec-5 predecessor
            // from host catalog state: that would create a second recovery authority.  A later
            // INSERT remains fail-closed unless a real GPU-authenticated predecessor exists.
            return Ok(None);
        };
        let expected_database_root = predecessor.database_root.ok_or_else(|| {
            EngineError::ApplyFailed(
                "typed CREATE INDEX has no GPU-authenticated database predecessor".to_string(),
            )
        })?;
        let expected_predecessor_index_roots = predecessor
            .table_index_roots(table.stable_table_id)
            .collect::<Vec<_>>();
        let raw_catalog_index_ordinal = table
            .indexes
            .iter()
            .position(|candidate| candidate.oid == index.oid)
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "CREATE INDEX lost its catalog ordinal before typed root enrollment"
                        .to_string(),
                )
            })?;
        if expected_table.logical_row_count != 0
            || expected_predecessor_index_roots.len() != raw_catalog_index_ordinal
            || table.indexes[..raw_catalog_index_ordinal]
                .iter()
                .zip(&expected_predecessor_index_roots)
                .any(|(catalog, root)| u64::from(catalog.oid) != root.stable_index_id)
        {
            return Ok(None);
        }
        let table_map_predecessor = predecessor
            .retained_table_map_predecessor(table.stable_table_id, expected_database_root)?
            .ok_or_else(|| {
                EngineError::ApplyFailed(
                    "typed CREATE INDEX database root has no retained table-map witness"
                        .to_string(),
                )
            })?;
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let target = self
            .cuda_driver_probe_runtime()
            .runtime_typed_insert_generation_target(0)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed CREATE INDEX CUDA target declined: {error:?}"
                ))
            })?;
        let attempt = gpu_db_execution::RuntimeTypedInsertGenerationAttempt::new(entry.index)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed CREATE INDEX generation attempt is invalid: {error:?}"
                ))
            })?;
        let prepared = gpu_db_execution::PreparedRuntimeTypedInsertGeneration::reserve(
            target,
            attempt,
            gpu_db_execution::RuntimeTypedInsertGenerationGeometry {
                rows: 0,
                cells: table.columns.len(),
                value_bytes: 0,
                indexes: 1,
                index_keys: keys.len(),
                index_effects: 0,
                index_effect_components: 0,
            },
        )
        .map_err(|error| {
            EngineError::ApplyFailed(format!(
                "typed CREATE INDEX generation reservation declined: {error:?}"
            ))
        })?;
        let allocator_frontier = self.read_state.mvcc.current_row_id();
        let runtime_table = gpu_db_execution::RuntimeTypedInsertGenerationTable {
            action: gpu_db_execution::RuntimeTypedInsertGenerationTableAction::EnrollIndex,
            table_map_predecessor,
            stable_table_id: table.stable_table_id,
            write001_final_image_ref: 0,
            base_data_generation: expected_table.data_generation,
            base_table_root: expected_table.table_root,
            row_allocator_before: allocator_frontier,
            row_allocator_high_water: allocator_frontier,
            initial_logical_row_count: 0,
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
        let key_count = u32::try_from(keys.len()).map_err(|_| {
            EngineError::ApplyFailed("typed CREATE INDEX key cardinality exceeds u32".to_string())
        })?;
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
                encoder.write_index(gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                    stable_index_id: u64::from(index.oid),
                    raw_catalog_index_ordinal: u32::try_from(raw_catalog_index_ordinal)
                        .expect("catalog index ordinal fits u32"),
                    index_flags: u32::from(index.unique)
                        | (u32::from(index.primary_key) << 1)
                        | (u32::from(index.unique_constraint) << 2)
                        | (1 << 3),
                    null_equality_policy: 1,
                    base_generation: 0,
                    base_root: [0; 32],
                    key_start: 0,
                    key_count,
                    effect_start: 0,
                    effect_count: 0,
                });
                for key in keys {
                    encoder.write_index_key(key);
                }
            })
            .complete();
        let proof = match completion {
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
                return Err(EngineError::ApplyFailed(format!(
                    "typed CREATE INDEX generation failed: {error}"
                )));
            }
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(
                unknown,
            ) => {
                let error = unknown.error().to_string();
                drop(unknown);
                return Err(EngineError::ApplyFailed(format!(
                    "typed CREATE INDEX generation quiescence is unproven: {error}"
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
                "typed CREATE INDEX returned inconsistent GPU roots".to_string(),
            ));
        }
        let before_columns = predecessor
            .table_columns(table.stable_table_id)
            .collect::<Vec<_>>();
        let mut shapes = vec![[0; 32]; table.columns.len()];
        let mut columns = vec![[0; 32]; table.columns.len()];
        logical
            .copy_column_roots_into(&mut shapes, &mut columns)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed CREATE INDEX GPU column-root cardinality drifted".to_string(),
                )
            })?;
        if before_columns.len() != table.columns.len()
            || before_columns.iter().enumerate().any(|(ordinal, before)| {
                before.column_shape_root != shapes[ordinal]
                    || before.column_root != columns[ordinal]
            })
        {
            return Err(EngineError::ApplyFailed(
                "typed CREATE INDEX changed an existing column root".to_string(),
            ));
        }
        let mut index_roots = [gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
            stable_index_id: 0,
            initial_generation: 0,
            initial_root: [0; 32],
            final_generation: 0,
            final_root: [0; 32],
        }];
        logical
            .copy_index_generation_roots_into(&mut index_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed CREATE INDEX GPU index-root cardinality drifted".to_string(),
                )
            })?;
        let index_root = index_roots[0];
        if index_root.stable_index_id != u64::from(index.oid)
            || index_root.initial_generation != 0
            || index_root.initial_root != [0; 32]
            || index_root.final_generation != entry.index
            || index_root.final_root == [0; 32]
        {
            return Err(EngineError::ApplyFailed(
                "typed CREATE INDEX GPU index root is invalid".to_string(),
            ));
        }
        let table_map_completion = TypedTableMapGpuCompletion::from_logical_completion(&logical)?;
        let _generation_authority = commitments;
        LiveTypedGenerationRootPublication::from_exact_gpu_completed_index_root_enrollment(
            predecessor,
            table.stable_table_id,
            expected_table,
            TypedTableGenerationRoot {
                data_generation: entry.index,
                table_root: final_table_root,
                logical_row_count: 0,
            },
            expected_database_root,
            &expected_predecessor_index_roots,
            TypedIndexGenerationRoot {
                stable_index_id: index_root.stable_index_id,
                index_generation: index_root.final_generation,
                index_root: index_root.final_root,
            },
            final_database_root,
            &table_map_completion,
        )
        .map(Some)
    }
}
