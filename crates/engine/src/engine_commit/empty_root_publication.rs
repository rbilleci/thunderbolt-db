//! GPU-authenticated zero-row CREATE root preparation.
//!
//! This leaf derives the sole immutable table-map successor after the ordinary catalog CREATE
//! allocates stable identities. It returns a publication candidate to the parent commit/apply
//! owner and neither appends WAL nor installs visibility.

use super::*;

impl Engine {
    /// After the ordinary catalog CREATE has allocated its stable identities, derive one empty
    /// table-map leaf substitution on the GPU and return its exact publication candidate.  The
    /// first CREATE derives the empty database on-device; every later eligible CREATE pins the
    /// current database root and its table-map sibling path before adding one absent leaf.
    pub(super) fn prepare_empty_typed_create_root_publication(
        &self,
        commit: &mut CommitState,
        entry: &LogEntry,
        cat: &DdlCatalogState,
    ) -> Result<Option<LiveTypedGenerationRootPublication>, EngineError> {
        let Some(Command::CreateTable(create)) = Self::decode_engine_command(&entry.payload)?
        else {
            return Ok(None);
        };
        let table = cat.relational_catalog.get(&create.table).ok_or_else(|| {
            EngineError::ApplyFailed(
                "CREATE TABLE did not publish its catalog table before typed root generation"
                    .to_string(),
            )
        })?;
        // CREATE TABLE and CREATE INDEX must establish the same GPU-root predecessor. Every
        // foldable inline index (including PRIMARY KEY/UNIQUE) is therefore encoded in this one
        // CREATE generation. Row-local CHECK and FK metadata are part of `schema_digest`, but
        // neither changes a zero-row physical generation; their device verdicts close before a
        // later INSERT reaches the terminal.
        if table.columns.is_empty() {
            return Ok(None);
        }
        let inline_indexes = if table.indexes.is_empty() {
            &table.indexes[..0]
        } else if crate::engine_residency::write001_empty_foldable_index_enrollment(table) {
            table.indexes.as_slice()
        } else {
            return Ok(None);
        };
        let mut inline_index_keys = Vec::with_capacity(inline_indexes.len());
        for index in inline_indexes {
            let mut keys = Vec::with_capacity(index.key_columns.len());
            for (key_ordinal, key_name) in index.key_columns.iter().enumerate() {
                let (catalog_column_ordinal, column) = table
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, column)| column.name == *key_name)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "inline CREATE index key is absent from its catalog table".to_string(),
                        )
                    })?;
                keys.push(
                    gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                        key_ordinal: u32::try_from(key_ordinal).map_err(|_| {
                            EngineError::ApplyFailed(
                                "inline CREATE index key ordinal exceeds u32".to_string(),
                            )
                        })?,
                        catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(
                            |_| {
                                EngineError::ApplyFailed(
                                    "inline CREATE index catalog ordinal exceeds u32".to_string(),
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
            inline_index_keys.push(keys);
        }
        let predecessor = self.read_state.typed_generation_roots.load_full();
        if predecessor.table(table.stable_table_id).is_some() {
            return Err(EngineError::ApplyFailed(
                "typed empty CREATE found an already-published table predecessor".to_string(),
            ));
        }
        let (table_map_predecessor, expected_database_root) = match predecessor.database_root {
            None => {
                if predecessor.table_map_root().is_some() {
                    return Err(EngineError::ApplyFailed(
                        "typed empty CREATE found a table map without a database root".to_string(),
                    ));
                }
                (
                    gpu_db_execution::RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase,
                    None,
                )
            }
            Some(database_root) => {
                let table_map_predecessor = predecessor
                    .retained_table_map_predecessor(table.stable_table_id, database_root)?
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "typed empty CREATE database root has no retained table-map witness"
                                .to_string(),
                        )
                    })?;
                (table_map_predecessor, Some(database_root))
            }
        };
        let Some(canonical) = commit
            .wal
            .last_record()
            .and_then(|record| {
                gpu_db_wal::decode_canonical_record_payload(&record.payload).transpose()
            })
            .transpose()?
        else {
            // Pre-canonical CREATE records retain their historical decoder/recovery behavior.
            // Only a current canonical envelope can authorize the GPU-native empty root.
            return Ok(None);
        };
        if canonical.header.commit_seq != entry.index
            || canonical.header.catalog_after_epoch == 0
            || canonical.header.catalog_after_digest == [0; 32]
            || canonical.header.identity != commit.canonical_identity
        {
            return Err(EngineError::ApplyFailed(
                "typed empty CREATE canonical identity differs from its committed entry"
                    .to_string(),
            ));
        }
        let schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let target = self
            .cuda_driver_probe_runtime()
            .runtime_typed_insert_generation_target(0)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed empty CREATE CUDA target declined: {error:?}"
                ))
            })?;
        let attempt = gpu_db_execution::RuntimeTypedInsertGenerationAttempt::new(entry.index)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "typed empty CREATE generation attempt is invalid: {error:?}"
                ))
            })?;
        let prepared = gpu_db_execution::PreparedRuntimeTypedInsertGeneration::reserve(
            target,
            attempt,
            gpu_db_execution::RuntimeTypedInsertGenerationGeometry {
                rows: 0,
                cells: table.columns.len(),
                value_bytes: 0,
                indexes: inline_indexes.len(),
                index_keys: inline_index_keys.iter().map(Vec::len).sum(),
                index_effects: 0,
                index_effect_components: 0,
            },
        )
        .map_err(|error| {
            EngineError::ApplyFailed(format!(
                "typed empty CREATE generation reservation declined: {error:?}"
            ))
        })?;
        let identity = gpu_db_execution::RuntimeTypedInsertGenerationIdentity {
            database_id: canonical.header.identity.database_id,
            catalog_epoch: canonical.header.catalog_after_epoch,
            catalog_digest: canonical.header.catalog_after_digest,
            stable_transaction_id: canonical.header.stable_transaction_id,
            commit_sequence: entry.index,
            write001_typed_statement_digest: [0; 32],
        };
        let allocator_frontier = self.read_state.mvcc.current_row_id();
        let runtime_table = gpu_db_execution::RuntimeTypedInsertGenerationTable {
            action: gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateEmpty,
            table_map_predecessor,
            stable_table_id: table.stable_table_id,
            write001_final_image_ref: 0,
            base_data_generation: entry.index,
            base_table_root: [0; 32],
            row_allocator_before: allocator_frontier,
            row_allocator_high_water: allocator_frontier,
            initial_logical_row_count: 0,
            final_logical_row_count: 0,
            image_layout_digest: schema_digest,
            image_content_digest: [0; 32],
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
                for (raw_catalog_index_ordinal, (index, keys)) in
                    inline_indexes.iter().zip(&inline_index_keys).enumerate()
                {
                    let key_count = u32::try_from(keys.len())
                        .expect("validated inline CREATE index key count fits u32");
                    encoder.write_index(gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                        stable_index_id: u64::from(index.oid),
                        raw_catalog_index_ordinal: u32::try_from(raw_catalog_index_ordinal)
                            .expect("validated inline CREATE index ordinal fits u32"),
                        index_flags: u32::from(index.unique)
                            | (u32::from(index.primary_key) << 1)
                            | (u32::from(index.unique_constraint) << 2)
                            | (1 << 3),
                        null_equality_policy: 1,
                        base_generation: 0,
                        base_root: [0; 32],
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
                        .expect("validated inline CREATE index key range fits u32");
                }
            })
            .complete();
        let proof = match completion {
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
                return Err(EngineError::ApplyFailed(format!(
                    "typed empty CREATE generation failed: {error}"
                )));
            }
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(
                unknown,
            ) => {
                let error = unknown.error().to_string();
                drop(unknown);
                return Err(EngineError::ApplyFailed(format!(
                    "typed empty CREATE generation quiescence is unproven: {error}"
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
        if initial_table_root != [0; 32]
            || final_table_root == [0; 32]
            || initial_database_root == [0; 32]
            || final_database_root == [0; 32]
            || initial_database_root == final_database_root
            || expected_database_root.is_some_and(|root| root != initial_database_root)
        {
            return Err(EngineError::ApplyFailed(
                "typed empty CREATE returned inconsistent GPU roots".to_string(),
            ));
        }
        let mut shape_roots = vec![[0; 32]; table.columns.len()];
        let mut column_roots = vec![[0; 32]; table.columns.len()];
        logical
            .copy_column_roots_into(&mut shape_roots, &mut column_roots)
            .map_err(|_| {
                EngineError::ApplyFailed(
                    "typed empty CREATE GPU column-root cardinality drifted".to_string(),
                )
            })?;
        let columns = table
            .columns
            .iter()
            .enumerate()
            .map(|(ordinal, column)| {
                Ok(TypedColumnGenerationRoot {
                    catalog_column_ordinal: u32::try_from(ordinal).map_err(|_| {
                        EngineError::ApplyFailed(
                            "typed empty CREATE column ordinal exceeds u32".to_string(),
                        )
                    })?,
                    stable_column_id: column.id,
                    attnum: column.attnum,
                    column_shape_root: shape_roots[ordinal],
                    column_root: column_roots[ordinal],
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let successor_index_roots = if inline_indexes.is_empty() {
            Vec::new()
        } else {
            let mut roots = vec![
                gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
                    stable_index_id: 0,
                    initial_generation: 0,
                    initial_root: [0; 32],
                    final_generation: 0,
                    final_root: [0; 32],
                };
                inline_indexes.len()
            ];
            logical
                .copy_index_generation_roots_into(&mut roots)
                .map_err(|_| {
                    EngineError::ApplyFailed(
                        "inline CREATE GPU index-root cardinality drifted".to_string(),
                    )
                })?;
            let mut successor = Vec::with_capacity(roots.len());
            for (index, root) in inline_indexes.iter().zip(roots) {
                if root.stable_index_id != u64::from(index.oid)
                    || root.initial_generation != 0
                    || root.initial_root != [0; 32]
                    || root.final_generation != entry.index
                    || root.final_root == [0; 32]
                {
                    return Err(EngineError::ApplyFailed(
                        "inline CREATE GPU index root is invalid".to_string(),
                    ));
                }
                successor.push(TypedIndexGenerationRoot {
                    stable_index_id: root.stable_index_id,
                    index_generation: root.final_generation,
                    index_root: root.final_root,
                });
            }
            successor
        };
        let table_map_completion = TypedTableMapGpuCompletion::from_logical_completion(&logical)?;
        // The opaque completion values are consumed only through their fixed, GPU-produced
        // publication slots. No host hash or alternative root derivation exists here.
        let _generation_authority = commitments;
        LiveTypedGenerationRootPublication::from_exact_gpu_completed_table_map_predecessor(
            predecessor,
            table.stable_table_id,
            None,
            false,
            expected_database_root,
            TypedTableGenerationRoot {
                data_generation: entry.index,
                table_root: final_table_root,
                logical_row_count: 0,
            },
            &columns,
            &[],
            &successor_index_roots,
            final_database_root,
            &table_map_completion,
        )
        .map(Some)
    }
}
