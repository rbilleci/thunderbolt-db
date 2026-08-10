//! Dense-batch UNIQUE/PRIMARY KEY proof below the composite pre-WAL owner.
//!
//! This module binds the complete raw index vector in catalog order, including non-unique
//! indexes.  It owns the exact-key primitive's descriptor construction, but not the common
//! source, allocation scope, CHECK scheduling, or cross-class diagnostic precedence.

use super::host_retention::{HostRetentionGeometry, HostRetentionReport};
use super::resident_constraint_generation::{self, ResidentConstraintColumnBinding};
#[cfg(test)]
use super::CatalogSnapshot;
use super::{EngineError, Index};
use crate::relational_model::{RelationalColumn, RelationalIndex, RelationalTable};
use crate::typed_insert_batch::{TypedInsertBatch, TypedInsertConstraintDeviceSource};
use crate::ExecuteError;
use gpu_db_execution::{insert_batch_key_verdict_scratch_bytes, CudaCompoundFoldColumn};

pub(super) struct CompiledBatchKeyConstraints {
    #[cfg(test)]
    target_schema: String,
    #[cfg(test)]
    target_name: String,
    indexes: Box<[IndexBinding]>,
    evaluated_keys: Box<[CompiledBatchKey]>,
}

/// Complete successful catalog witness for the raw index vector.  In particular, index OIDs are
/// explicit because the table schema digest does not own index identity.
pub(crate) struct BatchKeyConstraintProof {
    #[cfg(test)]
    table_oid: u32,
    target_schema: String,
    target_name: String,
    #[cfg(test)]
    schema_digest: gpu_db_wal::CanonicalDigest,
    #[cfg(test)]
    catalog_seq: Index,
    indexes: Box<[IndexBinding]>,
}

#[derive(Clone)]
pub(crate) struct IndexBinding {
    raw_ordinal: usize,
    #[cfg(test)]
    oid: u32,
    pub(super) name: String,
    table: String,
    column: String,
    key_columns: Box<[String]>,
    unique: bool,
    primary_key: bool,
    #[cfg(test)]
    unique_constraint: bool,
    resolved_columns: Box<[KeyColumnBinding]>,
}

#[derive(Clone)]
pub(super) struct KeyColumnBinding {
    pub(super) id: u32,
    table_oid: u32,
    pub(super) attnum: i16,
    pub(super) name: String,
    ty: crate::SqlType,
    type_oid: u32,
    type_size: i16,
}

struct CompiledBatchKey {
    index_ordinal: usize,
    key_columns: Box<[KeyColumnBinding]>,
    /// A validity descriptor occurs once per bitmap-backed key column, in attnum order.  Its
    /// ordinal is the CUDA terminal's bounded `validity_ordinal` and is retained exactly.
    validity_columns: Box<[KeyColumnBinding]>,
}

pub(super) enum BatchKeyCandidate {
    PrimaryKeyNull { row: u32, column: KeyColumnBinding },
    Duplicate { row: u32, index_ordinal: usize },
}

impl CompiledBatchKeyConstraints {
    pub(super) fn index(&self, ordinal: usize) -> &IndexBinding {
        &self.indexes[ordinal]
    }

    pub(super) fn requires_device_source(&self) -> bool {
        !self.evaluated_keys.is_empty()
    }
}

/// Compile all raw indexes and the exact subset that has batch-local UNIQUE/PRIMARY semantics.
/// No resident/history conflict is claimed here: this primitive proves only duplicates inside the
/// proposed dense batch.
pub(super) fn compile(
    batch: &TypedInsertBatch,
    table: &RelationalTable,
) -> Result<CompiledBatchKeyConstraints, EngineError> {
    let mut indexes = Vec::with_capacity(table.indexes.len());
    let mut evaluated_keys = Vec::new();
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let binding = bind_index(raw_ordinal, table, index)?;
        if index.unique || index.primary_key {
            if binding.key_columns.is_empty() {
                return Err(EngineError::ApplyFailed(format!(
                    "device key proof found an empty key for index \"{}\"",
                    index.name
                )));
            }
            let mut validity_columns = Vec::new();
            for column in binding.resolved_columns.iter() {
                if batch.row_local_constraint_column_has_validity_bitmap(column.id)? {
                    validity_columns.push(column.clone());
                }
            }
            // Repeated key members are legal. CUDA receives a validity suffix once per physical
            // bitmap-backed column, ordered by `(attnum, id)` rather than declaration/key
            // position. This exact sealed geometry is shared by scratch sizing and evaluation.
            validity_columns.sort_unstable_by_key(|column| (column.attnum, column.id));
            validity_columns.dedup_by_key(|column| column.id);
            if !binding
                .resolved_columns
                .iter()
                .all(|column| cuda_key_type_supported(column.ty))
            {
                return Err(EngineError::ApplyFailed(format!(
                    "device key proof is unavailable for index \"{}\"",
                    index.name
                )));
            }
            evaluated_keys.push(CompiledBatchKey {
                index_ordinal: raw_ordinal,
                key_columns: binding.resolved_columns.clone(),
                validity_columns: validity_columns.into(),
            });
        }
        indexes.push(binding);
    }
    Ok(CompiledBatchKeyConstraints {
        #[cfg(test)]
        target_schema: table.schema.clone(),
        #[cfg(test)]
        target_name: table.name.clone(),
        indexes: indexes.into(),
        evaluated_keys: evaluated_keys.into(),
    })
}

/// Exact high-water for the largest sequential key primitive, excluding the shared source.
pub(super) fn max_scratch_bytes(
    batch: &TypedInsertBatch,
    compiled: &CompiledBatchKeyConstraints,
) -> Result<u64, EngineError> {
    let rows = usize::try_from(batch.binary_insert_template_row_count())
        .expect("u32 row count fits usize on supported hosts");
    let mut maximum = 0_u64;
    for key in compiled.evaluated_keys.iter() {
        let descriptor_count = key
            .key_columns
            .len()
            .checked_add(key.validity_columns.len())
            .ok_or_else(|| {
                EngineError::ApplyFailed("device key descriptor count overflows".to_string())
            })?;
        let scratch =
            insert_batch_key_verdict_scratch_bytes(rows, descriptor_count).ok_or_else(|| {
                EngineError::ApplyFailed("device key scratch extent overflows".to_string())
            })?;
        maximum = maximum.max(scratch);
    }
    Ok(maximum)
}

/// Run every accepted UNIQUE/PRIMARY key primitive on the common source.  Every primitive runs
/// before the caller chooses a global candidate; this leaf never returns an SQL diagnostic.
pub(super) fn evaluate(
    batch: &TypedInsertBatch,
    source: &TypedInsertConstraintDeviceSource,
    compiled: &CompiledBatchKeyConstraints,
) -> Result<Box<[BatchKeyCandidate]>, EngineError> {
    let mut candidates = Vec::new();
    for key in compiled.evaluated_keys.iter() {
        let mut descriptors =
            Vec::with_capacity(key.key_columns.len() + key.validity_columns.len());
        for column in key.key_columns.iter() {
            let (descriptor, _) = source.key_column_layout(column.id, &column.name)?;
            descriptors.push(descriptor);
        }
        for column in key.validity_columns.iter() {
            let (_, validity_bitmap_byte_offset) =
                source.key_column_layout(column.id, &column.name)?;
            let bitmap_byte_offset = validity_bitmap_byte_offset.ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "device key source is missing expected validity bitmap for column \"{}\"",
                    column.name
                ))
            })?;
            descriptors.push(CudaCompoundFoldColumn::Validity { bitmap_byte_offset });
        }
        let index = compiled.index(key.index_ordinal);
        let verdict = source
            .memory()
            .insert_batch_key_verdict_from_payload(
                &descriptors,
                batch.binary_insert_template_row_count(),
            )
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "device key verdict failed for index \"{}\": {error}",
                    index.name
                ))
            })?;
        if index.primary_key {
            if let Some(null) = verdict.first_null {
                let suffix = usize::try_from(null.validity_ordinal).map_err(|_| {
                    EngineError::ApplyFailed("device key NULL ordinal overflows".to_string())
                })?;
                let column = key.validity_columns.get(suffix).ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "device key NULL ordinal is outside its suffix".to_string(),
                    )
                })?;
                candidates.push(BatchKeyCandidate::PrimaryKeyNull {
                    row: null.row,
                    column: (*column).clone(),
                });
            }
        }
        if let Some(row) = verdict.first_duplicate_row {
            candidates.push(BatchKeyCandidate::Duplicate {
                row,
                index_ordinal: key.index_ordinal,
            });
        }
    }
    Ok(candidates.into())
}

#[cfg(test)]
pub(super) fn seal_after_success(
    batch: &TypedInsertBatch,
    compiled: &CompiledBatchKeyConstraints,
) -> BatchKeyConstraintProof {
    let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
    BatchKeyConstraintProof {
        #[cfg(test)]
        table_oid,
        target_schema: compiled.target_schema.clone(),
        target_name: compiled.target_name.clone(),
        #[cfg(test)]
        schema_digest,
        #[cfg(test)]
        catalog_seq,
        indexes: compiled.indexes.clone(),
    }
}

/// Seal the raw catalog-index witness used by the transaction-terminal indexed append.
///
/// UNIQUE/PRIMARY conflict semantics are closed by the transaction's GPU verdict before this
/// physical witness is requested. This owner binds the same maintained index descriptor for
/// publication and does not create a second constraint decision.
pub(crate) fn seal_current_index_witness(
    table: &RelationalTable,
    _catalog_seq: Index,
) -> Result<BatchKeyConstraintProof, EngineError> {
    if table.indexes.is_empty()
        || table.indexes.iter().any(|index| {
            (index.primary_key && !index.unique)
                || (index.unique_constraint && !index.unique)
                || index.key_columns.is_empty()
        })
    {
        return Err(EngineError::ApplyFailed(
            "indexed codec-5 path requires coherent nonempty maintained indexes".to_string(),
        ));
    }
    #[cfg(test)]
    let schema_digest =
        crate::engine_transaction_reset::table_schema_digest(table).map_err(|error| {
            EngineError::ApplyFailed(format!("indexed codec-5 schema digest failed: {error}"))
        })?;
    let indexes = table
        .indexes
        .iter()
        .enumerate()
        .map(|(ordinal, index)| bind_index(ordinal, table, index))
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    Ok(BatchKeyConstraintProof {
        #[cfg(test)]
        table_oid: table.oid,
        target_schema: table.schema.clone(),
        target_name: table.name.clone(),
        #[cfg(test)]
        schema_digest,
        #[cfg(test)]
        catalog_seq: _catalog_seq,
        indexes,
    })
}

impl BatchKeyConstraintProof {
    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(crate) fn indexes(&self) -> &[IndexBinding] {
        &self.indexes
    }

    /// Current retained host backing for the sealed raw-index witness. It has no generation pin:
    /// string/catalog contents are data backings, while the current generation owner lives in
    /// the separate validation seal.
    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_string(&self.target_schema)?;
        report.retain_string(&self.target_name)?;
        report.retain_boxed_slice(&self.indexes)?;
        for index in self.indexes.iter() {
            index.append_host_retention(report)?;
        }
        Ok(())
    }

    /// Scalar pre-lease geometry for the sealed raw-index witness. Its strings and boxes are
    /// private copies made by the proof compiler, so they cannot alias another preview owner.
    pub(crate) fn host_retention_geometry(&self) -> Result<HostRetentionGeometry, EngineError> {
        let mut geometry = HostRetentionGeometry::default();
        append_string_geometry(
            &mut geometry,
            &self.target_schema,
            "key proof target schema",
        )?;
        append_string_geometry(&mut geometry, &self.target_name, "key proof target name")?;
        geometry.checked_add_backing_elements::<IndexBinding>(
            self.indexes.len(),
            "key proof index box",
        )?;
        for index in self.indexes.iter() {
            index.append_host_retention_geometry(&mut geometry)?;
        }
        Ok(geometry)
    }

    #[cfg(test)]
    pub(super) fn matches_current_catalog(&self, catalog: &CatalogSnapshot) -> bool {
        self.matches_catalog_binding(catalog) && catalog.commit_seq == self.catalog_seq
    }

    /// The proof-only resident pass may revalidate after ordinary DML has republished identical
    /// catalog contents at a newer commit sequence.  Its target binding stays exact; only the
    /// unrelated monotonic catalog sequence is permitted to advance.  Production keeps the
    /// sequence-exact [`Self::matches_current_catalog`] witness.
    #[cfg(test)]
    pub(super) fn matches_current_target_binding(&self, catalog: &CatalogSnapshot) -> bool {
        catalog.commit_seq >= self.catalog_seq && self.matches_catalog_binding(catalog)
    }

    #[cfg(test)]
    fn matches_catalog_binding(&self, catalog: &CatalogSnapshot) -> bool {
        let Some(table) = catalog
            .relational_catalog
            .values()
            .find(|table| table.oid == self.table_oid)
        else {
            return false;
        };
        table.schema == self.target_schema
            && table.name == self.target_name
            && crate::engine_transaction_reset::table_schema_digest(table).ok()
                == Some(self.schema_digest)
            && table.indexes.len() == self.indexes.len()
            && table
                .indexes
                .iter()
                .enumerate()
                .all(|(ordinal, live)| self.indexes[ordinal].matches(ordinal, table, live))
    }
}

impl IndexBinding {
    /// Visit the already sealed catalog members without materializing a second binding box.
    ///
    /// Fixed-rollover index preparation uses this after its all-resource permit. Keeping the
    /// callback borrowed makes the permit-time host high-water a function of the final CUDA
    /// descriptor vector only, rather than an otherwise redundant box of cloned names.
    pub(crate) fn try_for_each_resolved_catalog_column(
        &self,
        table: &RelationalTable,
        mut visit: impl FnMut(usize, &RelationalColumn) -> Result<(), ExecuteError>,
    ) -> Result<(), ExecuteError> {
        for binding in self.resolved_columns.iter() {
            let (position, column) = table
                .columns
                .iter()
                .enumerate()
                .find(|(_, column)| binding.matches(column))
                .ok_or_else(|| {
                    ExecuteError::Serialization(
                        "resident key binding lost its catalog column".to_string(),
                    )
                })?;
            visit(position, column)?;
        }
        Ok(())
    }

    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), EngineError> {
        report.retain_string(&self.name)?;
        report.retain_string(&self.table)?;
        report.retain_string(&self.column)?;
        report.retain_boxed_slice(&self.key_columns)?;
        for column in self.key_columns.iter() {
            report.retain_string(column)?;
        }
        report.retain_boxed_slice(&self.resolved_columns)?;
        for column in self.resolved_columns.iter() {
            column.append_host_retention(report)?;
        }
        Ok(())
    }

    fn append_host_retention_geometry(
        &self,
        geometry: &mut HostRetentionGeometry,
    ) -> Result<(), EngineError> {
        append_string_geometry(geometry, &self.name, "key proof index name")?;
        append_string_geometry(geometry, &self.table, "key proof index table")?;
        append_string_geometry(geometry, &self.column, "key proof index column")?;
        geometry.checked_add_backing_elements::<String>(
            self.key_columns.len(),
            "key proof index key-column box",
        )?;
        for column in self.key_columns.iter() {
            append_string_geometry(geometry, column, "key proof index key-column name")?;
        }
        geometry.checked_add_backing_elements::<KeyColumnBinding>(
            self.resolved_columns.len(),
            "key proof resolved-column box",
        )?;
        for column in self.resolved_columns.iter() {
            append_string_geometry(geometry, &column.name, "key proof resolved-column name")?;
        }
        Ok(())
    }
    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(crate) fn raw_ordinal(&self) -> usize {
        self.raw_ordinal
    }

    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(super) fn is_unique_or_primary(&self) -> bool {
        self.unique || self.primary_key
    }

    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(super) fn key_columns(&self) -> &[KeyColumnBinding] {
        &self.resolved_columns
    }

    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(crate) fn key_column_count(&self) -> usize {
        self.resolved_columns.len()
    }

    /// Adapt this already-sealed UNIQUE/index witness to the neutral resident constraint column
    /// seam. The generic binding remains impossible to fabricate from names or types alone:
    /// each source member must still match the current catalog column exactly.
    pub(crate) fn resident_constraint_columns(
        &self,
        table: &RelationalTable,
    ) -> Result<Box<[ResidentConstraintColumnBinding]>, ExecuteError> {
        let columns = self
            .resolved_columns
            .iter()
            .map(|binding| {
                table
                    .columns
                    .iter()
                    .find(|column| binding.matches(column))
                    .ok_or_else(|| {
                        ExecuteError::Serialization(
                            "resident key binding lost its catalog column".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        resident_constraint_generation::bind_catalog_columns(table, columns)
    }

    #[cfg(test)]
    fn matches(&self, ordinal: usize, table: &RelationalTable, live: &RelationalIndex) -> bool {
        self.raw_ordinal == ordinal
            && self.oid == live.oid
            && self.name == live.name
            && self.table == live.table
            && self.column == live.column
            && self.key_columns.as_ref() == live.key_columns.as_slice()
            && self.unique == live.unique
            && self.primary_key == live.primary_key
            && self.unique_constraint == live.unique_constraint
            && self.resolved_columns.len() == live.key_columns.len()
            && self
                .resolved_columns
                .iter()
                .zip(live.key_columns.iter())
                .all(|(proof, name)| {
                    table
                        .columns
                        .iter()
                        .find(|column| &column.name == name)
                        .is_some_and(|column| proof.matches(column))
                })
    }
}

fn append_string_geometry(
    geometry: &mut HostRetentionGeometry,
    value: &String,
    domain: &'static str,
) -> Result<(), EngineError> {
    if value.capacity() == 0 {
        return Ok(());
    }
    let bytes = u64::try_from(value.capacity())
        .map_err(|_| EngineError::Durability("key proof string capacity overflows".to_string()))?;
    geometry.checked_add_backing_bytes_slots(bytes, 1, domain)
}

impl KeyColumnBinding {
    fn append_host_retention(&self, report: &mut HostRetentionReport) -> Result<(), EngineError> {
        report.retain_string(&self.name)
    }
}

impl KeyColumnBinding {
    #[allow(dead_code)] // consumed by the production-compiled, unreachable reservation
    pub(super) fn ty(&self) -> crate::SqlType {
        self.ty
    }

    fn matches(&self, live: &RelationalColumn) -> bool {
        self.id == live.id
            && self.table_oid == live.table_oid
            && self.attnum == live.attnum
            && self.name == live.name
            && self.ty == live.ty
            && self.type_oid == live.type_oid
            && self.type_size == live.type_size
    }
}

fn bind_index(
    raw_ordinal: usize,
    table: &RelationalTable,
    index: &RelationalIndex,
) -> Result<IndexBinding, EngineError> {
    if index.table != table.name
        || index.key_columns.is_empty()
        || index.column != index.key_columns[0]
    {
        return Err(EngineError::ApplyFailed(format!(
            "device key proof found malformed index \"{}\" on relation \"{}\"",
            index.name, table.name
        )));
    }
    let resolved_columns = index
        .key_columns
        .iter()
        .map(|name| {
            table
                .columns
                .iter()
                .find(|column| column.name == *name)
                .map(bind_column)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device key column \"{}\" is absent from relation \"{}\"",
                        name, table.name
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IndexBinding {
        raw_ordinal,
        #[cfg(test)]
        oid: index.oid,
        name: index.name.clone(),
        table: index.table.clone(),
        column: index.column.clone(),
        key_columns: index.key_columns.clone().into(),
        unique: index.unique,
        primary_key: index.primary_key,
        #[cfg(test)]
        unique_constraint: index.unique_constraint,
        resolved_columns: resolved_columns.into(),
    })
}

fn bind_column(column: &RelationalColumn) -> KeyColumnBinding {
    KeyColumnBinding {
        id: column.id,
        table_oid: column.table_oid,
        attnum: column.attnum,
        name: column.name.clone(),
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    }
}

fn cuda_key_type_supported(ty: crate::SqlType) -> bool {
    matches!(
        ty,
        crate::SqlType::Int2
            | crate::SqlType::Int4
            | crate::SqlType::Date
            | crate::SqlType::Int8
            | crate::SqlType::Timestamp
            | crate::SqlType::Numeric { .. }
            | crate::SqlType::Uuid
            | crate::SqlType::Bool
            | crate::SqlType::Text
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_raw_index_proof_is_move_only() {
        let source = include_str!("batch_key_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("implementation precedes tests");
        assert!(source.contains("pub(crate) struct BatchKeyConstraintProof"));
        assert!(!source.contains("impl Clone for BatchKeyConstraintProof"));
        assert!(!source.contains(
            "#[cfg_attr(test, derive(Clone))]\npub(crate) struct BatchKeyConstraintProof"
        ));
    }

    #[test]
    fn raw_index_proof_rejects_every_identity_component_after_schema_digest_refresh() {
        let engine = crate::Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE raw_index_binding (id int4 PRIMARY KEY, a int4 UNIQUE, b int8)",
            )
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE INDEX raw_index_binding_b ON raw_index_binding (b)",
            )
            .unwrap();
        let mut catalog = (*engine.catalog_snapshot()).clone();
        let table = catalog
            .relational_catalog
            .get_mut("raw_index_binding")
            .unwrap();
        table
            .indexes
            .push(crate::relational_model::RelationalIndex {
                oid: 90_003,
                name: "raw_index_binding_repeated".to_string(),
                table: table.name.clone(),
                column: "a".to_string(),
                key_columns: vec!["a".to_string(), "a".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: false,
            });
        let command =
            gpu_db_sql::parse_command("INSERT INTO raw_index_binding VALUES (1, 2, 3)").unwrap();
        let batch = crate::typed_insert_batch::seal_typed_insert_batch_for_test(
            &command,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .unwrap();
        let table = catalog.relational_catalog.get("raw_index_binding").unwrap();
        let compiled = compile(&batch, table).unwrap();
        let base = seal_after_success(&batch, &compiled);
        assert!(base.matches_current_catalog(&catalog));
        let mut advanced_catalog = catalog.clone();
        advanced_catalog.commit_seq = advanced_catalog.commit_seq.saturating_add(1);
        assert!(
            base.matches_current_target_binding(&advanced_catalog),
            "ordinary DML may republish an unchanged target at a newer catalog sequence"
        );
        let mut regressed_catalog = catalog.clone();
        regressed_catalog.commit_seq = regressed_catalog.commit_seq.saturating_sub(1);
        assert!(
            !base.matches_current_target_binding(&regressed_catalog),
            "the current-generation binding must never accept a catalog sequence before its seal"
        );
        let mut target_index_drift = catalog.clone();
        target_index_drift
            .relational_catalog
            .get_mut("raw_index_binding")
            .unwrap()
            .indexes[0]
            .name
            .push_str("_drift");
        assert!(
            !base.matches_current_target_binding(&target_index_drift),
            "monotonic sequence advance must not mask target index identity drift"
        );
        let repeated = base
            .indexes
            .iter()
            .find(|index| index.key_columns.len() == 2)
            .expect("repeated index remains in the raw proof");
        assert_eq!(repeated.resolved_columns.len(), 2);
        assert_eq!(
            repeated.resolved_columns[0].id, repeated.resolved_columns[1].id,
            "repeated key occurrences retain separate resolved bindings"
        );

        for sabotage in [
            "index_oid",
            "index_name",
            "index_table",
            "legacy_column",
            "key_columns",
            "unique",
            "primary_key",
            "unique_constraint",
            "column_id",
            "column_table_oid",
            "column_attnum",
            "column_name",
            "column_type",
            "column_type_oid",
            "column_type_size",
        ] {
            let mut drift = catalog.clone();
            let table = drift
                .relational_catalog
                .get_mut("raw_index_binding")
                .unwrap();
            let index = &mut table.indexes[0];
            match sabotage {
                "index_oid" => index.oid = index.oid.wrapping_add(1),
                "index_name" => index.name.push_str("_drift"),
                "index_table" => index.table.push_str("_drift"),
                "legacy_column" => index.column = "a".to_string(),
                "key_columns" => index.key_columns = vec!["a".to_string()],
                "unique" => index.unique = !index.unique,
                "primary_key" => index.primary_key = !index.primary_key,
                "unique_constraint" => index.unique_constraint = !index.unique_constraint,
                "column_id" => table.columns[0].id = table.columns[0].id.wrapping_add(1),
                "column_table_oid" => {
                    table.columns[0].table_oid = table.columns[0].table_oid.wrapping_add(1)
                }
                "column_attnum" => {
                    table.columns[0].attnum = table.columns[0].attnum.saturating_add(1)
                }
                "column_name" => table.columns[0].name.push_str("_drift"),
                "column_type" => table.columns[0].ty = crate::SqlType::Int8,
                "column_type_oid" => {
                    table.columns[0].type_oid = table.columns[0].type_oid.wrapping_add(1)
                }
                "column_type_size" => {
                    table.columns[0].type_size = table.columns[0].type_size.saturating_add(1)
                }
                _ => unreachable!(),
            }
            let mut proof = seal_after_success(&batch, &compiled);
            proof.schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
                .expect("sabotage table remains digestible");
            assert!(
                !proof.matches_current_catalog(&drift),
                "raw index proof accepted {sabotage} drift"
            );
        }

        for sabotage in [
            "index_vector_reorder",
            "index_vector_length",
            "repeated_second_key",
            "target_schema",
            "target_name",
            "target_oid",
            "catalog_seq",
        ] {
            let mut drift = catalog.clone();
            let table = drift
                .relational_catalog
                .get_mut("raw_index_binding")
                .unwrap();
            match sabotage {
                "index_vector_reorder" => table.indexes.swap(0, 1),
                "index_vector_length" => {
                    table.indexes.pop();
                }
                "repeated_second_key" => {
                    let repeated = table
                        .indexes
                        .iter_mut()
                        .find(|index| index.key_columns.len() == 2)
                        .expect("catalog retains repeated index");
                    repeated.key_columns[1] = "b".to_string();
                }
                "target_schema" => table.schema.push_str("_drift"),
                "target_name" => table.name.push_str("_drift"),
                "target_oid" => table.oid = table.oid.wrapping_add(1),
                "catalog_seq" => drift.commit_seq = drift.commit_seq.saturating_add(1),
                _ => unreachable!(),
            }
            let mut proof = seal_after_success(&batch, &compiled);
            if sabotage != "catalog_seq" {
                proof.schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
                    .expect("sabotage table remains digestible");
            }
            assert!(
                !proof.matches_current_catalog(&drift),
                "raw index proof accepted {sabotage} drift"
            );
        }
    }
}
