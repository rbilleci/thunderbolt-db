//! Scalar-only views over a strictly validated decoded typed-INSERT model.
//!
//! This leaf deliberately exposes only the S2/S5 binding facts needed by a future aggregate
//! replay validator. It cannot expose decoded vectors, names, batches, plans, mutation state, or
//! any conversion into a live write path.

use super::*;

/// Target identity retained by one canonical typed-INSERT record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedInsertTargetFacts {
    pub(crate) oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedInsertTargetIdentityFacts<'a> {
    pub(crate) schema: &'a str,
    pub(crate) name: &'a str,
    pub(crate) oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

/// One resolved logical value selected from a retained column at one row.  This is deliberately
/// a scalar borrowing view: it cannot expose a vector, raw byte image, or a mutable carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodedTypedValueFacts<'a> {
    I32(i32),
    I64(i64),
    I128(i128),
    Uuid([u8; 16]),
    Bool(bool),
    Text(&'a str),
}

/// RETURNING geometry retained by one canonical typed-INSERT record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedReturningLayoutFacts {
    pub(crate) digest: gpu_db_wal::CanonicalDigest,
    pub(crate) row_count: u32,
    pub(crate) column_count: u32,
    pub(crate) cell_count: u64,
}

/// Scalar facts which bind a decoded typed record to its aggregate statement directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedInsertRecordFacts {
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) target: DecodedTypedInsertTargetFacts,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) row_count: u32,
    pub(crate) column_count: u32,
    pub(crate) returning: DecodedReturningLayoutFacts,
    pub(crate) sequence_effect_count: u32,
}

/// The one admitted sequence parent shared by a nonempty decoded effect vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceParentFacts {
    pub(crate) txn_id: TxnId,
    pub(crate) autocommit: bool,
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal_base: u32,
}

/// Stable scalar identity of one canonical sequence-effect request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceRequestFacts {
    pub(crate) effect_ordinal: u32,
    pub(crate) target_table_oid: u32,
    pub(crate) row_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) sequence_oid: u32,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal: u32,
    pub(crate) absolute_expression_ordinal: u32,
}

/// Read-only kind facts for one sequence effect. Published values are independently durable;
/// private state remains model-owned, so its view contains only the request-binding digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodedSequenceEffectKindFacts {
    Published {
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
        returned_value: i64,
    },
    Private {
        input_digest: gpu_db_wal::CanonicalDigest,
    },
}

/// Scalar-only record of one canonical sequence effect, in canonical effect order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceEffectFacts {
    pub(crate) request: DecodedSequenceRequestFacts,
    /// The resolved typed-vector value. This equals `returned_value` for a published effect.
    pub(crate) resolved_value: i64,
    pub(crate) kind: DecodedSequenceEffectKindFacts,
}

/// Borrowed catalog-order target column identity.  It intentionally omits vector values and
/// input-state arrays; S7 only needs the sealed catalog/source binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedCatalogColumnFacts<'a> {
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) attnum: i16,
    pub(crate) name: &'a str,
    pub(crate) ty: SqlType,
    pub(crate) type_oid: u32,
    pub(crate) type_size: i16,
    pub(crate) source_ordinal: Option<u32>,
    pub(crate) domain_ordinal: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedDependencyFacts<'a> {
    pub(crate) ordinal: u32,
    pub(crate) schema: &'a str,
    pub(crate) name: &'a str,
    pub(crate) oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedDomainFacts<'a> {
    pub(crate) ordinal: u32,
    pub(crate) schema: &'a str,
    pub(crate) name: &'a str,
    pub(crate) oid: u32,
    pub(crate) base_type: SqlType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedIndexFacts<'a> {
    pub(crate) owner_dependency_ordinal: u32,
    pub(crate) raw_ordinal: u32,
    pub(crate) oid: u32,
    pub(crate) name: &'a str,
    pub(crate) table_name: &'a str,
    pub(crate) first_column_name: &'a str,
    pub(crate) unique: bool,
    pub(crate) primary_key: bool,
    pub(crate) unique_constraint: bool,
    pub(crate) key_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedCatalogBindingFacts<'a> {
    pub(crate) dependency_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) attnum: i16,
    pub(crate) name: &'a str,
    pub(crate) ty: SqlType,
    pub(crate) type_oid: u32,
    pub(crate) type_size: i16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedForeignKeyFacts<'a> {
    pub(crate) raw_ordinal: u32,
    pub(crate) name: &'a str,
    pub(crate) child_column_name: &'a str,
    pub(crate) referenced_table_name: &'a str,
    pub(crate) referenced_column_name: &'a str,
    pub(crate) parent_dependency_ordinal: u32,
    pub(crate) child_column: DecodedCatalogBindingFacts<'a>,
    pub(crate) parent_column: DecodedCatalogBindingFacts<'a>,
    pub(crate) supporting_index: DecodedIndexFacts<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceBindingFacts<'a> {
    pub(crate) effect_ordinal: u32,
    pub(crate) source_name: &'a str,
    pub(crate) effective_name: &'a str,
    pub(crate) descriptor_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) request: DecodedSequenceRequestFacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedReturningProjectionFacts<'a> {
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) attnum: i16,
    pub(crate) name: &'a str,
    pub(crate) ty: SqlType,
    pub(crate) type_oid: u32,
    pub(crate) type_size: i16,
}

pub(super) fn record_facts(model: &DecodedModel) -> DecodedTypedInsertRecordFacts {
    DecodedTypedInsertRecordFacts {
        typed_statement_digest: model.typed_statement_digest,
        target: DecodedTypedInsertTargetFacts {
            oid: model.target.oid,
            schema_digest: model.target.schema_digest,
        },
        statement_ordinal: model.target.statement_ordinal,
        row_count: model.target.rows,
        column_count: model.target.column_count,
        returning: DecodedReturningLayoutFacts {
            digest: model.returning_digest,
            row_count: model.returning.rows,
            column_count: model.returning.columns,
            cell_count: model.returning.cells,
        },
        sequence_effect_count: u32::try_from(model.effects.effects.len())
            .expect("validated decoded sequence effect count fits u32"),
    }
}

pub(super) fn target_identity(model: &DecodedModel) -> DecodedTypedInsertTargetIdentityFacts<'_> {
    DecodedTypedInsertTargetIdentityFacts {
        schema: &model.target.schema,
        name: &model.target.name,
        oid: model.target.oid,
        schema_digest: model.target.schema_digest,
    }
}

pub(super) fn column_value_at(
    model: &DecodedModel,
    catalog_column_ordinal: u32,
    row_ordinal: u32,
) -> Result<(bool, DecodedTypedValueFacts<'_>), EngineError> {
    let column = model
        .columns
        .get(catalog_column_ordinal as usize)
        .ok_or_else(|| codec_error("decoded catalog column is absent"))?;
    let row = usize::try_from(row_ordinal).map_err(|_| codec_error("decoded row overflows"))?;
    if row >= column.states.len() {
        return Err(codec_error("decoded row is absent"));
    }
    let value = match &column.values {
        TypedInsertColumnValues::I32(values) => DecodedTypedValueFacts::I32(values[row]),
        TypedInsertColumnValues::I64(values) => DecodedTypedValueFacts::I64(values[row]),
        TypedInsertColumnValues::I128(values) => DecodedTypedValueFacts::I128(values[row]),
        TypedInsertColumnValues::Bytes16(values) => DecodedTypedValueFacts::Uuid(values[row]),
        TypedInsertColumnValues::BoolBits(words) => {
            DecodedTypedValueFacts::Bool(bit_is_set(words, row))
        }
        TypedInsertColumnValues::Text { offsets, bytes } => {
            let start = usize::try_from(offsets[row])
                .map_err(|_| codec_error("decoded text start overflows"))?;
            let end = usize::try_from(offsets[row + 1])
                .map_err(|_| codec_error("decoded text end overflows"))?;
            DecodedTypedValueFacts::Text(
                std::str::from_utf8(&bytes[start..end])
                    .map_err(|_| codec_error("validated decoded text is not UTF-8"))?,
            )
        }
    };
    Ok((column.validity.is_valid(row), value))
}

pub(super) fn sequence_parent_facts(model: &DecodedModel) -> Option<DecodedSequenceParentFacts> {
    model.effects.parent.map(parent_facts)
}

pub(super) fn sequence_effect_facts(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedSequenceEffectFacts> + '_ {
    model.effects.effects.iter().map(effect_facts)
}

pub(super) fn catalog_columns(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedCatalogColumnFacts<'_>> {
    model
        .columns
        .iter()
        .map(|column| DecodedCatalogColumnFacts {
            catalog_column_ordinal: column.ordinal,
            column_id: column.column_id,
            attnum: column.attnum,
            name: &column.name,
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
            source_ordinal: column.source_ordinal,
            domain_ordinal: column.domain_ordinal,
        })
}

pub(super) fn dependencies(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedDependencyFacts<'_>> {
    model
        .dependencies
        .iter()
        .enumerate()
        .map(|(ordinal, dependency)| DecodedDependencyFacts {
            ordinal: u32::try_from(ordinal).expect("validated dependency ordinal fits u32"),
            schema: &dependency.schema,
            name: &dependency.name,
            oid: dependency.oid,
            schema_digest: dependency.schema_digest,
        })
}

pub(super) fn domains(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedDomainFacts<'_>> {
    model
        .domains
        .iter()
        .enumerate()
        .map(|(ordinal, domain)| DecodedDomainFacts {
            ordinal: u32::try_from(ordinal).expect("validated domain ordinal fits u32"),
            schema: &domain.schema,
            name: &domain.name,
            oid: domain.oid,
            base_type: domain.base_type,
        })
}

pub(super) fn indexes(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedIndexFacts<'_>> {
    model.indexes.iter().map(index_facts)
}

pub(super) fn index_key_columns(
    model: &DecodedModel,
    index_ordinal: u32,
) -> Result<impl ExactSizeIterator<Item = DecodedCatalogBindingFacts<'_>>, EngineError> {
    let index = model
        .indexes
        .get(index_ordinal as usize)
        .ok_or_else(|| codec_error("decoded index is absent"))?;
    Ok(index.key_columns.iter().map(catalog_binding_facts))
}

pub(super) fn foreign_keys(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedForeignKeyFacts<'_>> {
    model
        .foreign_keys
        .iter()
        .map(|foreign_key| DecodedForeignKeyFacts {
            raw_ordinal: foreign_key.raw_ordinal,
            name: &foreign_key.name,
            child_column_name: &foreign_key.child_column_name,
            referenced_table_name: &foreign_key.referenced_table_name,
            referenced_column_name: &foreign_key.referenced_column_name,
            parent_dependency_ordinal: foreign_key.parent_dependency_ordinal,
            child_column: catalog_binding_facts(&foreign_key.child_column),
            parent_column: catalog_binding_facts(&foreign_key.parent_column),
            supporting_index: index_facts(&foreign_key.supporting_index),
        })
}

pub(super) fn foreign_key_supporting_index_keys(
    model: &DecodedModel,
    foreign_key_ordinal: u32,
) -> Result<impl ExactSizeIterator<Item = DecodedCatalogBindingFacts<'_>>, EngineError> {
    let foreign_key = model
        .foreign_keys
        .get(foreign_key_ordinal as usize)
        .ok_or_else(|| codec_error("decoded foreign key is absent"))?;
    Ok(foreign_key
        .supporting_index
        .key_columns
        .iter()
        .map(catalog_binding_facts))
}

pub(super) fn sequence_bindings(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedSequenceBindingFacts<'_>> {
    model
        .effects
        .effects
        .iter()
        .map(|effect| DecodedSequenceBindingFacts {
            effect_ordinal: effect.request.ordinal,
            source_name: &effect.request.source_name,
            effective_name: &effect.request.effective_name,
            descriptor_digest: effect.request.descriptor_digest,
            request: effect_facts(effect).request,
        })
}

pub(super) fn returning_projections(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedReturningProjectionFacts<'_>> {
    model
        .returning
        .projections
        .iter()
        .map(|projection| DecodedReturningProjectionFacts {
            catalog_column_ordinal: projection.catalog_column_ordinal,
            column_id: projection.column_id,
            attnum: projection.attnum,
            name: &projection.name,
            ty: projection.ty,
            type_oid: projection.type_oid,
            type_size: projection.type_size,
        })
}

fn index_facts(index: &DecodedIndex) -> DecodedIndexFacts<'_> {
    DecodedIndexFacts {
        owner_dependency_ordinal: index.owner_dependency_ordinal,
        raw_ordinal: index.raw_ordinal,
        oid: index.oid,
        name: &index.name,
        table_name: &index.table_name,
        first_column_name: &index.first_column_name,
        unique: index.unique,
        primary_key: index.primary_key,
        unique_constraint: index.unique_constraint,
        key_count: u32::try_from(index.key_columns.len())
            .expect("validated index key count fits u32"),
    }
}

fn catalog_binding_facts(column: &DecodedCatalogColumn) -> DecodedCatalogBindingFacts<'_> {
    DecodedCatalogBindingFacts {
        dependency_ordinal: column.dependency_ordinal,
        catalog_column_ordinal: column.catalog_column_ordinal,
        column_id: column.column_id,
        attnum: column.attnum,
        name: &column.name,
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    }
}

fn parent_facts(parent: DecodedParent) -> DecodedSequenceParentFacts {
    DecodedSequenceParentFacts {
        txn_id: parent.txn_id,
        autocommit: parent.autocommit,
        request_digest: parent.request_digest,
        statement_ordinal: parent.statement_ordinal,
        expression_ordinal_base: parent.expression_base,
    }
}

fn effect_facts(effect: &DecodedEffect) -> DecodedSequenceEffectFacts {
    let request = &effect.request;
    let kind = match &effect.kind {
        DecodedEffectKind::Published {
            transition_txn_id,
            input_digest,
            returned_value,
        } => DecodedSequenceEffectKindFacts::Published {
            transition_txn_id: *transition_txn_id,
            input_digest: *input_digest,
            returned_value: *returned_value,
        },
        DecodedEffectKind::Private { input_digest, .. } => {
            // The full private chain is retained by `DecodedModel`; S5 only needs the binding
            // digest while S2 still owns all private transition state.
            DecodedSequenceEffectKindFacts::Private {
                input_digest: *input_digest,
            }
        }
    };
    DecodedSequenceEffectFacts {
        request: DecodedSequenceRequestFacts {
            effect_ordinal: request.ordinal,
            target_table_oid: request.target_table_oid,
            row_ordinal: request.row_ordinal,
            catalog_column_ordinal: request.catalog_column_ordinal,
            column_id: request.column_id,
            sequence_oid: request.sequence_oid,
            statement_ordinal: request.statement_ordinal,
            expression_ordinal: request.expression_ordinal,
            absolute_expression_ordinal: request.absolute_expression_ordinal,
        },
        resolved_value: request.value,
        kind,
    }
}
