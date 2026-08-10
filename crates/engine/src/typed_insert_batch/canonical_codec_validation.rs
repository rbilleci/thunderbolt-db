//! Cross-section canonicality checks for the inert typed-INSERT codec.

use super::sequence::PrivateOwner;
use super::*;
use crate::typed_insert_batch::sequence_defaults::effects::{
    CanonicalSequenceParentView, CanonicalSequenceRequestView,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
pub(super) enum SequenceSectionKind {
    Published {
        transition_txn_id: TxnId,
    },
    Private {
        lifetime_origin: u8,
        owner: PrivateOwner,
    },
}

#[derive(Clone, Copy)]
pub(super) struct SequenceSectionEntry<'a> {
    pub(super) request: CanonicalSequenceRequestView<'a>,
    pub(super) absolute_expression_ordinal: u32,
    pub(super) kind: SequenceSectionKind,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_sequence_target_cell(
    target_oid: u32,
    rows: u32,
    request: CanonicalSequenceRequestView<'_>,
    column_id: u32,
    ty: SqlType,
    state: Option<TypedInsertInputState>,
    defaulted: bool,
    valid: bool,
    resolved_value: Option<i32>,
    request_value: i64,
) -> Result<(), EngineError> {
    if request.target_table_oid != target_oid
        || request.row_ordinal >= rows
        || request.column_id != column_id
        || ty != SqlType::Int4
        || !matches!(
            state,
            Some(TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault)
        )
        || !defaulted
        || !valid
        || i32::try_from(request_value).ok() != resolved_value
    {
        return Err(codec_error(
            "sequence effect does not match resolved typed vector output",
        ));
    }
    Ok(())
}

pub(super) fn validate_sequence_section(
    parent: CanonicalSequenceParentView,
    entries: &[SequenceSectionEntry<'_>],
) -> Result<(), EngineError> {
    if entries.is_empty() {
        return Err(codec_error("sequence section parent has no effects"));
    }
    if parent.autocommit
        && (parent.statement_ordinal != InsertStatementOrdinal::FIRST
            || parent.expression_ordinal_base != 0)
    {
        return Err(codec_error(
            "autocommit sequence parent geometry is invalid",
        ));
    }
    let active_columns = entries
        .iter()
        .map(|entry| entry.request.catalog_column_ordinal)
        .collect::<BTreeSet<_>>();
    let active_count = u32::try_from(active_columns.len())
        .map_err(|_| codec_error("active sequence column count overflows"))?;
    if active_count == 0 {
        return Err(codec_error("sequence section has no active columns"));
    }
    let slots = active_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, column)| {
            u32::try_from(slot)
                .map(|slot| (column, slot))
                .map_err(|_| codec_error("active sequence slot overflows"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut locals = BTreeSet::new();
    let mut absolutes = BTreeSet::new();
    let mut columns = BTreeMap::new();
    let mut sequences = BTreeMap::new();
    let mut effective_names = BTreeMap::new();
    let mut prior_key = None;
    let mut prior_transition = None;
    for entry in entries {
        let request = entry.request;
        if request.statement_ordinal != parent.statement_ordinal {
            return Err(codec_error("sequence request statement identity drifted"));
        }
        let key = (request.row_ordinal, request.catalog_column_ordinal);
        if prior_key.is_some_and(|prior| prior >= key) {
            return Err(codec_error(
                "sequence section is not row-major/catalog ordered",
            ));
        }
        prior_key = Some(key);
        let slot = *slots
            .get(&request.catalog_column_ordinal)
            .ok_or_else(|| codec_error("active sequence slot is absent"))?;
        let local = request
            .row_ordinal
            .checked_mul(active_count)
            .and_then(|base| base.checked_add(slot))
            .ok_or_else(|| codec_error("sequence local expression ordinal overflows"))?;
        let absolute = parent
            .expression_ordinal_base
            .checked_add(local)
            .ok_or_else(|| codec_error("sequence absolute expression ordinal overflows"))?;
        if request.expression_ordinal != local
            || entry.absolute_expression_ordinal != absolute
            || !locals.insert(local)
            || !absolutes.insert(absolute)
        {
            return Err(codec_error("sequence expression geometry is noncanonical"));
        }
        match columns.entry(request.catalog_column_ordinal) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert((
                    request.column_id,
                    request.sequence_oid,
                    request.sequence_source_name,
                    request.sequence_effective_name,
                ));
            }
            std::collections::btree_map::Entry::Occupied(slot)
                if *slot.get()
                    != (
                        request.column_id,
                        request.sequence_oid,
                        request.sequence_source_name,
                        request.sequence_effective_name,
                    ) =>
            {
                return Err(codec_error("sequence column identity drifted"));
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
        let identity = SequenceOidIdentity::from_entry(entry);
        match sequences.entry(request.sequence_oid) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(identity);
            }
            std::collections::btree_map::Entry::Occupied(slot) if *slot.get() != identity => {
                return Err(codec_error("sequence OID mode or identity drifted"));
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
        if let Some(previous) =
            effective_names.insert(request.sequence_effective_name, request.sequence_oid)
        {
            if previous != request.sequence_oid {
                return Err(codec_error("sequence effective-name identity drifted"));
            }
        }
        if let SequenceSectionKind::Published { transition_txn_id } = entry.kind {
            if transition_txn_id == 0
                || transition_txn_id == parent.txn_id
                || prior_transition.is_some_and(|prior| prior >= transition_txn_id)
            {
                return Err(codec_error("published sequence transition order drifted"));
            }
            prior_transition = Some(transition_txn_id);
        } else if parent.autocommit {
            return Err(codec_error(
                "autocommit sequence section has private effect",
            ));
        } else if let SequenceSectionKind::Private { owner, .. } = entry.kind {
            // The owner ordinal is in the complete transaction-operation program while the
            // parent ordinal is codec-5's typed-INSERT-only S1 order, so their numeric values are
            // not order-comparable here. The live aggregate binds the owner to an earlier catalog
            // operation; this statement-local codec only rejects an absent or aliased owner.
            if owner.statement_ordinal == u32::MAX
                || owner.statement_digest == parent.request_digest
            {
                return Err(codec_error(
                    "private sequence owner is not before parent statement",
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SequenceOidMode {
    Published,
    Private {
        lifetime_origin: u8,
        owner: PrivateOwner,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SequenceOidIdentity<'a> {
    effective_name: &'a str,
    mode: SequenceOidMode,
}

impl<'a> SequenceOidIdentity<'a> {
    fn from_entry(entry: &SequenceSectionEntry<'a>) -> Self {
        let mode = match entry.kind {
            SequenceSectionKind::Published { .. } => SequenceOidMode::Published,
            SequenceSectionKind::Private {
                lifetime_origin,
                owner,
            } => SequenceOidMode::Private {
                lifetime_origin,
                owner,
            },
        };
        Self {
            effective_name: entry.request.sequence_effective_name,
            mode,
        }
    }
}

/// Borrowed encoder/decoder-neutral target identity for the one canonical catalog registry.
#[derive(Clone, Copy)]
pub(super) struct GlobalTargetIdentity<'a> {
    pub(super) schema: &'a str,
    pub(super) name: &'a str,
    pub(super) oid: u32,
    pub(super) schema_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Clone, Copy)]
pub(super) struct GlobalTargetColumnIdentity<'a> {
    pub(super) name: &'a str,
    pub(super) column_id: u32,
    pub(super) attnum: i16,
    pub(super) ty: SqlType,
    pub(super) type_oid: u32,
    pub(super) type_size: i16,
}

#[derive(Clone, Copy)]
pub(super) struct GlobalSequenceDescriptor<'a> {
    pub(super) oid: u32,
    pub(super) effective_name: &'a str,
}

/// Validates the complete relation-shaped identity namespace from either canonical producer.
/// All registry keys are owned values so the same code accepts both typed-batch and decoded views.
pub(super) fn validate_global_identity_registry(
    target: GlobalTargetIdentity<'_>,
    target_columns: &[GlobalTargetColumnIdentity<'_>],
    dependencies: &[TypedInsertDependencyBinding],
    domains: &[TypedInsertDomainBinding],
    indexes: &[TypedInsertCanonicalIndexBinding],
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
    sequences: &[GlobalSequenceDescriptor<'_>],
) -> Result<(), EngineError> {
    let mut registry = GlobalIdentityRegistry::default();
    registry.register_relation(target.oid, target.schema, target.name, target.schema_digest)?;
    for dependency in dependencies {
        registry.register_relation(
            dependency.oid,
            &dependency.schema,
            &dependency.name,
            dependency.schema_digest,
        )?;
    }
    for domain in domains {
        registry.register_domain(domain)?;
    }
    let mut target_names = BTreeSet::new();
    let mut prior_attnum = None;
    for (ordinal, column) in target_columns.iter().enumerate() {
        let catalog_column_ordinal =
            u32::try_from(ordinal).map_err(|_| codec_error("target column ordinal overflows"))?;
        if column.name.is_empty()
            || column.column_id == 0
            || column.attnum <= 0
            || catalog_column_ordinal >= column.attnum as u32
            || column.type_oid == 0
            || column.type_size != column.ty.type_size()
            || !target_names.insert(column.name)
            || prior_attnum.is_some_and(|prior| prior >= column.attnum)
        {
            return Err(codec_error("target catalog column identity is invalid"));
        }
        prior_attnum = Some(column.attnum);
        registry.register_column(ColumnIdentity {
            relation_oid: target.oid,
            catalog_column_ordinal,
            column_id: column.column_id,
            attnum: column.attnum,
            name: column.name.to_owned(),
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
        })?;
    }
    for index in indexes {
        registry.register_index(index, dependencies)?;
    }
    for foreign_key in foreign_keys {
        registry.register_catalog_column(&foreign_key.child_column, dependencies)?;
        registry.register_catalog_column(&foreign_key.parent_column, dependencies)?;
        registry.register_index(&foreign_key.supporting_index, dependencies)?;
    }
    for sequence in sequences {
        registry.register_sequence(sequence.oid, sequence.effective_name)?;
    }
    Ok(())
}

#[derive(Default)]
struct GlobalIdentityRegistry {
    objects: BTreeMap<u32, GlobalObjectIdentity>,
    class_names: BTreeMap<(String, String), u32>,
    domain_names: BTreeMap<(String, String), u32>,
    sequence_effective_names: BTreeMap<String, u32>,
    columns: GlobalColumnRegistry,
}

impl GlobalIdentityRegistry {
    fn register_relation(
        &mut self,
        oid: u32,
        schema: &str,
        name: &str,
        schema_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), EngineError> {
        if !valid_relation_oid(oid)
            || schema.is_empty()
            || name.is_empty()
            || zero_digest(schema_digest)
        {
            return Err(codec_error("relation identity is invalid"));
        }
        self.remember_object(
            oid,
            GlobalObjectIdentity::Relation {
                schema: schema.to_owned(),
                name: name.to_owned(),
                schema_digest,
            },
        )?;
        remember_named_oid(
            &mut self.class_names,
            (schema.to_owned(), name.to_owned()),
            oid,
            "dependency qualified-name identity drifted",
        )
    }

    fn register_domain(&mut self, domain: &TypedInsertDomainBinding) -> Result<(), EngineError> {
        if !valid_relation_oid(domain.oid) || domain.schema.is_empty() || domain.name.is_empty() {
            return Err(codec_error("domain identity is invalid"));
        }
        self.register_domain_identity(
            domain.oid,
            domain.base_type,
            Some((domain.schema.to_string(), domain.name.to_string())),
        )?;
        remember_named_oid(
            &mut self.domain_names,
            (domain.schema.to_string(), domain.name.to_string()),
            domain.oid,
            "domain qualified-name identity drifted",
        )
    }

    fn register_index(
        &mut self,
        index: &TypedInsertCanonicalIndexBinding,
        dependencies: &[TypedInsertDependencyBinding],
    ) -> Result<(), EngineError> {
        let owner = dependencies
            .get(index.owner_dependency_ordinal as usize)
            .ok_or_else(|| codec_error("index owner dependency absent"))?;
        let keys = index
            .key_columns
            .iter()
            .map(|column| self.catalog_column_identity(column, dependencies))
            .collect::<Result<Vec<_>, _>>()?;
        for key in keys.iter().cloned() {
            self.register_column(key)?;
        }
        self.remember_object(
            index.oid,
            GlobalObjectIdentity::Index(IndexIdentity {
                owner_relation_oid: owner.oid,
                raw_ordinal: index.raw_ordinal,
                name: index.name.to_string(),
                table_name: index.table_name.to_string(),
                first_column_name: index.first_column_name.to_string(),
                unique: index.unique,
                primary_key: index.primary_key,
                unique_constraint: index.unique_constraint,
                key_columns: keys,
            }),
        )?;
        remember_named_oid(
            &mut self.class_names,
            (owner.schema.to_string(), index.name.to_string()),
            index.oid,
            "qualified class-name identity drifted",
        )
    }

    fn register_catalog_column(
        &mut self,
        column: &TypedInsertCanonicalColumnBinding,
        dependencies: &[TypedInsertDependencyBinding],
    ) -> Result<(), EngineError> {
        self.catalog_column_identity(column, dependencies)
            .and_then(|identity| self.register_column(identity))
    }

    fn catalog_column_identity(
        &self,
        column: &TypedInsertCanonicalColumnBinding,
        dependencies: &[TypedInsertDependencyBinding],
    ) -> Result<ColumnIdentity, EngineError> {
        let relation_oid = dependencies
            .get(column.dependency_ordinal as usize)
            .map(|dependency| dependency.oid)
            .ok_or_else(|| codec_error("catalog column dependency is absent"))?;
        if column.column_id == 0
            || column.attnum <= 0
            || column.catalog_column_ordinal >= column.attnum as u32
            || column.name.is_empty()
            || column.type_oid == 0
            || column.type_size != column.ty.type_size()
        {
            return Err(codec_error("catalog column identity is invalid"));
        }
        Ok(ColumnIdentity {
            relation_oid,
            catalog_column_ordinal: column.catalog_column_ordinal,
            column_id: column.column_id,
            attnum: column.attnum,
            name: column.name.to_string(),
            ty: column.ty,
            type_oid: column.type_oid,
            type_size: column.type_size,
        })
    }

    fn register_column(&mut self, identity: ColumnIdentity) -> Result<(), EngineError> {
        self.register_column_type(identity.ty, identity.type_oid)?;
        self.columns.register(identity)
    }

    fn register_column_type(&mut self, ty: SqlType, type_oid: u32) -> Result<(), EngineError> {
        if type_oid == ty.postgres_oid() {
            return Ok(());
        }
        if gpu_db_sql::SUPPORTED_SQL_TYPES
            .iter()
            .any(|builtin| type_oid == builtin.postgres_oid())
        {
            return Err(codec_error("builtin type OID has the wrong SQL type"));
        }
        self.register_domain_identity(type_oid, ty, None)
    }

    fn register_domain_identity(
        &mut self,
        oid: u32,
        base_type: SqlType,
        named: Option<(String, String)>,
    ) -> Result<(), EngineError> {
        if !valid_relation_oid(oid) {
            return Err(codec_error("domain type OID is invalid"));
        }
        match self.objects.get_mut(&oid) {
            None => {
                self.objects
                    .insert(oid, GlobalObjectIdentity::Domain { base_type, named });
                Ok(())
            }
            Some(GlobalObjectIdentity::Domain {
                base_type: previous_type,
                named: previous_name,
            }) if *previous_type == base_type => match (previous_name.as_ref(), named) {
                (Some(previous), Some(candidate)) if previous != &candidate => {
                    Err(codec_error("domain OID identity drifted"))
                }
                (None, Some(candidate)) => {
                    *previous_name = Some(candidate);
                    Ok(())
                }
                _ => Ok(()),
            },
            _ => Err(codec_error("global catalog OID identity drifted")),
        }
    }

    fn register_sequence(&mut self, oid: u32, effective_name: &str) -> Result<(), EngineError> {
        if !valid_relation_oid(oid) || effective_name.is_empty() {
            return Err(codec_error("sequence descriptor identity is invalid"));
        }
        self.remember_object(
            oid,
            GlobalObjectIdentity::Sequence {
                effective_name: effective_name.to_owned(),
            },
        )?;
        remember_named_oid(
            &mut self.class_names,
            ("public".to_string(), effective_name.to_owned()),
            oid,
            "qualified class-name identity drifted",
        )?;
        remember_named_oid(
            &mut self.sequence_effective_names,
            effective_name.to_owned(),
            oid,
            "sequence effective-name identity drifted",
        )
    }

    fn remember_object(
        &mut self,
        oid: u32,
        candidate: GlobalObjectIdentity,
    ) -> Result<(), EngineError> {
        if let Some(previous) = self.objects.insert(oid, candidate.clone()) {
            if previous != candidate {
                return Err(codec_error("global catalog OID identity drifted"));
            }
        }
        Ok(())
    }
}

fn valid_relation_oid(oid: u32) -> bool {
    (1..=i32::MAX as u32).contains(&oid)
}

fn remember_named_oid<K: Ord>(
    identities: &mut BTreeMap<K, u32>,
    key: K,
    oid: u32,
    message: &'static str,
) -> Result<(), EngineError> {
    if let Some(previous) = identities.insert(key, oid) {
        if previous != oid {
            return Err(codec_error(message));
        }
    }
    Ok(())
}

#[derive(Clone, PartialEq, Eq)]
enum GlobalObjectIdentity {
    Relation {
        schema: String,
        name: String,
        schema_digest: gpu_db_wal::CanonicalDigest,
    },
    Domain {
        base_type: SqlType,
        named: Option<(String, String)>,
    },
    Index(IndexIdentity),
    Sequence {
        effective_name: String,
    },
}

#[derive(Clone, PartialEq, Eq)]
struct IndexIdentity {
    owner_relation_oid: u32,
    raw_ordinal: u32,
    name: String,
    table_name: String,
    first_column_name: String,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
    key_columns: Vec<ColumnIdentity>,
}

#[derive(Clone, PartialEq, Eq)]
struct ColumnIdentity {
    relation_oid: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    attnum: i16,
    name: String,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

#[derive(Default)]
struct GlobalColumnRegistry {
    ids: BTreeMap<u32, ColumnIdentity>,
    ordinals: BTreeMap<(u32, u32), ColumnIdentity>,
    attnums: BTreeMap<(u32, i16), ColumnIdentity>,
    names: BTreeMap<(u32, String), ColumnIdentity>,
}

impl GlobalColumnRegistry {
    fn register(&mut self, identity: ColumnIdentity) -> Result<(), EngineError> {
        remember_column(&mut self.ids, identity.column_id, identity.clone())?;
        remember_column(
            &mut self.ordinals,
            (identity.relation_oid, identity.catalog_column_ordinal),
            identity.clone(),
        )?;
        remember_column(
            &mut self.attnums,
            (identity.relation_oid, identity.attnum),
            identity.clone(),
        )?;
        remember_column(
            &mut self.names,
            (identity.relation_oid, identity.name.clone()),
            identity,
        )
    }
}

fn remember_column<K: Ord>(
    identities: &mut BTreeMap<K, ColumnIdentity>,
    key: K,
    candidate: ColumnIdentity,
) -> Result<(), EngineError> {
    if let Some(previous) = identities.insert(key, candidate.clone()) {
        if previous != candidate {
            return Err(codec_error("global catalog column identity drifted"));
        }
    }
    Ok(())
}

pub(super) fn validate_catalog_closure(
    dependencies: &[TypedInsertDependencyBinding],
    indexes: &[TypedInsertCanonicalIndexBinding],
    foreign_keys: &[TypedInsertCanonicalForeignKeyBinding],
) -> Result<(), EngineError> {
    let mut index_ids = BTreeSet::new();
    let mut index_names = BTreeSet::new();
    let mut all_index_oids = BTreeMap::new();
    let mut columns = CatalogColumnRegistry::default();
    let mut primary_indexes = BTreeMap::new();
    for (ordinal, index) in indexes.iter().enumerate() {
        if index.owner_dependency_ordinal != 0
            || index.raw_ordinal != ordinal as u32
            || !index_ids.insert(index.oid)
            || !index_names.insert(index.name.as_ref())
        {
            return Err(codec_error("target index order or identity drifted"));
        }
        validate_index(index, dependencies, &mut columns)?;
        remember_primary_index(&mut primary_indexes, index)?;
        remember_index_identity(&mut all_index_oids, index.oid, index)?;
    }
    let mut fk_names = BTreeSet::new();
    let mut external_raw = BTreeMap::new();
    let mut external_name = BTreeMap::new();
    for (ordinal, foreign_key) in foreign_keys.iter().enumerate() {
        if foreign_key.raw_ordinal != ordinal as u32
            || !fk_names.insert(foreign_key.name.as_ref())
            || foreign_key.child_column.dependency_ordinal != 0
            || foreign_key.parent_column.dependency_ordinal != foreign_key.parent_dependency_ordinal
        {
            return Err(codec_error(
                "foreign-key order or dependency identity drifted",
            ));
        }
        columns.register(&foreign_key.child_column, dependencies)?;
        columns.register(&foreign_key.parent_column, dependencies)?;
        if foreign_key.child_column.name.as_ref() != foreign_key.child_column_name.as_ref()
            || foreign_key.parent_column.name.as_ref()
                != foreign_key.referenced_column_name.as_ref()
            || dependencies
                .get(foreign_key.parent_dependency_ordinal as usize)
                .map(|dependency| dependency.name.as_ref())
                != Some(foreign_key.referenced_table_name.as_ref())
            || foreign_key.child_column.ty != foreign_key.parent_column.ty
        {
            return Err(codec_error("foreign-key column/type binding drifted"));
        }
        validate_index(&foreign_key.supporting_index, dependencies, &mut columns)?;
        remember_primary_index(&mut primary_indexes, &foreign_key.supporting_index)?;
        if foreign_key.supporting_index.owner_dependency_ordinal
            != foreign_key.parent_dependency_ordinal
            || !foreign_key.supporting_index.unique
            || !(foreign_key.supporting_index.primary_key
                || foreign_key.supporting_index.unique_constraint)
            || foreign_key.supporting_index.key_columns.len() != 1
            || !same_catalog_column_identity(
                &foreign_key.supporting_index.key_columns[0],
                &foreign_key.parent_column,
            )
        {
            return Err(codec_error("foreign-key supporting unique index drifted"));
        }
        remember_index_identity(
            &mut all_index_oids,
            foreign_key.supporting_index.oid,
            &foreign_key.supporting_index,
        )?;
        if foreign_key.parent_dependency_ordinal == 0 {
            if usize::try_from(foreign_key.supporting_index.raw_ordinal)
                .ok()
                .and_then(|ordinal| indexes.get(ordinal))
                .is_none_or(|index| {
                    !same_catalog_index_identity(index, &foreign_key.supporting_index)
                })
            {
                return Err(codec_error(
                    "self-referencing foreign-key supporting index drifted",
                ));
            }
        } else {
            remember_external_index(
                &mut external_raw,
                (
                    foreign_key.parent_dependency_ordinal,
                    foreign_key.supporting_index.raw_ordinal,
                ),
                &foreign_key.supporting_index,
            )?;
            remember_external_index(
                &mut external_name,
                (
                    foreign_key.parent_dependency_ordinal,
                    foreign_key.supporting_index.name.clone(),
                ),
                &foreign_key.supporting_index,
            )?;
        }
    }
    Ok(())
}

fn remember_index_identity<'a>(
    identities: &mut BTreeMap<u32, &'a TypedInsertCanonicalIndexBinding>,
    oid: u32,
    candidate: &'a TypedInsertCanonicalIndexBinding,
) -> Result<(), EngineError> {
    if let Some(previous) = identities.insert(oid, candidate) {
        if !same_catalog_index_identity(previous, candidate) {
            return Err(codec_error("catalog index OID identity drifted"));
        }
    }
    Ok(())
}

fn remember_external_index<'a, K: Ord>(
    identities: &mut BTreeMap<K, &'a TypedInsertCanonicalIndexBinding>,
    key: K,
    candidate: &'a TypedInsertCanonicalIndexBinding,
) -> Result<(), EngineError> {
    if let Some(previous) = identities.insert(key, candidate) {
        if !same_catalog_index_identity(previous, candidate) {
            return Err(codec_error(
                "external foreign-key supporting-index copy drifted",
            ));
        }
    }
    Ok(())
}

fn same_catalog_column_identity(
    left: &TypedInsertCanonicalColumnBinding,
    right: &TypedInsertCanonicalColumnBinding,
) -> bool {
    left.dependency_ordinal == right.dependency_ordinal
        && left.catalog_column_ordinal == right.catalog_column_ordinal
        && left.column_id == right.column_id
        && left.attnum == right.attnum
        && left.name == right.name
        && left.ty == right.ty
        && left.type_oid == right.type_oid
        && left.type_size == right.type_size
}

#[derive(Default)]
struct CatalogColumnRegistry<'a> {
    ids: BTreeMap<u32, &'a TypedInsertCanonicalColumnBinding>,
    ordinals: BTreeMap<(u32, u32), &'a TypedInsertCanonicalColumnBinding>,
    attnums: BTreeMap<(u32, i16), &'a TypedInsertCanonicalColumnBinding>,
    names: BTreeMap<(u32, Arc<str>), &'a TypedInsertCanonicalColumnBinding>,
}

impl<'a> CatalogColumnRegistry<'a> {
    fn register(
        &mut self,
        column: &'a TypedInsertCanonicalColumnBinding,
        dependencies: &[TypedInsertDependencyBinding],
    ) -> Result<(), EngineError> {
        validate_catalog_column(column, dependencies)?;
        remember_catalog_column(&mut self.ids, column.column_id, column)?;
        remember_catalog_column(
            &mut self.ordinals,
            (column.dependency_ordinal, column.catalog_column_ordinal),
            column,
        )?;
        remember_catalog_column(
            &mut self.attnums,
            (column.dependency_ordinal, column.attnum),
            column,
        )?;
        remember_catalog_column(
            &mut self.names,
            (column.dependency_ordinal, column.name.clone()),
            column,
        )
    }
}

fn remember_catalog_column<'a, K: Ord>(
    identities: &mut BTreeMap<K, &'a TypedInsertCanonicalColumnBinding>,
    key: K,
    candidate: &'a TypedInsertCanonicalColumnBinding,
) -> Result<(), EngineError> {
    if let Some(previous) = identities.insert(key, candidate) {
        if !same_catalog_column_identity(previous, candidate) {
            return Err(codec_error("catalog column identity drifted"));
        }
    }
    Ok(())
}

fn same_catalog_index_identity(
    left: &TypedInsertCanonicalIndexBinding,
    right: &TypedInsertCanonicalIndexBinding,
) -> bool {
    left.owner_dependency_ordinal == right.owner_dependency_ordinal
        && left.raw_ordinal == right.raw_ordinal
        && left.oid == right.oid
        && left.name == right.name
        && left.table_name == right.table_name
        && left.first_column_name == right.first_column_name
        && left.unique == right.unique
        && left.primary_key == right.primary_key
        && left.unique_constraint == right.unique_constraint
        && left.key_columns.len() == right.key_columns.len()
        && left
            .key_columns
            .iter()
            .zip(&right.key_columns)
            .all(|(left, right)| same_catalog_column_identity(left, right))
}

fn remember_primary_index<'a>(
    identities: &mut BTreeMap<u32, &'a TypedInsertCanonicalIndexBinding>,
    candidate: &'a TypedInsertCanonicalIndexBinding,
) -> Result<(), EngineError> {
    if candidate.primary_key {
        if let Some(previous) = identities.insert(candidate.owner_dependency_ordinal, candidate) {
            if !same_catalog_index_identity(previous, candidate) {
                return Err(codec_error("relation has conflicting primary-key indexes"));
            }
        }
    }
    Ok(())
}

fn validate_index<'a>(
    index: &'a TypedInsertCanonicalIndexBinding,
    dependencies: &[TypedInsertDependencyBinding],
    columns: &mut CatalogColumnRegistry<'a>,
) -> Result<(), EngineError> {
    let owner = dependencies
        .get(index.owner_dependency_ordinal as usize)
        .ok_or_else(|| codec_error("index owner dependency absent"))?;
    if !valid_relation_oid(index.oid)
        || index.name.is_empty()
        || index.table_name.as_ref() != owner.name.as_ref()
        || index.first_column_name.is_empty()
        || !(1..=32).contains(&index.key_columns.len())
        || index.key_columns[0].name.as_ref() != index.first_column_name.as_ref()
        || ((index.primary_key || index.unique_constraint) && !index.unique)
        || (index.primary_key && index.unique_constraint)
    {
        return Err(codec_error("index metadata is invalid"));
    }
    let mut key_ids = BTreeSet::new();
    for column in &index.key_columns {
        if column.dependency_ordinal != index.owner_dependency_ordinal {
            return Err(codec_error("index key column owner dependency drifted"));
        }
        if !key_ids.insert(column.column_id) {
            return Err(codec_error("index repeats a key-column identity"));
        }
        columns.register(column, dependencies)?;
    }
    Ok(())
}

fn validate_catalog_column(
    column: &TypedInsertCanonicalColumnBinding,
    dependencies: &[TypedInsertDependencyBinding],
) -> Result<(), EngineError> {
    if column.column_id == 0
        || column.name.is_empty()
        || column.type_oid == 0
        || column.type_size != column.ty.type_size()
        || dependencies
            .get(column.dependency_ordinal as usize)
            .is_none()
    {
        return Err(codec_error("resolved catalog column identity is invalid"));
    }
    Ok(())
}
