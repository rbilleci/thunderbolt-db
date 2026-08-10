//! Dependency-token/guard-key closure independent of catalog witnesses.

use super::error;
use super::statements::{append_record_value, begin, exact, range, storage};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::ReservedSemanticsV2Graph;
use crate::EngineError;
use sha2::Digest;
use std::cmp::Ordering;

const ABSENT_U32: u32 = u32::MAX;
const TARGET_TABLE: u8 = 1;
const FOREIGN_PARENT_TABLE: u8 = 2;
const MAINTAINED_INDEX: u8 = 3;
const UNIQUE_KEY_GUARD: u8 = 4;
const FOREIGN_PARENT_INDEX: u8 = 5;
const FOREIGN_KEY_GUARD: u8 = 6;
const DOMAIN: u8 = 7;
const PUBLISHED_SEQUENCE: u8 = 8;
const NOT_NULL_GUARD: u8 = 9;
const CHECK_GUARD: u8 = 10;
const DOMAIN_CONSTRAINT_GUARD: u8 = 11;

const TARGET_TABLE_ROLE: u16 = 1;
const MAINTAINED_INDEX_ROLE: u16 = 2;
const UNIQUE_KEY_ROLE: u16 = 3;
const FOREIGN_PARENT_TABLE_ROLE: u16 = 4;
const FOREIGN_PARENT_INDEX_ROLE: u16 = 5;
const FOREIGN_KEY_ROLE: u16 = 6;
const DOMAIN_ROLE: u16 = 7;
const PUBLISHED_SEQUENCE_ROLE: u16 = 8;
const NOT_NULL_GUARD_ROLE: u16 = 9;
const CHECK_GUARD_ROLE: u16 = 10;
const DOMAIN_CONSTRAINT_GUARD_ROLE: u16 = 11;

pub(super) fn validate(
    graph: &ReservedSemanticsV2Graph,
    catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
) -> Result<(), EngineError> {
    validate_table_targets(graph, catalog_composition)?;
    validate_tokens(graph)?;
    validate_uses(graph)?;
    validate_terminal_binding(graph)
}

fn validate_tokens(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let mut terminal_count = 0_u32;
    for (ordinal, token) in graph.dependencies.iter().enumerate() {
        let descriptor = descriptor_for(graph, token)?;
        let effect = effect_for(graph, token)?;
        // A zero root/generation is admissible only for a transaction-created table or an index
        // owned by that table. The existing S7 table block and descriptor jointly prove that
        // fact; no recovery-time catalog convention is introduced.
        let target_is_initially_absent = token.kind == TARGET_TABLE
            && graph
                .tables
                .get(token.target_table_ref as usize)
                .is_some_and(|table| table.initial_table_absent);
        let index_is_initially_absent = matches!(token.kind, MAINTAINED_INDEX | UNIQUE_KEY_GUARD)
            && descriptor.is_some_and(|index| {
                index.owner_table_ref == token.target_table_ref
                    && ((index.base_index_generation == 0 && index.base_index_root == [0; 32])
                        || graph
                            .tables
                            .get(index.owner_table_ref as usize)
                            .is_some_and(|table| table.initial_table_absent))
            });
        // The catalog-only S3 record proves a transaction-private domain's creation. When the
        // target table is likewise transaction-created it has no published catalog predecessor,
        // so its domain dependency may retain the same zero snapshot generation.
        let domain_is_initially_absent = token.kind == DOMAIN
            && graph
                .tables
                .get(token.target_table_ref as usize)
                .is_some_and(|table| table.initial_table_absent);
        let absent_predecessor =
            target_is_initially_absent || index_is_initially_absent || domain_is_initially_absent;
        if token.dependency_ref != ordinal as u32
            || token.access != required_access(token.kind)
            || token.flags & !3 != 0
            || token.flags & 3 == 3
            || token.stable_object_id == 0
            || token.stable_object_id == u64::MAX
            || (token.display_oid == 0
                && !matches!(token.kind, NOT_NULL_GUARD | DOMAIN_CONSTRAINT_GUARD))
            || token.display_oid > 0x7fff_ffff
            || token.target_table_ref as usize >= graph.tables.len()
            || (token.base_generation == 0 && !absent_predecessor)
            || token.base_generation == u64::MAX
            || token.catalog_epoch != graph.header.catalog_before_epoch
            || (token.kind == PUBLISHED_SEQUENCE && token.snapshot_floor != 0)
            || (token.kind != PUBLISHED_SEQUENCE
                && !absent_predecessor
                && token.snapshot_floor == 0)
            || token.schema_digest == [0; 32] && token.kind != PUBLISHED_SEQUENCE
            || token.schema_digest != [0; 32] && token.kind == PUBLISHED_SEQUENCE
            || (token.base_root == [0; 32] && token.kind != DOMAIN && !absent_predecessor)
            || (token.base_root != [0; 32] && token.kind == DOMAIN)
            || token.name_digest == [0; 32]
            || token.identity_digest == [0; 32]
            || token.token_digest == [0; 32]
            || (matches!(token.kind, MAINTAINED_INDEX..=FOREIGN_KEY_GUARD)) != descriptor.is_some()
            || (!matches!(token.kind, MAINTAINED_INDEX..=FOREIGN_KEY_GUARD)
                && token.descriptor_ref != ABSENT_U32)
            || (token.flags & 1 != 0
                && (!matches!(token.kind, UNIQUE_KEY_GUARD | FOREIGN_KEY_GUARD)
                    || effect.is_none()))
            || (token.flags & 1 == 0 && token.key_effect_ref != ABSENT_U32)
            || (matches!(token.kind, UNIQUE_KEY_GUARD | FOREIGN_KEY_GUARD) && token.flags & 3 == 0)
            || (token.flags & 2 != 0
                && !matches!(
                    token.kind,
                    UNIQUE_KEY_GUARD
                        | FOREIGN_KEY_GUARD
                        | NOT_NULL_GUARD
                        | CHECK_GUARD
                        | DOMAIN_CONSTRAINT_GUARD
                ))
        {
            return Err(error(
                "dependency token scalar/type-state closure is invalid",
            ));
        }
        if let Some(index) = descriptor {
            // An FK may reference a parent table created earlier in this same transaction.  That
            // parent has one S7 index descriptor (the table's own descriptor); its absent base
            // is not a usable provider.  The FK token therefore pins the descriptor's already
            // authenticated final index successor.  This is still the exact same descriptor and
            // stable identity, not a second descriptor or a recovery-only FK path.
            let initial_parent_successor =
                matches!(token.kind, FOREIGN_PARENT_INDEX | FOREIGN_KEY_GUARD)
                    && graph
                        .tables
                        .get(index.owner_table_ref as usize)
                        .is_some_and(|table| table.initial_table_absent);
            let expected_index_generation = if initial_parent_successor {
                index.final_index_generation
            } else {
                index.base_index_generation
            };
            let expected_index_root = if initial_parent_successor {
                index.final_index_root
            } else {
                index.base_index_root
            };
            if token.stable_object_id != index.stable_index_id
                || token.display_oid != index.display_oid
                || token.catalog_epoch != index.catalog_epoch
                || token.base_generation != expected_index_generation
                || token.schema_digest != index.owner_schema_digest
                || token.base_root != expected_index_root
                || token.name_digest != index.index_name_digest
            {
                return Err(error(
                    "index dependency does not use its descriptor's exact stable identity",
                ));
            }
        }
        if let Some(effect) = effect {
            if effect.dependency_ref != token.dependency_ref
                || !matches!(token.kind, UNIQUE_KEY_GUARD | FOREIGN_KEY_GUARD)
            {
                return Err(error("live guard token does not own its key effect"));
            }
        }
        terminal_count = terminal_count
            .checked_add(u32::from(token.flags & 2 != 0))
            .ok_or_else(|| error("terminal dependency count overflows"))?;
        let effect_digest = effect.map_or([0; 32], |effect| effect.effect_digest);
        let descriptor_digest = descriptor.map_or([0; 32], |index| index.descriptor_digest);
        let identity = match token.kind {
            1 | 2 => exact(
                b"gpu-db/write001/s7-table-object/v2",
                &[
                    &[token.kind],
                    &token.stable_object_id.to_le_bytes(),
                    &token.display_oid.to_le_bytes(),
                    &token.catalog_epoch.to_le_bytes(),
                    &token.base_generation.to_le_bytes(),
                    &token.schema_digest,
                    &token.base_root,
                    &token.name_digest,
                ],
            ),
            3..=6 => exact(
                b"gpu-db/write001/s7-index-object/v2",
                &[
                    &[token.kind],
                    &token.stable_object_id.to_le_bytes(),
                    &token.display_oid.to_le_bytes(),
                    &token.catalog_epoch.to_le_bytes(),
                    &token.base_generation.to_le_bytes(),
                    &token.schema_digest,
                    &token.base_root,
                    &token.name_digest,
                    &descriptor_digest,
                    &effect_digest,
                ],
            ),
            7 => exact(
                b"gpu-db/write001/s7-domain-object/v2",
                &[
                    &token.stable_object_id.to_le_bytes(),
                    &token.display_oid.to_le_bytes(),
                    &token.catalog_epoch.to_le_bytes(),
                    &token.base_generation.to_le_bytes(),
                    &token.schema_digest,
                    &token.name_digest,
                ],
            ),
            9..=11 => {
                let table = graph
                    .tables
                    .get(token.target_table_ref as usize)
                    .ok_or_else(|| error("constraint token table is absent"))?;
                exact(
                    b"gpu-db/write001/s7-constraint-object/v2",
                    &[
                        &[token.kind],
                        &token.stable_object_id.to_le_bytes(),
                        &token.display_oid.to_le_bytes(),
                        &table.stable_table_id.to_le_bytes(),
                        &token.catalog_epoch.to_le_bytes(),
                        &token.base_generation.to_le_bytes(),
                        &token.schema_digest,
                        &token.base_root,
                        &token.name_digest,
                    ],
                )
            }
            8 => published_sequence_identity(graph, token)?,
            _ => return Err(error("dependency kind is invalid")),
        };
        if identity != token.identity_digest {
            return Err(error("dependency runtime guard key is invalid"));
        }
        let token_digest = token_digest(token, descriptor_digest, effect_digest);
        if token_digest != token.token_digest {
            return Err(error("dependency token digest is invalid"));
        }
        if runtime_guard_key(graph, token)? == [0; 32] {
            return Err(error("runtime guard key unexpectedly has the zero digest"));
        }
    }
    if terminal_count > 1 {
        return Err(error("codec closure has more than one terminal dependency"));
    }
    for window in graph.dependencies.windows(2) {
        if dependency_order(&window[0], &window[1]) != Ordering::Less {
            return Err(error(
                "dependency directory is not strictly identity ordered",
            ));
        }
    }
    Ok(())
}

fn validate_table_targets(
    graph: &ReservedSemanticsV2Graph,
    catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
) -> Result<(), EngineError> {
    for (table_ref, table) in graph.tables.iter().enumerate() {
        let token = graph
            .dependencies
            .get(table.target_dependency_ref as usize)
            .ok_or_else(|| error("table target dependency is absent"))?;
        if token.kind != TARGET_TABLE
            || token.target_table_ref != table_ref as u32
            || token.stable_object_id != table.stable_table_id
            || token.display_oid != table.display_oid
            || token.catalog_epoch != table.catalog_epoch
            || token.base_generation != table.data_generation_before
            || token.schema_digest != table.schema_digest
            || token.base_root != table.initial_table_root
        {
            return Err(error("target-table token does not close its table block"));
        }
        let mut source_count = 0_u32;
        let mut prior_record: Option<&crate::typed_insert_batch::DecodedTypedInsertRecord> = None;
        let mut final_schema_digest = None;
        for resolution in graph
            .resolutions
            .iter()
            .filter(|resolution| resolution.table_ref == table_ref as u32)
        {
            let record = graph
                .records
                .get(resolution.record_ref as usize)
                .ok_or_else(|| error("target-table S2 record is absent"))?;
            let target = record.target_identity();
            if target.oid != table.display_oid
                || token.name_digest != qualified_name_digest(target.schema, target.name)
            {
                return Err(error("table block and resolved S2 target diverge"));
            }
            if let Some(prior) = prior_record {
                if prior.target_identity().schema_digest != target.schema_digest
                    && !same_oid_sequence_name_transition(prior, record)
                    && !same_oid_s3_created_index_transition(graph, table_ref as u32, prior, record)
                {
                    return Err(error(
                        "S2 target schema changed without an ordered stable-OID sequence rename witness",
                    ));
                }
            }
            prior_record = Some(record);
            final_schema_digest = Some(target.schema_digest);
            source_count = source_count
                .checked_add(1)
                .ok_or_else(|| error("target-table source count overflows"))?;
        }
        if source_count == 0 {
            return Err(error("table block has no resolved S2 target"));
        }
        if final_schema_digest != Some(table.schema_digest)
            && !terminal_s3_sequence_rename_closes_table_schema(
                table,
                prior_record,
                catalog_composition,
            )
        {
            return Err(error(
                "table block does not close the final statement-time S2 target",
            ));
        }
        for other in graph.tables.iter().take(table_ref) {
            if other.target_dependency_ref == table.target_dependency_ref
                || other.stable_table_id == table.stable_table_id
                || other.display_oid == table.display_oid
            {
                return Err(error(
                    "table blocks are not a bijection over S2 target identities",
                ));
            }
        }
    }
    Ok(())
}

/// A terminal S3 sequence rename changes a dependent table's schema after the last S2 INSERT.
/// It can close that delta only with one exact old-name S2 private effect, stable sequence OID,
/// and before/after default-column table digests. The S3 record remains the sole catalog owner.
pub(super) fn terminal_s3_sequence_rename_closes_table_schema(
    table: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedTable,
    final_record: Option<&crate::typed_insert_batch::DecodedTypedInsertRecord>,
    catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
) -> bool {
    let (Some(final_record), Some(composition)) = (final_record, catalog_composition) else {
        return false;
    };
    let final_s2_schema = final_record.target_identity().schema_digest;
    if final_s2_schema == table.schema_digest {
        return false;
    }
    final_record.sequence_bindings().any(|binding| {
        let private_effect = final_record.sequence_effects().any(|effect| {
            effect.request.effect_ordinal == binding.effect_ordinal
                && matches!(
                    effect.kind,
                    crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Private { .. }
                )
        });
        if !private_effect {
            return false;
        }
        let mut matched_ordinal = None;
        for operation in &composition.sequence_lifecycle_operations {
            let is_exact_rename = composition
                .catalog_commands
                .get(operation.command_index as usize)
                .is_some_and(|command| {
                    command.ordinal == operation.ordinal
                        && matches!(
                            &command.command,
                            gpu_db_sql::Command::RenameSequence(rename)
                                if rename.old_name == binding.effective_name
                                    && rename.new_name != binding.effective_name
                        )
                });
            if !is_exact_rename {
                continue;
            }
            for target in &operation.targets {
                let closes_table_schema = target.before_name == binding.effective_name
                    && target
                        .after_name
                        .as_deref()
                        .is_some_and(|after| !after.is_empty() && after != binding.effective_name)
                    && target
                        .target_before
                        .as_ref()
                        .is_some_and(|before| before.oid == binding.request.sequence_oid)
                    && target
                        .target_after
                        .as_ref()
                        .is_some_and(|after| after.oid == binding.request.sequence_oid)
                    && target.dependencies_before.iter().any(|(key, before)| {
                        target.dependencies_after.get(key).is_some_and(|after| {
                            before.column_id == after.column_id
                                && before.table.oid == table.display_oid
                                && before.table.digest == final_s2_schema
                                && after.table.oid == table.display_oid
                                && after.table.digest == table.schema_digest
                        })
                    });
                if closes_table_schema && matched_ordinal.replace(operation.ordinal).is_some() {
                    return false;
                }
            }
        }
        let Some(matched_ordinal) = matched_ordinal else {
            return false;
        };
        !composition
            .sequence_lifecycle_operations
            .iter()
            .any(|operation| {
                operation.ordinal > matched_ordinal
                    && operation.targets.iter().any(|target| {
                        target
                            .target_before
                            .as_ref()
                            .is_some_and(|before| before.oid == binding.request.sequence_oid)
                            || target
                                .target_after
                                .as_ref()
                                .is_some_and(|after| after.oid == binding.request.sequence_oid)
                    })
            })
    })
}

fn same_oid_sequence_name_transition(
    before: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    after: &crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> bool {
    let before = before
        .sequence_bindings()
        .map(|binding| (binding.request.sequence_oid, binding.effective_name))
        .collect::<std::collections::BTreeMap<_, _>>();
    after.sequence_bindings().any(|binding| {
        before
            .get(&binding.request.sequence_oid)
            .is_some_and(|name| *name != binding.effective_name)
    })
}

/// An existing table's ordered S3 CREATE INDEX changes the statement-time target schema digest
/// between adjacent S2 records.  The paired-zero S7 descriptor is later bound to its exact S3
/// lifecycle identity; here we only recognize the corresponding absent→present S2 directory
/// transition, preserving the one statement-time catalog authority.
fn same_oid_s3_created_index_transition(
    graph: &ReservedSemanticsV2Graph,
    table_ref: u32,
    before: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    after: &crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> bool {
    graph.indexes.iter().any(|descriptor| {
        descriptor.owner_table_ref == table_ref
            && descriptor.base_index_generation == 0
            && descriptor.base_index_root == [0; 32]
            && !before
                .indexes()
                .any(|source| source.oid == descriptor.display_oid)
            && after
                .indexes()
                .any(|source| source.oid == descriptor.display_oid)
    })
}

fn validate_uses(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let mut next = 0_u32;
    for resolution in &graph.resolutions {
        let uses = range(
            &graph.dependency_uses,
            resolution.dependency_use_start,
            resolution.dependency_use_count,
            "dependency use",
        )?;
        if resolution.dependency_use_start != next {
            return Err(error("statement dependency-use ranges are not dense"));
        }
        for usage in uses {
            let token = graph
                .dependencies
                .get(usage.dependency_ref as usize)
                .ok_or_else(|| error("dependency use token is absent"))?;
            if usage.statement_ordinal != resolution.statement_ordinal
                || token.snapshot_floor > resolution.dependency_validation_floor
                || expected_kind_for_role(usage.role) != Some(token.kind)
                || !valid_use_shape(usage.role, usage.transition_ref, usage.key_effect_ref)
            {
                return Err(error(
                    "dependency use/floor/role is not closed by statement",
                ));
            }
            validate_use_source(graph, resolution, usage, token)?;
        }
        validate_static_source_inventory(graph, resolution, uses)?;
        next = next
            .checked_add(resolution.dependency_use_count)
            .ok_or_else(|| error("dependency-use range count overflows"))?;
    }
    if next as usize != graph.dependency_uses.len() {
        return Err(error(
            "dependency-use ranges do not exhaust their directory",
        ));
    }
    for window in graph.dependency_uses.windows(2) {
        if use_order(&window[0], &window[1]) != Ordering::Less {
            return Err(error(
                "dependency-use directory is not strictly role/source ordered",
            ));
        }
    }
    for token in &graph.dependencies {
        let used = graph
            .dependency_uses
            .iter()
            .any(|usage| usage.dependency_ref == token.dependency_ref);
        let target = graph
            .tables
            .iter()
            .any(|table| table.target_dependency_ref == token.dependency_ref);
        let effect = graph
            .key_effects
            .iter()
            .any(|effect| effect.dependency_ref == token.dependency_ref);
        if !used && !target && !effect {
            return Err(error(
                "dependency token is not reachable from any retained source",
            ));
        }
        if token.kind != PUBLISHED_SEQUENCE {
            let mut minimum: Option<u64> = None;
            for resolution in &graph.resolutions {
                let direct_use = graph.dependency_uses.iter().any(|usage| {
                    usage.statement_ordinal == resolution.statement_ordinal
                        && usage.dependency_ref == token.dependency_ref
                });
                if direct_use {
                    minimum = Some(match minimum {
                        Some(current) => current.min(resolution.dependency_validation_floor),
                        None => resolution.dependency_validation_floor,
                    });
                }
            }
            if minimum != Some(token.snapshot_floor) {
                return Err(error(
                    "dependency token floor is not the exact minimum of its retained uses",
                ));
            }
        }
    }
    validate_effect_use_bijections(graph)
}

/// Static sources are part of every resolved statement, even when no row survives.  In
/// particular, a table token cannot become reachable merely because an S7 table block names it:
/// the resolution must retain its own exact role-1 use.
fn validate_static_source_inventory(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
    uses: &[crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse],
) -> Result<(), EngineError> {
    let record = graph
        .records
        .get(resolution.record_ref as usize)
        .ok_or_else(|| error("static dependency-use S2 record is absent"))?;
    require_one_static_use(uses, TARGET_TABLE_ROLE, 0)?;
    for index in record
        .indexes()
        .filter(|index| index.owner_dependency_ordinal == 0)
    {
        require_one_static_use(uses, MAINTAINED_INDEX_ROLE, index.raw_ordinal)?;
    }
    for foreign_key in record.foreign_keys() {
        require_one_static_use(
            uses,
            FOREIGN_PARENT_TABLE_ROLE,
            foreign_key.parent_dependency_ordinal,
        )?;
        require_one_static_use(uses, FOREIGN_PARENT_INDEX_ROLE, foreign_key.raw_ordinal)?;
    }
    for domain in record.domains() {
        require_one_static_use(uses, DOMAIN_ROLE, domain.ordinal)?;
    }
    for sequence in record.sequence_effects() {
        if matches!(
            sequence.kind,
            crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Published { .. }
        ) {
            require_one_role_source_use(
                uses,
                PUBLISHED_SEQUENCE_ROLE,
                sequence.request.effect_ordinal,
            )?;
        }
    }
    Ok(())
}

fn require_one_static_use(
    uses: &[crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse],
    role: u16,
    source_ordinal: u32,
) -> Result<(), EngineError> {
    let usage = require_one_role_source_use(uses, role, source_ordinal)?;
    if usage.transition_ref != ABSENT_U32 || usage.key_effect_ref != ABSENT_U32 {
        return Err(error(
            "static S2 source has a dynamic dependency-use reference",
        ));
    }
    Ok(())
}

fn require_one_role_source_use(
    uses: &[crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse],
    role: u16,
    source_ordinal: u32,
) -> Result<
    &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse,
    EngineError,
> {
    let mut candidates = uses
        .iter()
        .filter(|usage| usage.role == role && usage.source_ordinal == source_ordinal);
    let usage = candidates
        .next()
        .ok_or_else(|| error("static S2 source has no dependency use"))?;
    if candidates.next().is_some() {
        return Err(error("static S2 source has duplicate dependency uses"));
    }
    Ok(usage)
}

fn validate_use_source(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
    usage: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<(), EngineError> {
    let record = graph
        .records
        .get(resolution.record_ref as usize)
        .ok_or_else(|| error("dependency-use S2 record is absent"))?;
    if token.target_table_ref != resolution.table_ref {
        return Err(error(
            "dependency use token does not retain its statement target-table context",
        ));
    }
    match usage.role {
        TARGET_TABLE_ROLE => {
            let table = graph
                .tables
                .get(resolution.table_ref as usize)
                .ok_or_else(|| error("target dependency-use table is absent"))?;
            if token.dependency_ref != table.target_dependency_ref || usage.source_ordinal != 0 {
                return Err(error("target-table use does not name its resolved table"));
            }
        }
        MAINTAINED_INDEX_ROLE | UNIQUE_KEY_ROLE => {
            let descriptor = descriptor_for(graph, token)?
                .ok_or_else(|| error("target-index use has no descriptor"))?;
            let Some(source) = record
                .indexes()
                .find(|source| source.raw_ordinal == usage.source_ordinal)
            else {
                // Pre-CREATE statements intentionally cannot name the new index. Its paired
                // zero base is admitted only after the retained S3 closure has authenticated
                // the terminal index source, so this is not a second dependency authority.
                if descriptor.base_index_generation == 0 && descriptor.base_index_root == [0; 32] {
                    return Ok(());
                }
                return Err(error("target-index use has no S2 index source"));
            };
            let paired_zero_s3_created_at_reused_ordinal = descriptor.base_index_generation == 0
                && descriptor.base_index_root == [0; 32]
                && descriptor.display_oid != source.oid;
            if paired_zero_s3_created_at_reused_ordinal {
                // An ordered DROP/CREATE may reuse the retiring index's final catalog ordinal.
                // The old S2 source must not be mistaken for the new paired-zero descriptor;
                // its identity is instead closed by the existing retained S3 CREATE proof.
                return Ok(());
            }
            if !descriptor_matches_s2_index(graph, record, descriptor, source)?
                || !descriptor_keys_match_s2_index(graph, record, descriptor, source)?
                || (usage.role == MAINTAINED_INDEX_ROLE && descriptor.flags & 8 == 0)
                || (usage.role == UNIQUE_KEY_ROLE && descriptor.flags & 1 == 0)
            {
                return Err(error("target-index use does not close its S2 source"));
            }
            validate_equality_use(graph, usage, token)?;
        }
        FOREIGN_PARENT_TABLE_ROLE => {
            let source = record
                .dependencies()
                .nth(usage.source_ordinal as usize)
                .ok_or_else(|| error("foreign-parent use has no S2 dependency"))?;
            if source.oid != token.display_oid
                || source.schema_digest != token.schema_digest
                || token.name_digest != qualified_name_digest(source.schema, source.name)
            {
                return Err(error(
                    "foreign-parent token does not equal its S2 dependency",
                ));
            }
        }
        FOREIGN_PARENT_INDEX_ROLE | FOREIGN_KEY_ROLE => {
            let foreign_key = record
                .foreign_keys()
                .find(|foreign_key| foreign_key.raw_ordinal == usage.source_ordinal)
                .ok_or_else(|| error("foreign-index use has no S2 FK source"))?;
            let descriptor = descriptor_for(graph, token)?
                .ok_or_else(|| error("foreign-index use has no descriptor"))?;
            if !descriptor_matches_foreign_key(graph, record, descriptor, foreign_key)?
                || !descriptor_keys_match_foreign_key(graph, record, descriptor, foreign_key)?
                || (usage.role == FOREIGN_KEY_ROLE && token.kind != FOREIGN_KEY_GUARD)
                || (usage.role == FOREIGN_PARENT_INDEX_ROLE && token.kind != FOREIGN_PARENT_INDEX)
            {
                return Err(error(
                    "foreign-index token does not close its exact S2 FK source",
                ));
            }
            // A transaction-created parent is represented by the table's one initial S7
            // descriptor, whose owner fields correctly retain the absent predecessor.  Its child
            // FK, however, can only be satisfied by the parent's final table/index successor.
            // Compare that FK parent use to the existing table block's final commitment; published
            // parents retain the descriptor's ordinary base commitment.
            let initial_parent = graph
                .tables
                .get(descriptor.owner_table_ref as usize)
                .filter(|table| table.initial_table_absent);
            let expected_parent_generation = initial_parent
                .map(|table| table.data_generation_after)
                .unwrap_or(descriptor.owner_data_generation);
            let expected_parent_root = initial_parent
                .map(|table| table.final_table_root)
                .unwrap_or(descriptor.owner_table_base_root);
            let parent_matches = graph
                .dependency_uses
                .iter()
                .filter(|candidate| {
                    candidate.statement_ordinal == usage.statement_ordinal
                        && candidate.role == FOREIGN_PARENT_TABLE_ROLE
                        && candidate.source_ordinal == foreign_key.parent_dependency_ordinal
                        && graph
                            .dependencies
                            .get(candidate.dependency_ref as usize)
                            .is_some_and(|parent| {
                                parent.kind == FOREIGN_PARENT_TABLE
                                    && parent.stable_object_id == descriptor.owner_stable_table_id
                                    && parent.display_oid == descriptor.owner_display_oid
                                    && parent.schema_digest == descriptor.owner_schema_digest
                                    && parent.base_generation == expected_parent_generation
                                    && parent.base_root == expected_parent_root
                                    && parent.name_digest == descriptor.owner_name_digest
                            })
                })
                .count();
            if parent_matches != 1 {
                return Err(error(
                    "FK descriptor and parent token do not bind the same S2 dependency",
                ));
            }
            validate_equality_use(graph, usage, token)?;
        }
        DOMAIN_ROLE => {
            let source = record
                .domains()
                .nth(usage.source_ordinal as usize)
                .ok_or_else(|| error("domain use has no S2 domain"))?;
            if source.oid != token.display_oid
                || token.schema_digest != domain_shape_digest(source.base_type)
                || token.name_digest != qualified_name_digest(source.schema, source.name)
            {
                return Err(error("domain token does not equal its S2 domain source"));
            }
        }
        PUBLISHED_SEQUENCE_ROLE => {
            let source = graph
                .sequence_effects
                .iter()
                .filter(|effect| {
                    effect.reference.as_ref().is_some_and(|reference| {
                        effect.statement_ordinal == usage.statement_ordinal
                            && effect.effect_ordinal == usage.source_ordinal
                            && reference.transition_txn_id == token.base_generation
                            && reference.sequence_oid == token.display_oid
                            && effect.reference_body_digest == token.base_root
                    })
                })
                .count();
            if source != 1 {
                return Err(error(
                    "published-sequence use does not select one S5 source",
                ));
            }
            let mut bindings = record
                .sequence_bindings()
                .filter(|binding| binding.effect_ordinal == usage.source_ordinal);
            let binding = bindings.next().ok_or_else(|| {
                error("published-sequence use has no effective S2 sequence binding")
            })?;
            if bindings.next().is_some()
                || token.name_digest
                    != qualified_name_digest(
                        record.target_identity().schema,
                        binding.effective_name,
                    )
            {
                return Err(error(
                    "published-sequence token name does not equal its effective S2 binding",
                ));
            }
        }
        NOT_NULL_GUARD_ROLE => {
            let table = graph
                .tables
                .get(resolution.table_ref as usize)
                .ok_or_else(|| error("NOT NULL dependency-use table is absent"))?;
            let column = record
                .catalog_columns()
                .nth(usage.source_ordinal as usize)
                .ok_or_else(|| error("NOT NULL dependency use has no S2 catalog column source"))?;
            if column.catalog_column_ordinal != usage.source_ordinal
                || token.name_digest
                    != synthesized_not_null_name_digest(
                        1,
                        table.stable_table_id,
                        usage.source_ordinal,
                    )
                || (token.flags & 2 != 0
                    && usage.source_ordinal != resolution.terminal_source_ordinal)
            {
                return Err(error(
                    "NOT NULL dependency use does not close its S2 column source",
                ));
            }
        }
        CHECK_GUARD_ROLE | DOMAIN_CONSTRAINT_GUARD_ROLE => {
            if usage.source_ordinal != resolution.terminal_source_ordinal && token.flags & 2 != 0 {
                return Err(error(
                    "terminal guard use does not echo its terminal source",
                ));
            }
        }
        _ => return Err(error("dependency use role is invalid")),
    }
    Ok(())
}

fn validate_effect_use_bijections(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    for effect in &graph.key_effects {
        let expected_kind = match effect.role {
            1 => MAINTAINED_INDEX,
            2 => UNIQUE_KEY_GUARD,
            3 => FOREIGN_KEY_GUARD,
            _ => return Err(error("key effect role is invalid for dependency closure")),
        };
        if !effect.participates {
            if effect.dependency_ref != ABSENT_U32 {
                return Err(error(
                    "NULL-skipped guard effect retains a dependency token",
                ));
            }
            continue;
        }
        let token = graph
            .dependencies
            .get(effect.dependency_ref as usize)
            .ok_or_else(|| error("participating effect dependency token is absent"))?;
        if token.kind != expected_kind
            || token.descriptor_ref != effect.index_ref
            || (effect.role == 1 && token.key_effect_ref != ABSENT_U32)
            || (effect.role != 1
                && (token.flags & 1 == 0 || token.key_effect_ref != effect.effect_ref))
        {
            return Err(error(
                "key effect and dependency token are not mutually bound",
            ));
        }
        if effect.role != 1 {
            let role = if effect.role == 2 {
                UNIQUE_KEY_ROLE
            } else {
                FOREIGN_KEY_ROLE
            };
            let count = graph
                .dependency_uses
                .iter()
                .filter(|usage| {
                    usage.role == role
                        && usage.dependency_ref == token.dependency_ref
                        && usage.transition_ref == effect.transition_ref
                        && usage.key_effect_ref == effect.effect_ref
                })
                .count();
            if count != 1 {
                return Err(error(
                    "live equality effect does not have one exact dependency use",
                ));
            }
        }
    }
    Ok(())
}

fn validate_equality_use(
    graph: &ReservedSemanticsV2Graph,
    usage: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<(), EngineError> {
    if !matches!(usage.role, UNIQUE_KEY_ROLE | FOREIGN_KEY_ROLE) {
        return Ok(());
    }
    match (usage.transition_ref, usage.key_effect_ref) {
        (ABSENT_U32, ABSENT_U32) if token.flags & 2 != 0 => {
            let terminal = graph
                .resolutions
                .iter()
                .find(|resolution| resolution.terminal_dependency_ref == token.dependency_ref)
                .ok_or_else(|| error("terminal equality guard has no terminal resolution"))?;
            let outcome = graph
                .outcomes
                .get(terminal.outcome_ref as usize)
                .ok_or_else(|| error("terminal equality guard outcome is absent"))?;
            if outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::AbortError
                || usage.statement_ordinal != terminal.statement_ordinal
                || usage.source_ordinal != terminal.terminal_source_ordinal
            {
                return Err(error(
                    "terminal equality guard use is outside its final abort source",
                ));
            }
            Ok(())
        }
        (transition_ref, effect_ref)
            if transition_ref != ABSENT_U32 && effect_ref != ABSENT_U32 =>
        {
            let effect = graph
                .key_effects
                .get(effect_ref as usize)
                .ok_or_else(|| error("equality use key effect is absent"))?;
            let transition = graph
                .transitions
                .get(transition_ref as usize)
                .ok_or_else(|| error("equality use transition is absent"))?;
            if effect.transition_ref != transition_ref
                || effect.dependency_ref != token.dependency_ref
                || token.key_effect_ref != effect_ref
                || transition.source_statement_ordinal != usage.statement_ordinal
            {
                return Err(error("equality use/token/effect relation is not bijective"));
            }
            Ok(())
        }
        _ => Err(error(
            "equality dependency use has an invalid terminal/live shape",
        )),
    }
}

fn validate_terminal_binding(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let abort = graph.resolutions.iter().find(|resolution| {
        graph
            .outcomes
            .get(resolution.outcome_ref as usize)
            .is_some_and(|outcome| {
                outcome.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError
            })
    });
    let terminal = graph.dependencies.iter().find(|token| token.flags & 2 != 0);
    match (abort, terminal) {
        (None, None) => Ok(()),
        (Some(resolution), Some(token)) => {
            if resolution.terminal_dependency_ref != token.dependency_ref {
                return Err(error(
                    "terminal resolution does not select the terminal token",
                ));
            }
            let use_count = graph
                .dependency_uses
                .iter()
                .filter(|usage| {
                    usage.statement_ordinal == resolution.statement_ordinal
                        && usage.dependency_ref == token.dependency_ref
                        && usage.role == token.kind as u16
                        && usage.source_ordinal == resolution.terminal_source_ordinal
                })
                .count();
            if use_count != 1 {
                return Err(error(
                    "terminal dependency does not have one exact statement use",
                ));
            }
            let outcome = graph
                .outcomes
                .get(resolution.outcome_ref as usize)
                .ok_or_else(|| error("terminal outcome is absent"))?;
            if !terminal_sqlstate_matches(token.kind, outcome.outcome.sqlstate) {
                return Err(error("terminal S6 outcome does not match its guard class"));
            }
            let expected_constraint_id = match token.kind {
                UNIQUE_KEY_GUARD => {
                    let descriptor = descriptor_for(graph, token)?
                        .ok_or_else(|| error("terminal unique guard has no index descriptor"))?;
                    if descriptor.stable_constraint_id == u64::MAX {
                        descriptor.stable_index_id
                    } else {
                        descriptor.stable_constraint_id
                    }
                }
                NOT_NULL_GUARD | CHECK_GUARD | DOMAIN_CONSTRAINT_GUARD => token.stable_object_id,
                FOREIGN_KEY_GUARD => 0,
                _ => unreachable!("terminal kind is checked above"),
            };
            if token.kind != FOREIGN_KEY_GUARD
                && outcome.outcome.constraint_id != expected_constraint_id
            {
                return Err(error("terminal S6 outcome does not match its guard class"));
            }
            if token.kind == NOT_NULL_GUARD {
                let record = graph
                    .records
                    .get(resolution.record_ref as usize)
                    .ok_or_else(|| error("terminal NOT NULL source S2 record is absent"))?;
                let column = record
                    .catalog_columns()
                    .nth(resolution.terminal_source_ordinal as usize)
                    .ok_or_else(|| error("terminal NOT NULL source catalog column is absent"))?;
                let (valid, _) = record.column_value_at(
                    column.catalog_column_ordinal,
                    resolution.terminal_row_ordinal,
                )?;
                if column.catalog_column_ordinal != resolution.terminal_source_ordinal || valid {
                    return Err(error(
                        "terminal NOT NULL does not select a NULL S2 column cell",
                    ));
                }
            }
            // A 23503 S6 carries the FK constraint stable ID. Its `ForeignKeyGuard` token is
            // intentionally keyed by the supporting parent index, so those identities cannot
            // be compared here.
            Ok(())
        }
        _ => Err(error("terminal outcome/token cardinalities diverge")),
    }
}

fn runtime_guard_key(
    graph: &ReservedSemanticsV2Graph,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<[u8; 32], EngineError> {
    let table = graph
        .tables
        .get(token.target_table_ref as usize)
        .ok_or_else(|| error("runtime guard target table is absent"))?;
    let mut digest = begin(b"gpu-db/write001/runtime-guard-key/v2");
    digest.update([token.kind]);
    digest.update(table.stable_table_id.to_le_bytes());
    digest.update(token.stable_object_id.to_le_bytes());
    digest.update(token.catalog_epoch.to_le_bytes());
    digest.update(token.base_generation.to_le_bytes());
    digest.update(token.schema_digest);
    digest.update(token.base_root);
    digest.update(token.name_digest);
    let (index_id, values) = if token.key_effect_ref != ABSENT_U32 {
        let effect = graph
            .key_effects
            .get(token.key_effect_ref as usize)
            .ok_or_else(|| error("runtime guard effect is absent"))?;
        let index = graph
            .indexes
            .get(effect.index_ref as usize)
            .ok_or_else(|| error("runtime guard index is absent"))?;
        let components = range(
            &graph.key_components,
            effect.new_component_start,
            effect.new_component_count,
            "runtime guard component",
        )?;
        digest.update(index.stable_index_id.to_le_bytes());
        digest.update(effect.key_arity.to_le_bytes());
        for component in components {
            digest.update(component.typed_value_digest);
        }
        (index.stable_index_id, effect.key_arity)
    } else if token.flags & 2 != 0 && matches!(token.kind, 4 | 6) {
        let resolution = graph
            .resolutions
            .iter()
            .find(|resolution| resolution.terminal_dependency_ref == token.dependency_ref)
            .ok_or_else(|| error("terminal runtime guard has no resolution"))?;
        let record = graph
            .records
            .get(resolution.record_ref as usize)
            .ok_or_else(|| error("terminal runtime guard S2 is absent"))?;
        let index = graph
            .indexes
            .get(token.descriptor_ref as usize)
            .ok_or_else(|| error("terminal runtime guard index is absent"))?;
        let keys = range(
            &graph.index_key_columns,
            index.key_start,
            index.key_count,
            "terminal runtime guard key",
        )?;
        if token.kind == FOREIGN_KEY_GUARD {
            let foreign_key = record
                .foreign_keys()
                .find(|foreign_key| foreign_key.raw_ordinal == resolution.terminal_source_ordinal)
                .ok_or_else(|| error("terminal FK runtime guard source is absent"))?;
            if keys.len() != 1
                || record
                    .foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?
                    .len()
                    != 1
            {
                return Err(error(
                    "current S2 terminal FK guard admits exactly one supporting key",
                ));
            }
        }
        digest.update(index.stable_index_id.to_le_bytes());
        digest.update(index.key_count.to_le_bytes());
        for key in keys {
            let source_column = if token.kind == 4 {
                key.owner_catalog_column_ordinal
            } else {
                let foreign_key = record
                    .foreign_keys()
                    .find(|foreign_key| {
                        foreign_key.raw_ordinal == resolution.terminal_source_ordinal
                    })
                    .ok_or_else(|| error("terminal FK runtime guard source is absent"))?;
                foreign_key.child_column.catalog_column_ordinal
            };
            let (valid, value) =
                record.column_value_at(source_column, resolution.terminal_row_ordinal)?;
            if !valid {
                return Err(error("terminal equality guard uses a NULL key"));
            }
            let mut value_digest = begin(b"gpu-db/write001/s7-typed-key-value/v2");
            value_digest.update(storage(if token.kind == 4 {
                key_type(record, key.owner_catalog_column_ordinal)?
            } else {
                key_type(record, source_column)?
            }));
            value_digest.update(
                if token.kind == 4 {
                    key.declared_type_oid
                } else {
                    record
                        .catalog_columns()
                        .nth(source_column as usize)
                        .ok_or_else(|| error("terminal child column is absent"))?
                        .type_oid
                }
                .to_le_bytes(),
            );
            value_digest.update(
                if token.kind == 4 {
                    key.signed_type_size
                } else {
                    record
                        .catalog_columns()
                        .nth(source_column as usize)
                        .ok_or_else(|| error("terminal child column is absent"))?
                        .type_size
                }
                .to_le_bytes(),
            );
            append_record_value(&mut value_digest, valid, value);
            digest.update(value_digest.finalize());
        }
        (index.stable_index_id, index.key_count)
    } else {
        digest.update(0_u64.to_le_bytes());
        digest.update(0_u32.to_le_bytes());
        (0, 0)
    };
    let _ = (index_id, values);
    Ok(digest.finalize().into())
}

fn descriptor_for<'a>(
    graph: &'a ReservedSemanticsV2Graph,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<
    Option<
        &'a crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexDescriptor,
    >,
    EngineError,
> {
    if token.descriptor_ref == ABSENT_U32 {
        return Ok(None);
    }
    graph
        .indexes
        .get(token.descriptor_ref as usize)
        .map(Some)
        .ok_or_else(|| error("dependency index descriptor is absent"))
}

fn effect_for<'a>(
    graph: &'a ReservedSemanticsV2Graph,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<
    Option<&'a crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedKeyEffect>,
    EngineError,
> {
    if token.key_effect_ref == ABSENT_U32 {
        return Ok(None);
    }
    graph
        .key_effects
        .get(token.key_effect_ref as usize)
        .map(Some)
        .ok_or_else(|| error("dependency key effect is absent"))
}

fn required_access(kind: u8) -> u8 {
    match kind {
        TARGET_TABLE | MAINTAINED_INDEX => 3,
        FOREIGN_PARENT_TABLE
        | UNIQUE_KEY_GUARD
        | FOREIGN_PARENT_INDEX
        | FOREIGN_KEY_GUARD
        | NOT_NULL_GUARD
        | CHECK_GUARD
        | DOMAIN_CONSTRAINT_GUARD => 2,
        DOMAIN => 1,
        PUBLISHED_SEQUENCE => 4,
        _ => 0,
    }
}

fn expected_kind_for_role(role: u16) -> Option<u8> {
    match role {
        TARGET_TABLE_ROLE => Some(TARGET_TABLE),
        MAINTAINED_INDEX_ROLE => Some(MAINTAINED_INDEX),
        UNIQUE_KEY_ROLE => Some(UNIQUE_KEY_GUARD),
        FOREIGN_PARENT_TABLE_ROLE => Some(FOREIGN_PARENT_TABLE),
        FOREIGN_PARENT_INDEX_ROLE => Some(FOREIGN_PARENT_INDEX),
        FOREIGN_KEY_ROLE => Some(FOREIGN_KEY_GUARD),
        DOMAIN_ROLE..=DOMAIN_CONSTRAINT_GUARD_ROLE => Some(role as u8),
        _ => None,
    }
}

fn valid_use_shape(role: u16, transition: u32, effect: u32) -> bool {
    let static_role = matches!(
        role,
        TARGET_TABLE_ROLE
            | MAINTAINED_INDEX_ROLE
            | FOREIGN_PARENT_TABLE_ROLE
            | FOREIGN_PARENT_INDEX_ROLE
            | DOMAIN_ROLE
            | NOT_NULL_GUARD_ROLE
            | CHECK_GUARD_ROLE
            | DOMAIN_CONSTRAINT_GUARD_ROLE
    );
    if static_role {
        return transition == ABSENT_U32 && effect == ABSENT_U32;
    }
    if role == PUBLISHED_SEQUENCE_ROLE {
        return effect == ABSENT_U32;
    }
    matches!(role, UNIQUE_KEY_ROLE | FOREIGN_KEY_ROLE)
        && (transition == ABSENT_U32) == (effect == ABSENT_U32)
}

pub(super) fn descriptor_matches_s2_index(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    descriptor: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexDescriptor,
    source: crate::typed_insert_batch::DecodedIndexFacts<'_>,
) -> Result<bool, EngineError> {
    let owner = if source.owner_dependency_ordinal == 0 {
        let target = record.target_identity();
        (target.oid, target.schema, target.name, target.schema_digest)
    } else {
        let parent = record
            .dependencies()
            .nth(source.owner_dependency_ordinal as usize)
            .ok_or_else(|| error("S2 index owner dependency is absent"))?;
        (parent.oid, parent.schema, parent.name, parent.schema_digest)
    };
    let owner_table_ref = owner_table_ref(graph, record, owner.0, owner.1, owner.2, owner.3)?;
    // S2 preserves its statement-time catalog binding while S7 names the final composed table.
    // A transaction-created table, or a pre-CREATE statement on a published table with a
    // paired-zero S3-created index, may therefore retain the same stable table identity across
    // a schema digest change. The latter bridge is limited to the existing S3 descriptor proof.
    let owner_schema_digest = owner_table_ref
        .and_then(|table_ref| graph.tables.get(table_ref as usize))
        .map_or(owner.3, |table| table.schema_digest);
    if source.table_name != owner.2
        || descriptor.owner_display_oid != owner.0
        || descriptor.owner_schema_digest != owner_schema_digest
        || descriptor.owner_name_digest != qualified_name_digest(owner.1, owner.2)
        || descriptor.index_name_digest != qualified_name_digest(owner.1, source.name)
        || descriptor.raw_catalog_ordinal != source.raw_ordinal
        || descriptor.display_oid != source.oid
        || descriptor.flags & 1 != u32::from(source.unique)
        || (descriptor.flags & 2 != 0) != source.primary_key
        || (descriptor.flags & 4 != 0) != source.unique_constraint
        || descriptor.key_count != source.key_count
        || descriptor.owner_table_ref != owner_table_ref.unwrap_or(ABSENT_U32)
        || (descriptor.flags & 8 != 0) != owner_table_ref.is_some()
    {
        return Ok(false);
    }
    Ok(true)
}

pub(super) fn descriptor_matches_foreign_key(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    descriptor: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexDescriptor,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
) -> Result<bool, EngineError> {
    let source = foreign_key.supporting_index;
    if source.owner_dependency_ordinal != foreign_key.parent_dependency_ordinal {
        return Ok(false);
    }
    descriptor_matches_s2_index(graph, record, descriptor, source)
}

pub(super) fn descriptor_keys_match_s2_index(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    descriptor: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexDescriptor,
    source: crate::typed_insert_batch::DecodedIndexFacts<'_>,
) -> Result<bool, EngineError> {
    let keys = range(
        &graph.index_key_columns,
        descriptor.key_start,
        descriptor.key_count,
        "S2 index descriptor key",
    )?;
    let source_keys = record.index_key_columns(source.raw_ordinal)?;
    if source_keys.len() != keys.len() {
        return Ok(false);
    }
    Ok(keys.iter().zip(source_keys).all(|(key, source)| {
        key.owner_catalog_column_ordinal == source.catalog_column_ordinal
            && key.stable_column_id == source.column_id
            && key.owner_display_table_oid == descriptor.owner_display_oid
            && key.attnum == source.attnum
            && key.storage == storage(source.ty)
            && key.declared_type_oid == source.type_oid
            && key.signed_type_size == source.type_size
            && key.column_name_digest == identifier_digest(source.name)
    }))
}

pub(super) fn descriptor_keys_match_foreign_key(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    descriptor: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexDescriptor,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
) -> Result<bool, EngineError> {
    let keys = range(
        &graph.index_key_columns,
        descriptor.key_start,
        descriptor.key_count,
        "S2 FK descriptor key",
    )?;
    let source_keys = record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?;
    if source_keys.len() != keys.len() {
        return Ok(false);
    }
    Ok(keys.iter().zip(source_keys).all(|(key, source)| {
        key.owner_catalog_column_ordinal == source.catalog_column_ordinal
            && key.stable_column_id == source.column_id
            && key.owner_display_table_oid == descriptor.owner_display_oid
            && key.attnum == source.attnum
            && key.storage == storage(source.ty)
            && key.declared_type_oid == source.type_oid
            && key.signed_type_size == source.type_size
            && key.column_name_digest == identifier_digest(source.name)
    }))
}

fn owner_table_ref(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    owner_display_oid: u32,
    schema: &str,
    name: &str,
    schema_digest: [u8; 32],
) -> Result<Option<u32>, EngineError> {
    let name_digest = qualified_name_digest(schema, name);
    let mut matched = None;
    for (table_ref, table) in graph.tables.iter().enumerate() {
        let target = graph
            .dependencies
            .get(table.target_dependency_ref as usize)
            .ok_or_else(|| error("index owner table target dependency is absent"))?;
        let table_ref_u32 =
            u32::try_from(table_ref).map_err(|_| error("index owner table ref exceeds u32"))?;
        let s3_created_index_schema_bridge = graph.indexes.iter().any(|descriptor| {
            descriptor.owner_table_ref == table_ref_u32
                && descriptor.base_index_generation == 0
                && descriptor.base_index_root == [0; 32]
                && !record
                    .indexes()
                    .any(|source| source.oid == descriptor.display_oid)
        });
        if table.display_oid == owner_display_oid
            && target.name_digest == name_digest
            && (table.schema_digest == schema_digest
                || table.initial_table_absent
                || s3_created_index_schema_bridge)
            && matched.replace(table_ref_u32).is_some()
        {
            return Err(error("S7 table blocks ambiguously name an index owner"));
        }
    }
    Ok(matched)
}

pub(super) fn qualified_name_digest(schema: &str, name: &str) -> [u8; 32] {
    exact(
        b"gpu-db/write001/s7-qualified-name/v2",
        &[
            &(schema.len() as u32).to_le_bytes(),
            schema.as_bytes(),
            &(name.len() as u32).to_le_bytes(),
            name.as_bytes(),
        ],
    )
}

fn domain_shape_digest(base_type: crate::SqlType) -> [u8; 32] {
    let storage = storage(base_type);
    let declared_type_oid = base_type.postgres_oid().to_le_bytes();
    let signed_type_size = base_type.type_size().to_le_bytes();
    exact(
        b"gpu-db/write001/s7-domain-shape/v2",
        &[&storage, &declared_type_oid, &signed_type_size],
    )
}

fn terminal_sqlstate_matches(kind: u8, sqlstate: Option<[u8; 5]>) -> bool {
    match kind {
        UNIQUE_KEY_GUARD => sqlstate == Some(*b"23505"),
        FOREIGN_KEY_GUARD => sqlstate == Some(*b"23503"),
        NOT_NULL_GUARD => sqlstate == Some(*b"23502"),
        CHECK_GUARD => sqlstate == Some(*b"23514"),
        // Q1 has no catalog witness to distinguish a domain's NOT NULL failure from its CHECK
        // failure. The terminal path still retains and checks the exact stable constraint ID.
        DOMAIN_CONSTRAINT_GUARD => sqlstate == Some(*b"23502") || sqlstate == Some(*b"23514"),
        _ => false,
    }
}

fn synthesized_not_null_name_digest(
    owner_kind: u8,
    stable_owner_id: u64,
    source_ordinal: u32,
) -> [u8; 32] {
    exact(
        b"gpu-db/write001/s7-synthesized-not-null-name/v2",
        &[
            &[owner_kind],
            &stable_owner_id.to_le_bytes(),
            &source_ordinal.to_le_bytes(),
        ],
    )
}

fn identifier_digest(name: &str) -> [u8; 32] {
    exact(
        b"gpu-db/write001/s7-identifier/v2",
        &[
            (name.len() as u32).to_le_bytes().as_slice(),
            name.as_bytes(),
        ],
    )
}

fn dependency_order(
    left: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
    right: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Ordering {
    let fields = [
        (u64::from(left.kind), u64::from(right.kind)),
        (
            u64::from(left.flags & 2 != 0),
            u64::from(right.flags & 2 != 0),
        ),
        (left.stable_object_id, right.stable_object_id),
        (u64::from(left.display_oid), u64::from(right.display_oid)),
        (
            u64::from(left.target_table_ref),
            u64::from(right.target_table_ref),
        ),
        (left.catalog_epoch, right.catalog_epoch),
        (left.base_generation, right.base_generation),
        (
            u64::from(left.key_effect_ref),
            u64::from(right.key_effect_ref),
        ),
        (
            u64::from(left.descriptor_ref),
            u64::from(right.descriptor_ref),
        ),
    ];
    for (left, right) in fields {
        match left.cmp(&right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    for (left, right) in [
        (&left.schema_digest, &right.schema_digest),
        (&left.base_root, &right.base_root),
        (&left.name_digest, &right.name_digest),
        (&left.identity_digest, &right.identity_digest),
    ] {
        match left.cmp(right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

fn use_order(
    left: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse,
    right: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementDependencyUse,
) -> Ordering {
    for (left, right) in [
        (left.statement_ordinal, right.statement_ordinal),
        (u32::from(left.role), u32::from(right.role)),
        (left.source_ordinal, right.source_ordinal),
        (left.transition_ref, right.transition_ref),
        (left.key_effect_ref, right.key_effect_ref),
        (left.dependency_ref, right.dependency_ref),
    ] {
        match left.cmp(&right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

fn key_type(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    ordinal: u32,
) -> Result<crate::SqlType, EngineError> {
    record
        .catalog_columns()
        .nth(ordinal as usize)
        .map(|column| column.ty)
        .ok_or_else(|| error("runtime guard source column is absent"))
}

fn published_sequence_identity(
    graph: &ReservedSemanticsV2Graph,
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
) -> Result<[u8; 32], EngineError> {
    let mut match_count = 0_u32;
    let mut bytes = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    let mut body_digest = [0_u8; 32];
    for effect in &graph.sequence_effects {
        let Some(reference) = effect.reference.as_ref() else {
            continue;
        };
        if reference.transition_txn_id == token.base_generation
            && reference.sequence_oid == token.display_oid
            && effect.reference_body_digest == token.base_root
        {
            match_count = match_count
                .checked_add(1)
                .ok_or_else(|| error("sequence token match count overflows"))?;
            crate::encode_sequence_value_reference_into_exact(reference, &mut bytes)?;
            body_digest = effect.reference_body_digest;
        }
    }
    if match_count != 1 {
        return Err(error(
            "published-sequence token does not select one S5 effect",
        ));
    }
    Ok(exact(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &token.stable_object_id.to_le_bytes(),
            &token.display_oid.to_le_bytes(),
            &token.catalog_epoch.to_le_bytes(),
            &token.base_generation.to_le_bytes(),
            &token.name_digest,
            &body_digest,
            &bytes,
        ],
    ))
}

fn token_digest(
    token: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDependencyToken,
    descriptor: [u8; 32],
    effect: [u8; 32],
) -> [u8; 32] {
    let mut raw = [0_u8; 192];
    raw[..4].copy_from_slice(&token.dependency_ref.to_le_bytes());
    raw[4] = token.kind;
    raw[5] = token.access;
    raw[6..8].copy_from_slice(&token.flags.to_le_bytes());
    raw[8..16].copy_from_slice(&token.stable_object_id.to_le_bytes());
    raw[16..20].copy_from_slice(&token.display_oid.to_le_bytes());
    raw[20..24].copy_from_slice(&token.target_table_ref.to_le_bytes());
    raw[24..32].copy_from_slice(&token.base_generation.to_le_bytes());
    raw[32..40].copy_from_slice(&token.snapshot_floor.to_le_bytes());
    raw[40..44].copy_from_slice(&token.key_effect_ref.to_le_bytes());
    raw[44..48].copy_from_slice(&token.descriptor_ref.to_le_bytes());
    raw[48..56].copy_from_slice(&token.catalog_epoch.to_le_bytes());
    raw[64..96].copy_from_slice(&token.schema_digest);
    raw[96..128].copy_from_slice(&token.base_root);
    raw[128..160].copy_from_slice(&token.name_digest);
    raw[160..192].copy_from_slice(&token.identity_digest);
    exact(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw, &[0; 32], &descriptor, &effect],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_shape_digest_uses_the_strict_s2_base_type_shape() {
        let base_type = crate::SqlType::Int4;
        assert_eq!(
            domain_shape_digest(base_type),
            exact(
                b"gpu-db/write001/s7-domain-shape/v2",
                &[
                    &storage(base_type),
                    &base_type.postgres_oid().to_le_bytes(),
                    &base_type.type_size().to_le_bytes(),
                ],
            )
        );
        assert_ne!(
            domain_shape_digest(base_type),
            domain_shape_digest(crate::SqlType::Int8)
        );
    }

    #[test]
    fn domain_constraint_sqlstate_is_provisional_but_other_guards_remain_exact() {
        assert!(terminal_sqlstate_matches(
            DOMAIN_CONSTRAINT_GUARD,
            Some(*b"23502")
        ));
        assert!(terminal_sqlstate_matches(
            DOMAIN_CONSTRAINT_GUARD,
            Some(*b"23514")
        ));
        assert!(!terminal_sqlstate_matches(
            DOMAIN_CONSTRAINT_GUARD,
            Some(*b"23503")
        ));
        assert!(!terminal_sqlstate_matches(
            DOMAIN_CONSTRAINT_GUARD,
            Some(*b"23505")
        ));
        assert!(terminal_sqlstate_matches(NOT_NULL_GUARD, Some(*b"23502")));
        assert!(!terminal_sqlstate_matches(NOT_NULL_GUARD, Some(*b"23514")));
        assert!(terminal_sqlstate_matches(CHECK_GUARD, Some(*b"23514")));
        assert!(!terminal_sqlstate_matches(CHECK_GUARD, Some(*b"23502")));
    }
}
