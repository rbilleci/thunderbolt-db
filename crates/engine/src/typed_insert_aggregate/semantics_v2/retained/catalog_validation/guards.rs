//! Guard applicability, terminal catalog matching, and foreign-key closure.

use super::super::{
    graph::{ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedTable},
    SemanticsV2BoundIdentity, SemanticsV2CatalogDomainWitness, SemanticsV2CatalogForeignKeyWitness,
    SemanticsV2CatalogGuardWitness, SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
};
use crate::typed_insert_batch::DecodedForeignKeyFacts;

const ABSENT_U32: u32 = u32::MAX;
const TERMINAL_ERROR_FLAG: u16 = 2;

#[cfg(test)]
#[path = "guards_tests.rs"]
mod guards_tests;

pub(super) fn validate_guard_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        super::require(
            row.stable_guard_id != 0
                && row.stable_guard_id != u64::MAX
                && matches!(
                    row.kind,
                    super::NOT_NULL_GUARD | super::CHECK_GUARD | super::DOMAIN_CONSTRAINT_GUARD
                )
                && previous.is_none_or(|prior| prior < row.stable_guard_id),
            "catalog guards are not in strict stable-ID order",
        )?;
        if row.synthesized_not_null {
            super::require(
                row.display_oid <= 0x7fff_ffff
                    && row.schema.is_empty()
                    && row.name.is_empty()
                    && matches!(
                        row.kind,
                        super::NOT_NULL_GUARD | super::DOMAIN_CONSTRAINT_GUARD
                    ),
                "only synthesized NOT NULL guards may omit a display OID",
            )?;
        } else {
            super::require(
                row.display_oid != 0
                    && row.display_oid <= 0x7fff_ffff
                    && super::valid_identifier(row.schema)
                    && super::valid_identifier(row.name),
                "named catalog guard identity is invalid",
            )?;
        }
        previous = Some(row.stable_guard_id);
    }
    Ok(())
}

pub(super) fn validate_guard_list_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
    kind: u8,
    owner: &str,
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    let mut expected_raw_constraint = 0_u32;
    for row in rows {
        let raw_constraint_is_dense = kind != super::NOT_NULL_GUARD;
        super::require(
            row.kind == kind
                && previous.is_none_or(|prior| prior < row.source_ordinal)
                && row.stable_guard_id != 0
                && row.stable_guard_id != u64::MAX
                && (!raw_constraint_is_dense
                    || (row.raw_constraint_ordinal == expected_raw_constraint
                        && row.source_ordinal == expected_raw_constraint)),
            "catalog guard list is not in exact source order",
        )?;
        previous = Some(row.source_ordinal);
        if raw_constraint_is_dense {
            expected_raw_constraint = expected_raw_constraint
                .checked_add(1)
                .ok_or_else(|| super::validation_error("catalog guard raw ordinal overflows"))?;
        }
    }
    let _ = owner;
    Ok(())
}

pub(super) fn validate_guard_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for guard in catalog.guards {
        let count = graph
            .dependencies
            .iter()
            .filter(|dependency| guard_matches_dependency(identity, guard, dependency))
            .count();
        super::require(
            count != 0,
            "pinned catalog guard is outside the exact S2/catalog guard closure",
        )?;
        validate_guard_owner(catalog, guard)?;
    }
    for dependency in graph.dependencies.iter().filter(|dependency| {
        matches!(
            dependency.kind,
            super::NOT_NULL_GUARD | super::CHECK_GUARD | super::DOMAIN_CONSTRAINT_GUARD
        )
    }) {
        let count = catalog
            .guards
            .iter()
            .filter(|guard| guard_matches_dependency(identity, guard, dependency))
            .count();
        super::require(
            count == 1,
            "S7 guard dependency does not select exactly one pinned catalog guard",
        )?;
    }
    validate_guard_nested_membership(catalog)?;
    validate_statement_guard_bijections(identity, graph, catalog)?;
    validate_terminal_catalog_identity(identity, graph, catalog)
}

fn guard_matches_dependency(
    identity: SemanticsV2BoundIdentity,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
    dependency: &RetainedDependencyToken,
) -> bool {
    guard.kind == dependency.kind
        && guard.stable_guard_id == dependency.stable_object_id
        && guard.display_oid == dependency.display_oid
        && guard.catalog_generation == dependency.base_generation
        && guard.shape_digest == dependency.schema_digest
        && guard.program_or_descriptor_root == dependency.base_root
        && dependency.catalog_epoch == identity.catalog_epoch
        && guard_name_digest(guard).is_ok_and(|digest| dependency.name_digest == digest)
}

fn validate_guard_owner(
    catalog: &SemanticsV2CatalogWitness<'_>,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<(), crate::EngineError> {
    match guard.kind {
        super::NOT_NULL_GUARD | super::CHECK_GUARD => {
            super::require(
                guard.owner_kind == 1,
                "table guard has an invalid owner kind",
            )?;
            let table = catalog
                .tables
                .iter()
                .find(|table| {
                    table.stable_table_id == guard.owner_stable_id
                        && table.display_oid == guard.owner_display_oid
                })
                .ok_or_else(|| {
                    super::validation_error("table guard owner is absent from the catalog")
                })?;
            if guard.kind == super::NOT_NULL_GUARD {
                let column = table
                    .catalog_columns
                    .iter()
                    .find(|column| {
                        column.catalog_column_ordinal == guard.owner_catalog_column_ordinal
                    })
                    .ok_or_else(|| {
                        super::validation_error("NOT NULL guard owner column is absent")
                    })?;
                super::require(
                    guard.synthesized_not_null
                        && guard.raw_constraint_ordinal == 0
                        && guard.shape_digest == column.column_shape_digest
                        && guard.program_or_descriptor_root == column.column_root,
                    "NOT NULL guard differs from its pinned catalog column descriptor",
                )?;
            } else {
                super::require(
                    !guard.synthesized_not_null
                        && guard.owner_catalog_column_ordinal == u32::MAX
                        && guard.domain_ordinal == u32::MAX,
                    "CHECK guard has an invalid table-column/domain binding",
                )?;
            }
        }
        super::DOMAIN_CONSTRAINT_GUARD => {
            super::require(
                guard.owner_kind == 2 && guard.owner_catalog_column_ordinal == u32::MAX,
                "domain guard has an invalid owner binding",
            )?;
            let domain = catalog
                .domains
                .iter()
                .find(|domain| {
                    domain.stable_domain_id == guard.owner_stable_id
                        && domain.display_oid == guard.owner_display_oid
                })
                .ok_or_else(|| {
                    super::validation_error("domain guard owner is absent from the catalog")
                })?;
            super::require(
                domain
                    .constraints
                    .iter()
                    .any(|candidate| same_guard(candidate, guard)),
                "domain guard is absent from its ordered domain-constraint list",
            )?;
        }
        _ => return Err(super::validation_error("catalog guard kind is invalid")),
    }
    Ok(())
}

fn validate_guard_nested_membership(
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for table in catalog.tables {
        for guard in table.not_null_guards.iter().chain(table.check_guards) {
            super::require(
                catalog
                    .guards
                    .iter()
                    .filter(|candidate| same_guard(candidate, guard))
                    .count()
                    == 1
                    && guard.owner_stable_id == table.stable_table_id
                    && guard.owner_display_oid == table.display_oid,
                "table guard list has an extra, missing, or foreign guard",
            )?;
        }
    }
    for domain in catalog.domains {
        for guard in domain.constraints {
            super::require(
                catalog
                    .guards
                    .iter()
                    .filter(|candidate| same_guard(candidate, guard))
                    .count()
                    == 1
                    && guard.owner_stable_id == domain.stable_domain_id
                    && guard.owner_display_oid == domain.display_oid,
                "domain guard list has an extra, missing, or foreign guard",
            )?;
        }
    }
    for guard in catalog.guards {
        let nested_count = match guard.kind {
            super::NOT_NULL_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.not_null_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            super::CHECK_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.check_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            super::DOMAIN_CONSTRAINT_GUARD => catalog
                .domains
                .iter()
                .flat_map(|domain| domain.constraints)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            _ => 0,
        };
        super::require(
            nested_count == 1,
            "catalog guard does not occur exactly once in its owner list",
        )?;
    }
    Ok(())
}

/// Build the expected static guard inventory directly from one resolved target table and its S2
/// record, then close it bijectively against roles 9--11 in that resolution's bounded use range.
/// No map or temporary inventory is materialized: every expected guard is checked in place and
/// the aggregate actual-use count proves there are no extras.
fn validate_statement_guard_bijections(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for resolution in &graph.resolutions {
        let (retained, target) = target_for_resolution(graph, catalog, resolution)?;
        let record = graph
            .records
            .get(usize::try_from(resolution.record_ref).map_err(|_| {
                super::validation_error("guard resolution record reference is unaddressable")
            })?)
            .ok_or_else(|| super::validation_error("guard resolution record is absent"))?;
        let uses = resolution_uses(graph, resolution)?;
        super::require(
            uses.iter()
                .all(|usage| usage.statement_ordinal == resolution.statement_ordinal),
            "guard resolution use range crosses a statement boundary",
        )?;

        let mut expected_count = 0_usize;
        for guard in target.not_null_guards {
            validate_table_not_null_expected(target, record, guard)?;
            require_exact_expected_guard_use(identity, graph, resolution, uses, guard)?;
            expected_count = expected_count.checked_add(1).ok_or_else(|| {
                super::validation_error("expected NOT NULL guard count overflows")
            })?;
        }
        for (raw_ordinal, guard) in target.check_guards.iter().enumerate() {
            validate_table_check_expected(target, guard, raw_ordinal)?;
            require_exact_expected_guard_use(identity, graph, resolution, uses, guard)?;
            expected_count = expected_count
                .checked_add(1)
                .ok_or_else(|| super::validation_error("expected CHECK guard count overflows"))?;
        }
        for (domain_ordinal, domain) in catalog.domains.iter().enumerate() {
            let Some(_source) = bound_domain_source(record, target, domain)? else {
                continue;
            };
            for (raw_ordinal, guard) in domain.constraints.iter().enumerate() {
                validate_domain_guard_expected(domain, guard, domain_ordinal, raw_ordinal)?;
                require_exact_expected_guard_use(identity, graph, resolution, uses, guard)?;
                expected_count = expected_count.checked_add(1).ok_or_else(|| {
                    super::validation_error("expected domain guard count overflows")
                })?;
            }
        }

        let actual_count = uses
            .iter()
            .filter(|usage| matches!(usage.role, 9..=11))
            .count();
        super::require(
            actual_count == expected_count,
            "statement guard uses have an extra, missing, or cross-owner entry",
        )?;
        let _ = retained;
    }
    Ok(())
}

fn target_for_resolution<'a, 'b>(
    graph: &'a ReservedSemanticsV2Graph,
    catalog: &'b SemanticsV2CatalogWitness<'b>,
    resolution: &super::super::graph::RetainedStatementResolution,
) -> Result<(&'a RetainedTable, &'b SemanticsV2CatalogTableWitness<'b>), crate::EngineError> {
    let mut retained_rows = graph
        .tables
        .iter()
        .filter(|table| table.table_ref == resolution.table_ref);
    let retained = retained_rows
        .next()
        .ok_or_else(|| super::validation_error("guard resolution target table is absent"))?;
    super::require(
        retained_rows.next().is_none(),
        "guard resolution target table reference is ambiguous",
    )?;
    let mut catalog_rows = catalog.tables.iter().filter(|table| {
        table.stable_table_id == retained.stable_table_id
            && table.display_oid == retained.display_oid
    });
    let target = catalog_rows.next().ok_or_else(|| {
        super::validation_error("guard resolution target catalog table is absent")
    })?;
    super::require(
        catalog_rows.next().is_none(),
        "guard resolution target catalog table is ambiguous",
    )?;
    Ok((retained, target))
}

fn resolution_uses<'a>(
    graph: &'a ReservedSemanticsV2Graph,
    resolution: &super::super::graph::RetainedStatementResolution,
) -> Result<&'a [super::super::graph::RetainedStatementDependencyUse], crate::EngineError> {
    let start = usize::try_from(resolution.dependency_use_start)
        .map_err(|_| super::validation_error("guard use range start is unaddressable"))?;
    let count = usize::try_from(resolution.dependency_use_count)
        .map_err(|_| super::validation_error("guard use range count is unaddressable"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| super::validation_error("guard use range overflows"))?;
    graph
        .dependency_uses
        .get(start..end)
        .ok_or_else(|| super::validation_error("guard use range exceeds retained graph"))
}

fn validate_table_not_null_expected(
    target: &SemanticsV2CatalogTableWitness<'_>,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<(), crate::EngineError> {
    let source = record
        .catalog_columns()
        .find(|column| column.catalog_column_ordinal == guard.owner_catalog_column_ordinal)
        .ok_or_else(|| super::validation_error("NOT NULL guard has no S2 target column"))?;
    let column = target
        .catalog_columns
        .iter()
        .find(|column| column.catalog_column_ordinal == guard.owner_catalog_column_ordinal)
        .ok_or_else(|| super::validation_error("NOT NULL guard owner column is not pinned"))?;
    super::require(
        guard.kind == super::NOT_NULL_GUARD
            && guard.owner_kind == 1
            && guard.owner_stable_id == target.stable_table_id
            && guard.owner_display_oid == target.display_oid
            && guard.source_ordinal == guard.owner_catalog_column_ordinal
            && guard.raw_constraint_ordinal == 0
            && guard.domain_ordinal == ABSENT_U32
            && guard.synthesized_not_null
            && guard.shape_digest == column.column_shape_digest
            && guard.program_or_descriptor_root == column.column_root
            && super::catalog_column_matches_source(column, source),
        "NOT NULL guard does not exactly bind its target owner and S2 column",
    )
}

fn validate_table_check_expected(
    target: &SemanticsV2CatalogTableWitness<'_>,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
    raw_ordinal: usize,
) -> Result<(), crate::EngineError> {
    let raw_ordinal = u32::try_from(raw_ordinal)
        .map_err(|_| super::validation_error("CHECK guard ordinal exceeds canonical domain"))?;
    super::require(
        guard.kind == super::CHECK_GUARD
            && guard.owner_kind == 1
            && guard.owner_stable_id == target.stable_table_id
            && guard.owner_display_oid == target.display_oid
            && !guard.synthesized_not_null
            && guard.owner_catalog_column_ordinal == ABSENT_U32
            && guard.domain_ordinal == ABSENT_U32
            && guard.raw_constraint_ordinal == raw_ordinal
            && guard.source_ordinal == raw_ordinal,
        "CHECK guard does not exactly bind its target owner and raw ordinal",
    )
}

fn bound_domain_source<'a>(
    record: &'a crate::typed_insert_batch::DecodedTypedInsertRecord,
    target: &SemanticsV2CatalogTableWitness<'_>,
    domain: &SemanticsV2CatalogDomainWitness<'_>,
) -> Result<Option<crate::typed_insert_batch::DecodedDomainFacts<'a>>, crate::EngineError> {
    let mut matches = record.domains().filter(|source| {
        source.oid == domain.display_oid
            && source.schema == domain.schema
            && source.name == domain.name
            && super::storage_bytes(source.base_type) == domain.storage
            && source.base_type.postgres_oid() == domain.declared_type_oid
            && source.base_type.type_size() == domain.signed_type_size
    });
    let Some(source) = matches.next() else {
        return Ok(None);
    };
    super::require(
        matches.next().is_none()
            && record.catalog_columns().any(|column| {
                column.domain_ordinal == Some(source.ordinal)
                    && target.catalog_columns.iter().any(|catalog_column| {
                        super::catalog_column_matches_source(catalog_column, column)
                    })
            }),
        "pinned domain is ambiguous or not bound by an exact target S2/catalog column",
    )?;
    Ok(Some(source))
}

fn validate_domain_guard_expected(
    domain: &SemanticsV2CatalogDomainWitness<'_>,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
    domain_ordinal: usize,
    raw_ordinal: usize,
) -> Result<(), crate::EngineError> {
    let domain_ordinal = u32::try_from(domain_ordinal)
        .map_err(|_| super::validation_error("catalog domain ordinal exceeds canonical domain"))?;
    let raw_ordinal = u32::try_from(raw_ordinal)
        .map_err(|_| super::validation_error("domain guard ordinal exceeds canonical domain"))?;
    super::require(
        guard.kind == super::DOMAIN_CONSTRAINT_GUARD
            && guard.owner_kind == 2
            && guard.owner_stable_id == domain.stable_domain_id
            && guard.owner_display_oid == domain.display_oid
            && guard.owner_catalog_column_ordinal == ABSENT_U32
            && guard.domain_ordinal == domain_ordinal
            && guard.raw_constraint_ordinal == raw_ordinal
            && guard.source_ordinal == raw_ordinal,
        "domain guard does not exactly bind its pinned domain and raw ordinal",
    )
}

fn require_exact_expected_guard_use(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    resolution: &super::super::graph::RetainedStatementResolution,
    uses: &[super::super::graph::RetainedStatementDependencyUse],
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<(), crate::EngineError> {
    let mut count = 0_u32;
    for usage in uses.iter().filter(|usage| {
        usage.role == u16::from(guard.kind) && usage.source_ordinal == guard.source_ordinal
    }) {
        let token = graph
            .dependencies
            .get(usize::try_from(usage.dependency_ref).map_err(|_| {
                super::validation_error("guard use dependency reference is unaddressable")
            })?)
            .ok_or_else(|| super::validation_error("guard use dependency is absent"))?;
        if guard_matches_dependency(identity, guard, token)
            && token.target_table_ref == resolution.table_ref
        {
            super::require(
                if token.flags & TERMINAL_ERROR_FLAG != 0 {
                    resolution.terminal_dependency_ref == token.dependency_ref
                        && resolution.terminal_source_ordinal == guard.source_ordinal
                } else {
                    resolution.terminal_dependency_ref != token.dependency_ref
                },
                "terminal guard use does not replace its exact ordinary guard use",
            )?;
            count = count
                .checked_add(1)
                .ok_or_else(|| super::validation_error("expected guard-use count overflows"))?;
        }
    }
    super::require(
        count == 1,
        "expected catalog guard does not have exactly one statement use",
    )
}

fn validate_terminal_catalog_identity(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for resolution in &graph.resolutions {
        let outcome = graph
            .outcomes
            .get(usize::try_from(resolution.outcome_ref).map_err(|_| {
                super::validation_error("terminal resolution outcome reference is unaddressable")
            })?)
            .ok_or_else(|| super::validation_error("terminal resolution outcome is absent"))?;
        if outcome.outcome.kind != gpu_db_wal::CanonicalOutcomeKind::AbortError {
            continue;
        }
        let (_, target) = target_for_resolution(graph, catalog, resolution)?;
        let record = graph
            .records
            .get(usize::try_from(resolution.record_ref).map_err(|_| {
                super::validation_error("terminal resolution record reference is unaddressable")
            })?)
            .ok_or_else(|| super::validation_error("terminal resolution record is absent"))?;
        let token = graph
            .dependencies
            .get(
                usize::try_from(resolution.terminal_dependency_ref).map_err(|_| {
                    super::validation_error("terminal dependency reference is unaddressable")
                })?,
            )
            .ok_or_else(|| super::validation_error("terminal dependency is absent"))?;
        let uses = resolution_uses(graph, resolution)?;
        super::require(
            token.flags & TERMINAL_ERROR_FLAG != 0
                && terminal_use_count(uses, resolution, token) == 1
                && token.target_table_ref == resolution.table_ref,
            "terminal token does not have one exact terminal statement use",
        )?;
        match token.kind {
            super::NOT_NULL_GUARD | super::CHECK_GUARD | super::DOMAIN_CONSTRAINT_GUARD => {
                validate_terminal_catalog_guard(identity, catalog, token, outcome)?;
            }
            super::UNIQUE_KEY_GUARD => {
                validate_terminal_unique(
                    graph, record, target, catalog, token, resolution, outcome,
                )?;
            }
            super::FOREIGN_KEY_GUARD => {
                validate_terminal_foreign_key(
                    graph, record, target, catalog, token, resolution, outcome,
                )?;
            }
            _ => {
                return Err(super::validation_error(
                    "terminal token kind is not a catalog guard",
                ));
            }
        }
    }
    Ok(())
}

fn terminal_use_count(
    uses: &[super::super::graph::RetainedStatementDependencyUse],
    resolution: &super::super::graph::RetainedStatementResolution,
    token: &RetainedDependencyToken,
) -> usize {
    uses.iter()
        .filter(|usage| {
            usage.dependency_ref == token.dependency_ref
                && usage.role == u16::from(token.kind)
                && usage.source_ordinal == resolution.terminal_source_ordinal
        })
        .count()
}

fn validate_terminal_catalog_guard(
    identity: SemanticsV2BoundIdentity,
    catalog: &SemanticsV2CatalogWitness<'_>,
    token: &RetainedDependencyToken,
    outcome: &super::super::graph::RetainedStatementOutcome,
) -> Result<(), crate::EngineError> {
    let mut guards = catalog
        .guards
        .iter()
        .filter(|guard| guard_matches_dependency(identity, guard, token));
    let guard = guards
        .next()
        .ok_or_else(|| super::validation_error("terminal token has no pinned catalog guard"))?;
    super::require(
        guards.next().is_none(),
        "terminal token selects an ambiguous pinned catalog guard",
    )?;
    let expected_sqlstate = match guard.kind {
        super::NOT_NULL_GUARD => *b"23502",
        super::CHECK_GUARD => *b"23514",
        super::DOMAIN_CONSTRAINT_GUARD if guard.synthesized_not_null => *b"23502",
        super::DOMAIN_CONSTRAINT_GUARD => *b"23514",
        _ => {
            return Err(super::validation_error(
                "terminal catalog guard kind is invalid",
            ));
        }
    };
    super::require(
        outcome.outcome.sqlstate == Some(expected_sqlstate)
            && outcome.outcome.constraint_id == guard.stable_guard_id,
        "terminal catalog guard SQLSTATE or stable constraint identity is invalid",
    )
}

fn validate_terminal_unique(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    target: &SemanticsV2CatalogTableWitness<'_>,
    catalog: &SemanticsV2CatalogWitness<'_>,
    token: &RetainedDependencyToken,
    resolution: &super::super::graph::RetainedStatementResolution,
    outcome: &super::super::graph::RetainedStatementOutcome,
) -> Result<(), crate::EngineError> {
    let source = record
        .indexes()
        .find(|index| index.raw_ordinal == resolution.terminal_source_ordinal)
        .ok_or_else(|| super::validation_error("terminal unique source index is absent"))?;
    let descriptor = graph
        .indexes
        .get(usize::try_from(token.descriptor_ref).map_err(|_| {
            super::validation_error("terminal unique descriptor reference is unaddressable")
        })?)
        .ok_or_else(|| super::validation_error("terminal unique descriptor is absent"))?;
    let mut matches = catalog.indexes.iter().filter(|index| {
        index.stable_index_id == descriptor.stable_index_id
            && index.display_oid == descriptor.display_oid
            && index.owner_stable_table_id == target.stable_table_id
            && index.owner_display_oid == target.display_oid
            && index.raw_catalog_ordinal == source.raw_ordinal
            && super::catalog_index_matches_s2(index, source)
    });
    let index = matches.next().ok_or_else(|| {
        super::validation_error("terminal unique index is absent from pinned catalog")
    })?;
    super::require(
        matches.next().is_none()
            && descriptor.raw_catalog_ordinal == source.raw_ordinal
            && descriptor.owner_stable_table_id == target.stable_table_id
            && descriptor.owner_display_oid == target.display_oid
            && token_matches_pinned_index(token, descriptor, index)?
            && index_constraint_matches_descriptor(index, descriptor)?
            && outcome.outcome.sqlstate == Some(*b"23505")
            && outcome.outcome.constraint_id == terminal_unique_constraint_id(index),
        "terminal unique source, descriptor, or stable constraint identity is invalid",
    )
}

fn validate_terminal_foreign_key(
    graph: &ReservedSemanticsV2Graph,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    target: &SemanticsV2CatalogTableWitness<'_>,
    catalog: &SemanticsV2CatalogWitness<'_>,
    token: &RetainedDependencyToken,
    resolution: &super::super::graph::RetainedStatementResolution,
    outcome: &super::super::graph::RetainedStatementOutcome,
) -> Result<(), crate::EngineError> {
    let source = record
        .foreign_keys()
        .find(|foreign_key| foreign_key.raw_ordinal == resolution.terminal_source_ordinal)
        .ok_or_else(|| super::validation_error("terminal foreign-key source is absent"))?;
    let mut matches = target.foreign_keys.iter().filter(|foreign_key| {
        foreign_key.raw_foreign_key_ordinal == source.raw_ordinal
            && foreign_key.schema == target.schema
            && foreign_key.name == source.name
    });
    let foreign_key = matches.next().ok_or_else(|| {
        super::validation_error("terminal foreign key is absent from target catalog")
    })?;
    let descriptor = graph
        .indexes
        .get(usize::try_from(token.descriptor_ref).map_err(|_| {
            super::validation_error("terminal foreign-key descriptor reference is unaddressable")
        })?)
        .ok_or_else(|| super::validation_error("terminal foreign-key descriptor is absent"))?;
    let mut supporting_indexes = catalog.indexes.iter().filter(|index| {
        index.stable_index_id == foreign_key.supporting_stable_index_id
            && index.owner_stable_table_id == foreign_key.parent_stable_table_id
            && index.owner_display_oid == foreign_key.parent_display_oid
            && super::catalog_index_matches_s2(index, source.supporting_index)
    });
    let supporting_index = supporting_indexes.next().ok_or_else(|| {
        super::validation_error(
            "terminal foreign-key supporting index is absent from pinned catalog",
        )
    })?;
    super::require(
        supporting_indexes.next().is_none()
            && token_matches_pinned_index(token, descriptor, supporting_index)?
            && matches.next().is_none()
            && descriptor.raw_catalog_ordinal == source.supporting_index.raw_ordinal
            && descriptor.owner_stable_table_id == foreign_key.parent_stable_table_id
            && descriptor.owner_display_oid == foreign_key.parent_display_oid
            && outcome.outcome.sqlstate == Some(*b"23503")
            && outcome.outcome.constraint_id == foreign_key.stable_constraint_id,
        "terminal foreign key source, supporting index, or stable constraint identity is invalid",
    )
}

fn token_matches_pinned_index(
    token: &RetainedDependencyToken,
    descriptor: &super::super::graph::RetainedIndexDescriptor,
    index: &super::super::SemanticsV2CatalogIndexWitness<'_>,
) -> Result<bool, crate::EngineError> {
    Ok(token.descriptor_ref == descriptor.index_ref
        && token.stable_object_id == index.stable_index_id
        && token.display_oid == index.display_oid
        && token.catalog_epoch == index.catalog_epoch
        && token.base_generation == index.base_generation
        && token.schema_digest == index.schema_digest
        && token.base_root == index.base_root
        && token.name_digest == super::qualified_name_digest(index.schema, index.name)?
        && descriptor.stable_index_id == index.stable_index_id
        && descriptor.display_oid == index.display_oid
        && descriptor.catalog_epoch == index.catalog_epoch
        && descriptor.base_index_generation == index.base_generation
        && descriptor.owner_schema_digest == index.schema_digest
        && descriptor.owner_name_digest
            == super::qualified_name_digest(index.owner_schema, index.owner_name)?
        && descriptor.base_index_root == index.base_root
        && descriptor.index_name_digest == super::qualified_name_digest(index.schema, index.name)?)
}

fn index_constraint_matches_descriptor(
    index: &super::super::SemanticsV2CatalogIndexWitness<'_>,
    descriptor: &super::super::graph::RetainedIndexDescriptor,
) -> Result<bool, crate::EngineError> {
    if index.index_flags & 0b110 != 0 {
        Ok(
            descriptor.stable_constraint_id == index.constraint_stable_id
                && descriptor.constraint_display_oid == index.constraint_display_oid
                && descriptor.constraint_name_digest
                    == super::qualified_name_digest(
                        index.constraint_schema,
                        index.constraint_name,
                    )?,
        )
    } else {
        Ok(descriptor.stable_constraint_id == u64::MAX
            && descriptor.constraint_display_oid == 0
            && descriptor.constraint_name_digest == [0; 32])
    }
}

fn terminal_unique_constraint_id(index: &super::super::SemanticsV2CatalogIndexWitness<'_>) -> u64 {
    if index.index_flags & 0b110 != 0 {
        index.constraint_stable_id
    } else {
        index.stable_index_id
    }
}

pub(super) fn validate_target_foreign_keys(
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    super::require(
        record.foreign_keys().len() == catalog.foreign_keys.len(),
        "S2 target foreign-key count differs from the pinned catalog table",
    )?;
    for (source, catalog_fk) in record.foreign_keys().zip(catalog.foreign_keys) {
        super::require(
            catalog_fk_matches_source(
                graph,
                all_catalog,
                retained,
                record,
                catalog,
                catalog_fk,
                source,
            )?,
            "S2 foreign key differs from its exact pinned catalog closure",
        )?;
    }
    Ok(())
}

fn catalog_fk_matches_source(
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    owner: &SemanticsV2CatalogTableWitness<'_>,
    catalog: &SemanticsV2CatalogForeignKeyWitness<'_>,
    source: DecodedForeignKeyFacts<'_>,
) -> Result<bool, crate::EngineError> {
    if catalog.raw_foreign_key_ordinal != source.raw_ordinal
        || catalog.schema != owner.schema
        || catalog.name != source.name
        || !owner.catalog_columns.iter().any(|column| {
            super::catalog_column_matches_binding(column, source.child_column)
                && column.catalog_column_ordinal == catalog.child_catalog_column_ordinal
                && column.stable_column_id == catalog.child_stable_column_id
        })
    {
        return Ok(false);
    }
    let parent_dependency = record
        .dependencies()
        .nth(
            usize::try_from(source.parent_dependency_ordinal).map_err(|_| {
                super::validation_error("FK parent dependency ordinal exceeds addressability")
            })?,
        )
        .ok_or_else(|| super::validation_error("FK parent dependency is absent"))?;
    let Some(parent) = all_catalog.tables.iter().find(|table| {
        table.stable_table_id == catalog.parent_stable_table_id
            && table.display_oid == catalog.parent_display_oid
    }) else {
        return Ok(false);
    };
    if parent.schema != parent_dependency.schema
        || parent.name != parent_dependency.name
        || parent.schema_digest != parent_dependency.schema_digest
        || !parent.catalog_columns.iter().any(|column| {
            super::catalog_column_matches_binding(column, source.parent_column)
                && column.catalog_column_ordinal == catalog.parent_catalog_column_ordinal
                && column.stable_column_id == catalog.parent_stable_column_id
        })
    {
        return Ok(false);
    }
    let supporting_index = all_catalog
        .indexes
        .iter()
        .find(|index| index.stable_index_id == catalog.supporting_stable_index_id);
    let Some(supporting_index) = supporting_index else {
        return Ok(false);
    };
    if !super::catalog_index_matches_s2(supporting_index, source.supporting_index)
        || supporting_index.owner_stable_table_id != catalog.parent_stable_table_id
        || supporting_index.owner_display_oid != catalog.parent_display_oid
        || !graph.dependencies.iter().any(|dependency| {
            dependency.kind == super::FOREIGN_KEY_GUARD
                && dependency.stable_object_id == supporting_index.stable_index_id
                && dependency.display_oid == supporting_index.display_oid
                && dependency.base_generation == supporting_index.base_generation
                && dependency.schema_digest == supporting_index.schema_digest
                && dependency.base_root == supporting_index.base_root
                && super::qualified_name_digest(supporting_index.schema, supporting_index.name)
                    .is_ok_and(|digest| dependency.name_digest == digest)
                && graph.indexes.iter().any(|descriptor| {
                    dependency.descriptor_ref == descriptor.index_ref
                        && descriptor.stable_index_id == supporting_index.stable_index_id
                        && descriptor.display_oid == supporting_index.display_oid
                        && descriptor.descriptor_digest != [0; 32]
                })
                && dependency.target_table_ref == retained.table_ref
        })
    {
        return Ok(false);
    }
    Ok(true)
}

fn guard_name_digest(
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<[u8; 32], crate::EngineError> {
    if guard.synthesized_not_null {
        let source_ordinal = match guard.owner_kind {
            1 => guard.owner_catalog_column_ordinal,
            2 => 0,
            _ => {
                return Err(super::validation_error(
                    "synthesized guard owner kind is invalid",
                ));
            }
        };
        Ok(super::v2_digest(
            b"gpu-db/write001/s7-synthesized-not-null-name/v2",
            &[
                &[guard.owner_kind],
                &guard.owner_stable_id.to_le_bytes(),
                &source_ordinal.to_le_bytes(),
            ],
        ))
    } else {
        super::qualified_name_digest(guard.schema, guard.name)
    }
}

fn same_guard(
    left: &SemanticsV2CatalogGuardWitness<'_>,
    right: &SemanticsV2CatalogGuardWitness<'_>,
) -> bool {
    left.kind == right.kind
        && left.stable_guard_id == right.stable_guard_id
        && left.display_oid == right.display_oid
        && left.schema == right.schema
        && left.name == right.name
        && left.synthesized_not_null == right.synthesized_not_null
        && left.owner_kind == right.owner_kind
        && left.owner_stable_id == right.owner_stable_id
        && left.owner_display_oid == right.owner_display_oid
        && left.owner_catalog_column_ordinal == right.owner_catalog_column_ordinal
        && left.domain_ordinal == right.domain_ordinal
        && left.raw_constraint_ordinal == right.raw_constraint_ordinal
        && left.source_ordinal == right.source_ordinal
        && left.shape_digest == right.shape_digest
        && left.program_or_descriptor_root == right.program_or_descriptor_root
        && left.catalog_generation == right.catalog_generation
}
