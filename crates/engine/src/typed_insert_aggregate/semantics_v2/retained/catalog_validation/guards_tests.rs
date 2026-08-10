use super::*;
use crate::typed_insert_aggregate::semantics_v2::{
    pass_zero::SemanticsV2S7HeaderIdentity,
    retained::{
        graph::{
            ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedIndexDescriptor,
            RetainedStatementDependencyUse, RetainedStatementOutcome, RetainedStatementResolution,
            RetainedTable,
        },
        SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness, SemanticsV2CatalogDomainWitness,
        SemanticsV2CatalogForeignKeyWitness, SemanticsV2CatalogGuardWitness,
        SemanticsV2CatalogIndexKeyWitness, SemanticsV2CatalogIndexWitness,
        SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
    },
};

const ABSENT: u32 = u32::MAX;
const TEST_CATALOG_EPOCH: u64 = 7;
const TEST_TABLE_ID: u64 = 101;
const TEST_TABLE_OID: u32 = 16_384;
const TEST_GUARD_ID: u64 = 501;
const UNIQUE_KEY_GUARD: u8 = super::super::UNIQUE_KEY_GUARD;
const NOT_NULL_GUARD: u8 = super::super::NOT_NULL_GUARD;
const CHECK_GUARD: u8 = super::super::CHECK_GUARD;
const DOMAIN_CONSTRAINT_GUARD: u8 = super::super::DOMAIN_CONSTRAINT_GUARD;
const FOREIGN_KEY_GUARD: u8 = super::super::FOREIGN_KEY_GUARD;
const ONE_INDEX_KEY: [SemanticsV2CatalogIndexKeyWitness<'static>; 1] =
    [SemanticsV2CatalogIndexKeyWitness {
        key_ordinal: 0,
        owner_catalog_column_ordinal: 0,
        stable_column_id: 1,
        attnum: 1,
        name: "key",
        storage: [2, 0, 0, 0],
        declared_type_oid: 23,
        signed_type_size: 4,
        column_name_digest: [0; 32],
    }];
const NOT_NULL_S2_HEX: &str = "47505544425459504544494e53310000010001000100010000000000080000005c01000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e44929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b010000004a000000060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c14610000000001000000010000000200000046000000010000000000000002000000696401000000010002000000170000000400010000000000010100000000000000000001000000020200000000010100000004000000000000000300000047000000010000000000000001060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c1461040000000400000000000000050000000400000000000000060000000400000000000000070000003400000001000000000000000000000000000000929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b0000000008000000050000000000000000";

#[test]
fn not_null_inventory_rejects_owner_source_and_use_sabotage() {
    let mut graph = graph_with_not_null_record();
    let identity = bound_identity();
    let source = graph.records[0]
        .catalog_columns()
        .next()
        .expect("frozen test S2 has one catalog column");
    let target = graph.records[0].target_identity();
    assert_eq!(source.catalog_column_ordinal, 0);
    assert_eq!(target.oid, TEST_TABLE_OID);

    let columns = [catalog_column(source)];
    let owner_table = SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid: TEST_TABLE_OID,
        schema: target.schema,
        name: target.name,
        schema_digest: target.schema_digest,
        data_generation: 11,
        data_root: [4; 32],
        catalog_columns: &columns,
        not_null_guards: &[],
        check_guards: &[],
        foreign_keys: &[],
    };
    let mut owner_swapped = not_null_guard();
    owner_swapped.owner_stable_id += 1;
    assert!(
        validate_table_not_null_expected(&owner_table, &graph.records[0], &owner_swapped).is_err()
    );

    let guards = [not_null_guard()];
    let table = SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid: TEST_TABLE_OID,
        schema: target.schema,
        name: target.name,
        schema_digest: target.schema_digest,
        data_generation: 11,
        data_root: [4; 32],
        catalog_columns: &columns,
        not_null_guards: &guards,
        check_guards: &[],
        foreign_keys: &[],
    };
    let tables = [table];
    let catalog = empty_catalog(&tables);

    validate_statement_guard_bijections(identity, &graph, &catalog)
        .expect("one exact target-column NOT NULL use closes");

    graph.dependency_uses[0].source_ordinal = 1;
    let missing = validate_statement_guard_bijections(identity, &graph, &catalog)
        .expect_err("redirecting the only use leaves the expected guard missing");
    assert!(missing
        .to_string()
        .contains("expected catalog guard does not have exactly one statement use"));
    graph.dependency_uses[0].source_ordinal = 0;

    graph.dependency_uses.push(RetainedStatementDependencyUse {
        statement_ordinal: 0,
        dependency_ref: 0,
        role: u16::from(NOT_NULL_GUARD),
        source_ordinal: 0,
        transition_ref: ABSENT,
        key_effect_ref: ABSENT,
    });
    graph.resolutions[0].dependency_use_count = 2;
    assert!(validate_statement_guard_bijections(identity, &graph, &catalog).is_err());
    graph.dependency_uses.pop();
    graph.resolutions[0].dependency_use_count = 1;
    validate_statement_guard_bijections(identity, &graph, &catalog)
        .expect("fresh ordinary use accepts after missing and extra-use failures");
}

#[test]
fn terminal_not_null_closure_requires_one_terminal_replacement_use() {
    terminal_not_null_guard_closure(|_| {})
        .expect("terminal NOT NULL dispatch closes through the full guard closure");

    assert!(
        terminal_not_null_guard_closure(|graph| graph.dependencies[0].flags = 0).is_err(),
        "an ordinary guard token cannot be selected as the terminal replacement"
    );
    assert!(
        terminal_not_null_guard_closure(|graph| {
            graph.resolutions[0].terminal_dependency_ref = ABSENT;
        })
        .is_err(),
        "the terminal resolution must select the exact terminal guard token"
    );
    assert!(
        terminal_not_null_guard_closure(|graph| {
            graph.dependencies.push(guard_token(&not_null_guard(), 1));
            graph.dependency_uses.push(RetainedStatementDependencyUse {
                statement_ordinal: 0,
                dependency_ref: 1,
                role: u16::from(NOT_NULL_GUARD),
                source_ordinal: 0,
                transition_ref: ABSENT,
                key_effect_ref: ABSENT,
            });
            graph.resolutions[0].dependency_use_count = 2;
        })
        .is_err(),
        "terminal and ordinary uses of the same guard cannot coexist"
    );

    terminal_not_null_guard_closure(|_| {})
        .expect("a fresh terminal NOT NULL owner accepts after hostile failures");
}

#[test]
fn table_check_and_domain_guards_reject_owner_and_ordinal_redirects() {
    let target = test_table();
    let mut check = SemanticsV2CatalogGuardWitness {
        kind: CHECK_GUARD,
        stable_guard_id: 502,
        display_oid: 702,
        schema: "public",
        name: "table_check",
        synthesized_not_null: false,
        owner_kind: 1,
        owner_stable_id: TEST_TABLE_ID,
        owner_display_oid: TEST_TABLE_OID,
        owner_catalog_column_ordinal: ABSENT,
        domain_ordinal: ABSENT,
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest: [5; 32],
        program_or_descriptor_root: [6; 32],
        catalog_generation: 12,
    };
    validate_table_check_expected(&target, &check, 0).expect("table CHECK is exact");
    check.owner_display_oid += 1;
    assert!(validate_table_check_expected(&target, &check, 0).is_err());
    check.owner_display_oid = TEST_TABLE_OID;
    check.source_ordinal = 1;
    assert!(validate_table_check_expected(&target, &check, 0).is_err());

    let domain = SemanticsV2CatalogDomainWitness {
        stable_domain_id: 801,
        display_oid: 901,
        schema: "public",
        name: "inventory_domain",
        storage: [2, 0, 0, 0],
        declared_type_oid: 23,
        signed_type_size: 4,
        storage_shape_digest: [7; 32],
        catalog_generation: 14,
        constraints: &[],
    };
    let mut guard = SemanticsV2CatalogGuardWitness {
        kind: DOMAIN_CONSTRAINT_GUARD,
        stable_guard_id: 802,
        display_oid: 902,
        schema: "public",
        name: "inventory_domain_check",
        synthesized_not_null: false,
        owner_kind: 2,
        owner_stable_id: domain.stable_domain_id,
        owner_display_oid: domain.display_oid,
        owner_catalog_column_ordinal: ABSENT,
        domain_ordinal: 1,
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest: [8; 32],
        program_or_descriptor_root: [9; 32],
        catalog_generation: domain.catalog_generation,
    };
    validate_domain_guard_expected(&domain, &guard, 1, 0).expect("domain CHECK is exact");
    guard.domain_ordinal = 0;
    assert!(validate_domain_guard_expected(&domain, &guard, 1, 0).is_err());
    guard.domain_ordinal = 1;
    guard.raw_constraint_ordinal = 1;
    assert!(validate_domain_guard_expected(&domain, &guard, 1, 0).is_err());
}

#[test]
fn check_inventory_bijection_rejects_a_source_ordinal_swap() {
    let mut graph = graph_with_not_null_record();
    let target = graph.records[0].target_identity();
    let columns = [catalog_column(
        graph.records[0]
            .catalog_columns()
            .next()
            .expect("fixture record has its target column"),
    )];
    let check_zero = table_check_guard(5_101, 7_101, 0);
    let check_one = table_check_guard(5_102, 7_102, 1);
    let checks = [check_zero, check_one];
    let table = SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid: target.oid,
        schema: target.schema,
        name: target.name,
        schema_digest: target.schema_digest,
        data_generation: 11,
        data_root: [4; 32],
        catalog_columns: &columns,
        not_null_guards: &[],
        check_guards: &checks,
        foreign_keys: &[],
    };
    let tables = [table];
    let catalog = empty_catalog(&tables);
    graph.dependencies = vec![guard_token(&checks[0], 0), guard_token(&checks[1], 1)];
    graph.dependency_uses = vec![
        RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 0,
            role: u16::from(CHECK_GUARD),
            source_ordinal: 0,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        },
        RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 1,
            role: u16::from(CHECK_GUARD),
            source_ordinal: 1,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        },
    ];
    graph.resolutions[0].dependency_use_count = 2;
    validate_statement_guard_bijections(bound_identity(), &graph, &catalog)
        .expect("each table CHECK has one exact source-ordinal use");

    graph.dependency_uses[0].source_ordinal = 1;
    graph.dependency_uses[1].source_ordinal = 0;
    assert!(
        validate_statement_guard_bijections(bound_identity(), &graph, &catalog).is_err(),
        "swapping CHECK use source ordinals breaks the full exact guard bijection"
    );
    graph.dependency_uses[0].source_ordinal = 0;
    graph.dependency_uses[1].source_ordinal = 1;
    validate_statement_guard_bijections(bound_identity(), &graph, &catalog)
        .expect("fresh CHECK source-order retry accepts after the swap");
}

#[test]
fn terminal_catalog_guards_have_exact_sqlstate_and_stable_identity() {
    for (kind, synthesized_not_null, sqlstate) in [
        (NOT_NULL_GUARD, true, *b"23502"),
        (CHECK_GUARD, false, *b"23514"),
        (DOMAIN_CONSTRAINT_GUARD, true, *b"23502"),
        (DOMAIN_CONSTRAINT_GUARD, false, *b"23514"),
    ] {
        let guard = SemanticsV2CatalogGuardWitness {
            kind,
            stable_guard_id: u64::from(kind) + 900,
            display_oid: if synthesized_not_null {
                0
            } else {
                u32::from(kind) + 1000
            },
            schema: if synthesized_not_null { "" } else { "public" },
            name: if synthesized_not_null {
                ""
            } else {
                "guard_name"
            },
            synthesized_not_null,
            owner_kind: if kind == DOMAIN_CONSTRAINT_GUARD {
                2
            } else {
                1
            },
            owner_stable_id: 700 + u64::from(kind),
            owner_display_oid: 800 + u32::from(kind),
            owner_catalog_column_ordinal: if kind == NOT_NULL_GUARD { 0 } else { ABSENT },
            domain_ordinal: if kind == DOMAIN_CONSTRAINT_GUARD {
                0
            } else {
                ABSENT
            },
            raw_constraint_ordinal: 0,
            source_ordinal: 0,
            shape_digest: [kind; 32],
            program_or_descriptor_root: [kind + 1; 32],
            catalog_generation: 15,
        };
        let guards = [guard];
        let catalog = SemanticsV2CatalogWitness {
            database_id: [0; 16],
            catalog_epoch: TEST_CATALOG_EPOCH,
            catalog_digest: [0; 32],
            tables: &[],
            indexes: &[],
            domains: &[],
            guards: &guards,
            sequences: &[],
        };
        let token = guard_token(&guards[0], 0);
        let mut outcome = abort_outcome(sqlstate, guards[0].stable_guard_id);
        validate_terminal_catalog_guard(bound_identity(), &catalog, &token, &outcome)
            .expect("exact terminal guard class closes");
        if kind == DOMAIN_CONSTRAINT_GUARD {
            outcome.outcome.sqlstate = Some(if synthesized_not_null {
                *b"23514"
            } else {
                *b"23502"
            });
            assert!(
                validate_terminal_catalog_guard(bound_identity(), &catalog, &token, &outcome)
                    .is_err(),
                "domain terminal class rejects the reversed SQLSTATE"
            );
            outcome.outcome.sqlstate = Some(sqlstate);
        }
        outcome.outcome.constraint_id += 1;
        assert!(
            validate_terminal_catalog_guard(bound_identity(), &catalog, &token, &outcome).is_err()
        );
    }
}

#[test]
fn synthesized_guard_order_requires_empty_name_and_a_valid_display_oid() {
    for kind in [NOT_NULL_GUARD, DOMAIN_CONSTRAINT_GUARD] {
        validate_guard_order(&[synthesized_guard_order_row(kind, 0, "", "")])
            .expect("synthesized guard may omit its catalog display OID");
        validate_guard_order(&[synthesized_guard_order_row(kind, 7_777, "", "")])
            .expect("synthesized guard may retain a real valid catalog display OID");
        assert!(
            validate_guard_order(&[synthesized_guard_order_row(kind, 7_777, "public", "")])
                .is_err(),
            "synthesized guard rejects a schema without a name"
        );
        assert!(
            validate_guard_order(&[synthesized_guard_order_row(kind, 7_777, "", "named")]).is_err(),
            "synthesized guard rejects a name without a schema"
        );
        assert!(
            validate_guard_order(&[synthesized_guard_order_row(kind, 0x8000_0000, "", "",)])
                .is_err(),
            "synthesized guard rejects a display OID outside PostgreSQL's signed domain"
        );
        validate_guard_order(&[synthesized_guard_order_row(kind, 7_777, "", "")])
            .expect("fresh synthesized guard order retry accepts after hostile rows");
    }
}

#[test]
fn terminal_index_tokens_require_the_selected_pinned_index_and_constraint_form() {
    let (constraint_index, constraint_descriptor, constraint_token) = index_fixture(true);
    assert!(token_matches_pinned_index(
        &constraint_token,
        &constraint_descriptor,
        &constraint_index
    )
    .expect("test index name is valid"));
    assert!(
        index_constraint_matches_descriptor(&constraint_index, &constraint_descriptor)
            .expect("test constraint name is valid")
    );
    assert_eq!(terminal_unique_constraint_id(&constraint_index), 601);

    let (plain_index, plain_descriptor, plain_token) = index_fixture(false);
    assert!(
        token_matches_pinned_index(&plain_token, &plain_descriptor, &plain_index)
            .expect("test index name is valid")
    );
    assert!(
        index_constraint_matches_descriptor(&plain_index, &plain_descriptor)
            .expect("plain index uses the S7 absent constraint form")
    );
    assert_eq!(
        terminal_unique_constraint_id(&plain_index),
        plain_index.stable_index_id
    );

    let mut redirected = plain_token;
    redirected.catalog_epoch += 1;
    assert!(
        !token_matches_pinned_index(&redirected, &plain_descriptor, &plain_index)
            .expect("test index name is valid")
    );
    redirected.catalog_epoch -= 1;
    redirected.descriptor_ref = 1;
    assert!(
        !token_matches_pinned_index(&redirected, &plain_descriptor, &plain_index)
            .expect("test index name is valid")
    );
}

#[test]
fn terminal_unique_uses_the_exact_s2_index_and_stable_constraint_form() {
    let mut graph = graph_with_catalog_closure_record();
    let target = graph.records[0].target_identity();
    let target_oid = target.oid;
    let target_schema = target.schema.to_owned();
    let target_name = target.name.to_owned();
    let target_schema_digest = target.schema_digest;
    let constraint_source = owned_index_source(&graph.records[0], |source| {
        source.unique && (source.primary_key || source.unique_constraint)
    });
    let plain_source = owned_index_source(&graph.records[0], |source| {
        source.unique && !source.primary_key && !source.unique_constraint
    });
    graph.tables.push(retained_table(TEST_TABLE_ID, target_oid));

    for (source, constraint_backed, stable_index_id, stable_constraint_id) in [
        (constraint_source, true, 6_001, 6_101),
        (plain_source, false, 6_002, 0),
    ] {
        assert_eq!(source.key_count, 1, "terminal unique fixture is single-key");
        let index = catalog_index(
            &source,
            stable_index_id,
            IndexOwner {
                stable_table_id: TEST_TABLE_ID,
                display_oid: target_oid,
                schema: &target_schema,
                name: &target_name,
            },
            stable_constraint_id,
            constraint_backed,
        );
        let descriptor = descriptor_for_index(&index, constraint_backed);
        let token = terminal_index_token(UNIQUE_KEY_GUARD, &index, 0);
        graph.indexes.clear();
        graph.dependencies.clear();
        graph.dependency_uses.clear();
        graph.outcomes.clear();
        graph.resolutions.clear();
        graph.indexes.push(descriptor);
        graph.dependencies.push(token);
        graph.dependency_uses.push(RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 0,
            role: u16::from(UNIQUE_KEY_GUARD),
            source_ordinal: source.raw_ordinal,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        });
        graph.outcomes.push(abort_outcome(
            *b"23505",
            if constraint_backed {
                stable_constraint_id
            } else {
                stable_index_id
            },
        ));
        graph
            .resolutions
            .push(terminal_resolution(source.raw_ordinal));

        let table = terminal_target_table(
            target_oid,
            &target_schema,
            &target_name,
            target_schema_digest,
            &[],
        );
        let tables = [table];
        let indexes = [index];
        let catalog = SemanticsV2CatalogWitness {
            database_id: [0; 16],
            catalog_epoch: TEST_CATALOG_EPOCH,
            catalog_digest: [0; 32],
            tables: &tables,
            indexes: &indexes,
            domains: &[],
            guards: &[],
            sequences: &[],
        };
        validate_terminal_catalog_identity(bound_identity(), &graph, &catalog)
            .expect("terminal dispatch selects the exact S2/catalog UNIQUE index");

        graph.outcomes[0].outcome.constraint_id = if constraint_backed {
            stable_index_id
        } else {
            stable_index_id + 1
        };
        assert!(
            validate_terminal_catalog_identity(bound_identity(), &graph, &catalog).is_err(),
            "terminal UNIQUE rejects the wrong stable constraint form"
        );
        graph.outcomes[0].outcome.constraint_id = if constraint_backed {
            stable_constraint_id
        } else {
            stable_index_id
        };
        validate_terminal_catalog_identity(bound_identity(), &graph, &catalog)
            .expect("fresh terminal UNIQUE dispatch accepts after the hostile outcome");
    }
}

#[test]
fn terminal_foreign_key_uses_exact_support_and_fk_stable_constraint() {
    let mut graph = graph_with_catalog_closure_record();
    let target = graph.records[0].target_identity();
    let target_oid = target.oid;
    let target_schema = target.schema.to_owned();
    let target_name = target.name.to_owned();
    let target_schema_digest = target.schema_digest;
    let [source, other_source] = owned_foreign_key_sources(&graph.records[0]);
    assert_eq!(source.supporting.key_count, 1, "FK support is single-key");
    let parent_stable_id = 7_001;
    let supporting_stable_index_id = 7_101;
    let foreign_key_stable_id = 7_201;
    let support = catalog_index(
        &source.supporting,
        supporting_stable_index_id,
        IndexOwner {
            stable_table_id: parent_stable_id,
            display_oid: 70_001,
            schema: &target_schema,
            name: &source.supporting.owner_name,
        },
        supporting_stable_index_id,
        true,
    );
    let descriptor = descriptor_for_index(&support, true);
    let token = terminal_index_token(FOREIGN_KEY_GUARD, &support, 0);
    graph.tables.push(retained_table(TEST_TABLE_ID, target_oid));
    graph.indexes.push(descriptor);
    graph.dependencies.push(token);
    graph.dependency_uses.push(RetainedStatementDependencyUse {
        statement_ordinal: 0,
        dependency_ref: 0,
        role: u16::from(FOREIGN_KEY_GUARD),
        source_ordinal: source.raw_ordinal,
        transition_ref: ABSENT,
        key_effect_ref: ABSENT,
    });
    graph
        .outcomes
        .push(abort_outcome(*b"23503", foreign_key_stable_id));
    graph
        .resolutions
        .push(terminal_resolution(source.raw_ordinal));

    let other_foreign_key_stable_id = 7_202;
    let foreign_keys = [
        foreign_key_witness(
            &source,
            foreign_key_stable_id,
            72_001,
            &target_schema,
            parent_stable_id,
            supporting_stable_index_id,
        ),
        foreign_key_witness(
            &other_source,
            other_foreign_key_stable_id,
            72_002,
            &target_schema,
            parent_stable_id,
            supporting_stable_index_id,
        ),
    ];
    let table = terminal_target_table(
        target_oid,
        &target_schema,
        &target_name,
        target_schema_digest,
        &foreign_keys,
    );
    let tables = [table];
    let indexes = [support];
    let catalog = SemanticsV2CatalogWitness {
        database_id: [0; 16],
        catalog_epoch: TEST_CATALOG_EPOCH,
        catalog_digest: [0; 32],
        tables: &tables,
        indexes: &indexes,
        domains: &[],
        guards: &[],
        sequences: &[],
    };
    validate_terminal_catalog_identity(bound_identity(), &graph, &catalog)
        .expect("terminal dispatch selects the exact FK supporting index and constraint");

    for wrong_constraint_id in [supporting_stable_index_id, other_foreign_key_stable_id] {
        graph.outcomes[0].outcome.constraint_id = wrong_constraint_id;
        assert!(
            validate_terminal_catalog_identity(bound_identity(), &graph, &catalog).is_err(),
            "terminal FK rejects a supporting-index or another actual FK constraint ID"
        );
    }
    graph.outcomes[0].outcome.constraint_id = foreign_key_stable_id;
    validate_terminal_catalog_identity(bound_identity(), &graph, &catalog)
        .expect("fresh terminal FK dispatch accepts after the hostile outcomes");
}

#[test]
fn same_shape_domains_require_their_own_s2_column_and_statement_guard_use() {
    let mut graph = graph_with_same_shape_domain_record();
    let target = graph.records[0].target_identity();
    let target_oid = target.oid;
    let target_schema = target.schema.to_owned();
    let target_name = target.name.to_owned();
    let target_schema_digest = target.schema_digest;
    let sources = owned_domain_sources(&graph.records[0]);
    assert_eq!(
        sources[0].storage, sources[1].storage,
        "domains intentionally share shape"
    );
    assert_eq!(sources[0].declared_type_oid, sources[1].declared_type_oid);

    let columns = [
        owned_catalog_column(&sources[0].column),
        owned_catalog_column(&sources[1].column),
    ];
    // Catalog order is intentionally the reverse of S2 domain order.  The increasing stable
    // IDs still make that catalog slice plausible, while `domain_ordinal` remains catalog-local
    // rather than mirroring the S2 domain ordinal.
    let domain_b_guard = domain_check_guard(&sources[1], 8_101, 8_201, 0);
    let domain_a_guard = domain_check_guard(&sources[0], 8_102, 8_202, 1);
    let domain_a_guards = [domain_a_guard];
    let domain_b_guards = [domain_b_guard];
    let domains = [
        catalog_domain(&sources[1], 8_001, &domain_b_guards),
        catalog_domain(&sources[0], 8_002, &domain_a_guards),
    ];
    assert_ne!(
        domain_b_guards[0].domain_ordinal, sources[1].ordinal,
        "domain B uses its catalog slice ordinal, not its S2 ordinal"
    );
    assert_ne!(
        domain_a_guards[0].domain_ordinal, sources[0].ordinal,
        "domain A uses its catalog slice ordinal, not its S2 ordinal"
    );
    graph.tables.push(retained_table(TEST_TABLE_ID, target_oid));
    // S7 keeps the S2 identity order (A then B), independently of the reversed catalog slice.
    graph.dependencies.push(guard_token(&domain_a_guards[0], 0));
    graph.dependencies.push(guard_token(&domain_b_guards[0], 1));
    graph.dependency_uses.extend([
        RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 0,
            role: u16::from(DOMAIN_CONSTRAINT_GUARD),
            source_ordinal: 0,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        },
        RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 1,
            role: u16::from(DOMAIN_CONSTRAINT_GUARD),
            source_ordinal: 0,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        },
    ]);
    graph.resolutions.push(statement_resolution(2));
    let table = terminal_target_table(
        target_oid,
        &target_schema,
        &target_name,
        target_schema_digest,
        &[],
    );
    let table = SemanticsV2CatalogTableWitness {
        catalog_columns: &columns,
        ..table
    };
    let tables = [table];
    let catalog = SemanticsV2CatalogWitness {
        database_id: [0; 16],
        catalog_epoch: TEST_CATALOG_EPOCH,
        catalog_digest: [0; 32],
        tables: &tables,
        indexes: &[],
        domains: &domains,
        guards: &[],
        sequences: &[],
    };
    validate_statement_guard_bijections(bound_identity(), &graph, &catalog)
        .expect("same-shape domains each close through their own exact S2 target column");

    graph.dependency_uses[1].dependency_ref = 0;
    assert!(
        validate_statement_guard_bijections(bound_identity(), &graph, &catalog).is_err(),
        "an actually-bound same-shape domain cannot redirect its guard use to the other domain"
    );
    graph.dependency_uses[1].dependency_ref = 1;
    validate_statement_guard_bijections(bound_identity(), &graph, &catalog)
        .expect("fresh same-shape domain retry accepts after redirect rejection");
}

#[derive(Clone)]
struct OwnedIndexSource {
    raw_ordinal: u32,
    oid: u32,
    name: String,
    owner_name: String,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
    key_count: u32,
}

struct IndexOwner<'a> {
    stable_table_id: u64,
    display_oid: u32,
    schema: &'a str,
    name: &'a str,
}

struct OwnedForeignKeySource {
    raw_ordinal: u32,
    name: String,
    supporting: OwnedIndexSource,
}

struct OwnedColumnSource {
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    name: String,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
}

struct OwnedDomainSource {
    ordinal: u32,
    oid: u32,
    schema: String,
    name: String,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
    column: OwnedColumnSource,
}

fn owned_index_source(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    select: impl Fn(crate::typed_insert_batch::DecodedIndexFacts<'_>) -> bool,
) -> OwnedIndexSource {
    let source = record
        .indexes()
        .find(|source| select(*source))
        .expect("sealed S2 fixture has the selected index source");
    OwnedIndexSource {
        raw_ordinal: source.raw_ordinal,
        oid: source.oid,
        name: source.name.to_owned(),
        owner_name: source.table_name.to_owned(),
        unique: source.unique,
        primary_key: source.primary_key,
        unique_constraint: source.unique_constraint,
        key_count: source.key_count,
    }
}

fn owned_foreign_key_sources(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> [OwnedForeignKeySource; 2] {
    let mut sources = record.foreign_keys();
    let first = sources
        .next()
        .expect("sealed S2 fixture has its first foreign key");
    let second = sources
        .next()
        .expect("sealed S2 fixture has its second foreign key");
    assert!(
        sources.next().is_none(),
        "fixture has exactly two foreign keys"
    );
    [first, second].map(|source| OwnedForeignKeySource {
        raw_ordinal: source.raw_ordinal,
        name: source.name.to_owned(),
        supporting: OwnedIndexSource {
            raw_ordinal: source.supporting_index.raw_ordinal,
            oid: source.supporting_index.oid,
            name: source.supporting_index.name.to_owned(),
            owner_name: source.supporting_index.table_name.to_owned(),
            unique: source.supporting_index.unique,
            primary_key: source.supporting_index.primary_key,
            unique_constraint: source.supporting_index.unique_constraint,
            key_count: source.supporting_index.key_count,
        },
    })
}

fn foreign_key_witness<'a>(
    source: &'a OwnedForeignKeySource,
    stable_constraint_id: u64,
    display_oid: u32,
    schema: &'a str,
    parent_stable_table_id: u64,
    supporting_stable_index_id: u64,
) -> SemanticsV2CatalogForeignKeyWitness<'a> {
    SemanticsV2CatalogForeignKeyWitness {
        stable_constraint_id,
        display_oid,
        raw_foreign_key_ordinal: source.raw_ordinal,
        schema,
        name: &source.name,
        child_catalog_column_ordinal: 0,
        child_stable_column_id: 1,
        parent_stable_table_id,
        parent_display_oid: 70_001,
        parent_catalog_column_ordinal: 0,
        parent_stable_column_id: 1,
        supporting_stable_index_id,
    }
}

fn owned_domain_sources(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> [OwnedDomainSource; 2] {
    let mut domains = record.domains();
    let first = domains
        .next()
        .expect("sealed S2 has first same-shape domain");
    let second = domains
        .next()
        .expect("sealed S2 has second same-shape domain");
    assert!(
        domains.next().is_none(),
        "fixture contains exactly two domains"
    );
    [first, second].map(|domain| {
        let column = record
            .catalog_columns()
            .find(|column| column.domain_ordinal == Some(domain.ordinal))
            .expect("each domain has its exact S2 target column");
        OwnedDomainSource {
            ordinal: domain.ordinal,
            oid: domain.oid,
            schema: domain.schema.to_owned(),
            name: domain.name.to_owned(),
            storage: super::super::storage_bytes(domain.base_type),
            declared_type_oid: domain.base_type.postgres_oid(),
            signed_type_size: domain.base_type.type_size(),
            column: OwnedColumnSource {
                catalog_column_ordinal: column.catalog_column_ordinal,
                stable_column_id: column.column_id,
                attnum: column.attnum,
                name: column.name.to_owned(),
                storage: super::super::storage_bytes(column.ty),
                declared_type_oid: column.type_oid,
                signed_type_size: column.type_size,
            },
        }
    })
}

fn catalog_index<'a>(
    source: &'a OwnedIndexSource,
    stable_index_id: u64,
    owner: IndexOwner<'a>,
    stable_constraint_id: u64,
    constraint_backed: bool,
) -> SemanticsV2CatalogIndexWitness<'a> {
    SemanticsV2CatalogIndexWitness {
        stable_index_id,
        display_oid: source.oid,
        owner_stable_table_id: owner.stable_table_id,
        owner_display_oid: owner.display_oid,
        schema: owner.schema,
        name: &source.name,
        owner_schema: owner.schema,
        owner_name: owner.name,
        constraint_stable_id: if constraint_backed {
            stable_constraint_id
        } else {
            0
        },
        constraint_display_oid: if constraint_backed { source.oid } else { 0 },
        constraint_schema: if constraint_backed { owner.schema } else { "" },
        constraint_name: if constraint_backed { &source.name } else { "" },
        schema_digest: [0x71; 32],
        index_flags: u32::from(source.unique)
            | (u32::from(source.primary_key) << 1)
            | (u32::from(source.unique_constraint) << 2),
        null_equality_policy: 1,
        raw_catalog_ordinal: source.raw_ordinal,
        key_columns: &ONE_INDEX_KEY,
        catalog_epoch: TEST_CATALOG_EPOCH,
        base_generation: 31,
        base_root: [0x72; 32],
    }
}

fn descriptor_for_index(
    index: &SemanticsV2CatalogIndexWitness<'_>,
    constraint_backed: bool,
) -> RetainedIndexDescriptor {
    RetainedIndexDescriptor {
        index_ref: 0,
        owner_table_ref: 0,
        raw_catalog_ordinal: index.raw_catalog_ordinal,
        stable_index_id: index.stable_index_id,
        display_oid: index.display_oid,
        stable_constraint_id: if constraint_backed {
            index.constraint_stable_id
        } else {
            u64::MAX
        },
        constraint_display_oid: index.constraint_display_oid,
        flags: index.index_flags,
        null_equality_policy: index.null_equality_policy,
        key_start: 0,
        key_count: 1,
        catalog_epoch: index.catalog_epoch,
        owner_stable_table_id: index.owner_stable_table_id,
        owner_display_oid: index.owner_display_oid,
        owner_schema_digest: index.schema_digest,
        owner_name_digest: super::super::qualified_name_digest(
            index.owner_schema,
            index.owner_name,
        )
        .expect("fixture owner name is canonical"),
        index_name_digest: super::super::qualified_name_digest(index.schema, index.name)
            .expect("fixture index name is canonical"),
        constraint_name_digest: if constraint_backed {
            super::super::qualified_name_digest(index.constraint_schema, index.constraint_name)
                .expect("fixture constraint name is canonical")
        } else {
            [0; 32]
        },
        owner_table_base_root: [0x73; 32],
        base_index_root: index.base_root,
        final_index_root: [0x74; 32],
        descriptor_digest: [0x75; 32],
        owner_data_generation: 30,
        base_index_generation: index.base_generation,
        final_index_generation: 32,
    }
}

fn terminal_index_token(
    kind: u8,
    index: &SemanticsV2CatalogIndexWitness<'_>,
    target_table_ref: u32,
) -> RetainedDependencyToken {
    RetainedDependencyToken {
        dependency_ref: 0,
        kind,
        access: 0,
        flags: TERMINAL_ERROR_FLAG,
        stable_object_id: index.stable_index_id,
        display_oid: index.display_oid,
        target_table_ref,
        base_generation: index.base_generation,
        snapshot_floor: 0,
        key_effect_ref: ABSENT,
        descriptor_ref: 0,
        catalog_epoch: index.catalog_epoch,
        schema_digest: index.schema_digest,
        base_root: index.base_root,
        name_digest: super::super::qualified_name_digest(index.schema, index.name)
            .expect("fixture index name is canonical"),
        identity_digest: [0; 32],
        token_digest: [0; 32],
    }
}

fn terminal_target_table<'a>(
    display_oid: u32,
    schema: &'a str,
    name: &'a str,
    schema_digest: [u8; 32],
    foreign_keys: &'a [SemanticsV2CatalogForeignKeyWitness<'a>],
) -> SemanticsV2CatalogTableWitness<'a> {
    SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid,
        schema,
        name,
        schema_digest,
        data_generation: 30,
        data_root: [0x73; 32],
        catalog_columns: &[],
        not_null_guards: &[],
        check_guards: &[],
        foreign_keys,
    }
}

fn terminal_resolution(terminal_source_ordinal: u32) -> RetainedStatementResolution {
    RetainedStatementResolution {
        statement_ordinal: 0,
        record_ref: 0,
        outcome_ref: 0,
        table_ref: 0,
        flags: 0,
        s4_start: 0,
        s4_count: 0,
        s5_start: 0,
        s5_count: 0,
        dependency_use_start: 0,
        dependency_use_count: 1,
        projection_start: 0,
        projection_count: 0,
        input_row_count: 0,
        surviving_row_count: 0,
        affected_row_count: 0,
        dependency_validation_floor: 0,
        record_bytes: 0,
        terminal_dependency_ref: 0,
        terminal_row_ordinal: 0,
        terminal_source_ordinal,
        request_digest: [0; 32],
        typed_statement_digest: [0; 32],
        record_digest: [0; 32],
        returning_digest: [0; 32],
        overlay_before: [0; 32],
        overlay_after: [0; 32],
        outcome_digest: [0; 32],
    }
}

fn statement_resolution(dependency_use_count: u32) -> RetainedStatementResolution {
    RetainedStatementResolution {
        outcome_ref: ABSENT,
        terminal_dependency_ref: ABSENT,
        terminal_row_ordinal: ABSENT,
        terminal_source_ordinal: ABSENT,
        dependency_use_count,
        ..terminal_resolution(ABSENT)
    }
}

fn retained_table(stable_table_id: u64, display_oid: u32) -> RetainedTable {
    RetainedTable {
        table_ref: 0,
        resets_existing_rows: false,
        initial_table_absent: false,
        stable_table_id,
        display_oid,
        target_dependency_ref: ABSENT,
        catalog_epoch: TEST_CATALOG_EPOCH,
        data_generation_before: 30,
        data_generation_after: 30,
        row_allocator_before: 0,
        row_allocator_high_water: 0,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        disposition_start: 0,
        disposition_count: 0,
        transition_start: 0,
        transition_count: 0,
        key_effect_start: 0,
        key_effect_count: 0,
        owned_index_start: 0,
        owned_index_count: 0,
        image_ref: ABSENT,
        catalog_column_count: 0,
        schema_digest: [0; 32],
        initial_table_root: [0; 32],
        final_table_root: [0; 32],
        transition_root: [0; 32],
        index_effect_root: [0; 32],
        image_layout_digest: [0; 32],
        image_content_digest: [0; 32],
        image_arena_offset: 0,
        image_encoded_bytes: 0,
        image_descriptor_digest: [0; 32],
        manifest_digest: [0; 32],
    }
}

fn owned_catalog_column(source: &OwnedColumnSource) -> SemanticsV2CatalogColumnWitness<'_> {
    SemanticsV2CatalogColumnWitness {
        catalog_column_ordinal: source.catalog_column_ordinal,
        stable_column_id: source.stable_column_id,
        attnum: source.attnum,
        name: &source.name,
        storage: source.storage,
        declared_type_oid: source.declared_type_oid,
        signed_type_size: source.signed_type_size,
        column_shape_digest: [0x81; 32],
        column_root: [0x82; 32],
    }
}

fn domain_check_guard<'a>(
    source: &'a OwnedDomainSource,
    stable_guard_id: u64,
    display_oid: u32,
    domain_ordinal: u32,
) -> SemanticsV2CatalogGuardWitness<'a> {
    SemanticsV2CatalogGuardWitness {
        kind: DOMAIN_CONSTRAINT_GUARD,
        stable_guard_id,
        display_oid,
        schema: &source.schema,
        name: "same_shape_check",
        synthesized_not_null: false,
        owner_kind: 2,
        owner_stable_id: stable_guard_id - 100,
        owner_display_oid: source.oid,
        owner_catalog_column_ordinal: ABSENT,
        domain_ordinal,
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest: [0x83; 32],
        program_or_descriptor_root: [0x84; 32],
        catalog_generation: 33,
    }
}

fn catalog_domain<'a>(
    source: &'a OwnedDomainSource,
    stable_domain_id: u64,
    constraints: &'a [SemanticsV2CatalogGuardWitness<'a>],
) -> SemanticsV2CatalogDomainWitness<'a> {
    SemanticsV2CatalogDomainWitness {
        stable_domain_id,
        display_oid: source.oid,
        schema: &source.schema,
        name: &source.name,
        storage: source.storage,
        declared_type_oid: source.declared_type_oid,
        signed_type_size: source.signed_type_size,
        storage_shape_digest: [0x85; 32],
        catalog_generation: 33,
        constraints,
    }
}

fn graph_with_catalog_closure_record() -> ReservedSemanticsV2Graph {
    let engine = crate::Engine::new_local();
    for (txn_id, sql) in [
        (1, "CREATE DOMAIN codec_amount AS int4"),
        (2, "CREATE TABLE codec_parent (id int4 PRIMARY KEY)"),
        (
            3,
            "CREATE TABLE codec_child (id int4 UNIQUE, pid int4, other_pid int4, amount codec_amount)",
        ),
        (
            4,
            "CREATE UNIQUE INDEX codec_child_pid_unique_idx ON codec_child (pid)",
        ),
        (
            5,
            "ALTER TABLE ONLY codec_child ADD CONSTRAINT codec_child_pid_fkey FOREIGN KEY (pid) REFERENCES codec_parent(id)",
        ),
        (
            6,
            "ALTER TABLE ONLY codec_child ADD CONSTRAINT codec_child_other_pid_fkey FOREIGN KEY (other_pid) REFERENCES codec_parent(id)",
        ),
    ] {
        engine
            .execute_text(txn_id, sql)
            .expect("terminal catalog fixture DDL applies");
    }
    blank_graph(sealed_record(
        &engine,
        "INSERT INTO codec_child (id, pid, other_pid, amount) VALUES (7, 11, 11, 123)",
    ))
}

fn graph_with_same_shape_domain_record() -> ReservedSemanticsV2Graph {
    let engine = crate::Engine::new_local();
    for (txn_id, sql) in [
        (1, "CREATE DOMAIN same_shape_a AS int4"),
        (2, "CREATE DOMAIN same_shape_b AS int4"),
        (
            3,
            "CREATE TABLE same_shape_target (left_value same_shape_a, right_value same_shape_b)",
        ),
    ] {
        engine
            .execute_text(txn_id, sql)
            .expect("same-shape domain fixture DDL applies");
    }
    blank_graph(sealed_record(
        &engine,
        "INSERT INTO same_shape_target (left_value, right_value) VALUES (7, 11)",
    ))
}

fn sealed_record(
    engine: &crate::Engine,
    sql: &str,
) -> crate::typed_insert_batch::DecodedTypedInsertRecord {
    let crate::Command::Insert(insert) = crate::parse_command(sql).expect("fixture INSERT parses")
    else {
        panic!("fixture command must be INSERT");
    };
    let catalog = engine.catalog_snapshot();
    let prepared = crate::typed_insert_batch::prepare_typed_insert_semantics_at(
        &insert,
        &catalog,
        catalog.commit_seq,
        None,
        crate::insert_semantic_ir::InsertStatementOrdinal::FIRST,
    )
    .expect("fixture semantic preparation succeeds")
    .expect("fixture target is current");
    let batch = prepared
        .seal(crate::typed_insert_batch::sequence_defaults::SequenceDefaultBindings::empty())
        .expect("fixture batch seals");
    let bytes = crate::typed_insert_batch::encode_canonical_typed_insert_record_for_test(&batch)
        .expect("fixture S2 encodes");
    crate::typed_insert_batch::decode_canonical_typed_insert_record(&bytes)
        .expect("fixture S2 decodes")
}

fn blank_graph(
    record: crate::typed_insert_batch::DecodedTypedInsertRecord,
) -> ReservedSemanticsV2Graph {
    ReservedSemanticsV2Graph {
        header: zero_header(),
        statements: Vec::new(),
        records: vec![record],
        dispositions: Vec::new(),
        sequence_effects: Vec::new(),
        outcomes: Vec::new(),
        tables: Vec::new(),
        table_dispositions: Vec::new(),
        resolutions: Vec::new(),
        dependencies: Vec::new(),
        dependency_uses: Vec::new(),
        indexes: Vec::new(),
        index_key_columns: Vec::new(),
        transitions: Vec::new(),
        key_effects: Vec::new(),
        key_components: Vec::new(),
        projections: Vec::new(),
        images: Vec::new(),
        response:
            crate::typed_insert_aggregate::semantics_v2::retained::graph::empty_response_for_test(),
    }
}

fn terminal_not_null_guard_closure(
    sabotage: impl FnOnce(&mut ReservedSemanticsV2Graph),
) -> Result<(), crate::EngineError> {
    let mut graph = graph_with_not_null_record();
    graph.dependencies[0] = guard_token(&not_null_guard(), 0);
    graph.dependencies[0].flags = TERMINAL_ERROR_FLAG;
    graph.outcomes.push(abort_outcome(*b"23502", TEST_GUARD_ID));
    graph.resolutions[0].outcome_ref = 0;
    graph.resolutions[0].terminal_dependency_ref = 0;
    graph.resolutions[0].terminal_row_ordinal = 0;
    graph.resolutions[0].terminal_source_ordinal = 0;
    sabotage(&mut graph);

    let source = graph.records[0]
        .catalog_columns()
        .next()
        .expect("terminal NOT NULL fixture has its target column");
    let target = graph.records[0].target_identity();
    let columns = [catalog_column(source)];
    let nested_guards = [not_null_guard()];
    let global_guards = [not_null_guard()];
    let table = SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid: target.oid,
        schema: target.schema,
        name: target.name,
        schema_digest: target.schema_digest,
        data_generation: 11,
        data_root: [4; 32],
        catalog_columns: &columns,
        not_null_guards: &nested_guards,
        check_guards: &[],
        foreign_keys: &[],
    };
    let tables = [table];
    let catalog = SemanticsV2CatalogWitness {
        database_id: [0; 16],
        catalog_epoch: TEST_CATALOG_EPOCH,
        catalog_digest: [0; 32],
        tables: &tables,
        indexes: &[],
        domains: &[],
        guards: &global_guards,
        sequences: &[],
    };
    validate_guard_closure(bound_identity(), &graph, &catalog)
}

fn graph_with_not_null_record() -> ReservedSemanticsV2Graph {
    let record = crate::typed_insert_batch::decode_canonical_typed_insert_record(&decode_hex(
        NOT_NULL_S2_HEX,
    ))
    .expect("frozen S2 test record decodes");
    ReservedSemanticsV2Graph {
        header: zero_header(),
        statements: Vec::new(),
        records: vec![record],
        dispositions: Vec::new(),
        sequence_effects: Vec::new(),
        outcomes: Vec::new(),
        tables: vec![RetainedTable {
            table_ref: 0,
            resets_existing_rows: false,
            initial_table_absent: false,
            stable_table_id: TEST_TABLE_ID,
            display_oid: TEST_TABLE_OID,
            target_dependency_ref: ABSENT,
            catalog_epoch: TEST_CATALOG_EPOCH,
            data_generation_before: 11,
            data_generation_after: 11,
            row_allocator_before: 0,
            row_allocator_high_water: 0,
            initial_logical_row_count: 0,
            final_logical_row_count: 0,
            disposition_start: 0,
            disposition_count: 0,
            transition_start: 0,
            transition_count: 0,
            key_effect_start: 0,
            key_effect_count: 0,
            owned_index_start: 0,
            owned_index_count: 0,
            image_ref: ABSENT,
            catalog_column_count: 1,
            schema_digest: [1; 32],
            initial_table_root: [4; 32],
            final_table_root: [4; 32],
            transition_root: [0; 32],
            index_effect_root: [0; 32],
            image_layout_digest: [0; 32],
            image_content_digest: [0; 32],
            image_arena_offset: 0,
            image_encoded_bytes: 0,
            image_descriptor_digest: [0; 32],
            manifest_digest: [0; 32],
        }],
        table_dispositions: Vec::new(),
        resolutions: vec![RetainedStatementResolution {
            statement_ordinal: 0,
            record_ref: 0,
            outcome_ref: ABSENT,
            table_ref: 0,
            flags: 0,
            s4_start: 0,
            s4_count: 0,
            s5_start: 0,
            s5_count: 0,
            dependency_use_start: 0,
            dependency_use_count: 1,
            projection_start: 0,
            projection_count: 0,
            input_row_count: 0,
            surviving_row_count: 0,
            affected_row_count: 0,
            dependency_validation_floor: 0,
            record_bytes: 0,
            terminal_dependency_ref: ABSENT,
            terminal_row_ordinal: ABSENT,
            terminal_source_ordinal: ABSENT,
            request_digest: [0; 32],
            typed_statement_digest: [0; 32],
            record_digest: [0; 32],
            returning_digest: [0; 32],
            overlay_before: [0; 32],
            overlay_after: [0; 32],
            outcome_digest: [0; 32],
        }],
        dependencies: vec![guard_token(&not_null_guard(), 0)],
        dependency_uses: vec![RetainedStatementDependencyUse {
            statement_ordinal: 0,
            dependency_ref: 0,
            role: u16::from(NOT_NULL_GUARD),
            source_ordinal: 0,
            transition_ref: ABSENT,
            key_effect_ref: ABSENT,
        }],
        indexes: Vec::new(),
        index_key_columns: Vec::new(),
        transitions: Vec::new(),
        key_effects: Vec::new(),
        key_components: Vec::new(),
        projections: Vec::new(),
        images: Vec::new(),
        response:
            crate::typed_insert_aggregate::semantics_v2::retained::graph::empty_response_for_test(),
    }
}

fn not_null_guard() -> SemanticsV2CatalogGuardWitness<'static> {
    SemanticsV2CatalogGuardWitness {
        kind: NOT_NULL_GUARD,
        stable_guard_id: TEST_GUARD_ID,
        display_oid: 0,
        schema: "",
        name: "",
        synthesized_not_null: true,
        owner_kind: 1,
        owner_stable_id: TEST_TABLE_ID,
        owner_display_oid: TEST_TABLE_OID,
        owner_catalog_column_ordinal: 0,
        domain_ordinal: ABSENT,
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest: [2; 32],
        program_or_descriptor_root: [3; 32],
        catalog_generation: 12,
    }
}

fn synthesized_guard_order_row<'a>(
    kind: u8,
    display_oid: u32,
    schema: &'a str,
    name: &'a str,
) -> SemanticsV2CatalogGuardWitness<'a> {
    SemanticsV2CatalogGuardWitness {
        kind,
        stable_guard_id: 5_500 + u64::from(kind),
        display_oid,
        schema,
        name,
        synthesized_not_null: true,
        owner_kind: if kind == DOMAIN_CONSTRAINT_GUARD {
            2
        } else {
            1
        },
        owner_stable_id: 5_600 + u64::from(kind),
        owner_display_oid: 5_700 + u32::from(kind),
        owner_catalog_column_ordinal: if kind == NOT_NULL_GUARD { 0 } else { ABSENT },
        domain_ordinal: if kind == DOMAIN_CONSTRAINT_GUARD {
            0
        } else {
            ABSENT
        },
        raw_constraint_ordinal: 0,
        source_ordinal: 0,
        shape_digest: [0x91; 32],
        program_or_descriptor_root: [0x92; 32],
        catalog_generation: 34,
    }
}

fn table_check_guard(
    stable_guard_id: u64,
    display_oid: u32,
    raw_constraint_ordinal: u32,
) -> SemanticsV2CatalogGuardWitness<'static> {
    SemanticsV2CatalogGuardWitness {
        kind: CHECK_GUARD,
        stable_guard_id,
        display_oid,
        schema: "public",
        name: if raw_constraint_ordinal == 0 {
            "check_zero"
        } else {
            "check_one"
        },
        synthesized_not_null: false,
        owner_kind: 1,
        owner_stable_id: TEST_TABLE_ID,
        owner_display_oid: TEST_TABLE_OID,
        owner_catalog_column_ordinal: ABSENT,
        domain_ordinal: ABSENT,
        raw_constraint_ordinal,
        source_ordinal: raw_constraint_ordinal,
        shape_digest: [0x31 + raw_constraint_ordinal as u8; 32],
        program_or_descriptor_root: [0x41 + raw_constraint_ordinal as u8; 32],
        catalog_generation: 12,
    }
}

fn catalog_column(
    source: crate::typed_insert_batch::DecodedCatalogColumnFacts<'_>,
) -> SemanticsV2CatalogColumnWitness<'_> {
    SemanticsV2CatalogColumnWitness {
        catalog_column_ordinal: source.catalog_column_ordinal,
        stable_column_id: source.column_id,
        attnum: source.attnum,
        name: source.name,
        storage: super::super::storage_bytes(source.ty),
        declared_type_oid: source.type_oid,
        signed_type_size: source.type_size,
        column_shape_digest: [2; 32],
        column_root: [3; 32],
    }
}

fn test_table() -> SemanticsV2CatalogTableWitness<'static> {
    SemanticsV2CatalogTableWitness {
        stable_table_id: TEST_TABLE_ID,
        display_oid: TEST_TABLE_OID,
        schema: "public",
        name: "table_name",
        schema_digest: [1; 32],
        data_generation: 11,
        data_root: [4; 32],
        catalog_columns: &[],
        not_null_guards: &[],
        check_guards: &[],
        foreign_keys: &[],
    }
}

fn empty_catalog<'a>(
    tables: &'a [SemanticsV2CatalogTableWitness<'a>],
) -> SemanticsV2CatalogWitness<'a> {
    SemanticsV2CatalogWitness {
        database_id: [0; 16],
        catalog_epoch: TEST_CATALOG_EPOCH,
        catalog_digest: [0; 32],
        tables,
        indexes: &[],
        domains: &[],
        guards: &[],
        sequences: &[],
    }
}

fn guard_token(
    guard: &SemanticsV2CatalogGuardWitness<'_>,
    dependency_ref: u32,
) -> RetainedDependencyToken {
    RetainedDependencyToken {
        dependency_ref,
        kind: guard.kind,
        access: 0,
        flags: 0,
        stable_object_id: guard.stable_guard_id,
        display_oid: guard.display_oid,
        target_table_ref: 0,
        base_generation: guard.catalog_generation,
        snapshot_floor: 0,
        key_effect_ref: ABSENT,
        descriptor_ref: ABSENT,
        catalog_epoch: TEST_CATALOG_EPOCH,
        schema_digest: guard.shape_digest,
        base_root: guard.program_or_descriptor_root,
        name_digest: guard_name_digest(guard).expect("test guard has a canonical name"),
        identity_digest: [0; 32],
        token_digest: [0; 32],
    }
}

fn index_fixture(
    constraint_backed: bool,
) -> (
    SemanticsV2CatalogIndexWitness<'static>,
    RetainedIndexDescriptor,
    RetainedDependencyToken,
) {
    let (constraint_stable_id, constraint_display_oid, constraint_schema, constraint_name) =
        if constraint_backed {
            (601, 701, "public", "table_idx_key")
        } else {
            (0, 0, "", "")
        };
    let index = SemanticsV2CatalogIndexWitness {
        stable_index_id: 600,
        display_oid: 700,
        owner_stable_table_id: TEST_TABLE_ID,
        owner_display_oid: TEST_TABLE_OID,
        schema: "public",
        name: "table_idx",
        owner_schema: "public",
        owner_name: "table_name",
        constraint_stable_id,
        constraint_display_oid,
        constraint_schema,
        constraint_name,
        schema_digest: [7; 32],
        index_flags: if constraint_backed { 0b101 } else { 1 },
        null_equality_policy: 1,
        raw_catalog_ordinal: 0,
        key_columns: &[],
        catalog_epoch: TEST_CATALOG_EPOCH,
        base_generation: 16,
        base_root: [8; 32],
    };
    let descriptor = RetainedIndexDescriptor {
        index_ref: 0,
        owner_table_ref: 0,
        raw_catalog_ordinal: 0,
        stable_index_id: index.stable_index_id,
        display_oid: index.display_oid,
        stable_constraint_id: if constraint_backed {
            index.constraint_stable_id
        } else {
            u64::MAX
        },
        constraint_display_oid: index.constraint_display_oid,
        flags: index.index_flags,
        null_equality_policy: 1,
        key_start: 0,
        key_count: 0,
        catalog_epoch: TEST_CATALOG_EPOCH,
        owner_stable_table_id: TEST_TABLE_ID,
        owner_display_oid: TEST_TABLE_OID,
        owner_schema_digest: index.schema_digest,
        owner_name_digest: super::super::qualified_name_digest("public", "table_name")
            .expect("test owner name is valid"),
        index_name_digest: super::super::qualified_name_digest(index.schema, index.name)
            .expect("test index name is valid"),
        constraint_name_digest: if constraint_backed {
            super::super::qualified_name_digest(index.constraint_schema, index.constraint_name)
                .expect("test constraint name is valid")
        } else {
            [0; 32]
        },
        owner_table_base_root: [4; 32],
        base_index_root: index.base_root,
        final_index_root: [9; 32],
        descriptor_digest: [10; 32],
        owner_data_generation: 11,
        base_index_generation: index.base_generation,
        final_index_generation: 17,
    };
    let token = RetainedDependencyToken {
        dependency_ref: 0,
        kind: UNIQUE_KEY_GUARD,
        access: 0,
        flags: TERMINAL_ERROR_FLAG,
        stable_object_id: index.stable_index_id,
        display_oid: index.display_oid,
        target_table_ref: 0,
        base_generation: index.base_generation,
        snapshot_floor: 0,
        key_effect_ref: ABSENT,
        descriptor_ref: descriptor.index_ref,
        catalog_epoch: index.catalog_epoch,
        schema_digest: index.schema_digest,
        base_root: index.base_root,
        name_digest: super::super::qualified_name_digest(index.schema, index.name)
            .expect("test index name is valid"),
        identity_digest: [0; 32],
        token_digest: [0; 32],
    };
    (index, descriptor, token)
}

fn abort_outcome(sqlstate: [u8; 5], constraint_id: u64) -> RetainedStatementOutcome {
    RetainedStatementOutcome {
        statement_ordinal: 0,
        family_ordinal: 0,
        semantic_class: 0,
        flags: 0,
        typed_statement_digest: [0; 32],
        outcome_digest: [0; 32],
        outcome: gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(sqlstate),
            constraint_id,
            target_digest: [0; 32],
            returning_digest: [0; 32],
        },
    }
}

fn bound_identity() -> SemanticsV2BoundIdentity {
    SemanticsV2BoundIdentity {
        database_id: [0; 16],
        cluster_id: [0; 16],
        timeline_id: [0; 16],
        format_epoch: 0,
        leader_epoch: 0,
        catalog_epoch: TEST_CATALOG_EPOCH,
        catalog_digest: [0; 32],
        stable_transaction_id: 0,
        request_digest: [0; 32],
        autocommit: true,
        commit_sequence: 0,
        initial_database_root: [0; 32],
    }
}

fn zero_header() -> SemanticsV2S7HeaderIdentity {
    SemanticsV2S7HeaderIdentity {
        total_bytes: 0,
        root_descriptor_version: 0,
        catalog_before_epoch: 0,
        catalog_after_epoch: 0,
        catalog_before_digest: [0; 32],
        catalog_after_digest: [0; 32],
        initial_database_root: [0; 32],
        final_database_root: [0; 32],
        initial_overlay_root: [0; 32],
        final_overlay_root: [0; 32],
        root_descriptor: [0; 32],
        payload_digest: [0; 32],
    }
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert!(value.len().is_multiple_of(2), "test hex has complete bytes");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let high = hex_digit(chunk[0]);
            let low = hex_digit(chunk[1]);
            (high << 4) | low
        })
        .collect()
}

fn hex_digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("test fixture is lower-case hex"),
    }
}
