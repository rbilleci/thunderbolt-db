// Legacy DDL mutation ownership. This is not a product execution path.

use super::ddl_syntax::{
    parse_alter_table_drop_constraint, parse_drop_table, parse_truncate_table, DropConstraint,
    DropTable, ParsedTruncateTable,
};
use super::{
    add_check_constraint_to_session, add_column_default_supported, add_foreign_key_to_session,
    add_primary_key_to_session, add_unique_constraint_to_session, column_default_matches_type,
    create_implicit_sequence, drop_column_from_session, evaluate_column_default,
    preflight_column_default_target, rename_column_in_session, rename_constraint_in_session,
    rename_table_in_session, resolve_column_domain_type, schema_permission_error,
    validate_foreign_keys, write_command_complete, write_error, CatalogColumn,
    CatalogCommentTarget, ColumnDefault, Command, ErrorField, ReadWrite, SchemaPrivilege, Session,
    Table,
};
use std::collections::BTreeSet;
use std::io;

pub(super) fn try_execute_ddl_statement(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    statement: &str,
) -> Option<io::Result<()>> {
    if let Some(truncate) = parse_truncate_table(statement) {
        return Some(execute_truncate(stream, session, truncate));
    }
    if let Some(drop) = parse_drop_table(statement) {
        return Some(execute_drop_table(stream, session, drop));
    }
    if let Some(drop) = parse_alter_table_drop_constraint(statement) {
        return Some(execute_drop_constraint(stream, session, drop));
    }
    None
}

fn execute_truncate(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    truncate: ParsedTruncateTable,
) -> io::Result<()> {
    if session.views.contains_key(&truncate.table)
        || session.materialized_views.contains_key(&truncate.table)
        || session.sequences.contains_key(&truncate.table)
    {
        return write_error(
            stream,
            &ErrorField {
                code: "42809",
                message: "relation is not a table",
                position: None,
            },
        );
    }
    let restart_sequences = if truncate.restart_identity {
        session
            .tables
            .get(&truncate.table)
            .map(|table| {
                table
                    .columns
                    .iter()
                    .filter_map(|column| match &column.def.default {
                        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                            Some(sequence.clone())
                        }
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default()
    } else {
        BTreeSet::new()
    };
    for sequence in &restart_sequences {
        if !session.sequences.contains_key(sequence) {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "sequence does not exist",
                    position: None,
                },
            );
        }
    }
    let Some(table) = session.tables.get(&truncate.table).cloned() else {
        return write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        );
    };
    let mut candidate_table = table.clone();
    candidate_table.rows.clear();
    let old_table = session
        .tables
        .insert(truncate.table.clone(), candidate_table)
        .expect("table existence checked");
    if let Err(error) = validate_foreign_keys(session) {
        session.tables.insert(truncate.table.clone(), old_table);
        return write_error(stream, &error);
    }
    session.mark_table_dirty(truncate.table);
    for sequence in restart_sequences {
        let sequence_state = session
            .sequences
            .get_mut(&sequence)
            .expect("truncate restart identity sequence preflighted");
        sequence_state.last_value = 1;
        sequence_state.is_called = false;
        session.currval_sequences.remove(&sequence);
        session.mark_sequence_dirty(sequence);
    }
    session.persist_catalog_snapshot();
    write_command_complete(stream, "TRUNCATE TABLE")
}

fn execute_drop_table(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    drop: DropTable,
) -> io::Result<()> {
    let mut seen = BTreeSet::new();
    for table in &drop.tables {
        if !seen.insert(table) {
            return write_error(
                stream,
                &ErrorField {
                    code: "42710",
                    message: "table specified more than once",
                    position: None,
                },
            );
        }
        if session.views.contains_key(table)
            || session.materialized_views.contains_key(table)
            || session.sequences.contains_key(table)
        {
            return write_error(
                stream,
                &ErrorField {
                    code: "42809",
                    message: "relation is not a table",
                    position: None,
                },
            );
        }
        if !drop.if_exists && !session.tables.contains_key(table) {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            );
        }
    }
    let drop_tables = drop.tables.iter().cloned().collect::<BTreeSet<_>>();
    if session.tables.values().any(|table| {
        table.foreign_keys.iter().any(|constraint| {
            drop_tables.contains(&constraint.table)
                || drop_tables.contains(&constraint.referenced_table)
        })
    }) {
        return write_error(
            stream,
            &ErrorField {
                code: "2BP01",
                message: "cannot drop table because a foreign key constraint depends on it",
                position: None,
            },
        );
    }
    let dropped_index_names = session
        .indexes
        .iter()
        .filter(|index| drop_tables.contains(&index.table))
        .map(|index| index.name.clone())
        .collect::<BTreeSet<_>>();
    for table in &drop.tables {
        session.tables.remove(table);
        session.mark_table_dirty(table.clone());
    }
    let old_index_count = session.indexes.len();
    session
        .indexes
        .retain(|index| !drop_tables.contains(&index.table));
    session.dirty_indexes |= session.indexes.len() != old_index_count;
    let dropped_comment_targets = session
        .comments
        .keys()
        .filter(|target| match target {
            CatalogCommentTarget::Table { table }
            | CatalogCommentTarget::Column { table, .. }
            | CatalogCommentTarget::Constraint { table, .. } => drop_tables.contains(table),
            CatalogCommentTarget::Index { index } => dropped_index_names.contains(index),
            CatalogCommentTarget::Database { .. }
            | CatalogCommentTarget::Role { .. }
            | CatalogCommentTarget::Schema { .. }
            | CatalogCommentTarget::Tablespace { .. }
            | CatalogCommentTarget::View { .. }
            | CatalogCommentTarget::MaterializedView { .. }
            | CatalogCommentTarget::Extension { .. }
            | CatalogCommentTarget::Function { .. }
            | CatalogCommentTarget::Sequence { .. }
            | CatalogCommentTarget::Domain { .. }
            | CatalogCommentTarget::Publication { .. }
            | CatalogCommentTarget::Subscription { .. } => false,
        })
        .cloned()
        .collect::<Vec<_>>();
    for target in dropped_comment_targets {
        session.comments.remove(&target);
        session.mark_comment_dirty(target);
    }
    session.persist_catalog_snapshot();
    write_command_complete(stream, "DROP TABLE")
}

fn execute_drop_constraint(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    drop: DropConstraint,
) -> io::Result<()> {
    if !session.tables.contains_key(&drop.table) {
        if drop.table_if_exists {
            return write_command_complete(stream, "ALTER TABLE");
        }
        return write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        );
    }
    let old_index_count = session.indexes.len();
    session.indexes.retain(|index| {
        !(index.table == drop.table
            && index.name == drop.constraint
            && (index.primary_key || index.unique_constraint))
    });
    let mut dropped_check = false;
    let mut dropped_foreign_key = false;
    if let Some(table) = session.tables.get_mut(&drop.table) {
        let old_check_count = table.check_constraints.len();
        table
            .check_constraints
            .retain(|constraint| constraint.name != drop.constraint);
        dropped_check = table.check_constraints.len() != old_check_count;
        let old_foreign_key_count = table.foreign_keys.len();
        table
            .foreign_keys
            .retain(|constraint| constraint.name != drop.constraint);
        dropped_foreign_key = table.foreign_keys.len() != old_foreign_key_count;
        if dropped_check || dropped_foreign_key {
            session.mark_table_dirty(drop.table.clone());
        }
    }
    if session.indexes.len() == old_index_count
        && !dropped_check
        && !dropped_foreign_key
        && !drop.if_exists
    {
        return write_error(
            stream,
            &ErrorField {
                code: "42704",
                message: "constraint does not exist",
                position: None,
            },
        );
    }
    session.dirty_indexes |= session.indexes.len() != old_index_count;
    if session.indexes.len() != old_index_count || dropped_check || dropped_foreign_key {
        for target in [
            CatalogCommentTarget::Index {
                index: drop.constraint.clone(),
            },
            CatalogCommentTarget::Constraint {
                table: drop.table.clone(),
                constraint: drop.constraint.clone(),
            },
        ] {
            session.comments.remove(&target);
            session.mark_comment_dirty(target);
        }
    }
    session.persist_catalog_snapshot();
    write_command_complete(stream, "ALTER TABLE")
}

pub(super) fn execute_parsed_table_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateTable(create) => {
            if !session.public_schema_exists {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                );
            }
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if session.tables.contains_key(&create.table)
                || session.views.contains_key(&create.table)
                || session.materialized_views.contains_key(&create.table)
                || session.sequences.contains_key(&create.table)
                || session.domains.contains_key(&create.table)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P07",
                        message: "relation already exists",
                        position: None,
                    },
                );
            }
            let mut columns = Vec::with_capacity(create.columns.len());
            for (idx, mut def) in create.columns.into_iter().enumerate() {
                let Ok(attnum) = i16::try_from(idx + 1) else {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "54000",
                            message: "too many columns for bootstrap catalog",
                            position: None,
                        },
                    );
                };
                if let Err(error) = resolve_column_domain_type(session, &mut def) {
                    return write_error(stream, &error);
                }
                columns.push(CatalogColumn { attnum, def });
            }
            let primary_key = create.primary_key.clone();
            let check_constraints = create.check_constraints.clone();
            let name = create.table;
            let table_name = name.clone();
            for column in &columns {
                if let Some(default) = column.def.default.as_ref() {
                    if let Some(error) = preflight_column_default_target(session, default) {
                        return write_error(stream, &error);
                    }
                }
            }
            let oid = session.next_relation_oid;
            session.next_relation_oid = match session.next_relation_oid.checked_add(1) {
                Some(next) => next,
                None => {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "54000",
                            message: "relation OID allocation exhausted",
                            position: None,
                        },
                    );
                }
            };
            let implicit_sequences = columns
                .iter()
                .filter_map(|column| match &column.def.default {
                    Some(ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    }) => Some(sequence.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for sequence in &implicit_sequences {
                if let Err(error) = create_implicit_sequence(session, sequence) {
                    return write_error(stream, &error);
                }
            }
            session.tables.insert(
                table_name.clone(),
                Table {
                    oid,
                    name,
                    columns,
                    rows: Vec::new(),
                    check_constraints: Vec::new(),
                    foreign_keys: Vec::new(),
                },
            );
            if let Some(primary_key) = primary_key {
                let constraint_name = primary_key
                    .name
                    .unwrap_or_else(|| format!("{}_pkey", table_name));
                if let Err(error) = add_primary_key_to_session(
                    session,
                    &table_name,
                    constraint_name,
                    primary_key.column,
                ) {
                    session.tables.remove(&table_name);
                    session.indexes.retain(|index| index.table != table_name);
                    for sequence in &implicit_sequences {
                        session.sequences.remove(sequence);
                        session.mark_sequence_dirty(sequence.clone());
                    }
                    return write_error(stream, &error);
                }
            }
            for unique in create.unique_constraints {
                let constraint_name = unique
                    .name
                    .unwrap_or_else(|| format!("{}_{}_key", table_name, unique.column));
                if let Err(error) = add_unique_constraint_to_session(
                    session,
                    &table_name,
                    constraint_name,
                    unique.column,
                ) {
                    session.tables.remove(&table_name);
                    session.indexes.retain(|index| index.table != table_name);
                    for sequence in &implicit_sequences {
                        session.sequences.remove(sequence);
                        session.mark_sequence_dirty(sequence.clone());
                    }
                    return write_error(stream, &error);
                }
            }
            for check in check_constraints {
                let constraint_name = check
                    .name
                    .unwrap_or_else(|| format!("{}_{}_check", table_name, check.filter.column));
                if let Err(error) = add_check_constraint_to_session(
                    session,
                    &table_name,
                    constraint_name,
                    check.filter,
                ) {
                    session.tables.remove(&table_name);
                    session.indexes.retain(|index| index.table != table_name);
                    for sequence in &implicit_sequences {
                        session.sequences.remove(sequence);
                        session.mark_sequence_dirty(sequence.clone());
                    }
                    return write_error(stream, &error);
                }
            }
            if !session.default_table_acl.is_empty() {
                session
                    .table_acls
                    .insert(table_name.clone(), session.default_table_acl.clone());
                session.mark_table_acl_dirty(table_name.clone());
            }
            session.mark_table_dirty(table_name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE TABLE")
        }
        Command::AddPrimaryKey(add) => {
            if let Err(error) = add_primary_key_to_session(
                session,
                &add.table,
                add.name.clone(),
                add.column.clone(),
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::AddUniqueConstraint(add) => {
            if let Err(error) = add_unique_constraint_to_session(
                session,
                &add.table,
                add.name.clone(),
                add.column.clone(),
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::AddCheckConstraint(add) => {
            if let Err(error) =
                add_check_constraint_to_session(session, &add.table, add.name, add.filter)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::AddForeignKey(add) => {
            if let Err(error) = add_foreign_key_to_session(
                session,
                &add.table,
                add.name,
                add.column,
                add.referenced_table,
                add.referenced_column,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::DropConstraint(drop) => {
            if !session.tables.contains_key(&drop.table) {
                if drop.table_if_exists {
                    return write_command_complete(stream, "ALTER TABLE");
                }
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            }
            let old_index_count = session.indexes.len();
            session.indexes.retain(|index| {
                !(index.table == drop.table
                    && index.name == drop.name
                    && (index.primary_key || index.unique_constraint))
            });
            let mut dropped_check = false;
            let mut dropped_foreign_key = false;
            if let Some(table) = session.tables.get_mut(&drop.table) {
                let old_check_count = table.check_constraints.len();
                table
                    .check_constraints
                    .retain(|constraint| constraint.name != drop.name);
                dropped_check = table.check_constraints.len() != old_check_count;
                let old_foreign_key_count = table.foreign_keys.len();
                table
                    .foreign_keys
                    .retain(|constraint| constraint.name != drop.name);
                dropped_foreign_key = table.foreign_keys.len() != old_foreign_key_count;
                if dropped_check || dropped_foreign_key {
                    session.mark_table_dirty(drop.table.clone());
                }
            }
            if session.indexes.len() == old_index_count
                && !dropped_check
                && !dropped_foreign_key
                && !drop.if_exists
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42704",
                        message: "constraint does not exist",
                        position: None,
                    },
                );
            }
            session.dirty_indexes |= session.indexes.len() != old_index_count;
            if session.indexes.len() != old_index_count || dropped_check || dropped_foreign_key {
                for target in [
                    CatalogCommentTarget::Index {
                        index: drop.name.clone(),
                    },
                    CatalogCommentTarget::Constraint {
                        table: drop.table.clone(),
                        constraint: drop.name.clone(),
                    },
                ] {
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::RenameConstraint(rename) => {
            if let Err(error) = rename_constraint_in_session(
                session,
                &rename.table,
                &rename.old_name,
                &rename.new_name,
                rename.table_if_exists,
            ) {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::RenameTable(rename) => {
            if let Err(error) = rename_table_in_session(
                session,
                &rename.old_name,
                &rename.new_name,
                rename.if_exists,
            ) {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::DropTable(drop) => {
            let mut seen = BTreeSet::new();
            for name in &drop.names {
                if !seen.insert(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "table specified more than once",
                            position: None,
                        },
                    );
                }
                if session.views.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a table",
                            position: None,
                        },
                    );
                }
                if session.materialized_views.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a table",
                            position: None,
                        },
                    );
                }
                if session.sequences.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a table",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.tables.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P01",
                            message: "relation does not exist",
                            position: None,
                        },
                    );
                }
            }
            let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
            let dropped_index_names = session
                .indexes
                .iter()
                .filter(|index| drop_names.contains(&index.table))
                .map(|index| index.name.clone())
                .collect::<BTreeSet<_>>();
            for name in &drop.names {
                session.tables.remove(name);
                session.table_acls.remove(name);
                session.mark_table_dirty(name.clone());
                session.mark_table_acl_dirty(name.clone());
            }
            let old_index_count = session.indexes.len();
            session
                .indexes
                .retain(|index| !drop_names.contains(&index.table));
            session.dirty_indexes |= session.indexes.len() != old_index_count;
            let dropped_comment_targets = session
                .comments
                .keys()
                .filter(|target| match target {
                    CatalogCommentTarget::Table { table }
                    | CatalogCommentTarget::Column { table, .. }
                    | CatalogCommentTarget::Constraint { table, .. } => drop_names.contains(table),
                    CatalogCommentTarget::Index { index } => dropped_index_names.contains(index),
                    CatalogCommentTarget::Database { .. }
                    | CatalogCommentTarget::Role { .. }
                    | CatalogCommentTarget::Schema { .. }
                    | CatalogCommentTarget::Tablespace { .. }
                    | CatalogCommentTarget::View { .. }
                    | CatalogCommentTarget::MaterializedView { .. }
                    | CatalogCommentTarget::Extension { .. }
                    | CatalogCommentTarget::Function { .. }
                    | CatalogCommentTarget::Sequence { .. }
                    | CatalogCommentTarget::Domain { .. }
                    | CatalogCommentTarget::Publication { .. }
                    | CatalogCommentTarget::Subscription { .. } => false,
                })
                .cloned()
                .collect::<Vec<_>>();
            for target in dropped_comment_targets {
                session.comments.remove(&target);
                session.mark_comment_dirty(target);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP TABLE")
        }
        Command::AlterColumnDefault(alter) => {
            if session.views.contains_key(&alter.table)
                || session.materialized_views.contains_key(&alter.table)
                || session.sequences.contains_key(&alter.table)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42809",
                        message: "relation is not a table",
                        position: None,
                    },
                );
            }
            let Some(table) = session.tables.get(&alter.table) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            let Some(column) = table
                .columns
                .iter()
                .find(|column| column.def.name == alter.column)
            else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42703",
                        message: "column does not exist",
                        position: None,
                    },
                );
            };
            let Some(default) = alter.default.clone() else {
                session
                    .tables
                    .get_mut(&alter.table)
                    .expect("table existence checked")
                    .columns
                    .iter_mut()
                    .find(|column| column.def.name == alter.column)
                    .expect("column existence checked")
                    .def
                    .default = None;
                session.mark_table_dirty(alter.table);
                session.persist_catalog_snapshot();
                return write_command_complete(stream, "ALTER TABLE");
            };
            if !column_default_matches_type(&default, column.def.ty) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42804",
                        message: "column default type mismatch",
                        position: None,
                    },
                );
            }
            if let Some(error) = preflight_column_default_target(session, &default) {
                return write_error(stream, &error);
            }
            session
                .tables
                .get_mut(&alter.table)
                .expect("table existence checked")
                .columns
                .iter_mut()
                .find(|column| column.def.name == alter.column)
                .expect("column existence checked")
                .def
                .default = Some(default);
            session.mark_table_dirty(alter.table);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::AddColumn(add) => {
            if session.views.contains_key(&add.table)
                || session.materialized_views.contains_key(&add.table)
                || session.sequences.contains_key(&add.table)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42809",
                        message: "relation is not a table",
                        position: None,
                    },
                );
            }
            let Some(default) = add.column.default.clone() else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "ADD COLUMN requires a supported DEFAULT",
                        position: None,
                    },
                );
            };
            if !add_column_default_supported(&default) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "ADD COLUMN SERIAL is unsupported",
                        position: None,
                    },
                );
            }
            if !column_default_matches_type(&default, add.column.ty) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42804",
                        message: "column default type mismatch",
                        position: None,
                    },
                );
            }
            let Some(table) = session.tables.get(&add.table) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            if table
                .columns
                .iter()
                .any(|column| column.def.name == add.column.name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42701",
                        message: "column already exists",
                        position: None,
                    },
                );
            }
            if let Some(error) = preflight_column_default_target(session, &default) {
                return write_error(stream, &error);
            }
            let row_count = table.rows.len();
            let attnum = match i16::try_from(table.columns.len() + 1) {
                Ok(attnum) => attnum,
                Err(_) => {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "54000",
                            message: "too many columns for bootstrap catalog",
                            position: None,
                        },
                    );
                }
            };
            let mut default_values = Vec::with_capacity(row_count);
            for _ in 0..row_count {
                match evaluate_column_default(session, &default) {
                    Ok(default_value) => default_values.push(default_value),
                    Err(error) => return write_error(stream, &error),
                }
            }
            let table = session
                .tables
                .get_mut(&add.table)
                .expect("table existence checked");
            table.columns.push(CatalogColumn {
                attnum,
                def: add.column,
            });
            for (row, default_value) in table.rows.iter_mut().zip(default_values) {
                row.push(default_value);
            }
            session.mark_table_dirty(add.table);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::RenameColumn(rename) => {
            if let Err(error) =
                rename_column_in_session(session, &rename.table, &rename.old_name, &rename.new_name)
            {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER TABLE")
        }
        Command::DropColumn(drop) => {
            if let Err(error) = drop_column_from_session(session, &drop.table, &drop.column) {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER TABLE")
        }
        _ => unreachable!("parsed table DDL executor called with an unrelated command"),
    }
}
