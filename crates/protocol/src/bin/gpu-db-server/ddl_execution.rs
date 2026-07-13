// Legacy DDL mutation ownership. This is not a product execution path.

use super::ddl_syntax::{
    parse_alter_table_drop_constraint, parse_drop_table, parse_truncate_table, DropConstraint,
    DropTable, ParsedTruncateTable,
};
use super::{
    validate_foreign_keys, write_command_complete, write_error, CatalogCommentTarget,
    ColumnDefault, ErrorField, ReadWrite, Session,
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
