// Legacy sequence ownership. This is not a product execution path.

use super::{
    int4_column, int8_column, object_access_permission_error, schema_permission_error, text_column,
    write_command_complete, write_error, write_select_rows, write_single_row, CatalogCommentTarget,
    Command, ErrorField, ReadWrite, SchemaPrivilege, Sequence, Session, TablePrivilege,
};
use std::collections::BTreeSet;
use std::io;

fn psql_describe_sequences_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_sequences_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_sequence_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .sequences
        .values()
        .map(|sequence| {
            vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("sequence".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_sequence_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .sequences
        .values()
        .map(|sequence| {
            vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("sequence".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("0 bytes".to_string()),
                session
                    .comments
                    .get(&CatalogCommentTarget::Sequence {
                        sequence: sequence.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

#[cfg(test)]
pub(super) fn test_psql_describe_sequences_catalog_query() -> &'static str {
    psql_describe_sequences_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_sequences_verbose_catalog_query() -> &'static str {
    psql_describe_sequences_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_sequence_verbose_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    psql_describe_sequence_verbose_rows(session)
}

pub(super) fn try_execute_sequence_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_describe_sequences_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &psql_describe_sequence_rows(session),
        ));
    }
    if canonical == psql_describe_sequences_verbose_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Size"),
                text_column("Description"),
            ],
            &psql_describe_sequence_verbose_rows(session),
        ));
    }
    None
}

fn pg_catalog_class_sequences_query() -> &'static str {
    "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 's' order by c.relname"
}

fn pg_catalog_class_sequence_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    sequences
        .into_iter()
        .map(|sequence| {
            vec![
                Some(sequence.oid.to_string()),
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("s".to_string()),
                Some("p".to_string()),
            ]
        })
        .collect()
}

pub(super) fn try_execute_sequence_class_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != pg_catalog_class_sequences_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &[
            int4_column("oid"),
            text_column("nspname"),
            text_column("relname"),
            text_column("relkind"),
            text_column("relpersistence"),
        ],
        &pg_catalog_class_sequence_rows(session),
    ))
}

#[cfg(test)]
pub(super) fn test_pg_catalog_class_sequence_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_class_sequence_rows(session)
}

pub(super) fn sequence_target_error(session: &Session, name: &str) -> Option<ErrorField> {
    if session.tables.contains_key(name)
        || session.views.contains_key(name)
        || session.materialized_views.contains_key(name)
    {
        return Some(ErrorField {
            code: "42809",
            message: "relation is not a sequence",
            position: None,
        });
    }
    if !session.sequences.contains_key(name) {
        return Some(ErrorField {
            code: "42P01",
            message: "sequence does not exist",
            position: None,
        });
    }
    None
}

pub(super) fn next_sequence_value(sequence: &mut Sequence) -> Result<i64, ErrorField> {
    let value = if sequence.is_called {
        sequence.last_value.checked_add(1).ok_or(ErrorField {
            code: "2200H",
            message: "sequence value overflow",
            position: None,
        })?
    } else {
        sequence.last_value
    };
    sequence.last_value = value;
    sequence.is_called = true;
    Ok(value)
}

pub(super) fn create_implicit_sequence(
    session: &mut Session,
    name: &str,
) -> Result<(), ErrorField> {
    if session.tables.contains_key(name)
        || session.views.contains_key(name)
        || session.materialized_views.contains_key(name)
        || session.sequences.contains_key(name)
    {
        return Err(ErrorField {
            code: "42P07",
            message: "relation already exists",
            position: None,
        });
    }
    let oid = session.next_relation_oid;
    session.next_relation_oid = session.next_relation_oid.checked_add(1).ok_or(ErrorField {
        code: "54000",
        message: "relation OID allocation exhausted",
        position: None,
    })?;
    session.sequences.insert(
        name.to_string(),
        Sequence {
            oid,
            name: name.to_string(),
            last_value: 1,
            is_called: false,
        },
    );
    session.mark_sequence_dirty(name.to_string());
    Ok(())
}

pub(super) fn rename_sequence_in_session(
    session: &mut Session,
    old_name: &str,
    new_name: &str,
) -> Result<(), ErrorField> {
    if session.tables.contains_key(old_name)
        || session.views.contains_key(old_name)
        || session.materialized_views.contains_key(old_name)
    {
        return Err(ErrorField {
            code: "42809",
            message: "relation is not a sequence",
            position: None,
        });
    }
    if !session.sequences.contains_key(old_name) {
        return Err(ErrorField {
            code: "42P01",
            message: "sequence does not exist",
            position: None,
        });
    }
    if session.tables.contains_key(new_name)
        || session.views.contains_key(new_name)
        || session.materialized_views.contains_key(new_name)
        || session.sequences.contains_key(new_name)
    {
        return Err(ErrorField {
            code: "42P07",
            message: "relation already exists",
            position: None,
        });
    }
    let mut sequence = session
        .sequences
        .remove(old_name)
        .expect("sequence existence validated");
    sequence.name = new_name.to_string();
    session.sequences.insert(new_name.to_string(), sequence);
    if let Some(value) = session.currval_sequences.remove(old_name) {
        session
            .currval_sequences
            .insert(new_name.to_string(), value);
    }
    if let Some(acl) = session.table_acls.remove(old_name) {
        session.table_acls.insert(new_name.to_string(), acl);
        session.mark_table_acl_dirty(old_name.to_string());
        session.mark_table_acl_dirty(new_name.to_string());
    }
    session.mark_sequence_dirty(old_name.to_string());
    session.mark_sequence_dirty(new_name.to_string());

    let old_target = CatalogCommentTarget::Sequence {
        sequence: old_name.to_string(),
    };
    if let Some(comment) = session.comments.remove(&old_target) {
        let new_target = CatalogCommentTarget::Sequence {
            sequence: new_name.to_string(),
        };
        session.comments.insert(new_target.clone(), comment);
        session.mark_comment_dirty(old_target);
        session.mark_comment_dirty(new_target);
    }

    session.persist_catalog_snapshot();
    Ok(())
}

pub(super) fn execute_sequence_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
    include_row_description: bool,
) -> io::Result<()> {
    match command {
        Command::CreateSequence(create) => {
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
            if session.tables.contains_key(&create.name)
                || session.views.contains_key(&create.name)
                || session.materialized_views.contains_key(&create.name)
                || session.sequences.contains_key(&create.name)
                || session.domains.contains_key(&create.name)
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
            let name = create.name;
            session.sequences.insert(
                name.clone(),
                Sequence {
                    oid,
                    name: name.clone(),
                    last_value: 1,
                    is_called: false,
                },
            );
            session.mark_sequence_dirty(name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE SEQUENCE")
        }
        Command::SequenceNextVal(nextval) => {
            if let Some(error) = sequence_target_error(session, &nextval.name) {
                return write_error(stream, &error);
            }
            if let Some(error) =
                object_access_permission_error(session, &nextval.name, TablePrivilege::Update)
            {
                return write_error(stream, &error);
            }
            let sequence = session
                .sequences
                .get_mut(&nextval.name)
                .expect("sequence target checked");
            let value = match next_sequence_value(sequence) {
                Ok(value) => value,
                Err(error) => return write_error(stream, &error),
            };
            session
                .currval_sequences
                .insert(nextval.name.clone(), value);
            session.mark_sequence_dirty(nextval.name);
            session.persist_catalog_snapshot();
            write_select_rows(
                stream,
                &[int8_column("nextval")],
                &[vec![Some(value.to_string())]],
                include_row_description,
            )
        }
        Command::SequenceCurrVal(currval) => {
            if let Some(error) = sequence_target_error(session, &currval.name) {
                return write_error(stream, &error);
            }
            if let Some(error) =
                object_access_permission_error(session, &currval.name, TablePrivilege::Select)
            {
                return write_error(stream, &error);
            }
            let Some(value) = session.currval_sequences.get(&currval.name).copied() else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "55000",
                        message: "currval of sequence is not yet defined in this session",
                        position: None,
                    },
                );
            };
            write_select_rows(
                stream,
                &[int8_column("currval")],
                &[vec![Some(value.to_string())]],
                include_row_description,
            )
        }
        Command::SequenceSetVal(setval) => {
            if let Some(error) = sequence_target_error(session, &setval.name) {
                return write_error(stream, &error);
            }
            if let Some(error) =
                object_access_permission_error(session, &setval.name, TablePrivilege::Update)
            {
                return write_error(stream, &error);
            }
            let value = setval.value;
            let sequence = session
                .sequences
                .get_mut(&setval.name)
                .expect("sequence target checked");
            sequence.last_value = value;
            sequence.is_called = setval.is_called;
            session.mark_sequence_dirty(setval.name);
            session.persist_catalog_snapshot();
            write_select_rows(
                stream,
                &[int8_column("setval")],
                &[vec![Some(value.to_string())]],
                include_row_description,
            )
        }
        Command::RenameSequence(rename) => {
            if let Err(error) =
                rename_sequence_in_session(session, &rename.old_name, &rename.new_name)
            {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER SEQUENCE")
        }
        Command::DropSequence(drop) => {
            let mut seen = BTreeSet::new();
            for name in &drop.names {
                if !seen.insert(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "sequence specified more than once",
                            position: None,
                        },
                    );
                }
                if session.tables.contains_key(name)
                    || session.views.contains_key(name)
                    || session.materialized_views.contains_key(name)
                {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a sequence",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.sequences.contains_key(name) {
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
            for name in &drop.names {
                if session.sequences.remove(name).is_some() {
                    session.table_acls.remove(name);
                    session.mark_table_acl_dirty(name.clone());
                    let target = CatalogCommentTarget::Sequence {
                        sequence: name.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                    session.currval_sequences.remove(name);
                }
                session.mark_sequence_dirty(name.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP SEQUENCE")
        }
        _ => unreachable!("sequence executor called with an unrelated command"),
    }
}
