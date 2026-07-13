// Legacy sequence ownership. This is not a product execution path.

use super::{
    int8_column, object_access_permission_error, schema_permission_error, write_command_complete,
    write_error, write_select_rows, CatalogCommentTarget, Command, ErrorField, ReadWrite,
    SchemaPrivilege, Sequence, Session, TablePrivilege,
};
use std::collections::BTreeSet;
use std::io;

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
