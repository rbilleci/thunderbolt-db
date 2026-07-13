// Legacy bounded-function ownership. This is not a product execution path.

use super::{
    execute_function_result, rename_function_in_session, schema_permission_error,
    write_command_complete, write_error, write_select_rows, CatalogCommentTarget, Command,
    ErrorField, FunctionInfo, ReadWrite, SchemaPrivilege, Session,
};
use std::collections::BTreeMap;
use std::io;

pub(super) fn execute_function_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
    include_row_description: bool,
) -> io::Result<()> {
    match command {
        Command::CreateFunction(create) => {
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
            if session.functions.contains_key(&create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42723",
                        message: "function already exists with same argument types",
                        position: None,
                    },
                );
            }
            let oid = session.next_relation_oid;
            let Some(next_oid) = session.next_relation_oid.checked_add(1) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "54000",
                        message: "function OID allocation exhausted",
                        position: None,
                    },
                );
            };
            session.next_relation_oid = next_oid;
            session.functions.insert(
                create.name.clone(),
                FunctionInfo {
                    oid,
                    name: create.name.clone(),
                    return_type: create.return_type,
                    body: create.body,
                    acl: BTreeMap::new(),
                },
            );
            session.mark_function_dirty(create.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE FUNCTION")
        }
        Command::RenameFunction(rename) => {
            if let Err(error) =
                rename_function_in_session(session, &rename.old_name, &rename.new_name)
            {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER FUNCTION")
        }
        Command::DropFunction(drop) => {
            if !drop.if_exists && !session.functions.contains_key(&drop.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42883",
                        message: "function does not exist",
                        position: None,
                    },
                );
            }
            if session.functions.remove(&drop.name).is_some() {
                let target = CatalogCommentTarget::Function {
                    function: drop.name.clone(),
                };
                session.comments.remove(&target);
                session.mark_comment_dirty(target);
            }
            session.mark_function_dirty(drop.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP FUNCTION")
        }
        Command::SelectFunction(call) => {
            let result = match execute_function_result(session, &call) {
                Ok(result) => result,
                Err(error) => return write_error(stream, &error),
            };
            write_select_rows(
                stream,
                &result.columns,
                &result.rows,
                include_row_description,
            )
        }
        _ => unreachable!("function executor called with an unrelated command"),
    }
}
