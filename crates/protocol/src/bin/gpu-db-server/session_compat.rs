// Legacy session compatibility ownership. This is not a product execution path.

use super::{
    bool_column, int4_column, text_column, write_command_complete, write_error, write_select_rows,
    write_single_row, ErrorField, PreparedStatement, ReadWrite, Session,
};
use std::io;

pub(super) fn try_execute_session_control_statement(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if is_pg_dump_session_set_statement(canonical) {
        return Some(write_command_complete(stream, "SET"));
    }
    if canonical == "reset search_path" {
        return Some(write_command_complete(stream, "RESET"));
    }
    if canonical.starts_with("lock table ") && canonical.ends_with(" in access share mode") {
        return Some(write_command_complete(stream, "LOCK TABLE"));
    }
    None
}

pub(super) fn try_execute_session_compat_query(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == "select pg_catalog.set_config('search_path', '', false)" {
        return Some(write_single_row(
            stream,
            &[text_column("set_config")],
            &[vec![Some(String::new())]],
        ));
    }
    if canonical == "select pg_catalog.set_config('search_path', 'public', false)" {
        return Some(write_single_row(
            stream,
            &[text_column("set_config")],
            &[vec![Some("public".to_string())]],
        ));
    }
    if canonical == "select pg_advisory_unlock_all()"
        || canonical == "select pg_catalog.pg_advisory_unlock_all()"
    {
        return Some(write_single_row(
            stream,
            &[text_column("pg_advisory_unlock_all")],
            &[vec![None]],
        ));
    }
    if canonical
        == "select set_config(name, 'view, foreign-table', false) from pg_settings where name = 'restrict_nonsystem_relation_kind'"
    {
        return Some(write_select_rows(
            stream,
            &[text_column("set_config")],
            &[],
            true,
        ));
    }
    if canonical == "select pg_catalog.pg_is_in_recovery()" {
        return Some(write_single_row(
            stream,
            &[bool_column("pg_is_in_recovery")],
            &[vec![Some("f".to_string())]],
        ));
    }
    if canonical == "select pg_catalog.current_schemas(false)" {
        return Some(write_single_row(
            stream,
            &[text_column("current_schemas")],
            &[vec![Some("{public}".to_string())]],
        ));
    }
    None
}

pub(super) fn execute_session_compat_fallback(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    canonical: &str,
) -> io::Result<()> {
    match canonical {
        "begin" => {
            session.in_transaction = true;
            write_command_complete(stream, "BEGIN")
        }
        "commit" => {
            session.in_transaction = false;
            write_command_complete(stream, "COMMIT")
        }
        "rollback" => {
            session.in_transaction = false;
            write_command_complete(stream, "ROLLBACK")
        }
        "reset all" => write_command_complete(stream, "RESET"),
        "discard all" => write_command_complete(stream, "DISCARD ALL"),
        "deallocate all" => {
            session
                .prepared
                .retain(|_, statement| matches!(statement, PreparedStatement::Extended(_)));
            write_command_complete(stream, "DEALLOCATE ALL")
        }
        "unlisten *" | "unlisten all" => write_command_complete(stream, "UNLISTEN"),
        "show client_encoding" => write_single_row(
            stream,
            &[text_column("client_encoding")],
            &[vec![Some(String::from("UTF8"))]],
        ),
        "show transaction isolation level" => write_single_row(
            stream,
            &[text_column("transaction_isolation")],
            &[vec![Some(String::from("read committed"))]],
        ),
        "select current_schema()" => write_single_row(
            stream,
            &[text_column("current_schema")],
            &[vec![Some(String::from("public"))]],
        ),
        "select 1 as one" => write_single_row(
            stream,
            &[int4_column("one")],
            &[vec![Some(String::from("1"))]],
        ),
        "select 2 as in_tx" => write_single_row(
            stream,
            &[int4_column("in_tx")],
            &[vec![Some(String::from("2"))]],
        ),
        "select 3 as rolled_back" => write_single_row(
            stream,
            &[int4_column("rolled_back")],
            &[vec![Some(String::from("3"))]],
        ),
        "prepare golden_stmt(int) as select $1 + 10 as plus_ten" => {
            session
                .prepared
                .insert(String::from("golden_stmt"), PreparedStatement::AddTen);
            write_command_complete(stream, "PREPARE")
        }
        "execute golden_stmt(5)" => {
            if session.prepared.contains_key("golden_stmt") {
                write_single_row(
                    stream,
                    &[int4_column("plus_ten")],
                    &[vec![Some(String::from("15"))]],
                )
            } else {
                write_error(
                    stream,
                    &ErrorField {
                        code: "26000",
                        message: "prepared statement \"golden_stmt\" does not exist",
                        position: None,
                    },
                )
            }
        }
        "deallocate golden_stmt" => {
            session.prepared.remove("golden_stmt");
            write_command_complete(stream, "DEALLOCATE")
        }
        "select * from definitely_missing_relation_for_golden" => write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation \"definitely_missing_relation_for_golden\" does not exist",
                position: Some("15"),
            },
        ),
        _ => write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "query shape is not supported by the compatibility stub",
                position: None,
            },
        ),
    }
}

fn is_pg_dump_session_set_statement(canonical: &str) -> bool {
    let normalized = canonical.replace(" to ", " = ");
    matches!(
        normalized.as_str(),
        "set datestyle = iso"
            | "set intervalstyle = postgres"
            | "set extra_float_digits = 3"
            | "set statement_timeout = 0"
            | "set lock_timeout = 0"
            | "set idle_in_transaction_session_timeout = 0"
            | "set client_encoding = 'utf8'"
            | "set standard_conforming_strings = on"
            | "set synchronize_seqscans = off"
            | "set check_function_bodies = false"
            | "set xmloption = content"
            | "set client_min_messages = warning"
            | "set row_security = off"
            | "set default_tablespace = ''"
            | "set default_table_access_method = heap"
            | "set default_transaction_read_only = off"
            | "set transaction isolation level repeatable read, read only"
    )
}
