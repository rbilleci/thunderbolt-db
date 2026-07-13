// Legacy session compatibility ownership. This is not a product execution path.

use super::{
    bool_column, text_column, write_command_complete, write_select_rows, write_single_row,
    ReadWrite,
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
