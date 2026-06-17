//! SQL -> `ResidentExpr` binding via libpg_query (Charter rule 2;
//! `docs/architecture/18-sql-to-expr-handoff.md`). The `pg_query` crate vendors libpg_query — the
//! real PostgreSQL parser — so a SQL string is parsed by Postgres's own grammar and then walked into
//! the engine's general `ResidentExpr` IR (`engine_expr.rs`) and routed to
//! `execute_resident_expr_select` on the GPU. This is the "close the loop" path: SQL text -> general
//! GPU execution. It deliberately does NOT extend the hand-rolled `gpu_db_sql` parser and is NOT a
//! catalog of query shapes — coverage grows by node/type/operator (Charter rule 2).
//!
//! Build slices (doc 18 §2.5): this commit lands the **parse foundation** — `parse_single_select`
//! lifts the one `SELECT` statement out of the libpg_query parse tree. The AST -> `ResidentExpr`
//! mapper and the SQL-text engine entry are the next slices; until a non-test caller lands they are
//! unused in a normal build, so the module allows dead code (dropped when the entry is wired).
#![allow(dead_code)]

use super::*;

use pg_query::protobuf::SelectStmt;
use pg_query::NodeEnum;

/// Parse `sql` with libpg_query (Postgres's grammar) and lift out the single `SELECT` statement's
/// parse tree. Errors — mapped to the engine's `ApplyFailed` — on a parse failure, an empty or
/// multi-statement string, or a non-`SELECT` command: the general GPU executor binds read queries
/// only, and the bind/map layers above resolve the table, projection, and predicate from this tree.
pub(crate) fn parse_single_select(sql: &str) -> Result<SelectStmt, ExecuteError> {
    let parsed = pg_query::parse(sql).map_err(|err| sql_pg_error(format!("SQL parse error: {err}")))?;
    let mut stmts = parsed.protobuf.stmts;
    if stmts.len() != 1 {
        return Err(sql_pg_error(format!(
            "expected exactly one SQL statement, found {}",
            stmts.len()
        )));
    }
    let node = stmts
        .remove(0)
        .stmt
        .and_then(|boxed| boxed.node)
        .ok_or_else(|| sql_pg_error("empty SQL statement".to_string()))?;
    match node {
        NodeEnum::SelectStmt(select) => Ok(*select),
        _ => Err(sql_pg_error(
            "the general GPU executor binds SELECT statements only".to_string(),
        )),
    }
}

/// Wrap a SQL->Expr binding failure as the engine's standard `ApplyFailed` execution error (the same
/// surface the rest of the relational path uses), so callers handle it uniformly.
fn sql_pg_error(message: String) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message))
}
