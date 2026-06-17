//! SQL -> `ResidentExpr` binding via libpg_query (Charter rule 2;
//! `docs/architecture/18-sql-to-expr-handoff.md`). The `pg_query` crate vendors libpg_query — the
//! real PostgreSQL parser — so a SQL string is parsed by Postgres's own grammar and then walked into
//! the engine's general `ResidentExpr` IR (`engine_expr.rs`) and executed by the general GPU executor.
//! This is the "close the loop" path: SQL text -> general GPU execution. It deliberately does NOT
//! extend the hand-rolled `gpu_db_sql` parser and is NOT a catalog of query shapes — coverage grows by
//! node / type / operator (Charter rule 2). Today it binds a single-table int4 `SELECT ... WHERE`
//! with an arithmetic/comparison/column-vs-column predicate; `AND`/`OR` and richer types/operators are
//! the next slices. Anything the mapper cannot represent is a hard error — never a silent mis-answer.

use super::*;

use pg_query::protobuf::{
    a_const, AExpr, AExprKind, BoolExpr, BoolExprType, ColumnRef, Node, SelectStmt, SetOperation,
};
use pg_query::NodeEnum;

use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};

impl Engine {
    /// Parse `sql` (libpg_query / real Postgres grammar), map it to the general `ResidentExpr` IR, and
    /// execute it on the GPU general executor (Charter rule 2). Supported shape: a single-table SELECT
    /// of int4 columns with an int4 `WHERE` predicate over arithmetic (`+ - *`), comparisons
    /// (`= <> < <= > >=`), and column-vs-column. The catalog is bound ONCE and the predicate's column
    /// indices are resolved against that SAME bound table, so the columns, projection, and residency
    /// snapshot all derive from one catalog generation (a concurrent shape-changing DDL cannot split
    /// the column resolution from the execution). Errors — not a silent fallback — on any shape the
    /// mapper cannot represent; the routing layer (slice 5) decides what to do with a rejection.
    pub fn execute_resident_expr_select_sql(
        &self,
        sql: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let stmt = parse_single_select(sql)?;
        let (select, qualifier) = build_select_from_select_stmt(&stmt)?;
        let where_node = stmt.where_clause.as_deref().ok_or_else(|| {
            sql_pg_error(
                "the general GPU executor requires a WHERE predicate (a full-table SELECT is not an \
                 expression filter)"
                    .to_string(),
            )
        })?;
        // Bind once; map the predicate against that SAME bound table; execute against that binding.
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(&select)?;
        let predicate = map_predicate_node(where_node, &table, &qualifier)?;
        self.execute_resident_expr_select_with_binding(&select, &table, bound, copin_s, &predicate)
    }
}

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

/// Build the hand-rolled `Select` (table + int4 projection) that the bind/execute path consumes. The
/// WHERE is NOT carried here — it is mapped to a `ResidentExpr` and evaluated by the general executor,
/// so `filter`/`filters`/`filter_groups` stay empty. Rejects the shapes the Expr path does not yet
/// model (joins / multiple FROM relations, set operations, CTEs, DISTINCT, GROUP BY / HAVING, window
/// functions, ORDER BY, LIMIT / OFFSET, row locking, SELECT INTO, VALUES).
fn build_select_from_select_stmt(stmt: &SelectStmt) -> Result<(Select, String), ExecuteError> {
    let [from] = stmt.from_clause.as_slice() else {
        return Err(sql_pg_error(
            "the general GPU executor supports exactly one FROM relation (no joins yet)".to_string(),
        ));
    };
    let NodeEnum::RangeVar(range_var) = node_enum(from)? else {
        return Err(sql_pg_error(
            "FROM must be a single base table (joins / subqueries are not on the Expr path yet)"
                .to_string(),
        ));
    };
    let table = range_var.relname.clone();
    // The single qualifier a `tbl.col` reference must use: PG hides the relation name behind an alias,
    // so an aliased relation is named only by its alias; an unaliased one by its relation name. The
    // mapper validates qualified column references against this and rejects any other (PG's "missing
    // FROM-clause entry"), so a stray qualifier is never silently resolved to this relation's column.
    let qualifier = range_var
        .alias
        .as_ref()
        .map(|alias| alias.aliasname.clone())
        .unwrap_or_else(|| table.clone());

    let unsupported = [
        (!stmt.distinct_clause.is_empty(), "DISTINCT"),
        (!stmt.group_clause.is_empty(), "GROUP BY"),
        (stmt.having_clause.is_some(), "HAVING"),
        (!stmt.window_clause.is_empty(), "window functions"),
        (!stmt.sort_clause.is_empty(), "ORDER BY"),
        (stmt.limit_count.is_some() || stmt.limit_offset.is_some(), "LIMIT / OFFSET"),
        (!stmt.locking_clause.is_empty(), "row locking (FOR UPDATE/SHARE)"),
        (stmt.with_clause.is_some(), "WITH / CTEs"),
        // A plain SELECT is SETOP_NONE (= 1; proto enums prefix Undefined = 0). UNION/INTERSECT/EXCEPT
        // put their inputs in larg/rarg with an empty top-level from_clause, so the single-FROM check
        // above already rejects them — this guard is defense-in-depth for a tree that carries a set-op
        // tag without the expected shape.
        (
            stmt.op != SetOperation::SetopNone as i32,
            "set operations (UNION/INTERSECT/EXCEPT)",
        ),
        (stmt.into_clause.is_some(), "SELECT INTO"),
        (!stmt.values_lists.is_empty(), "VALUES"),
    ];
    if let Some((_, clause)) = unsupported.iter().find(|(present, _)| *present) {
        return Err(sql_pg_error(format!(
            "{clause} is not on the general GPU executor's Expr path yet"
        )));
    }

    let projection = build_projection(&stmt.target_list, &qualifier)?;
    let select = Select {
        table,
        distinct: false,
        projection,
        group_by: None,
        having_groups: Vec::new(),
        filter: None,
        filters: Vec::new(),
        filter_groups: Vec::new(),
        order_by: None,
        limit: None,
        offset: None,
    };
    Ok((select, qualifier))
}

/// Map the SELECT target list to a projection: a lone `*` -> `All`, otherwise a list of plain column
/// names. Rejects column aliases and projected expressions / aggregates (only int4 column projection
/// is materialized today).
fn build_projection(
    target_list: &[Node],
    qualifier: &str,
) -> Result<SelectProjection, ExecuteError> {
    if let [only] = target_list {
        if let NodeEnum::ResTarget(res_target) = node_enum(only)? {
            if let Some(val) = res_target.val.as_deref() {
                if let NodeEnum::ColumnRef(column_ref) = node_enum(val)? {
                    if column_ref.fields.len() == 1
                        && matches!(node_enum(&column_ref.fields[0])?, NodeEnum::AStar(_))
                    {
                        return Ok(SelectProjection::All);
                    }
                }
            }
        }
    }

    let mut columns = Vec::with_capacity(target_list.len());
    for target in target_list {
        let NodeEnum::ResTarget(res_target) = node_enum(target)? else {
            return Err(sql_pg_error("unexpected SELECT target".to_string()));
        };
        if !res_target.name.is_empty() {
            return Err(sql_pg_error(
                "column aliases are not on the Expr path yet".to_string(),
            ));
        }
        let val = res_target
            .val
            .as_deref()
            .ok_or_else(|| sql_pg_error("SELECT target has no value expression".to_string()))?;
        let NodeEnum::ColumnRef(column_ref) = node_enum(val)? else {
            return Err(sql_pg_error(
                "the general GPU executor projects plain columns only (no expressions / aggregates \
                 yet)"
                    .to_string(),
            ));
        };
        columns.push(resolve_column_name(column_ref, qualifier)?.to_string());
    }
    if columns.is_empty() {
        return Err(sql_pg_error("SELECT has no projected columns".to_string()));
    }
    Ok(SelectProjection::Columns(columns))
}

/// Map a WHERE / predicate parse node to the general `ResidentExpr` IR, resolving column names to
/// indices against `table`. Recurses through `A_Expr` (arithmetic + comparison). Anything the IR
/// cannot represent — `BoolExpr` (`AND`/`OR`, slice 4), non-operator `A_Expr` (IN / LIKE / BETWEEN),
/// non-integer literals, function calls, subqueries — is a hard error so the routing layer never
/// silently mis-answers.
fn map_predicate_node(
    node: &Node,
    table: &RelationalTable,
    qualifier: &str,
) -> Result<ResidentExpr, ExecuteError> {
    match node_enum(node)? {
        NodeEnum::ColumnRef(column_ref) => Ok(ResidentExpr::Column(relational_column_index(
            table,
            resolve_column_name(column_ref, qualifier)?,
        )?)),
        NodeEnum::AConst(constant) => match &constant.val {
            Some(a_const::Val::Ival(integer)) => Ok(ResidentExpr::Int4Literal(integer.ival)),
            _ => Err(sql_pg_error(
                "the general GPU executor supports int4 literals only".to_string(),
            )),
        },
        NodeEnum::AExpr(a_expr) => map_a_expr(a_expr, table, qualifier),
        NodeEnum::BoolExpr(bool_expr) => map_bool_expr(bool_expr, table, qualifier),
        _ => Err(sql_pg_error(
            "unsupported expression node for the general GPU executor".to_string(),
        )),
    }
}

/// Map a `BoolExpr` (`AND` / `OR` / `NOT`) to the general IR. `AND` / `OR` left-fold their operands
/// into a binary tree of `Binary{And/Or}` — libpg_query flattens `a AND b AND c` into one BoolExpr
/// with N args, so a chain becomes `((a AND b) AND c)`. `NOT` (a unary boolean) has no IR node yet and
/// is a hard error.
fn map_bool_expr(
    bool_expr: &BoolExpr,
    table: &RelationalTable,
    qualifier: &str,
) -> Result<ResidentExpr, ExecuteError> {
    let op = if bool_expr.boolop == BoolExprType::AndExpr as i32 {
        ResidentBinaryOp::And
    } else if bool_expr.boolop == BoolExprType::OrExpr as i32 {
        ResidentBinaryOp::Or
    } else {
        // NOT_EXPR (or an unexpected/Undefined boolop): unary NOT is not on the Expr path yet.
        return Err(sql_pg_error(
            "NOT predicates are not on the Expr path yet".to_string(),
        ));
    };
    let mut args = bool_expr.args.iter();
    let first = args
        .next()
        .ok_or_else(|| sql_pg_error("boolean expression has no operands".to_string()))?;
    let mut folded = map_predicate_node(first, table, qualifier)?;
    for arg in args {
        folded = ResidentExpr::Binary {
            op,
            lhs: Box::new(folded),
            rhs: Box::new(map_predicate_node(arg, table, qualifier)?),
        };
    }
    Ok(folded)
}

/// Map an `A_Expr` (a binary-operator expression) to a `ResidentExpr::Binary`. Only `AEXPR_OP` (a
/// normal operator: `+ - * = <> < <= > >=`) with both operands present is supported; other kinds
/// (`IN` / `LIKE` / `BETWEEN` / ...) and unsupported operator tokens (`/`, `%`, ...) are rejected.
fn map_a_expr(
    a_expr: &AExpr,
    table: &RelationalTable,
    qualifier: &str,
) -> Result<ResidentExpr, ExecuteError> {
    if a_expr.kind != AExprKind::AexprOp as i32 {
        return Err(sql_pg_error(
            "only normal operator predicates are supported (no IN / LIKE / BETWEEN yet)".to_string(),
        ));
    }
    let op = map_operator(aexpr_op_token(a_expr)?)?;
    let lexpr = a_expr
        .lexpr
        .as_deref()
        .ok_or_else(|| sql_pg_error("operator expression is missing its left operand".to_string()))?;
    let rexpr = a_expr.rexpr.as_deref().ok_or_else(|| {
        sql_pg_error(
            "operator expression is missing its right operand (unary operators unsupported)"
                .to_string(),
        )
    })?;
    Ok(ResidentExpr::Binary {
        op,
        lhs: Box::new(map_predicate_node(lexpr, table, qualifier)?),
        rhs: Box::new(map_predicate_node(rexpr, table, qualifier)?),
    })
}

/// Map a Postgres operator token to a `ResidentBinaryOp`. `<>` is PG's not-equal (PG normalizes `!=`
/// to `<>` in the grammar).
fn map_operator(token: &str) -> Result<ResidentBinaryOp, ExecuteError> {
    Ok(match token {
        "+" => ResidentBinaryOp::Add,
        "-" => ResidentBinaryOp::Sub,
        "*" => ResidentBinaryOp::Mul,
        "=" => ResidentBinaryOp::Eq,
        "<>" => ResidentBinaryOp::Ne,
        "<" => ResidentBinaryOp::Lt,
        "<=" => ResidentBinaryOp::Le,
        ">" => ResidentBinaryOp::Gt,
        ">=" => ResidentBinaryOp::Ge,
        other => {
            return Err(sql_pg_error(format!(
                "operator \"{other}\" is not supported by the general GPU executor yet"
            )))
        }
    })
}

/// The populated `NodeEnum` of a parse `Node` (a node with no inner value is a parser inconsistency).
fn node_enum(node: &Node) -> Result<&NodeEnum, ExecuteError> {
    node.node
        .as_ref()
        .ok_or_else(|| sql_pg_error("empty libpg_query parse node".to_string()))
}

/// The column name a `ColumnRef` resolves to, VALIDATING any table qualifier against the bound
/// relation's `qualifier` (its alias, else its name). `a` -> "a"; `t.a` / `alias.a` -> "a" only when
/// the qualifier names the FROM relation, else PG's "missing FROM-clause entry for table ..." (so a
/// stray `wrongtab.a` is a hard error, never silently resolved to this relation's column — load-
/// bearing once joins make two same-named columns ambiguous). Rejects `*`, qualified-star, and
/// schema/catalog-qualified (3+ part) references.
fn resolve_column_name<'a>(
    column_ref: &'a ColumnRef,
    qualifier: &str,
) -> Result<&'a str, ExecuteError> {
    let parts = column_ref
        .fields
        .iter()
        .map(|field| match field.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(string.sval.as_str()),
            _ => Err(sql_pg_error(
                "expected a column name (got `*` or a non-name column reference)".to_string(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match parts.as_slice() {
        [column] => Ok(column),
        [table_qualifier, column] => {
            if *table_qualifier == qualifier {
                Ok(column)
            } else {
                Err(sql_pg_error(format!(
                    "missing FROM-clause entry for table \"{table_qualifier}\""
                )))
            }
        }
        _ => Err(sql_pg_error(
            "schema-qualified or multi-part column references are not supported".to_string(),
        )),
    }
}

/// The operator token of an `A_Expr` (`name` carries it as a single `String` node).
fn aexpr_op_token(a_expr: &AExpr) -> Result<&str, ExecuteError> {
    match a_expr.name.as_slice() {
        [name] => match name.node.as_ref() {
            Some(NodeEnum::String(string)) => Ok(&string.sval),
            _ => Err(sql_pg_error("operator name is not a String node".to_string())),
        },
        _ => Err(sql_pg_error(
            "schema-qualified or multi-part operators are not supported".to_string(),
        )),
    }
}

/// Wrap a SQL->Expr binding failure as the engine's standard `ApplyFailed` execution error (the same
/// surface the rest of the relational path uses), so callers handle it uniformly.
fn sql_pg_error(message: String) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message))
}
