//! SQL -> `ResidentExpr` binding via libpg_query (`engine_sql_pg`, Charter rule 2,
//! docs/architecture/18-sql-to-expr-handoff.md). Slice 1 pins the libpg_query (`pg_query` v6) parse
//! API this binding walks: it parses an arithmetic-predicate `SELECT` and asserts the exact parse-
//! tree shape (`SelectStmt` -> target_list/from_clause/where_clause -> `ResTarget`/`RangeVar`/
//! `A_Expr`/`ColumnRef`/`A_Const`) the AST -> `ResidentExpr` mapper consumes in the next slice. This
//! both proves the heavy libpg_query C build links and documents the navigation, so a future
//! `pg_query` API drift fails here loudly rather than silently in the mapper.

use super::*;

use crate::engine_sql_pg::parse_single_select;
use pg_query::protobuf::{a_const, AConst, AExpr, AExprKind, ColumnRef, Node};
use pg_query::NodeEnum;

/// The populated `NodeEnum` inside a parse `Node` (every node libpg_query emits in a valid tree is
/// populated; an empty one is a parser bug we want to surface).
fn node_enum(node: &Node) -> &NodeEnum {
    node.node.as_ref().expect("libpg_query parse node is populated")
}

/// The (single) column name of a `ColumnRef` — `fields` is the dotted path; an unqualified `a` is a
/// one-element list of a `String` node.
fn column_ref_name(column_ref: &ColumnRef) -> &str {
    match column_ref.fields.first().and_then(|node| node.node.as_ref()) {
        Some(NodeEnum::String(string)) => &string.sval,
        other => panic!("column ref field is not a String node: {other:?}"),
    }
}

/// The operator token of an `A_Expr` (`name` carries the operator as a `String` node, e.g. `+`, `>`,
/// `<>`).
fn aexpr_op(expr: &AExpr) -> &str {
    match expr.name.first().and_then(|node| node.node.as_ref()) {
        Some(NodeEnum::String(string)) => &string.sval,
        other => panic!("a_expr operator name is not a String node: {other:?}"),
    }
}

/// The int32 value of an integer `A_Const` (a literal that fits int4 arrives as `Ival(Integer)`).
fn aconst_int(constant: &AConst) -> i32 {
    match &constant.val {
        Some(a_const::Val::Ival(integer)) => integer.ival,
        other => panic!("a_const is not an integer literal: {other:?}"),
    }
}

#[test]
fn libpg_query_parses_arithmetic_predicate_select_into_the_expected_tree() {
    // `SELECT a FROM t WHERE a + b > 400` is the slice-2 end-to-end target. Its parse tree is exactly
    // the node set the AST -> ResidentExpr mapper walks: a single projected ColumnRef, a single
    // RangeVar table, and a WHERE that is A_Expr(`>`, A_Expr(`+`, ColumnRef a, ColumnRef b),
    // A_Const 400). Asserting the full shape proves the libpg_query build + pins the v6 API.
    let select = parse_single_select("SELECT a FROM t WHERE a + b > 400")
        .expect("libpg_query parses a single SELECT");

    // Projection: target_list = [ ResTarget { val: ColumnRef "a" } ].
    assert_eq!(select.target_list.len(), 1, "one projected column");
    let NodeEnum::ResTarget(res_target) = node_enum(&select.target_list[0]) else {
        panic!("target_list entry is not a ResTarget");
    };
    let projected = res_target.val.as_deref().expect("ResTarget projects a value");
    let NodeEnum::ColumnRef(projected_col) = node_enum(projected) else {
        panic!("projected value is not a ColumnRef");
    };
    assert_eq!(column_ref_name(projected_col), "a");

    // FROM: from_clause = [ RangeVar "t" ].
    assert_eq!(select.from_clause.len(), 1, "one FROM relation");
    let NodeEnum::RangeVar(range_var) = node_enum(&select.from_clause[0]) else {
        panic!("from_clause entry is not a RangeVar");
    };
    assert_eq!(range_var.relname, "t");

    // WHERE: A_Expr(AEXPR_OP `>`, lexpr = A_Expr(`+`, a, b), rexpr = A_Const 400).
    let where_node = select.where_clause.as_deref().expect("WHERE clause present");
    let NodeEnum::AExpr(comparison) = node_enum(where_node) else {
        panic!("WHERE is not an A_Expr");
    };
    assert_eq!(
        comparison.kind,
        AExprKind::AexprOp as i32,
        "comparison is a normal operator A_Expr (AEXPR_OP)"
    );
    assert_eq!(aexpr_op(comparison), ">");

    let lhs = comparison.lexpr.as_deref().expect("comparison lhs");
    let NodeEnum::AExpr(addition) = node_enum(lhs) else {
        panic!("comparison lhs is not an A_Expr");
    };
    assert_eq!(aexpr_op(addition), "+");
    let NodeEnum::ColumnRef(add_lhs) = node_enum(addition.lexpr.as_deref().expect("add lhs")) else {
        panic!("addition lhs is not a ColumnRef");
    };
    let NodeEnum::ColumnRef(add_rhs) = node_enum(addition.rexpr.as_deref().expect("add rhs")) else {
        panic!("addition rhs is not a ColumnRef");
    };
    assert_eq!(column_ref_name(add_lhs), "a");
    assert_eq!(column_ref_name(add_rhs), "b");

    let rhs = comparison.rexpr.as_deref().expect("comparison rhs");
    let NodeEnum::AConst(literal) = node_enum(rhs) else {
        panic!("comparison rhs is not an A_Const");
    };
    assert_eq!(aconst_int(literal), 400);
}

#[test]
fn parse_single_select_rejects_non_select_and_multi_statement() {
    // The general GPU executor binds read queries: a non-SELECT command and a multi-statement string
    // are hard errors (not silent mis-binds), so the routing layer can fall through cleanly.
    assert!(
        parse_single_select("INSERT INTO t (a) VALUES (1)").is_err(),
        "a non-SELECT command must not bind as a SELECT"
    );
    assert!(
        parse_single_select("SELECT a FROM t; SELECT b FROM t").is_err(),
        "a multi-statement string must be rejected"
    );
    assert!(
        parse_single_select("SELECT a FROM").is_err(),
        "a syntactically invalid statement surfaces the libpg_query parse error"
    );
}

// ---- Slice 3: SQL text -> ResidentExpr -> general GPU executor ----

/// Assert `sql` errors through the SQL->Expr entry with a message containing `needle` (never returns
/// rows). Rejections must be hard errors so the routing layer decides, not silent mis-answers.
fn assert_sql_err_contains(engine: &Engine, sql: &str, needle: &str) {
    match engine.execute_resident_expr_select_sql(sql) {
        Ok(_) => panic!("expected `{sql}` to error, but it returned rows"),
        Err(err) => assert!(
            err.to_string().contains(needle),
            "`{sql}` should error mentioning `{needle}`, got: {err}"
        ),
    }
}

#[test]
fn execute_resident_expr_select_sql_rejects_unsupported_shapes() {
    // Every shape the mapper cannot represent is a HARD error (deterministic, no GPU needed): build-
    // stage rejections (multiple FROM relations, aggregate projection, ORDER BY) fire before the
    // residency check; mapper-stage rejections (unsupported operator, AND/OR, non-int literal) fire
    // after the single bind. None silently mis-answer.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    assert_sql_err_contains(&e, "SELECT a FROM t", "WHERE"); // a filter needs a predicate
    assert_sql_err_contains(&e, "SELECT a FROM t x, t y WHERE a > 0", "one FROM relation"); // join
    assert_sql_err_contains(&e, "SELECT count(*) FROM t WHERE a > 0", "plain columns"); // aggregate
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE a > 0 ORDER BY a", "ORDER BY"); // ordering
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE a / b > 1", "/"); // unsupported operator
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE a > 1 AND b < 2", "AND"); // BoolExpr (slice 4)
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE a > 'x'", "int4 literals"); // non-int literal
    // A column qualifier that does not name the FROM relation is PG's "missing FROM-clause entry",
    // never silently resolved to t.a (load-bearing once joins make same-named columns ambiguous).
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE wrong.a > 0", "missing FROM-clause");
    assert_sql_err_contains(&e, "SELECT wrong.a FROM t WHERE a > 0", "missing FROM-clause");
    // An alias HIDES the relation name (PG): once `FROM t AS x`, `t.a` no longer names the relation.
    assert_sql_err_contains(&e, "SELECT a FROM t x WHERE t.a > 0", "missing FROM-clause");
}

#[test]
fn execute_resident_expr_select_sql_maps_supported_predicate_and_reaches_gpu_dispatch() {
    // A supported single-table int4 SELECT parses + maps + binds and reaches the GPU residency check —
    // proving the full SQL -> ResidentExpr binding succeeds end to end up to device dispatch. With no
    // residency snapshot populated it stops at the residency error (deterministic on any box), which
    // is PAST parse/map/bind — i.e. NOT a mapper rejection. (The GPU e2e test below runs it through.)
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    match e.execute_resident_expr_select_sql("SELECT a FROM t WHERE a + b > 400") {
        Ok(_) => panic!("no residency snapshot populated, so this cannot return rows"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("resident"),
                "a supported predicate must reach GPU dispatch (residency stage), not a mapper \
                 rejection; got: {msg}"
            );
        }
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_predicate_from_sql_text() {
    // The whole loop: a SQL STRING -> libpg_query -> ResidentExpr -> general GPU executor, end to end.
    // Same GPU-native closed-form oracle as the programmatic Expr tests, now driven from real SQL.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    const N: i32 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, b) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // Arithmetic predicate (the canonical handoff query): a[i]=b[i]=i, a+b=2i monotone, so
    // {i : a+b>400} = [201, 600); projected a[i]=i. Distinct from "only a" (a>400 => [401, 600)).
    let added = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a + b > 400")
        .expect("SQL `a + b > 400` -> GPU");
    let added_expected: Vec<Vec<SqlValue>> = (201..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(added.columns.len(), 1);
    assert_eq!(added.columns[0].name, "a");
    assert_eq!(
        added.rows, added_expected,
        "SQL `a + b > 400` on GPU must be a-values for i in [201, 600)"
    );
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(added.fallback_reason, None);
    assert_ne!(
        added.rows.len(),
        (N - 401) as usize,
        "must differ from a>400 — proves column b was read, not just a"
    );

    // A plain column-vs-literal comparison from SQL (exercises the comparison mapping + the simple
    // value-buffer compare path): a > 500 => [501, 600).
    let compared = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a > 500")
        .expect("SQL `a > 500` -> GPU");
    let compared_expected: Vec<Vec<SqlValue>> = (501..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        compared.rows, compared_expected,
        "SQL `a > 500` on GPU must be a-values for i in [501, 600)"
    );

    // A qualifier that DOES name the FROM relation resolves to its column: bare `t.a` and the alias
    // `x.a` both mean column a (a stray qualifier hard-errors — covered by the host rejection test).
    let qualified = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE t.a > 500")
        .expect("SQL `t.a > 500` -> GPU");
    assert_eq!(
        qualified.rows, compared_expected,
        "qualified `t.a > 500` must equal `a > 500`"
    );
    let aliased = e
        .execute_resident_expr_select_sql("SELECT x.a FROM t AS x WHERE x.a > 500")
        .expect("SQL alias `x.a > 500` -> GPU");
    assert_eq!(
        aliased.rows, compared_expected,
        "alias-qualified `x.a > 500` must equal `a > 500`"
    );
}
