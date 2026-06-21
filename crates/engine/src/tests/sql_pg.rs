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
    // stage rejections (multiple FROM relations, aggregate projection) fire before the residency check;
    // mapper-stage rejections (unsupported operator, AND/OR, non-int literal) fire after the single
    // bind. None silently mis-answer.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();

    // NB: `SELECT a FROM t` (no WHERE) is NOT rejected any more -- it is a supported full-table scan
    // (covered by gpu_execute_resident_expr_select_sql_full_table_no_where).
    // A comma join WITH an equi-join condition is supported now (gpu_inner_join_comma_join_from_where);
    // a comma list with NO join condition between the relations is a cartesian product -> a clean reject.
    assert_sql_err_contains(&e, "SELECT a FROM t x, t y WHERE a > 0", "equi-join condition");
    // count(*) / sum / min / max / avg are supported now (operator axis, GPU-tested); count(col) and
    // other functions are follow-ons, still rejected at the parser.
    assert_sql_err_contains(&e, "SELECT count(a) FROM t WHERE a > 0", "COUNT(*) / SUM / MIN / MAX");
    // NB: a non-grouped ORDER BY over an int column is SUPPORTED now -- routed to the general GPU Expr
    // executor + the bitonic sort (covered by gpu_nongrouped_order_by_via_gpu_sort), no longer rejected.
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE a / b > 1", "/"); // unsupported operator
    assert_sql_err_contains(&e, "SELECT a FROM t WHERE NOT a > 1", "NOT"); // unary NOT (AND/OR are ok)
    // int4 / numeric / text / bool literals all map now (bool literals + `flag = true` / `NOT flag`
    // are GPU-tested in the bool slice); an unsupported expression NODE -- a function call, subquery,
    // etc. -- is still rejected at the mapper.
    assert_sql_err_contains(
        &e,
        "SELECT a FROM t WHERE a > abs(b)",
        "unsupported expression node",
    );
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

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_boolean_predicates_from_sql_text() {
    // AND / OR from SQL text -> BoolExpr -> Binary{And/Or} -> the mask VM on the GPU. Closed-form
    // oracle: a[i]=b[i]=i, so each comparison is a contiguous range and the boolean combinator is set
    // algebra on those ranges.
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

    // a>200 AND b<400 => i in (200,400) = [201,400). Differs from `a>200` alone ([201,600)), so a pass
    // proves the b<400 conjunct was AND-combined, not dropped.
    let and_rows = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a > 200 AND b < 400")
        .expect("SQL `a>200 AND b<400` -> GPU");
    let and_expected: Vec<Vec<SqlValue>> = (201..400).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(and_rows.rows, and_expected, "a>200 AND b<400 => [201, 400)");
    assert_eq!(and_rows.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(and_rows.fallback_reason, None);

    // a<100 OR b>500 => [0,100) U [501,600).
    let or_rows = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a < 100 OR b > 500")
        .expect("SQL `a<100 OR b>500` -> GPU");
    let mut or_expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int4(i)]).collect();
    or_expected.extend((501..N).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(or_rows.rows, or_expected, "a<100 OR b>500 => [0,100) U [501,600)");

    // A 3-way chain: libpg_query flattens `a AND a AND b` into one BoolExpr with 3 args, so the
    // left-fold must handle N>2. a>100 AND a<500 AND b>300 => i in (300,500) = [301,500).
    let chain = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a > 100 AND a < 500 AND b > 300")
        .expect("SQL 3-way AND chain -> GPU");
    let chain_expected: Vec<Vec<SqlValue>> = (301..500).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        chain.rows, chain_expected,
        "a>100 AND a<500 AND b>300 => [301, 500) (N-arg BoolExpr left-folded)"
    );

    // Nested mixed AND/OR with NO parens: AND binds tighter than OR, so libpg_query yields
    // OR(a<50, AND(a>200, b<400)) and the mapper must preserve that nesting. => [0,50) U [201,400).
    let nested = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a < 50 OR a > 200 AND b < 400")
        .expect("SQL nested AND/OR -> GPU");
    let mut nested_expected: Vec<Vec<SqlValue>> = (0..50).map(|i| vec![SqlValue::Int4(i)]).collect();
    nested_expected.extend((201..400).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(
        nested.rows, nested_expected,
        "a<50 OR a>200 AND b<400 => [0,50) U [201,400) (AND precedence preserved)"
    );
}

#[test]
fn hand_rolled_parser_rejects_arithmetic_so_routing_is_unambiguous() {
    // Routing precondition (slice 5): the text dispatch tries the hand-rolled parser FIRST (it gates
    // the tuned enumerated fast-paths + catalog) and only routes to the general Expr path when the
    // hand-rolled parser CANNOT express the SELECT. For that to never be a silent mis-answer, the
    // hand-rolled parser must ERROR (not mis-parse) on exactly the predicates the general path adds:
    // ARITHMETIC (column op column, scalar arithmetic, deeper trees), incl. arithmetic mixed with a
    // boolean. This test guards the boundary — if the hand-rolled parser ever starts accepting one of
    // these, the routing would silently skip the general path, so this must fail loudly.
    for sql in [
        "SELECT a FROM t WHERE a + b > 400",
        "SELECT a FROM t WHERE a * 2 > b",
        "SELECT a FROM t WHERE (a + b) * 2 - 5 > 395",
        "SELECT a FROM t WHERE a + b > 400 AND a < 500",
    ] {
        assert!(
            parse_command(sql).is_err(),
            "hand-rolled parser must REJECT `{sql}` so the text dispatch routes it to the general \
             Expr path"
        );
    }

    // Conversely, SIMPLE comparison conjunctions ARE hand-rolled-parseable (column op literal, ANDed/
    // ORed) and are handled by the existing path (resident-route filter groups), NOT the general path.
    // The general path's own boolean support still covers them when reached directly or mixed with
    // arithmetic; here we just pin that the routing boundary is "arithmetic", not "boolean".
    for sql in [
        "SELECT a FROM t WHERE a > 200 AND b < 400",
        "SELECT a FROM t WHERE a < 100 OR b > 500",
    ] {
        assert!(
            parse_command(sql).is_ok(),
            "hand-rolled parser handles simple comparison conjunctions (`{sql}`) — they stay on the \
             existing path"
        );
    }
}

#[test]
fn select_text_keeps_simple_predicates_on_the_existing_path() {
    // A hand-rolled-parseable SELECT (a simple comparison) is NOT routed to the general Expr path: the
    // text dispatch runs it through the existing path (here CPU, no residency populated) and returns
    // the right rows. Proves the routing only ADDS arithmetic coverage and never hijacks the shapes
    // the hand-rolled parser + tuned fast-paths already own.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (5, 1), (6, 2), (7, 3)")
        .unwrap();
    let result = e
        .execute_relational_select_text("SELECT a FROM t WHERE a = 6")
        .expect("simple equality runs on the existing path");
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(6)]],
        "a = 6 -> the single matching row, via the existing path"
    );
}

#[test]
fn select_text_non_resident_order_by_falls_through_to_existing_path() {
    // A non-grouped int ORDER BY routes to the general GPU (bitonic-sort) path -- but ONLY when the
    // table is GPU-RESIDENT (that path has no CPU fallback). On a NON-resident table the routing gate
    // falls through to the existing path, which sorts correctly rather than hard-erroring with "no
    // resident snapshot". Guards the residency condition on the GPU-sort routing.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (5), (2), (8), (1)")
        .unwrap();
    // Deliberately do NOT populate residency -> t is not GPU-resident.
    let result = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a DESC")
        .expect("non-resident ORDER BY falls through to the existing path, not a hard error");
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(8)],
            vec![SqlValue::Int4(5)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(1)],
        ],
        "non-resident ORDER BY DESC sorted via the existing path"
    );
}

#[test]
fn select_text_non_resident_text_order_by_falls_through_to_existing_path() {
    // A single-text-key ORDER BY routes to the GPU text sort ONLY when the table is GPU-resident. On a
    // NON-resident table the gate falls through to the existing path, which sorts the text correctly
    // rather than hard-erroring. Guards the residency condition for the text leg of the GPU sort.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (s TEXT)").unwrap();
    e.execute_text(2, "INSERT INTO t (s) VALUES ('banana'), ('apple'), ('cherry')")
        .unwrap();
    // Deliberately do NOT populate residency -> t is not GPU-resident.
    let result = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s")
        .expect("non-resident text ORDER BY falls through to the existing path, not a hard error");
    let got: Vec<String> = result
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(v) => v.to_string(),
            other => panic!("expected Text, got {other:?}"),
        })
        .collect();
    assert_eq!(
        got,
        vec!["apple", "banana", "cherry"],
        "non-resident text ORDER BY sorted via the existing path"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_multikey_order_by_sorts_on_the_gpu() {
    // Multi-key ORDER BY on a GROUPED result now sorts ON THE GPU (the grouped-sort migration): no
    // host-side sort, no first-key-only. A count tie is broken by the secondary key on-device. A
    // multi-aggregate GROUP BY routes via the Err arm to the general path's grouped branch.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE g (a INT)").unwrap();
    // counts: a=1->2, a=2->2, a=3->1. ORDER BY count ASC, a DESC -> count 1 (a=3), then the count-2 tie
    // by a DESC (a=2 then a=1) -> the `a` column = [3, 2, 1].
    e.execute_text(2, "INSERT INTO g (a) VALUES (1), (1), (2), (2), (3)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("g").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text(
            "SELECT a, COUNT(*), SUM(a) FROM g GROUP BY a ORDER BY count ASC, a DESC",
        )
        .expect("multi-key grouped ORDER BY now sorts on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    let a_order: Vec<SqlValue> = s.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(
        a_order,
        vec![SqlValue::Int4(3), SqlValue::Int4(2), SqlValue::Int4(1)],
        "count ASC then a DESC: a=3 (count 1), then the count-2 tie a=2,a=1 by a DESC"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_count_distinct_routes_through_text_entry() {
    // COUNT(DISTINCT v) via the wire/text dispatch (execute_relational_select_text): the hand-rolled
    // parser rejects DISTINCT, so the Err arm falls through to the general libpg_query path's grouped
    // branch -- the same user-facing route the PG-wire server uses.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g, v) VALUES (1,10),(1,10),(1,20),(2,5),(2,15)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
        .expect("COUNT(DISTINCT v) via the text dispatch");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(2)],
            vec![SqlValue::Int4(2), SqlValue::Int8(2)],
        ],
        "g=1 -> 2 distinct (10,20), g=2 -> 2 distinct (5,15)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn scalar_count_distinct_routes_through_text_entry() {
    // Scalar COUNT(DISTINCT v) (no GROUP BY) via the wire/text dispatch: the hand-rolled parser rejects
    // DISTINCT, so the Err arm falls through to the general path's scalar-aggregate branch.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (v) VALUES (10),(10),(20),(30),(30)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT COUNT(DISTINCT v) FROM t")
        .expect("scalar COUNT(DISTINCT v) via the text dispatch");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        vec![vec![SqlValue::Int8(3)]],
        "3 distinct values (10,20,30) across the table"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_two_relations_int_key() {
    // M5: a 2-relation INNER equi-join on an int key, via the general path's GPU hash join. parent.id
    // is UNIQUE (the build side); child.parent_id is the FK (1:N + an orphan + a childless parent).
    //   parent: (1,a),(2,b),(3,c)   child: (1,x),(1,y),(2,z),(99,orphan)
    //   parent JOIN child ON parent.id = child.parent_id -> (a,x),(a,y),(b,z); 99 + parent 3 dropped.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE parent (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE child (parent_id INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO parent (id, name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO child (parent_id, label) VALUES (1,'x'),(1,'y'),(2,'z'),(99,'orphan')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("parent").unwrap();
    let cs = e.populate_relational_residency_snapshot("child").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    // Extract (name, label) text pairs + sort (the join emit order is unspecified without ORDER BY).
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let s = |c: &SqlValue| match c {
                    SqlValue::Text(t) => t.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (s(&r[0]), s(&r[1]))
            })
            .collect();
        v.sort();
        v
    };
    let expected = vec![
        ("a".to_string(), "x".to_string()),
        ("a".to_string(), "y".to_string()),
        ("b".to_string(), "z".to_string()),
    ];
    // Via the general path directly + via the production wire/text dispatch (the hand-rolled parser
    // rejects JOIN -> the Err arm -> the general path).
    let direct = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join (general path)");
    assert_eq!(direct.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(direct.columns.len(), 2);
    assert!(direct.columns[0].name.eq_ignore_ascii_case("name"));
    assert!(direct.columns[1].name.eq_ignore_ascii_case("label"));
    assert_eq!(pairs(&direct), expected, "1:N inner join, orphan + childless dropped");
    let wire = e
        .execute_relational_select_text(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join (text/wire dispatch)");
    assert_eq!(wire.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(pairs(&wire), expected, "the wire dispatch routes JOIN to the general path");
    // Qualified projection (parent.name, child.label) resolves each column to its relation.
    let qualified = e
        .execute_resident_expr_select_sql(
            "SELECT parent.name, child.label FROM parent JOIN child ON parent.id = child.parent_id",
        )
        .expect("inner join, qualified projection");
    assert_eq!(pairs(&qualified), expected, "qualified column refs resolve per-relation");
    // The ON written the other way around (child.parent_id = parent.id) is the same join.
    let swapped = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON child.parent_id = parent.id",
        )
        .expect("inner join, ON operands swapped");
    assert_eq!(pairs(&swapped), expected, "ON operand order does not matter");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_build_fallback_when_smaller_side_not_unique() {
    // The SMALLER side (s, the left/probe-by-size) has a DUPLICATE join key, so build-on-smaller hits
    // DuplicateBuildKey -> the executor falls back to building on the LARGER (unique) side. This
    // exercises the fallback + the left_is_build = !smaller_is_left index mapping (a left column must
    // still read the left relation's rows after the build side flips).
    //   s: (1,'p'),(1,'q')  [smaller, key 1 duplicated]   l: (1,'A'),(2,'B'),(3,'C')  [larger, unique]
    //   s JOIN l ON s.k = l.k -> (p,A),(q,A).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE s (k INT, sv TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE l (k INT, lv TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO s (k, sv) VALUES (1,'p'),(1,'q')")
        .unwrap();
    e.execute_text(4, "INSERT INTO l (k, lv) VALUES (1,'A'),(2,'B'),(3,'C')")
        .unwrap();
    let ss = e.populate_relational_residency_snapshot("s").unwrap();
    let ls = e.populate_relational_residency_snapshot("l").unwrap();
    if ss.device_memory_proof.is_none() || ls.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT sv, lv FROM s JOIN l ON s.k = l.k")
        .expect("inner join with build fallback");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            other => panic!("expected text pair, got {other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![("p".to_string(), "A".to_string()), ("q".to_string(), "A".to_string())],
        "fallback build-on-larger keeps left(sv)/right(lv) rows correctly paired"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_with_where_pushed_per_side() {
    // M5 J3: a WHERE on a join, with per-relation conjuncts pushed to each side's GPU pre-filter.
    //   parent: (1,a),(2,b),(3,c)   child: (1,10,x),(1,20,y),(2,30,z),(3,40,w)
    //   ON parent.id = child.parent_id WHERE parent.id >= 2 AND child.v > 15
    //   -> parent survivors abs{1,2}, child survivors abs{1,2,3} (BOTH sides drop an EARLY row, so the
    //      matched survivor POSITION != the absolute row on both sides) -> (b,z),(c,w).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE parent (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE child (parent_id INT, v INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO parent (id, name) VALUES (1,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO child (parent_id, v, label) VALUES (1,10,'x'),(1,20,'y'),(2,30,'z'),(3,40,'w')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("parent").unwrap();
    let cs = e.populate_relational_residency_snapshot("child").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id \
             WHERE parent.id >= 2 AND child.v > 15",
        )
        .expect("inner join with per-side WHERE");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut pairs: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            other => panic!("expected text pair, got {other:?}"),
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![("b".to_string(), "z".to_string()), ("c".to_string(), "w".to_string())],
        "per-side WHERE pre-filters both relations (early-row drop on both -> survivor pos != abs row)"
    );
    // A WHERE that filters everything out -> no rows.
    let empty = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM parent JOIN child ON parent.id = child.parent_id \
             WHERE parent.id > 100",
        )
        .expect("inner join with an all-filtering WHERE");
    assert!(empty.rows.is_empty(), "a WHERE that drops all left rows -> no join output");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_int8_and_mixed_int_keys() {
    // M5 J4a: int8/bigint join keys (i64 section) -- incl. a value beyond the int4 range -- and a MIXED
    // int4=int8 join (both project to i64, so 5_i32 == 5_i64 matches).
    let mut e = Engine::new_local();
    // (a) both BIGINT keys, one beyond int4 range.
    e.execute_text(1, "CREATE TABLE big (id BIGINT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE rref (big_id BIGINT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO big (id, name) VALUES (9000000000,'a'),(2,'b'),(3,'c')")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO rref (big_id, label) VALUES (9000000000,'x'),(2,'y'),(2,'z'),(99,'w')",
    )
    .unwrap();
    // (b) mixed: INT key joined to a BIGINT key.
    e.execute_text(5, "CREATE TABLE p4 (id INT, name TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE c8 (pid BIGINT, label TEXT)")
        .unwrap();
    // Include a NEGATIVE key so the int4 sign-extension to i64 (-3 -> i64 -3) is checked against the
    // int8 -3 in the mixed join.
    e.execute_text(7, "INSERT INTO p4 (id, name) VALUES (5,'a'),(6,'b'),(-3,'n')")
        .unwrap();
    e.execute_text(8, "INSERT INTO c8 (pid, label) VALUES (5,'x'),(6,'y'),(6,'z'),(-3,'m')")
        .unwrap();
    for t in ["big", "rref", "p4", "c8"] {
        if e.populate_relational_residency_snapshot(t)
            .unwrap()
            .device_memory_proof
            .is_none()
        {
            return;
        }
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = res
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
                other => panic!("expected text pair, got {other:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let expected = vec![
        ("a".to_string(), "x".to_string()),
        ("b".to_string(), "y".to_string()),
        ("b".to_string(), "z".to_string()),
    ];
    let r1 = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM big JOIN rref ON big.id = rref.big_id",
        )
        .expect("int8 join");
    assert_eq!(r1.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(pairs(&r1), expected, "int8 keys incl. a value beyond int4 range");
    let r2 = e
        .execute_resident_expr_select_sql("SELECT name, label FROM p4 JOIN c8 ON p4.id = c8.pid")
        .expect("mixed int4=int8 join");
    let expected_mixed = vec![
        ("a".to_string(), "x".to_string()),
        ("b".to_string(), "y".to_string()),
        ("b".to_string(), "z".to_string()),
        ("n".to_string(), "m".to_string()),
    ];
    assert_eq!(
        pairs(&r2),
        expected_mixed,
        "int4 key == int8 key (both i64), incl. a negative key (-3) via sign extension"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_star_projection() {
    // M5: `SELECT *` (all columns of both relations, left then right) and `SELECT alias.*` (one
    // relation) on a join.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE parent (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE child (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO parent (id, name) VALUES (1,'a'),(2,'b')")
        .unwrap();
    e.execute_text(4, "INSERT INTO child (pid, label) VALUES (1,'x'),(2,'y')")
        .unwrap();
    if e.populate_relational_residency_snapshot("parent")
        .unwrap()
        .device_memory_proof
        .is_none()
        || e.populate_relational_residency_snapshot("child")
            .unwrap()
            .device_memory_proof
            .is_none()
    {
        return;
    }
    // SELECT * -> 4 columns (id, name, pid, label), in left-then-right order.
    let star = e
        .execute_resident_expr_select_sql("SELECT * FROM parent JOIN child ON parent.id = child.pid")
        .expect("SELECT * join");
    assert_eq!(
        star.columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["id", "name", "pid", "label"],
        "bare * = all left columns then all right"
    );
    let mut srows: Vec<(i32, String, i32, String)> = star
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1], &r[2], &r[3]) {
            (SqlValue::Int4(a), SqlValue::Text(b), SqlValue::Int4(c), SqlValue::Text(d)) => {
                (*a, b.clone(), *c, d.clone())
            }
            other => panic!("unexpected row shape {other:?}"),
        })
        .collect();
    srows.sort();
    assert_eq!(
        srows,
        vec![
            (1, "a".to_string(), 1, "x".to_string()),
            (2, "b".to_string(), 2, "y".to_string()),
        ]
    );
    // SELECT parent.* -> only the left relation's columns.
    let qual = e
        .execute_resident_expr_select_sql(
            "SELECT parent.* FROM parent JOIN child ON parent.id = child.pid",
        )
        .expect("SELECT alias.* join");
    assert_eq!(
        qual.columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["id", "name"],
        "alias.* = only that relation's columns"
    );
    assert_eq!(qual.rows.len(), 2);
    // SELECT child.* -> only the RIGHT relation's columns (exercises the right-side alias.* path).
    let qual_r = e
        .execute_resident_expr_select_sql(
            "SELECT child.* FROM parent JOIN child ON parent.id = child.pid",
        )
        .expect("right alias.* join");
    assert_eq!(
        qual_r.columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["pid", "label"],
        "right alias.* = only the right relation's columns"
    );
    // Mixed: an explicit column followed by `*` -> the explicit col, then all-left, then all-right.
    let mixed = e
        .execute_resident_expr_select_sql(
            "SELECT parent.name, * FROM parent JOIN child ON parent.id = child.pid",
        )
        .expect("mixed explicit + star join");
    assert_eq!(
        mixed.columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["name", "id", "name", "pid", "label"],
        "an explicit column then bare * (left then right)"
    );
}

#[test]
fn gpu_inner_join_rejects_unsupported_shapes() {
    // Host-side clean rejections (no GPU): the parser gates the join slice's scope.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE a (k INT, x INT)").unwrap();
    e.execute_text(2, "CREATE TABLE b (k INT, y INT)").unwrap();
    let reject = |sql: &str| {
        let err = e
            .execute_resident_expr_select_sql(sql)
            .err()
            .unwrap_or_else(|| panic!("expected a rejection for: {sql}"));
        format!("{err:?}").to_lowercase()
    };
    assert!(
        reject("SELECT x, y FROM a LEFT JOIN b ON a.k = b.k").contains("inner"),
        "LEFT JOIN rejected"
    );
    assert!(
        reject("SELECT x, y FROM a JOIN b ON a.k > b.k").contains("equality")
            || reject("SELECT x, y FROM a JOIN b ON a.k > b.k").contains("equi"),
        "non-equi ON rejected"
    );
    // A comma join `FROM a, b WHERE a.k = b.k` is SUPPORTED now -- it routes to the join path (which, with
    // a/b not resident here, stops at the GPU-only residency check rather than a parser rejection).
    let comma = reject("SELECT x, y FROM a, b WHERE a.k = b.k");
    assert!(
        comma.contains("resident snapshot") || comma.contains("join path"),
        "comma join routes to the join path, got: {comma}"
    );
    // A per-relation WHERE conjunct is supported (J3); a CROSS-relation WHERE predicate (beyond the ON)
    // is a follow-up -> clean reject.
    let cross_where = reject("SELECT x, y FROM a JOIN b ON a.k = b.k WHERE a.x > b.y");
    assert!(
        cross_where.contains("cross-relation") || cross_where.contains("one relation"),
        "cross-relation WHERE rejected, got: {cross_where}"
    );
    // An UNQUALIFIED column present in BOTH relations is ambiguous (both a.k and b.k exist).
    let ambig = reject("SELECT k FROM a JOIN b ON a.k = b.k");
    assert!(
        ambig.contains("ambiguous"),
        "ambiguous unqualified column rejected, got: {ambig}"
    );
    // An ON that equates two columns of the SAME relation does not join the newly added relation.
    let same_rel = reject("SELECT x, y FROM a JOIN b ON a.k = a.x");
    assert!(
        same_rel.contains("already-joined") || same_rel.contains("ambiguous"),
        "ON within one relation rejected, got: {same_rel}"
    );
    // A qualifier naming no FROM relation is rejected.
    let bad_qual = reject("SELECT a.x, b.y FROM a JOIN b ON a.k = c.k");
    assert!(
        bad_qual.contains("missing from-clause") || bad_qual.contains("\"c\""),
        "unknown qualifier rejected, got: {bad_qual}"
    );
    // `SELECT *` / `alias.*` are supported, but a schema-qualified 3-part `s.t.*` is not.
    let three_part_star = reject("SELECT s.t.* FROM a JOIN b ON a.k = b.k");
    assert!(
        three_part_star.contains("schema-qualified") || three_part_star.contains("not supported"),
        "3-part star rejected, got: {three_part_star}"
    );
    // A `*` is invalid as an ON operand (the ON path must not accept a star).
    let star_in_on = reject("SELECT x, y FROM a JOIN b ON a.k = b.*");
    assert!(
        star_in_on.contains("`*`") || star_in_on.contains("non-name") || star_in_on.contains("name"),
        "star in ON rejected, got: {star_in_on}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_catalog_relations_transient_payload() {
    // M5 J5: a 2-relation INNER equi-join where BOTH sides are SYNTHESIZED pg_catalog relations (which
    // have NO residency snapshot). The executor uploads each as a TRANSIENT device payload + descriptor
    // and runs the SAME GPU per-side WHERE pushdown + hash join as a user-table join (no CPU relational
    // join -- charter). Construction oracle: the user tables we CREATE are EXACTLY the relkind='r' rows
    // of pg_class in the public namespace, so `pg_class JOIN pg_namespace ON n.oid = c.relnamespace`
    // filtered to public/'r' is identity over their names -> {(alpha,public),(beta,public)}.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE alpha (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE beta (id INT)").unwrap();
    // Gate on GPU availability via ANY resident snapshot (the catalog relations are not resident -- they
    // are synthesized + transiently uploaded inside the join; this just detects a usable driver/GPU).
    let snapshot = e.populate_relational_residency_snapshot("alpha").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Extract (relname, nspname) text pairs + sort (the join emit order is unspecified without ORDER BY,
    // and pg_class rows come from a HashMap).
    let names = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let s = |c: &SqlValue| match c {
                    SqlValue::Text(t) => t.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (s(&r[0]), s(&r[1]))
            })
            .collect();
        v.sort();
        v
    };
    let expected = vec![
        ("alpha".to_string(), "public".to_string()),
        ("beta".to_string(), "public".to_string()),
    ];
    // pg_class JOIN pg_namespace, with a per-side GPU text WHERE on EACH synthesized relation (c.relkind
    // on pg_class, n.nspname on pg_namespace) pushed down over its transient payload.
    let q = "SELECT c.relname, n.nspname FROM pg_catalog.pg_class c \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relkind = 'r' AND n.nspname = 'public'";
    let direct = e
        .execute_resident_expr_select_sql(q)
        .expect("catalog inner join (general path)");
    assert_eq!(direct.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(direct.columns.len(), 2);
    assert!(direct.columns[0].name.eq_ignore_ascii_case("relname"));
    assert!(direct.columns[1].name.eq_ignore_ascii_case("nspname"));
    assert_eq!(names(&direct), expected, "both user tables join to the public namespace");
    // Same query through the production wire/text dispatch (the hand-rolled parser rejects JOIN -> the
    // Err arm routes to the general path).
    let wire = e
        .execute_relational_select_text(q)
        .expect("catalog inner join (text/wire dispatch)");
    assert_eq!(wire.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(names(&wire), expected, "the wire dispatch routes the catalog JOIN to the general path");
    // Reversed FROM order (pg_namespace is now the LEFT/build side -- its `oid` key is UNIQUE, so the
    // build-on-smaller path succeeds here too); same result.
    let reversed = e
        .execute_resident_expr_select_sql(
            "SELECT c.relname, n.nspname FROM pg_catalog.pg_namespace n \
             JOIN pg_catalog.pg_class c ON n.oid = c.relnamespace \
             WHERE c.relkind = 'r' AND n.nspname = 'public'",
        )
        .expect("catalog inner join, reversed FROM order");
    assert_eq!(names(&reversed), expected, "join is symmetric in FROM order");
    // No-WHERE variant: the join key itself (relnamespace = oid) selects only the public namespace
    // (pg_catalog/information_schema oids match no relnamespace), so the result is the same WITHOUT any
    // per-side filter -- exercising the all-rows survivor path over the transient payloads.
    let no_where = e
        .execute_resident_expr_select_sql(
            "SELECT c.relname, n.nspname FROM pg_catalog.pg_class c \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        )
        .expect("catalog inner join, no WHERE");
    assert_eq!(names(&no_where), expected, "the join key alone selects the public namespace");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_three_way_user_tables() {
    // M5 J6: a 3-way LEFT-DEEP chain of INNER joins over RESIDENT user tables, pipelined by carrying
    // per-relation absolute-row indices (no intermediate materialized to a device payload). The second
    // step joins on a column from the FIRST relation joined in step 0 (cust.rid), exercising
    // accumulated-set resolution.
    //   ord JOIN cust ON cust.cid = ord.cid JOIN region ON region.rid = cust.rid
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE region (rid INT, rname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE cust (cid INT, rid INT, cname TEXT)").unwrap();
    e.execute_text(3, "CREATE TABLE ord (oid INT, cid INT, label TEXT)").unwrap();
    e.execute_text(4, "INSERT INTO region (rid, rname) VALUES (1,'west'),(2,'east')").unwrap();
    e.execute_text(5, "INSERT INTO cust (cid, rid, cname) VALUES (10,1,'alice'),(20,2,'bob')").unwrap();
    e.execute_text(6, "INSERT INTO ord (oid, cid, label) VALUES (100,10,'x'),(101,10,'y'),(102,20,'z')").unwrap();
    let rs = e.populate_relational_residency_snapshot("region").unwrap();
    let cs = e.populate_relational_residency_snapshot("cust").unwrap();
    let os = e.populate_relational_residency_snapshot("ord").unwrap();
    if rs.device_memory_proof.is_none() || cs.device_memory_proof.is_none() || os.device_memory_proof.is_none() {
        return;
    }
    let triples = |res: &RelationalSelectResult| -> Vec<(String, String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let mut v: Vec<(String, String, String)> =
            res.rows.iter().map(|r| (s(&r[0]), s(&r[1]), s(&r[2]))).collect();
        v.sort();
        v
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ord.label, cust.cname, region.rname FROM ord \
             JOIN cust ON cust.cid = ord.cid \
             JOIN region ON region.rid = cust.rid",
        )
        .expect("3-way inner join over user tables");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(res.columns.len(), 3);
    assert_eq!(
        triples(&res),
        vec![
            ("x".to_string(), "alice".to_string(), "west".to_string()),
            ("y".to_string(), "alice".to_string(), "west".to_string()),
            ("z".to_string(), "bob".to_string(), "east".to_string()),
        ],
        "each order -> its customer -> that customer's region"
    );
    // Right-nested (bushy) joins are a follow-up: the right arg of a JOIN must be a base table.
    let bushy = e
        .execute_resident_expr_select_sql(
            "SELECT ord.label FROM ord JOIN (cust JOIN region ON region.rid = cust.rid) \
             ON cust.cid = ord.cid",
        )
        .unwrap_err()
        .to_string();
    assert!(bushy.contains("base table"), "right-nested join rejected, got: {bushy}");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_three_way_catalog_describe_shape() {
    // M5 J6: the psql `\d`-family 3-way catalog chain -- pg_attribute JOIN pg_class JOIN pg_namespace --
    // entirely over SYNTHESIZED (transient-payload) relations, with a per-relation GPU WHERE on each.
    //   pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
    // Construction oracle: `people` has exactly columns (id, name); filtering c.relname='people' selects
    // them out of multiple tables -> the people columns, each tagged public.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE teams (tid INT)").unwrap();
    let snapshot = e.populate_relational_residency_snapshot("people").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT a.attname, c.relname, n.nspname \
             FROM pg_catalog.pg_attribute a \
             JOIN pg_catalog.pg_class c ON c.oid = a.attrelid \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' AND c.relname = 'people' AND a.attnum > 0",
        )
        .expect("3-way catalog \\d-shape join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let s = |c: &SqlValue| match c {
                SqlValue::Text(t) => t.clone(),
                other => panic!("expected text, got {other:?}"),
            };
            (s(&r[0]), s(&r[1]), s(&r[2]))
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("id".to_string(), "people".to_string(), "public".to_string()),
            ("name".to_string(), "people".to_string(), "public".to_string()),
        ],
        "people's columns join to its pg_class row and the public namespace; teams is filtered out"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_four_way_back_reference_to_first_relation() {
    // M5 J6: a 4-way chain whose LAST step joins BACK to relation 0 (`d.aid = a.aid`), with a 1:N fan-out
    // at relation 2 (c) BEFORE it -- so the carried `work_idx[0]` (a's rows, repeated by the fan-out) and
    // `work_idx[2]` (c's rows) DIFFER. This is the regression guard for accumulated-set resolution: a step
    // must project the accumulated key from the relation the ON names (a, index 0), NOT the immediately
    // prior relation -- the very distinction the index-vector pipeline exists for.
    //   a JOIN b ON b.aid=a.aid JOIN c ON c.bid=b.bid JOIN d ON d.aid=a.aid
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE a (aid INT, label TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE b (bid INT, aid INT)").unwrap();
    e.execute_text(3, "CREATE TABLE c (cid INT, bid INT)").unwrap();
    e.execute_text(4, "CREATE TABLE d (did INT, aid INT, dlabel TEXT)").unwrap();
    e.execute_text(5, "INSERT INTO a (aid, label) VALUES (1,'x'),(2,'y')").unwrap();
    e.execute_text(6, "INSERT INTO b (bid, aid) VALUES (10,1)").unwrap();
    e.execute_text(7, "INSERT INTO c (cid, bid) VALUES (100,10),(101,10)").unwrap();
    e.execute_text(8, "INSERT INTO d (did, aid, dlabel) VALUES (1000,1,'d1')").unwrap();
    let mut ok = true;
    for t in ["a", "b", "c", "d"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT a.label, d.dlabel FROM a \
             JOIN b ON b.aid = a.aid \
             JOIN c ON c.bid = b.bid \
             JOIN d ON d.aid = a.aid",
        )
        .expect("4-way join with a back-reference to the first relation");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let pairs = |r: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let mut v: Vec<(String, String)> =
            r.rows.iter().map(|row| (s(&row[0]), s(&row[1]))).collect();
        v.sort();
        v
    };
    // a(1) -> b(10) -> {c(100), c(101)} -> back to a(1) -> d(1000): the fan-out at c gives TWO tuples,
    // both carrying a(1)'s label 'x' and d(1000)'s 'd1'. a(2) never joins (no b row), so it is absent.
    assert_eq!(
        pairs(&res),
        vec![("x".to_string(), "d1".to_string()), ("x".to_string(), "d1".to_string())],
        "the c-fan-out duplicates the (a,d) pairing; the back-join reads a's carried rows, not c's"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_composite_on_two_column_key() {
    // M5 J6: a 2-conjunct (composite) ON `parent.pa = child.ca AND parent.pb = child.cb`. The executor
    // packs each side's two <=32-bit keys into one i64 (pa<<32|pb) for the existing hash join. The oracle
    // is constructed so a SINGLE-column join would mis-match: child(1,20) shares pa=1 with parent(1,10)
    // but must map to parent(1,20) -- proving BOTH members are compared.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE parent (pa INT, pb INT, pname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE child (ca INT, cb INT, label TEXT)").unwrap();
    // Negative members (-1,-2)/(-1,5) also exercise the pack on sign-extended values: -1's low 32 bits
    // are all-ones, so an incorrect mask/shift would corrupt the other member and cross-match.
    e.execute_text(3, "INSERT INTO parent (pa, pb, pname) VALUES (1,10,'p1'),(1,20,'p2'),(2,10,'p3'),(-1,-2,'pneg'),(-1,5,'pneg2')").unwrap();
    e.execute_text(
        4,
        "INSERT INTO child (ca, cb, label) VALUES (1,10,'x'),(1,20,'y'),(2,10,'z'),(1,99,'orphan'),(9,10,'orphan2'),(-1,-2,'neg'),(-1,5,'neg2')",
    )
    .unwrap();
    let ps = e.populate_relational_residency_snapshot("parent").unwrap();
    let cs = e.populate_relational_residency_snapshot("child").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let pairs = |r: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let mut v: Vec<(String, String)> =
            r.rows.iter().map(|row| (s(&row[0]), s(&row[1]))).collect();
        v.sort();
        v
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT child.label, parent.pname FROM child \
             JOIN parent ON parent.pa = child.ca AND parent.pb = child.cb",
        )
        .expect("composite 2-column ON join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&res),
        vec![
            ("neg".to_string(), "pneg".to_string()),
            ("neg2".to_string(), "pneg2".to_string()),
            ("x".to_string(), "p1".to_string()),
            ("y".to_string(), "p2".to_string()),
            ("z".to_string(), "p3".to_string()),
        ],
        "(ca,cb) matches (pa,pb) on BOTH columns; the pa=1/-1 rows do not cross-match; orphans dropped"
    );
    // >2 ON conjuncts (a key wider than 64 bits) is a clean follow-up error.
    let three = e
        .execute_resident_expr_select_sql(
            "SELECT child.label FROM child JOIN parent \
             ON parent.pa = child.ca AND parent.pb = child.cb AND parent.pa = child.cb",
        )
        .unwrap_err()
        .to_string();
    assert!(three.contains("more than 2"), "3-conjunct ON rejected, got: {three}");
}

#[test]
fn gpu_inner_join_composite_on_int8_member_rejected() {
    // A 2-column composite key packs two members into ONE i64, so each member must be <=32 bits. An
    // int8/timestamp composite member would overflow -> a clean reject (the `narrow_key` gate). This runs
    // BEFORE residency (the key-type precompute), so it needs no GPU. A SINGLE int8 key is still allowed.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE pp (pa BIGINT, pb INT)").unwrap();
    e.execute_text(2, "CREATE TABLE cc (ca BIGINT, cb INT)").unwrap();
    let err = e
        .execute_resident_expr_select_sql(
            "SELECT pp.pb FROM pp JOIN cc ON pp.pa = cc.ca AND pp.pb = cc.cb",
        )
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("composite") || err.contains("int2/int4/date"),
        "int8 composite member rejected, got: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_comma_join_from_where() {
    // M5 J6: a comma join `FROM ord, cust, region WHERE ...` -- the join conditions live in the WHERE and
    // are lifted into the SAME left-deep `JoinStep` pipeline as an explicit JOIN. A single-relation WHERE
    // conjunct stays a per-relation GPU filter; a cross-relation `=` becomes a join edge.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE region (rid INT, rname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE cust (cid INT, rid INT, cname TEXT)").unwrap();
    e.execute_text(3, "CREATE TABLE ord (oid INT, cid INT, label TEXT)").unwrap();
    e.execute_text(4, "INSERT INTO region (rid, rname) VALUES (1,'west'),(2,'east')").unwrap();
    e.execute_text(5, "INSERT INTO cust (cid, rid, cname) VALUES (10,1,'alice'),(20,2,'bob')").unwrap();
    e.execute_text(6, "INSERT INTO ord (oid, cid, label) VALUES (100,10,'x'),(101,10,'y'),(102,20,'z')").unwrap();
    let mut ok = true;
    for t in ["region", "cust", "ord"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let triples = |res: &RelationalSelectResult| -> Vec<(String, String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let mut v: Vec<(String, String, String)> =
            res.rows.iter().map(|r| (s(&r[0]), s(&r[1]), s(&r[2]))).collect();
        v.sort();
        v
    };
    // 3-way comma join (same result as the explicit-JOIN 3-way test).
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ord.label, cust.cname, region.rname FROM ord, cust, region \
             WHERE cust.cid = ord.cid AND region.rid = cust.rid",
        )
        .expect("3-way comma join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        triples(&res),
        vec![
            ("x".to_string(), "alice".to_string(), "west".to_string()),
            ("y".to_string(), "alice".to_string(), "west".to_string()),
            ("z".to_string(), "bob".to_string(), "east".to_string()),
        ],
        "comma join derives the same left-deep chain as the explicit JOIN"
    );
    // A per-relation filter in the WHERE alongside the join edges (label filter -> ord's GPU pre-filter).
    let filtered = e
        .execute_resident_expr_select_sql(
            "SELECT ord.label, cust.cname, region.rname FROM ord, cust, region \
             WHERE cust.cid = ord.cid AND region.rid = cust.rid AND ord.label = 'z'",
        )
        .expect("comma join with a per-relation filter");
    assert_eq!(
        triples(&filtered),
        vec![("z".to_string(), "bob".to_string(), "east".to_string())],
        "the ord.label='z' filter is pushed to ord; only that row survives"
    );
    // A relation with no join condition to an earlier one (a cartesian product) is a clean follow-up error.
    let cartesian = e
        .execute_resident_expr_select_sql("SELECT ord.label, cust.cname, region.rname FROM ord, cust, region WHERE cust.cid = ord.cid")
        .unwrap_err()
        .to_string();
    assert!(
        cartesian.contains("no equi-join condition") || cartesian.contains("cartesian"),
        "disconnected comma join rejected, got: {cartesian}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_text_key() {
    // M5 J4b: a join on a TEXT key, run by the GPU text hash join (FNV-hash + full byte-verify -- a
    // 64-bit hash collision between distinct names can never mis-join). users.name is UNIQUE (the build
    // side); logins.name is the FK (1:N + a userless 'dave' + a loginless 'carol').
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE users (uid INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE logins (name TEXT, ts INT)").unwrap();
    e.execute_text(3, "INSERT INTO users (uid, name) VALUES (1,'alice'),(2,'bob'),(3,'carol')").unwrap();
    e.execute_text(4, "INSERT INTO logins (name, ts) VALUES ('alice',100),('alice',101),('bob',200),('dave',300)").unwrap();
    let us = e.populate_relational_residency_snapshot("users").unwrap();
    let ls = e.populate_relational_residency_snapshot("logins").unwrap();
    if us.device_memory_proof.is_none() || ls.device_memory_proof.is_none() {
        return;
    }
    let int_pairs = |res: &RelationalSelectResult| -> Vec<(i32, i32)> {
        let n = |c: &SqlValue| match c {
            SqlValue::Int4(v) => *v,
            other => panic!("expected int4, got {other:?}"),
        };
        let mut v: Vec<(i32, i32)> = res.rows.iter().map(|r| (n(&r[0]), n(&r[1]))).collect();
        v.sort();
        v
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT users.uid, logins.ts FROM users JOIN logins ON users.name = logins.name",
        )
        .expect("text-key inner join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int_pairs(&res),
        vec![(1, 100), (1, 101), (2, 200)],
        "alice 1:N, bob 1:1; carol (no login) and dave (no user) dropped"
    );
    // A mixed text/int ON is not comparable -> a clean reject.
    let mixed = e
        .execute_resident_expr_select_sql(
            "SELECT users.uid FROM users JOIN logins ON users.name = logins.ts",
        )
        .unwrap_err()
        .to_string();
    assert!(
        mixed.contains("SAME type on BOTH sides") || mixed.contains("integer column"),
        "mixed text/int key rejected, got: {mixed}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_text_key_step_in_multi_way() {
    // M5 J4b: a TEXT-key step (cust.email = ord.email) followed by an INT-key step (region.rid = cust.rid)
    // in the carried-index multi-way pipeline -- the text join's matched indices feed the next step.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE region (rid INT, rname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE cust (cid INT, rid INT, email TEXT)").unwrap();
    e.execute_text(3, "CREATE TABLE ord (oid INT, email TEXT, label TEXT)").unwrap();
    e.execute_text(4, "INSERT INTO region (rid, rname) VALUES (1,'west'),(2,'east')").unwrap();
    e.execute_text(5, "INSERT INTO cust (cid, rid, email) VALUES (10,1,'a@x'),(20,2,'b@x')").unwrap();
    e.execute_text(6, "INSERT INTO ord (oid, email, label) VALUES (100,'a@x','o1'),(101,'a@x','o2'),(102,'b@x','o3')").unwrap();
    let mut ok = true;
    for t in ["region", "cust", "ord"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ord.label, region.rname FROM ord \
             JOIN cust ON cust.email = ord.email \
             JOIN region ON region.rid = cust.rid",
        )
        .expect("text step + int step multi-way join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let s = |c: &SqlValue| match c {
        SqlValue::Text(t) => t.clone(),
        other => panic!("expected text, got {other:?}"),
    };
    let mut got: Vec<(String, String)> =
        res.rows.iter().map(|r| (s(&r[0]), s(&r[1]))).collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("o1".to_string(), "west".to_string()),
            ("o2".to_string(), "west".to_string()),
            ("o3".to_string(), "east".to_string()),
        ],
        "each order -> its customer (by email) -> that customer's region (by rid)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_uuid_and_numeric_keys() {
    // M5 J4c: NUMERIC and UUID join keys reuse the J4b text/byte hash join over each value's 16-byte
    // canonical form (uuid = raw bytes; numeric = i128 mantissa, both columns the same scale).
    let mut e = Engine::new_local();
    // --- UUID key: users.gid -> groups.gid (groups.gid unique build side) ---
    e.execute_text(1, "CREATE TABLE groups (gid UUID, gname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE users (uid INT, gid UUID)").unwrap();
    let g1 = "11111111-1111-1111-1111-111111111111";
    let g2 = "22222222-2222-2222-2222-222222222222";
    let g3 = "33333333-3333-3333-3333-333333333333";
    e.execute_text(3, &format!("INSERT INTO groups (gid, gname) VALUES ('{g1}','admins'),('{g2}','members')")).unwrap();
    e.execute_text(4, &format!("INSERT INTO users (uid, gid) VALUES (1,'{g1}'),(2,'{g1}'),(3,'{g2}'),(4,'{g3}')")).unwrap();
    // --- NUMERIC key: accounts.bal -> targets.bal (same scale (10,2); targets.bal unique) ---
    e.execute_text(5, "CREATE TABLE targets (bal NUMERIC(10,2), tname TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE accounts (aid INT, bal NUMERIC(10,2))").unwrap();
    e.execute_text(7, "INSERT INTO targets (bal, tname) VALUES (100.00,'hundred'),(200.50,'two-fifty')").unwrap();
    e.execute_text(8, "INSERT INTO accounts (aid, bal) VALUES (1,100.00),(2,100.00),(3,200.50),(4,999.99)").unwrap();
    let mut ok = true;
    for t in ["groups", "users", "targets", "accounts"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let int_text = |res: &RelationalSelectResult| -> Vec<(i32, String)> {
        let mut v: Vec<(i32, String)> = res
            .rows
            .iter()
            .map(|r| {
                let n = match &r[0] {
                    SqlValue::Int4(v) => *v,
                    other => panic!("expected int4, got {other:?}"),
                };
                let s = match &r[1] {
                    SqlValue::Text(t) => t.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (n, s)
            })
            .collect();
        v.sort();
        v
    };
    let uuid_res = e
        .execute_resident_expr_select_sql(
            "SELECT users.uid, groups.gname FROM users JOIN groups ON users.gid = groups.gid",
        )
        .expect("uuid-key inner join");
    assert_eq!(uuid_res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int_text(&uuid_res),
        vec![(1, "admins".to_string()), (2, "admins".to_string()), (3, "members".to_string())],
        "uuid FK 1:N; the g3 user (no group) is dropped"
    );
    let num_res = e
        .execute_resident_expr_select_sql(
            "SELECT accounts.aid, targets.tname FROM accounts JOIN targets ON accounts.bal = targets.bal",
        )
        .expect("numeric-key inner join");
    assert_eq!(num_res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int_text(&num_res),
        vec![(1, "hundred".to_string()), (2, "hundred".to_string()), (3, "two-fifty".to_string())],
        "numeric (same scale) FK 1:N; the 999.99 account is dropped"
    );
    // Different-scale numeric on the two sides -> a clean reject (the mantissas are not comparable).
    e.execute_text(9, "CREATE TABLE precise (bal NUMERIC(10,4), pname TEXT)").unwrap();
    e.execute_text(10, "INSERT INTO precise (bal, pname) VALUES (100.0000,'p')").unwrap();
    let _ = e.populate_relational_residency_snapshot("precise");
    let scale_err = e
        .execute_resident_expr_select_sql(
            "SELECT accounts.aid FROM accounts JOIN precise ON accounts.bal = precise.bal",
        )
        .unwrap_err()
        .to_string();
    assert!(
        scale_err.contains("same scale") || scale_err.contains("SAME type"),
        "different-scale numeric join rejected, got: {scale_err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_n_to_n_cross_product() {
    // M5 N:N: BOTH sides have duplicate join keys -> the chaining many-to-many join emits each key's
    // (left rows x right rows). On k=100, left {lid 1,2} x right {rid 10,11} = 4 pairs; k=200 (left-only)
    // and k=300 (right-only) drop. (Previously a hard "N:N is a follow-up" reject.)
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE lhs (lid INT, k INT)").unwrap();
    e.execute_text(2, "CREATE TABLE rhs (rid INT, k INT)").unwrap();
    e.execute_text(3, "INSERT INTO lhs (lid, k) VALUES (1,100),(2,100),(3,200)").unwrap();
    e.execute_text(4, "INSERT INTO rhs (rid, k) VALUES (10,100),(11,100),(12,300)").unwrap();
    let ls = e.populate_relational_residency_snapshot("lhs").unwrap();
    let rs = e.populate_relational_residency_snapshot("rhs").unwrap();
    if ls.device_memory_proof.is_none() || rs.device_memory_proof.is_none() {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(i32, i32)> {
        let n = |c: &SqlValue| match c {
            SqlValue::Int4(v) => *v,
            other => panic!("expected int4, got {other:?}"),
        };
        let mut v: Vec<(i32, i32)> = res.rows.iter().map(|r| (n(&r[0]), n(&r[1]))).collect();
        v.sort();
        v
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT lhs.lid, rhs.rid FROM lhs JOIN rhs ON lhs.k = rhs.k",
        )
        .expect("N:N many-to-many join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&res),
        vec![(1, 10), (1, 11), (2, 10), (2, 11)],
        "k=100 yields the full 2x2 cross product; the left-only/right-only keys drop"
    );
    // N:N composes with a per-side WHERE (still GPU-pre-filtered): restrict to lid <= 1 -> only lid 1.
    let filtered = e
        .execute_resident_expr_select_sql(
            "SELECT lhs.lid, rhs.rid FROM lhs JOIN rhs ON lhs.k = rhs.k WHERE lhs.lid <= 1",
        )
        .expect("N:N join with a per-side filter");
    assert_eq!(
        pairs(&filtered),
        vec![(1, 10), (1, 11)],
        "the lid<=1 filter leaves one left row -> a 1x2 cross product"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_n_to_n_text_and_numeric_keys() {
    // M5: N:N many-to-many over NON-int keys (text + numeric/uuid reuse the chaining text/byte kernel).
    // Both sides duplicate the key -> each key's (left x right) cross product.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE lt (lid INT, tag TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE rt (rid INT, tag TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO lt (lid, tag) VALUES (1,'x'),(2,'x'),(3,'y')").unwrap();
    e.execute_text(4, "INSERT INTO rt (rid, tag) VALUES (10,'x'),(11,'x'),(12,'z')").unwrap();
    e.execute_text(5, "CREATE TABLE la (laid INT, amt NUMERIC(10,2))").unwrap();
    e.execute_text(6, "CREATE TABLE ra (raid INT, amt NUMERIC(10,2))").unwrap();
    e.execute_text(7, "INSERT INTO la (laid, amt) VALUES (1,5.00),(2,5.00),(3,9.00)").unwrap();
    e.execute_text(8, "INSERT INTO ra (raid, amt) VALUES (10,5.00),(11,5.00),(12,1.00)").unwrap();
    let mut ok = true;
    for t in ["lt", "rt", "la", "ra"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(i32, i32)> {
        let n = |c: &SqlValue| match c {
            SqlValue::Int4(v) => *v,
            other => panic!("expected int4, got {other:?}"),
        };
        let mut v: Vec<(i32, i32)> = res.rows.iter().map(|r| (n(&r[0]), n(&r[1]))).collect();
        v.sort();
        v
    };
    // text N:N: tag 'x' -> lt {1,2} x rt {10,11} = 4 pairs; 'y'/'z' drop.
    let text_nn = e
        .execute_resident_expr_select_sql("SELECT lt.lid, rt.rid FROM lt JOIN rt ON lt.tag = rt.tag")
        .expect("text N:N join");
    assert_eq!(text_nn.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&text_nn),
        vec![(1, 10), (1, 11), (2, 10), (2, 11)],
        "text key 'x' yields the 2x2 cross product"
    );
    // numeric N:N: amt 5.00 -> la {1,2} x ra {10,11} = 4 pairs (reuses the same chaining kernel).
    let num_nn = e
        .execute_resident_expr_select_sql("SELECT la.laid, ra.raid FROM la JOIN ra ON la.amt = ra.amt")
        .expect("numeric N:N join");
    assert_eq!(
        pairs(&num_nn),
        vec![(1, 10), (1, 11), (2, 10), (2, 11)],
        "numeric key 5.00 yields the 2x2 cross product"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_order_by_limit_offset() {
    // M5 (catalog \d prerequisite): ORDER BY / LIMIT / OFFSET on a join. ORDER BY is a GPU sort over the
    // join result (int key via the matrix path, text+int multi-key via the hetero payload path).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, lid INT, score INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'charlie'),(2,'alice'),(3,'bob')").unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid, lid, score) VALUES (10,1,50),(11,2,90),(12,3,70),(13,1,30)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // Join is {charlie:50, charlie:30, alice:90, bob:70} over (name, score).
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, i32)> {
        res.rows
            .iter()
            .map(|r| {
                let name = match &r[0] {
                    SqlValue::Text(s) => s.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                let score = match &r[1] {
                    SqlValue::Int4(v) => *v,
                    other => panic!("expected int4, got {other:?}"),
                };
                (name, score)
            })
            .collect()
    };
    // (a) ORDER BY an INT key, DESC -> the matrix sort path; full descending order.
    let desc = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.lid ORDER BY r.score DESC",
        )
        .expect("order by int desc");
    assert_eq!(desc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&desc),
        vec![
            ("alice".to_string(), 90),
            ("bob".to_string(), 70),
            ("charlie".to_string(), 50),
            ("charlie".to_string(), 30),
        ],
        "ORDER BY r.score DESC sorts the join result on the GPU"
    );
    // (b) ORDER BY INT ASC + LIMIT 2 -> the two smallest scores, in order.
    let asc_lim = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.lid ORDER BY r.score ASC LIMIT 2",
        )
        .expect("order by asc + limit");
    assert_eq!(
        pairs(&asc_lim),
        vec![("charlie".to_string(), 30), ("charlie".to_string(), 50)],
        "ORDER BY ASC LIMIT 2 keeps the two smallest after the GPU sort"
    );
    // (c) Multi-key ORDER BY (TEXT ASC, INT DESC) -> the hetero payload sort path.
    let multi = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.lid ORDER BY l.name ASC, r.score DESC",
        )
        .expect("multi-key order by");
    assert_eq!(
        pairs(&multi),
        vec![
            ("alice".to_string(), 90),
            ("bob".to_string(), 70),
            ("charlie".to_string(), 50),
            ("charlie".to_string(), 30),
        ],
        "ORDER BY name ASC, score DESC (text+int multi-key) sorts on the GPU"
    );
    // (d) OFFSET + LIMIT after the sort -> the middle window.
    let window = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.lid ORDER BY r.score DESC OFFSET 1 LIMIT 2",
        )
        .expect("offset + limit");
    assert_eq!(
        pairs(&window),
        vec![("bob".to_string(), 70), ("charlie".to_string(), 50)],
        "OFFSET 1 LIMIT 2 slices the sorted result"
    );
    // A non-projected ORDER BY key is a follow-up -> a clear error, not a wrong/partial answer.
    let err = e
        .execute_resident_expr_select_sql(
            "SELECT l.name FROM l JOIN r ON l.id = r.lid ORDER BY r.score DESC",
        )
        .expect_err("non-projected ORDER BY key must be rejected, not silently dropped");
    assert!(
        format!("{err:?}").contains("must appear in the SELECT list"),
        "got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_inner_join_using_and_natural() {
    // M5: USING / NATURAL joins -- the join column is COALESCED (appears once in `*`, PG order: join cols,
    // then left's rest, then right's; an unqualified ref resolves to the left copy).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE emp (eid INT, dept INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE dept (dept INT, dname TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO emp (eid, dept, name) VALUES (1,10,'alice'),(2,10,'bob'),(3,20,'carol')").unwrap();
    e.execute_text(4, "INSERT INTO dept (dept, dname) VALUES (10,'eng'),(20,'sales'),(30,'hr')").unwrap();
    let es = e.populate_relational_residency_snapshot("emp").unwrap();
    let ds = e.populate_relational_residency_snapshot("dept").unwrap();
    if es.device_memory_proof.is_none() || ds.device_memory_proof.is_none() {
        return;
    }
    // bare `*`: coalesced `dept` once, then emp's other cols (eid, name), then dept's other (dname).
    let rows4 = |res: &RelationalSelectResult| -> Vec<(i32, i32, String, String)> {
        let i = |c: &SqlValue| match c {
            SqlValue::Int4(v) => *v,
            other => panic!("expected int4, got {other:?}"),
        };
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let mut v: Vec<(i32, i32, String, String)> =
            res.rows.iter().map(|r| (i(&r[0]), i(&r[1]), s(&r[2]), s(&r[3]))).collect();
        v.sort();
        v
    };
    let star = e
        .execute_resident_expr_select_sql("SELECT * FROM emp JOIN dept USING (dept)")
        .expect("USING join with `*`");
    assert_eq!(star.executed_target, DeviceTarget::Gpu(0));
    let names: Vec<String> = star.columns.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["dept", "eid", "name", "dname"], "USING coalesces `dept` once, PG `*` order");
    assert_eq!(
        rows4(&star),
        vec![
            (10, 1, "alice".to_string(), "eng".to_string()),
            (10, 2, "bob".to_string(), "eng".to_string()),
            (20, 3, "carol".to_string(), "sales".to_string()),
        ],
        "USING(dept) joins emp.dept=dept.dept; dept 30 (no emp) dropped"
    );
    // NATURAL JOIN derives USING(common columns) = USING(dept) here -> identical result.
    let nat = e
        .execute_resident_expr_select_sql("SELECT * FROM emp NATURAL JOIN dept")
        .expect("NATURAL join");
    assert_eq!(
        nat.columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        vec!["dept", "eid", "name", "dname"],
        "NATURAL joins on (and coalesces) the common column `dept`"
    );
    assert_eq!(rows4(&nat), rows4(&star), "NATURAL == USING(dept) here");
    // An UNQUALIFIED reference to the coalesced join column resolves (not ambiguous).
    let explicit = e
        .execute_resident_expr_select_sql("SELECT name, dept, dname FROM emp JOIN dept USING (dept)")
        .expect("USING join, unqualified coalesced column");
    let mut got: Vec<(String, i32, String)> = explicit
        .rows
        .iter()
        .map(|r| {
            let s = |c: &SqlValue| match c {
                SqlValue::Text(t) => t.clone(),
                other => panic!("text expected, got {other:?}"),
            };
            let n = match &r[1] {
                SqlValue::Int4(v) => *v,
                other => panic!("int expected, got {other:?}"),
            };
            (s(&r[0]), n, s(&r[2]))
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("alice".to_string(), 10, "eng".to_string()),
            ("bob".to_string(), 10, "eng".to_string()),
            ("carol".to_string(), 20, "sales".to_string()),
        ],
        "unqualified `dept` is the coalesced column"
    );
    // NATURAL/USING in a MULTI-WAY join is a clean follow-up reject.
    e.execute_text(5, "CREATE TABLE loc (dept INT, city TEXT)").unwrap();
    let multi = e
        .execute_resident_expr_select_sql(
            "SELECT * FROM emp JOIN dept USING (dept) JOIN loc ON loc.dept = emp.dept",
        )
        .unwrap_err()
        .to_string();
    assert!(multi.contains("multi-way"), "multi-way USING rejected, got: {multi}");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_order_by_text_key_sorts_on_the_gpu() {
    // GROUP BY a TEXT column, ORDER BY that text key: the grouped GPU sort builds a resident-like TEXT
    // payload (offsets + bytes) from the host result + sorts on-device -- the trickiest payload path.
    // Multi-aggregate forces the general path.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (name TEXT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (name, v) VALUES ('cara', 1), ('amy', 2), ('bob', 3), ('amy', 4)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text(
            "SELECT name, COUNT(*), SUM(v) FROM t GROUP BY name ORDER BY name DESC",
        )
        .expect("grouped text-key ORDER BY now sorts on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    let names: Vec<String> = s
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(t) => t.to_string(),
            other => panic!("expected text, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["cara", "bob", "amy"], "name DESC -> cara, bob, amy");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_order_by_numeric_key_sorts_on_the_gpu() {
    // GROUP BY a NUMERIC column, ORDER BY it DESC: the grouped GPU sort builds a resident-like 16-byte
    // (b128) payload section + sorts on-device. Assert via the per-group COUNT (distinct) to avoid a
    // numeric-literal compare. Multi-aggregate forces the general path.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (v NUMERIC(10,2), w INT)")
        .unwrap();
    // counts by v: 1.00->2, 2.00->3, 3.00->1. ORDER BY v DESC -> 3.00(1), 2.00(3), 1.00(2).
    e.execute_text(
        2,
        "INSERT INTO t (v, w) VALUES (1.00,1),(1.00,1),(2.00,1),(2.00,1),(2.00,1),(3.00,1)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text(
            "SELECT v, COUNT(*), SUM(w) FROM t GROUP BY v ORDER BY v DESC",
        )
        .expect("grouped numeric-key ORDER BY now sorts on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    let counts: Vec<i64> = s
        .rows
        .iter()
        .map(|r| match &r[1] {
            SqlValue::Int8(c) => *c,
            SqlValue::Int4(c) => i64::from(*c),
            other => panic!("unexpected count type: {other:?}"),
        })
        .collect();
    assert_eq!(
        counts,
        vec![1, 3, 2],
        "v DESC: 3.00(count 1), 2.00(count 3), 1.00(count 2)"
    );
}

#[test]
fn grouped_order_by_expression_is_rejected() {
    // ORDER BY an EXPRESSION on a GROUPED result has no GPU sort yet (the grouped-sort migration lands
    // that) -> a clean reject, not a silent host first-key sort. Multi-aggregate forces the general path.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE g (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO g (a) VALUES (1), (1), (2)")
        .unwrap();
    let err = e
        .execute_relational_select_text("SELECT a, COUNT(*), SUM(a) FROM g GROUP BY a ORDER BY a + a")
        .expect_err("grouped ORDER BY expression must be rejected");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("expression") && msg.contains("grouped"),
        "expected a grouped-expression rejection, got: {err:?}"
    );
}

#[test]
fn select_text_multikey_order_by_non_resident_rejects() {
    // Multi-key ORDER BY is GPU-only: it sorts on the general Expr executor's bitonic-sort path, routed
    // ONLY when every key is an i64-sortable base column on a GPU-RESIDENT table. On a non-resident
    // table the routing gate falls through to the enumerated/CPU path, which has NO multi-key sort and
    // must reject it cleanly (never silently sort by the first key). No GPU needed -- the rejection is
    // on the CPU choke point. Charter: no CPU relational multi-key sort.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (1, 9), (1, 3), (2, 5)")
        .unwrap();
    // Deliberately NOT resident -> the multi-key sort cannot take the GPU path.
    match e.execute_relational_select_text("SELECT a, b FROM t ORDER BY a ASC, b DESC") {
        Ok(_) => panic!("multi-key ORDER BY on a non-resident table must error, not return rows"),
        Err(err) => assert!(
            err.to_string().contains("multi-key ORDER BY"),
            "expected a clean multi-key rejection, got: {err}"
        ),
    }
    // A SINGLE-key ORDER BY on the same non-resident table still works (unchanged) via the CPU path.
    let single = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a DESC")
        .expect("single-key ORDER BY still sorts via the existing path");
    assert_eq!(
        single.rows,
        vec![
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(1)],
        ],
        "single-key ORDER BY unchanged"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_select_text_routes_arithmetic_predicate_to_general_expr_path() {
    // The TEXT dispatch (execute_relational_select_text — what a consolidated server calls) routes an
    // arithmetic-WHERE SELECT, which the hand-rolled parser cannot express, to the general GPU Expr
    // executor, end to end. Same closed-form oracle as the direct entry; this proves the ROUTING hook
    // (not just the standalone SQL->Expr method).
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

    let result = e
        .execute_relational_select_text("SELECT a FROM t WHERE a + b > 400")
        .expect("text dispatch routes the arithmetic WHERE to the general GPU path");
    let expected: Vec<Vec<SqlValue>> = (201..N).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        result.rows, expected,
        "routed `a + b > 400` must materialize a-values for i in [201, 600) on the GPU"
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}
