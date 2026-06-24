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
fn execute_resident_expr_select_sql_maps_is_null_predicate_and_reaches_gpu_dispatch() {
    // `WHERE v IS NULL` / `IS NOT NULL` (a libpg_query NullTest node) maps to `ResidentExpr::IsNull` and
    // binds, reaching the GPU residency stage (PAST parse/map/bind) -- not a mapper "unsupported node"
    // rejection. With no residency populated it stops at the residency error, which is deterministic on
    // any box. (The GPU e2e validity-bitmap test in resident_expr.rs runs it through end to end.)
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (id INT, v INT)").unwrap();
    for sql in [
        "SELECT id FROM t WHERE v IS NULL",
        "SELECT id FROM t WHERE v IS NOT NULL",
    ] {
        match e.execute_resident_expr_select_sql(sql) {
            Ok(_) => panic!("no residency snapshot populated, so `{sql}` cannot return rows"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("resident"),
                    "`{sql}` must map (NullTest -> IsNull) and reach the residency stage, not a mapper \
                     rejection; got: {msg}"
                );
            }
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
fn gpu_inner_join_excludes_null_keys_three_valued_logic() {
    // M3 NULL-key gate (doc 21): in an equi-join `NULL = x` is UNKNOWN, so a row whose join key is NULL
    // matches NOTHING -- on BOTH sides. INT keys would otherwise share the 0 placeholder and SPURIOUSLY
    // match each other; TEXT/b128 keys would otherwise ERROR in the key gather. Both NULL-key rows drop.
    let mut e = Engine::new_local();
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

    // --- INT key: without the gate, the two NULL ids share the 0 placeholder and spuriously pair. ---
    e.execute_text(1, "CREATE TABLE p (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE c (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO p (id, name) VALUES (1,'a'),(2,'b'),(NULL,'pnull')")
        .unwrap();
    e.execute_text(4, "INSERT INTO c (pid, label) VALUES (1,'x'),(2,'z'),(NULL,'cnull')")
        .unwrap();
    let ps = e.populate_relational_residency_snapshot("p").unwrap();
    let cs = e.populate_relational_residency_snapshot("c").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let int_join = e
        .execute_resident_expr_select_sql("SELECT name, label FROM p JOIN c ON p.id = c.pid")
        .expect("int join with NULL keys");
    assert_eq!(int_join.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        pairs(&int_join),
        vec![
            ("a".to_string(), "x".to_string()),
            ("b".to_string(), "z".to_string())
        ],
        "NULL int keys match nothing -- no spurious ('pnull','cnull') pair"
    );

    // --- TEXT key: without the gate, the key-bytes gather would ERROR on a NULL cell. ---
    e.execute_text(5, "CREATE TABLE s (k TEXT, x TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE u (k TEXT, y TEXT)").unwrap();
    e.execute_text(7, "INSERT INTO s (k, x) VALUES ('m','sm'),(NULL,'snull')")
        .unwrap();
    e.execute_text(8, "INSERT INTO u (k, y) VALUES ('m','tm'),(NULL,'tnull')")
        .unwrap();
    let ss = e.populate_relational_residency_snapshot("s").unwrap();
    let us = e.populate_relational_residency_snapshot("u").unwrap();
    if ss.device_memory_proof.is_none() || us.device_memory_proof.is_none() {
        return;
    }
    let text_join = e
        .execute_resident_expr_select_sql("SELECT x, y FROM s JOIN u ON s.k = u.k")
        .expect("text join with NULL keys must not error");
    assert_eq!(
        pairs(&text_join),
        vec![("sm".to_string(), "tm".to_string())],
        "NULL text keys match nothing (and the gather does not error)"
    );

    // --- COMPOSITE int key (a.k1=b.k1 AND a.k2=b.k2): a row with ANY member NULL is excluded (else the
    // NULL member's 0 placeholder would pack to the same composite key and spuriously match). ---
    e.execute_text(9, "CREATE TABLE ca (k1 INT, k2 INT, x TEXT)")
        .unwrap();
    e.execute_text(10, "CREATE TABLE cb (k1 INT, k2 INT, y TEXT)")
        .unwrap();
    e.execute_text(11, "INSERT INTO ca (k1, k2, x) VALUES (1,1,'ca1'),(2,NULL,'canull')")
        .unwrap();
    e.execute_text(12, "INSERT INTO cb (k1, k2, y) VALUES (1,1,'cb1'),(2,NULL,'cbnull')")
        .unwrap();
    let cas = e.populate_relational_residency_snapshot("ca").unwrap();
    let cbs = e.populate_relational_residency_snapshot("cb").unwrap();
    if cas.device_memory_proof.is_none() || cbs.device_memory_proof.is_none() {
        return;
    }
    let comp_join = e
        .execute_resident_expr_select_sql(
            "SELECT x, y FROM ca JOIN cb ON ca.k1 = cb.k1 AND ca.k2 = cb.k2",
        )
        .expect("composite int join with a NULL member");
    assert_eq!(
        pairs(&comp_join),
        vec![("ca1".to_string(), "cb1".to_string())],
        "a NULL composite-key member excludes the row -- no ('canull','cbnull')"
    );

    // --- NUMERIC (b128) key: a NULL numeric key is excluded (and the 16-byte gather does not error). ---
    e.execute_text(13, "CREATE TABLE na (k NUMERIC(10,2), x TEXT)")
        .unwrap();
    e.execute_text(14, "CREATE TABLE nb (k NUMERIC(10,2), y TEXT)")
        .unwrap();
    e.execute_text(15, "INSERT INTO na (k, x) VALUES (1.50,'na1'),(NULL,'nanull')")
        .unwrap();
    e.execute_text(16, "INSERT INTO nb (k, y) VALUES (1.50,'nb1'),(NULL,'nbnull')")
        .unwrap();
    let nas = e.populate_relational_residency_snapshot("na").unwrap();
    let nbs = e.populate_relational_residency_snapshot("nb").unwrap();
    if nas.device_memory_proof.is_none() || nbs.device_memory_proof.is_none() {
        return;
    }
    let num_join = e
        .execute_resident_expr_select_sql("SELECT x, y FROM na JOIN nb ON na.k = nb.k")
        .expect("numeric join with NULL keys must not error");
    assert_eq!(
        pairs(&num_join),
        vec![("na1".to_string(), "nb1".to_string())],
        "NULL numeric keys match nothing (the b128 gather does not error)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_at_scale_grid_stride_v1b() {
    // V1b WIRE: the NULL-key skip is now ON-DEVICE (the hash-join kernel reads a per-side validity bitmap),
    // not a host pre-filter. At SCALE (>256 rows => multiple 256-thread blocks, grid-stride) a NULL key on
    // either side must (a) match nothing and (b) NOT spuriously match a REAL key 0 -- every NULL fixed-width
    // cell stores a 0 placeholder on-device, so the on-device skip is the ONLY thing preventing the NULL rows
    // from colliding with the real key 0 (which DOES exist here). bigp.id is unique on its non-NULL rows, so
    // the unique-build kernel runs; if the skip were broken, a NULL build row (placeholder 0) would collide
    // with real key 0 -> a spurious DuplicateBuildKey N:N fallback + wrong pairs, caught by the exact set.
    let mut e = Engine::new_local();
    const N: i32 = 600;
    let mut pv = String::new();
    let mut cv = String::new();
    for i in 0..N {
        if i > 0 {
            pv.push(',');
            cv.push(',');
        }
        // NULL conditions chosen so id 0 stays a REAL key on both sides.
        if i % 13 == 5 {
            pv.push_str(&format!("(NULL,'p{i}')"));
        } else {
            pv.push_str(&format!("({i},'p{i}')"));
        }
        if i % 17 == 3 {
            cv.push_str(&format!("(NULL,'c{i}')"));
        } else {
            cv.push_str(&format!("({i},'c{i}')"));
        }
    }
    e.execute_text(1, "CREATE TABLE bigp (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE bigc (pid INT, label TEXT)").unwrap();
    e.execute_text(3, &format!("INSERT INTO bigp (id, name) VALUES {pv}")).unwrap();
    e.execute_text(4, &format!("INSERT INTO bigc (pid, label) VALUES {cv}")).unwrap();
    let ps = e.populate_relational_residency_snapshot("bigp").unwrap();
    let cs = e.populate_relational_residency_snapshot("bigc").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT name, label FROM bigp JOIN bigc ON bigp.id = bigc.pid")
        .expect("scale int join with NULL keys");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let s = |c: &SqlValue| match c {
                SqlValue::Text(t) => t.clone(),
                o => panic!("expected text, got {o:?}"),
            };
            (s(&r[0]), s(&r[1]))
        })
        .collect();
    got.sort();
    let mut expected: Vec<(String, String)> = (0..N)
        .filter(|&i| i % 13 != 5 && i % 17 != 3)
        .map(|i| (format!("p{i}"), format!("c{i}")))
        .collect();
    expected.sort();
    assert_eq!(
        got, expected,
        "every non-NULL key matches 1:1; NULL rows (incl. the placeholder-0 rows) match nothing"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_int2_int8_uuid_v1b() {
    // V1b: the on-device NULL-key skip works for the int2 + int8 (i64 section) widths and the uuid (b128)
    // type (int4/text/numeric/composite are covered by gpu_inner_join_excludes_null_keys_*). int8 keys
    // exercise the i64-section gather; key 0 is a real key on both sides, so the NULL row's 0 placeholder
    // (a NULL int8 cell stores 0, NOT i64::MIN, so the launcher's i64::MIN reject is not tripped) must not
    // collide with it.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE i2a (k INT2, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE i2b (k INT2, y TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO i2a (k,x) VALUES (0,'a0'),(7,'a7'),(NULL,'anull')").unwrap();
    e.execute_text(4, "INSERT INTO i2b (k,y) VALUES (0,'b0'),(7,'b7'),(NULL,'bnull')").unwrap();
    e.execute_text(5, "CREATE TABLE i8a (k INT8, x TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE i8b (k INT8, y TEXT)").unwrap();
    e.execute_text(7, "INSERT INTO i8a (k,x) VALUES (0,'a0'),(9000000000,'abig'),(NULL,'anull')").unwrap();
    e.execute_text(8, "INSERT INTO i8b (k,y) VALUES (0,'b0'),(9000000000,'bbig'),(NULL,'bnull')").unwrap();
    let u1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    e.execute_text(9, "CREATE TABLE uxa (k UUID, x TEXT)").unwrap();
    e.execute_text(10, "CREATE TABLE uxb (k UUID, y TEXT)").unwrap();
    e.execute_text(11, &format!("INSERT INTO uxa (k,x) VALUES ('{u1}','a1'),(NULL,'anull')")).unwrap();
    e.execute_text(12, &format!("INSERT INTO uxb (k,y) VALUES ('{u1}','b1'),(NULL,'bnull')")).unwrap();
    let mut ok = true;
    for t in ["i2a", "i2b", "i8a", "i8b", "uxa", "uxb"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            o => panic!("expected text, got {o:?}"),
        };
        let mut v: Vec<(String, String)> = res.rows.iter().map(|r| (s(&r[0]), s(&r[1]))).collect();
        v.sort();
        v
    };
    let i2 = e
        .execute_resident_expr_select_sql("SELECT x, y FROM i2a JOIN i2b ON i2a.k = i2b.k")
        .expect("int2 NULL-key join");
    assert_eq!(
        pairs(&i2),
        vec![("a0".to_string(), "b0".to_string()), ("a7".to_string(), "b7".to_string())],
        "int2 NULL key skipped; key 0 still matches"
    );
    let i8 = e
        .execute_resident_expr_select_sql("SELECT x, y FROM i8a JOIN i8b ON i8a.k = i8b.k")
        .expect("int8 NULL-key join");
    assert_eq!(
        pairs(&i8),
        vec![("a0".to_string(), "b0".to_string()), ("abig".to_string(), "bbig".to_string())],
        "int8 NULL key skipped; real key 0 + the big key match (placeholder 0 != i64::MIN)"
    );
    let ux = e
        .execute_resident_expr_select_sql("SELECT x, y FROM uxa JOIN uxb ON uxa.k = uxb.k")
        .expect("uuid NULL-key join");
    assert_eq!(
        pairs(&ux),
        vec![("a1".to_string(), "b1".to_string())],
        "uuid NULL key skipped on the b128 path"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_n_to_n_int_and_text_v1b() {
    // V1b: the N:N (many-to-many chaining) kernels skip a NULL key too -- a NULL BUILD key is never chained,
    // a NULL PROBE key emits nothing. Both sides carry DUPLICATE keys (forcing the N:N fallback) PLUS NULLs;
    // key 0 (int) / 'm' (text) is a real DUPLICATED key, so a broken skip would chain the placeholder-0 /
    // empty-string NULL rows into that key's cross-product and emit spurious pairs.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE nna (k INT, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE nnb (k INT, y TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO nna (k,x) VALUES (0,'a0a'),(0,'a0b'),(1,'a1'),(NULL,'anull')").unwrap();
    e.execute_text(4, "INSERT INTO nnb (k,y) VALUES (0,'b0a'),(0,'b0b'),(1,'b1'),(NULL,'bnull')").unwrap();
    e.execute_text(5, "CREATE TABLE tta (k TEXT, x TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE ttb (k TEXT, y TEXT)").unwrap();
    e.execute_text(7, "INSERT INTO tta (k,x) VALUES ('m','a1'),('m','a2'),(NULL,'anull')").unwrap();
    e.execute_text(8, "INSERT INTO ttb (k,y) VALUES ('m','b1'),('m','b2'),(NULL,'bnull')").unwrap();
    let mut ok = true;
    for t in ["nna", "nnb", "tta", "ttb"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let pairs = |res: &RelationalSelectResult| -> Vec<(String, String)> {
        let s = |c: &SqlValue| match c {
            SqlValue::Text(t) => t.clone(),
            o => panic!("expected text, got {o:?}"),
        };
        let mut v: Vec<(String, String)> = res.rows.iter().map(|r| (s(&r[0]), s(&r[1]))).collect();
        v.sort();
        v
    };
    let int_nn = e
        .execute_resident_expr_select_sql("SELECT x, y FROM nna JOIN nnb ON nna.k = nnb.k")
        .expect("int N:N NULL-key join");
    assert_eq!(
        pairs(&int_nn),
        vec![
            ("a0a".to_string(), "b0a".to_string()),
            ("a0a".to_string(), "b0b".to_string()),
            ("a0b".to_string(), "b0a".to_string()),
            ("a0b".to_string(), "b0b".to_string()),
            ("a1".to_string(), "b1".to_string()),
        ],
        "int N:N: key 0's 2x2 cross product + key 1's 1x1; NULL rows never chain/emit"
    );
    let text_nn = e
        .execute_resident_expr_select_sql("SELECT x, y FROM tta JOIN ttb ON tta.k = ttb.k")
        .expect("text N:N NULL-key join");
    assert_eq!(
        pairs(&text_nn),
        vec![
            ("a1".to_string(), "b1".to_string()),
            ("a1".to_string(), "b2".to_string()),
            ("a2".to_string(), "b1".to_string()),
            ("a2".to_string(), "b2".to_string()),
        ],
        "text N:N: 'm's 2x2 cross product; NULL rows never chain/emit"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_null_keys_right_and_full_outer_padded_v1b() {
    // V1b: a NULL-key row in an OUTER join matches nothing (skipped on-device) and so must be PADDED on the
    // outer side, never dropped or spuriously matched against the OTHER side's NULL-key row. RIGHT keeps
    // every right row (its NULL-key row left-padded; the left NULL-key row is left-only -> dropped); FULL
    // keeps both sides' unmatched rows (incl. both NULL-key rows).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE ol (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE orr (rid INT, label TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO ol (id, name) VALUES (1,'a'),(NULL,'lnull')").unwrap();
    e.execute_text(4, "INSERT INTO orr (rid, label) VALUES (1,'x'),(NULL,'rnull')").unwrap();
    let ls = e.populate_relational_residency_snapshot("ol").unwrap();
    let rs = e.populate_relational_residency_snapshot("orr").unwrap();
    if ls.device_memory_proof.is_none() || rs.device_memory_proof.is_none() {
        return;
    }
    let opt_pairs = |res: &RelationalSelectResult| -> Vec<(Option<String>, Option<String>)> {
        let opt = |c: &SqlValue| match c {
            SqlValue::Text(t) => Some(t.clone()),
            SqlValue::Null => None,
            o => panic!("expected text/null, got {o:?}"),
        };
        let mut v: Vec<(Option<String>, Option<String>)> =
            res.rows.iter().map(|r| (opt(&r[0]), opt(&r[1]))).collect();
        v.sort();
        v
    };
    let right = e
        .execute_resident_expr_select_sql("SELECT name, label FROM ol RIGHT JOIN orr ON ol.id = orr.rid")
        .expect("right outer join with NULL keys");
    assert_eq!(right.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        opt_pairs(&right),
        vec![
            (None, Some("rnull".to_string())), // right NULL-key row -> left padded (NOT matched to 'lnull')
            (Some("a".to_string()), Some("x".to_string())),
        ],
        "RIGHT: the right NULL-key row is left-padded; the left NULL-key row is dropped (left-only)"
    );
    let full = e
        .execute_resident_expr_select_sql("SELECT name, label FROM ol FULL JOIN orr ON ol.id = orr.rid")
        .expect("full outer join with NULL keys");
    assert_eq!(
        opt_pairs(&full),
        vec![
            (None, Some("rnull".to_string())),              // right-only NULL-key row
            (Some("a".to_string()), Some("x".to_string())), // the lone match
            (Some("lnull".to_string()), None),              // left-only NULL-key row
        ],
        "FULL: both NULL-key rows are kept and padded; they do NOT match each other"
    );
}

// ── V1b WIRE adversarial regression net (adopted from the independent audit of `5724bf55`) ───────────
// Each was fault-injection-proven non-vacuous by the auditor: with the on-device skip disabled (bitmaps
// forced to None), each FAILS with the exact spurious match noted. They isolate cases the author's 4 v1b
// tests don't: build-side swap, an all-NULL build column, composite member-AND, the anti-join silent
// data-loss, and the 32-bit bitmap word boundary.

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_all_null_build_column_empty_effective_build() {
    // An ENTIRELY-NULL build key column: every build key is skipped on-device => the effective build is
    // empty => nothing matches, EVEN against a real probe key 0. The build's placeholder index 0 is itself
    // a NULL row. Skip-disabled: the two NULL build rows (placeholder key 0) collide => DuplicateBuildKey =>
    // N:N => both spuriously match the probe's real key 0.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE ba (k INT, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE bb (k INT, y TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO ba (k,x) VALUES (NULL,'a0'),(NULL,'a1')").unwrap();
    e.execute_text(4, "INSERT INTO bb (k,y) VALUES (0,'b0'),(5,'b5')").unwrap();
    let bas = e.populate_relational_residency_snapshot("ba").unwrap();
    let bbs = e.populate_relational_residency_snapshot("bb").unwrap();
    if bas.device_memory_proof.is_none() || bbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM ba JOIN bb ON ba.k = bb.k")
        .expect("all-NULL build column join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert!(
        res.rows.is_empty(),
        "an all-NULL build key column matches nothing -- not even the probe's real key 0; got {:?}",
        res.rows
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_null_on_build_side_when_smaller_side_swaps() {
    // The bitmaps must SWAP with the keys when the smaller side becomes the build. Here the NEW (right) side
    // is smaller, so the hash join builds on it (build/probe swap); the right bitmap must follow to the build
    // slot. NULLs on BOTH sides. Skip-disabled or a mis-swapped bitmap => a NULL row leaks into the result.
    let mut e = Engine::new_local();
    // acc (left) = 3 rows incl a NULL; new (right) = 2 rows incl a NULL => smaller_is_left = false => the
    // build swaps to the right side.
    e.execute_text(1, "CREATE TABLE sa (k INT, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE sb (k INT, y TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO sa (k,x) VALUES (1,'a1'),(2,'a2'),(NULL,'anull')").unwrap();
    e.execute_text(4, "INSERT INTO sb (k,y) VALUES (1,'b1'),(NULL,'bnull')").unwrap();
    let sas = e.populate_relational_residency_snapshot("sa").unwrap();
    let sbs = e.populate_relational_residency_snapshot("sb").unwrap();
    if sas.device_memory_proof.is_none() || sbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM sa JOIN sb ON sa.k = sb.k")
        .expect("build-side-swap NULL-key join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![("a1".to_string(), "b1".to_string())],
        "only key 1 matches; the NULL rows on both sides are skipped even though build/probe swapped"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_composite_one_member_null_other_equals_real_row() {
    // A composite key where ONE member is NULL but the other equals a real row's member. Validity must be
    // AND'd across BOTH members, so `(5,NULL)` is skipped. The NULL member's 0 placeholder makes `(5,NULL)`
    // pack identically to a real `(5,0)` row on the other side -- if validity were checked on member0 only,
    // `(5,NULL)` would spuriously match `(5,0)`.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE ca (k1 INT, k2 INT, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE cb (k1 INT, k2 INT, y TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO ca (k1,k2,x) VALUES (5,7,'match'),(5,NULL,'pnull')").unwrap();
    e.execute_text(4, "INSERT INTO cb (k1,k2,y) VALUES (5,7,'cb7'),(5,0,'cb0')").unwrap();
    let cas = e.populate_relational_residency_snapshot("ca").unwrap();
    let cbs = e.populate_relational_residency_snapshot("cb").unwrap();
    if cas.device_memory_proof.is_none() || cbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT x, y FROM ca JOIN cb ON ca.k1 = cb.k1 AND ca.k2 = cb.k2",
        )
        .expect("composite NULL-member join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![("match".to_string(), "cb7".to_string())],
        "(5,NULL) is skipped (validity AND'd across members); no spurious ('pnull','cb0')"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_anti_join_left_where_inner_is_null_with_null_keys() {
    // The anti-join `LEFT JOIN ... WHERE inner IS NULL` with NULL keys on BOTH sides. The left NULL-key row
    // matches nothing => it is NULL-padded => WHERE inner IS NULL KEEPS it. If the two NULL rows spuriously
    // matched, the left NULL-key row would be a MATCH (inner not NULL) and silently DROPPED -- data loss.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE la (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE lb (rid INT, label TEXT)").unwrap();
    e.execute_text(3, "INSERT INTO la (id,name) VALUES (1,'a'),(NULL,'lnull')").unwrap();
    e.execute_text(4, "INSERT INTO lb (rid,label) VALUES (1,'x'),(NULL,'rnull')").unwrap();
    let las = e.populate_relational_residency_snapshot("la").unwrap();
    let lbs = e.populate_relational_residency_snapshot("lb").unwrap();
    if las.device_memory_proof.is_none() || lbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name FROM la LEFT JOIN lb ON la.id = lb.rid WHERE lb.label IS NULL",
        )
        .expect("anti-join with NULL keys");
    let mut got: Vec<String> = res
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(s) => s.clone(),
            o => panic!("expected text, got {o:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec!["lnull".to_string()],
        "the unmatched NULL-key left row is padded + kept by WHERE inner IS NULL (not spuriously matched)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_v1b_null_at_word_boundary_index_32() {
    // The ONLY NULL key is at row index 32 -- bitmap word 1 (32>>5), bit 0 (32&31). Catches any off-by-one
    // in the kernel's bitmap word/bit math: exactly row 32 must be skipped, every other row matches 1:1.
    let mut e = Engine::new_local();
    const N: i32 = 40;
    let mut wa = String::new();
    let mut wb = String::new();
    for i in 0..N {
        if i > 0 {
            wa.push(',');
            wb.push(',');
        }
        if i == 32 {
            wa.push_str(&format!("(NULL,'a{i}')"));
        } else {
            wa.push_str(&format!("({i},'a{i}')"));
        }
        wb.push_str(&format!("({i},'b{i}')"));
    }
    e.execute_text(1, "CREATE TABLE wa (k INT, x TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE wb (k INT, y TEXT)").unwrap();
    e.execute_text(3, &format!("INSERT INTO wa (k,x) VALUES {wa}")).unwrap();
    e.execute_text(4, &format!("INSERT INTO wb (k,y) VALUES {wb}")).unwrap();
    let was = e.populate_relational_residency_snapshot("wa").unwrap();
    let wbs = e.populate_relational_residency_snapshot("wb").unwrap();
    if was.device_memory_proof.is_none() || wbs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT x, y FROM wa JOIN wb ON wa.k = wb.k")
        .expect("word-boundary NULL-key join");
    let mut got: Vec<(String, String)> = res
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Text(a), SqlValue::Text(b)) => (a.clone(), b.clone()),
            o => panic!("expected (text,text), got {o:?}"),
        })
        .collect();
    got.sort();
    let mut expected: Vec<(String, String)> = (0..N)
        .filter(|&i| i != 32)
        .map(|i| (format!("a{i}"), format!("b{i}")))
        .collect();
    expected.sort();
    assert_eq!(got, expected, "exactly row 32 (word 1, bit 0) is skipped; all others match 1:1");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_left_outer_join_null_pads_unmatched_left_rows() {
    // 2-relation LEFT OUTER join (M3 -- doc 21): every LEFT row appears; an unmatched left row -- the
    // CHILDLESS parent 3, AND the NULL-key left row 'nokey' (which matches nothing, 3VL) -- is kept with
    // the right relation's columns NULL-padded.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE lp (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE lc (pid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO lp (id, name) VALUES (1,'a'),(2,'b'),(3,'c'),(NULL,'nokey')")
        .unwrap();
    e.execute_text(4, "INSERT INTO lc (pid, label) VALUES (1,'x'),(1,'y'),(2,'z')")
        .unwrap();
    let ps = e.populate_relational_residency_snapshot("lp").unwrap();
    let cs = e.populate_relational_residency_snapshot("lc").unwrap();
    if ps.device_memory_proof.is_none() || cs.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT name, label FROM lp LEFT JOIN lc ON lp.id = lc.pid")
        .expect("left outer join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, Option<String>)> = res
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".to_string(), Some("x".to_string())),
            ("a".to_string(), Some("y".to_string())),
            ("b".to_string(), Some("z".to_string())),
            ("c".to_string(), None),     // childless parent 3 -> NULL-padded right columns
            ("nokey".to_string(), None), // NULL-key left row matches nothing -> NULL-padded
        ],
        "LEFT JOIN keeps every left row; unmatched (incl. NULL-key) rows are NULL-padded"
    );

    // TEXT-key LEFT join: the NULL pad maps over the GPU text hash join too (a different build/probe
    // orientation than int). 'z' has no match -> NULL-padded.
    e.execute_text(5, "CREATE TABLE tp (k TEXT, name TEXT)")
        .unwrap();
    e.execute_text(6, "CREATE TABLE tc (k TEXT, label TEXT)")
        .unwrap();
    e.execute_text(7, "INSERT INTO tp (k, name) VALUES ('a','pa'),('b','pb'),('z','pz')")
        .unwrap();
    e.execute_text(8, "INSERT INTO tc (k, label) VALUES ('a','ca'),('b','cb')")
        .unwrap();
    let tps = e.populate_relational_residency_snapshot("tp").unwrap();
    let tcs = e.populate_relational_residency_snapshot("tc").unwrap();
    if tps.device_memory_proof.is_none() || tcs.device_memory_proof.is_none() {
        return;
    }
    let tres = e
        .execute_resident_expr_select_sql("SELECT name, label FROM tp LEFT JOIN tc ON tp.k = tc.k")
        .expect("text-key left join");
    let mut tgot: Vec<(String, Option<String>)> = tres
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    tgot.sort();
    assert_eq!(
        tgot,
        vec![
            ("pa".to_string(), Some("ca".to_string())),
            ("pb".to_string(), Some("cb".to_string())),
            ("pz".to_string(), None), // 'z' has no match -> NULL-padded
        ],
        "text-key LEFT join NULL-pads the unmatched left row"
    );

    // A WHERE on a LEFT join filters the JOINED RESULT (PG semantics), not a per-side pushdown. The
    // predicate runs on the GPU; only lc row (1,'x') passes `lc.label = 'x'`, so every other tuple --
    // including the NULL-padded (c) / (nokey) rows whose lc.label is NULL (UNKNOWN) -- is dropped.
    let wres = e
        .execute_resident_expr_select_sql(
            "SELECT name, label FROM lp LEFT JOIN lc ON lp.id = lc.pid WHERE lc.label = 'x'",
        )
        .expect("LEFT JOIN with WHERE on the inner side filters the result");
    let mut wgot: Vec<(String, Option<String>)> = wres
        .rows
        .iter()
        .map(|r| {
            let name = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("name: {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => Some(t.clone()),
                SqlValue::Null => None,
                o => panic!("label: {o:?}"),
            };
            (name, label)
        })
        .collect();
    wgot.sort();
    assert_eq!(
        wgot,
        vec![("a".to_string(), Some("x".to_string()))],
        "WHERE on the inner side of a LEFT join drops the non-matching + NULL-padded tuples"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_right_and_full_outer_join_null_pad_the_correct_side() {
    // RIGHT keeps every RIGHT (new) row (unmatched -> the left columns NULL-padded); FULL keeps both
    // sides' unmatched rows (M3 -- doc 21). rl 2 'b' is left-only; rr 3 'z' is right-only.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE rl (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE rr (rid INT, label TEXT)")
        .unwrap();
    e.execute_text(3, "INSERT INTO rl (id, name) VALUES (1,'a'),(2,'b')")
        .unwrap();
    e.execute_text(4, "INSERT INTO rr (rid, label) VALUES (1,'x'),(3,'z')")
        .unwrap();
    let ls = e.populate_relational_residency_snapshot("rl").unwrap();
    let rs = e.populate_relational_residency_snapshot("rr").unwrap();
    if ls.device_memory_proof.is_none() || rs.device_memory_proof.is_none() {
        return;
    }
    let opt_pairs = |res: &RelationalSelectResult| -> Vec<(Option<String>, Option<String>)> {
        let opt = |c: &SqlValue| match c {
            SqlValue::Text(t) => Some(t.clone()),
            SqlValue::Null => None,
            o => panic!("expected text/null, got {o:?}"),
        };
        let mut v: Vec<(Option<String>, Option<String>)> =
            res.rows.iter().map(|r| (opt(&r[0]), opt(&r[1]))).collect();
        v.sort();
        v
    };

    // RIGHT: every rr row appears; (3,'z') is right-only -> name NULL. rl's left-only 'b' is DROPPED.
    let right = e
        .execute_resident_expr_select_sql("SELECT name, label FROM rl RIGHT JOIN rr ON rl.id = rr.rid")
        .expect("right outer join");
    assert_eq!(right.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        opt_pairs(&right),
        vec![
            (None, Some("z".to_string())), // right-only -> left columns NULL
            (Some("a".to_string()), Some("x".to_string())),
        ],
        "RIGHT JOIN keeps every right row; left-only 'b' dropped"
    );

    // FULL: the match + BOTH unmatched sides.
    let full = e
        .execute_resident_expr_select_sql("SELECT name, label FROM rl FULL JOIN rr ON rl.id = rr.rid")
        .expect("full outer join");
    assert_eq!(
        opt_pairs(&full),
        vec![
            (None, Some("z".to_string())),                  // right-only
            (Some("a".to_string()), Some("x".to_string())), // match
            (Some("b".to_string()), None),                  // left-only
        ],
        "FULL JOIN keeps the match + both unmatched sides"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nway_outer_join_null_pads_through_the_pipeline() {
    // N-way (multi-step) OUTER join (M3 -- doc 21): a prior step's NULL pad carries a JOIN_NULL_ROW
    // sentinel; the next step reads it as a NULL key (matches nothing; a LEFT step re-pads it) instead of
    // OOB-reading host_rows. Two cases: (1) the carried NULL is in a relation NOT used as the next key (the
    // tuple still participates via a non-padded key); (2) the carried NULL IS the next key (the tuple is
    // re-padded). All on the GPU join pipeline.
    let mut e = Engine::new_local();
    // Case 1: A LEFT JOIN B (on A.id) LEFT JOIN C (on A.id). A=3 has no B (B NULL-padded) but matches C=3.
    e.execute_text(1, "CREATE TABLE a3 (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE b3 (aid INT, bl TEXT)").unwrap();
    e.execute_text(3, "CREATE TABLE c3 (aid INT, cl TEXT)").unwrap();
    e.execute_text(4, "INSERT INTO a3 (id,name) VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
    e.execute_text(5, "INSERT INTO b3 (aid,bl) VALUES (1,'b1'),(2,'b2')").unwrap();
    e.execute_text(6, "INSERT INTO c3 (aid,cl) VALUES (1,'c1'),(3,'c3')").unwrap();
    for t in ["a3", "b3", "c3"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let opt = |v: &SqlValue| match v {
        SqlValue::Text(t) => Some(t.clone()),
        SqlValue::Null => None,
        o => panic!("unexpected {o:?}"),
    };
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, bl, cl FROM a3 LEFT JOIN b3 ON a3.id = b3.aid LEFT JOIN c3 ON a3.id = c3.aid",
        )
        .expect("N-way LEFT-LEFT join");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut got: Vec<(String, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]).unwrap(), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".into(), Some("b1".into()), Some("c1".into())),
            ("b".into(), Some("b2".into()), None), // B match, no C
            ("c".into(), None, Some("c3".into())), // B NULL-padded (step 1), still matches C on A.id
        ],
        "N-way LEFT-LEFT: a carried NULL in a non-key relation still joins on a non-padded key"
    );

    // Case 2: A LEFT JOIN B (on A.id) LEFT JOIN C (on B.cid). A=3's B is NULL-padded, so B.cid is a carried
    // NULL key at step 2 -> A=3 matches no C -> C re-padded (would OOB-read host_rows without the fix).
    e.execute_text(7, "CREATE TABLE a4 (id INT, name TEXT)").unwrap();
    e.execute_text(8, "CREATE TABLE b4 (aid INT, cid INT, bl TEXT)").unwrap();
    e.execute_text(9, "CREATE TABLE c4 (id INT, cl TEXT)").unwrap();
    e.execute_text(10, "INSERT INTO a4 (id,name) VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
    e.execute_text(11, "INSERT INTO b4 (aid,cid,bl) VALUES (1,100,'b1'),(2,200,'b2')").unwrap();
    e.execute_text(12, "INSERT INTO c4 (id,cl) VALUES (100,'c1'),(200,'c2'),(300,'c3')").unwrap();
    for t in ["a4", "b4", "c4"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, bl, cl FROM a4 LEFT JOIN b4 ON a4.id = b4.aid LEFT JOIN c4 ON b4.cid = c4.id",
        )
        .expect("N-way LEFT-LEFT join keyed on a previously-padded relation");
    let mut got: Vec<(String, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]).unwrap(), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".into(), Some("b1".into()), Some("c1".into())),
            ("b".into(), Some("b2".into()), Some("c2".into())),
            ("c".into(), None, None), // B NULL-padded -> B.cid NULL key at step 2 -> C re-padded
        ],
        "N-way: a carried NULL used AS the next join key matches nothing and is re-padded (no OOB)"
    );

    // Case 3: a RIGHT step over a MULTI-relation accumulated side -- A JOIN B (inner) RIGHT JOIN C. An
    // unmatched C row pads the ENTIRE accumulated side (BOTH A and B), exercising the RIGHT pad's
    // `take(new_rel)` over >1 accumulated relations.
    e.execute_text(13, "CREATE TABLE a5 (id INT, an TEXT)").unwrap();
    e.execute_text(14, "CREATE TABLE b5 (aid INT, bn TEXT)").unwrap();
    e.execute_text(15, "CREATE TABLE c5 (cx INT, cn TEXT)").unwrap();
    e.execute_text(16, "INSERT INTO a5 (id,an) VALUES (1,'a1'),(2,'a2')").unwrap();
    e.execute_text(17, "INSERT INTO b5 (aid,bn) VALUES (1,'b1'),(2,'b2')").unwrap();
    e.execute_text(18, "INSERT INTO c5 (cx,cn) VALUES (1,'c1'),(3,'c3')").unwrap();
    for t in ["a5", "b5", "c5"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT an, bn, cn FROM a5 JOIN b5 ON a5.id = b5.aid RIGHT JOIN c5 ON b5.aid = c5.cx",
        )
        .expect("RIGHT JOIN over a multi-relation accumulated side");
    let mut got: Vec<(Option<String>, Option<String>, Option<String>)> = res
        .rows
        .iter()
        .map(|r| (opt(&r[0]), opt(&r[1]), opt(&r[2])))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            (None, None, Some("c3".into())), // right-only C row pads BOTH accumulated relations (A and B)
            (Some("a1".into()), Some("b1".into()), Some("c1".into())),
        ],
        "RIGHT step over a 2-relation accumulated side pads the WHOLE accumulated tuple (A and B)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_explicit_nulls_first_last_honored_on_device() {
    // M3 (doc 21): explicit NULLS FIRST / NULLS LAST OVERRIDES PG's default placement, honored ON-DEVICE
    // in the GPU sort comparator (the per-key nulls_first bitmask), DECOUPLED from ASC/DESC. Without an
    // override ASC = NULLS LAST and DESC = NULLS FIRST (the default, covered elsewhere).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a,b) VALUES (3,100),(NULL,100),(1,100),(NULL,100),(2,100)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    let ints = |rows: &[Vec<SqlValue>]| -> Vec<Option<i32>> {
        rows.iter()
            .map(|row| match row[0] {
                SqlValue::Int4(v) => Some(v),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // ASC NULLS FIRST: NULLs first, then ascending (overrides the ASC default of NULLS LAST).
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a ASC NULLS FIRST")
        .unwrap();
    assert_eq!(ints(&r.rows), vec![None, None, Some(1), Some(2), Some(3)], "ASC NULLS FIRST");
    // DESC NULLS LAST: descending, then NULLs last (overrides the DESC default of NULLS FIRST).
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a DESC NULLS LAST")
        .unwrap();
    assert_eq!(ints(&r.rows), vec![Some(3), Some(2), Some(1), None, None], "DESC NULLS LAST");
    // Sanity: the default is unchanged (ASC => NULLS LAST) when no override is given.
    let r = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE b >= 0 ORDER BY a")
        .unwrap();
    assert_eq!(ints(&r.rows), vec![Some(1), Some(2), Some(3), None, None], "ASC default = NULLS LAST");
    // Explicit NULLS FIRST/LAST is now ALSO honored on the GROUP BY result path. a=[3,NULL,1,NULL,2] ->
    // groups {1,2,3,NULL}; ORDER BY a NULLS FIRST -> NULL first, then ascending.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT a, COUNT(*) FROM t WHERE b >= 0 GROUP BY a ORDER BY a NULLS FIRST",
        )
        .expect("grouped ORDER BY a NULLS FIRST");
    assert_eq!(
        ints(&r.rows),
        vec![None, Some(1), Some(2), Some(3)],
        "explicit NULLS FIRST on a GROUP BY result places the NULL group first"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_composite_key_per_member_null_on_device() {
    // M3 (doc 21): GROUP BY a COMPOSITE key (`a, b`) with nullable members. Per-member NULL is encoded in
    // the wide key (a trailing validity word written ON-DEVICE by gpu_db_build_wide_key), so (NULL,5),
    // (1,5), (NULL,6), (NULL,NULL), (1,NULL) are all DISTINCT groups, each member rendered SqlValue::Null
    // from the representative row. A nullable composite routes to the wide-key path (the i64 pack has no
    // room for validity).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (a,b) VALUES (1,5),(NULL,5),(1,5),(NULL,6),(NULL,NULL),(1,NULL)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    let groups = |rows: &[Vec<SqlValue>]| -> Vec<(Option<i32>, Option<i32>, i64)> {
        let opt = |v: &SqlValue| match v {
            SqlValue::Int4(x) => Some(*x),
            SqlValue::Null => None,
            o => panic!("unexpected key {o:?}"),
        };
        let mut v: Vec<(Option<i32>, Option<i32>, i64)> = rows
            .iter()
            .map(|r| {
                let c = match r[2] {
                    SqlValue::Int8(x) => x,
                    SqlValue::Int4(x) => i64::from(x),
                    ref o => panic!("unexpected count {o:?}"),
                };
                (opt(&r[0]), opt(&r[1]), c)
            })
            .collect();
        v.sort();
        v
    };
    let r = e
        .execute_resident_expr_select_sql("SELECT a, b, COUNT(*) FROM t GROUP BY a, b")
        .expect("GROUP BY a nullable composite key");
    assert_eq!(
        groups(&r.rows),
        vec![
            (None, None, 1),       // (NULL, NULL)
            (None, Some(5), 1),    // (NULL, 5)
            (None, Some(6), 1),    // (NULL, 6)
            (Some(1), None, 1),    // (1, NULL)
            (Some(1), Some(5), 2), // (1, 5) x2
        ],
        "composite NULL members form distinct groups: (NULL,5) != (1,5) != (NULL,6) != (NULL,NULL) != (1,NULL)"
    );
    // COUNT(DISTINCT) over a nullable composite key clean-errors (its dedup sub-pass builds the wide key
    // without validity, so it would merge NULL with a real value) -- a clean error, not a wrong answer.
    assert!(
        e.execute_resident_expr_select_sql("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a, b")
            .is_err(),
        "COUNT(DISTINCT) over a nullable composite key clean-errors"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_count_distinct_over_a_nullable_value_clean_errors() {
    // M3 (doc 21): COUNT(DISTINCT v) over a NULLABLE value must EXCLUDE NULLs (PG); the sort-based reps
    // pass has no value validity (it would over-count NULL as a distinct value), so clean-error rather than
    // silently mis-count. (A non-null COUNT(DISTINCT) is unaffected -- covered by the count_distinct suite.)
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g,v) VALUES (1,10),(1,NULL),(1,10),(2,20)").unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    assert!(
        e.execute_resident_expr_select_sql("SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g")
            .is_err(),
        "grouped COUNT(DISTINCT) over a nullable value clean-errors (no silent over-count)"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT COUNT(DISTINCT v) FROM t").is_err(),
        "scalar COUNT(DISTINCT) over a nullable value clean-errors (no silent over-count)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_nullable_expression_key_forms_a_null_group_on_device() {
    // M3 (doc 21): GROUP BY a NULLABLE int4 EXPRESSION (`a + b`, b non-null) -- the rows where a is NULL
    // (so a+b is NULL) form their OWN group, rendered SqlValue::Null. Reuses the single-column NULL-key
    // reserved slot via the one nullable operand's validity bitmap (ZERO kernel change).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a,b) VALUES (1,10),(NULL,10),(1,10),(2,10),(NULL,10)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    let groups = |rows: &[Vec<SqlValue>]| -> Vec<(Option<i32>, i64)> {
        let mut v: Vec<(Option<i32>, i64)> = rows
            .iter()
            .map(|r| {
                let k = match r[0] {
                    SqlValue::Int4(x) => Some(x),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected key {o:?}"),
                };
                let c = match r[1] {
                    SqlValue::Int8(x) => x,
                    SqlValue::Int4(x) => i64::from(x),
                    ref o => panic!("unexpected count {o:?}"),
                };
                (k, c)
            })
            .collect();
        v.sort();
        v
    };
    // a+b: 11, NULL, 11, 12, NULL -> groups {11:2, 12:1, NULL:2}.
    let r = e
        .execute_resident_expr_select_sql("SELECT a + b, COUNT(*) FROM t GROUP BY a + b")
        .expect("GROUP BY nullable int4 expression");
    assert_eq!(
        groups(&r.rows),
        vec![(None, 2), (Some(11), 2), (Some(12), 1)],
        "GROUP BY a+b: the NULL-result rows form their own group (rendered NULL)"
    );
    // GROUP BY over an expression with TWO nullable operands clean-errors (needs a derived validity AND).
    e.execute_text(3, "CREATE TABLE t2 (a INT, b INT)").unwrap();
    e.execute_text(4, "INSERT INTO t2 (a,b) VALUES (1,2),(NULL,NULL)").unwrap();
    if e.populate_relational_residency_snapshot("t2").unwrap().device_memory_proof.is_some() {
        assert!(
            e.execute_resident_expr_select_sql("SELECT a + b, COUNT(*) FROM t2 GROUP BY a + b")
                .is_err(),
            "GROUP BY an expression over two nullable operands clean-errors"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_nullable_expression_places_null_results_on_device() {
    // M3 (doc 21): ORDER BY a NULLABLE int4 EXPRESSION (`a + b`). A NULL result (any operand NULL) becomes
    // the i64::MAX default-end sentinel, blended ON-DEVICE (a validity-mask VM run + the blend kernel), so
    // NULL-expression rows sort to PG's default end. No host NULL decision.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (id INT, a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (id,a,b) VALUES (1,5,1),(2,NULL,1),(3,2,1),(4,NULL,1)")
        .unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    let ids = |rows: &[Vec<SqlValue>]| -> Vec<i32> {
        rows.iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // a+b: id1=6, id2=NULL, id3=3, id4=NULL. ASC default = NULLS LAST; the secondary `id` orders the
    // (tied) NULL-result group deterministically -> [3 (=3), 1 (=6), 2 (NULL), 4 (NULL)].
    let r = e
        .execute_resident_expr_select_sql("SELECT id FROM t ORDER BY a + b, id")
        .expect("nullable int4 expression ORDER BY");
    assert_eq!(
        ids(&r.rows),
        vec![3, 1, 2, 4],
        "nullable a+b: non-NULL ascending, then NULL results last (PG default), on-device"
    );
    // A nullable int8 expression clean-errors (the i64::MAX NULL sentinel could collide with a real bigint).
    e.execute_text(3, "CREATE TABLE t8 (id INT, a BIGINT, b BIGINT)").unwrap();
    e.execute_text(4, "INSERT INTO t8 (id,a,b) VALUES (1,5,1),(2,NULL,1)").unwrap();
    if e.populate_relational_residency_snapshot("t8").unwrap().device_memory_proof.is_some() {
        assert!(
            e.execute_resident_expr_select_sql("SELECT id FROM t8 ORDER BY a + b").is_err(),
            "nullable int8 expression ORDER BY clean-errors (sentinel collision)"
        );
    }
    // Explicit NULLS FIRST/LAST on a nullable expression clean-errors (value-sentinel only does default).
    assert!(
        e.execute_resident_expr_select_sql("SELECT id FROM t ORDER BY a + b NULLS FIRST").is_err(),
        "explicit NULLS FIRST on a nullable expression clean-errors"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_result_order_by_explicit_nulls_first_last_on_device() {
    // M3 (doc 21): explicit NULLS FIRST/LAST on a GROUP BY result ORDER BY, honored ON-DEVICE in
    // gpu_sort_result_rows (an int result key's NULL sentinel value is chosen per the request). The NULL
    // group's key renders SqlValue::Null and places per the override.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (g,v) VALUES (1,10),(NULL,10),(2,10),(NULL,10)").unwrap();
    if e.populate_relational_residency_snapshot("t").unwrap().device_memory_proof.is_none() {
        return;
    }
    let keys = |rows: &[Vec<SqlValue>]| -> Vec<Option<i32>> {
        rows.iter()
            .map(|r| match r[0] {
                SqlValue::Int4(x) => Some(x),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect()
    };
    // groups {1, 2, NULL}. ORDER BY g NULLS FIRST overrides the ASC default (NULLS LAST) -> NULL first.
    let r = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g NULLS FIRST")
        .expect("grouped ORDER BY NULLS FIRST");
    assert_eq!(keys(&r.rows), vec![None, Some(1), Some(2)], "grouped ORDER BY g NULLS FIRST");
    // Default ASC = NULLS LAST.
    let r = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g")
        .expect("grouped ORDER BY default");
    assert_eq!(keys(&r.rows), vec![Some(1), Some(2), None], "grouped ORDER BY g default = NULLS LAST");
    // DESC NULLS LAST overrides the DESC default (NULLS FIRST) -> descending then NULL last.
    let r = e
        .execute_resident_expr_select_sql(
            "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g DESC NULLS LAST",
        )
        .expect("grouped ORDER BY DESC NULLS LAST");
    assert_eq!(keys(&r.rows), vec![Some(2), Some(1), None], "grouped ORDER BY g DESC NULLS LAST");
    // A BIGINT result key is now exact too: its NULL is marked by an on-device validity bitmap (not a
    // value sentinel), so explicit NULLS FIRST is honored with no collision risk. g8=[1,NULL] -> NULL first.
    e.execute_text(3, "CREATE TABLE t8 (g BIGINT, v INT)").unwrap();
    e.execute_text(4, "INSERT INTO t8 (g,v) VALUES (1,10),(NULL,10)").unwrap();
    if e.populate_relational_residency_snapshot("t8").unwrap().device_memory_proof.is_some() {
        let r8 = e
            .execute_resident_expr_select_sql(
                "SELECT g, COUNT(*) FROM t8 GROUP BY g ORDER BY g NULLS FIRST",
            )
            .expect("bigint grouped ORDER BY g NULLS FIRST");
        let g8: Vec<Option<i64>> = r8
            .rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int8(x) => Some(x),
                SqlValue::Null => None,
                ref o => panic!("unexpected {o:?}"),
            })
            .collect();
        assert_eq!(g8, vec![None, Some(1)], "bigint grouped ORDER BY g NULLS FIRST: NULL group first");
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_order_by_explicit_nulls_first_last_on_device() {
    // M3 (doc 21): explicit NULLS FIRST/LAST on a JOIN-result ORDER BY, honored ON-DEVICE via
    // gpu_sort_result_rows (the override is threaded through JoinPlan.order_by_nulls_first). A LEFT join
    // pads the unmatched row's x to NULL; the override places it.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, n TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id,n) VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
    e.execute_text(4, "INSERT INTO r (rid,x) VALUES (1,5),(2,7)").unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let rows = |res: &RelationalSelectResult| -> Vec<(String, Option<i32>)> {
        res.rows
            .iter()
            .map(|r| {
                let n = match &r[0] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("unexpected {o:?}"),
                };
                let x = match r[1] {
                    SqlValue::Int4(v) => Some(v),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected {o:?}"),
                };
                (n, x)
            })
            .collect()
    };
    // result: (a,5),(b,7),(c,NULL). ORDER BY x NULLS FIRST overrides the ASC default -> NULL (c) first.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT n, x FROM l LEFT JOIN r ON l.id = r.rid ORDER BY x NULLS FIRST",
        )
        .expect("join ORDER BY x NULLS FIRST");
    assert_eq!(
        rows(&res),
        vec![("c".into(), None), ("a".into(), Some(5)), ("b".into(), Some(7))],
        "join ORDER BY x NULLS FIRST places the NULL-padded row first"
    );
    // Default ASC = NULLS LAST.
    let res = e
        .execute_resident_expr_select_sql("SELECT n, x FROM l LEFT JOIN r ON l.id = r.rid ORDER BY x")
        .expect("join ORDER BY x default");
    assert_eq!(
        rows(&res),
        vec![("a".into(), Some(5)), ("b".into(), Some(7)), ("c".into(), None)],
        "join ORDER BY x default = NULLS LAST"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_outer_join_with_where_filters_the_result_not_the_inputs() {
    // M3 (doc 21): a WHERE on an OUTER join filters the JOINED RESULT (PG semantics), NOT a per-side
    // pushdown (which is not filter-commutative for an outer join). The predicate runs on the GPU
    // (lower_resident_predicate); its survivor set post-filters the padded result: a JOIN_NULL_ROW pad
    // means the relation's columns are NULL -> the predicate is UNKNOWN -> the tuple is dropped.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, x INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id,name) VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
    e.execute_text(4, "INSERT INTO r (rid,x) VALUES (1,5),(2,7)").unwrap();
    for t in ["l", "r"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let rows = |res: &RelationalSelectResult| -> Vec<(String, Option<i32>)> {
        let mut v: Vec<(String, Option<i32>)> = res
            .rows
            .iter()
            .map(|row| {
                let name = match &row[0] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("unexpected {o:?}"),
                };
                let x = match row[1] {
                    SqlValue::Int4(v) => Some(v),
                    SqlValue::Null => None,
                    ref o => panic!("unexpected {o:?}"),
                };
                (name, x)
            })
            .collect();
        v.sort();
        v
    };

    // WHERE on the INNER (padded) side: `r.x = 5` drops the NULL-padded row (c) AND the non-matching
    // matched row (b, x=7) -- effectively inner on that condition. Only (a, 5) survives.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE r.x = 5",
        )
        .expect("LEFT JOIN with WHERE on the inner side");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        rows(&res),
        vec![("a".into(), Some(5))],
        "WHERE on the padded side filters the result (UNKNOWN on the NULL pad drops it)"
    );

    // WHERE on the PRESERVED (left) side: `l.id >= 2` keeps id 2 (matched, x=7) and id 3 (padded, NULL),
    // dropping id 1 -- the padding is preserved for the surviving left rows.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE l.id >= 2",
        )
        .expect("LEFT JOIN with WHERE on the preserved side");
    assert_eq!(
        rows(&res),
        vec![("b".into(), Some(7)), ("c".into(), None)],
        "WHERE on the preserved side keeps the NULL pad for surviving left rows"
    );

    // ANTI-JOIN: `WHERE r.x IS NULL` is TRUE on the NULL pad, so the post-filter must KEEP the padded
    // (unmatched-left) rows -- and drop every matched row (whose r.x is non-NULL). l=3 has no r match.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT name, x FROM l LEFT JOIN r ON l.id = r.rid WHERE r.x IS NULL",
        )
        .expect("LEFT JOIN anti-join (WHERE inner IS NULL)");
    assert_eq!(
        rows(&res),
        vec![("c".into(), None)],
        "anti-join: WHERE inner.col IS NULL keeps the NULL-padded (unmatched) left rows"
    );
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
    // LEFT JOIN is SUPPORTED now (2-relation, ON-only) -- it routes to the join path (the residency
    // check here, since a/b are not resident), NOT a parser rejection. RIGHT/FULL remain follow-ups.
    let left = reject("SELECT x, y FROM a LEFT JOIN b ON a.k = b.k");
    assert!(
        left.contains("resident snapshot") || left.contains("join path"),
        "LEFT JOIN routes to the join path now, got: {left}"
    );
    // RIGHT/FULL JOIN are SUPPORTED now (2-relation, ON-only) -- they route to the join path too.
    let right = reject("SELECT x, y FROM a RIGHT JOIN b ON a.k = b.k");
    assert!(
        right.contains("resident snapshot") || right.contains("join path"),
        "RIGHT JOIN routes to the join path now, got: {right}"
    );
    // An OUTER JOIN with NATURAL/USING is still a follow-up -> clean parser rejection.
    assert!(
        reject("SELECT x, y FROM a NATURAL LEFT JOIN b").contains("natural")
            || reject("SELECT x, y FROM a NATURAL LEFT JOIN b").contains("using"),
        "OUTER NATURAL/USING rejected (a follow-up)"
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
fn gpu_where_in_and_not_in() {
    // IN / NOT IN lower to an OR-chain of `=` / AND-chain of `<>` on the general GPU executor (no new
    // kernel) -- INT keys here (text IN awaits text AND/OR on the executor; see gpu-type-matrix).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE inq (id INT, tag INT)").unwrap();
    e.execute_text(2, "INSERT INTO inq (id, tag) VALUES (1,10),(2,20),(3,30),(4,40)").unwrap();
    if e.populate_relational_residency_snapshot("inq").unwrap().device_memory_proof.is_none() {
        return;
    }
    let ids = |res: &RelationalSelectResult| -> Vec<i32> {
        let mut v: Vec<i32> = res
            .rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Int4(v) => *v,
                other => panic!("expected int4, got {other:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let run = |sql: &str| e.execute_resident_expr_select_sql(sql).expect(sql);
    let in_int = run("SELECT id FROM inq WHERE id IN (1, 3)");
    assert_eq!(in_int.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(ids(&in_int), vec![1, 3], "id IN (1,3)");
    assert_eq!(
        ids(&run("SELECT id FROM inq WHERE id NOT IN (1, 3)")),
        vec![2, 4],
        "id NOT IN (1,3) is the complement"
    );
    assert_eq!(
        ids(&run("SELECT id FROM inq WHERE id IN (4, 2, 4)")),
        vec![2, 4],
        "multi-element IN with a duplicate"
    );
    assert_eq!(
        ids(&run("SELECT id FROM inq WHERE id IN (2)")),
        vec![2],
        "single-element IN"
    );
    // IN composes with AND under the general boolean executor.
    assert_eq!(
        ids(&run("SELECT id FROM inq WHERE id IN (1, 2, 3) AND tag <> 20")),
        vec![1, 3],
        "IN AND <> composes"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_where_text_and_or_and_in() {
    // Text AND/OR on the general executor: each text `=`/`<>` becomes a TextEqMask the mask VM combines
    // with AND/OR (and with int4) -- so text IN / NOT IN / multi-text WHERE run on the GPU.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE tq (id INT, tag TEXT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO tq (id, tag) VALUES (1,'apple'),(2,'banana'),(3,'cherry'),(4,'apple')",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("tq").unwrap().device_memory_proof.is_none() {
        return;
    }
    let ids = |res: &RelationalSelectResult| -> Vec<i32> {
        let mut v: Vec<i32> = res
            .rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Int4(v) => *v,
                other => panic!("expected int4, got {other:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let run = |sql: &str| e.execute_resident_expr_select_sql(sql).expect(sql);
    let or = run("SELECT id FROM tq WHERE tag = 'apple' OR tag = 'cherry'");
    assert_eq!(or.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(ids(&or), vec![1, 3, 4], "text OR (apple, cherry) runs on the GPU");
    assert_eq!(
        ids(&run("SELECT id FROM tq WHERE tag = 'apple' AND id > 1")),
        vec![4],
        "text = AND an int4 comparison (mixed masks under the i32 VM)"
    );
    assert_eq!(
        ids(&run("SELECT id FROM tq WHERE tag IN ('banana', 'cherry')")),
        vec![2, 3],
        "text IN -> OR-chain of text ="
    );
    assert_eq!(
        ids(&run("SELECT id FROM tq WHERE tag NOT IN ('apple')")),
        vec![2, 3],
        "text NOT IN -> AND-chain of text <>"
    );
    assert_eq!(
        ids(&run("SELECT id FROM tq WHERE tag <> 'apple' AND tag <> 'banana'")),
        vec![3],
        "text <> AND text <>"
    );
    assert_eq!(
        ids(&run("SELECT id FROM tq WHERE tag IN ('apple', 'banana') AND id <> 2")),
        vec![1, 4],
        "text IN AND an int4 <> compose"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_catalog_pg_class_join_pg_namespace_d_metadata() {
    // Closes the function-free `\d` family (golden 23/29): pg_class JOIN pg_namespace, projecting
    // relpersistence (newly synthesized), filtered by nspname + relkind, ORDER BY relname -- the whole
    // query on the GPU join + GPU sort path. BOTH sides are SYNTHESIZED catalog relations (transient
    // device payloads), schema-qualified `pg_catalog.<rel>`.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE dz_people (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE dz_teams (id INT)").unwrap();
    e.execute_text(3, "CREATE TABLE dz_ignored (id INT)").unwrap();
    // Probe: the transient catalog payload needs a GPU; skip cleanly if unavailable.
    let probe = e.execute_resident_expr_select_sql(
        "SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n \
         ON n.oid = c.relnamespace WHERE c.relname = 'dz_people'",
    );
    match &probe {
        Ok(r) if r.executed_target == DeviceTarget::Gpu(0) => {}
        _ => return,
    }
    let meta_rows = |res: &RelationalSelectResult| -> Vec<(String, String, String, String)> {
        res.rows
            .iter()
            .map(|r| {
                let t = |c: &SqlValue| match c {
                    SqlValue::Text(s) => s.clone(),
                    other => panic!("expected text, got {other:?}"),
                };
                (t(&r[0]), t(&r[1]), t(&r[2]), t(&r[3]))
            })
            .collect()
    };
    let people = ("public".to_string(), "dz_people".to_string(), "r".to_string(), "p".to_string());
    let teams = ("public".to_string(), "dz_teams".to_string(), "r".to_string(), "p".to_string());
    let ignored =
        ("public".to_string(), "dz_ignored".to_string(), "r".to_string(), "p".to_string());
    // Golden 23 form: every public relkind='r' relation, ordered by name (dz_ignored < dz_people < dz_teams).
    // The whole query -- the join, the per-side text filters, the ORDER BY, and relpersistence -- is on GPU.
    let all = e
        .execute_resident_expr_select_sql(
            "SELECT n.nspname, c.relname, c.relkind, c.relpersistence \
             FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' AND c.relkind = 'r' ORDER BY c.relname",
        )
        .expect("catalog \\d metadata join (golden 23)");
    assert_eq!(all.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        meta_rows(&all),
        vec![ignored.clone(), people.clone(), teams.clone()],
        "golden 23: the join filters + orders on the GPU and projects relpersistence='p'"
    );
    // Golden 29 form: the same, restricted by `relname IN (...)` (a text IN -> GPU mask VM) -> dz_ignored
    // is excluded. The IN runs as a per-side text OR-chain on the pg_class side before the join.
    let subset = e
        .execute_resident_expr_select_sql(
            "SELECT n.nspname, c.relname, c.relkind, c.relpersistence \
             FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' AND c.relname IN ('dz_people', 'dz_teams') \
             AND c.relkind = 'r' ORDER BY c.relname",
        )
        .expect("catalog \\d metadata join (golden 29, text IN)");
    assert_eq!(
        meta_rows(&subset),
        vec![people, teams],
        "golden 29: relname IN (...) excludes dz_ignored, still GPU join + sort"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_where_bool_and_or() {
    // Bool column in AND/OR on the general executor: a bool bitmap -> i32 mask the VM combines with
    // AND/OR (and int4). `NOT flag` is `flag = false`. Reuses gpu_db_resident_bool_to_mask (no new kernel).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE bq (id INT, active BOOL, qty INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO bq (id, active, qty) VALUES (1,true,5),(2,false,3),(3,true,10),(4,false,8)",
    )
    .unwrap();
    if e.populate_relational_residency_snapshot("bq").unwrap().device_memory_proof.is_none() {
        return;
    }
    let ids = |res: &RelationalSelectResult| -> Vec<i32> {
        let mut v: Vec<i32> = res
            .rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Int4(v) => *v,
                other => panic!("expected int4, got {other:?}"),
            })
            .collect();
        v.sort();
        v
    };
    let run = |sql: &str| e.execute_resident_expr_select_sql(sql).expect(sql);
    let and = run("SELECT id FROM bq WHERE active AND qty > 6");
    assert_eq!(and.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(ids(&and), vec![3], "bare bool AND an int4 comparison");
    assert_eq!(
        ids(&run("SELECT id FROM bq WHERE NOT active AND qty > 4")),
        vec![4],
        "NOT bool (-> bool=false) AND int4"
    );
    assert_eq!(
        ids(&run("SELECT id FROM bq WHERE active OR qty > 7")),
        vec![1, 3, 4],
        "bool OR int4"
    );
    assert_eq!(
        ids(&run("SELECT id FROM bq WHERE active = false AND qty < 5")),
        vec![2],
        "bool = false AND int4"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_catalog_pg_attribute_d_table_columns() {
    // Function-free `\d <table>` column listing (golden 24 minus format_type): 3-way pg_attribute JOIN
    // pg_class JOIN pg_namespace, the per-side `attnum > 0 AND NOT attisdropped` (int4 AND bool, now on
    // the GPU mask VM), ORDER BY attnum, projecting the newly-synthesized atttypmod/attisdropped.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE dt_widget (id INT, label TEXT, qty INT)").unwrap();
    let probe = e.execute_resident_expr_select_sql(
        "SELECT a.attname FROM pg_catalog.pg_attribute a \
         JOIN pg_catalog.pg_class c ON c.oid = a.attrelid WHERE c.relname = 'dt_widget'",
    );
    match &probe {
        Ok(r) if r.executed_target == DeviceTarget::Gpu(0) => {}
        _ => return,
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT a.attnum, a.attname, a.attnotnull, a.atttypmod \
             FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid = a.attrelid \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE n.nspname = 'public' AND c.relname = 'dt_widget' \
             AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
        )
        .expect("catalog \\d <table> column listing");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let cols: Vec<(i32, String, bool, i32)> = res
        .rows
        .iter()
        .map(|r| {
            let num = match &r[0] {
                SqlValue::Int4(v) => *v,
                o => panic!("attnum int4, got {o:?}"),
            };
            let name = match &r[1] {
                SqlValue::Text(s) => s.clone(),
                o => panic!("attname text, got {o:?}"),
            };
            let notnull = match &r[2] {
                SqlValue::Bool(b) => *b,
                o => panic!("attnotnull bool, got {o:?}"),
            };
            let typmod = match &r[3] {
                SqlValue::Int4(v) => *v,
                o => panic!("atttypmod int4, got {o:?}"),
            };
            (num, name, notnull, typmod)
        })
        .collect();
    assert_eq!(
        cols,
        vec![
            (1, "id".to_string(), false, -1),
            (2, "label".to_string(), false, -1),
            (3, "qty".to_string(), false, -1),
        ],
        "the \\d <table> column join filters (attnum>0 AND NOT attisdropped) + orders on the GPU"
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
fn gpu_join_result_nullable_value_columns_from_device() {
    // S7/V3: the join result VALUES are gathered from each relation's DEVICE payload (not a host_rows
    // copy). The critical new surface vs the prior host gather: a MATCHED row whose projected NON-KEY
    // column is NULL must emit SqlValue::Null via the device validity bitmap -- NOT the 0/"" placeholder
    // the device stores for a NULL cell. The existing NULL-key tests only DROP NULL-KEY rows; here the join
    // KEY is non-null and the NULLs live in projected value columns of MATCHED rows (int, text, numeric).
    // Plus a LEFT-outer variant where a pad-NULL (JOIN_NULL_ROW) and a value-NULL coexist.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, v INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w NUMERIC(10,2))").unwrap();
    // id=1: l.v NULL, matched to r.w NULL. id=2: l.name NULL, matched to r.w=2.50. id=3: unmatched (pad).
    e.execute_text(
        3,
        "INSERT INTO l (id, v, name) VALUES (1, NULL, 'a'), (2, 20, NULL), (3, 30, 'c')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO r (rid, w) VALUES (1, NULL), (2, 2.50)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // INNER: matched rows (1, NULL, 'a', NULL) and (2, 20, NULL, 2.50). Every NULL is a projected non-key
    // value of a MATCHED row -> it must come back as SqlValue::Null from the device validity bitmap.
    let inner = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.v, l.name, r.w FROM l JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("inner join projecting nullable value columns");
    assert_eq!(inner.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        inner.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Null,
                SqlValue::Text("a".to_string()),
                SqlValue::Null,
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int4(20),
                SqlValue::Null,
                SqlValue::Numeric(Decimal128::new(250, 2)),
            ],
        ],
        "matched-row NULL values come from the device validity bitmap, not the 0/\"\" placeholder"
    );
    // LEFT OUTER: id=3 matches nothing -> r.w is a JOIN_NULL_ROW pad NULL; id=1's value-NULLs coexist.
    let left = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.v, l.name, r.w FROM l LEFT JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("left outer join projecting nullable value columns + a pad");
    assert_eq!(
        left.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Null,
                SqlValue::Text("a".to_string()),
                SqlValue::Null,
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int4(20),
                SqlValue::Null,
                SqlValue::Numeric(Decimal128::new(250, 2)),
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Int4(30),
                SqlValue::Text("c".to_string()),
                SqlValue::Null,
            ],
        ],
        "a pad-NULL (unmatched right) and real value-NULLs both render as SqlValue::Null"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_join_limit_offset_window_edges() {
    // S7/V3: OFFSET/LIMIT on the join result now WINDOWS the device sort permutation (gpu_sort_permutation),
    // gathering only the kept window -- no host drain/truncate on result data. Edge cases vs the old
    // drain/truncate: OFFSET past the end -> empty, LIMIT 0 -> empty, OFFSET+LIMIT past the end -> clamped.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, score INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d')").unwrap();
    e.execute_text(4, "INSERT INTO r (rid, score) VALUES (1,10),(2,20),(3,30),(4,40)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let names = |res: &RelationalSelectResult| -> Vec<String> {
        res.rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Text(s) => s.clone(),
                other => panic!("expected text, got {other:?}"),
            })
            .collect()
    };
    // Sorted by score ASC the join result names are [a, b, c, d].
    // OFFSET past the end -> empty.
    let beyond = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.rid ORDER BY r.score OFFSET 10",
        )
        .expect("offset past the end");
    assert_eq!(beyond.executed_target, DeviceTarget::Gpu(0));
    assert!(beyond.rows.is_empty(), "OFFSET past the end -> no rows");
    // LIMIT 0 -> empty.
    let zero = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.rid ORDER BY r.score LIMIT 0",
        )
        .expect("limit 0");
    assert!(zero.rows.is_empty(), "LIMIT 0 -> no rows");
    // OFFSET 2 + LIMIT 100 past the end -> clamped to the tail [c, d].
    let tail = e
        .execute_resident_expr_select_sql(
            "SELECT l.name, r.score FROM l JOIN r ON l.id = r.rid ORDER BY r.score LIMIT 100 OFFSET 2",
        )
        .expect("limit past the end");
    assert_eq!(
        names(&tail),
        vec!["c".to_string(), "d".to_string()],
        "LIMIT past the end clamps to the tail"
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

// ============================================================================================
// S7/V3 INDEPENDENT ADVERSARIAL AUDIT (commit 70758557): join result materialization on-device.
// Goal: break the device gather -- a MATCHED-row value-NULL emitting a placeholder, a pad emitting
// row-0's value, a type-narrowing/tagging error, or a LIMIT/OFFSET window divergence.
// ============================================================================================

// Helper: pull the single (col 0) value of a one-row resident SELECT -- the "truth" produced by the
// already-audited resident materialization path (S1), used as the expected value for the join gather.
#[cfg(test)]
fn audit_one_val(e: &Engine, sql: &str) -> SqlValue {
    let r = e.execute_resident_expr_select_sql(sql).expect("reference select");
    assert_eq!(r.rows.len(), 1, "reference select must return exactly one row: {sql}");
    r.rows[0][0].clone()
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_matched_row_value_null_every_type() {
    // HUNT #1: a MATCHED join row whose projected NON-KEY column is NULL must come back SqlValue::Null
    // for EVERY nullable type -- not the device placeholder (0/""/0-mantissa/zero-uuid/false). The
    // committed test only proves int4/text/numeric. Here: int2, int8, date, timestamp, uuid, bool,
    // numeric@scale4. The join KEY (id) is non-null; every value column carries a NULL on the matched row.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, s2 SMALLINT, s8 BIGINT, d DATE, ts TIMESTAMP, u UUID, b BOOLEAN, n4 NUMERIC(12,4))",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2)").unwrap();
    // rid=1: every value column NULL. rid=2: every value column a distinctive NON-null value.
    e.execute_text(
        4,
        "INSERT INTO r (rid, s2, s8, d, ts, u, b, n4) VALUES \
         (1, NULL, NULL, NULL, NULL, NULL, NULL, NULL), \
         (2, -12345, 9000000000, '2024-03-14', '2024-03-14 13:37:00', \
          'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', true, 1234.5678)",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // The non-null reference values (rid=2), from the audited resident path.
    let v_s2 = audit_one_val(&e, "SELECT s2 FROM r WHERE rid = 2");
    let v_s8 = audit_one_val(&e, "SELECT s8 FROM r WHERE rid = 2");
    let v_d = audit_one_val(&e, "SELECT d FROM r WHERE rid = 2");
    let v_ts = audit_one_val(&e, "SELECT ts FROM r WHERE rid = 2");
    let v_u = audit_one_val(&e, "SELECT u FROM r WHERE rid = 2");
    let v_b = audit_one_val(&e, "SELECT b FROM r WHERE rid = 2");
    let v_n4 = audit_one_val(&e, "SELECT n4 FROM r WHERE rid = 2");
    // sanity: the references are the real (non-Null) values and exercise the placeholder hazard.
    assert_eq!(v_s2, SqlValue::Int2(-12345));
    assert_eq!(v_b, SqlValue::Bool(true));
    assert!(matches!(v_u, SqlValue::Uuid(_)));
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.s2, r.s8, r.d, r.ts, r.u, r.b, r.n4 \
             FROM l JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("inner join projecting every nullable type");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        res.rows,
        vec![
            // rid=1 matched: EVERY value column is NULL (validity bitmap), NOT a placeholder.
            vec![
                SqlValue::Int4(1),
                SqlValue::Null, // s2 (placeholder would be Int2(0))
                SqlValue::Null, // s8 (placeholder Int8(0))
                SqlValue::Null, // d  (placeholder Date(0))
                SqlValue::Null, // ts (placeholder Timestamp(0))
                SqlValue::Null, // u  (placeholder Uuid([0;16]))
                SqlValue::Null, // b  (placeholder Bool(false))
                SqlValue::Null, // n4 (placeholder Numeric(0))
            ],
            // rid=2 matched: every value column the exact non-null value (type-exact narrow/tag/bytes).
            vec![SqlValue::Int4(2), v_s2, v_s8, v_d, v_ts, v_u, v_b, v_n4],
        ],
        "matched-row NULLs across all types must be SqlValue::Null, non-nulls type-exact"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_outer_pad_on_nonnullable_columns_all_types() {
    // HUNT #2 + #3: an OUTER pad must force NULL on a column that has NO validity bitmap (non-nullable),
    // independent of validity -- AND must not leak row-0's value (pads use placeholder index 0). Row 0 of
    // the padded relation holds DISTINCTIVE values for every type; the unmatched left rows must be NULL.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    // r has NO nullable columns (none of these ever hold a NULL -> no validity bitmap is built).
    e.execute_text(
        2,
        "CREATE TABLE r (rid INT, s2 SMALLINT, s8 BIGINT, d DATE, ts TIMESTAMP, u UUID, b BOOLEAN, n NUMERIC(10,2), name TEXT)",
    )
    .unwrap();
    // l has id 1,2,3. r has rid=1 (ROW 0, distinctive) and rid=2. id=3 is UNMATCHED -> a pad over r.
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2),(3)").unwrap();
    e.execute_text(
        4,
        "INSERT INTO r (rid, s2, s8, d, ts, u, b, n, name) VALUES \
         (1, 777, 123456789012, '2030-12-31', '2030-12-31 23:59:59', \
          '11111111-2222-3333-4444-555555555555', true, 42.42, 'ROW0'), \
         (2, 1, 1, '2000-01-01', '2000-01-01 00:00:00', \
          '00000000-0000-0000-0000-000000000001', false, 1.00, 'row2')",
    )
    .unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.s2, r.s8, r.d, r.ts, r.u, r.b, r.n, r.name \
             FROM l LEFT JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("left outer join, pad over a fully non-nullable relation");
    assert_eq!(res.rows.len(), 3);
    // id=3 row: every r column is a pad -> must be NULL, NOT row-0's distinctive value (777/'ROW0'/...).
    let pad_row = &res.rows[2];
    assert_eq!(pad_row[0], SqlValue::Int4(3), "left id survives");
    for (c, v) in pad_row.iter().enumerate().skip(1) {
        assert_eq!(
            *v,
            SqlValue::Null,
            "padded non-nullable column {c} must be NULL, not row-0's value (placeholder-0 leak)"
        );
    }
    // And the matched id=1 row really carries row-0's distinctive values (proves the gather isn't dead).
    assert_eq!(res.rows[0][1], SqlValue::Int2(777), "matched id=1 gets row-0 s2");
    assert_eq!(res.rows[0][8], SqlValue::Text("ROW0".to_string()), "matched id=1 gets row-0 name");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_empty_padded_side_right_full() {
    // HUNT #3: a relation whose row_count==0 appears as a PADDED side (RIGHT/FULL with an empty side).
    // gather_col must early-return all-NULL (no device read at index 0 into an empty payload) -- no panic.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b')").unwrap();
    // r is EMPTY.
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // LEFT JOIN with empty right -> both left rows survive, r columns NULL.
    let left = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.name, r.w FROM l LEFT JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("left join over an empty right relation");
    assert_eq!(
        left.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("a".to_string()), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Text("b".to_string()), SqlValue::Null],
        ],
        "empty padded side -> r.w all NULL, no panic/OOB"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_limit_offset_no_orderby_matches_join_order() {
    // HUNT #5: LIMIT/OFFSET WITHOUT ORDER BY must window the join result in JOIN ORDER (identity perm),
    // exactly as the old drain/truncate did. We make the join order deterministic (unique 1:1 keys, build
    // on the unique side) and verify the windowed slice is a contiguous slice of the full result.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, score INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e')").unwrap();
    e.execute_text(4, "INSERT INTO r (rid, score) VALUES (1,10),(2,20),(3,30),(4,40),(5,50)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let full = e
        .execute_resident_expr_select_sql("SELECT l.name FROM l JOIN r ON l.id = r.rid")
        .expect("full join, no window");
    let full_names: Vec<SqlValue> = full.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(full_names.len(), 5);
    // OFFSET 1 LIMIT 2 (no ORDER BY) -> the contiguous slice [1..3) of the join order.
    let win = e
        .execute_resident_expr_select_sql("SELECT l.name FROM l JOIN r ON l.id = r.rid LIMIT 2 OFFSET 1")
        .expect("windowed join, no order by");
    let win_names: Vec<SqlValue> = win.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(
        win_names,
        full_names[1..3].to_vec(),
        "LIMIT 2 OFFSET 1 with no ORDER BY must equal the contiguous join-order slice"
    );
    // OFFSET only (no LIMIT) -> the tail [2..].
    let tail = e
        .execute_resident_expr_select_sql("SELECT l.name FROM l JOIN r ON l.id = r.rid OFFSET 2")
        .expect("offset-only join");
    let tail_names: Vec<SqlValue> = tail.rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(tail_names, full_names[2..].to_vec(), "OFFSET 2, no LIMIT -> tail in join order");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_empty_result_no_matches() {
    // HUNT #9: no matches -> work_n == 0 -> the gather returns empty, transpose -> 0 rows, no panic.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w NUMERIC(10,2))").unwrap();
    e.execute_text(3, "INSERT INTO l (id, name) VALUES (1,'a'),(2,'b')").unwrap();
    e.execute_text(4, "INSERT INTO r (rid, w) VALUES (100, 1.00),(200, 2.00)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, l.name, r.w FROM l JOIN r ON l.id = r.rid ORDER BY l.id",
        )
        .expect("inner join with no matches");
    assert!(res.rows.is_empty(), "no matches -> empty result, no panic");
    // also with a window applied on top of empty.
    let res2 = e
        .execute_resident_expr_select_sql(
            "SELECT l.id FROM l JOIN r ON l.id = r.rid LIMIT 5 OFFSET 0",
        )
        .expect("inner join no matches + window");
    assert!(res2.rows.is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_nn_and_multiway_gather_right_rows() {
    // HUNT #8: N:N many-to-many + a 3-way chain. The carried index vectors must gather the RIGHT rows
    // from each side's device payload (text + numeric + null values), not misaligned values.
    let mut e = Engine::new_local();
    // N:N on an int key: l has key 1 twice, r has key 1 twice -> 4 result rows, each a distinct (l,r) pair.
    e.execute_text(1, "CREATE TABLE l (k INT, lv TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (k INT, rv NUMERIC(10,2))").unwrap();
    e.execute_text(3, "INSERT INTO l (k, lv) VALUES (1,'x'),(1,'y')").unwrap();
    e.execute_text(4, "INSERT INTO r (k, rv) VALUES (1, 1.10),(1, NULL)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let nn = e
        .execute_resident_expr_select_sql(
            "SELECT l.lv, r.rv FROM l JOIN r ON l.k = r.k ORDER BY l.lv, r.rv",
        )
        .expect("N:N int join");
    // 4 pairs: (x,1.10),(x,NULL),(y,1.10),(y,NULL). r.rv NULL must be Null from the validity bitmap.
    // ORDER BY r.rv places NULL last (ASC PG default). So per lv: [1.10, NULL].
    assert_eq!(
        nn.rows,
        vec![
            vec![SqlValue::Text("x".to_string()), SqlValue::Numeric(Decimal128::new(110, 2))],
            vec![SqlValue::Text("x".to_string()), SqlValue::Null],
            vec![SqlValue::Text("y".to_string()), SqlValue::Numeric(Decimal128::new(110, 2))],
            vec![SqlValue::Text("y".to_string()), SqlValue::Null],
        ],
        "N:N gather: each (l,r) pair's text+numeric (incl. a NULL numeric value) is correct"
    );
    // 3-way chain a JOIN b JOIN c. Carried indices into 3 payloads.
    e.execute_text(10, "CREATE TABLE a (aid INT, an TEXT)").unwrap();
    e.execute_text(11, "CREATE TABLE b (bid INT, bref INT, bn TEXT)").unwrap();
    e.execute_text(12, "CREATE TABLE c (cid INT, cn TEXT)").unwrap();
    e.execute_text(13, "INSERT INTO a (aid, an) VALUES (1,'a1'),(2,'a2')").unwrap();
    e.execute_text(14, "INSERT INTO b (bid, bref, bn) VALUES (1,1,'b1'),(2,2,'b2')").unwrap();
    e.execute_text(15, "INSERT INTO c (cid, cn) VALUES (1,'c1'),(2,'c2')").unwrap();
    for t in ["a", "b", "c"] {
        if e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_none() {
            return;
        }
    }
    let threeway = e
        .execute_resident_expr_select_sql(
            "SELECT a.an, b.bn, c.cn FROM a JOIN b ON a.aid = b.bref JOIN c ON b.bid = c.cid ORDER BY a.an",
        )
        .expect("3-way chain join");
    assert_eq!(
        threeway.rows,
        vec![
            vec![
                SqlValue::Text("a1".to_string()),
                SqlValue::Text("b1".to_string()),
                SqlValue::Text("c1".to_string())
            ],
            vec![
                SqlValue::Text("a2".to_string()),
                SqlValue::Text("b2".to_string()),
                SqlValue::Text("c2".to_string())
            ],
        ],
        "3-way chain: each side's text gathered from its own payload at the carried row"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_using_natural_and_star_gather() {
    // HUNT #7: USING coalesced column (mapped to rel 0) + bare `*` gather correctly.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT, lname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (id INT, rscore INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id, lname) VALUES (1,'a'),(2,'b'),(3,'c')").unwrap();
    e.execute_text(4, "INSERT INTO r (id, rscore) VALUES (1,10),(2,20)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // USING (id): the coalesced id from rel0, then l.lname, then r.rscore.
    let star = e
        .execute_resident_expr_select_sql("SELECT * FROM l JOIN r USING (id) ORDER BY id")
        .expect("USING join star");
    assert_eq!(
        star.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("a".to_string()), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Text("b".to_string()), SqlValue::Int4(20)],
        ],
        "USING star: coalesced id (rel0) + l.lname + r.rscore gathered from device"
    );
    // unqualified id reference resolves to the left copy.
    let bare = e
        .execute_resident_expr_select_sql("SELECT id, lname, rscore FROM l JOIN r USING (id) ORDER BY id")
        .expect("USING explicit columns");
    assert_eq!(bare.rows, star.rows, "explicit list equals star for USING");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_nonvacuity_placeholder_would_leak_without_override() {
    // NON-VACUITY PROOF: the device DOES store a 0/false/""/zero placeholder for a NULL cell. This test
    // asserts the placeholder values directly via a deliberately-WRONG expectation -- it MUST PANIC,
    // proving the SqlValue::Null in the real test is the validity override doing real work (not that the
    // device happens to be empty/absent). If this test ever PASSES, the placeholder is leaking == bug.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, b BOOLEAN, name TEXT, s2 SMALLINT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1)").unwrap();
    e.execute_text(4, "INSERT INTO r (rid, b, name, s2) VALUES (1, NULL, NULL, NULL)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT r.b, r.name, r.s2 FROM l JOIN r ON l.id = r.rid")
        .expect("join");
    let row = &res.rows[0];
    // The CORRECT result is all-Null. The placeholder (the WRONG result) would be Bool(false)/Text("")/Int2(0).
    let placeholder = vec![SqlValue::Bool(false), SqlValue::Text(String::new()), SqlValue::Int2(0)];
    let result = std::panic::catch_unwind(|| {
        assert_eq!(*row, placeholder, "if this matched, the placeholder LEAKED");
    });
    assert!(
        result.is_err(),
        "NON-VACUITY: the result must NOT equal the device placeholder (got {row:?}) -- \
         the validity override is load-bearing"
    );
    // Belt-and-suspenders: it IS all-Null.
    assert_eq!(*row, vec![SqlValue::Null, SqlValue::Null, SqlValue::Null]);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_join_order_by_nullable_value_with_window() {
    // HUNT #5 + #1 cross: ORDER BY a device-gathered NULLABLE value column on the join, then window it.
    // The gathered NULL must (a) render Null and (b) sort to PG default (NULLs last ASC), and the window
    // must slice the SORTED order (not join order). A divergence would silently reorder/mis-window.
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE l (id INT)").unwrap();
    e.execute_text(2, "CREATE TABLE r (rid INT, w INT)").unwrap();
    e.execute_text(3, "INSERT INTO l (id) VALUES (1),(2),(3),(4)").unwrap();
    // w values: 30, NULL, 10, 20 -> sorted ASC: 10(id3),20(id4),30(id1),NULL(id2).
    e.execute_text(4, "INSERT INTO r (rid, w) VALUES (1,30),(2,NULL),(3,10),(4,20)").unwrap();
    let mut ok = true;
    for t in ["l", "r"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let sorted = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w",
        )
        .expect("order by nullable value");
    assert_eq!(
        sorted.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int4(10)],
            vec![SqlValue::Int4(4), SqlValue::Int4(20)],
            vec![SqlValue::Int4(1), SqlValue::Int4(30)],
            vec![SqlValue::Int4(2), SqlValue::Null], // NULL last (ASC default)
        ],
        "ORDER BY r.w ASC: NULL sorts last, value device-gathered"
    );
    // Window the sorted order: OFFSET 1 LIMIT 2 -> rows [20(id4), 30(id1)].
    let win = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w LIMIT 2 OFFSET 1",
        )
        .expect("order by + window");
    assert_eq!(
        win.rows,
        vec![
            vec![SqlValue::Int4(4), SqlValue::Int4(20)],
            vec![SqlValue::Int4(1), SqlValue::Int4(30)],
        ],
        "window slices the SORTED order, not join order"
    );
    // NULLS FIRST explicitly: NULL should now be the first row; window OFFSET 0 LIMIT 1 -> the NULL row.
    let nf = e
        .execute_resident_expr_select_sql(
            "SELECT l.id, r.w FROM l JOIN r ON l.id = r.rid ORDER BY r.w NULLS FIRST LIMIT 1",
        )
        .expect("nulls first + limit 1");
    assert_eq!(
        nf.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Null]],
        "NULLS FIRST + LIMIT 1 -> the NULL row first"
    );
}

// ============================================================================
// S5/V1a adversarial audit: join text/numeric/uuid KEY values from the DEVICE
// payload (not host_rows). The charter claim is BEHAVIOR-PRESERVING vs the old
// host_rows gather. These tests target byte-order/value exactness, negatives,
// multi-way carried-index gathers, N:N, empty inputs, and the NULL gate.
// ============================================================================

/// HUNT #1: UUID byte-order / value exactness AND that the joined-on uuid VALUE is projected
/// byte-identically. The matched uuids are ASYMMETRIC/non-palindromic byte patterns, so a byte-order
/// bug in `project_i128 -> to_le_bytes()` (vs the old `SqlValue::Uuid(bytes)` host read) would EITHER
/// mismatch the join (wrong/missing pairs) OR project a byte-swapped uuid. The existing test only
/// projected `gname` (another column), so a uuid-value byte-swap would have been invisible there.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_uuid_byteorder_value_exactness() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE gd (gid UUID, gname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE ud (uid INT, gid UUID)").unwrap();
    // Deliberately asymmetric byte patterns: a byte-swap would change the value AND break matching.
    let a = "0102030405060708090a0b0c0d0e0f10"; // bare 32-hex form
    let b = "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0"; // descending, high bit set in byte 0
    let c = "00112233-4455-6677-8899-aabbccddeeff";
    let unmatched = "deadbeef-0000-1111-2222-333344445555";
    let a_canon = gpu_db_sql::uuid::format_uuid(&gpu_db_sql::uuid::parse_uuid(a).unwrap());
    e.execute_text(3, &format!("INSERT INTO gd (gid, gname) VALUES ('{a}','A'),('{b}','B'),('{c}','C')")).unwrap();
    e.execute_text(4, &format!("INSERT INTO ud (uid, gid) VALUES (1,'{a}'),(2,'{b}'),(3,'{c}'),(4,'{unmatched}')")).unwrap();
    let mut ok = true;
    for t in ["gd", "ud"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // Project BOTH the uuid join key (from each side) AND a tag, so a byte-order bug surfaces in the
    // projected value, not just the match set.
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ud.uid, ud.gid, gd.gid, gd.gname FROM ud JOIN gd ON ud.gid = gd.gid ORDER BY ud.uid",
        )
        .expect("uuid-key join projecting the uuid value");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    // Exactly 3 matches (the deadbeef user drops). Both projected uuids must be the canonical value.
    let rows: Vec<(i32, String, String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let uid = match &r[0] {
                SqlValue::Int4(v) => *v,
                o => panic!("uid {o:?}"),
            };
            let u_gid = match &r[1] {
                SqlValue::Uuid(bytes) => gpu_db_sql::uuid::format_uuid(bytes),
                o => panic!("ud.gid not uuid: {o:?}"),
            };
            let g_gid = match &r[2] {
                SqlValue::Uuid(bytes) => gpu_db_sql::uuid::format_uuid(bytes),
                o => panic!("gd.gid not uuid: {o:?}"),
            };
            let gname = match &r[3] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("gname {o:?}"),
            };
            (uid, u_gid, g_gid, gname)
        })
        .collect();
    assert_eq!(rows.len(), 3, "exactly the 3 matched uuids (deadbeef drops)");
    // Row 1: uuid `a` -> both projected uuids equal `a`'s canonical form (NOT byte-swapped).
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[0].1, a_canon, "ud.gid value exact (no byte swap)");
    assert_eq!(rows[0].2, a_canon, "gd.gid value exact (no byte swap)");
    assert_eq!(rows[0].3, "A");
    // Row 2: uuid `b` (descending, high-bit byte0) round-trips exactly + matched the right group.
    assert_eq!(rows[1].0, 2);
    assert_eq!(rows[1].1, "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0");
    assert_eq!(rows[1].2, "fffefdfc-fbfa-f9f8-f7f6-f5f4f3f2f1f0");
    assert_eq!(rows[1].3, "B");
    // Row 3: uuid `c`.
    assert_eq!(rows[2].0, 3);
    assert_eq!(rows[2].1, "00112233-4455-6677-8899-aabbccddeeff");
    assert_eq!(rows[2].3, "C");
}

/// HUNT #2: NUMERIC mantissa exactness across scales + NEGATIVES + large i128 magnitudes. The old host
/// path used `value.mantissa.to_le_bytes()`; the device path projects the i128 then `.to_le_bytes()`.
/// A sign-extension/limb bug in `project_i128` would surface as a missing/extra join pair OR a
/// byte-swapped projected value. Existing tests used only small POSITIVE mantissas (100.00/200.50).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_numeric_mantissa_exactness_negatives_and_large() {
    let mut e = Engine::new_local();
    // scale 0. NEGATIVES are the high-limb probe: a -1 mantissa is 0xFF..FF across ALL 16 bytes
    // (including the HIGH 64-bit limb); a dropped/zeroed high limb in project_i128 would turn it into a
    // large POSITIVE value and break the match. Also a beyond-i32 positive (3e9) crosses the 32-bit line.
    e.execute_text(1, "CREATE TABLE t0 (k NUMERIC(30,0), name TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE a0 (aid INT, k NUMERIC(30,0))").unwrap();
    let big = "3000000000"; // > i32::MAX (2.1e9): a 32-bit-limb bug would corrupt it
    e.execute_text(3, &format!("INSERT INTO t0 (k, name) VALUES (-1,'neg1'),({big},'big'),(-999999999,'negbil')")).unwrap();
    e.execute_text(4, &format!("INSERT INTO a0 (aid, k) VALUES (1,-1),(2,{big}),(3,-999999999),(4,777)")).unwrap();
    // high scale + negative fraction
    e.execute_text(5, "CREATE TABLE th (k NUMERIC(20,6), name TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE ah (aid INT, k NUMERIC(20,6))").unwrap();
    e.execute_text(7, "INSERT INTO th (k, name) VALUES (-12.345678,'negfrac'),(0.000001,'tiny')").unwrap();
    e.execute_text(8, "INSERT INTO ah (aid, k) VALUES (1,-12.345678),(2,0.000001),(3,5.000000)").unwrap();
    let mut ok = true;
    for t in ["t0", "a0", "th", "ah"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let collect = |res: &RelationalSelectResult| -> Vec<(i32, String, String)> {
        let mut v: Vec<(i32, String, String)> = res
            .rows
            .iter()
            .map(|r| {
                let aid = match &r[0] {
                    SqlValue::Int4(v) => *v,
                    o => panic!("aid {o:?}"),
                };
                let k = match &r[1] {
                    SqlValue::Numeric(d) => d.mantissa.to_string(),
                    o => panic!("k not numeric: {o:?}"),
                };
                let name = match &r[2] {
                    SqlValue::Text(t) => t.clone(),
                    o => panic!("name {o:?}"),
                };
                (aid, k, name)
            })
            .collect();
        v.sort();
        v
    };
    let r0 = e
        .execute_resident_expr_select_sql(
            "SELECT a0.aid, a0.k, t0.name FROM a0 JOIN t0 ON a0.k = t0.k",
        )
        .expect("scale-0 negative/large numeric join");
    assert_eq!(r0.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        collect(&r0),
        vec![
            (1, "-1".to_string(), "neg1".to_string()),
            (2, big.to_string(), "big".to_string()),
            (3, "-999999999".to_string(), "negbil".to_string()),
        ],
        "scale-0 negatives + a near-i128::MAX mantissa match exactly; 777 (unmatched) drops"
    );
    let rh = e
        .execute_resident_expr_select_sql(
            "SELECT ah.aid, ah.k, th.name FROM ah JOIN th ON ah.k = th.k",
        )
        .expect("high-scale numeric join");
    assert_eq!(
        collect(&rh),
        vec![
            (1, "-12345678".to_string(), "negfrac".to_string()),
            (2, "1".to_string(), "tiny".to_string()),
        ],
        "high-scale negative fraction + tiny value match exactly; 5.0 (unmatched) drops"
    );
}

/// HUNT #3: a numeric/uuid (b128) key used in a LATER step of a 3-way join. The accumulated side's key
/// is gathered at CARRIED indices that came from a prior step -- those indices must index the
/// relation's OWN payload, not the prior result. A wrong base would gather the wrong key -> wrong matches.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_b128_key_in_later_multiway_step() {
    let mut e = Engine::new_local();
    // r0 (int pk) -> r1 (int fk to r0, uuid u) -> r2 (uuid u). The uuid join is the SECOND step, joining
    // the ACCUMULATED (r0,r1) on r1.u against r2.u. r1.u is gathered at carried r1 indices.
    e.execute_text(1, "CREATE TABLE r0 (id INT, tag TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE r1 (id INT, u UUID)").unwrap();
    e.execute_text(3, "CREATE TABLE r2 (u UUID, label TEXT)").unwrap();
    let ux = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    let uy = "12345678-9abc-def0-1234-567890abcdef";
    let uz = "00000000-0000-0000-0000-000000000001";
    // r0 rows 1..3; r1 maps id->uuid (id1->ux, id2->uy, id3->uz); r2 has ux,uy only.
    e.execute_text(4, "INSERT INTO r0 (id, tag) VALUES (1,'one'),(2,'two'),(3,'three')").unwrap();
    e.execute_text(5, &format!("INSERT INTO r1 (id, u) VALUES (1,'{ux}'),(2,'{uy}'),(3,'{uz}')")).unwrap();
    e.execute_text(6, &format!("INSERT INTO r2 (u, label) VALUES ('{ux}','X'),('{uy}','Y')")).unwrap();
    let mut ok = true;
    for t in ["r0", "r1", "r2"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT r0.tag, r2.label, r1.u FROM r0 JOIN r1 ON r0.id = r1.id JOIN r2 ON r1.u = r2.u ORDER BY r0.tag",
        )
        .expect("3-way: int step then uuid step on the accumulated side");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let rows: Vec<(String, String, String)> = res
        .rows
        .iter()
        .map(|r| {
            let tag = match &r[0] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("tag {o:?}"),
            };
            let label = match &r[1] {
                SqlValue::Text(t) => t.clone(),
                o => panic!("label {o:?}"),
            };
            let u = match &r[2] {
                SqlValue::Uuid(b) => gpu_db_sql::uuid::format_uuid(b),
                o => panic!("u {o:?}"),
            };
            (tag, label, u)
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            ("one".to_string(), "X".to_string(), ux.to_string()),
            ("two".to_string(), "Y".to_string(), uy.to_string()),
        ],
        "id3->uz has no r2 match and drops; the carried r1.u gathers the RIGHT uuid per accumulated tuple"
    );
}

/// HUNT #5: empty `abs` early return on a b128/text step. A per-side WHERE filters one side to ZERO
/// rows before a uuid-key join -> the `if abs.is_empty()` path must yield an empty result, no panic.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_empty_side_before_b128_and_text_step() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE gu (gid UUID, gname TEXT)").unwrap();
    e.execute_text(2, "CREATE TABLE uu (uid INT, gid UUID)").unwrap();
    let g = "11112222-3333-4444-5555-666677778888";
    e.execute_text(3, &format!("INSERT INTO gu (gid, gname) VALUES ('{g}','g')")).unwrap();
    e.execute_text(4, &format!("INSERT INTO uu (uid, gid) VALUES (1,'{g}'),(2,'{g}')")).unwrap();
    e.execute_text(5, "CREATE TABLE ls (lid INT, t TEXT)").unwrap();
    e.execute_text(6, "CREATE TABLE rs (rid INT, t TEXT)").unwrap();
    e.execute_text(7, "INSERT INTO ls (lid, t) VALUES (1,'x'),(2,'y')").unwrap();
    e.execute_text(8, "INSERT INTO rs (rid, t) VALUES (10,'x')").unwrap();
    let mut ok = true;
    for t in ["gu", "uu", "ls", "rs"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    // WHERE filters uu to empty (no uid > 100) before the uuid join.
    let uuid_empty = e
        .execute_resident_expr_select_sql(
            "SELECT uu.uid, gu.gname FROM uu JOIN gu ON uu.gid = gu.gid WHERE uu.uid > 100",
        )
        .expect("empty uuid side must not panic");
    assert!(uuid_empty.rows.is_empty(), "filtered-to-empty uuid side -> empty result");
    // WHERE filters rs to empty before the text join.
    let text_empty = e
        .execute_resident_expr_select_sql(
            "SELECT ls.lid, rs.rid FROM ls JOIN rs ON ls.t = rs.t WHERE rs.rid > 100",
        )
        .expect("empty text side must not panic");
    assert!(text_empty.rows.is_empty(), "filtered-to-empty text side -> empty result");
    // A NON-empty text match through the device gather (so this test also covers the text-key gather
    // correctness, not just the empty path): 'x' matches lid 1 -> rid 10.
    let text_match = e
        .execute_resident_expr_select_sql("SELECT ls.lid, rs.rid FROM ls JOIN rs ON ls.t = rs.t")
        .expect("text-key match through the device gather");
    let pairs: Vec<(i32, i32)> = text_match
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (SqlValue::Int4(a), SqlValue::Int4(b)) => (*a, *b),
            o => panic!("{o:?}"),
        })
        .collect();
    assert_eq!(pairs, vec![(1, 10)], "text key 'x' matches lid 1 -> rid 10 (device-gathered keys)");
}

/// HUNT #4 + #6: N:N numeric/uuid AND the NULL gate together. Both sides duplicate the numeric key; one
/// side ALSO has a NULL-key row. The NULL row must match nothing (excluded by the host gate BEFORE the
/// device gather) and the cross product of the non-NULL key must be exact. Confirms the device gather is
/// never reached on a NULL index.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_nn_numeric_with_null_key_gate() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE ln (lid INT, amt NUMERIC(10,2))").unwrap();
    e.execute_text(2, "CREATE TABLE rn (rid INT, amt NUMERIC(10,2))").unwrap();
    // key 5.00 duplicated on both sides; a NULL-key row on each side must drop, NOT spuriously match.
    e.execute_text(3, "INSERT INTO ln (lid, amt) VALUES (1,5.00),(2,5.00),(3,NULL)").unwrap();
    e.execute_text(4, "INSERT INTO rn (rid, amt) VALUES (10,5.00),(11,5.00),(12,NULL)").unwrap();
    let mut ok = true;
    for t in ["ln", "rn"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql(
            "SELECT ln.lid, rn.rid FROM ln JOIN rn ON ln.amt = rn.amt",
        )
        .expect("N:N numeric with NULL keys");
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let mut pairs: Vec<(i32, i32)> = res
        .rows
        .iter()
        .map(|r| {
            let a = match &r[0] {
                SqlValue::Int4(v) => *v,
                o => panic!("{o:?}"),
            };
            let b = match &r[1] {
                SqlValue::Int4(v) => *v,
                o => panic!("{o:?}"),
            };
            (a, b)
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![(1, 10), (1, 11), (2, 10), (2, 11)],
        "5.00 cross product only; NULL=NULL is UNKNOWN -> the (3,_)/(12,_) NULL rows match nothing"
    );
}

/// CROSS-CHECK that the new device gather is BEHAVIOR-PRESERVING by comparing the same query/data the
/// way the diff claims: a uuid+numeric join whose RESULT (matches AND projected key values) is the
/// expected set computed from the host data. This is the non-vacuous "device == host bytes" proof for a
/// dataset with both an asymmetric uuid and a negative numeric in the SAME query, projected end-to-end.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn audit_s5_mixed_uuid_numeric_end_to_end() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE k (gid UUID, amt NUMERIC(12,3))").unwrap();
    e.execute_text(2, "CREATE TABLE p (gid UUID, tag TEXT)").unwrap();
    let u = "80706050-4030-2010-0fef-dfcfbfaf9f8f"; // high bit set, asymmetric
    e.execute_text(3, &format!("INSERT INTO k (gid, amt) VALUES ('{u}',-42.500)")).unwrap();
    e.execute_text(4, &format!("INSERT INTO p (gid, tag) VALUES ('{u}','hit'),('deadbeef-0000-0000-0000-000000000000','miss')")).unwrap();
    let mut ok = true;
    for t in ["k", "p"] {
        ok &= e.populate_relational_residency_snapshot(t).unwrap().device_memory_proof.is_some();
    }
    if !ok {
        return;
    }
    let res = e
        .execute_resident_expr_select_sql("SELECT k.gid, k.amt, p.tag FROM k JOIN p ON k.gid = p.gid")
        .expect("uuid join projecting uuid + negative numeric");
    assert_eq!(res.rows.len(), 1, "only the matching uuid pair");
    let row = &res.rows[0];
    match &row[0] {
        SqlValue::Uuid(b) => assert_eq!(gpu_db_sql::uuid::format_uuid(b), u, "uuid value exact"),
        o => panic!("gid {o:?}"),
    }
    match &row[1] {
        SqlValue::Numeric(d) => assert_eq!(d.mantissa, -42500, "negative numeric mantissa exact"),
        o => panic!("amt {o:?}"),
    }
    match &row[2] {
        SqlValue::Text(t) => assert_eq!(t, "hit"),
        o => panic!("tag {o:?}"),
    }
}
