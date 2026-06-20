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
    assert_sql_err_contains(&e, "SELECT a FROM t x, t y WHERE a > 0", "one FROM relation"); // join
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
