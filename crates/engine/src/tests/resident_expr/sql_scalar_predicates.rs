use crate::{Engine, ExecuteError, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_predicates() {
    // int8 (BIGINT) end to end on the general GPU executor from SQL text (the type matrix, doc 19):
    // scalar comparison, column-vs-column with values ABOVE i32::MAX (proving genuine 64-bit), the Ne
    // operator, and both int8 + int4 projection. Plus the unsupported-int8-shape hard errors.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (a INT, big BIGINT, big2 BIGINT, small BIGINT)",
    )
    .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX (2_147_483_647)
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {}, {}, {i})", BASE + i, BASE + (N - 1 - i)));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, big, big2, small) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // (1) int8 scalar comparison + int8 projection: small[i]=i, small > 400 => [401, 600).
    let scalar = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small > 400")
        .expect("int8 scalar comparison on GPU");
    let scalar_expected: Vec<Vec<SqlValue>> = (401..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        scalar.rows, scalar_expected,
        "small > 400 => small=i for i in [401, 600)"
    );
    assert_eq!(scalar.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(scalar.fallback_reason, None);

    // (2) int8 column-vs-column + 64-bit + i64 projection: big[i]=BASE+i, big2[i]=BASE+(N-1-i);
    // big < big2 <=> i < N-1-i <=> [0, 300). Projected big = BASE+i (values above i32::MAX).
    let cols = e
        .execute_resident_expr_select_sql("SELECT big FROM t WHERE big < big2")
        .expect("int8 col-vs-col on GPU");
    let cols_expected: Vec<Vec<SqlValue>> =
        (0..300).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(
        cols.rows, cols_expected,
        "big < big2 => big=BASE+i for i in [0, 300) (64-bit values)"
    );

    // (3) int8 Ne predicate + int4 projection (the projection type is independent of the predicate
    // type): small <> 300 => every row but i=300, projecting the int4 column a.
    let ne = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE small <> 300")
        .expect("int8 Ne on GPU");
    let mut ne_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![SqlValue::Int4(i)]).collect();
    ne_expected.extend((301..N as i32).map(|i| vec![SqlValue::Int4(i)]));
    assert_eq!(
        ne.rows, ne_expected,
        "small <> 300 => all rows but i=300, projecting int4 a"
    );

    // (4) A MIXED int4/int8 comparison is a HARD error (never a silent mis-answer). int8 arithmetic
    // is now supported — see gpu_execute_resident_expr_select_sql_runs_int8_arithmetic.
    assert!(
        e.execute_resident_expr_select_sql("SELECT big FROM t WHERE a < big")
            .is_err(),
        "mixed int4/int8 comparison -> hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_arithmetic() {
    // int8 (BIGINT) ARITHMETIC on the general GPU executor (the type matrix, doc 19): the i64 buffer
    // VM evaluates int8 arith trees (add/sub/mul, col-vs-col + scalar) with values ABOVE i32::MAX.
    // Closed-form oracle: a[i]=BASE+i, b[i]=BASE, c[i]=2*BASE+300, small[i]=i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT, small BIGINT)",
    )
    .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({}, {BASE}, {}, {i})", BASE + i, 2 * BASE + 300));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, b, c, small) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // col-vs-col ADD: a + b > c <=> (2*BASE+i) > (2*BASE+300) <=> i > 300 => [301, 600). 64-bit, no
    // overflow (2*BASE ~ 8e9 < i64::MAX). Projects a = BASE+i.
    let added = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE a + b > c")
        .expect("int8 a+b>c on GPU");
    let added_expected: Vec<Vec<SqlValue>> =
        (301..N).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(
        added.rows, added_expected,
        "a+b>c => a=BASE+i for i in [301, 600)"
    );
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(added.fallback_reason, None);

    // SUB then scalar compare: a - b > 200 <=> i > 200 => [201, 600). Projects small = i.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE a - b > 200")
        .expect("int8 a-b>200 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> = (201..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        subbed.rows, subbed_expected,
        "a-b>200 => small=i for i in [201, 600)"
    );

    // scalar MUL: small * 2 > 800 <=> i > 400 => [401, 600). Projects small = i.
    let scaled = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small * 2 > 800")
        .expect("int8 small*2>800 on GPU");
    let scaled_expected: Vec<Vec<SqlValue>> = (401..N).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        scaled.rows, scaled_expected,
        "small*2>800 => small=i for i in [401, 600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_raises_int8_integer_out_of_range_on_overflow() {
    // int8 arithmetic overflow raises Postgres `integer out of range` (the i64 buffer VM's checked mul
    // via mul.hi vs the sign-extension of mul.lo). Construction oracle (the int64 boundary):
    // 3037000500^2 = 9223372037000250000 > i64::MAX (9223372036854775807); 3037000499^2 =
    // 9223372030926249001 is in range.
    let overflow = run_int8_square_gt_zero(3_037_000_500);
    if let Some(result) = overflow {
        match result {
            Ok(_) => {
                panic!("a*a over a=3037000500 overflows int64 -> must raise bigint out of range")
            }
            Err(err) => assert!(
                err.to_string().contains("bigint out of range"),
                "int8 overflow must be PG's `bigint out of range` (not `integer`), got: {err}"
            ),
        }
    }

    // The largest in-range square does NOT error and returns the rows (a*a > 0 for both rows).
    if let Some(result) = run_int8_square_gt_zero(3_037_000_499) {
        let rows =
            result.expect("a*a at the int64 boundary 3037000499 is in range, must not error");
        assert_eq!(
            rows.rows,
            vec![vec![SqlValue::Int8(2)], vec![SqlValue::Int8(3_037_000_499)]],
            "a*a>0 in range -> both rows"
        );
    }
}

/// Build a single-BIGINT-column table `t(a) = [2, boundary]`, push a GPU snapshot, and run
/// `SELECT a FROM t WHERE a * a > 0` on the general executor. Returns `None` off-GPU.
fn run_int8_square_gt_zero(boundary: i64) -> Option<Result<RelationalSelectResult, ExecuteError>> {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a BIGINT)").unwrap();
    e.execute_text(2, &format!("INSERT INTO t (a) VALUES (2), ({boundary})"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    snapshot.device_memory_proof.as_ref()?;
    Some(e.execute_resident_expr_select_sql("SELECT a FROM t WHERE a * a > 0"))
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_boolean_predicates() {
    // REGRESSION (the audit's P0): int8 AND/OR predicates must run on the i64 VM, NOT silently route
    // to the i32 VM (which read int8 columns at the wrong 4-byte stride -> garbage rows). The same
    // routing gap also bypassed the mixed-int4/int8 guard, so a mixed AND must still hard-error.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (a INT, small BIGINT, big BIGINT, big2 BIGINT)",
    )
    .unwrap();

    const N: i64 = 600;
    const BASE: i64 = 4_000_000_000; // > i32::MAX
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}, {i}, {}, {})", BASE + i, BASE + (N - 1 - i)));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (a, small, big, big2) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // int8 AND (i32-range literals over an int8 column): small > 200 AND small < 400 => [201, 400).
    // The buggy i32-VM route read `small` at a 4-byte stride and returned a garbled set.
    let and_rows = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small > 200 AND small < 400")
        .expect("int8 AND on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (201..400).map(|i| vec![SqlValue::Int8(i)]).collect();
    assert_eq!(
        and_rows.rows, and_expected,
        "small>200 AND small<400 => [201, 400)"
    );
    assert_eq!(and_rows.executed_target, DeviceTarget::Gpu(0));

    // int8 OR: small < 100 OR small > 500 => [0,100) U [501,600).
    let or_rows = e
        .execute_resident_expr_select_sql("SELECT small FROM t WHERE small < 100 OR small > 500")
        .expect("int8 OR on GPU");
    let mut or_expected: Vec<Vec<SqlValue>> = (0..100).map(|i| vec![SqlValue::Int8(i)]).collect();
    or_expected.extend((501..N).map(|i| vec![SqlValue::Int8(i)]));
    assert_eq!(
        or_rows.rows, or_expected,
        "small<100 OR small>500 => [0,100) U [501,600)"
    );

    // 64-bit AND over two int8 columns (values above i32::MAX): big > 100 AND big2 > 100 => all rows.
    // An i32-stride read of big/big2 would NOT yield all rows, so this pins the genuine 64-bit read.
    let big_and = e
        .execute_resident_expr_select_sql("SELECT big FROM t WHERE big > 100 AND big2 > 100")
        .expect("int8 64-bit AND on GPU");
    let big_and_expected: Vec<Vec<SqlValue>> =
        (0..N).map(|i| vec![SqlValue::Int8(BASE + i)]).collect();
    assert_eq!(
        big_and.rows, big_and_expected,
        "big>100 AND big2>100 => all rows (64-bit)"
    );

    // MIXED int4/int8 inside AND now RUNS at I32 (ADR-006 mixed-width groups: the int8 leaf is a
    // width-safe `LoadColumnI64` scalar compare) — it used to hard-error. `big` is all > i32::MAX
    // (and > 5), so the pin is row-exactness: a 4-byte mis-read could not return exactly a<3.
    let mixed = e
        .execute_resident_expr_select_sql("SELECT a FROM t WHERE big > 5 AND a < 3")
        .expect("mixed int4/int8 AND runs on the GPU (mixed-width group)");
    let mixed_expected: Vec<Vec<SqlValue>> = (0..3).map(|i| vec![SqlValue::Int4(i)]).collect();
    assert_eq!(
        mixed.rows, mixed_expected,
        "big>5 AND a<3 => a in {{0,1,2}} (all bigs exceed 5)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_comparisons() {
    // numeric (NUMERIC / i128) comparison + projection end-to-end from SQL (the type matrix, doc 19).
    // price[i] = i.50 (NUMERIC(10,2)), cost[i] = (N-1-i).50. Closed-form oracles; the i128 signedness
    // is proven separately in the execution-crate primitive test.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        // price = i.50, cost = (N-1-i).50, label = i
        values.push_str(&format!("({i}.50, {}.50, {i})", N - 1 - i));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let num = |whole: i64| SqlValue::Numeric(Decimal128::new(i128::from(whole) * 100 + 50, 2)); // whole.50

    // literal at the column scale: price > 10.50 => i+0.50 > 10.50 => i > 10 => [11, N). Project price.
    let gt = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.50")
        .expect("price > 10.50 on GPU");
    let gt_expected: Vec<Vec<SqlValue>> = (11..N).map(|i| vec![num(i)]).collect();
    assert_eq!(
        gt.rows, gt_expected,
        "price > 10.50 => i.50 for i in [11, 600)"
    );
    assert_eq!(gt.executed_target, DeviceTarget::Gpu(0));

    // integer literal coerced to numeric: price < 5 => i+0.50 < 5 => i <= 4 => [0, 5).
    let lt_int = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price < 5")
        .expect("price < 5 on GPU");
    let lt_int_expected: Vec<Vec<SqlValue>> = (0..5).map(|i| vec![num(i)]).collect();
    assert_eq!(
        lt_int.rows, lt_int_expected,
        "price < 5 (int coerced) => [0, 5)"
    );

    // lower-scale literal rescales UP exactly: price > 10.5 (scale 1) == price > 10.50 => [11, N).
    let gt_low = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.5")
        .expect("price > 10.5 on GPU");
    assert_eq!(
        gt_low.rows, gt_expected,
        "price > 10.5 (scale 1) == price > 10.50"
    );

    // trailing zeros are insignificant: price > 10.500 (written scale 3) == price > 10.50 => [11, N).
    let gt_trailing = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.500")
        .expect("price > 10.500 on GPU");
    assert_eq!(
        gt_trailing.rows, gt_expected,
        "price > 10.500 (trailing zeros) == price > 10.50"
    );

    // col-vs-col, equal scale: price < cost => i+0.50 < (N-1-i)+0.50 => 2i < N-1 => [0, 300).
    let cols = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price < cost")
        .expect("price < cost on GPU");
    let cols_expected: Vec<Vec<SqlValue>> = (0..300).map(|i| vec![num(i)]).collect();
    assert_eq!(cols.rows, cols_expected, "price < cost => [0, 300)");

    // a literal FINER than the column now rescales the column UP (cross-scale): price > 10.555
    // (scale 3) <=> i+0.50 > 10.555 <=> i >= 11 => [11, N).
    let finer = e
        .execute_resident_expr_select_sql("SELECT price FROM t WHERE price > 10.555")
        .expect("price > 10.555 (cross-scale) on GPU");
    let finer_expected: Vec<Vec<SqlValue>> = (11..N).map(|i| vec![num(i)]).collect();
    assert_eq!(
        finer.rows, finer_expected,
        "price > 10.555 (finer literal, cross-scale) => [11, N)"
    );

    // REJECTION — a mixed numeric vs an int4 column is a hard error, never wrong rows.
    assert!(
        e.execute_resident_expr_select_sql("SELECT price FROM t WHERE price > label")
            .is_err(),
        "mixed numeric/int4 column => hard error"
    );
    // numeric AND/OR now lowers via the i128 mask VM: price > 1.00 AND price < 100.00 (price = i.50)
    // <=> 0 < i < 100 => [1, 100). Project price.
    let and = e
        .execute_resident_expr_select_sql(
            "SELECT price FROM t WHERE price > 1.00 AND price < 100.00",
        )
        .expect("price>1.00 AND price<100.00 on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (1..100).map(|i| vec![num(i)]).collect();
    assert_eq!(
        and.rows, and_expected,
        "price>1.00 AND price<100.00 => price for i in [1, 100)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_arithmetic() {
    // numeric (i128) CHECKED add/sub arithmetic end-to-end from SQL (the type matrix, doc 19).
    // price[i]=i.50, cost[i]=i.25 (NUMERIC(10,2)), label[i]=i. Closed-form; mul + mixed are rejected.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.25, {i})"));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // col+col add then scalar compare: price+cost = 2i+0.75; > 100 => 200i+75 > 10000 => i >= 50.
    let added = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > 100")
        .expect("price+cost>100 on GPU");
    let added_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        added.rows, added_expected,
        "price+cost>100 => label in [50, 600)"
    );
    assert_eq!(added.executed_target, DeviceTarget::Gpu(0));

    // scalar sub then compare: price-5 = (i-5)+0.50; > 100 => 100i-450 > 10000 => i >= 105.
    let subbed = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price - 5 > 100")
        .expect("price-5>100 on GPU");
    let subbed_expected: Vec<Vec<SqlValue>> =
        (105..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        subbed.rows, subbed_expected,
        "price-5>100 => label in [105, 600)"
    );

    // buffer-vs-buffer (arith on the left, column on the right): price+cost > price <=> cost > 0 => all.
    let cmp_buffers = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price + cost > price")
        .expect("price+cost>price on GPU");
    let all_expected: Vec<Vec<SqlValue>> = (0..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cmp_buffers.rows, all_expected,
        "price+cost>price <=> cost>0 => all rows"
    );

    // REJECTION — a mixed numeric + int4 column in arithmetic is a hard error, never wrong rows.
    // (integer-literal multiply is now supported — see ..._runs_numeric_multiply.)
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE price + label > 100")
            .is_err(),
        "mixed numeric + int4 column in arithmetic => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_multiply() {
    // numeric (i128) CHECKED multiply by an INTEGER literal end-to-end from SQL (the type matrix,
    // doc 19): price*2 = 2i+1.00 (mantissa 200i+100). Fractional + column*column multipliers (which
    // change the result scale) are rejected.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,2), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.25, {i})"));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // price * 2 > 100 : (200i+100) > 10000 => 200i > 9900 => i >= 50 => [50, N).
    let mul = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 2 > 100")
        .expect("price*2>100 on GPU");
    let mul_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(mul.rows, mul_expected, "price*2>100 => label in [50, 600)");
    assert_eq!(mul.executed_target, DeviceTarget::Gpu(0));

    // commutative: 2 * price > 100 is the same set.
    let mul_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 2 * price > 100")
        .expect("2*price>100 on GPU");
    assert_eq!(
        mul_left.rows, mul_expected,
        "2*price>100 == price*2>100 (commutative)"
    );

    // fractional-literal multiply: price*1.5 (result scale 2+1=3): (1500i+750) > 100000 => i >= 67.
    let frac = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * 1.5 > 100")
        .expect("price*1.5>100 on GPU");
    let frac_expected: Vec<Vec<SqlValue>> =
        (67..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        frac.rows, frac_expected,
        "price*1.5>100 => label in [67, 600)"
    );

    // column*column multiply: price*cost (result scale 2+2=4): (100i+50)(100i+25) > 1000000 => i >= 10.
    let cols = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > 100")
        .expect("price*cost>100 on GPU");
    let cols_expected: Vec<Vec<SqlValue>> =
        (10..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cols.rows, cols_expected,
        "price*cost>100 => label in [10, 600)"
    );

    // cross-scale arith-vs-arith: price*cost (scale 4) > price (scale 2): rescale price up, then
    // (100i+25) > 100 <=> i >= 1 => [1, N).
    let cross = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE price * cost > price")
        .expect("price*cost>price (cross-scale) on GPU");
    let cross_expected: Vec<Vec<SqlValue>> =
        (1..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cross.rows, cross_expected,
        "price*cost > price (cross-scale) => [1, 600)"
    );

    // REJECTION — a mixed numeric * int4 column is a hard error, never wrong rows.
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE price * label > 100")
            .is_err(),
        "mixed numeric * int4 column => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_cross_scale_numeric_comparisons() {
    // CROSS-SCALE numeric comparison (the type matrix, doc 19): operands of different scales are
    // rescaled UP to the common (max) scale on-device (mantissa * 10^k) before comparing.
    // p2 = i.50 (NUMERIC(10,2)), p4 = (2i).0000 (NUMERIC(10,4)), label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {}.0000, {i})", 2 * i));
    }
    e.execute_text(2, &format!("INSERT INTO t (p2, p4, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // cross-scale column-vs-column: p2 > p4 <=> i+0.50 > 2i <=> i < 0.5 => only row 0.
    let cols = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > p4")
        .expect("p2>p4 on GPU");
    assert_eq!(
        cols.rows,
        vec![vec![SqlValue::Int4(0)]],
        "p2>p4 (cross-scale col-vs-col) => row 0 only"
    );
    assert_eq!(cols.executed_target, DeviceTarget::Gpu(0));

    // cross-scale column-vs-FINER-literal: p2 > 1.555 <=> i+0.50 > 1.555 <=> i >= 2 => [2, N).
    let lit = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > 1.555")
        .expect("p2>1.555 on GPU");
    let lit_expected: Vec<Vec<SqlValue>> = (2..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        lit.rows, lit_expected,
        "p2>1.555 (finer literal) => [2, 600)"
    );

    // literal on the left: 1.555 < p2 is the same set.
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 1.555 < p2")
        .expect("1.555<p2 on GPU");
    assert_eq!(lit_left.rows, lit_expected, "1.555<p2 == p2>1.555");

    // same-scale still uses the resident peephole (regression): p2 > 10.50 => [11, N).
    let same = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 > 10.50")
        .expect("p2>10.50 on GPU");
    let same_expected: Vec<Vec<SqlValue>> =
        (11..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        same.rows, same_expected,
        "p2>10.50 (same scale) => [11, 600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_cross_scale_numeric_add_sub() {
    // CROSS-SCALE numeric ADD/SUB (the type matrix, doc 19): operands of different scales are rescaled
    // UP to the common (max) scale before the buffer add/sub. p2 = i.50 (NUMERIC(10,2)), p4 = (2i).2500
    // (NUMERIC(10,4)), label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (p2 NUMERIC(10,2), p4 NUMERIC(10,4), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {}.2500, {i})", 2 * i));
    }
    e.execute_text(2, &format!("INSERT INTO t (p2, p4, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // cross-scale ADD (col+col): p2 + p4 = 3i + 0.75 (scale 4); > 100 => 3i > 99.25 => i >= 34.
    let add = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 + p4 > 100")
        .expect("p2+p4>100 on GPU");
    let add_expected: Vec<Vec<SqlValue>> =
        (34..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        add.rows, add_expected,
        "p2+p4>100 (cross-scale add) => [34, 600)"
    );
    assert_eq!(add.executed_target, DeviceTarget::Gpu(0));

    // cross-scale SUB (col-col): p4 - p2 = i - 0.25 (scale 4); > 100 => i >= 101.
    let sub = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p4 - p2 > 100")
        .expect("p4-p2>100 on GPU");
    let sub_expected: Vec<Vec<SqlValue>> =
        (101..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        sub.rows, sub_expected,
        "p4-p2>100 (cross-scale sub) => [101, 600)"
    );

    // cross-scale add with a FINER literal: p2 + 0.0001 = i.5001 (scale 4); > 50.5 => i >= 50.
    let lit = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE p2 + 0.0001 > 50.5")
        .expect("p2+0.0001>50.5 on GPU");
    let lit_expected: Vec<Vec<SqlValue>> =
        (50..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        lit.rows, lit_expected,
        "p2 + 0.0001 (finer literal) > 50.5 => [50, 600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_and_or() {
    // numeric AND/OR (the type matrix, doc 19): each comparison -> a mask via the i128 VM, MaskBinary
    // combines, terminal compact. price = i.50 (NUMERIC(10,2)), cost = i.2500 (NUMERIC(10,4)).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (price NUMERIC(10,2), cost NUMERIC(10,4), label INT)",
    )
    .unwrap();

    const N: i64 = 600;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i}.50, {i}.2500, {i})"));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (price, cost, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // AND: price > 10.50 AND price < 100.50 <=> 10 < i < 100 => [11, 100).
    let and = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price > 10.50 AND price < 100.50",
        )
        .expect("AND on GPU");
    let and_expected: Vec<Vec<SqlValue>> = (11i64..100)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(
        and.rows, and_expected,
        "price>10.50 AND price<100.50 => [11, 100)"
    );
    assert_eq!(and.executed_target, DeviceTarget::Gpu(0));

    // OR: price < 5.50 OR price > 595.50 => [0,5) U [596, N).
    let or = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price < 5.50 OR price > 595.50",
        )
        .expect("OR on GPU");
    let or_expected: Vec<Vec<SqlValue>> = (0..5)
        .chain(596..N)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(
        or.rows, or_expected,
        "price<5.50 OR price>595.50 => [0,5) U [596,600)"
    );

    // CROSS-SCALE AND (price scale 2, cost scale 4 -- each comparison rescales independently):
    // price > 10.50 AND cost > 50.2500 <=> i>10 AND i>50 => [51, N).
    let cross = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE price > 10.50 AND cost > 50.2500",
        )
        .expect("cross-scale AND on GPU");
    let cross_expected: Vec<Vec<SqlValue>> =
        (51..N).map(|i| vec![SqlValue::Int4(i as i32)]).collect();
    assert_eq!(
        cross.rows, cross_expected,
        "price>10.50 AND cost>50.2500 (cross-scale) => [51, 600)"
    );

    // NESTED: (price > 10.50 AND price < 100.50) OR price > 595.50 => [11,100) U [596, N).
    let nested = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE (price > 10.50 AND price < 100.50) OR price > 595.50",
        )
        .expect("nested AND/OR on GPU");
    let nested_expected: Vec<Vec<SqlValue>> = (11..100)
        .chain(596..N)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(
        nested.rows, nested_expected,
        "(price>10.50 AND price<100.50) OR price>595.50 => [11,100) U [596,600)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_text_equality() {
    // Text equality on the general GPU executor (the type matrix, doc 19): byte-wise = / <>. The
    // residency builder pads the text offsets section to its required 8-byte alignment even after an
    // odd-sized int4 section.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)")
        .unwrap();
    let names = ["alice", "bob", "alice", "carol", "bob", "alice", "dave"];
    let mut values = String::new();
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{n}', {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (name, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // = 'alice' -> rows 0,2,5
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name = 'alice'")
        .expect("name = 'alice' on GPU");
    assert_eq!(
        eq.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(5)]
        ],
        "name = 'alice' => rows 0,2,5"
    );
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // <> 'alice' -> rows 1,3,4,6
    let ne = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name <> 'alice'")
        .expect("name <> 'alice' on GPU");
    assert_eq!(
        ne.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(6)]
        ],
        "name <> 'alice' => rows 1,3,4,6"
    );

    // literal on the LEFT (equality is symmetric): 'bob' = name -> rows 1,4
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 'bob' = name")
        .expect("'bob' = name on GPU");
    assert_eq!(
        lit_left.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(4)]],
        "'bob' = name => rows 1,4"
    );

    // no match
    let none = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name = 'zzz'")
        .expect("name = 'zzz' on GPU");
    assert!(none.rows.is_empty(), "name = 'zzz' matches nothing");

    // Text INEQUALITY now runs on-device (ADR-006: the lexicographic byte-compare kernel). `name < 'bob'`
    // = the three 'alice' rows (a < b byte-wise) -> labels 0,2,5, matching the host `str::cmp` order.
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name < 'bob'")
        .expect("name < 'bob' on GPU");
    assert_eq!(
        lt.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(5)]
        ],
        "name < 'bob' => the 'alice' rows 0,2,5 (byte-wise lexicographic)"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE name = label")
            .is_err(),
        "mixed text/int => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_text_like() {
    // Text LIKE on the general GPU executor (the type matrix, doc 19): general %/_ backtracking match.
    // The offsets section remains 8-byte aligned after the odd-sized int4 section. Includes the `\_`
    // escape vs a bare `_`.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (name TEXT, label INT)")
        .unwrap();
    // "a_b" stores a literal underscore; "axb" distinguishes the `_` wildcard from the `\_` escape.
    let names = ["alice", "alicia", "bob", "alfred", "carol", "a_b", "axb"];
    let mut values = String::new();
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{n}', {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (name, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |idx: &[i32]| -> Vec<Vec<SqlValue>> {
        idx.iter().map(|i| vec![SqlValue::Int4(*i)]).collect()
    };

    // prefix: 'al%' -> alice, alicia, alfred
    let prefix = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'al%'")
        .expect("LIKE 'al%' on GPU");
    assert_eq!(
        prefix.rows,
        labels(&[0, 1, 3]),
        "LIKE 'al%' => alice/alicia/alfred"
    );
    assert_eq!(prefix.executed_target, DeviceTarget::Gpu(0));

    // contains: '%i%' -> alice, alicia
    let contains = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE '%i%'")
        .expect("LIKE '%i%' on GPU");
    assert_eq!(contains.rows, labels(&[0, 1]), "LIKE '%i%' => alice/alicia");

    // single-char wildcard: 'a_b' -> a_b AND axb (the `_` matches any one char)
    let wild = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'a_b'")
        .expect("LIKE 'a_b' on GPU");
    assert_eq!(wild.rows, labels(&[5, 6]), "LIKE 'a_b' => a_b AND axb");

    // ESCAPED underscore: 'a\\_b' -> only the literal "a_b" (NOT axb)
    let escaped = e
        .execute_resident_expr_select_sql(r"SELECT label FROM t WHERE name LIKE 'a\_b'")
        .expect("LIKE 'a\\_b' on GPU");
    assert_eq!(
        escaped.rows,
        labels(&[5]),
        "LIKE 'a\\_b' => only the literal a_b"
    );

    // exact (no wildcards) behaves like equality
    let exact = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE 'bob'")
        .expect("LIKE 'bob' on GPU");
    assert_eq!(exact.rows, labels(&[2]), "LIKE 'bob' => bob");

    // '%' matches every row
    let all = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE name LIKE '%'")
        .expect("LIKE '%' on GPU");
    assert_eq!(
        all.rows,
        labels(&[0, 1, 2, 3, 4, 5, 6]),
        "LIKE '%' => all rows"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_date_comparisons() {
    // Date comparison on the general GPU executor (the type matrix, doc 19): a `date` is i32 days
    // since 2000-01-01, reusing the int4 residency section + the I32 VM. hire_date[i] = 2024-01-(i+1),
    // label = i. The string literal is coerced to a day count at lowering (like PG).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (hire_date DATE, label INT)")
        .unwrap();
    const N: i64 = 30; // 2024-01-01 .. 2024-01-30
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('2024-01-{:02}', {i})", i + 1));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (hire_date, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '2024-01-15' -> the day i+1 == 15 -> row 14
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = '2024-01-15'")
        .expect("hire_date = date on GPU");
    assert_eq!(
        eq.rows,
        vec![vec![SqlValue::Int4(14)]],
        "= '2024-01-15' => row 14"
    );
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '2024-01-15' -> [15, 30)
    let gt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date > '2024-01-15'")
        .expect("hire_date > date on GPU");
    assert_eq!(gt.rows, labels(15..N), "> '2024-01-15' => [15, 30)");

    // < '2024-01-10' -> [0, 9)
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date < '2024-01-10'")
        .expect("hire_date < date on GPU");
    assert_eq!(lt.rows, labels(0..9), "< '2024-01-10' => [0, 9)");

    // literal on the LEFT: '2024-01-15' < hire_date -> [15, 30)
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE '2024-01-15' < hire_date")
        .expect("date < hire_date on GPU");
    assert_eq!(
        lit_left.rows,
        labels(15..N),
        "'2024-01-15' < hire_date => [15, 30)"
    );

    // projecting the DATE column yields SqlValue::Date(days)
    let proj = e
        .execute_resident_expr_select_sql("SELECT hire_date FROM t WHERE hire_date = '2024-01-15'")
        .expect("project date on GPU");
    let days = gpu_db_sql::datetime::parse_date("2024-01-15").expect("valid date");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Date(days)]],
        "projecting hire_date returns the date value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = 5")
            .is_err(),
        "date compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE hire_date = 'not-a-date'")
            .is_err(),
        "invalid date literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_timestamp_comparisons() {
    // Timestamp comparison on the general GPU executor (the type matrix, doc 19): a `timestamp` is
    // i64 microseconds since 2000-01-01, reusing the int8 section + the i64 compare kernels (the i64
    // micro literal exceeds the i32 VM scalar, so it uses expr_i64_compare_scalar_filter directly).
    // event_at[i] = 2024-01-15 i:00:00, created_at = constant 2024-01-15 12:00:00, label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE t (event_at TIMESTAMP, created_at TIMESTAMP, label INT)",
    )
    .unwrap();
    const N: i64 = 24; // hours 00:00:00 .. 23:00:00
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!(
            "('2024-01-15 {i:02}:00:00', '2024-01-15 12:00:00', {i})"
        ));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (event_at, created_at, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '2024-01-15 10:00:00' -> hour 10 -> row 10
    let eq = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at = '2024-01-15 10:00:00'",
        )
        .expect("event_at = ts on GPU");
    assert_eq!(
        eq.rows,
        vec![vec![SqlValue::Int4(10)]],
        "= 10:00:00 => row 10"
    );
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '2024-01-15 10:00:00' -> [11, 24)
    let gt = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at > '2024-01-15 10:00:00'",
        )
        .expect("event_at > ts on GPU");
    assert_eq!(gt.rows, labels(11..N), "> 10:00:00 => [11, 24)");

    // fractional / sub-hour literal: < '2024-01-15 05:30:00' -> hours 0..5 -> [0, 6)
    let lt = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE event_at < '2024-01-15 05:30:00'",
        )
        .expect("event_at < ts on GPU");
    assert_eq!(lt.rows, labels(0..6), "< 05:30:00 => [0, 6)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql(
            "SELECT label FROM t WHERE '2024-01-15 10:00:00' < event_at",
        )
        .expect("ts < event_at on GPU");
    assert_eq!(
        lit_left.rows,
        labels(11..N),
        "10:00:00 < event_at => [11, 24)"
    );

    // col-vs-col: event_at > created_at (noon) -> hours > 12 -> [13, 24)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at > created_at")
        .expect("event_at > created_at on GPU");
    assert_eq!(
        col_col.rows,
        labels(13..N),
        "event_at > created_at (noon) => [13, 24)"
    );

    // projecting the TIMESTAMP column yields SqlValue::Timestamp(micros)
    let proj = e
        .execute_resident_expr_select_sql(
            "SELECT event_at FROM t WHERE event_at = '2024-01-15 10:00:00'",
        )
        .expect("project timestamp on GPU");
    let micros = gpu_db_sql::datetime::parse_timestamp("2024-01-15 10:00:00").expect("valid ts");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Timestamp(micros)]],
        "projecting event_at returns the timestamp value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at = 5")
            .is_err(),
        "timestamp compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE event_at = 'not-a-ts'")
            .is_err(),
        "invalid timestamp literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_uuid_comparisons() {
    // UUID comparison on the general GPU executor (the type matrix, doc 19): a `uuid` is 16 raw bytes
    // in the i128 (16-byte) section, compared by an unsigned big-endian memcmp kernel (PG's uuid
    // order). id[i] = ...{i:02x} (last byte = i, so byte-wise ascending), peer = constant ...0a,
    // label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (id UUID, peer UUID, label INT)")
        .unwrap();
    const N: i64 = 20;
    let uuid_for = |i: i64| format!("00000000-0000-0000-0000-0000000000{i:02x}");
    let peer = uuid_for(10);
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{}', '{peer}', {i})", uuid_for(i)));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (id, peer, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = '...0a' -> last byte 10 -> row 10
    let eq = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id = '{}'",
            uuid_for(10)
        ))
        .expect("id = uuid on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(10)]], "= ...0a => row 10");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > '...0a' -> [11, 20)
    let gt = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id > '{}'",
            uuid_for(10)
        ))
        .expect("id > uuid on GPU");
    assert_eq!(gt.rows, labels(11..N), "> ...0a => [11, 20)");

    // < '...05' -> [0, 5)
    let lt = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id < '{}'",
            uuid_for(5)
        ))
        .expect("id < uuid on GPU");
    assert_eq!(lt.rows, labels(0..5), "< ...05 => [0, 5)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE '{}' < id",
            uuid_for(10)
        ))
        .expect("uuid < id on GPU");
    assert_eq!(lit_left.rows, labels(11..N), "...0a < id => [11, 20)");

    // col-vs-col: id > peer (constant ...0a) -> [11, 20)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE id > peer")
        .expect("id > peer on GPU");
    assert_eq!(col_col.rows, labels(11..N), "id > peer (...0a) => [11, 20)");

    // <> excludes only the equal row
    let ne = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT label FROM t WHERE id <> '{}'",
            uuid_for(10)
        ))
        .expect("id <> uuid on GPU");
    let mut expected_ne = labels(0..10);
    expected_ne.extend(labels(11..N));
    assert_eq!(ne.rows, expected_ne, "<> ...0a => all but row 10");

    // projecting the UUID column yields SqlValue::Uuid(bytes)
    let proj = e
        .execute_resident_expr_select_sql(&format!(
            "SELECT id FROM t WHERE id = '{}'",
            uuid_for(10)
        ))
        .expect("project uuid on GPU");
    let bytes = gpu_db_sql::uuid::parse_uuid(&uuid_for(10)).expect("valid uuid");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Uuid(bytes)]],
        "projecting id returns the uuid value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE id = 5")
            .is_err(),
        "uuid compared to an integer => hard error"
    );
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE id = 'not-a-uuid'")
            .is_err(),
        "invalid uuid literal => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int2_comparisons() {
    // smallint comparison on the general GPU executor (the type matrix, doc 19): a `smallint` is
    // stored WIDENED to i32 in the int4 section, so it reuses the i32 compare VM. sz[i] = i - 10
    // (so -10..9, exercising negatives + sign extension), peer = 0, label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (sz SMALLINT, peer SMALLINT, label INT)")
        .unwrap();
    const N: i64 = 20;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({}, 0, {i})", i - 10));
    }
    e.execute_text(
        2,
        &format!("INSERT INTO t (sz, peer, label) VALUES {values}"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    let labels = |range: std::ops::Range<i64>| -> Vec<Vec<SqlValue>> {
        range.map(|i| vec![SqlValue::Int4(i as i32)]).collect()
    };

    // = 0 -> i-10 = 0 -> row 10
    let eq = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz = 0")
        .expect("sz = 0 on GPU");
    assert_eq!(eq.rows, vec![vec![SqlValue::Int4(10)]], "= 0 => row 10");
    assert_eq!(eq.executed_target, DeviceTarget::Gpu(0));

    // > 0 -> [11, 20)
    let gt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz > 0")
        .expect("sz > 0 on GPU");
    assert_eq!(gt.rows, labels(11..N), "> 0 => [11, 20)");

    // < -5 -> i < 5 -> [0, 5) (NEGATIVE literal + values -> sign extension is correct)
    let lt = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz < -5")
        .expect("sz < -5 on GPU");
    assert_eq!(lt.rows, labels(0..5), "< -5 => [0, 5)");

    // literal on the LEFT
    let lit_left = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE 0 < sz")
        .expect("0 < sz on GPU");
    assert_eq!(lit_left.rows, labels(11..N), "0 < sz => [11, 20)");

    // col-vs-col: sz > peer (constant 0) -> [11, 20)
    let col_col = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz > peer")
        .expect("sz > peer on GPU");
    assert_eq!(col_col.rows, labels(11..N), "sz > peer (0) => [11, 20)");

    // An out-of-int16 literal is a VALID comparison (PG widens both to int4): sz < 30000 -> all rows.
    let wide = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE sz < 30000")
        .expect("sz < 30000 on GPU");
    assert_eq!(wide.rows, labels(0..N), "< 30000 (> i16::MAX) => all rows");

    // projecting the SMALLINT column yields SqlValue::Int2(i16)
    let proj = e
        .execute_resident_expr_select_sql("SELECT sz FROM t WHERE sz = 0")
        .expect("project smallint on GPU");
    assert_eq!(
        proj.rows,
        vec![vec![SqlValue::Int2(0)]],
        "projecting sz returns the smallint value"
    );

    // REJECTIONS -- hard errors, never wrong rows:
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE sz = 'x'")
            .is_err(),
        "smallint compared to a text literal => hard error"
    );
    // INSERT out of the int16 range is rejected ("smallint out of range").
    assert!(
        e.execute_text(3, "INSERT INTO t (sz, peer, label) VALUES (40000, 0, 99)")
            .is_err(),
        "INSERT 40000 into smallint => out of range error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_bool_predicate() {
    // bool-predicate on the general GPU executor (the type matrix, doc 19): a bool column is a
    // 1-bit-per-row BITMAP, so `WHERE flag` expands the bitmap straight to the row mask -- no compare.
    // flag[i] = (i even), label = i.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, label INT)")
        .unwrap();
    const N: i64 = 20;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let flag = if i % 2 == 0 { "true" } else { "false" };
        values.push_str(&format!("({flag}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (flag, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // WHERE flag -> the rows where flag is true (the even labels)
    let r = e
        .execute_resident_expr_select_sql("SELECT label FROM t WHERE flag")
        .expect("WHERE flag on GPU");
    let expected: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 == 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    assert_eq!(r.rows, expected, "WHERE flag => the true (even) rows");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    let even: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 == 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();
    let odd: Vec<Vec<SqlValue>> = (0..N)
        .filter(|i| i % 2 != 0)
        .map(|i| vec![SqlValue::Int4(i as i32)])
        .collect();

    // flag = true / = false / <> (bool literal compares lower to the bitmap->mask kernel via negate).
    for (sql, expected, label) in [
        (
            "SELECT label FROM t WHERE flag = true",
            &even,
            "= true => even",
        ),
        (
            "SELECT label FROM t WHERE flag = false",
            &odd,
            "= false => odd",
        ),
        (
            "SELECT label FROM t WHERE true = flag",
            &even,
            "true = flag => even",
        ),
        (
            "SELECT label FROM t WHERE flag <> true",
            &odd,
            "<> true => odd",
        ),
        (
            "SELECT label FROM t WHERE flag <> false",
            &even,
            "<> false => even",
        ),
        // NOT flag === flag = false (the mapper rewrites it).
        (
            "SELECT label FROM t WHERE NOT flag",
            &odd,
            "NOT flag => odd",
        ),
    ] {
        let got = e.execute_resident_expr_select_sql(sql).expect(label);
        assert_eq!(&got.rows, expected, "{label}");
        assert_eq!(got.executed_target, DeviceTarget::Gpu(0), "{label} on GPU");
    }

    // PROJECT the bool column (gathered straight from the bitmap): flag for rows 0..4 = T,F,T,F.
    let proj = e
        .execute_resident_expr_select_sql("SELECT flag FROM t WHERE label < 4")
        .expect("project bool on GPU");
    assert_eq!(
        proj.rows,
        vec![
            vec![SqlValue::Bool(true)],
            vec![SqlValue::Bool(false)],
            vec![SqlValue::Bool(true)],
            vec![SqlValue::Bool(false)],
        ],
        "SELECT flag => the bitmap bits gathered as bools"
    );

    // A bare NON-bool column predicate is invalid SQL -> hard error (never wrong rows).
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE label")
            .is_err(),
        "WHERE <int column> => argument-of-WHERE-must-be-boolean error"
    );
    // NOT over a non-bool column is also invalid.
    assert!(
        e.execute_resident_expr_select_sql("SELECT label FROM t WHERE NOT label")
            .is_err(),
        "NOT <int column> => hard error"
    );
}
