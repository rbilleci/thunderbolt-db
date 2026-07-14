use crate::{average_sql_value, Engine};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlType, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_count_star() {
    // First operator-axis aggregate: COUNT(*) WHERE <pred> on the general GPU executor. The count is
    // the GPU filter's surviving-row count (the compaction result); PG returns bigint. a[i] = i,
    // flag[i] = (i % 3 == 0).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, flag BOOL)")
        .unwrap();
    const N: i64 = 50;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        let flag = if i % 3 == 0 { "true" } else { "false" };
        values.push_str(&format!("({i}, {flag})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (a, flag) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // COUNT(*) WHERE a > 10 -> a in [11, 50) = 39 rows.
    let r = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE a > 10")
        .expect("count on GPU");
    assert_eq!(
        r.rows,
        vec![vec![SqlValue::Int8(39)]],
        "COUNT(*) WHERE a > 10 => 39"
    );
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));

    // COUNT(*) WHERE flag -- a popcount over the bool bitmap. i%3==0 in [0,50) = 17 rows.
    let flag_count = (0..N).filter(|i| i % 3 == 0).count() as i64;
    let r2 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE flag")
        .expect("count flag on GPU");
    assert_eq!(
        r2.rows,
        vec![vec![SqlValue::Int8(flag_count)]],
        "COUNT(*) WHERE flag"
    );

    // COUNT(*) of an empty result -> 0 (not an error / not NULL).
    let r3 = e
        .execute_resident_expr_select_sql("SELECT COUNT(*) FROM t WHERE a > 1000")
        .expect("count empty on GPU");
    assert_eq!(
        r3.rows,
        vec![vec![SqlValue::Int8(0)]],
        "COUNT(*) empty => 0"
    );

    // count(*) is case-insensitive.
    let r4 = e
        .execute_resident_expr_select_sql("SELECT count(*) FROM t WHERE a >= 0")
        .expect("lowercase count on GPU");
    assert_eq!(
        r4.rows,
        vec![vec![SqlValue::Int8(N)]],
        "count(*) WHERE a >= 0 => all"
    );

    // SUM(int4) over a filtered set -- a GPU reduction over the gathered column; PG returns bigint.
    let sum_of = |keep: &dyn Fn(i64) -> bool| -> i64 { (0..N).filter(|&i| keep(i)).sum() };
    let s1 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 10")
        .expect("sum on GPU");
    assert_eq!(
        s1.rows,
        vec![vec![SqlValue::Int8(sum_of(&|a| a > 10))]],
        "SUM(a) WHERE a > 10"
    );
    assert_eq!(s1.executed_target, DeviceTarget::Gpu(0));
    let s2 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE flag")
        .expect("sum where flag on GPU");
    assert_eq!(
        s2.rows,
        vec![vec![SqlValue::Int8(sum_of(&|a| a % 3 == 0))]],
        "SUM(a) WHERE flag"
    );
    let s3 = e
        .execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a >= 0")
        .expect("sum all on GPU");
    assert_eq!(
        s3.rows,
        vec![vec![SqlValue::Int8(sum_of(&|_| true))]],
        "SUM(a) WHERE a >= 0 => total"
    );
    // SUM over an EMPTY set is SQL NULL (PG) -- one row, NULL (was the M3 hard error).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT SUM(a) FROM t WHERE a > 1000")
            .expect("SUM empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "SUM over empty => SQL NULL"
    );

    // MIN / MAX(int4) over a filtered set -- GPU reductions; PG preserves the type (int4 -> int4).
    let min_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).min().unwrap() as i32;
    let max_of = |keep: &dyn Fn(i64) -> bool| (0..N).filter(|&i| keep(i)).max().unwrap() as i32;
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 10")
        .expect("min on GPU");
    assert_eq!(
        mn.rows,
        vec![vec![SqlValue::Int4(min_of(&|a| a > 10))]],
        "MIN(a) WHERE a > 10 => 11"
    );
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE flag")
        .expect("max where flag on GPU");
    assert_eq!(
        mx.rows,
        vec![vec![SqlValue::Int4(max_of(&|a| a % 3 == 0))]],
        "MAX(a) WHERE flag"
    );
    let mx2 = e
        .execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a >= 0")
        .expect("max all on GPU");
    assert_eq!(
        mx2.rows,
        vec![vec![SqlValue::Int4((N - 1) as i32)]],
        "MAX(a) WHERE a >= 0 => N-1"
    );
    let mn2 = e
        .execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a >= 0")
        .expect("min all on GPU");
    assert_eq!(
        mn2.rows,
        vec![vec![SqlValue::Int4(0)]],
        "MIN(a) WHERE a >= 0 => 0"
    );
    // MIN/MAX over an EMPTY set is SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MIN(a) FROM t WHERE a > 1000")
            .expect("MIN empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MIN over empty => SQL NULL"
    );
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MAX(a) FROM t WHERE a > 1000")
            .expect("MAX empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MAX over empty => SQL NULL"
    );

    // The remaining aggregates are follow-ons -> hard error (clear message, never a wrong/blank
    // answer). FILTER / OVER live INSIDE the FuncCall: they must reject, not silently drop (audit P0).
    for sql in [
        "SELECT COUNT(a) FROM t WHERE a > 0",
        "SELECT COUNT(*) FILTER (WHERE a > 90) FROM t WHERE a >= 0",
        "SELECT COUNT(*) OVER () FROM t WHERE a >= 0",
        "SELECT COUNT(*) OVER (ORDER BY a) FROM t WHERE a >= 0",
        "SELECT COUNT(*) AS c FROM t WHERE a > 0",
        "SELECT SUM(*) FROM t WHERE a > 0",
        "SELECT MIN(*) FROM t WHERE a > 0",
        "SELECT AVG(*) FROM t WHERE a > 0",
    ] {
        assert!(
            e.execute_resident_expr_select_sql(sql).is_err(),
            "{sql} => aggregate follow-on error (no silent FILTER/OVER drop)"
        );
    }

    // AVG(int4) = the GPU sum / the count, as numeric (PG, scale 16). The reduction is on the GPU;
    // the scalar divide reuses average_sql_value (so this checks the pipeline picks the right
    // sum + count). a > 10 -> 1170/39 = 30 exactly; a >= 0 -> 1225/50 = 24.5.
    let avg_expected = |keep: &dyn Fn(i64) -> bool| -> Vec<Vec<SqlValue>> {
        let sum: i128 = (0..N).filter(|&i| keep(i)).map(i128::from).sum();
        let count = (0..N).filter(|&i| keep(i)).count();
        vec![vec![average_sql_value(sum, count)]]
    };
    let a1 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a > 10")
        .expect("avg on GPU");
    assert_eq!(
        a1.rows,
        avg_expected(&|a| a > 10),
        "AVG(a) WHERE a > 10 => 30"
    );
    assert_eq!(a1.executed_target, DeviceTarget::Gpu(0));
    // 30 exactly, at scale 16.
    assert_eq!(
        a1.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(
            30 * 10_i128.pow(16),
            16
        ))]],
        "AVG = 30.0000000000000000"
    );
    let a2 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a >= 0")
        .expect("avg all on GPU");
    assert_eq!(a2.rows, avg_expected(&|_| true), "AVG(a) all => 24.5");
    let a3 = e
        .execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE flag")
        .expect("avg where flag on GPU");
    assert_eq!(a3.rows, avg_expected(&|a| a % 3 == 0), "AVG(a) WHERE flag");
    // AVG over an EMPTY set is SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT AVG(a) FROM t WHERE a > 1000")
            .expect("AVG empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "AVG over empty => SQL NULL"
    );
}

#[test]
fn average_sql_value_matches_postgres_dynamic_scale_and_rounding() {
    // AVG = sum/count must match PostgreSQL's numeric division EXACTLY: a dynamic result scale
    // (PG select_div_scale, ~16 significant digits) + round half-away-from-zero. The hardcoded
    // expectations are PostgreSQL 18's text output (NOT computed via average_sql_value -- the prior
    // self-referential tests could not catch the fixed-scale-16 / truncation P0 the AVG audit found).
    let avg_str = |sum: i128, count: usize| -> String {
        match average_sql_value(sum, count) {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("AVG must be numeric, got {other:?}"),
        }
    };
    // 1-4 integer digits -> scale 16.
    assert_eq!(
        avg_str(3, 1),
        "3.0000000000000000",
        "exact integer, scale 16"
    );
    assert_eq!(avg_str(7, 2), "3.5000000000000000", "3.5, scale 16");
    // sub-1 quotient -> scale 20; repeating, ROUNDS the last digit up.
    assert_eq!(
        avg_str(2, 3),
        "0.66666666666666666667",
        "2/3 rounds, scale 20"
    );
    assert_eq!(avg_str(1, 2), "0.50000000000000000000", "1/2, scale 20");
    // 5-digit integer part -> scale 12 (even exact integers carry the dynamic scale).
    assert_eq!(avg_str(234_000, 4), "58500.000000000000", "58500, scale 12");
    // 11-digit integer part -> scale 8.
    assert_eq!(
        avg_str(100_000_000_000, 7),
        "14285714285.71428571",
        "5-digit-group scale 8"
    );
    // Negative: magnitude + sign both correct (round away from zero).
    assert_eq!(
        avg_str(-2, 3),
        "-0.66666666666666666667",
        "negative rounds away"
    );
    // PG select_div_scale decrements the quotient weight when the dividend's leading base-10000 digit
    // <= the divisor's -- so an exact 1.0 from sum==count renders at scale 20, NOT 16. The naive
    // "quotient decimal weight" formula shipped scale 16 here; these lock the fix (verified vs PG 18).
    assert_eq!(
        avg_str(3, 3),
        "1.00000000000000000000",
        "sum==count -> scale 20 (firstdigit decr)"
    );
    assert_eq!(
        avg_str(5, 5),
        "1.00000000000000000000",
        "leading-digit-equal -> scale 20"
    );
    assert_eq!(
        avg_str(9, 3),
        "3.0000000000000000",
        "fd1(9) > fd2(3) -> no decr, scale 16"
    );
    // Zero sum (e.g. AVG over cancelling rows): PG renders 0 at scale max(S, 20), not 16.
    assert_eq!(
        avg_str(0, 2),
        "0.00000000000000000000",
        "zero sum -> scale 20"
    );
    assert_eq!(
        avg_str(0, 5),
        "0.00000000000000000000",
        "zero sum, count 5 -> scale 20"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_int8_aggregates() {
    // int8 aggregates: MIN/MAX -> int8, SUM/AVG -> numeric (a sum of int8 can exceed i64, so SUM
    // reduces to i128 via the two-atomic carry kernel). Values span > i32::MAX, negatives, and a
    // subset (rows 0,1) whose SUM EXCEEDS i64::MAX. label = row index (the int4 filter column).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (b BIGINT, label INT)")
        .unwrap();
    let vals: [i64; 6] = [
        9_000_000_000_000_000_000,
        8_000_000_000_000_000_000,
        0,
        5_000_000_000,
        -7_000_000_000,
        -9_000_000_000_000_000_000,
    ];
    let mut values = String::new();
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({v}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (b, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // Closed-form oracles (S9): vals = [9e18, 8e18, 0, 5e9, -7e9, -9e18] (label = row index, the int4
    // filter col). keep_all = all rows; keep_lt2 = rows 0,1 (label < 2) = {9e18, 8e18}. Explicit
    // constants, not a host `vals.iter()...min()/max()/sum()` re-implementation of the aggregate.
    let min_all: i64 = -9_000_000_000_000_000_000; // MIN over all rows
    let max_lt2: i64 = 9_000_000_000_000_000_000; // MAX of {9e18, 8e18}
    let sum_lt2: i128 = 17_000_000_000_000_000_000; // 9e18 + 8e18 (> i64::MAX -> i128 carry)
    let sum_all: i128 = 7_999_999_998_000_000_000; // 9e18 + 8e18 + 0 + 5e9 - 7e9 - 9e18

    // MIN/MAX(int8) -> int8.
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(b) FROM t WHERE label >= 0")
        .expect("min");
    assert_eq!(mn.rows, vec![vec![SqlValue::Int8(min_all)]], "MIN(b) all");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(b) FROM t WHERE label < 2")
        .expect("max subset");
    assert_eq!(
        mx.rows,
        vec![vec![SqlValue::Int8(max_lt2)]],
        "MAX(b) subset => 9e18"
    );

    // SUM(int8) -> numeric (scale 0). The subset {9e18, 8e18} sums to 17e18 -- EXCEEDS i64::MAX, so
    // the i128 two-atomic carry must be correct.
    let s_sub = e
        .execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label < 2")
        .expect("sum subset");
    assert_eq!(
        s_sub.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_lt2, 0))]],
        "SUM subset => 17e18 (> i64::MAX, i128 carry)"
    );
    assert!(sum_lt2 > i128::from(i64::MAX), "test really exceeds i64");
    // The result COLUMN descriptor must match the numeric value (the int8-agg audit caught SUM(int8)
    // declaring Int4/oid 23 -- a wire-decode mismatch). PG SUM(int8) -> numeric (oid 1700).
    assert!(
        matches!(s_sub.columns[0].ty, SqlType::Numeric { .. }),
        "SUM(int8) result column must be numeric, got {:?}",
        s_sub.columns[0].ty
    );
    assert_eq!(
        s_sub.columns[0].type_oid, 1700,
        "SUM(int8) wire oid = numeric 1700"
    );
    let s_all = e
        .execute_resident_expr_select_sql("SELECT SUM(b) FROM t WHERE label >= 0")
        .expect("sum all");
    assert_eq!(
        s_all.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(sum_all, 0))]],
        "SUM all (incl. negatives)"
    );

    // AVG(int8) -> numeric. 17e18 / 2 = 8.5e18 (scale 0 at this magnitude).
    let av = e
        .execute_resident_expr_select_sql("SELECT AVG(b) FROM t WHERE label < 2")
        .expect("avg subset");
    assert_eq!(
        av.rows,
        vec![vec![average_sql_value(sum_lt2, 2)]],
        "AVG subset"
    );
    match &av.rows[0][0] {
        SqlValue::Numeric(d) => {
            assert_eq!(d.to_decimal_string(), "8500000000000000000", "AVG = 8.5e18")
        }
        other => panic!("AVG must be numeric, got {other:?}"),
    }

    // Empty SUM/MIN/AVG -> SQL NULL (PG).
    for sql in [
        "SELECT MIN(b) FROM t WHERE label > 1000",
        "SELECT SUM(b) FROM t WHERE label > 1000",
        "SELECT AVG(b) FROM t WHERE label > 1000",
    ] {
        assert_eq!(
            e.execute_resident_expr_select_sql(sql).expect(sql).rows,
            vec![vec![SqlValue::Null]],
            "{sql} => SQL NULL"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_minmax() {
    // MIN/MAX(numeric) over a filtered set -> numeric (PG preserves the type). Reduces the i128
    // mantissas via the partials + host-combine reduction. label = row index (int4 filter col).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)")
        .unwrap();
    let prices = ["12.50", "-3.75", "100.00", "0.01", "-99.99", "42.42"];
    let mut values = String::new();
    for (i, p) in prices.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({p}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (p, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    // mantissas at scale 2.
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 2));

    // All rows: MIN = -99.99, MAX = 100.00.
    let mn = e
        .execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label >= 0")
        .expect("min num");
    assert_eq!(mn.rows, vec![vec![num(-9999)]], "MIN(p) all => -99.99");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));
    let mx = e
        .execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label >= 0")
        .expect("max num");
    assert_eq!(mx.rows, vec![vec![num(10000)]], "MAX(p) all => 100.00");

    // Subset (label < 3 -> 12.50, -3.75, 100.00): MIN = -3.75, MAX = 100.00.
    let mn2 = e
        .execute_resident_expr_select_sql("SELECT MIN(p) FROM t WHERE label < 3")
        .expect("min subset");
    assert_eq!(mn2.rows, vec![vec![num(-375)]], "MIN subset => -3.75");
    let mx2 = e
        .execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label < 3")
        .expect("max subset");
    assert_eq!(mx2.rows, vec![vec![num(10000)]], "MAX subset => 100.00");

    // Empty -> SQL NULL (PG).
    assert_eq!(
        e.execute_resident_expr_select_sql("SELECT MAX(p) FROM t WHERE label > 1000")
            .expect("MAX(numeric) empty on GPU")
            .rows,
        vec![vec![SqlValue::Null]],
        "MAX(numeric) over empty => SQL NULL"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_numeric_sum_avg() {
    // SUM(numeric) -> numeric at the column scale (i128 mantissa sum); AVG(numeric) -> numeric at PG's
    // division scale. Expected values derived from PG's numeric semantics (SUM keeps scale 2; AVG of a
    // weight-0 quotient over a scale-2 dividend has rscale max(2, 16) = 16).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (p NUMERIC(10,2), label INT)")
        .unwrap();
    let prices = ["12.50", "-3.75", "100.00", "0.01", "-99.99", "42.42"];
    let mut values = String::new();
    for (i, p) in prices.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({p}, {i})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (p, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }

    // SUM all = 12.50 - 3.75 + 100.00 + 0.01 - 99.99 + 42.42 = 51.19 (mantissa 5119, scale 2).
    let s_all = e
        .execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label >= 0")
        .expect("sum all");
    assert_eq!(
        s_all.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(5119, 2))]],
        "SUM all = 51.19"
    );
    assert_eq!(s_all.executed_target, DeviceTarget::Gpu(0));
    assert!(
        matches!(s_all.columns[0].ty, SqlType::Numeric { .. }),
        "SUM(numeric) col numeric"
    );
    assert_eq!(s_all.columns[0].type_oid, 1700, "SUM(numeric) oid 1700");
    // SUM subset (12.50, -3.75) = 8.75.
    let s_sub = e
        .execute_resident_expr_select_sql("SELECT SUM(p) FROM t WHERE label < 2")
        .expect("sum subset");
    assert_eq!(
        s_sub.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(875, 2))]],
        "SUM subset = 8.75"
    );

    // AVG all = 51.19 / 6 = 8.5316666... -> scale 16, round half-away.
    let a_all = e
        .execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label >= 0")
        .expect("avg all");
    match &a_all.rows[0][0] {
        SqlValue::Numeric(d) => assert_eq!(d.to_decimal_string(), "8.5316666666666667", "AVG all"),
        other => panic!("AVG numeric, got {other:?}"),
    }
    assert!(
        matches!(a_all.columns[0].ty, SqlType::Numeric { .. }),
        "AVG(numeric) col numeric"
    );
    // AVG subset = 8.75 / 2 = 4.375 -> scale 16.
    let a_sub = e
        .execute_resident_expr_select_sql("SELECT AVG(p) FROM t WHERE label < 2")
        .expect("avg subset");
    match &a_sub.rows[0][0] {
        SqlValue::Numeric(d) => {
            assert_eq!(d.to_decimal_string(), "4.3750000000000000", "AVG subset")
        }
        other => panic!("AVG numeric, got {other:?}"),
    }

    // Empty SUM/AVG -> SQL NULL (PG).
    for sql in [
        "SELECT SUM(p) FROM t WHERE label > 1000",
        "SELECT AVG(p) FROM t WHERE label > 1000",
    ] {
        assert_eq!(
            e.execute_resident_expr_select_sql(sql).expect(sql).rows,
            vec![vec![SqlValue::Null]],
            "{sql} => SQL NULL"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_numeric_sum_overflow_errors() {
    // SUM(numeric) is CHECKED: a mantissa sum exceeding i128 is PG `numeric field overflow`, NEVER a
    // silent wrap. Each mantissa is 9e18 * 10^19 = 9e37 (column NUMERIC(38,19), integer part 9e18 fits
    // the legacy parser's i64 literal range); two sum to 1.8e38 > i128::MAX (~1.7e38).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE big (v NUMERIC(38,19), label INT)")
        .unwrap();
    let big = "9000000000000000000"; // 9e18, fits i64
    e.execute_text(
        2,
        &format!("INSERT INTO big (v, label) VALUES ({big}, 0), ({big}, 1)"),
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let mantissa = 9_000_000_000_000_000_000_i128 * 10_i128.pow(19); // 9e37
    assert!(
        mantissa.checked_add(mantissa).is_none(),
        "sanity: 2 * 9e37 must overflow i128 (else the test does not exercise overflow)"
    );
    let err = e
        .execute_resident_expr_select_sql("SELECT SUM(v) FROM big WHERE label >= 0")
        .expect_err("SUM overflow must error");
    assert!(
        err.to_string().contains("numeric field overflow"),
        "expected numeric field overflow, got: {err}"
    );
    // A single value (no overflow) still sums correctly to its mantissa at scale 19.
    let ok = e
        .execute_resident_expr_select_sql("SELECT SUM(v) FROM big WHERE label < 1")
        .expect("single value sums");
    assert_eq!(
        ok.rows,
        vec![vec![SqlValue::Numeric(Decimal128::new(mantissa, 19))]],
        "SUM of one 9e37-mantissa value"
    );
}
