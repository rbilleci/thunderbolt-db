use crate::{Engine, RelationalSelectResult};
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_expression() {
    // ORDER BY an EXPRESSION (`a+b`, `a*2`) on the general GPU path: the device Expr interpreter
    // evaluates it into an i64 key column feeding the GPU bitonic sort -- single key, multi-key
    // (expr + column), expr + a text key (hetero), WHERE, LIMIT. executed_target==Gpu throughout.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT, name TEXT)")
        .unwrap();
    let rows: &[(i32, i32, i32, &str)] = &[
        (5, 1, 1, "bob"),  // a+b=6
        (2, 9, 2, "amy"),  // a+b=11
        (8, 0, 3, "cara"), // a+b=8
        (1, 1, 4, "dan"),  // a+b=2
        (3, 5, 5, "amy"),  // a+b=8 (ties c=3 on the sum; "amy" ties c=2 on the name)
        (4, 3, 6, "bob"),  // a+b=7 ("bob" ties c=1 on the name)
    ];
    let values = rows
        .iter()
        .map(|(a, b, c, n)| format!("({a}, {b}, {c}, '{n}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (a, b, c, name) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) ORDER BY a+b ASC -- the sum ties (c=3,c=5 both 8; bitonic is unstable), so assert the SUM
    // sequence (computed from the projected a,b) is monotonic, not the exact rows.
    let s = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a + b")
        .unwrap();
    assert_eq!(
        s.executed_target,
        DeviceTarget::Gpu(0),
        "ORDER BY a+b on GPU"
    );
    let sums: Vec<i32> = i4(&s, 0)
        .iter()
        .zip(i4(&s, 1))
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(sums, vec![2, 6, 7, 8, 8, 11], "ORDER BY a+b ASC monotonic");

    // (b) ORDER BY a*2 DESC -- monotonic in a, no ties -> exact (c identifies rows).
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY a * 2 DESC")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![3, 1, 6, 5, 2, 4], "ORDER BY a*2 DESC");

    // (c) multi-key expr-primary: ORDER BY a+b, c -- the (sum,c) tuple is distinct -> deterministic.
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY a + b, c")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![4, 1, 6, 3, 5, 2], "ORDER BY a+b, c");

    // (d) hetero (text + expr): ORDER BY name, a+b -- (name,sum) distinct -> deterministic.
    let s = e
        .execute_relational_select_text("SELECT c FROM t ORDER BY name, a + b")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        i4(&s, 0),
        vec![5, 2, 1, 6, 3, 4],
        "ORDER BY name, a+b (hetero)"
    );

    // (e) WHERE + expr ORDER BY + LIMIT. a>2: c1(sum6),c3(sum8),c5(sum8),c6(sum7) -> sorted 6,7,8,8;
    // LIMIT 2 -> c1, c6.
    let s = e
        .execute_relational_select_text("SELECT c FROM t WHERE a > 2 ORDER BY a + b LIMIT 2")
        .unwrap();
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(i4(&s, 0), vec![1, 6], "WHERE a>2 ORDER BY a+b LIMIT 2");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_int8_expression_sorts_at_i64_width() {
    // ORDER BY a BIGINT expression must read the arith value buffer at i64 width. Reading it as i32
    // (the pre-fix bug) would stride the 8-byte BIGINT column by 4 bytes -> garbage keys. A value
    // beyond i32::MAX also exercises the i64 range (an i32 read could not even represent it).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a BIGINT, b BIGINT, id INT)")
        .unwrap();
    // a+b: id1->15, id2->2, id3->5000000001 (> i32::MAX), id4->7. asc by a+b: 2,7,15,5e9 -> ids 2,4,1,3.
    e.execute_text(
        2,
        "INSERT INTO t (a, b, id) VALUES (10,5,1),(1,1,2),(5000000000,1,3),(3,4,4)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT id FROM t ORDER BY a + b")
        .expect("int8 expression ORDER BY runs on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        vec![
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(4)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(3)],
        ],
        "BIGINT a+b sorted at i64 width"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_numeric_b128_width() {
    // ORDER BY a NUMERIC (i128) column on the GPU: the 16-byte comparator (signed HIGH limb, unsigned
    // LOW limb). Mixed-sign values exercise both limbs -- the signed hi distinguishes sign (negatives
    // hi=-1 below positives hi=0); the unsigned lo decides within a sign (two's-complement low bits).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (v NUMERIC(10,2), label INT)")
        .unwrap();
    // labels = the sorted rank (inserted shuffled): -20 < -10 < 0 < 5 < 10 < 20. 6 rows -> npot 8.
    e.execute_text(
        2,
        "INSERT INTO t (v, label) VALUES \
         (20.00, 5), (-10.00, 1), (5.00, 3), (-20.00, 0), (10.00, 4), (0.00, 2)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let asc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v")
        .expect("numeric ORDER BY on the GPU");
    assert_eq!(asc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        asc.rows,
        (0..6).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "numeric ASC (signed hi, unsigned lo)"
    );
    let desc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v DESC")
        .expect("numeric DESC on the GPU");
    assert_eq!(
        desc.rows,
        (0..6)
            .rev()
            .map(|i| vec![SqlValue::Int4(i)])
            .collect::<Vec<_>>(),
        "numeric DESC"
    );
    // WHERE v>0 -> ranks 3,4,5 ; LIMIT 2 -> 3,4.
    let win = e
        .execute_relational_select_text("SELECT label FROM t WHERE v > 0 ORDER BY v LIMIT 2")
        .expect("numeric WHERE+LIMIT on the GPU");
    assert_eq!(
        win.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(4)]],
        "v>0 asc limit2 -> ranks 3,4"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_b128_secondary_key_dispatch() {
    // A b128 (numeric) key as a tie-broken SECONDARY: the int primary `grp` ties, so the numeric `v`
    // decides. Guards the 2-bit key_plan dispatch for a b128 key that is NOT the primary -- a
    // `kind >> 31` (instead of >> 30) bug would misdispatch the secondary numeric onto the text leg
    // and misorder within each group. (The other b128 multi-key test uses a b128 PRIMARY.)
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (grp INT, v NUMERIC(10,2), label INT)")
        .unwrap();
    // ORDER BY grp ASC, v ASC: grp=1 {v=10,20,30 -> 0,1,2}, grp=2 {v=5,15,25 -> 3,4,5}. label = rank.
    e.execute_text(
        2,
        "INSERT INTO t (grp, v, label) VALUES \
         (1, 30.00, 2), (1, 10.00, 0), (1, 20.00, 1), (2, 15.00, 4), (2, 5.00, 3), (2, 25.00, 5)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let s = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY grp ASC, v ASC")
        .expect("int-primary + numeric-secondary ORDER BY on the GPU");
    assert_eq!(s.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        s.rows,
        (0..6).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "numeric as a tie-broken SECONDARY key dispatches correctly"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_uuid_big_endian_unsigned() {
    // ORDER BY a UUID column: 16 raw bytes, UNSIGNED BIG-ENDIAN (byte 0 most significant). Bytes >= 0x80
    // sort ABOVE 0x7f (unsigned). byte-15 breaks a byte-0 tie. 7 rows -> npot 8 exercises padding.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (id UUID, label INT)")
        .unwrap();
    let uuid_b0 = |b: u32| format!("{b:02x}000000-0000-0000-0000-000000000000");
    // sorted order: 00/00, 00/ff, 10, 40, 7f, 80, ff -> labels = rank, inserted shuffled.
    let rows: [(String, i32); 7] = [
        (uuid_b0(0xff), 6),
        (uuid_b0(0x00), 0),
        (uuid_b0(0x80), 5),
        (uuid_b0(0x10), 2),
        ("00000000-0000-0000-0000-0000000000ff".to_string(), 1),
        (uuid_b0(0x7f), 4),
        (uuid_b0(0x40), 3),
    ];
    let mut values = String::new();
    for (i, (uuid, label)) in rows.iter().enumerate() {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("('{uuid}', {label})"));
    }
    e.execute_text(2, &format!("INSERT INTO t (id, label) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let asc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY id")
        .expect("uuid ORDER BY on the GPU");
    assert_eq!(asc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        asc.rows,
        (0..7).map(|i| vec![SqlValue::Int4(i)]).collect::<Vec<_>>(),
        "uuid ASC big-endian unsigned"
    );
    let desc = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY id DESC")
        .expect("uuid DESC on the GPU");
    assert_eq!(
        desc.rows,
        (0..7)
            .rev()
            .map(|i| vec![SqlValue::Int4(i)])
            .collect::<Vec<_>>(),
        "uuid DESC"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_mixed_b128_keys() {
    // Multi-key ORDER BY with a b128 key as the tie-broken primary, on the heterogeneous comparator.
    let mut e = Engine::new_local_test_engine();
    // (a) numeric DESC, int ASC: a numeric tie is broken by the int key.
    e.execute_text(1, "CREATE TABLE t (v NUMERIC(10,2), tb INT, label INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (v, tb, label) VALUES (10.00, 2, 2), (10.00, 1, 1), (20.00, 5, 0)",
    )
    .unwrap();
    let s1 = e.populate_relational_residency_snapshot("t").unwrap();
    if s1.device_memory_proof.is_none() {
        return;
    }
    let r = e
        .execute_relational_select_text("SELECT label FROM t ORDER BY v DESC, tb ASC")
        .expect("numeric+int hetero sort");
    assert_eq!(r.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        r.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)]
        ],
        "v=20 first; the v=10 tie broken by tb ASC"
    );
    // (b) uuid ASC, text ASC: a uuid tie is broken by the text key.
    e.execute_text(3, "CREATE TABLE u (id UUID, name TEXT, label INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO u (id, name, label) VALUES \
         ('00000000-0000-0000-0000-000000000001', 'bob', 1), \
         ('00000000-0000-0000-0000-000000000001', 'amy', 0), \
         ('00000000-0000-0000-0000-000000000002', 'zoe', 2)",
    )
    .unwrap();
    let s2 = e.populate_relational_residency_snapshot("u").unwrap();
    if s2.device_memory_proof.is_none() {
        return;
    }
    let r2 = e
        .execute_relational_select_text("SELECT label FROM u ORDER BY id ASC, name ASC")
        .expect("uuid+text hetero sort");
    assert_eq!(r2.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        r2.rows,
        vec![
            vec![SqlValue::Int4(0)],
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)]
        ],
        "uuid tie (..01) broken by name ASC (amy<bob), then ..02"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_order_by_expression_overflow_is_pg_error() {
    // ORDER BY a+b where a+b overflows int4 -> a clean PG "integer out of range" error (checked
    // arithmetic on-device), NOT a wrapped value, NOT a CPU re-execution.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (2147483647, 1), (1, 1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a + b")
        .expect_err("a+b overflow must surface as a PG error, not wrap or CPU-fallback");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("out of range") || msg.contains("overflow"),
        "expected integer out of range, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_via_gpu_sort() {
    // A non-grouped ORDER BY over an int column runs on the GENERAL GPU Expr executor + the GPU bitonic
    // sort (NOT the enumerated ordered-projection shape, NOT the CPU path). executed_target==Gpu proves
    // it took the general GPU path through the routing gate (`execute_relational_select_text`).
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c BIGINT)")
        .unwrap();
    let rows: &[(i32, i32, i64)] = &[
        (5, 50, 500),
        (2, 20, 200),
        (8, 80, 800),
        (1, 10, 100),
        (9, 90, 900),
        (3, 30, 300),
    ];
    let values = rows
        .iter()
        .map(|(a, b, c)| format!("({a}, {b}, {c})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (a, b, c) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let int4_col = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) ORDER BY a ASC -- and confirm it took the general GPU path.
    let asc = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a")
        .unwrap();
    assert_eq!(
        asc.executed_target,
        DeviceTarget::Gpu(0),
        "non-grouped ORDER BY must run on the general GPU path"
    );
    assert_eq!(int4_col(&asc, 0), vec![1, 2, 3, 5, 8, 9], "ORDER BY a ASC");

    // (b) ORDER BY a DESC.
    let desc = e
        .execute_relational_select_text("SELECT a, b FROM t ORDER BY a DESC")
        .unwrap();
    assert_eq!(desc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&desc, 0),
        vec![9, 8, 5, 3, 2, 1],
        "ORDER BY a DESC"
    );

    // (c) WHERE b > 25 ORDER BY a -> a in {3,5,8,9} (their b are 30/50/80/90).
    let filtered = e
        .execute_relational_select_text("SELECT a FROM t WHERE b > 25 ORDER BY a")
        .unwrap();
    assert_eq!(filtered.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(int4_col(&filtered, 0), vec![3, 5, 8, 9], "WHERE + ORDER BY");

    // (d) ORDER BY c DESC (an int8 key).
    let by_c = e
        .execute_relational_select_text("SELECT a, c FROM t ORDER BY c DESC")
        .unwrap();
    assert_eq!(by_c.executed_target, DeviceTarget::Gpu(0));
    let c_col: Vec<i64> = by_c
        .rows
        .iter()
        .map(|r| match r[1] {
            SqlValue::Int8(v) => v,
            ref other => panic!("expected Int8, got {other:?}"),
        })
        .collect();
    assert_eq!(c_col, vec![900, 800, 500, 300, 200, 100], "ORDER BY c DESC");

    // (e) ORDER BY a LIMIT 3 OFFSET 1 -> [2, 3, 5].
    let limited = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 3 OFFSET 1")
        .unwrap();
    assert_eq!(limited.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&limited, 0),
        vec![2, 3, 5],
        "ORDER BY a LIMIT 3 OFFSET 1"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_resident_select_limit_offset_window_edges() {
    // S4: OFFSET/LIMIT on the projection path now slices `indices_u64` (the device-ordered index vector)
    // BEFORE the column gather -- only the kept window is materialized from the device, no host
    // drain/truncate. These edge cases pin the windowing math against the prior drain/truncate: OFFSET
    // past the end, LIMIT 0, OFFSET+LIMIT past the end (clamped), and a DESC window. The ORDER BY makes
    // every window deterministic.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a) VALUES (5),(2),(8),(1),(9),(3)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };
    // Sorted ascending the rows are [1,2,3,5,8,9].

    // OFFSET past the end -> empty (start clamps to len).
    let beyond = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a OFFSET 10")
        .unwrap();
    assert_eq!(beyond.executed_target, DeviceTarget::Gpu(0));
    assert!(beyond.rows.is_empty(), "OFFSET past the end -> no rows");

    // LIMIT 0 -> empty.
    let zero = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 0")
        .unwrap();
    assert!(zero.rows.is_empty(), "LIMIT 0 -> no rows");

    // OFFSET 4 + LIMIT 100 past the end -> clamped to the tail [8,9].
    let tail = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a LIMIT 100 OFFSET 4")
        .unwrap();
    assert_eq!(
        col(&tail),
        vec![8, 9],
        "LIMIT past the end clamps to the tail"
    );

    // OFFSET only (no LIMIT) -> drop the first four, keep [8,9].
    let off = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a OFFSET 4")
        .unwrap();
    assert_eq!(col(&off), vec![8, 9], "OFFSET only keeps the tail");

    // DESC + LIMIT 2 OFFSET 1 -> from [9,8,5,3,2,1] skip 1, take 2 -> [8,5].
    let desc = e
        .execute_relational_select_text("SELECT a FROM t ORDER BY a DESC LIMIT 2 OFFSET 1")
        .unwrap();
    assert_eq!(col(&desc), vec![8, 5], "DESC LIMIT 2 OFFSET 1 window");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_500_rows() {
    // 500 rows (not a power of two -> padding) shuffled via a coprime stride (a permutation of 0..500),
    // sorted on the GPU. Exercises the bitonic sort at scale on the projection path.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE big (a INT)").unwrap();
    let vals = (0..500i32)
        .map(|i| format!("({})", (i * 137 + 11).rem_euclid(500)))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (a) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_relational_select_text("SELECT a FROM big ORDER BY a")
        .unwrap();
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let a_col: Vec<i32> = res
        .rows
        .iter()
        .map(|r| match r[0] {
            SqlValue::Int4(v) => v,
            ref other => panic!("expected Int4, got {other:?}"),
        })
        .collect();
    assert_eq!(
        a_col,
        (0..500).collect::<Vec<i32>>(),
        "500-row GPU ORDER BY a"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_radix_above_crossover() {
    // 11_000 rows (> the 10_000 adaptive crossover) -> the single-int-key ORDER BY takes the GPU RADIX
    // arm (engine_expr order_by_sort_i64), end to end. A coprime-stride (137, gcd(137,11000)=1)
    // permutation of 0..11000 must sort back to 0..11000 (ASC) / its reverse (DESC), on the GPU.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 11_000;
    let vals = (0..N)
        .map(|i| format!("({})", (i * 137 + 11).rem_euclid(N)))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (a) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let col = |sql: &str| -> Vec<i32> {
        let res = e.execute_relational_select_text(sql).unwrap();
        assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };
    assert_eq!(
        col("SELECT a FROM big ORDER BY a"),
        (0..N).collect::<Vec<i32>>(),
        "11k-row GPU radix ORDER BY a"
    );
    assert_eq!(
        col("SELECT a FROM big ORDER BY a DESC"),
        (0..N).rev().collect::<Vec<i32>>(),
        "11k-row GPU radix ORDER BY a DESC"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_text() {
    // A non-grouped ORDER BY over a TEXT column sorts on the GENERAL GPU Expr executor via the byte-wise
    // text bitonic comparator (lexicographic, UNSIGNED bytes, a prefix sorts smaller) -- NOT a CPU sort.
    // executed_target==Gpu proves the general GPU path. Covers prefixes, the empty string, duplicates, DESC.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (s TEXT, id INT)")
        .unwrap();
    // Deliberate corner cases: prefixes ('a' < 'ab'), empty string (sorts first), duplicates, mixed length.
    let rows: &[(&str, i32)] = &[
        ("banana", 1),
        ("apple", 2),
        ("ab", 3),
        ("a", 4),
        ("", 5),
        ("apple", 6),
        ("ab", 7),
        ("cherry", 8),
    ];
    let values = rows
        .iter()
        .map(|(s, id)| format!("('{s}', {id})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (s, id) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let text_col = |res: &RelationalSelectResult| -> Vec<String> {
        res.rows
            .iter()
            .map(|r| match &r[0] {
                SqlValue::Text(v) => v.to_string(),
                other => panic!("expected Text, got {other:?}"),
            })
            .collect()
    };
    // Closed-form oracle (S9): the 8 rows in unsigned-byte (lexicographic) order -- empty string first,
    // a < ab (a prefix sorts smaller), duplicates kept -- stated explicitly, not via a host Rust sort of
    // the input (which would re-implement the ORDER BY comparator on the CPU).
    let oracle: Vec<&str> = vec!["", "a", "ab", "ab", "apple", "apple", "banana", "cherry"];

    // (a) ORDER BY s ASC -- and confirm it took the general GPU path.
    let asc = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s")
        .unwrap();
    assert_eq!(
        asc.executed_target,
        DeviceTarget::Gpu(0),
        "text ORDER BY must run on the general GPU path"
    );
    assert_eq!(
        text_col(&asc),
        oracle,
        "ORDER BY s ASC (prefixes, empty, dups)"
    );

    // (b) ORDER BY s DESC -- the reverse key order (ties are identical strings, so order among them is moot).
    let desc = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s DESC")
        .unwrap();
    assert_eq!(desc.executed_target, DeviceTarget::Gpu(0));
    let mut oracle_desc = oracle.clone();
    oracle_desc.reverse();
    assert_eq!(text_col(&desc), oracle_desc, "ORDER BY s DESC");

    // (c) WHERE id > 4 ORDER BY s -> id in {5,6,7,8} = {"", "apple", "ab", "cherry"} sorted.
    let filtered = e
        .execute_relational_select_text("SELECT s FROM t WHERE id > 4 ORDER BY s")
        .unwrap();
    assert_eq!(filtered.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        text_col(&filtered),
        vec!["", "ab", "apple", "cherry"],
        "WHERE + text ORDER BY"
    );

    // (d) ORDER BY s LIMIT 3 -> the first three: ["", "a", "ab"].
    let limited = e
        .execute_relational_select_text("SELECT s FROM t ORDER BY s LIMIT 3")
        .unwrap();
    assert_eq!(limited.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        text_col(&limited),
        vec!["", "a", "ab"],
        "text ORDER BY LIMIT 3"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_text_300_rows() {
    // 300 rows (not a power of two -> bitonic padding), distinct zero-padded strings shuffled via a
    // coprime stride (a permutation of 0..300), GPU-sorted by the text comparator back to order.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE big (s TEXT)").unwrap();
    let vals = (0..300usize)
        .map(|i| format!("('{:04}')", (i * 137 + 11) % 300))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO big (s) VALUES {vals}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("big").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let res = e
        .execute_relational_select_text("SELECT s FROM big ORDER BY s")
        .unwrap();
    assert_eq!(res.executed_target, DeviceTarget::Gpu(0));
    let got: Vec<String> = res
        .rows
        .iter()
        .map(|r| match &r[0] {
            SqlValue::Text(v) => v.to_string(),
            other => panic!("expected Text, got {other:?}"),
        })
        .collect();
    let expected: Vec<String> = (0..300).map(|i| format!("{i:04}")).collect();
    assert_eq!(got, expected, "300-row GPU text ORDER BY");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_multikey() {
    // MULTI-KEY ORDER BY (`ORDER BY a ASC, b DESC, ...`) sorts on the GPU via the multi-key bitonic
    // comparator on the general Expr executor: each key, in significance order with its own direction,
    // breaks ties for the next. executed_target==Gpu proves the general GPU path (routing gate + GPU
    // multi-key sort), NOT the CPU/enumerated path. Rows are engineered so EVERY key is the real
    // tie-breaker -- drop any key and the expected order changes.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t2 (a INT, b INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t2 (a, b) VALUES (1, 10), (1, 30), (1, 20), (2, 50), (2, 40)",
    )
    .unwrap();
    e.execute_text(3, "CREATE TABLE t3 (a INT, b INT, c INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO t3 (a, b, c) VALUES (1, 5, 100), (1, 5, 50), (1, 8, 10), (2, 3, 7), (2, 3, 9)",
    )
    .unwrap();
    e.execute_text(5, "CREATE TABLE td (d DATE, x INT)")
        .unwrap();
    e.execute_text(
        6,
        "INSERT INTO td (d, x) VALUES ('2024-01-02', 5), ('2024-01-01', 9), \
         ('2024-01-01', 3), ('2024-01-02', 7)",
    )
    .unwrap();
    let s2 = e.populate_relational_residency_snapshot("t2").unwrap();
    e.populate_relational_residency_snapshot("t3").unwrap();
    e.populate_relational_residency_snapshot("td").unwrap();
    if s2.device_memory_proof.is_none() {
        return;
    }
    let int4_col = |res: &RelationalSelectResult, col: usize| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[col] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) two keys: a ASC, b DESC -- a-ties (1,1,1 / 2,2) broken by b DESCENDING.
    let ab = e
        .execute_relational_select_text("SELECT a, b FROM t2 ORDER BY a ASC, b DESC")
        .unwrap();
    assert_eq!(
        ab.executed_target,
        DeviceTarget::Gpu(0),
        "multi-key ORDER BY must run on the general GPU path"
    );
    assert_eq!(int4_col(&ab, 0), vec![1, 1, 1, 2, 2], "key a ASC");
    assert_eq!(
        int4_col(&ab, 1),
        vec![30, 20, 10, 50, 40],
        "key b DESC breaks a-ties"
    );

    // (b) three keys: a ASC, b DESC, c ASC -- a-ties broken by b, then (a,b)-ties broken by c. The two
    // (1,5,*) rows tie on a AND b and are ordered ONLY by c ASC (50 before 100).
    let abc = e
        .execute_relational_select_text("SELECT a, b, c FROM t3 ORDER BY a ASC, b DESC, c ASC")
        .unwrap();
    assert_eq!(abc.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(int4_col(&abc, 0), vec![1, 1, 1, 2, 2], "key a ASC");
    assert_eq!(int4_col(&abc, 1), vec![8, 5, 5, 3, 3], "key b DESC");
    assert_eq!(
        int4_col(&abc, 2),
        vec![10, 50, 100, 7, 9],
        "key c ASC breaks (a,b)-ties"
    );

    // (c) DATE + int: d ASC (primary), x DESC. The x order [9,3,7,5] proves d is the PRIMARY key --
    // sorting by x DESC alone would give [9,7,5,3]. (A mixed key-type multi-key sort.)
    let dx = e
        .execute_relational_select_text("SELECT x FROM td ORDER BY d ASC, x DESC")
        .unwrap();
    assert_eq!(dx.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&dx, 0),
        vec![9, 3, 7, 5],
        "date primary ASC, int secondary DESC"
    );

    // (d) single-key down the SAME path is unchanged (K=1) -- regression guard.
    let single = e
        .execute_relational_select_text("SELECT a FROM t2 ORDER BY a ASC")
        .unwrap();
    assert_eq!(single.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        int4_col(&single, 0),
        vec![1, 1, 1, 2, 2],
        "single key a ASC unchanged"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nongrouped_order_by_mixed_int_text() {
    // MIXED int+text multi-key ORDER BY (`ORDER BY name /*text*/, age /*int*/, id /*int*/`) sorts on the
    // GPU via the HETEROGENEOUS comparator -- each key dispatched to the s64 compare (int) or the byte
    // compare (text). executed_target==Gpu proves the general GPU path. Rows are engineered so EVERY key
    // is the real tie-breaker. Completes the canonical `ORDER BY last_name, age, id`.
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (name TEXT, age INT, id INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (name, age, id) VALUES \
         ('bob', 30, 1), ('alice', 25, 2), ('alice', 25, 3), ('alice', 40, 4), \
         ('bob', 30, 5), ('bob', 20, 6)",
    )
    .unwrap();
    e.execute_text(3, "CREATE TABLE names (last TEXT, first TEXT, id INT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO names (last, first, id) VALUES \
         ('smith', 'bob', 1), ('jones', 'amy', 2), ('smith', 'amy', 3), ('smith', 'al', 4)",
    )
    .unwrap();
    let snap = e.populate_relational_residency_snapshot("people").unwrap();
    e.populate_relational_residency_snapshot("names").unwrap();
    if snap.device_memory_proof.is_none() {
        return;
    }
    let id_col = |res: &RelationalSelectResult| -> Vec<i32> {
        res.rows
            .iter()
            .map(|r| match r[0] {
                SqlValue::Int4(v) => v,
                ref other => panic!("expected Int4, got {other:?}"),
            })
            .collect()
    };

    // (a) the canonical 3-key: name ASC (text primary), age DESC (int), id ASC (int). alice<bob; within
    // a name, age DESC; within name+age (the two alice/25 rows), id ASC. Each key a real tie-breaker.
    let abc = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY name ASC, age DESC, id ASC")
        .unwrap();
    assert_eq!(
        abc.executed_target,
        DeviceTarget::Gpu(0),
        "mixed int+text ORDER BY must run on the general GPU path"
    );
    assert_eq!(
        id_col(&abc),
        vec![4, 2, 3, 1, 5, 6],
        "name ASC / age DESC / id ASC"
    );

    // (b) int primary, text secondary: age ASC, name ASC, id ASC. ages group; name (degenerate tie
    // within each age here) then id ASC. Proves age is primary (not name).
    let ba = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY age ASC, name ASC, id ASC")
        .unwrap();
    assert_eq!(ba.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        id_col(&ba),
        vec![6, 2, 3, 1, 5, 4],
        "age ASC / name ASC / id ASC"
    );

    // (c) DESC on the TEXT key: name DESC, id ASC. bob before alice; within a name, id ASC. (The text
    // key must sort DESC via the comparator direction, not a key sentinel.)
    let nd = e
        .execute_relational_select_text("SELECT id FROM people ORDER BY name DESC, id ASC")
        .unwrap();
    assert_eq!(nd.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(id_col(&nd), vec![1, 5, 6, 2, 3, 4], "name DESC / id ASC");

    // (d) TWO text keys: last ASC, first ASC. jones<smith; within smith, first ASC ('al'<'amy'<'bob').
    // Two text slots in the key_plan, both dispatched to the byte compare.
    let lf = e
        .execute_relational_select_text("SELECT id FROM names ORDER BY last ASC, first ASC")
        .unwrap();
    assert_eq!(lf.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        id_col(&lf),
        vec![2, 4, 3, 1],
        "last ASC / first ASC (two text keys)"
    );
}
