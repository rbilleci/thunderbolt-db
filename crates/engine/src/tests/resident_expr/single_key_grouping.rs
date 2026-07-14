use crate::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{Decimal128, SqlValue};

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_group_by() {
    // GROUP BY an int4 key on the general GPU executor (hash aggregation): COUNT/SUM/AVG per group,
    // results sorted by key for determinism. Groups: g=1 -> v{10,20,30}, g=2 -> v{5,15}, g=3 -> v{100}.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(2,5),(1,20),(3,100),(2,15),(1,30)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;

    // GROUP BY g, COUNT(*): g=1->3, g=2->2, g=3->1.
    let c = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("grouped count");
    assert_eq!(
        c.rows,
        vec![vec![i4(1), i8(3)], vec![i4(2), i8(2)], vec![i4(3), i8(1)]],
        "GROUP BY count"
    );
    assert_eq!(c.executed_target, DeviceTarget::Gpu(0));

    // GROUP BY g, SUM(v): g=1->60, g=2->20, g=3->100.
    let s = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("grouped sum");
    assert_eq!(
        s.rows,
        vec![
            vec![i4(1), i8(60)],
            vec![i4(2), i8(20)],
            vec![i4(3), i8(100)]
        ],
        "GROUP BY sum"
    );

    // GROUP BY g, AVG(v): 60/3=20, 20/2=10, 100/1=100 -> numeric scale 16 (PG select_div_scale).
    let a = e
        .execute_resident_expr_select_sql("SELECT g, AVG(v) FROM t GROUP BY g")
        .expect("grouped avg");
    let avg_strs: Vec<String> = a
        .rows
        .iter()
        .map(|r| match &r[1] {
            SqlValue::Numeric(d) => d.to_decimal_string(),
            other => panic!("AVG must be numeric, got {other:?}"),
        })
        .collect();
    assert_eq!(
        a.rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![i4(1), i4(2), i4(3)],
        "AVG keys"
    );
    assert_eq!(
        avg_strs,
        vec![
            "20.0000000000000000",
            "10.0000000000000000",
            "100.0000000000000000"
        ],
        "GROUP BY avg"
    );

    // GROUP BY with a WHERE: group only the surviving rows. v > 10 -> (1,20),(3,100),(2,15),(1,30):
    // g=1 -> 2 (20,30), g=2 -> 1 (15), g=3 -> 1 (100).
    let w = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t WHERE v > 10 GROUP BY g")
        .expect("grouped count + where");
    assert_eq!(
        w.rows,
        vec![vec![i4(1), i8(2)], vec![i4(2), i8(1)], vec![i4(3), i8(1)]],
        "GROUP BY count + WHERE"
    );

    // The result schema is [group key, aggregate]: 2 columns named g + the aggregate.
    assert_eq!(
        c.columns.len(),
        2,
        "grouped result has key + aggregate columns"
    );
    assert_eq!(
        c.columns[0].name, "g",
        "first result column is the group key"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression() {
    // GROUP BY a+b: the expression is materialized ON-DEVICE into a derived int key column the kernel
    // groups by (key_base_override); the result group VALUE is the distinct a+b (not raw a/b), and the
    // SELECT projection of the same expression reads it. Result is key-sorted.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // a+b: (1,2)=3,(2,1)=3,(5,5)=10,(4,4)=8,(3,0)=3,(6,4)=10. groups 3{c:10,20,30}/8{c:40}/10{c:100,200}.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,2,10),(2,1,20),(5,5,100),(4,4,40),(3,0,30),(6,4,200)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let i8 = SqlValue::Int8;
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*), SUM(c) FROM t GROUP BY a+b")
        .expect("GROUP BY a+b");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![i4(3), i8(3), i8(60)],
            vec![i4(8), i8(1), i8(40)],
            vec![i4(10), i8(2), i8(300)],
        ],
        "GROUP BY a+b -> derived group key + per-group count/sum"
    );
    // GROUP BY a*2 -> 6 distinct doubled keys, key-sorted.
    let m = e
        .execute_resident_expr_select_sql("SELECT a*2, COUNT(*) FROM t GROUP BY a*2")
        .expect("GROUP BY a*2");
    assert_eq!(
        m.rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![i4(2), i4(4), i4(6), i4(8), i4(10), i4(12)],
        "GROUP BY a*2 -> 6 doubled keys"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_minmax() {
    // MIN/MAX over a value column with an EXPRESSION group key -- orthogonal mechanisms (group by the
    // derived a+b, MIN/MAX over the real column c). Closes the audit-flagged coverage gap.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, c INT)")
        .unwrap();
    // a+b groups: (1,2)->3{c=10,30}, (5,3)->8{c=40}, (5,5)->10{c=100}.
    e.execute_text(
        2,
        "INSERT INTO t (a,b,c) VALUES (1,2,10),(2,1,30),(5,3,40),(5,5,100)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, MIN(c), MAX(c) FROM t GROUP BY a+b")
        .expect("GROUP BY a+b with MIN/MAX(c)");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Int4(3), SqlValue::Int4(10), SqlValue::Int4(30)],
            vec![SqlValue::Int4(8), SqlValue::Int4(40), SqlValue::Int4(40)],
            vec![SqlValue::Int4(10), SqlValue::Int4(100), SqlValue::Int4(100)],
        ],
        "MIN/MAX(c) per derived a+b group"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_empty_table() {
    // GROUP BY <expr> on an EMPTY table -> 0 groups (PG returns no rows), matching the plain-column
    // path. Guards the audit P1: the on-device arith materialize rejects n=0, so the grouped branch
    // now skips it for 0 rows instead of erroring.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*) FROM t GROUP BY a+b")
        .expect("empty-table GROUP BY a+b -> 0 rows, not an error");
    assert!(
        g.rows.is_empty(),
        "0 groups on empty input, got: {:?}",
        g.rows
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_expression_overflow_is_pg_error() {
    // GROUP BY a+b where a+b overflows int4 -> a clean PG "integer out of range" (checked on-device,
    // no wrap, no CPU), inherited from the arith VM -- same as the WHERE/ORDER BY expression paths.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT)").unwrap();
    e.execute_text(2, "INSERT INTO t (a,b) VALUES (2147483647, 1), (1, 1)")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let err = e
        .execute_resident_expr_select_sql("SELECT a+b, COUNT(*) FROM t GROUP BY a+b")
        .expect_err("a+b overflows int4 -> error");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("out of range") || msg.contains("overflow"),
        "checked overflow -> PG error, got: {err:?}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_execute_resident_expr_select_sql_runs_grouped_min_max() {
    // GROUP BY with per-group MIN/MAX on the general GPU executor (single-level kernel; signed s64
    // atom.min/max). Expected values are CONSTRUCTED from the inserted rows (a GPU-native oracle, not
    // a CPU re-fold): g=1 -> v{10,30,20}; g=2 -> v{5,15,-7}; g=3 -> v{100}. A NEGATIVE value exercises
    // the signed min.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, v INT)").unwrap();
    e.execute_text(
        2,
        "INSERT INTO t (g,v) VALUES (1,10),(2,5),(1,30),(3,100),(2,15),(1,20),(2,-7)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;

    // g=1 -> min 10; g=2 -> min -7 (signed); g=3 -> min 100.
    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t GROUP BY g")
        .expect("grouped min");
    assert_eq!(
        mn.rows,
        vec![
            vec![i4(1), i4(10)],
            vec![i4(2), i4(-7)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY min"
    );
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    // g=1 -> max 30; g=2 -> max 15; g=3 -> max 100.
    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(v) FROM t GROUP BY g")
        .expect("grouped max");
    assert_eq!(
        mx.rows,
        vec![
            vec![i4(1), i4(30)],
            vec![i4(2), i4(15)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY max"
    );

    // MIN with a WHERE: the predicate-filtered indices feed the same kernel. v > 0 drops (2,-7), so
    // g=2's min becomes 5.
    let mw = e
        .execute_resident_expr_select_sql("SELECT g, MIN(v) FROM t WHERE v > 0 GROUP BY g")
        .expect("grouped min + where");
    assert_eq!(
        mw.rows,
        vec![
            vec![i4(1), i4(10)],
            vec![i4(2), i4(5)],
            vec![i4(3), i4(100)]
        ],
        "GROUP BY min + WHERE"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_min_max_over_uuid_value() {
    // GROUP BY an int4 key, MIN/MAX of a UUID value on the general GPU executor. uuid MIN/MAX runs the
    // LOCK-FREE b128 CAS-loop kernel (native 128-bit atomic, sm_90+), comparing the two UNSIGNED 64-bit
    // halves = PG's unsigned big-endian memcmp order. Constructed oracle: each group's min/max is the
    // byte-wise (memcmp) min/max of its uuids -- and SqlValue::Uuid bytes ARE in memcmp order, so the
    // lexicographic min/max of the parsed byte arrays is the definitional answer (not a CPU re-fold of
    // the kernel's logic).
    //
    //   g=1  HIGH limb (bytes 0-7) all tie -> the LOW limb (bytes 8-15) decides, and 0x80.. must
    //        compare UNSIGNED (a signed low-limb compare would mis-rank it as negative):
    //          ..-0100-..   byte 8 = 0x01   low limb 0x0100000000000000
    //          ..-8000-..   byte 8 = 0x80   low limb 0x8000000000000000   <- MAX
    //          ..-..00ff    byte 15 = 0xff  low limb 0x00000000000000ff   <- MIN
    //   g=2  the HIGH limb (bytes 0-7) decides; 0xff.. must compare UNSIGNED and DOMINATE a huge low
    //        limb (the min has the LARGEST possible low limb but the smallest high limb):
    //          00000000-..-ffff-ffffffffffff   uhi 0, ulo max              <- MIN
    //          01000000-..                      uhi 0x0100000000000000
    //          ff000000-..                      uhi 0xff00000000000000      <- MAX
    //   g=3  single row -> min == max == the value.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g INT, u UUID)").unwrap();
    let rows: &[(i32, &str)] = &[
        (1, "00000000-0000-0000-0100-000000000000"),
        (1, "00000000-0000-0000-8000-000000000000"),
        (1, "00000000-0000-0000-0000-0000000000ff"),
        (2, "00000000-0000-0000-ffff-ffffffffffff"),
        (2, "01000000-0000-0000-0000-000000000000"),
        (2, "ff000000-0000-0000-0000-000000000000"),
        (3, "12345678-9abc-def0-1234-56789abcdef0"),
    ];
    let values = rows
        .iter()
        .map(|(g, u)| format!("({g}, '{u}')"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, u) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let i4 = SqlValue::Int4;
    let uuid = |s: &str| SqlValue::Uuid(gpu_db_sql::uuid::parse_uuid(s).expect("valid uuid"));

    // Construction oracle: byte-wise (memcmp) min/max per group, built by sorting the parsed bytes.
    let mut expected_min: Vec<Vec<SqlValue>> = Vec::new();
    let mut expected_max: Vec<Vec<SqlValue>> = Vec::new();
    for g in [1i32, 2, 3] {
        let mut bytes: Vec<[u8; 16]> = rows
            .iter()
            .filter(|(rg, _)| *rg == g)
            .map(|(_, u)| gpu_db_sql::uuid::parse_uuid(u).expect("valid uuid"))
            .collect();
        bytes.sort();
        expected_min.push(vec![i4(g), SqlValue::Uuid(*bytes.first().unwrap())]);
        expected_max.push(vec![i4(g), SqlValue::Uuid(*bytes.last().unwrap())]);
    }

    let mn = e
        .execute_resident_expr_select_sql("SELECT g, MIN(u) FROM t GROUP BY g")
        .expect("uuid grouped min");
    assert_eq!(mn.rows, expected_min, "GROUP BY uuid MIN");
    assert_eq!(mn.executed_target, DeviceTarget::Gpu(0));

    let mx = e
        .execute_resident_expr_select_sql("SELECT g, MAX(u) FROM t GROUP BY g")
        .expect("uuid grouped max");
    assert_eq!(mx.rows, expected_max, "GROUP BY uuid MAX");

    // Spell the discriminating cases out so a regression names itself.
    // g=1: LOW-limb decides; the 0x80.. value is the MAX only under an UNSIGNED compare.
    assert_eq!(
        mn.rows[0],
        vec![i4(1), uuid("00000000-0000-0000-0000-0000000000ff")],
        "g=1 MIN -- late byte (low limb)"
    );
    assert_eq!(
        mx.rows[0],
        vec![i4(1), uuid("00000000-0000-0000-8000-000000000000")],
        "g=1 MAX -- low-limb high bit set, must be unsigned"
    );
    // g=2: HIGH-limb decides and dominates the low limb; ff.. is the MAX only under an UNSIGNED compare.
    assert_eq!(
        mn.rows[1],
        vec![i4(2), uuid("00000000-0000-0000-ffff-ffffffffffff")],
        "g=2 MIN -- smallest high limb despite a huge low limb"
    );
    assert_eq!(
        mx.rows[1],
        vec![i4(2), uuid("ff000000-0000-0000-0000-000000000000")],
        "g=2 MAX -- early byte (high limb), must be unsigned"
    );

    // SUM/AVG over a uuid have no meaning -> hard error, never a wrong row.
    assert!(
        e.execute_resident_expr_select_sql("SELECT g, SUM(u) FROM t GROUP BY g")
            .is_err(),
        "SUM(uuid) => hard error"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_grouped_by_numeric_key() {
    // GROUP BY a NUMERIC (i128) key -- the 128-bit key is claimed via the native atom.cas.b128 into
    // slot_keys_i128 (single-level kernel). Covers a NEGATIVE key + a key whose mantissa exceeds 2^64
    // (non-zero HIGH limb), and reconstructs the mantissa @ the column scale. EMPTY128 = i128::MIN is
    // outside the +/-10^38 numeric range, so no real numeric key ever collides with the sentinel.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (g NUMERIC(30,4), v INT)")
        .unwrap();
    let rows: &[(&str, i32)] = &[
        ("12.5000", 10),
        ("12.5000", 5),
        ("-7.2500", 100),
        ("-7.2500", 3),
        ("1234567890123456789.0000", 1), // mantissa 1.23e22 > 2^64 -> exercises the high limb
        ("1234567890123456789.0000", 2),
    ];
    let values = rows
        .iter()
        .map(|(g, v)| format!("({g}, {v})"))
        .collect::<Vec<_>>()
        .join(",");
    e.execute_text(2, &format!("INSERT INTO t (g, v) VALUES {values}"))
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let num = |m: i128| SqlValue::Numeric(Decimal128::new(m, 4));
    let big: i128 = 1_234_567_890_123_456_789_i128 * 10_000;

    // Output sorts by numeric VALUE: -7.25 < 12.5 < 1.23e18.
    let count = e
        .execute_resident_expr_select_sql("SELECT g, COUNT(*) FROM t GROUP BY g")
        .expect("numeric-key COUNT");
    assert_eq!(count.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        count.rows,
        vec![
            vec![num(-72_500), SqlValue::Int8(2)],
            vec![num(125_000), SqlValue::Int8(2)],
            vec![num(big), SqlValue::Int8(2)],
        ],
        "GROUP BY numeric key, COUNT"
    );

    let sum = e
        .execute_resident_expr_select_sql("SELECT g, SUM(v) FROM t GROUP BY g")
        .expect("numeric-key SUM");
    assert_eq!(
        sum.rows,
        vec![
            vec![num(-72_500), SqlValue::Int8(103)],
            vec![num(125_000), SqlValue::Int8(15)],
            vec![num(big), SqlValue::Int8(3)],
        ],
        "GROUP BY numeric key, SUM(int4)->bigint"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_bool_key() {
    // GROUP BY a BOOL column -- 2 groups (false<true) via the bool->int4 materialize + key_base_override
    // (the audited int4 path; NO bool GROUP BY kernel -> no concurrency hazard).
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (flag BOOL, i INT)")
        .unwrap();
    // false: i={10,30} (count 2, sum 40); true: i={20,40,50} (count 3, sum 110).
    e.execute_text(
        2,
        "INSERT INTO t (flag, i) VALUES (false,10),(true,20),(false,30),(true,40),(true,50)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT flag, COUNT(*), SUM(i) FROM t GROUP BY flag")
        .expect("GROUP BY bool key");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int8(2), SqlValue::Int8(40)],
            vec![SqlValue::Bool(true), SqlValue::Int8(3), SqlValue::Int8(110)],
        ],
        "GROUP BY bool -> false then true, count + sum"
    );
    let m = e
        .execute_resident_expr_select_sql("SELECT flag, MIN(i) FROM t GROUP BY flag")
        .expect("bool key + MIN(int)");
    assert_eq!(
        m.rows,
        vec![
            vec![SqlValue::Bool(false), SqlValue::Int4(10)],
            vec![SqlValue::Bool(true), SqlValue::Int4(20)],
        ],
        "bool key + MIN(int4)"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_group_by_bool_minmax_value() {
    // MIN/MAX over a BOOL VALUE (int key): group all-false -> min=max=false; all-true -> true; mixed ->
    // min=false, max=true. Via bool->int4 materialize + value_base_override.
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE t (k INT, flag BOOL)")
        .unwrap();
    // k=1: {false,false}; k=2: {true,true}; k=3: {false,true}.
    e.execute_text(
        2,
        "INSERT INTO t (k, flag) VALUES (1,false),(1,false),(2,true),(2,true),(3,false),(3,true)",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("t").unwrap();
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let g = e
        .execute_resident_expr_select_sql("SELECT k, MIN(flag), MAX(flag) FROM t GROUP BY k")
        .expect("MIN/MAX over bool value");
    assert_eq!(g.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        g.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Bool(false),
                SqlValue::Bool(false)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Bool(true),
                SqlValue::Bool(true)
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Bool(false),
                SqlValue::Bool(true)
            ],
        ],
        "MIN/MAX(bool) per int group"
    );
}
