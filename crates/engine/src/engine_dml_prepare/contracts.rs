use gpu_db_sql::{SelectFilterOp, SqlType, SqlValue};

use crate::{RelationalTable, WriteSet};

/// One device-resolved DML match: stable entity id, derived entity key, and bounded row image.
pub(crate) type DmlResolvedMatch = (u64, String, Vec<SqlValue>);

/// Applied INSERT data surfaced to residency publication: table, stored row images, conflict
/// write-set, and stable row identities.
pub(crate) type AppliedInsert = (String, Vec<Vec<SqlValue>>, WriteSet, Vec<u64>);

/// ADR-006 (FK child elision, all fk column types): the CANONICAL Eq literal for a device
/// scan-probe of `value` against a column of type `ty` — one arm per device-scannable type,
/// mirroring `dml_filter_groups_to_device_predicate`'s Eq lowering EXACTLY (Date/Uuid round-trip
/// their canonical strings — a raw-days/raw-bytes literal is a hard error in the lowering; Int2
/// compares as its i32 section image; Timestamp as raw micros via the type-discriminating
/// Int8Literal). A mismatched `(ty, value)` pair declines and the caller fails loud.
pub(crate) fn device_eq_scan_literal(
    ty: gpu_db_sql::SqlType,
    value: &SqlValue,
) -> Option<crate::engine_expr::ResidentExpr> {
    use crate::engine_expr::ResidentExpr;
    use gpu_db_sql::SqlType;
    Some(match (ty, value) {
        (SqlType::Int4, SqlValue::Int4(v)) => ResidentExpr::Int4Literal(*v),
        (SqlType::Int2, SqlValue::Int2(v)) => ResidentExpr::Int4Literal(i32::from(*v)),
        (SqlType::Date, SqlValue::Date(v)) => {
            ResidentExpr::TextLiteral(gpu_db_sql::datetime::format_date(*v))
        }
        (SqlType::Int8, SqlValue::Int8(v)) => ResidentExpr::Int8Literal(*v),
        (SqlType::Timestamp, SqlValue::Timestamp(v)) => ResidentExpr::Int8Literal(*v),
        (SqlType::Numeric { .. }, SqlValue::Numeric(d)) => ResidentExpr::NumericLiteral(*d),
        (SqlType::Text, SqlValue::Text(s)) => ResidentExpr::TextLiteral(s.clone()),
        (SqlType::Uuid, SqlValue::Uuid(bytes)) => {
            ResidentExpr::TextLiteral(gpu_db_sql::uuid::format_uuid(bytes))
        }
        (SqlType::Bool, SqlValue::Bool(v)) => ResidentExpr::BoolLiteral(*v),
        _ => return None,
    })
}

/// Canonical device predicate for structural tuple equality. SQL `=` is intentionally not used
/// for NULL members: uniqueness treats two NULL members as the same structural key in this engine,
/// so those leaves lower as `IS NULL`. Keeping this builder shared makes validation and mutation
/// locate use the same typed/NULL semantics when a compound fingerprint is absent or declines.
pub(crate) fn device_structural_tuple_predicate(
    table: &RelationalTable,
    key_cols: &[(usize, SqlValue)],
) -> Option<crate::engine_expr::ResidentExpr> {
    use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};

    let mut predicate = None;
    for leaf in device_structural_tuple_predicates(table, key_cols)? {
        predicate = Some(match predicate {
            None => leaf,
            Some(previous) => ResidentExpr::Binary {
                op: ResidentBinaryOp::And,
                lhs: Box::new(previous),
                rhs: Box::new(leaf),
            },
        });
    }
    predicate
}

/// Independently typed leaves for structural tuple equality. Consumers that must combine unlike
/// physical widths retain each leaf as a device mask and AND those masks on-device.
pub(crate) fn device_structural_tuple_predicates(
    table: &RelationalTable,
    key_cols: &[(usize, SqlValue)],
) -> Option<Vec<crate::engine_expr::ResidentExpr>> {
    use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};

    key_cols
        .iter()
        .map(|(column_idx, value)| {
            if matches!(value, SqlValue::Null) {
                Some(ResidentExpr::IsNull {
                    col: *column_idx,
                    is_not_null: false,
                })
            } else {
                let ty = table.columns.get(*column_idx)?.ty;
                Some(ResidentExpr::Binary {
                    op: ResidentBinaryOp::Eq,
                    lhs: Box::new(ResidentExpr::Column(*column_idx)),
                    rhs: Box::new(device_eq_scan_literal(ty, value)?),
                })
            }
        })
        .collect()
}

/// Lower a DELETE/UPDATE's `filter_groups` (OR of AND-groups) into an
/// `ResidentExpr` DNF (`Column(catalog_idx) <op> literal`, AND within a group, OR across groups) for the
/// device predicate scan-locate. Supports INT4/INT8/TIMESTAMP (I32/I64 VM), NUMERIC (I128 VM), and TEXT
/// equality/ordering/LIKE, UUID, DATE, BOOL, and NULL-aware typed predicates. Any unsupported or
/// empty group returns `None`; production callers fail loud rather than dispatch to a host scan. `Column`
/// carries the FULL-CATALOG index, which `lower_resident_predicate` translates to the shard's section
/// offset (int4 or int8 by the column's catalog type). MIXED-WIDTH groups (int8/timestamp scalar
/// leaves beside int4/text/bool/date/uuid — e.g. `big > 5 AND name = 'x'`) lower at I32 via the
/// width-safe `LoadColumnI64` scalar arms (ADR-006, `mixed_width_i32_elem`); an int8 ARITH subtree
/// in a mixed group still hard-errors on lowering and declines.
pub(crate) fn dml_filter_groups_to_device_predicate(
    table: &RelationalTable,
    filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
) -> Option<crate::engine_expr::ResidentExpr> {
    use crate::engine_expr::{ResidentBinaryOp, ResidentExpr};
    if filter_groups.is_empty() || filter_groups.iter().any(Vec::is_empty) {
        return None;
    }
    let mut dnf: Option<ResidentExpr> = None;
    for group in filter_groups {
        let mut conj: Option<ResidentExpr> = None;
        for (idx, op, value) in group {
            // LIKE-prefix is a TEXT-ONLY op (audit hardening): a non-text column with a `LikePrefix`
            // whose literal coerced to that type (e.g. `ts LIKE '2020-...%'` → Timestamp) must NOT
            // ride an unguarded numeric/timestamp value_leaf arm — decline cleanly here rather than
            // emit a `Column Like <non-text-literal>` that only errors deeper in lowering.
            if matches!(op, SelectFilterOp::LikePrefix)
                && table.columns.get(*idx).map(|c| c.ty) != Some(SqlType::Text)
            {
                return None;
            }
            // The value leaf by the column's section: Int4 -> Int4Literal (I32 VM); Int8 / Timestamp ->
            // Int8Literal (I64 VM — a timestamp is i64 microseconds in the same i64 section, lowered by the
            // timestamp peephole which accepts a raw-micros Int8Literal); Numeric -> NumericLiteral (I128 VM
            // via the numeric peephole, which rescales to the column scale and handles AND/OR). Any other
            // column type / value declines to the host.
            let value_leaf = match (table.columns.get(*idx).map(|c| c.ty), value) {
                (Some(SqlType::Int2), SqlValue::Int2(v)) => {
                    ResidentExpr::Int4Literal(i32::from(*v))
                }
                (Some(SqlType::Int4), SqlValue::Int4(v)) => ResidentExpr::Int4Literal(*v),
                (Some(SqlType::Int8), SqlValue::Int8(v)) => ResidentExpr::Int8Literal(*v),
                (Some(SqlType::Timestamp), SqlValue::Timestamp(v)) => ResidentExpr::Int8Literal(*v),
                // A date bound lowers to the CANONICAL date string (ADR-006 date compound): the date
                // VM leaf / date peephole parse it back to the identical days via `parse_date` (a
                // lossless round-trip, the uuid `format_uuid` pattern). NOT a raw-days Int4Literal —
                // that shape must stay a hard error (`date = 5`, PG semantics) on the read path.
                (Some(SqlType::Date), SqlValue::Date(v)) => {
                    ResidentExpr::TextLiteral(gpu_db_sql::datetime::format_date(*v))
                }
                (Some(SqlType::Numeric { .. }), SqlValue::Numeric(d)) => {
                    ResidentExpr::NumericLiteral(*d)
                }
                // TEXT `=` and `<`/`<=`/`>`/`>=` (ADR-006, charter-pure): lowers to a `TextLiteral` that
                // `try_lower_text_predicate` evaluates on-device — equality via the byte-eq kernel,
                // ordering via the lexicographic byte-compare kernel (memcmp, shorter sorts first,
                // BYTE-IDENTICAL to the host `compare_sql_values` Text order the recheck uses). The
                // located slots materialize their text on-device for the recheck. `LIKE 'p%'` is a
                // separate arm below; no other text op lowers.
                (Some(SqlType::Text), SqlValue::Text(s))
                    if matches!(
                        op,
                        SelectFilterOp::Eq
                            | SelectFilterOp::Lt
                            | SelectFilterOp::Lte
                            | SelectFilterOp::Gt
                            | SelectFilterOp::Gte
                    ) =>
                {
                    ResidentExpr::TextLiteral(s.clone())
                }
                // UUID / BOOL EQUALITY (ADR-006, charter-pure): reuse the DEVICE equality kernels the
                // read path already has — uuid via `try_lower_uuid_predicate` (byte-wise b128 compare;
                // the needle is the canonical uuid string a `TextLiteral` parses back to the same 16
                // bytes), bool via `try_lower_bool_predicate` (the 1-bit bitmap → mask). The recheck
                // compares uuid/bool exactly (`compare_sql_values`). `=` only. The WHERE literal is
                // coerced to the column type at bind (Text→Uuid), so a still-Text value declines here.
                // UUID supports ORDERING too (byte-wise, PG's uuid order == the device kernel's cmp
                // code == the recheck `compare_sql_values`), so `=`/`<`/`<=`/`>`/`>=` all lower;
                // LikePrefix already declined at the text-only guard above.
                (Some(SqlType::Uuid), SqlValue::Uuid(bytes))
                    if matches!(
                        op,
                        SelectFilterOp::Eq
                            | SelectFilterOp::Lt
                            | SelectFilterOp::Lte
                            | SelectFilterOp::Gt
                            | SelectFilterOp::Gte
                    ) =>
                {
                    ResidentExpr::TextLiteral(gpu_db_sql::uuid::format_uuid(bytes))
                }
                // Bool supports ORDERING too (ADR-006 bool inequalities: PG `false < true`; the
                // bool leaves constant-fold `<`/`<=`/`>`/`>=` to equality masks or constants, and
                // the recheck's `compare_sql_values` Bool arm is `bool::cmp` — identical order).
                (Some(SqlType::Bool), SqlValue::Bool(v))
                    if matches!(
                        op,
                        SelectFilterOp::Eq
                            | SelectFilterOp::Lt
                            | SelectFilterOp::Lte
                            | SelectFilterOp::Gt
                            | SelectFilterOp::Gte
                    ) =>
                {
                    ResidentExpr::BoolLiteral(*v)
                }
                // TEXT `LIKE 'prefix%'` (ADR-006, charter-pure): reuse the DEVICE text-LIKE kernel the
                // read path already has (`try_lower_text_predicate`'s `expr_text_like_scalar_filter`).
                // A `LikePrefix` bound carries the BARE literal prefix; reconstruct the faithful escaped
                // `LIKE '<prefix>%'` pattern (byte-identical to the read path's `map_predicate_node`), so
                // the device match == the recheck's `left.starts_with(prefix)`. Text columns only.
                (Some(SqlType::Text), SqlValue::Text(s))
                    if matches!(op, SelectFilterOp::LikePrefix) =>
                {
                    ResidentExpr::TextLiteral(crate::engine_expr::like_pattern_for_literal_prefix(
                        s,
                    ))
                }
                _ => return None,
            };
            let bop = match op {
                SelectFilterOp::Eq => ResidentBinaryOp::Eq,
                SelectFilterOp::Lt => ResidentBinaryOp::Lt,
                SelectFilterOp::Lte => ResidentBinaryOp::Le,
                SelectFilterOp::Gt => ResidentBinaryOp::Gt,
                SelectFilterOp::Gte => ResidentBinaryOp::Ge,
                // Only reached for a text column (the value_leaf `LikePrefix` arm above; every other
                // type's `LikePrefix` already declined at value_leaf) → the device text-LIKE op.
                SelectFilterOp::LikePrefix => ResidentBinaryOp::Like,
            };
            let leaf = ResidentExpr::Binary {
                op: bop,
                lhs: Box::new(ResidentExpr::Column(*idx)),
                rhs: Box::new(value_leaf),
            };
            conj = Some(match conj {
                None => leaf,
                Some(prev) => ResidentExpr::Binary {
                    op: ResidentBinaryOp::And,
                    lhs: Box::new(prev),
                    rhs: Box::new(leaf),
                },
            });
        }
        let c = conj?;
        dnf = Some(match dnf {
            None => c,
            Some(prev) => ResidentExpr::Binary {
                op: ResidentBinaryOp::Or,
                lhs: Box::new(prev),
                rhs: Box::new(c),
            },
        });
    }
    dnf
}

/// Device-history coverage: how much constraint validation `prepare_insert` runs. `Full` everywhere EXCEPT
/// the wave sequencer's under-lock RE-RESOLVE, where unique/CHECK re-validation of an FK-FREE
/// table is PROVABLY REDUNDANT — the coverage argument, verified against the sequencer:
///  - a dup committed at C <= S (the item's read snapshot): the OFF-LOCK prepare validated
///    against every row visible at S and errored the statement before it ever enqueued;
///  - a dup committed in (S, commit] — INCLUDING an earlier item of the SAME wave: the
///    exact device history plus wave-local conflict check runs BEFORE the re-resolve and aborts with a retryable
///    serialization conflict; the item's registered snapshot guard pins version reclamation <= S,
///    so no physical history it needs can vanish mid-flight;
///  - a WITHIN-STATEMENT dup (VALUES (1),(1)): deterministic on the statement text — the
///    off-lock prepare's in-batch check already rejected it;
///  - CHECK constraints are row-local and deterministic on the values: same verdict as the
///    off-lock pass.
///
/// FK re-validation is NOT covered by unique-key device history, so FK-bearing tables always
/// validate fully. PK NOT-NULL (O(new), pure) runs unconditionally as cheap defense.
///
/// PRECONDITION (audit 3b1be580): the skip is granted ONLY while the catalog generation
/// still matches the off-lock prepare's (`CommitWaveItem::prepared_catalog_seq`) — a
/// constraint-adding DDL (ADD UNIQUE/CHECK) committing in (S, wave] is absent from the item's
/// off-lock key projection, so an unguarded skip silently bypassed it (sabotage-verified by
/// `wave_insert_prepared_before_add_check_is_revalidated`). Any DDL bumps the stamp -> Full.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InsertPrepareValidation {
    Full,
    /// Off-lock preparation for an item that is guaranteed to enter the classic commit-wave
    /// sequencer. Eligible single-row unique checks may be deferred to its batched device locate.
    WaveOffLock,
    /// Wave-time fallback after a batched device needle could not bind or a locate declined. This
    /// performs the full validator ladder but MUST NOT re-enter wave deferral, which would turn the
    /// fallback into a no-op (notably for structural NULL unique keys).
    WaveFallbackFull,
    ReResolveDeviceCovered,
}
