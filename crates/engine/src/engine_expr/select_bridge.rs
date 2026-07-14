//! Production DISTINCT and general-select bridges. Grouped execution and test-only benchmark
//! helpers remain with their established owners.

use super::execution_source::{ResidentExecSource, ResidentVisibility};
use crate::relational_model::{RelationalSelectResult, RowBlock};
use crate::{Engine, ExecuteError};
use gpu_db_sql::{GroupedAggKind, GroupedAggregate, Select, SelectProjection, SqlValue};
use gpu_db_types::EngineError;
use std::sync::Arc;

impl Engine {
    /// S10b: `&Select`->general BRIDGE for a single-column `SELECT DISTINCT` (the int4_[filtered_]distinct
    /// route shapes). DISTINCT over column `a` is exactly `GROUP BY a` with no aggregate; the most-audited
    /// on-device grouped path (S8) is built around >=1 aggregate, so synthesize a `COUNT(*)` grouped select,
    /// run it through `execute_resident_grouped_via_general` (one row per distinct key, ON THE DEVICE -- the
    /// retired probe deduped on a HOST `BTreeSet`), then DROP the trailing COUNT column. WHERE / ORDER BY a /
    /// LIMIT / OFFSET ride the grouped select unchanged. **PG-correct on NULLs** (a behavior change vs the
    /// retired probe, like the S10a ordered projection): GROUP BY groups a NULL key into one group (M3
    /// NULL-key slot) -> DISTINCT yields ONE `SqlValue::Null` row, whereas the NULL-blind probe read the int4
    /// column directly and surfaced a NULL as a phantom `Int4(0)`. For the FILTERED shape a NULL fails the
    /// range predicate (3VL) so it is excluded either way. **No-ORDER-BY order:** the probe returned
    /// first-seen order; the grouped path returns the deterministic default order (key ASC) -- same SET, a
    /// PG-unspecified sequence made deterministic (like the S8 tie-break).
    pub(crate) fn execute_resident_distinct_via_general(
        &self,
        select: &Select,
        // `None` = look up the table's whole-table single resident store by name (the original callers).
        // `Some(src)` forwards an injected source to the grouped bridge it synthesizes (S10c slice 2b:
        // the unified multi-shard buffer), so DISTINCT runs on-device over the whole table.
        src: Option<&ResidentExecSource>,
        // R-ver PART 2: the versioned unified src's visibility, forwarded to the grouped bridge so
        // the distinct SET is over VISIBLE rows only. `None` for a `src: None` caller.
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let SelectProjection::Columns(columns) = &select.projection else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge requires a column projection".to_string(),
            )));
        };
        let [column] = columns.as_slice() else {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident DISTINCT bridge currently supports a single projected column".to_string(),
            )));
        };
        // SELECT DISTINCT a [WHERE ..] [ORDER BY a] [LIMIT ..] == SELECT a, COUNT(*) FROM t [WHERE ..]
        // GROUP BY a [ORDER BY a] [LIMIT ..], with the COUNT column dropped from the result.
        let mut grouped = select.clone();
        grouped.distinct = false;
        grouped.group_by = Some(column.clone());
        grouped.projection = SelectProjection::GroupedAggregates {
            group_column: column.clone(),
            aggregates: vec![GroupedAggregate {
                kind: GroupedAggKind::Count,
                value_column: None,
            }],
        };
        let mut result = self.execute_resident_grouped_via_general(&grouped, src, visibility)?;
        // Drop the trailing COUNT(*) column -> the bare distinct keys (column 0 is the group key).
        // `columns` is now `Arc`-shared; `make_mut` gives an owned `&mut Vec` (no clone — this freshly
        // produced result holds the only reference).
        Arc::make_mut(&mut result.columns).truncate(1);
        // Drop the trailing COUNT(*) value from every row too — `rows` is a flat RowBlock, so reshape it to
        // a single column (the bare distinct group key, column 0).
        let keys: Vec<SqlValue> = result.rows.iter().map(|row| row[0].clone()).collect();
        result.rows = RowBlock::flat(keys, 1);
        Ok(result)
    }

    /// CPU-ENGINE RETIREMENT (ADR-006): the `&Select`->general-executor DISPATCH for the DECLINED-shape
    /// read fallback (`execute_relational_select_instrumented`, when the specialized resident route did
    /// not recognize the shape). Routes by shape to the correct `src: None` sub-bridge — a `SELECT
    /// DISTINCT col` to the DISTINCT bridge (which synthesizes `GROUP BY col` and dedups on-device;
    /// `with_binding` does NOT dedup a bare distinct projection itself), and everything else (GROUP BY /
    /// single-key ORDER BY / plain projection / scalar aggregate) to the grouped bridge, whose non-grouped
    /// path runs the plain projection / aggregate. `src: None` lets `with_binding` resolve the payload —
    /// the whole-table buffer OR the TYPE-COMPLETE unified shard source (THE FLIP), so wider-type
    /// (int8 / numeric / uuid / bool / text) shapes run on-device, unlike the int4-only
    /// `execute_resident_sharded_via_general`. Mirrors that method's shape dispatch (distinct-first, then
    /// grouped for GROUP BY / ORDER BY). Errors (never mis-answers) on a shape it cannot express; the
    /// caller then serves it from the CPU pinned path.
    pub(crate) fn execute_resident_select_via_general(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        if select.distinct {
            self.execute_resident_distinct_via_general(select, None, None)
        } else {
            self.execute_resident_grouped_via_general(select, None, None)
        }
    }
}
