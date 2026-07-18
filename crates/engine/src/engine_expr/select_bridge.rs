//! Production general-select entry, grouped, DISTINCT, and declined-shape dispatch bridges.
//! Test-only benchmark helpers remain with their established owner.

use super::execution_source::{ResidentExecSource, ResidentVisibility};
use super::normalization::{
    grouped_projection_to_aggregates, resident_predicate_from_bound_filters,
};
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::{RelationalSelectResult, RelationalTable, RowBlock};
use crate::resident_route::BoundRelationalSelect;
use crate::{Engine, ExecuteError};
use gpu_db_sql::{GroupedAggKind, GroupedAggregate, Select, SelectProjection, SqlValue};
use gpu_db_types::{EngineError, Index};
use std::sync::Arc;

impl Engine {
    /// General GPU executor entry (Charter rule 2): run `SELECT <int4 columns> FROM <table>` filtered
    /// by a general predicate [`ResidentExpr`], evaluating the predicate on the GPU via the device
    /// interpreter and materializing the surviving rows by gathering the projected columns. The
    /// predicate is supplied as an expression tree (the unit of execution is an expression, not a
    /// recognized shape); the `Select` carries the table + projection + MVCC binding. NOT routed
    /// through `resident_route_query_shape` — this is the general path, parallel to the (frozen)
    /// enumerated probe dispatch.
    // Forward API: run a select with a PROGRAMMATIC predicate (binds the catalog internally). The GPU
    // parity tests use it; the production caller is the SQL->Expr entry, which binds once itself and
    // calls `execute_resident_expr_select_with_binding` directly, so this wrapper has no non-test
    // caller yet.
    #[allow(dead_code)]
    pub(crate) fn execute_resident_expr_select(
        &self,
        select: &Select,
        predicate: &ResidentExpr,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        self.execute_resident_expr_select_with_binding(
            select,
            &table,
            None,
            bound,
            copin_s,
            Some(predicate),
            None, // SV3b visibility: single-buffer path is never a versioned shard
            &[],
            &[],
            None,
            &[],
        )
    }

    /// `&Select`->general-executor BRIDGE (S8 grouped int4 aggregates; S10a non-grouped projections).
    /// Runs a `Select` through the SAME on-device general executor as the SQL->Expr path, retiring the
    /// legacy resident-probe methods. For a GROUPED select it normalizes the legacy 1-aggregate
    /// projection and runs the grouped path (the int4-group + int4-value shapes the route classifier
    /// accepts); for a NON-grouped select (`group_by == None`) `grouped_projection_to_aggregates` is a
    /// no-op and the group-key list is EMPTY, so the binding executor runs the plain-projection path
    /// (WHERE VM + GPU sort + LIMIT/OFFSET window) -- this is how S10a routes the `int4_ordered_projection`
    /// shape, replacing its probe. The grouped probes' `!gpu_ordered` branches did a HOST sort / HAVING /
    /// LIMIT (a charter violation -- relational finalization on the host). The general executor does ORDER BY / HAVING / LIMIT
    /// ON-DEVICE (S2/S3/S4), so this is behavior-preserving (enumerated == general proven 0/24
    /// differential; bridge == general re-verified before the probe methods were deleted; the general
    /// grouped ORDER BY now appends a group-key tie-break, matching the legacy group-ASC tie order).
    ///
    /// Unlike `execute_resident_expr_select_sql`, this sources its inputs from the engine `&Select`
    /// (the hand-rolled parse) rather than the libpg_query parse tree, so it covers the text entry AND
    /// the `&Select` callers with no raw SQL -- CTAS and view/matview -- uniformly. The WHERE predicate
    /// is rebuilt from the bound's resolved filters (a `ResidentExpr` DNF -- the third predicate path),
    /// then the bound filters are CLEARED so the executor filters SOLELY via the predicate, exactly as
    /// the SQL->Expr path does (whose bound carries no filters -- the WHERE rides the predicate). The
    /// grouped ORDER BY keys are RESULT columns (the group column or an aggregate), resolved by the
    /// executor against the projection, so every `order_by_exprs` entry is `None` (a plain key, not an
    /// expression); NULLS FIRST/LAST is `None` (PG default) since the hand-rolled `SelectOrder` carries
    /// no explicit override -- byte-identical to the probe path it replaces.
    pub(crate) fn execute_resident_grouped_via_general(
        &self,
        select: &Select,
        // `None` = look up the table's whole-table single resident store by name (the original
        // text/CTAS/view callers). `Some(src)` INJECTS an already-built source (S10c slice 2b: the
        // unified multi-shard buffer recompacted by `execute_resident_sharded_via_general`), so
        // the same on-device grouped/distinct/ordered path serves the sharded shapes.
        src: Option<&ResidentExecSource>,
        // R-ver PART 2: the SV3b/SV6 visibility for a VERSIONED unified `src` — threaded into
        // `indices` so GROUP BY / DISTINCT / ORDER BY group/sort/dedup over VISIBLE rows only
        // (tombstoned + too-new versions never reach the keys). MUST be `None` for a `src: None`
        // caller (with_binding builds + resolves the unified source's visibility itself — the
        // debug_assert there enforces it).
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Normalize the legacy 1-aggregate grouped projection (GroupedCount / GroupedSum / GroupedAvg /
        // GroupedMin / GroupedMax produced by the hand-rolled parser) to the general GroupedAggregates
        // form the Expr executor consumes -- and bind against THAT, so the binding (selected columns,
        // result schema, ORDER-BY-result-column resolution) is byte-identical to the SQL->Expr path,
        // which always produces GroupedAggregates. The WHERE / GROUP BY / ORDER BY / HAVING / LIMIT are
        // carried over unchanged in the clone.
        let mut select_owned = select.clone();
        if let Some(projection) = grouped_projection_to_aggregates(&select.projection) {
            select_owned.projection = projection;
        }
        let select = &select_owned;
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        self.execute_resident_grouped_via_general_with_binding(
            select, &table, bound, copin_s, src, visibility,
        )
    }

    pub(crate) fn execute_resident_grouped_via_general_with_binding(
        &self,
        select: &Select,
        table: &RelationalTable,
        mut bound: BoundRelationalSelect,
        copin_s: Index,
        src: Option<&ResidentExecSource>,
        visibility: Option<ResidentVisibility>,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Rebuild the WHERE predicate from the bound filters BEFORE clearing them; then clear so the
        // executor's filter (and the access-path planner it feeds) sees an empty-filter bound, matching
        // the SQL->Expr path exactly. The predicate ResidentExpr is now the sole filter.
        let predicate = resident_predicate_from_bound_filters(&bound)?;
        bound.filter = None;
        bound.filters.clear();
        bound.filter_groups.clear();
        // Single bare-column GROUP BY (the only grouped shape the route accepts); the executor packs a
        // composite key when len()==2, but the grouped routes are single-key, so this is one column.
        let group_key_columns: Vec<String> = match &select.group_by {
            Some(column) => vec![column.clone()],
            None => Vec::new(),
        };
        // Grouped ORDER BY keys are result columns (None = plain key); no expression GROUP BY here.
        let order_by_exprs: Vec<Option<ResidentExpr>> = vec![None; select.order_by.len()];
        let order_by_nulls_first: Vec<Option<bool>> = vec![None; select.order_by.len()];
        self.execute_resident_expr_select_with_binding(
            select,
            table,
            src,
            bound,
            copin_s,
            predicate.as_ref(),
            // R-ver PART 2: forward the versioned unified src's visibility so the grouped/ordered
            // survivors are the VISIBLE rows (was hard-`None`, which forced the caller refusal).
            visibility,
            &order_by_exprs,
            &order_by_nulls_first,
            None,
            &group_key_columns,
        )
    }

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
    /// caller then fails loudly through the GPU-required boundary.
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
