use super::*;

/// Capture the one immutable generation used by an autocommit libpg_query SELECT.
///
/// Residency repair must precede capture: once the scope is entered, every catalog and device
/// lookup deliberately resolves through the retained descriptor bundle. Repairing afterward would
/// publish a cold replacement that this statement cannot see and could leave a scratch-heavy join
/// or rank window executing against the oversized generation it just retained.
pub(super) fn capture_general_select_statement_snapshot(
    engine: &Engine,
    stmt: &SelectStmt,
) -> Result<Option<Arc<TransactionSnapshot>>, ExecuteError> {
    let mut relation_names = Vec::new();
    for from in &stmt.from_clause {
        collect_range_tables(from, &mut relation_names)?;
    }
    if let Some(snapshot) = engine.current_transaction_read_snapshot() {
        engine.acquire_transaction_table_access(&snapshot, relation_names)?;
        return Ok(None);
    }
    if engine.mvcc_read_skips_leader_check() {
        return Ok(None);
    }

    let _pre_capture_access = relation_names
        .iter()
        .map(|table| engine.acquire_autocommit_table_access(table))
        .collect::<Result<Vec<_>, _>>()?;

    let rank_window = select_has_inline_window(stmt)?;
    let streaming_join = from_clause_is_join(stmt) || stmt.from_clause.len() > 1;
    if rank_window || streaming_join {
        prepare_streaming_inputs(engine, stmt, rank_window)?;
    }

    let commit = engine.commit_state();
    engine
        .ensure_commit_path_available()
        .map_err(ExecuteError::Engine)?;
    let snapshot = engine.capture_statement_snapshot(engine.committed_seq());
    engine.acquire_transaction_table_access(&snapshot, relation_names)?;
    drop(commit);
    Ok(Some(snapshot))
}

fn prepare_streaming_inputs(
    engine: &Engine,
    stmt: &SelectStmt,
    rank_window: bool,
) -> Result<(), ExecuteError> {
    let gpu_id = engine.planner.default_gpu_id();
    let Some(budget) = engine.relational_residency_budget_bytes(gpu_id) else {
        return Ok(());
    };
    if !rank_window && budget < 128 {
        return Ok(());
    }
    let mut relation_names = Vec::new();
    for from in &stmt.from_clause {
        collect_range_tables(from, &mut relation_names)?;
    }
    let relation_count = relation_names.len() as u64;
    let catalog = engine.catalog_snapshot();
    // The streaming join implementation consumes cold user-table inputs only. A synthesized
    // catalog side stays transient and must keep every user peer resident for the mixed GPU join;
    // publishing a cold-only peer here would leave no executable representation after capture.
    if relation_names.is_empty()
        || relation_names
            .iter()
            .any(|table| !catalog.relational_catalog.contains_key(table))
    {
        return Ok(());
    }
    let tables = relation_names.into_iter().collect::<BTreeSet<_>>();
    let input_target = if rank_window {
        (budget / 2).max(1)
    } else if relation_count > 2 {
        (budget / relation_count.saturating_mul(4)).max(1)
    } else {
        (budget / 8).max(1)
    };
    let transition_ceiling = if rank_window {
        input_target
    } else {
        budget.max(1)
    };
    for table_name in &tables {
        engine
            .transition_device_table_to_streaming_repair_above(table_name, transition_ceiling)
            .map_err(ExecuteError::Engine)?;
    }
    if tables
        .iter()
        .all(|table| engine.table_is_gpu_resident(table))
    {
        return Ok(());
    }
    // Streaming joins do not mix a device-authoritative input with cold chunks. Once any relation
    // needs streaming, move every authoritative peer through the same complete repair bridge before
    // building the exact-size chunks. A zero ceiling means "transition any allocated generation";
    // the bridge still refuses a source it cannot prove and never de-authorizes into host execution.
    if !rank_window {
        for table_name in &tables {
            engine
                .transition_device_table_to_streaming_repair_above(table_name, 0)
                .map_err(ExecuteError::Engine)?;
        }
    }
    for table_name in tables {
        if let Some(table) = catalog.relational_catalog.get(&table_name) {
            let _ = engine.ensure_streaming_join_cold(
                table,
                engine.committed_seq(),
                gpu_id,
                input_target,
            );
        }
    }
    Ok(())
}

fn collect_range_tables(node: &Node, tables: &mut Vec<String>) -> Result<(), ExecuteError> {
    match node.node.as_ref() {
        Some(NodeEnum::RangeVar(range)) => {
            tables.push(catalog_range_relation_key(range)?);
            Ok(())
        }
        Some(NodeEnum::JoinExpr(join)) => {
            if let Some(left) = join.larg.as_deref() {
                collect_range_tables(left, tables)?;
            }
            if let Some(right) = join.rarg.as_deref() {
                collect_range_tables(right, tables)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
