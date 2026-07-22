//! Named-window resolution and inline-window classification for the GPU window route.

use super::*;

pub(super) fn resolve_rank_window_def(
    over: &pg_query::protobuf::WindowDef,
    clauses: &[Node],
    depth: usize,
) -> Result<pg_query::protobuf::WindowDef, ExecuteError> {
    fn named<'a>(
        name: &str,
        clauses: &'a [Node],
    ) -> Result<&'a pg_query::protobuf::WindowDef, ExecuteError> {
        clauses
            .iter()
            .find_map(|node| match node.node.as_ref() {
                Some(NodeEnum::WindowDef(window)) if window.name == name => Some(window.as_ref()),
                _ => None,
            })
            .ok_or_else(|| sql_pg_error(format!("window \"{name}\" does not exist")))
    }
    fn source(
        window: &pg_query::protobuf::WindowDef,
        clauses: &[Node],
        depth: usize,
    ) -> Result<pg_query::protobuf::WindowDef, ExecuteError> {
        if depth > clauses.len().saturating_add(1) {
            return Err(sql_pg_error("cyclic named WINDOW reference".to_string()));
        }
        let mut resolved = if window.refname.is_empty() {
            window.clone()
        } else {
            let mut base = source(named(&window.refname, clauses)?, clauses, depth + 1)?;
            if !window.partition_clause.is_empty() {
                base.partition_clause = window.partition_clause.clone();
            }
            if !window.order_clause.is_empty() {
                base.order_clause = window.order_clause.clone();
            }
            if window.frame_options & 0x00001 != 0 {
                base.frame_options = window.frame_options;
                base.start_offset = window.start_offset.clone();
                base.end_offset = window.end_offset.clone();
            }
            base
        };
        resolved.name.clear();
        resolved.refname.clear();
        Ok(resolved)
    }
    if over.name.is_empty() {
        source(over, clauses, depth)
    } else {
        source(named(&over.name, clauses)?, clauses, depth + 1)
    }
}

pub(super) fn select_has_inline_window(stmt: &SelectStmt) -> Result<bool, ExecuteError> {
    for target in &stmt.target_list {
        let NodeEnum::ResTarget(target) = node_enum(target)? else {
            continue;
        };
        let Some(value) = target.val.as_deref() else {
            continue;
        };
        if matches!(node_enum(value)?, NodeEnum::FuncCall(func) if func.over.is_some()) {
            return Ok(true);
        }
    }
    Ok(false)
}
