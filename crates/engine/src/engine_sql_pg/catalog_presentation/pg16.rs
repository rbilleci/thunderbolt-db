//! Exact PostgreSQL 16 psql verbose relation/attribute display shapes.

use super::*;

pub(in crate::engine_sql_pg) fn catalog_relation_device_sizes(
    engine: &Engine,
) -> BTreeMap<String, u64> {
    // The legacy PostgreSQL-facing heap estimate is derived from the authoritative GPU-residency
    // descriptor, never from a host row shadow. Derive the typed payload from device column layouts
    // because `resident_bytes` intentionally has different accounting meanings for dense snapshots
    // (logical key + values) and shards (allocated device sections). Add PostgreSQL's frozen 24-byte
    // tuple overhead. This remains rename-stable while residency/proof accounting continues to own
    // the actual device allocation.
    let boundary = engine.read_snapshot_boundary();
    let catalog = engine.read_catalog_as_of(boundary);
    let mut sizes = engine
        .read_residency_snapshots()
        .iter()
        .filter(|(_, entry)| entry.descriptor.row_count != 0)
        .map(|(name, entry)| {
            (
                name.clone(),
                pg16_heap_estimate(
                    catalog.relational_catalog.get(name),
                    entry.descriptor.row_count as u64,
                    &entry.descriptor.resident_device_int4_columns,
                    &entry.descriptor.resident_device_int8_columns,
                    &entry.descriptor.resident_device_numeric_columns,
                    entry.descriptor.resident_device_bool_columns.len(),
                    &entry.descriptor.resident_device_text_columns,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (name, shards) in engine.read_residency_shards().iter() {
        let bytes = shards
            .iter()
            .filter(|shard| shard.row_count != 0)
            .map(|shard| {
                pg16_heap_estimate(
                    catalog.relational_catalog.get(name),
                    shard.row_count as u64,
                    &shard.resident_device_int4_columns,
                    &shard.resident_device_int8_columns,
                    &shard.resident_device_numeric_columns,
                    shard.resident_device_bool_columns.len(),
                    &shard.resident_device_text_columns,
                )
            })
            .sum::<u64>();
        if bytes != 0 {
            sizes.insert(name.clone(), bytes);
        }
    }
    for (name, chunks) in engine.read_streaming_cold_chunks().iter() {
        let bytes = chunks
            .chunks
            .iter()
            .filter(|chunk| chunk.row_count != 0)
            .map(|chunk| {
                pg16_heap_estimate(
                    catalog.relational_catalog.get(name),
                    chunk.row_count,
                    &chunk.snapshot.resident_device_int4_columns,
                    &chunk.snapshot.resident_device_int8_columns,
                    &chunk.snapshot.resident_device_numeric_columns,
                    chunk.snapshot.resident_device_bool_columns.len(),
                    &chunk.snapshot.resident_device_text_columns,
                )
            })
            .sum::<u64>();
        if bytes != 0 {
            sizes.insert(name.clone(), bytes);
        }
    }
    sizes
}

fn pg16_heap_estimate(
    table: Option<&RelationalTable>,
    rows: u64,
    int4_columns: &[String],
    int8_columns: &[String],
    wide_columns: &[String],
    bool_columns: usize,
    text_columns: &[ResidentDeviceTextColumnLayout],
) -> u64 {
    let wide_bytes_per_row = wide_columns
        .iter()
        .map(|name| {
            table
                .and_then(|table| table.columns.iter().find(|column| column.name == *name))
                .map_or(16, |column| match column.ty {
                    // The legacy compatibility estimate counted the scale byte as part of a
                    // NUMERIC value even though the GPU layout stores scale in catalog metadata.
                    SqlType::Numeric { .. } => 17,
                    _ => 16,
                })
        })
        .sum::<u64>();
    let fixed_bytes_per_row = (int4_columns.len() as u64)
        .saturating_mul(4)
        .saturating_add((int8_columns.len() as u64).saturating_mul(8))
        .saturating_add(wide_bytes_per_row)
        .saturating_add(bool_columns as u64);
    let text_bytes = text_columns
        .iter()
        .map(|layout| layout.bytes_len)
        .sum::<u64>();
    rows.saturating_mul(24u64.saturating_add(fixed_bytes_per_row))
        .saturating_add(text_bytes)
}

pub(super) fn catalog_attstattarget_case_is_exact(
    case: &pg_query::protobuf::CaseExpr,
    qualifier: &str,
) -> bool {
    if case.arg.is_some() || case.args.len() != 1 {
        return false;
    }
    let Some(otherwise) = case.defresult.as_deref() else {
        return false;
    };
    let Ok(NodeEnum::ColumnRef(otherwise)) = node_enum(otherwise) else {
        return false;
    };
    if resolve_column_name(otherwise, qualifier).ok() != Some("attstattarget") {
        return false;
    }
    let Ok(NodeEnum::CaseWhen(arm)) = node_enum(&case.args[0]) else {
        return false;
    };
    let (Some(condition), Some(result)) = (arm.expr.as_deref(), arm.result.as_deref()) else {
        return false;
    };
    if !matches!(
        result.node.as_ref(),
        Some(NodeEnum::AConst(constant)) if constant.isnull
    ) {
        return false;
    }
    let Ok(NodeEnum::AExpr(condition)) = node_enum(condition) else {
        return false;
    };
    condition.kind == AExprKind::AexprOp as i32
        && aexpr_op_token(condition).ok() == Some("=")
        && condition.lexpr.as_deref().is_some_and(|left| {
            matches!(
                node_enum(left),
                Ok(NodeEnum::ColumnRef(column))
                    if resolve_column_name(column, qualifier).ok() == Some("attstattarget")
            )
        })
        && matches!(
            condition
                .rexpr
                .as_deref()
                .and_then(|right| right.node.as_ref()),
            Some(NodeEnum::AConst(constant))
                if matches!(
                    &constant.val,
                    Some(a_const::Val::Ival(value)) if value.ival == -1
                )
        )
}

pub(super) fn catalog_single_col_description_source(
    function: &pg_query::protobuf::FuncCall,
    table: &str,
    qualifier: &str,
) -> Result<(String, String), ExecuteError> {
    let [relation, attnum] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "col_description requires relation OID and attribute number columns".to_string(),
        ));
    };
    let (NodeEnum::ColumnRef(relation), NodeEnum::ColumnRef(attnum)) =
        (node_enum(relation)?, node_enum(attnum)?)
    else {
        return Err(sql_pg_error(
            "col_description requires relation OID and attribute number columns".to_string(),
        ));
    };
    if !table.ends_with("pg_attribute")
        || resolve_column_name(relation, qualifier)? != "attrelid"
        || resolve_column_name(attnum, qualifier)? != "attnum"
    {
        return Err(sql_pg_error(
            "col_description requires pg_attribute relation and attribute columns".to_string(),
        ));
    }
    Ok((
        GPU_CATALOG_DESCRIPTION.to_string(),
        "col_description".to_string(),
    ))
}

pub(super) fn catalog_relpersistence_case_is_exact(arms: &[(SqlValue, SqlValue)]) -> bool {
    const EXPECTED: [(&str, &str); 3] = [("p", "permanent"), ("t", "temporary"), ("u", "unlogged")];
    arms.len() == EXPECTED.len()
        && arms
            .iter()
            .zip(EXPECTED)
            .all(|((input, output), expected)| {
                input == &SqlValue::Text(expected.0.to_string())
                    && output == &SqlValue::Text(expected.1.to_string())
            })
}

pub(super) fn catalog_reloptions_array_source_is_exact(source: &Node) -> bool {
    let Ok(NodeEnum::AExpr(concat)) = node_enum(source) else {
        return false;
    };
    if concat.kind != AExprKind::AexprOp as i32
        || aexpr_op_token(concat).ok() != Some("||")
        || concat
            .lexpr
            .as_deref()
            .is_none_or(|left| !catalog_column_is_exact(left, "c", "reloptions"))
    {
        return false;
    }
    let Some(right) = concat.rexpr.as_deref() else {
        return false;
    };
    let Ok(NodeEnum::SubLink(sublink)) = node_enum(right) else {
        return false;
    };
    if sublink.sub_link_type != SubLinkType::ArraySublink as i32
        || sublink.testexpr.is_some()
        || !sublink.oper_name.is_empty()
    {
        return false;
    }
    let Some(subselect) = sublink.subselect.as_deref() else {
        return false;
    };
    let Ok(NodeEnum::SelectStmt(select)) = node_enum(subselect) else {
        return false;
    };
    if !catalog_scalar_subselect_is_plain(select)
        || select.where_clause.is_some()
        || select.from_clause.len() != 1
    {
        return false;
    }
    let Some(target) = catalog_target_value(select) else {
        return false;
    };
    let Ok(NodeEnum::AExpr(target_concat)) = node_enum(target) else {
        return false;
    };
    if target_concat.kind != AExprKind::AexprOp as i32
        || aexpr_op_token(target_concat).ok() != Some("||")
        || target_concat
            .lexpr
            .as_deref()
            .is_none_or(|left| !catalog_text_constant_is_exact(left, "toast."))
        || target_concat
            .rexpr
            .as_deref()
            .is_none_or(|right| !catalog_unqualified_column_is_exact(right, "x"))
    {
        return false;
    }
    catalog_reloptions_unnest_range_is_exact(&select.from_clause[0])
}

fn catalog_reloptions_unnest_range_is_exact(node: &Node) -> bool {
    let Ok(NodeEnum::RangeFunction(range)) = node_enum(node) else {
        return false;
    };
    if range.lateral
        || range.ordinality
        || range.is_rowsfrom
        || !range.coldeflist.is_empty()
        || range.functions.len() != 1
        || range
            .alias
            .as_ref()
            .is_none_or(|alias| alias.aliasname != "x" || !alias.colnames.is_empty())
    {
        return false;
    }
    let Some(function) = catalog_range_function_call(&range.functions[0]) else {
        return false;
    };
    catalog_function_name(function).ok().as_deref() == Some("unnest")
        && function.args.len() == 1
        && catalog_column_is_exact(&function.args[0], "tc", "reloptions")
}

fn catalog_range_function_call(node: &Node) -> Option<&pg_query::protobuf::FuncCall> {
    match node.node.as_ref()? {
        NodeEnum::FuncCall(function) => Some(function),
        NodeEnum::List(list) => list.items.iter().find_map(catalog_range_function_call),
        _ => None,
    }
}

fn catalog_unqualified_column_is_exact(node: &Node, expected: &str) -> bool {
    let Ok(NodeEnum::ColumnRef(column)) = node_enum(node) else {
        return false;
    };
    matches!(
        column.fields.as_slice(),
        [name] if matches!(
            name.node.as_ref(),
            Some(NodeEnum::String(name)) if name.sval == expected
        )
    )
}

fn catalog_text_constant_is_exact(node: &Node, expected: &str) -> bool {
    matches!(
        node.node.as_ref(),
        Some(NodeEnum::AConst(constant))
            if matches!(
                &constant.val,
                Some(a_const::Val::Sval(value)) if value.sval == expected
            )
    )
}

pub(super) fn catalog_join_relation_size_source(
    function: &pg_query::protobuf::FuncCall,
) -> Result<JoinColRef, ExecuteError> {
    let [inner] = function.args.as_slice() else {
        return Err(sql_pg_error(
            "pg_size_pretty relation presentation requires one size expression".to_string(),
        ));
    };
    let NodeEnum::FuncCall(inner) = node_enum(inner)? else {
        return Err(sql_pg_error(
            "pg_size_pretty relation presentation requires pg_table_size".to_string(),
        ));
    };
    if catalog_function_name(inner)? != "pg_table_size" {
        return Err(sql_pg_error(
            "pg_size_pretty relation presentation requires pg_table_size".to_string(),
        ));
    }
    let [oid] = inner.args.as_slice() else {
        return Err(sql_pg_error(
            "pg_table_size relation presentation requires one OID column".to_string(),
        ));
    };
    let mut source = parse_join_col_ref(oid)?;
    if source.qualifier.as_deref() != Some("c") || source.column != "oid" {
        return Err(sql_pg_error(
            "pg_table_size relation presentation requires pg_class c.oid".to_string(),
        ));
    }
    source.column = GPU_CATALOG_RELATION_SIZE.to_string();
    Ok(source)
}
