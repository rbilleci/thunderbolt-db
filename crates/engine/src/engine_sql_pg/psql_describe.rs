//! PostgreSQL `psql \gdesc` result-type presentation over a transient GPU relation.
//!
//! After psql describes the user's statement through the extended protocol, it renders those
//! fields by issuing one stable SQL program:
//!
//! ```sql
//! SELECT name AS "Column", pg_catalog.format_type(tp, tpm) AS "Type"
//! FROM (VALUES (...)) s(name, tp, tpm)
//! ```
//!
//! This leaf recognizes that complete libpg_query AST fail-closed, uploads the raw VALUES rows
//! plus complete pinned type metadata, and receives the final name/type cells from one bounded
//! GPU operator.  Those final device rows are read back once into the protocol result; they are
//! never rebuilt as host `SqlValue` rows and uploaded into a second transient relation.

use super::*;
use gpu_db_execution::{
    DeviceFormatTypeCandidate, DeviceFormatTypeRow, DeviceFormatTypeValue, DeviceFormatTypeVerdict,
};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

#[derive(Debug, PartialEq, Eq)]
struct PsqlDescribeValue {
    name: String,
    type_oid: i32,
    typmod: i32,
}

impl Engine {
    pub(super) fn execute_psql_describe_route_if_applicable(
        &self,
        stmt: &SelectStmt,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Some(values) = psql_describe_values(stmt) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        self.execute_psql_describe_format_types(&values, &catalog)
            .map(Some)
    }

    fn execute_psql_describe_format_types(
        &self,
        values: &[PsqlDescribeValue],
        catalog: &CatalogSnapshot,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let mut input = DeviceFormatTypeInput::default();
        for (ordinal, value) in values.iter().enumerate() {
            let (name_offset, name_len) = input.text(&value.name)?;
            input.values.push(DeviceFormatTypeValue {
                ordinal: u32::try_from(ordinal).map_err(|_| {
                    psql_describe_device_error("VALUES ordinal exceeds device range")
                })?,
                name_offset,
                name_len,
                type_oid: value.type_oid,
                typmod: value.typmod,
            });
        }
        for (oid, display) in modeled_format_type_candidates(catalog) {
            let (display_offset, display_len) = input.text(&display)?;
            input.candidates.push(DeviceFormatTypeCandidate {
                oid,
                display_offset,
                display_len,
            });
        }
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__psql_describe_format_type_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) =
            self.build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])?;
        let (verdict, device_rows) = memory
            .catalog_format_types(&input.values, &input.candidates, &input.bytes)
            .map_err(|error| psql_describe_device_error(error.to_string()))?;
        #[cfg(test)]
        let verdict = if DEVICE_FORMAT_TYPE_SABOTAGE.swap(false, AtomicOrdering::SeqCst) {
            // The kernel above has already consumed the complete candidate relation.  Model a
            // corrupted terminal device return, rather than substituting a host formatter.
            DeviceFormatTypeVerdict::InvalidInput
        } else {
            verdict
        };
        match verdict {
            DeviceFormatTypeVerdict::Complete => {
                let table = catalog_relation_table(
                    "pg_catalog",
                    "__psql_describe_result",
                    &[("Column", SqlType::Text), ("Type", SqlType::Text)],
                );
                psql_describe_result_from_device(table, device_rows, memory.metadata().gpu_id)
            }
            DeviceFormatTypeVerdict::UnknownOid => Err(sql_pg_error(
                "format_type does not model result type OID from the device type relation"
                    .to_string(),
            )),
            DeviceFormatTypeVerdict::InvalidTypmod => Err(sql_pg_error(
                "format_type received invalid numeric typmod on the device".to_string(),
            )),
            DeviceFormatTypeVerdict::InvalidInput => Err(psql_describe_device_error(
                "format_type device operator rejected malformed candidates",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn sabotage_next_psql_describe_format_type_verdict(&self) {
        DEVICE_FORMAT_TYPE_SABOTAGE.store(true, AtomicOrdering::SeqCst);
    }
}

#[cfg(test)]
static DEVICE_FORMAT_TYPE_SABOTAGE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static DEVICE_FORMAT_TYPE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Default)]
struct DeviceFormatTypeInput {
    bytes: Vec<u8>,
    values: Vec<DeviceFormatTypeValue>,
    candidates: Vec<DeviceFormatTypeCandidate>,
}

impl DeviceFormatTypeInput {
    fn text(&mut self, value: &str) -> Result<(u32, u32), ExecuteError> {
        let offset = u32::try_from(self.bytes.len()).map_err(|_| {
            psql_describe_device_error("format_type text staging exceeds device range")
        })?;
        let len = u32::try_from(value.len()).map_err(|_| {
            psql_describe_device_error("format_type text staging exceeds device range")
        })?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok((offset, len))
    }
}

fn modeled_format_type_candidates(catalog: &CatalogSnapshot) -> Vec<(i32, String)> {
    let mut candidates = vec![
        (16, "boolean".to_string()),
        (20, "bigint".to_string()),
        (21, "smallint".to_string()),
        (23, "integer".to_string()),
        (25, "text".to_string()),
        (1082, "date".to_string()),
        (1114, "timestamp without time zone".to_string()),
        (1700, "numeric".to_string()),
        (2950, "uuid".to_string()),
    ];
    candidates.extend(
        catalog
            .relational_domains
            .values()
            .map(|domain| (domain.oid as i32, domain.name.clone())),
    );
    candidates
}

fn psql_describe_result_from_device(
    table: RelationalTable,
    rows: Vec<DeviceFormatTypeRow>,
    gpu_id: u16,
) -> Result<RelationalSelectResult, ExecuteError> {
    let rows = psql_describe_rows_from_device(rows)?;
    Ok(RelationalSelectResult {
        columns: Arc::new(table.columns),
        rows: rows.into(),
        planned_target: DeviceTarget::Gpu(gpu_id),
        executed_target: DeviceTarget::Gpu(gpu_id),
        fallback_reason: None,
        access_path: Arc::new(RelationalAccessPath::FullTableScan),
    })
}

fn psql_describe_rows_from_device(
    rows: Vec<DeviceFormatTypeRow>,
) -> Result<Vec<Vec<SqlValue>>, ExecuteError> {
    rows.into_iter()
        .enumerate()
        .map(|(ordinal, row)| {
            if row.ordinal != ordinal as u32 {
                return Err(psql_describe_device_error(
                    "format_type device operator lost VALUES ordinal",
                ));
            }
            let name = psql_describe_device_text(&row.name, row.name_len)?;
            let display = psql_describe_device_text(&row.display, row.type_len)?;
            Ok(vec![SqlValue::Text(name), SqlValue::Text(display)])
        })
        .collect()
}

fn psql_describe_device_text(bytes: &[u8], len: u32) -> Result<String, ExecuteError> {
    let len = usize::try_from(len)
        .ok()
        .filter(|len| *len <= bytes.len())
        .ok_or_else(|| psql_describe_device_error("format_type device cell length is invalid"))?;
    std::str::from_utf8(&bytes[..len])
        .map(str::to_string)
        .map_err(|_| psql_describe_device_error("format_type device cell is not UTF-8"))
}

fn psql_describe_device_error(detail: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(format!(
        "psql format_type device stage failed closed: {}",
        detail.into()
    )))
}

fn psql_describe_values(stmt: &SelectStmt) -> Option<Vec<PsqlDescribeValue>> {
    if !is_exact_outer_select(stmt)
        || !is_column_target(&stmt.target_list[0], "Column", "name")
        || !is_format_type_target(&stmt.target_list[1])
    {
        return None;
    }
    let NodeEnum::RangeSubselect(range) = stmt.from_clause[0].node.as_ref()? else {
        return None;
    };
    if range.lateral || !is_exact_values_alias(range.alias.as_ref()?) {
        return None;
    }
    let NodeEnum::SelectStmt(values) = range.subquery.as_deref()?.node.as_ref()? else {
        return None;
    };
    if !is_exact_values_select(values) {
        return None;
    }
    values
        .values_lists
        .iter()
        .map(parse_describe_value)
        .collect()
}

fn is_exact_outer_select(stmt: &SelectStmt) -> bool {
    stmt.distinct_clause.is_empty()
        && stmt.into_clause.is_none()
        && stmt.target_list.len() == 2
        && stmt.from_clause.len() == 1
        && stmt.where_clause.is_none()
        && stmt.group_clause.is_empty()
        && !stmt.group_distinct
        && stmt.having_clause.is_none()
        && stmt.window_clause.is_empty()
        && stmt.values_lists.is_empty()
        && stmt.sort_clause.is_empty()
        && stmt.limit_offset.is_none()
        && stmt.limit_count.is_none()
        && stmt.locking_clause.is_empty()
        && stmt.with_clause.is_none()
        && stmt.op == SetOperation::SetopNone as i32
        && !stmt.all
        && stmt.larg.is_none()
        && stmt.rarg.is_none()
}

fn is_exact_values_select(stmt: &SelectStmt) -> bool {
    stmt.distinct_clause.is_empty()
        && stmt.into_clause.is_none()
        && stmt.target_list.is_empty()
        && stmt.from_clause.is_empty()
        && stmt.where_clause.is_none()
        && stmt.group_clause.is_empty()
        && !stmt.group_distinct
        && stmt.having_clause.is_none()
        && stmt.window_clause.is_empty()
        && !stmt.values_lists.is_empty()
        && stmt.sort_clause.is_empty()
        && stmt.limit_offset.is_none()
        && stmt.limit_count.is_none()
        && stmt.locking_clause.is_empty()
        && stmt.with_clause.is_none()
        && stmt.op == SetOperation::SetopNone as i32
        && !stmt.all
        && stmt.larg.is_none()
        && stmt.rarg.is_none()
}

fn is_column_target(node: &Node, alias: &str, column: &str) -> bool {
    let Some(NodeEnum::ResTarget(target)) = node.node.as_ref() else {
        return false;
    };
    target.name == alias
        && target.indirection.is_empty()
        && target
            .val
            .as_deref()
            .is_some_and(|value| is_column_ref(value, column))
}

fn is_format_type_target(node: &Node) -> bool {
    let Some(NodeEnum::ResTarget(target)) = node.node.as_ref() else {
        return false;
    };
    if target.name != "Type" || !target.indirection.is_empty() {
        return false;
    }
    let Some(NodeEnum::FuncCall(function)) =
        target.val.as_deref().and_then(|value| value.node.as_ref())
    else {
        return false;
    };
    is_name_path(&function.funcname, &["pg_catalog", "format_type"])
        && matches!(
            function.args.as_slice(),
            [type_oid, typmod]
                if is_column_ref(type_oid, "tp") && is_column_ref(typmod, "tpm")
        )
        && function.agg_order.is_empty()
        && function.agg_filter.is_none()
        && function.over.is_none()
        && !function.agg_within_group
        && !function.agg_star
        && !function.agg_distinct
        && !function.func_variadic
}

fn is_exact_values_alias(alias: &pg_query::protobuf::Alias) -> bool {
    alias.aliasname == "s" && is_name_path(&alias.colnames, &["name", "tp", "tpm"])
}

fn is_column_ref(node: &Node, expected: &str) -> bool {
    matches!(
        node.node.as_ref(),
        Some(NodeEnum::ColumnRef(column)) if is_name_path(&column.fields, &[expected])
    )
}

fn is_name_path(nodes: &[Node], expected: &[&str]) -> bool {
    nodes.len() == expected.len()
        && nodes.iter().zip(expected).all(|(node, expected)| {
            matches!(
                node.node.as_ref(),
                Some(NodeEnum::String(name)) if name.sval == *expected
            )
        })
}

fn parse_describe_value(node: &Node) -> Option<PsqlDescribeValue> {
    let NodeEnum::List(row) = node.node.as_ref()? else {
        return None;
    };
    let [name, type_oid, typmod] = row.items.as_slice() else {
        return None;
    };
    Some(PsqlDescribeValue {
        name: string_constant(name)?.to_string(),
        type_oid: oid_cast_constant(type_oid)?.parse().ok()?,
        typmod: integer_constant(typmod)?,
    })
}

fn string_constant(node: &Node) -> Option<&str> {
    let NodeEnum::AConst(constant) = node.node.as_ref()? else {
        return None;
    };
    if constant.isnull {
        return None;
    }
    match constant.val.as_ref()? {
        a_const::Val::Sval(value) => Some(value.sval.as_str()),
        _ => None,
    }
}

fn integer_constant(node: &Node) -> Option<i32> {
    let NodeEnum::AConst(constant) = node.node.as_ref()? else {
        return None;
    };
    if constant.isnull {
        return None;
    }
    match constant.val.as_ref()? {
        a_const::Val::Ival(value) => Some(value.ival),
        _ => None,
    }
}

fn oid_cast_constant(node: &Node) -> Option<&str> {
    let NodeEnum::TypeCast(cast) = node.node.as_ref()? else {
        return None;
    };
    let type_name = cast.type_name.as_ref()?;
    if type_name.type_oid != 0
        || type_name.setof
        || type_name.pct_type
        || !type_name.typmods.is_empty()
        || !type_name.array_bounds.is_empty()
        || !is_name_path(&type_name.names, &["pg_catalog", "oid"])
    {
        return None;
    }
    string_constant(cast.arg.as_deref()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESCRIBE_SQL: &str = r#"
        SELECT name AS "Column", pg_catalog.format_type(tp, tpm) AS "Type"
        FROM (VALUES
            ('id', '23'::pg_catalog.oid, -1),
            ('quo''te, column', '25'::pg_catalog.oid, -1),
            ('amount', '1700'::pg_catalog.oid, 786440)
        ) s(name, tp, tpm)
    "#;

    #[test]
    fn recognizes_exact_psql_describe_ast_without_string_splitting() {
        let stmt = parse_single_select(DESCRIBE_SQL).unwrap();
        assert_eq!(
            psql_describe_values(&stmt),
            Some(vec![
                PsqlDescribeValue {
                    name: "id".to_string(),
                    type_oid: 23,
                    typmod: -1,
                },
                PsqlDescribeValue {
                    name: "quo'te, column".to_string(),
                    type_oid: 25,
                    typmod: -1,
                },
                PsqlDescribeValue {
                    name: "amount".to_string(),
                    type_oid: 1700,
                    typmod: 786440,
                },
            ])
        );
    }

    #[test]
    fn near_miss_programs_do_not_enter_the_psql_describe_route() {
        for sql in [
            DESCRIBE_SQL.replace("pg_catalog.format_type", "format_type"),
            DESCRIBE_SQL.replace("AS \"Type\"", "AS \"Other\""),
            DESCRIBE_SQL.replace("::pg_catalog.oid", "::integer"),
            DESCRIBE_SQL.replace(") s(name, tp, tpm)", ") s(name, tp, tpm) WHERE true"),
            DESCRIBE_SQL.replace("786440", "786440 + 0"),
        ] {
            let stmt = parse_single_select(&sql).unwrap();
            assert_eq!(psql_describe_values(&stmt), None, "{sql}");
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn psql_describe_rows_and_format_type_execute_on_the_gpu() {
        let _gpu_test = DEVICE_FORMAT_TYPE_TEST_LOCK.lock().unwrap();
        let result = Engine::new_local_test_engine()
            .execute_resident_expr_select_sql(DESCRIBE_SQL)
            .unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            result.rows,
            vec![
                vec![
                    SqlValue::Text("id".to_string()),
                    SqlValue::Text("integer".to_string()),
                ],
                vec![
                    SqlValue::Text("quo'te, column".to_string()),
                    SqlValue::Text("text".to_string()),
                ],
                vec![
                    SqlValue::Text("amount".to_string()),
                    SqlValue::Text("numeric(12,4)".to_string()),
                ],
            ]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn psql_describe_device_format_type_preserves_order_duplicates_decoys_and_typmod_minus_one() {
        let _gpu_test = DEVICE_FORMAT_TYPE_TEST_LOCK.lock().unwrap();
        let sql = r#"
            SELECT name AS "Column", pg_catalog.format_type(tp, tpm) AS "Type"
            FROM (VALUES
                ('numeric_default', '1700'::pg_catalog.oid, -1),
                ('plain', '23'::pg_catalog.oid, -1),
                ('numeric_scaled', '1700'::pg_catalog.oid, 786440),
                ('plain_again', '23'::pg_catalog.oid, -1)
            ) s(name, tp, tpm)
        "#;
        let engine = Engine::new_local_test_engine();
        // This domain is a complete type-relation decoy: the selected OIDs below must still be
        // resolved by the device join, not by a host builtin-type shortcut.
        engine
            .submit_transaction(
                910,
                gpu_db_sql::ParsedCommand::parse("CREATE DOMAIN gdesc_decoy_type AS int4").unwrap(),
            )
            .unwrap();
        let result = engine.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            result.rows,
            vec![
                vec![
                    SqlValue::Text("numeric_default".to_string()),
                    SqlValue::Text("numeric".to_string())
                ],
                vec![
                    SqlValue::Text("plain".to_string()),
                    SqlValue::Text("integer".to_string())
                ],
                vec![
                    SqlValue::Text("numeric_scaled".to_string()),
                    SqlValue::Text("numeric(12,4)".to_string())
                ],
                vec![
                    SqlValue::Text("plain_again".to_string()),
                    SqlValue::Text("integer".to_string())
                ],
            ]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn psql_describe_unknown_oid_and_invalid_numeric_typmods_fail_at_device_stage() {
        let _gpu_test = DEVICE_FORMAT_TYPE_TEST_LOCK.lock().unwrap();
        for sql in [
            DESCRIBE_SQL.replace("'23'::pg_catalog.oid", "'999999'::pg_catalog.oid"),
            DESCRIBE_SQL.replace("786440", "4"),
            DESCRIBE_SQL.replace("786440", "2555908"),
        ] {
            let error = Engine::new_local_test_engine()
                .execute_resident_expr_select_sql(&sql)
                .unwrap_err();
            assert!(
                error.to_string().contains("format_type") || error.to_string().contains("device"),
                "{error:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn psql_describe_device_join_handles_reordered_candidates_and_duplicate_values() {
        let _gpu_test = DEVICE_FORMAT_TYPE_TEST_LOCK.lock().unwrap();
        let engine = Engine::new_local_test_engine();
        let mut input = DeviceFormatTypeInput::default();
        for (ordinal, name) in ["first", "second"].into_iter().enumerate() {
            let (name_offset, name_len) = input.text(name).unwrap();
            input.values.push(DeviceFormatTypeValue {
                ordinal: ordinal as u32,
                name_offset,
                name_len,
                type_oid: 23,
                typmod: -1,
            });
        }
        // Candidate order is deliberately unrelated to request order, with a nonmatching type
        // preceding the matching OID.  The two values deliberately duplicate their type OID.
        for (oid, display) in [(70_001, "device_decoy"), (23, "integer")] {
            let (display_offset, display_len) = input.text(display).unwrap();
            input.candidates.push(DeviceFormatTypeCandidate {
                oid,
                display_offset,
                display_len,
            });
        }
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__psql_describe_reordered_candidate_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) = engine
            .build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])
            .unwrap();
        let (verdict, rows) = memory
            .catalog_format_types(&input.values, &input.candidates, &input.bytes)
            .unwrap();
        assert_eq!(verdict, DeviceFormatTypeVerdict::Complete);
        assert_eq!(
            psql_describe_rows_from_device(rows).unwrap(),
            vec![
                vec![
                    SqlValue::Text("first".to_string()),
                    SqlValue::Text("integer".to_string())
                ],
                vec![
                    SqlValue::Text("second".to_string()),
                    SqlValue::Text("integer".to_string())
                ],
            ]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn psql_describe_duplicate_type_candidates_and_terminal_sabotage_fail_closed() {
        let _gpu_test = DEVICE_FORMAT_TYPE_TEST_LOCK.lock().unwrap();
        let engine = Engine::new_local_test_engine();
        let mut input = DeviceFormatTypeInput::default();
        let (name_offset, name_len) = input.text("id").unwrap();
        input.values.push(DeviceFormatTypeValue {
            ordinal: 0,
            name_offset,
            name_len,
            type_oid: 23,
            typmod: -1,
        });
        for display in ["integer", "integer_shadow"] {
            let (display_offset, display_len) = input.text(display).unwrap();
            input.candidates.push(DeviceFormatTypeCandidate {
                oid: 23,
                display_offset,
                display_len,
            });
        }
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__psql_describe_duplicate_candidate_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) = engine
            .build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])
            .unwrap();
        assert_eq!(
            memory
                .catalog_format_types(&input.values, &input.candidates, &input.bytes)
                .unwrap()
                .0,
            DeviceFormatTypeVerdict::InvalidInput
        );

        engine.sabotage_next_psql_describe_format_type_verdict();
        let error = engine
            .execute_resident_expr_select_sql(DESCRIBE_SQL)
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::ApplyFailed(_))
        ));
        assert!(error.to_string().contains("failed closed"));
    }
}
