//! Typed GPU execution helpers for recognized PostgreSQL 16 catalog programs.
//!
//! Route modules may encode catalog source rows and query-independent presentation values on the
//! host. Query-time filtering, joining, aggregation, ordering, and final value projection must pass
//! through these helpers so every returned scalar is gathered from device memory.

use super::*;

pub(super) fn catalog_column_index(
    table: &RelationalTable,
    column: &str,
) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|candidate| candidate.name == column)
        .ok_or_else(|| sql_pg_error(format!("catalog GPU source is missing column {column:?}")))
}

pub(super) fn int4_comparison(
    table: &RelationalTable,
    column: &str,
    op: ResidentBinaryOp,
    value: i32,
) -> Result<ResidentExpr, ExecuteError> {
    Ok(ResidentExpr::Binary {
        op,
        lhs: Box::new(ResidentExpr::Column(catalog_column_index(table, column)?)),
        rhs: Box::new(ResidentExpr::Int4Literal(value)),
    })
}

pub(super) fn text_comparison(
    table: &RelationalTable,
    column: &str,
    op: ResidentBinaryOp,
    value: &str,
) -> Result<ResidentExpr, ExecuteError> {
    Ok(ResidentExpr::Binary {
        op,
        lhs: Box::new(ResidentExpr::Column(catalog_column_index(table, column)?)),
        rhs: Box::new(ResidentExpr::TextLiteral(value.to_string())),
    })
}

pub(super) fn bool_comparison(
    table: &RelationalTable,
    column: &str,
    value: bool,
) -> Result<ResidentExpr, ExecuteError> {
    Ok(ResidentExpr::Binary {
        op: ResidentBinaryOp::Eq,
        lhs: Box::new(ResidentExpr::Column(catalog_column_index(table, column)?)),
        rhs: Box::new(ResidentExpr::BoolLiteral(value)),
    })
}

pub(super) fn and(left: ResidentExpr, right: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::And,
        lhs: Box::new(left),
        rhs: Box::new(right),
    }
}

pub(super) fn or(left: ResidentExpr, right: ResidentExpr) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Or,
        lhs: Box::new(left),
        rhs: Box::new(right),
    }
}

pub(super) fn any_text_values(
    table: &RelationalTable,
    column: &str,
    values: &[&str],
) -> Result<ResidentExpr, ExecuteError> {
    let mut predicates = values
        .iter()
        .map(|value| text_comparison(table, column, ResidentBinaryOp::Eq, value));
    let first = predicates
        .next()
        .ok_or_else(|| sql_pg_error("catalog GPU membership list is empty".to_string()))??;
    predicates.try_fold(first, |combined, predicate| {
        predicate.map(|predicate| or(combined, predicate))
    })
}

pub(super) fn pg_dump_select(
    table: &RelationalTable,
    projection: SelectProjection,
    order_by: &[&str],
) -> Select {
    Select {
        table: table.name.clone(),
        public_only: false,
        distinct: false,
        projection,
        group_by: None,
        having_groups: Vec::new(),
        filter: None,
        filters: Vec::new(),
        filter_groups: Vec::new(),
        order_by: order_by
            .iter()
            .map(|column| SelectOrder {
                column: (*column).to_string(),
                descending: false,
            })
            .collect(),
        limit: None,
        offset: None,
    }
}

impl Engine {
    pub(super) fn execute_pg_dump_gpu_select(
        &self,
        table: RelationalTable,
        rows: Vec<Vec<SqlValue>>,
        projection: SelectProjection,
        predicate: Option<ResidentExpr>,
        order_by: &[&str],
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let select = pg_dump_select(&table, projection, order_by);
        let bound = bind_relational_select(&table, &select)?;
        let row_count = rows.len() as u64;
        let (snapshot, memory) = self.build_transient_relation_residency(&table, &rows)?;
        let source = ResidentExecSource {
            descriptor: Arc::new(snapshot),
            device_memory: Arc::new(memory),
            row_count,
        };
        self.execute_resident_expr_select_with_binding(
            &select,
            &table,
            Some(&source),
            bound,
            boundary,
            predicate.as_ref(),
            None,
            &vec![None; order_by.len()],
            &vec![None; order_by.len()],
            None,
            &[],
        )
    }

    /// Execute the scalar `COUNT(*) WHERE int4_column = needle` catalog shape as a device
    /// reduction. The general expression executor currently returns `indices.len()` for
    /// `CountAll`; catalog compatibility cannot use that host cardinality shortcut because every
    /// returned catalog scalar must be read from a device result.
    pub(super) fn execute_pg_dump_gpu_int4_equal_count(
        &self,
        table: RelationalTable,
        rows: Vec<Vec<SqlValue>>,
        column: &str,
        needle: i32,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let select = pg_dump_select(&table, SelectProjection::CountAll, &[]);
        let bound = bind_relational_select(&table, &select)?;
        let row_count = u64::try_from(rows.len()).map_err(|_| {
            sql_pg_error("catalog GPU count source exceeds the device row-count range".to_string())
        })?;
        let (snapshot, memory) = self.build_transient_relation_residency(&table, &rows)?;
        let column_idx = catalog_column_index(&table, column)?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, column_idx)?;
        let null_bitmap_offset = resident_device_null_column_offset(&snapshot, &table, column_idx)?;
        let count = memory
            .count_i32_equal_from_payload(byte_offset, row_count, needle, null_bitmap_offset)
            .map_err(|error| ExecuteError::Engine(EngineError::ApplyFailed(error.to_string())))?;
        let count = i64::try_from(count).map_err(|_| {
            sql_pg_error("catalog GPU count exceeds the PostgreSQL bigint range".to_string())
        })?;
        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: vec![vec![SqlValue::Int8(count)]].into(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path: Arc::new(RelationalAccessPath::FullTableScan),
        })
    }

    pub(super) fn execute_pg_dump_gpu_join(
        &self,
        plan: &JoinPlan,
        tables: Vec<RelationalTable>,
        rows: Vec<Vec<Vec<SqlValue>>>,
        predicates: Vec<Option<ResidentExpr>>,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_resident_expr_inner_join(
            plan,
            tables,
            rows.into_iter().map(Some).collect(),
            predicates,
            boundary,
            None,
            None,
            false,
        )
    }
}

pub(super) fn join_relation(table: &RelationalTable, alias: &str) -> JoinRelationRef {
    JoinRelationRef {
        table: table.name.clone(),
        alias: alias.to_string(),
        public_only: false,
    }
}

pub(super) fn join_column(alias: &str, column: &str) -> JoinColRef {
    JoinColRef {
        qualifier: Some(alias.to_string()),
        column: column.to_string(),
    }
}

pub(super) fn join_step(
    left_alias: &str,
    left_column: &str,
    right_alias: &str,
    right_column: &str,
    outer_left: bool,
) -> JoinStep {
    JoinStep {
        conjuncts: vec![(
            join_column(left_alias, left_column),
            join_column(right_alias, right_column),
        )],
        natural: false,
        coalesce: Vec::new(),
        outer_left,
        outer_right: false,
    }
}

pub(super) fn projected_column(alias: &str, column: &str) -> JoinProjItem {
    JoinProjItem::Column(join_column(alias, column))
}

pub(super) fn join_plan(
    relations: &[(&RelationalTable, &str)],
    steps: Vec<JoinStep>,
    projection: Vec<(&str, &str, &str)>,
    order_by: Vec<(&str, &str)>,
) -> JoinPlan {
    let order_by_len = order_by.len();
    JoinPlan {
        relations: relations
            .iter()
            .map(|(table, alias)| join_relation(table, alias))
            .collect(),
        steps,
        projection: projection
            .iter()
            .map(|(alias, column, _)| projected_column(alias, column))
            .collect(),
        projection_aliases: projection
            .into_iter()
            .map(|(_, _, output)| Some(output.to_string()))
            .collect(),
        order_by: order_by
            .into_iter()
            .map(|(alias, column)| (join_column(alias, column), false))
            .collect(),
        order_by_nulls_first: vec![None; order_by_len],
        limit: None,
        offset: None,
    }
}

impl Engine {
    pub(super) fn execute_pg16_dump_class_gpu_plan(
        &self,
        classes: RelationalTable,
        class_rows: Vec<Vec<SqlValue>>,
        catalog: &CatalogSnapshot,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // The modeled catalog has no sequence-ownership dependency rows yet. Keeping an independent
        // typed source is nevertheless important: once ownership is modeled, the existing outer GPU
        // join determines multiplicity and joined values without changing this route boundary.
        let dependencies = catalog_relation_table(
            "pg_catalog",
            "pg_depend",
            &[
                ("objid", SqlType::Int4),
                ("refobjid", SqlType::Int4),
                ("refobjsubid", SqlType::Int4),
            ],
        );
        let dependency_rows = Vec::new();
        let (tablespaces, tablespace_rows) = synthesize_pg_tablespace(catalog);
        let (access_methods, access_method_rows) = synthesize_pg_am();

        let predicate = any_text_values(&classes, "relkind", &["r", "S", "v", "c", "m", "f", "p"])?;
        let projection = vec![
            ("c", "tableoid", "tableoid"),
            ("c", "oid", "oid"),
            ("c", "relname", "relname"),
            ("c", "relnamespace", "relnamespace"),
            ("c", "relkind", "relkind"),
            ("c", "reltype", "reltype"),
            ("c", "relowner", "relowner"),
            ("c", "relchecks", "relchecks"),
            ("c", "relhasindex", "relhasindex"),
            ("c", "relhasrules", "relhasrules"),
            ("c", "relpages", "relpages"),
            ("c", "relhastriggers", "relhastriggers"),
            ("c", "relpersistence", "relpersistence"),
            ("c", "reloftype", "reloftype"),
            ("c", "relacl", "relacl"),
            ("c", "acldefault", "acldefault"),
            ("c", "foreignserver", "foreignserver"),
            ("c", "relfrozenxid", "relfrozenxid"),
            ("tc", "relfrozenxid", "tfrozenxid"),
            ("tc", "oid", "toid"),
            ("tc", "relpages", "toastpages"),
            ("tc", "reloptions", "toast_reloptions"),
            ("d", "refobjid", "owning_tab"),
            ("d", "refobjsubid", "owning_col"),
            ("tsp", "spcname", "reltablespace"),
            ("c", "relhasoids", "relhasoids"),
            ("c", "relispopulated", "relispopulated"),
            ("c", "relreplident", "relreplident"),
            ("c", "relrowsecurity", "relrowsecurity"),
            ("c", "relforcerowsecurity", "relforcerowsecurity"),
            ("c", "relminmxid", "relminmxid"),
            ("tc", "relminmxid", "tminmxid"),
            ("c", "reloptions", "reloptions"),
            ("c", "checkoption", "checkoption"),
            ("am", "amname", "amname"),
            ("c", "is_identity_sequence", "is_identity_sequence"),
            ("c", "ispartition", "ispartition"),
        ];
        let plan = join_plan(
            &[
                (&classes, "c"),
                (&dependencies, "d"),
                (&tablespaces, "tsp"),
                (&access_methods, "am"),
                (&classes, "tc"),
            ],
            vec![
                join_step("c", "oid", "d", "objid", true),
                join_step("c", "__reltablespace_oid", "tsp", "oid", true),
                join_step("c", "__relam", "am", "oid", true),
                join_step("c", "__reltoastrelid", "tc", "oid", true),
            ],
            projection,
            vec![("c", "oid")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![
                classes.clone(),
                dependencies,
                tablespaces,
                access_methods,
                classes,
            ],
            vec![
                class_rows.clone(),
                dependency_rows,
                tablespace_rows,
                access_method_rows,
                class_rows,
            ],
            vec![Some(predicate), None, None, None, None],
            boundary,
        )
    }

    pub(super) fn execute_pg16_dump_function_gpu_plan(
        &self,
        functions: RelationalTable,
        function_rows: Vec<Vec<SqlValue>>,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // Engine functions are user-schema functions. Built-in aggregates, casts, transforms, and
        // internal dependency rows are not represented in this source generation; pg_init_privs is
        // nevertheless an independent source and the LEFT JOIN itself is a GPU operator.
        let init_privs =
            catalog_relation_table("pg_catalog", "pg_init_privs", &[("objoid", SqlType::Int4)]);
        let predicate = text_comparison(&functions, "__prokind", ResidentBinaryOp::Ne, "a")?;
        let projection = [
            "tableoid",
            "oid",
            "proname",
            "prolang",
            "pronargs",
            "proargtypes",
            "prorettype",
            "proacl",
            "acldefault",
            "pronamespace",
            "proowner",
        ]
        .into_iter()
        .map(|column| ("p", column, column))
        .collect();
        let plan = join_plan(
            &[(&functions, "p"), (&init_privs, "pip")],
            vec![join_step("p", "oid", "pip", "objoid", true)],
            projection,
            Vec::new(),
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![functions, init_privs],
            vec![function_rows, Vec::new()],
            vec![Some(predicate), None],
            boundary,
        )
    }

    pub(super) fn execute_pg16_dump_type_gpu_plan(
        &self,
        types: RelationalTable,
        type_rows: Vec<Vec<SqlValue>>,
        catalog: &CatalogSnapshot,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let classes = catalog_relation_table(
            "pg_catalog",
            "pg_class",
            &[("oid", SqlType::Int4), ("relkind", SqlType::Text)],
        );
        let mut class_rows = vec![vec![SqlValue::Int4(0), SqlValue::Text(" ".to_string())]];
        class_rows.extend(
            catalog
                .relational_catalog
                .values()
                .map(|relation| class_kind_row(relation.oid, "r")),
        );
        class_rows.extend(
            catalog
                .relational_views
                .values()
                .map(|relation| class_kind_row(relation.oid, "v")),
        );
        class_rows.extend(
            catalog
                .relational_materialized_views
                .values()
                .map(|relation| class_kind_row(relation.oid, "m")),
        );

        // Arrays are not modeled yet. The oid=0 sentinel represents every modeled base/domain row's
        // `typelem=0`; retaining it as an independent relation keeps the scalar-subquery lookup in the
        // GPU join plan and makes a future array source an additive change.
        let elements = catalog_relation_table(
            "pg_catalog",
            "__pg16_dump_type_elements",
            &[("oid", SqlType::Int4), ("isarray", SqlType::Bool)],
        );
        let element_rows = vec![vec![SqlValue::Int4(0), SqlValue::Bool(false)]];
        let projection = vec![
            ("t", "tableoid", "tableoid"),
            ("t", "oid", "oid"),
            ("t", "typname", "typname"),
            ("t", "typnamespace", "typnamespace"),
            ("t", "typacl", "typacl"),
            ("t", "acldefault", "acldefault"),
            ("t", "typowner", "typowner"),
            ("t", "typelem", "typelem"),
            ("t", "typrelid", "typrelid"),
            ("c", "relkind", "typrelkind"),
            ("t", "typtype", "typtype"),
            ("t", "typisdefined", "typisdefined"),
            ("e", "isarray", "isarray"),
        ];
        let plan = join_plan(
            &[(&types, "t"), (&classes, "c"), (&elements, "e")],
            vec![
                join_step("t", "typrelid", "c", "oid", false),
                join_step("t", "typelem", "e", "oid", false),
            ],
            projection,
            Vec::new(),
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![types, classes, elements],
            vec![type_rows, class_rows, element_rows],
            vec![None, None, None],
            boundary,
        )
    }
}

fn class_kind_row(oid: u32, kind: &str) -> Vec<SqlValue> {
    vec![SqlValue::Int4(oid as i32), SqlValue::Text(kind.to_string())]
}

pub(super) fn oid_source_relation(oids: &[u32]) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_requested_oids",
        &[("tbloid", SqlType::Int4)],
    );
    let rows = oids
        .iter()
        .map(|oid| vec![SqlValue::Int4(*oid as i32)])
        .collect();
    (table, rows)
}
