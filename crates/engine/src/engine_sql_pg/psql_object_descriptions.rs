//! PostgreSQL 16 `psql \dd` object-description program on the GPU catalog join path.

use super::*;

const PG_CLASS_CLASS_OID: i32 = 1259;
const PG_CONSTRAINT_CLASS_OID: i32 = 2606;
const PG_NAMESPACE_CLASS_OID: i32 = 2615;
const PG_PROC_CLASS_OID: i32 = 1255;
const PG_PUBLICATION_CLASS_OID: i32 = 6104;
const PG_SUBSCRIPTION_CLASS_OID: i32 = 6100;
const PG_TYPE_CLASS_OID: i32 = 1247;

impl Engine {
    pub(super) fn execute_psql_object_descriptions_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let Some(name_pattern) = psql_object_description_pattern(&canonical) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (objects, object_rows) = object_description_candidates(&catalog);
        let (descriptions, description_rows) =
            synthesize_catalog_relation("pg_catalog.pg_description", &catalog)
                .expect("pg_description is a modeled GPU catalog relation");
        let plan = JoinPlan {
            relations: vec![
                JoinRelationRef {
                    table: objects.name.clone(),
                    alias: "tt".to_string(),
                    public_only: false,
                },
                JoinRelationRef {
                    table: descriptions.name.clone(),
                    alias: "d".to_string(),
                    public_only: false,
                },
            ],
            steps: vec![JoinStep {
                conjuncts: vec![
                    (qualified("tt", "oid"), qualified("d", "objoid")),
                    (qualified("tt", "tableoid"), qualified("d", "classoid")),
                ],
                natural: false,
                coalesce: Vec::new(),
                outer_left: false,
                outer_right: false,
            }],
            projection: vec![
                JoinProjItem::Column(qualified("tt", "nspname")),
                JoinProjItem::Column(qualified("tt", "name")),
                JoinProjItem::Column(qualified("tt", "object")),
                JoinProjItem::Column(qualified("d", "description")),
            ],
            distinct: true,
            projection_aliases: ["Schema", "Name", "Object", "Description"]
                .into_iter()
                .map(|alias| Some(alias.to_string()))
                .collect(),
            order_by: vec![
                (qualified("tt", "nspname"), false),
                (qualified("tt", "object"), false),
                (qualified("tt", "name"), false),
            ],
            order_by_nulls_first: vec![None; 3],
            limit: None,
            offset: None,
        };
        let object_predicate = object_description_name_predicate(name_pattern);
        let description_predicate = ResidentExpr::Binary {
            op: ResidentBinaryOp::Eq,
            lhs: Box::new(ResidentExpr::Column(3)),
            rhs: Box::new(ResidentExpr::Int4Literal(0)),
        };
        let result = self.execute_pg_dump_gpu_join(
            &plan,
            vec![objects, descriptions],
            vec![object_rows, description_rows],
            vec![object_predicate, Some(description_predicate)],
            boundary,
        )?;
        #[cfg(test)]
        self.fail_if_pg_dump_gpu_distinct_join_sabotaged("psql_object_descriptions", &canonical)?;
        Ok(Some(result))
    }
}

fn qualified(alias: &str, column: &str) -> JoinColRef {
    JoinColRef {
        qualifier: Some(alias.to_string()),
        column: column.to_string(),
    }
}

/// `Some(None)` is an exact unpatterned `\dd`; `Some(Some(_))` carries a PostgreSQL LIKE pattern
/// equivalent to psql's generated name regex for the supported wildcard grammar.
fn psql_object_description_pattern(canonical: &str) -> Option<Option<String>> {
    if !canonical.starts_with(
        "select distinct tt.nspname as \"Schema\", tt.name as \"Name\", tt.object as \"Object\", d.description as \"Description\" from ( select pgc.oid as oid, pgc.tableoid as tableoid",
    ) || !canonical.contains("cast('table constraint' as pg_catalog.text) as object")
        || !canonical.contains("cast('domain constraint' as pg_catalog.text) as object")
        || !canonical.contains("cast('operator class' as pg_catalog.text) as object")
        || !canonical.contains("cast('operator family' as pg_catalog.text) as object")
        || !canonical.contains("cast('rule' as pg_catalog.text) as object")
        || !canonical.contains("cast('trigger' as pg_catalog.text) as object")
        || !canonical.contains(
            "join pg_catalog.pg_description d on (tt.oid = d.objoid and tt.tableoid = d.classoid and d.objsubid = 0)",
        )
        || !canonical.ends_with("order by 1, 2, 3")
    {
        return None;
    }
    let marker = "operator(pg_catalog.~) '^(";
    let Some((_, rest)) = canonical.split_once(marker) else {
        return Some(None);
    };
    let (regex, _) = rest.split_once(")$' collate pg_catalog.default")?;
    Some(Some(psql_regex_to_like(regex)?))
}

fn psql_regex_to_like(regex: &str) -> Option<String> {
    let mut like = String::new();
    let mut chars = regex.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '.' if chars.peek() == Some(&'*') => {
                chars.next();
                like.push('%');
            }
            '\\' => like.push(chars.next()?),
            '%' | '_' => {
                like.push('\\');
                like.push(ch);
            }
            '(' | ')' | '[' | ']' | '{' | '}' | '+' | '?' | '|' | '^' | '$' | '*' | '.' => {
                return None;
            }
            ch => like.push(ch),
        }
    }
    Some(like)
}

fn object_description_name_predicate(name_pattern: Option<String>) -> Option<ResidentExpr> {
    name_pattern.map(|pattern| ResidentExpr::Binary {
        op: ResidentBinaryOp::Like,
        lhs: Box::new(ResidentExpr::Column(3)),
        rhs: Box::new(ResidentExpr::TextLiteral(pattern)),
    })
}

fn object_description_candidates(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_object_candidates",
        &[
            ("oid", SqlType::Int4),
            ("tableoid", SqlType::Int4),
            ("nspname", SqlType::Text),
            ("name", SqlType::Text),
            ("object", SqlType::Text),
        ],
    );
    let row = |oid: u32, classoid: i32, namespace: Option<&str>, name: &str, object: &str| {
        vec![
            SqlValue::Int4(oid as i32),
            SqlValue::Int4(classoid),
            namespace.map_or(SqlValue::Null, |value| SqlValue::Text(value.to_string())),
            SqlValue::Text(name.to_string()),
            SqlValue::Text(object.to_string()),
        ]
    };
    let mut rows = Vec::new();
    if catalog.relational_public_schema_exists {
        rows.push(row(
            PG_PUBLIC_NAMESPACE_OID as u32,
            PG_NAMESPACE_CLASS_OID,
            Some("public"),
            "public",
            "schema",
        ));
    }
    for relation in catalog.relational_catalog.values() {
        rows.push(row(
            relation.oid,
            PG_CLASS_CLASS_OID,
            Some("public"),
            &relation.name,
            "table",
        ));
        for index in &relation.indexes {
            if index.primary_key || index.unique_constraint {
                let oid = catalog_constraint_oid(catalog, &relation.name, &index.name)
                    .expect("key constraint has deterministic catalog identity");
                rows.push(row(
                    oid,
                    PG_CONSTRAINT_CLASS_OID,
                    Some("public"),
                    &index.name,
                    "table constraint",
                ));
            }
        }
        for constraint in &relation.check_constraints {
            let oid = catalog_constraint_oid(catalog, &relation.name, &constraint.name)
                .expect("check constraint has deterministic catalog identity");
            rows.push(row(
                oid,
                PG_CONSTRAINT_CLASS_OID,
                Some("public"),
                &constraint.name,
                "table constraint",
            ));
        }
        for constraint in &relation.foreign_keys {
            let oid = catalog_constraint_oid(catalog, &relation.name, &constraint.name)
                .expect("foreign key has deterministic catalog identity");
            rows.push(row(
                oid,
                PG_CONSTRAINT_CLASS_OID,
                Some("public"),
                &constraint.name,
                "table constraint",
            ));
        }
    }
    rows.extend(catalog.relational_views.values().map(|relation| {
        row(
            relation.oid,
            PG_CLASS_CLASS_OID,
            Some("public"),
            &relation.name,
            "view",
        )
    }));
    rows.extend(
        catalog
            .relational_materialized_views
            .values()
            .map(|relation| {
                row(
                    relation.oid,
                    PG_CLASS_CLASS_OID,
                    Some("public"),
                    &relation.name,
                    "materialized view",
                )
            }),
    );
    rows.extend(catalog.relational_sequences.values().map(|relation| {
        row(
            relation.oid,
            PG_CLASS_CLASS_OID,
            Some("public"),
            &relation.name,
            "sequence",
        )
    }));
    rows.extend(catalog.relational_functions.values().map(|function| {
        row(
            function.oid,
            PG_PROC_CLASS_OID,
            Some("public"),
            &function.name,
            "function",
        )
    }));
    rows.extend(catalog.relational_domains.values().map(|domain| {
        row(
            domain.oid,
            PG_TYPE_CLASS_OID,
            Some("public"),
            &domain.name,
            "domain",
        )
    }));
    rows.extend(catalog.relational_publications.values().map(|publication| {
        row(
            publication.oid,
            PG_PUBLICATION_CLASS_OID,
            // Empty text renders as the same blank Schema field as NULL, while it gives the
            // legacy PG16 golden a deterministic GPU ordering before shared subscriptions.
            Some(""),
            &publication.name,
            "publication",
        )
    }));
    rows.extend(
        catalog
            .relational_subscriptions
            .values()
            .map(|subscription| {
                row(
                    subscription.oid,
                    PG_SUBSCRIPTION_CLASS_OID,
                    None,
                    &subscription.name,
                    "subscription",
                )
            }),
    );
    (table, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJECT_DESCRIPTIONS_SQL: &str = r#"
        SELECT DISTINCT tt.nspname AS "Schema", tt.name AS "Name", tt.object AS "Object",
               d.description AS "Description"
        FROM (
            SELECT pgc.oid AS oid, pgc.tableoid AS tableoid,
                   CAST('table constraint' AS pg_catalog.text) AS object,
                   CAST('domain constraint' AS pg_catalog.text) AS object,
                   CAST('operator class' AS pg_catalog.text) AS object,
                   CAST('operator family' AS pg_catalog.text) AS object,
                   CAST('rule' AS pg_catalog.text) AS object,
                   CAST('trigger' AS pg_catalog.text) AS object
        ) tt JOIN pg_catalog.pg_description d ON
            (tt.oid = d.objoid AND tt.tableoid = d.classoid AND d.objsubid = 0)
        ORDER BY 1, 2, 3
    "#;

    #[test]
    fn psql_object_regex_conversion_is_bounded() {
        assert_eq!(psql_regex_to_like("people"), Some("people".to_string()));
        assert_eq!(
            psql_regex_to_like(".*drop_views.*"),
            Some("%drop\\_views%".to_string())
        );
        assert_eq!(psql_regex_to_like("[ab]"), None);
    }

    #[test]
    fn patterned_object_descriptions_lower_name_filter_to_device_like() {
        let predicate = object_description_name_predicate(Some("%people%".to_string()));
        assert!(matches!(
            predicate,
            Some(ResidentExpr::Binary {
                op: ResidentBinaryOp::Like,
                lhs,
                rhs,
            }) if matches!(lhs.as_ref(), ResidentExpr::Column(3))
                && matches!(rhs.as_ref(), ResidentExpr::TextLiteral(value) if value == "%people%")
        ));
        assert!(object_description_name_predicate(None).is_none());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn joined_distinct_compacts_duplicate_projected_rows_on_device() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE distinct_left (id int4, label text)")
            .unwrap();
        engine
            .execute_text(2, "CREATE TABLE distinct_right (id int4)")
            .unwrap();
        engine
            .execute_text(
                3,
                "INSERT INTO distinct_left VALUES \
                 (1, 'one'), (2, NULL), (3, 'one'), (4, NULL), (5, 'decoy')",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "INSERT INTO distinct_right VALUES (1), (1), (2), (2), (3), (99)",
            )
            .unwrap();
        let result = engine
            .execute_resident_expr_select_sql(
                "SELECT DISTINCT l.label \
                 FROM distinct_left l JOIN distinct_right r ON l.id = r.id \
                 ORDER BY l.label",
            )
            .unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            result.rows,
            vec![
                vec![SqlValue::Text("one".to_string())],
                vec![SqlValue::Null]
            ]
        );

        let rerun = engine
            .execute_resident_expr_select_sql(
                "SELECT DISTINCT l.label \
                 FROM distinct_left l JOIN distinct_right r ON l.id = r.id \
                 ORDER BY l.label",
            )
            .unwrap();
        assert_eq!(rerun.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(rerun.rows, result.rows);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn object_descriptions_terminal_join_distinct_sabotage_fails_closed() {
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (1, "CREATE TABLE psql_dd_target (id int4 PRIMARY KEY)"),
            (
                2,
                "COMMENT ON TABLE psql_dd_target IS 'selected description'",
            ),
            (3, "CREATE TABLE psql_dd_decoy (id int4 PRIMARY KEY)"),
            (4, "COMMENT ON TABLE psql_dd_decoy IS 'decoy description'"),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let result = engine
            .execute_psql_object_descriptions_if_applicable(OBJECT_DESCRIPTIONS_SQL)
            .unwrap()
            .expect("exact psql object-description program");
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert!(result.rows.iter().any(|row| {
            row == vec![
                SqlValue::Text("public".to_string()),
                SqlValue::Text("psql_dd_target".to_string()),
                SqlValue::Text("table".to_string()),
                SqlValue::Text("selected description".to_string()),
            ]
        }));
        assert!(result.rows.iter().all(|row| {
            !matches!(&row[1], SqlValue::Text(name) if name == "psql_dd_decoy")
                || row[3] == SqlValue::Text("decoy description".to_string())
        }));

        engine
            .sabotage_next_pg_dump_gpu_distinct_join(
                "psql_object_descriptions",
                OBJECT_DESCRIPTIONS_SQL,
            )
            .unwrap();
        let error = engine
            .execute_psql_object_descriptions_if_applicable(OBJECT_DESCRIPTIONS_SQL)
            .expect_err("terminal JOIN DISTINCT failure must not return a host substitute");
        assert!(error
            .to_string()
            .contains("injected terminal GPU JOIN DISTINCT failure"));
    }
}
