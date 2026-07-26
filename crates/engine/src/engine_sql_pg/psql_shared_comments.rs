//! Bounded shared-object comments through raw GPU catalog candidates.
//!
//! The host serializes object and comment facts from one pinned catalog generation. The device
//! joins the request to those facts, selects a real comment over an object default, and returns one
//! final scalar. No host object-to-comment lookup or missing-comment decision participates.

use super::*;

const REQUEST_DEFAULT_COMMENT_KEY: &str = "__psql_shared_comment_request_default";

impl Engine {
    pub(super) fn execute_psql_shared_comments_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let Some(program) = shared_comment_program(&canonical) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let request_key = shared_comment_object_key(program.class, program.oid);
        let (request, request_rows) = shared_comment_request_candidates(&request_key);
        let (objects, mut object_rows) = shared_comment_object_candidates(&catalog);
        let (descriptions, mut description_rows) =
            shared_comment_description_candidates(&catalog, &object_rows);
        // This is a typed request fallback, not a catalog lookup. Its lower device order yields
        // only when no complete raw object candidate matches the requested class/OID.
        object_rows.push(vec![
            SqlValue::Text(request_key),
            SqlValue::Text(REQUEST_DEFAULT_COMMENT_KEY.to_string()),
            SqlValue::Int4(1),
        ]);
        description_rows.push(vec![
            SqlValue::Text(REQUEST_DEFAULT_COMMENT_KEY.to_string()),
            SqlValue::Null,
            SqlValue::Bool(true),
            SqlValue::Int4(0),
        ]);
        let projection = if program.is_null {
            "description_is_null"
        } else {
            "description"
        };
        let plan = JoinPlan {
            relations: vec![
                shared_comment_join_relation(&request, "request"),
                shared_comment_join_relation(&objects, "object"),
                shared_comment_join_relation(&descriptions, "description"),
            ],
            steps: vec![
                shared_comment_join_step("request", "__object_key", "object", "__object_key"),
                shared_comment_join_step("object", "__comment_key", "description", "__comment_key"),
            ],
            projection: vec![shared_comment_projected_column("description", projection)],
            distinct: false,
            projection_aliases: vec![Some(program.alias.to_string())],
            order_by: vec![
                (shared_comment_join_column("object", "__priority"), false),
                (
                    shared_comment_join_column("description", "__priority"),
                    false,
                ),
            ],
            order_by_nulls_first: vec![None; 2],
            limit: Some(1),
            offset: None,
        };
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![request, objects, descriptions],
            vec![request_rows, object_rows, description_rows],
            vec![None; 3],
            boundary,
        )
        .map(Some)
    }
}

struct SharedCommentProgram<'a> {
    oid: u32,
    class: &'a str,
    alias: &'a str,
    is_null: bool,
}

fn shared_comment_program(canonical: &str) -> Option<SharedCommentProgram<'_>> {
    let rest = canonical.strip_prefix("select pg_catalog.shobj_description(")?;
    let (args, tail) = rest.split_once(')')?;
    let (oid, class) = args.split_once(',')?;
    let oid = oid.trim().parse().ok()?;
    let class = class.trim().strip_prefix('\'')?.strip_suffix('\'')?;
    if !matches!(class, "pg_authid" | "pg_database" | "pg_tablespace") {
        return None;
    }
    let tail = tail.trim();
    if tail.is_empty() {
        return Some(SharedCommentProgram {
            oid,
            class,
            alias: "shobj_description",
            is_null: false,
        });
    }
    if let Some(alias) = tail.strip_prefix("as ") {
        return valid_alias(alias).then_some(SharedCommentProgram {
            oid,
            class,
            alias,
            is_null: false,
        });
    }
    let alias = tail.strip_prefix("is null as ")?;
    valid_alias(alias).then_some(SharedCommentProgram {
        oid,
        class,
        alias,
        is_null: true,
    })
}

fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn shared_comment_request_candidates(key: &str) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_shared_comment_request",
        &[("__object_key", SqlType::Text)],
    );
    (table, vec![vec![SqlValue::Text(key.to_string())]])
}

fn shared_comment_object_candidates(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_shared_comment_objects",
        &[
            ("__object_key", SqlType::Text),
            ("__comment_key", SqlType::Text),
            ("__priority", SqlType::Int4),
        ],
    );
    let row = |class: &str, oid: u32, kind: &str, name: &str| {
        vec![
            SqlValue::Text(shared_comment_object_key(class, oid)),
            SqlValue::Text(shared_comment_target_key(kind, name)),
            SqlValue::Int4(0),
        ]
    };
    let mut rows = vec![
        row("pg_authid", 10, "role", "postgres"),
        row("pg_database", 5, "database", "postgres"),
        row("pg_tablespace", 1663, "tablespace", "pg_default"),
        row("pg_tablespace", 1664, "tablespace", "pg_global"),
    ];
    rows.extend(
        catalog
            .relational_roles
            .values()
            .map(|role| row("pg_authid", role.oid, "role", &role.name)),
    );
    rows.extend(
        catalog
            .relational_databases
            .values()
            .map(|database| row("pg_database", database.oid, "database", &database.name)),
    );
    rows.extend(catalog.relational_tablespaces.values().map(|tablespace| {
        row(
            "pg_tablespace",
            tablespace.oid,
            "tablespace",
            &tablespace.name,
        )
    }));
    (table, rows)
}

fn shared_comment_description_candidates(
    catalog: &CatalogSnapshot,
    object_rows: &[Vec<SqlValue>],
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_shared_comment_descriptions",
        &[
            ("__comment_key", SqlType::Text),
            ("description", SqlType::Text),
            ("description_is_null", SqlType::Bool),
            ("__priority", SqlType::Int4),
        ],
    );
    // Every object has a raw NULL default. A real raw comment with the same key has a lower
    // device order, so selection remains a GPU ORDER BY/LIMIT decision rather than a host map
    // lookup.
    let mut rows = object_rows
        .iter()
        .filter_map(|row| row.get(1))
        .map(|key| {
            vec![
                key.clone(),
                SqlValue::Null,
                SqlValue::Bool(true),
                SqlValue::Int4(1),
            ]
        })
        .collect::<Vec<_>>();
    rows.extend(
        catalog
            .relational_comments
            .iter()
            .filter_map(|(target, description)| {
                shared_comment_target_key_from_comment(target).map(|key| {
                    vec![
                        SqlValue::Text(key),
                        SqlValue::Text(description.clone()),
                        SqlValue::Bool(false),
                        SqlValue::Int4(0),
                    ]
                })
            }),
    );
    (table, rows)
}

fn shared_comment_target_key_from_comment(target: &RelationalCommentTarget) -> Option<String> {
    match target {
        RelationalCommentTarget::Role { role } => Some(shared_comment_target_key("role", role)),
        RelationalCommentTarget::Database { database } => {
            Some(shared_comment_target_key("database", database))
        }
        RelationalCommentTarget::Tablespace { tablespace } => {
            Some(shared_comment_target_key("tablespace", tablespace))
        }
        _ => None,
    }
}

fn shared_comment_object_key(class: &str, oid: u32) -> String {
    format!("{class}:{oid}")
}

fn shared_comment_target_key(kind: &str, name: &str) -> String {
    format!("{kind}:{name}")
}

fn shared_comment_join_relation(table: &RelationalTable, alias: &str) -> JoinRelationRef {
    JoinRelationRef {
        table: table.name.clone(),
        alias: alias.to_string(),
        public_only: false,
    }
}

fn shared_comment_join_column(alias: &str, column: &str) -> JoinColRef {
    JoinColRef {
        qualifier: Some(alias.to_string()),
        column: column.to_string(),
    }
}

fn shared_comment_join_step(
    left_alias: &str,
    left_column: &str,
    right_alias: &str,
    right_column: &str,
) -> JoinStep {
    JoinStep {
        conjuncts: vec![(
            shared_comment_join_column(left_alias, left_column),
            shared_comment_join_column(right_alias, right_column),
        )],
        natural: false,
        coalesce: Vec::new(),
        outer_left: false,
        outer_right: false,
    }
}

fn shared_comment_projected_column(alias: &str, column: &str) -> JoinProjItem {
    JoinProjItem::Column(shared_comment_join_column(alias, column))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(oid: u32, class: &str, suffix: &str) -> String {
        format!("SELECT pg_catalog.shobj_description({oid}, '{class}') {suffix}")
    }

    fn object_row<'a>(rows: &'a [Vec<SqlValue>], class: &str, oid: u32) -> &'a [SqlValue] {
        let key = SqlValue::Text(shared_comment_object_key(class, oid));
        rows.iter()
            .find(|row| row[0] == key)
            .unwrap_or_else(|| panic!("missing candidate for {class}/{oid}"))
    }

    #[test]
    fn shared_comment_candidates_keep_objects_and_comments_as_separate_raw_relations() {
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (1, "COMMENT ON ROLE postgres IS 'bootstrap role'"),
            (2, "COMMENT ON DATABASE postgres IS 'bootstrap database'"),
            (
                3,
                "COMMENT ON TABLESPACE pg_default IS 'default tablespace'",
            ),
            (4, "CREATE ROLE shared_comment_role"),
            (
                5,
                "COMMENT ON ROLE shared_comment_role IS 'role description'",
            ),
            (6, "CREATE DATABASE shared_comment_database"),
            (
                7,
                "CREATE TABLESPACE shared_comment_tablespace LOCATION '/tmp/shared-comments'",
            ),
            (
                8,
                "COMMENT ON TABLESPACE shared_comment_tablespace IS 'tablespace description'",
            ),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let role_oid = engine.relational_role("shared_comment_role").unwrap().oid;
        let database_oid = engine
            .relational_database("shared_comment_database")
            .unwrap()
            .oid;
        let tablespace_oid = engine
            .relational_tablespace("shared_comment_tablespace")
            .unwrap()
            .oid;
        let catalog = engine.catalog_snapshot();
        let (_, objects) = shared_comment_object_candidates(&catalog);
        let (_, descriptions) = shared_comment_description_candidates(&catalog, &objects);

        assert_eq!(
            object_row(&objects, "pg_authid", 10)[1],
            SqlValue::Text(shared_comment_target_key("role", "postgres"))
        );
        assert!(object_row(&objects, "pg_database", database_oid)
            .iter()
            .all(|value| !matches!(value, SqlValue::Text(text) if text == "bootstrap database")));
        assert_eq!(
            object_row(&objects, "pg_authid", role_oid)[1],
            SqlValue::Text(shared_comment_target_key("role", "shared_comment_role"))
        );
        assert!(object_row(&objects, "pg_tablespace", tablespace_oid)
            .iter()
            .any(|value| value
                == &SqlValue::Text(shared_comment_target_key(
                    "tablespace",
                    "shared_comment_tablespace"
                ))));
        assert!(descriptions.iter().any(|row| {
            row[0] == SqlValue::Text(shared_comment_target_key("role", "postgres"))
                && row[1] == SqlValue::Text("bootstrap role".to_string())
                && row[2] == SqlValue::Bool(false)
        }));
        assert!(descriptions.iter().any(|row| {
            row[0] == SqlValue::Text(shared_comment_target_key("database", "postgres"))
                && row[1] == SqlValue::Text("bootstrap database".to_string())
        }));
    }

    #[test]
    fn shared_comment_invalid_gpu_sabotage_fails_closed() {
        let engine = Engine::with_planner_config(PlannerConfig {
            default_gpu_id: u16::MAX,
        });
        let error = engine
            .execute_resident_expr_select_sql(&query(10, "pg_authid", "AS description"))
            .expect_err("shared-comment route must not fabricate a host scalar");
        let message = error.to_string().to_ascii_lowercase();
        assert!(
            message.contains("gpu") || message.contains("device") || message.contains("cuda"),
            "sabotaged shared-comment route must report a device failure: {error}"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shared_comment_class_and_oid_selection_runs_on_gpu() {
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (1, "CREATE ROLE shared_comment_target"),
            (2, "COMMENT ON ROLE shared_comment_target IS 'selected role'"),
            (3, "CREATE ROLE shared_comment_decoy"),
            (4, "COMMENT ON ROLE shared_comment_decoy IS 'decoy role'"),
            (5, "CREATE DATABASE shared_comment_target_database"),
            (6, "CREATE DATABASE shared_comment_decoy_database"),
            (
                7,
                "COMMENT ON DATABASE shared_comment_decoy_database IS 'decoy database'",
            ),
            (
                8,
                "CREATE TABLESPACE shared_comment_target_space LOCATION '/tmp/shared-comment-target'",
            ),
            (
                9,
                "COMMENT ON TABLESPACE shared_comment_target_space IS 'selected tablespace'",
            ),
            (
                10,
                "CREATE TABLESPACE shared_comment_decoy_space LOCATION '/tmp/shared-comment-decoy'",
            ),
            (
                11,
                "COMMENT ON TABLESPACE shared_comment_decoy_space IS 'decoy tablespace'",
            ),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let role_oid = engine.relational_role("shared_comment_target").unwrap().oid;
        let database_oid = engine
            .relational_database("shared_comment_target_database")
            .unwrap()
            .oid;
        let tablespace_oid = engine
            .relational_tablespace("shared_comment_target_space")
            .unwrap()
            .oid;

        let role = engine
            .execute_resident_expr_select_sql(&query(role_oid, "pg_authid", "AS role_comment"))
            .unwrap();
        assert_eq!(role.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(role.columns[0].name, "role_comment");
        assert_eq!(
            role.rows,
            vec![vec![SqlValue::Text("selected role".to_string())]]
        );

        let database = engine
            .execute_resident_expr_select_sql(&query(
                database_oid,
                "pg_database",
                "IS NULL AS database_comment_is_null",
            ))
            .unwrap();
        assert_eq!(database.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(database.columns[0].name, "database_comment_is_null");
        assert_eq!(database.rows, vec![vec![SqlValue::Bool(true)]]);

        let tablespace = engine
            .execute_resident_expr_select_sql(&query(
                tablespace_oid,
                "pg_tablespace",
                "AS tablespace_comment",
            ))
            .unwrap();
        assert_eq!(tablespace.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            tablespace.rows,
            vec![vec![SqlValue::Text("selected tablespace".to_string())]]
        );

        let missing = engine
            .execute_resident_expr_select_sql(&query(
                2_000_000,
                "pg_authid",
                "IS NULL AS missing_comment_is_null",
            ))
            .unwrap();
        assert_eq!(missing.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(missing.rows, vec![vec![SqlValue::Bool(true)]]);

        let missing_value = engine
            .execute_resident_expr_select_sql(&query(2_000_000, "pg_authid", "AS missing_comment"))
            .unwrap();
        assert_eq!(missing_value.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(missing_value.rows, vec![vec![SqlValue::Null]]);
    }
}
