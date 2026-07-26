//! Exact PostgreSQL 16 `psql \d[+]` relation-detail programs.
//!
//! `psql` issues a stable sequence of catalog queries after resolving a relation OID. The general
//! GPU catalog executor handles the relation flags and attribute programs. This leaf owns the
//! remaining mixed comma/outer-join programs: it recognizes the complete normalized SQL
//! fail-closed and derives deterministic metadata from the transaction-pinned catalog. The host
//! only encodes complete candidate relations; query OID filtering, joins, ordering, and final
//! projection execute through the ordinary GPU paths.

use super::*;

const INDEX_PREFIX: &str = "select c2.relname, i.indisprimary, i.indisunique, i.indisclustered, i.indisvalid, pg_catalog.pg_get_indexdef(i.indexrelid, 0, true), pg_catalog.pg_get_constraintdef(con.oid, true), contype, condeferrable, condeferred, i.indisreplident, c2.reltablespace from pg_catalog.pg_class c, pg_catalog.pg_class c2, pg_catalog.pg_index i left join pg_catalog.pg_constraint con on (conrelid = i.indrelid and conindid = i.indexrelid and contype in ('p','u','x')) where c.oid = '";
const INDEX_SUFFIX: &str =
    "' and c.oid = i.indrelid and i.indexrelid = c2.oid order by i.indisprimary desc, c2.relname";
const CHECK_PREFIX: &str = "select r.conname, pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where r.conrelid = '";
const CHECK_SUFFIX: &str = "' and r.contype = 'c' order by 1";
const POLICY_PREFIX: &str = "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '";
const POLICY_SUFFIX: &str = "' order by 1";
const STATISTICS_PREFIX: &str = "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '";
const STATISTICS_SUFFIX: &str = "' order by nsp, stxname";
const INHERITS_PARENT_PREFIX: &str = "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '";
const INHERITS_PARENT_SUFFIX: &str =
    "' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno";
const INHERITS_CHILD_PREFIX: &str = "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '";
const INHERITS_CHILD_SUFFIX: &str = "' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text";

enum RelationDetailProgram {
    Index(u32),
    Check(u32),
    ForeignKeys(u32),
    ReferencedBy(u32),
    Publications(u32),
    Empty(RelationalTable),
}

const INDEX_OUTPUT_COLUMNS: [&str; 12] = [
    "relname",
    "indisprimary",
    "indisunique",
    "indisclustered",
    "indisvalid",
    "pg_get_indexdef",
    "pg_get_constraintdef",
    "contype",
    "condeferrable",
    "condeferred",
    "indisreplident",
    "reltablespace",
];
const CHECK_OUTPUT_COLUMNS: [&str; 2] = ["conname", "pg_get_constraintdef"];
const FOREIGN_KEY_OUTPUT_COLUMNS: [&str; 4] = ["sametable", "conname", "condef", "ontable"];
const PUBLICATION_ALL_TABLES_KEY: &str = "__psql_publication_all_tables";

impl Engine {
    pub(super) fn execute_psql_relation_detail_route_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let Some(program) = relation_detail_program(&canonical) else {
            return Ok(None);
        };
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        match program {
            RelationDetailProgram::Index(oid) => {
                let (table, rows) = index_candidates(&catalog);
                self.execute_pg_dump_gpu_select(
                    table,
                    rows,
                    detail_projection(&INDEX_OUTPUT_COLUMNS),
                    Some(detail_oid_predicate(0, oid)),
                    &["__primary_order", "relname"],
                    boundary,
                )
                .map(Some)
            }
            RelationDetailProgram::Check(oid) => {
                let (table, rows) = check_candidates(&catalog)?;
                self.execute_pg_dump_gpu_select(
                    table,
                    rows,
                    detail_projection(&CHECK_OUTPUT_COLUMNS),
                    Some(detail_oid_predicate(0, oid)),
                    &["conname"],
                    boundary,
                )
                .map(Some)
            }
            RelationDetailProgram::ForeignKeys(oid) => {
                let (table, rows) = foreign_key_candidates(&catalog);
                self.execute_pg_dump_gpu_select(
                    table,
                    rows,
                    detail_projection(&FOREIGN_KEY_OUTPUT_COLUMNS),
                    Some(detail_oid_predicate(0, oid)),
                    &["conname"],
                    boundary,
                )
                .map(Some)
            }
            RelationDetailProgram::ReferencedBy(oid) => self
                .execute_referenced_by_gpu_plan(&catalog, oid, boundary)
                .map(Some),
            RelationDetailProgram::Publications(oid) => self
                .execute_publications_gpu_plan(&catalog, oid, boundary)
                .map(Some),
            RelationDetailProgram::Empty(table) => self
                .execute_pg_dump_transient_relation(table, Vec::new(), &[], boundary)
                .map(Some),
        }
    }

    fn execute_referenced_by_gpu_plan(
        &self,
        catalog: &CatalogSnapshot,
        oid: u32,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (edges, edge_rows) = referenced_by_candidates(catalog);
        let (relations, relation_rows) = relation_name_candidates(catalog);
        self.execute_referenced_by_gpu_candidates(
            edges,
            edge_rows,
            relations,
            relation_rows,
            oid,
            boundary,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_referenced_by_gpu_candidates(
        &self,
        edges: RelationalTable,
        edge_rows: Vec<Vec<SqlValue>>,
        relations: RelationalTable,
        relation_rows: Vec<Vec<SqlValue>>,
        oid: u32,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let plan = JoinPlan {
            relations: vec![
                detail_join_relation(&edges, "edge"),
                detail_join_relation(&relations, "parent"),
                detail_join_relation(&relations, "child"),
            ],
            steps: vec![
                detail_join_step("edge", "__referenced_table", "parent", "name"),
                detail_join_step("edge", "__child_oid", "child", "oid"),
            ],
            projection: vec![
                detail_projected_column("edge", "conname"),
                detail_projected_column("child", "ontable"),
                detail_projected_column("edge", "condef"),
            ],
            distinct: false,
            projection_aliases: vec![None; 3],
            order_by: vec![
                (detail_join_column("edge", "conname"), false),
                (detail_join_column("child", "ontable"), false),
            ],
            order_by_nulls_first: vec![None; 2],
            limit: None,
            offset: None,
        };
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![edges, relations.clone(), relations],
            vec![edge_rows, relation_rows.clone(), relation_rows],
            vec![None, Some(detail_oid_predicate(0, oid)), None],
            boundary,
        )
    }

    fn execute_publications_gpu_plan(
        &self,
        catalog: &CatalogSnapshot,
        oid: u32,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (publications, publication_rows) = publication_candidates(catalog);
        let (memberships, membership_rows) = publication_membership_candidates(catalog);
        let (relations, relation_rows) = publication_relation_membership_candidates(catalog);
        self.execute_publications_gpu_candidates(
            publications,
            publication_rows,
            memberships,
            membership_rows,
            relations,
            relation_rows,
            oid,
            boundary,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_publications_gpu_candidates(
        &self,
        publications: RelationalTable,
        publication_rows: Vec<Vec<SqlValue>>,
        memberships: RelationalTable,
        membership_rows: Vec<Vec<SqlValue>>,
        relations: RelationalTable,
        relation_rows: Vec<Vec<SqlValue>>,
        oid: u32,
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let plan = JoinPlan {
            relations: vec![
                detail_join_relation(&memberships, "member"),
                detail_join_relation(&publications, "publication"),
                detail_join_relation(&relations, "relation"),
            ],
            steps: vec![
                detail_join_step("member", "__publication_oid", "publication", "oid"),
                detail_join_step("member", "__membership_key", "relation", "__membership_key"),
            ],
            projection: vec![
                detail_projected_column("publication", "pubname"),
                detail_projected_column("member", "pg_get_expr"),
                detail_projected_column("member", "case"),
            ],
            // An `all_tables` selector and an explicit selector may reach the same relation. The
            // device DISTINCT owns that result decision just as PostgreSQL's UNION does.
            distinct: true,
            projection_aliases: vec![None; 3],
            order_by: vec![(detail_join_column("publication", "pubname"), false)],
            order_by_nulls_first: vec![None],
            limit: None,
            offset: None,
        };
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![memberships, publications, relations],
            vec![membership_rows, publication_rows, relation_rows],
            vec![None, None, Some(detail_oid_predicate(0, oid))],
            boundary,
        )
    }
}

fn relation_detail_program(canonical: &str) -> Option<RelationDetailProgram> {
    if let Some(oid) = between_oid(canonical, INDEX_PREFIX, INDEX_SUFFIX) {
        return Some(RelationDetailProgram::Index(oid));
    }
    if let Some(oid) = between_oid(canonical, CHECK_PREFIX, CHECK_SUFFIX) {
        return Some(RelationDetailProgram::Check(oid));
    }
    if let Some(oid) = foreign_key_query_oid(canonical) {
        return Some(RelationDetailProgram::ForeignKeys(oid));
    }
    if let Some(oid) = referenced_by_query_oid(canonical) {
        return Some(RelationDetailProgram::ReferencedBy(oid));
    }
    if let Some(oid) = publication_query_oid(canonical) {
        return Some(RelationDetailProgram::Publications(oid));
    }
    if trigger_query_oid(canonical).is_some() {
        return Some(RelationDetailProgram::Empty(catalog_relation_table(
            "pg_catalog",
            "__psql_triggers",
            &[
                ("tgname", SqlType::Text),
                ("pg_get_triggerdef", SqlType::Text),
                ("tgenabled", SqlType::Text),
                ("tgisinternal", SqlType::Bool),
                ("parent", SqlType::Text),
            ],
        )));
    }
    if between_oid(canonical, POLICY_PREFIX, POLICY_SUFFIX).is_some() {
        return Some(RelationDetailProgram::Empty(catalog_relation_table(
            "pg_catalog",
            "__psql_policies",
            &[
                ("polname", SqlType::Text),
                ("polpermissive", SqlType::Bool),
                ("array_to_string", SqlType::Text),
                ("pg_get_expr", SqlType::Text),
                ("pg_get_expr", SqlType::Text),
                ("cmd", SqlType::Text),
            ],
        )));
    }
    if between_oid(canonical, STATISTICS_PREFIX, STATISTICS_SUFFIX).is_some() {
        return Some(RelationDetailProgram::Empty(catalog_relation_table(
            "pg_catalog",
            "__psql_statistics",
            &[
                ("oid", SqlType::Int4),
                ("stxrelid", SqlType::Text),
                ("nsp", SqlType::Text),
                ("stxname", SqlType::Text),
                ("columns", SqlType::Text),
                ("ndist_enabled", SqlType::Bool),
                ("deps_enabled", SqlType::Bool),
                ("mcv_enabled", SqlType::Bool),
                ("stxstattarget", SqlType::Int4),
            ],
        )));
    }
    if between_oid(canonical, INHERITS_PARENT_PREFIX, INHERITS_PARENT_SUFFIX).is_some() {
        return Some(RelationDetailProgram::Empty(catalog_relation_table(
            "pg_catalog",
            "__psql_inherits_parent",
            &[("oid", SqlType::Text)],
        )));
    }
    if between_oid(canonical, INHERITS_CHILD_PREFIX, INHERITS_CHILD_SUFFIX).is_some() {
        return Some(RelationDetailProgram::Empty(catalog_relation_table(
            "pg_catalog",
            "__psql_inherits_child",
            &[
                ("oid", SqlType::Text),
                ("relkind", SqlType::Text),
                ("inhdetachpending", SqlType::Bool),
                ("pg_get_expr", SqlType::Text),
            ],
        )));
    }
    None
}

fn between_oid(canonical: &str, prefix: &str, suffix: &str) -> Option<u32> {
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn foreign_key_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint r")
        || !canonical.contains("r.contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    quoted_oid_after(canonical, "r.conrelid = '")
}

fn referenced_by_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint c")
        || !canonical.contains("confrelid in (select pg_catalog.pg_partition_ancestors('")
        || !canonical.contains("and contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    quoted_oid_after(canonical, "pg_partition_ancestors('")
}

fn trigger_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_trigger t")
        || !canonical.contains("pg_catalog.pg_get_triggerdef(t.oid, true)")
    {
        return None;
    }
    quoted_oid_after(canonical, "where t.tgrelid = '")
}

fn publication_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.starts_with("select pubname , null , null from pg_catalog.pg_publication p")
        || !canonical.contains("join pg_catalog.pg_publication_namespace pn")
        || !canonical.contains("union select pubname , pg_get_expr(pr.prqual, c.oid)")
        || !canonical.contains("join pg_catalog.pg_publication_rel pr")
        || !canonical.contains("union select pubname , null , null")
        || !canonical.ends_with("order by 1")
    {
        return None;
    }
    quoted_oid_after(canonical, "where pc.oid ='")
}

fn quoted_oid_after(canonical: &str, marker: &str) -> Option<u32> {
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn index_candidates(catalog: &CatalogSnapshot) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_indexes",
        &[
            ("__table_oid", SqlType::Int4),
            ("__primary_order", SqlType::Int4),
            ("relname", SqlType::Text),
            ("indisprimary", SqlType::Bool),
            ("indisunique", SqlType::Bool),
            ("indisclustered", SqlType::Bool),
            ("indisvalid", SqlType::Bool),
            ("pg_get_indexdef", SqlType::Text),
            ("pg_get_constraintdef", SqlType::Text),
            ("contype", SqlType::Text),
            ("condeferrable", SqlType::Bool),
            ("condeferred", SqlType::Bool),
            ("indisreplident", SqlType::Bool),
            ("reltablespace", SqlType::Int4),
        ],
    );
    // Every modeled index enters this immutable snapshot.  The requested table OID remains a
    // typed hidden column so the GPU filter owns route selection.
    let rows = catalog
        .relational_catalog
        .values()
        .flat_map(|relation| {
            relation.indexes.iter().map(move |index| {
                let constraint = index.primary_key || index.unique_constraint;
                let constraint_label = if index.primary_key {
                    "PRIMARY KEY"
                } else {
                    "UNIQUE"
                };
                let contype = if index.primary_key { "p" } else { "u" };
                vec![
                    SqlValue::Int4(relation.oid as i32),
                    SqlValue::Int4(if index.primary_key { 0 } else { 1 }),
                    SqlValue::Text(index.name.clone()),
                    SqlValue::Bool(index.primary_key),
                    SqlValue::Bool(index.unique),
                    SqlValue::Bool(false),
                    SqlValue::Bool(true),
                    SqlValue::Text(format!(
                        "CREATE {}INDEX {} ON public.{} USING btree ({})",
                        if index.unique { "UNIQUE " } else { "" },
                        index.name,
                        relation.name,
                        index.key_columns.join(", ")
                    )),
                    if constraint {
                        SqlValue::Text(format!(
                            "{constraint_label} ({})",
                            index.key_columns.join(", ")
                        ))
                    } else {
                        SqlValue::Null
                    },
                    if constraint {
                        SqlValue::Text(contype.to_string())
                    } else {
                        SqlValue::Null
                    },
                    if constraint {
                        SqlValue::Bool(false)
                    } else {
                        SqlValue::Null
                    },
                    if constraint {
                        SqlValue::Bool(false)
                    } else {
                        SqlValue::Null
                    },
                    SqlValue::Bool(false),
                    SqlValue::Int4(0),
                ]
            })
        })
        .collect();
    (table, rows)
}

fn check_candidates(
    catalog: &CatalogSnapshot,
) -> Result<(RelationalTable, Vec<Vec<SqlValue>>), ExecuteError> {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_checks",
        &[
            ("__table_oid", SqlType::Int4),
            ("conname", SqlType::Text),
            ("pg_get_constraintdef", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        for constraint in &relation.check_constraints {
            rows.push(vec![
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text(constraint.name.clone()),
                SqlValue::Text(check_constraint_definition(constraint)?),
            ]);
        }
    }
    Ok((table, rows))
}

fn check_constraint_definition(
    constraint: &RelationalCheckConstraint,
) -> Result<String, ExecuteError> {
    let operator = match constraint.op {
        SelectFilterOp::Eq => "=",
        SelectFilterOp::Lt => "<",
        SelectFilterOp::Lte => "<=",
        SelectFilterOp::Gt => ">",
        SelectFilterOp::Gte => ">=",
        SelectFilterOp::LikePrefix => "LIKE",
    };
    let value = render_sql_value_literal(&constraint.value).map_err(ExecuteError::Engine)?;
    Ok(format!(
        "CHECK (({} {operator} {value}))",
        constraint.column
    ))
}

fn foreign_key_candidates(catalog: &CatalogSnapshot) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_foreign_keys",
        &[
            ("__table_oid", SqlType::Int4),
            ("sametable", SqlType::Bool),
            ("conname", SqlType::Text),
            ("condef", SqlType::Text),
            ("ontable", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_catalog
        .values()
        .flat_map(|relation| {
            relation.foreign_keys.iter().map(move |constraint| {
                vec![
                    SqlValue::Int4(relation.oid as i32),
                    SqlValue::Bool(true),
                    SqlValue::Text(constraint.name.clone()),
                    SqlValue::Text(foreign_key_definition(constraint)),
                    SqlValue::Text(format!("public.{}", relation.name)),
                ]
            })
        })
        .collect();
    (table, rows)
}

fn referenced_by_candidates(catalog: &CatalogSnapshot) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_referenced_by_edges",
        &[
            ("__child_oid", SqlType::Int4),
            ("__referenced_table", SqlType::Text),
            ("conname", SqlType::Text),
            ("condef", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_catalog
        .values()
        .flat_map(|relation| {
            relation.foreign_keys.iter().map(|constraint| {
                vec![
                    SqlValue::Int4(relation.oid as i32),
                    SqlValue::Text(constraint.referenced_table.clone()),
                    SqlValue::Text(constraint.name.clone()),
                    SqlValue::Text(foreign_key_definition(constraint)),
                ]
            })
        })
        .collect();
    (table, rows)
}

fn relation_name_candidates(catalog: &CatalogSnapshot) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_relation_names",
        &[
            ("oid", SqlType::Int4),
            ("name", SqlType::Text),
            ("ontable", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_catalog
        .values()
        .map(|relation| {
            vec![
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Text(relation.name.clone()),
                SqlValue::Text(format!("public.{}", relation.name)),
            ]
        })
        .collect();
    (table, rows)
}

fn publication_candidates(catalog: &CatalogSnapshot) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_publication_candidates",
        &[("oid", SqlType::Int4), ("pubname", SqlType::Text)],
    );
    let rows = catalog
        .relational_publications
        .values()
        .map(|publication| {
            vec![
                SqlValue::Int4(publication.oid as i32),
                SqlValue::Text(publication.name.clone()),
            ]
        })
        .collect();
    (table, rows)
}

fn publication_membership_candidates(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_publication_memberships",
        &[
            ("__publication_oid", SqlType::Int4),
            ("__membership_key", SqlType::Text),
            ("pg_get_expr", SqlType::Text),
            ("case", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_publications
        .values()
        .flat_map(|publication| {
            publication
                .all_tables
                .then_some(PUBLICATION_ALL_TABLES_KEY.to_string())
                .into_iter()
                .chain(publication.tables.iter().cloned())
                .map(move |membership_key| {
                    vec![
                        SqlValue::Int4(publication.oid as i32),
                        SqlValue::Text(membership_key),
                        SqlValue::Null,
                        SqlValue::Null,
                    ]
                })
        })
        .collect();
    (table, rows)
}

fn publication_relation_membership_candidates(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__psql_publication_relation_memberships",
        &[
            ("__table_oid", SqlType::Int4),
            ("__membership_key", SqlType::Text),
        ],
    );
    // Each relation has its raw name key plus the all-tables key. The GPU joins a publication's
    // explicit names or its all-tables selector to this complete relation; no host loop expands a
    // publication into result rows.
    let rows = catalog
        .relational_catalog
        .values()
        .flat_map(|relation| {
            [
                relation.name.clone(),
                PUBLICATION_ALL_TABLES_KEY.to_string(),
            ]
            .into_iter()
            .map(move |membership_key| {
                vec![
                    SqlValue::Int4(relation.oid as i32),
                    SqlValue::Text(membership_key),
                ]
            })
        })
        .collect();
    (table, rows)
}

fn detail_projection(columns: &[&str]) -> SelectProjection {
    SelectProjection::Columns(columns.iter().map(|column| (*column).to_string()).collect())
}

fn detail_oid_predicate(column: usize, oid: u32) -> ResidentExpr {
    ResidentExpr::Binary {
        op: ResidentBinaryOp::Eq,
        lhs: Box::new(ResidentExpr::Column(column)),
        rhs: Box::new(ResidentExpr::Int4Literal(oid as i32)),
    }
}

fn detail_join_relation(table: &RelationalTable, alias: &str) -> JoinRelationRef {
    JoinRelationRef {
        table: table.name.clone(),
        alias: alias.to_string(),
        public_only: false,
    }
}

fn detail_join_column(alias: &str, column: &str) -> JoinColRef {
    JoinColRef {
        qualifier: Some(alias.to_string()),
        column: column.to_string(),
    }
}

fn detail_projected_column(alias: &str, column: &str) -> JoinProjItem {
    JoinProjItem::Column(detail_join_column(alias, column))
}

fn detail_join_step(
    left_alias: &str,
    left_column: &str,
    right_alias: &str,
    right_column: &str,
) -> JoinStep {
    JoinStep {
        conjuncts: vec![(
            detail_join_column(left_alias, left_column),
            detail_join_column(right_alias, right_column),
        )],
        natural: false,
        coalesce: Vec::new(),
        outer_left: false,
        outer_right: false,
    }
}

fn foreign_key_definition(constraint: &RelationalForeignKey) -> String {
    format!(
        "FOREIGN KEY ({}) REFERENCES {}({})",
        constraint.column, constraint.referenced_table, constraint.referenced_column
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_detail_matchers_are_exact_and_fail_closed() {
        assert_eq!(
            between_oid(
                &format!("{INDEX_PREFIX}16384{INDEX_SUFFIX}"),
                INDEX_PREFIX,
                INDEX_SUFFIX
            ),
            Some(16_384)
        );
        assert_eq!(
            between_oid(
                &format!("{CHECK_PREFIX}42{CHECK_SUFFIX}"),
                CHECK_PREFIX,
                CHECK_SUFFIX
            ),
            Some(42)
        );
        assert!(relation_detail_program(
            &format!("{INDEX_PREFIX}16384{INDEX_SUFFIX}").replace("indisvalid", "indisready")
        )
        .is_none());
    }

    #[test]
    fn relation_detail_results_use_the_transient_gpu_target() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE psql_detail (id int4 PRIMARY KEY, name text UNIQUE, \
                 CONSTRAINT psql_detail_positive CHECK (id > 0))",
            )
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE TABLE psql_detail_decoy (id int4 PRIMARY KEY, \
                 CONSTRAINT psql_detail_decoy_positive CHECK (id > 0))",
            )
            .unwrap();
        let relation = engine.relational_catalog_table("psql_detail").unwrap();
        let sql = format!("{INDEX_PREFIX}{}{INDEX_SUFFIX}", relation.oid);
        let result = engine.execute_resident_expr_select_sql(&sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.rows.len(), 2);
        assert_eq!(
            result.rows[0][0],
            SqlValue::Text("psql_detail_pkey".to_string())
        );
        assert_eq!(
            result.rows[1][0],
            SqlValue::Text("psql_detail_name_key".to_string())
        );

        let sql = format!("{CHECK_PREFIX}{}{CHECK_SUFFIX}", relation.oid);
        let result = engine.execute_resident_expr_select_sql(&sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            result.rows[0][1],
            SqlValue::Text("CHECK ((id > 0))".to_string())
        );
        assert!(result
            .rows
            .iter()
            .all(|row| { !matches!(&row[0], SqlValue::Text(name) if name.contains("decoy")) }));
    }

    #[test]
    fn referenced_by_and_publication_joins_fail_closed_without_a_gpu() {
        let engine = Engine::with_planner_config(PlannerConfig {
            default_gpu_id: u16::MAX,
        });
        let catalog = engine.catalog_snapshot();
        let (edges, edge_rows) = referenced_by_candidates(&catalog);
        let (names, name_rows) = relation_name_candidates(&catalog);
        let referenced_by = engine.execute_referenced_by_gpu_candidates(
            edges,
            edge_rows,
            names,
            name_rows,
            1,
            engine.read_snapshot_boundary(),
        );
        assert!(
            referenced_by.is_err(),
            "referenced-by must not use a host name/OID join"
        );

        let (publications, publication_rows) = publication_candidates(&catalog);
        let (memberships, membership_rows) = publication_membership_candidates(&catalog);
        let (relations, relation_rows) = publication_relation_membership_candidates(&catalog);
        let publications = engine.execute_publications_gpu_candidates(
            publications,
            publication_rows,
            memberships,
            membership_rows,
            relations,
            relation_rows,
            1,
            engine.read_snapshot_boundary(),
        );
        assert!(
            publications.is_err(),
            "publication membership must not expand on the host"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn referenced_by_name_to_oid_and_publication_membership_stay_on_the_gpu() {
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (1, "CREATE TABLE psql_ref_parent (id int4 PRIMARY KEY)"),
            (2, "CREATE TABLE psql_ref_child_zeta (id int4 PRIMARY KEY, parent_id int4)"),
            (3, "CREATE TABLE psql_ref_child_alpha (id int4 PRIMARY KEY, parent_id int4)"),
            (4, "CREATE TABLE psql_ref_decoy_parent (id int4 PRIMARY KEY)"),
            (5, "CREATE TABLE psql_ref_decoy_child (id int4 PRIMARY KEY, parent_id int4)"),
            (
                6,
                "ALTER TABLE ONLY psql_ref_child_zeta ADD CONSTRAINT psql_ref_zeta_fk FOREIGN KEY (parent_id) REFERENCES psql_ref_parent(id)",
            ),
            (
                7,
                "ALTER TABLE ONLY psql_ref_child_alpha ADD CONSTRAINT psql_ref_alpha_fk FOREIGN KEY (parent_id) REFERENCES psql_ref_parent(id)",
            ),
            (
                8,
                "ALTER TABLE ONLY psql_ref_decoy_child ADD CONSTRAINT psql_ref_decoy_fk FOREIGN KEY (parent_id) REFERENCES psql_ref_decoy_parent(id)",
            ),
            (9, "CREATE PUBLICATION psql_pub_all FOR ALL TABLES"),
            (10, "CREATE PUBLICATION psql_pub_explicit FOR TABLE psql_ref_parent"),
            (11, "CREATE PUBLICATION psql_pub_decoy FOR TABLE psql_ref_decoy_parent"),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let catalog = engine.catalog_snapshot();
        let parent_oid = catalog.relational_catalog["psql_ref_parent"].oid;
        let child_alpha_oid = catalog.relational_catalog["psql_ref_child_alpha"].oid;
        let (edges, mut edge_rows) = referenced_by_candidates(&catalog);
        let (names, mut name_rows) = relation_name_candidates(&catalog);
        // Reordered full candidates, a decoy FK, and a NULL join key all remain input facts; the
        // device joins the referenced table name to the parent OID and the child OID to its name.
        edge_rows.reverse();
        edge_rows.push(vec![
            SqlValue::Int4(child_alpha_oid as i32),
            SqlValue::Null,
            SqlValue::Text("psql_ref_null_fk".to_string()),
            SqlValue::Text("NULL decoy".to_string()),
        ]);
        name_rows.reverse();
        let referenced_by = engine
            .execute_referenced_by_gpu_candidates(
                edges,
                edge_rows,
                names,
                name_rows,
                parent_oid,
                engine.read_snapshot_boundary(),
            )
            .unwrap();
        assert_eq!(referenced_by.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            referenced_by.rows,
            vec![
                vec![
                    SqlValue::Text("psql_ref_alpha_fk".to_string()),
                    SqlValue::Text("public.psql_ref_child_alpha".to_string()),
                    SqlValue::Text(
                        "FOREIGN KEY (parent_id) REFERENCES psql_ref_parent(id)".to_string()
                    ),
                ],
                vec![
                    SqlValue::Text("psql_ref_zeta_fk".to_string()),
                    SqlValue::Text("public.psql_ref_child_zeta".to_string()),
                    SqlValue::Text(
                        "FOREIGN KEY (parent_id) REFERENCES psql_ref_parent(id)".to_string()
                    ),
                ],
            ]
        );

        let (publications, publication_rows) = publication_candidates(&catalog);
        let (memberships, mut membership_rows) = publication_membership_candidates(&catalog);
        let (relations, mut relation_rows) = publication_relation_membership_candidates(&catalog);
        membership_rows.reverse();
        membership_rows.push(
            membership_rows
                .iter()
                .find(|row| {
                    row[0]
                        == SqlValue::Int4(
                            catalog.relational_publications["psql_pub_all"].oid as i32,
                        )
                })
                .expect("all-tables raw selector")
                .clone(),
        );
        membership_rows.push(vec![
            SqlValue::Int4(catalog.relational_publications["psql_pub_explicit"].oid as i32),
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Null,
        ]);
        relation_rows.reverse();
        relation_rows.push(
            relation_rows
                .iter()
                .find(|row| row[0] == SqlValue::Int4(parent_oid as i32))
                .expect("target relation membership")
                .clone(),
        );
        relation_rows.push(vec![SqlValue::Int4(parent_oid as i32), SqlValue::Null]);
        let publications = engine
            .execute_publications_gpu_candidates(
                publications,
                publication_rows,
                memberships,
                membership_rows,
                relations,
                relation_rows,
                parent_oid,
                engine.read_snapshot_boundary(),
            )
            .unwrap();
        assert_eq!(publications.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            publications.rows,
            vec![
                vec![
                    SqlValue::Text("psql_pub_all".to_string()),
                    SqlValue::Null,
                    SqlValue::Null
                ],
                vec![
                    SqlValue::Text("psql_pub_explicit".to_string()),
                    SqlValue::Null,
                    SqlValue::Null
                ],
            ]
        );
    }
}
