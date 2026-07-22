//! PostgreSQL 16 dump dependency and view-definition metadata.

use super::*;

pub(super) fn is_pg16_dump_dependency_program(canonical: &str) -> bool {
    canonical
        == "select classid, objid, refclassid, refobjid, deptype from pg_depend where deptype != 'p' and deptype != 'e' union all select 'pg_opfamily'::regclass as classid, amopfamily as objid, refclassid, refobjid, deptype from pg_depend d, pg_amop o where deptype not in ('p', 'e', 'i') and classid = 'pg_amop'::regclass and objid = o.oid and not (refclassid = 'pg_opfamily'::regclass and amopfamily = refobjid) union all select 'pg_opfamily'::regclass as classid, amprocfamily as objid, refclassid, refobjid, deptype from pg_depend d, pg_amproc p where deptype not in ('p', 'e', 'i') and classid = 'pg_amproc'::regclass and objid = p.oid and not (refclassid = 'pg_opfamily'::regclass and amprocfamily = refobjid) order by 1,2"
}

pub(super) fn pg16_dump_view_definition_oid(canonical: &str) -> Option<u32> {
    let oid = canonical.strip_prefix("select pg_catalog.pg_get_viewdef('")?;
    let oid = oid.strip_suffix("'::pg_catalog.oid) as viewdef")?;
    oid.parse().ok()
}

impl Engine {
    pub(super) fn execute_pg16_dump_view_definition(
        &self,
        view_oid: u32,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__pg16_dump_view_definitions",
            &[("viewdef", SqlType::Text), ("__oid", SqlType::Int4)],
        );
        let mut rows = catalog
            .relational_views
            .values()
            .map(|view| {
                vec![
                    SqlValue::Text(format!("{};", view.definition)),
                    SqlValue::Int4(view.oid as i32),
                ]
            })
            .collect::<Vec<_>>();
        rows.extend(catalog.relational_materialized_views.values().map(|view| {
            vec![
                SqlValue::Text(format!("{};", view.definition)),
                SqlValue::Int4(view.oid as i32),
            ]
        }));
        let predicate = int4_comparison(&table, "__oid", ResidentBinaryOp::Eq, view_oid as i32)?;
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(vec!["viewdef".to_string()]),
            Some(predicate),
            &[],
            boundary,
        )
    }

    pub(super) fn execute_pg16_dump_dependencies(
        &self,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (dependencies, dependency_rows) = pg16_dump_dependency_candidates(&catalog);
        let (relations, relation_rows) = pg16_dump_dependency_relations(&catalog);
        let predicate = and(
            text_comparison(&dependencies, "deptype", ResidentBinaryOp::Ne, "p")?,
            text_comparison(&dependencies, "deptype", ResidentBinaryOp::Ne, "e")?,
        );
        let plan = join_plan(
            &[(&dependencies, "d"), (&relations, "c")],
            vec![join_step("d", "__referenced_name", "c", "relname", false)],
            vec![
                ("d", "classid", "classid"),
                ("d", "objid", "objid"),
                ("d", "refclassid", "refclassid"),
                ("c", "oid", "refobjid"),
                ("d", "deptype", "deptype"),
            ],
            vec![("d", "classid"), ("d", "objid")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![dependencies, relations],
            vec![dependency_rows, relation_rows],
            vec![Some(predicate), None],
            boundary,
        )
    }
}

fn pg16_dump_dependency_candidates(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_depend",
        &[
            ("classid", SqlType::Int4),
            ("objid", SqlType::Int4),
            ("refclassid", SqlType::Int4),
            ("deptype", SqlType::Text),
            ("__referenced_name", SqlType::Text),
        ],
    );
    let mut rows = catalog
        .relational_views
        .values()
        .map(|view| dependency_candidate(view.oid, &view.query.table))
        .collect::<Vec<_>>();
    rows.extend(
        catalog
            .relational_materialized_views
            .values()
            .map(|view| dependency_candidate(view.oid, &view.query.table)),
    );
    (table, rows)
}

fn dependency_candidate(object_oid: u32, referenced_name: &str) -> Vec<SqlValue> {
    vec![
        SqlValue::Int4(1259),
        SqlValue::Int4(object_oid as i32),
        SqlValue::Int4(1259),
        SqlValue::Text("n".to_string()),
        SqlValue::Text(referenced_name.to_string()),
    ]
}

fn pg16_dump_dependency_relations(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_class",
        &[("oid", SqlType::Int4), ("relname", SqlType::Text)],
    );
    let row =
        |oid: u32, name: &str| vec![SqlValue::Int4(oid as i32), SqlValue::Text(name.to_string())];
    let mut rows = catalog
        .relational_catalog
        .values()
        .map(|relation| row(relation.oid, &relation.name))
        .collect::<Vec<_>>();
    rows.extend(
        catalog
            .relational_views
            .values()
            .map(|relation| row(relation.oid, &relation.name)),
    );
    rows.extend(
        catalog
            .relational_materialized_views
            .values()
            .map(|relation| row(relation.oid, &relation.name)),
    );
    (table, rows)
}
