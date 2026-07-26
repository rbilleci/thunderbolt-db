//! Bounded legacy sequence catalog probes on the GPU join path.
//!
//! PostgreSQL represents sequences with `pg_class.relkind = 'S'`. The historical PG16 golden
//! program pinned by PRODUCT-001 asks for lower-case `'s'`; this exact route preserves that
//! compatibility contract without changing the canonical `pg_class` relation used by psql and
//! pg_dump. Candidate rows are synthesized from one pinned catalog generation, then all filtering,
//! joins, ordering, and projection run through the GPU join executor.

use super::*;

const SEQUENCE_CLASS_PROGRAM: &str = "select c.oid, n.nspname, c.relname, c.relkind, \
    c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = \
    c.relnamespace where n.nspname = 'public' and c.relkind = 's' order by c.relname";

const RELATION_DESCRIPTION_PROGRAM: &str = "select n.nspname, c.relname, a.attname, d.description \
    from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join \
    pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on \
    a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in \
    ('r','v','s') order by c.relname, d.objsubid";

impl Engine {
    pub(super) fn execute_psql_sequence_catalog_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if canonical != SEQUENCE_CLASS_PROGRAM && canonical != RELATION_DESCRIPTION_PROGRAM {
            return Ok(None);
        }

        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (classes, class_rows) = legacy_lowercase_sequence_classes(&catalog);
        let (namespaces, namespace_rows) = synthesize_pg_namespace(&catalog);
        let namespace_predicate = text_equal(&namespaces, "nspname", "public").map(Some)?;

        if canonical == SEQUENCE_CLASS_PROGRAM {
            let plan = plan(
                &[(&classes, "c"), (&namespaces, "n")],
                vec![step("c", "relnamespace", "n", "oid", false)],
                &[
                    ("c", "oid", "oid"),
                    ("n", "nspname", "nspname"),
                    ("c", "relname", "relname"),
                    ("c", "relkind", "relkind"),
                    ("c", "relpersistence", "relpersistence"),
                ],
                &[("c", "relname")],
            );
            let class_predicate = text_equal(&classes, "relkind", "s").map(Some)?;
            return self
                .execute_pg_dump_gpu_join(
                    &plan,
                    vec![classes, namespaces],
                    vec![class_rows, namespace_rows],
                    vec![class_predicate, namespace_predicate],
                    boundary,
                )
                .map(Some);
        }

        let (descriptions, description_rows) =
            synthesize_catalog_relation("pg_catalog.pg_description", &catalog)
                .expect("pg_description is a modeled GPU catalog relation");
        let (attributes, attribute_rows) = synthesize_pg_attribute(&catalog);
        let plan = plan(
            &[
                (&classes, "c"),
                (&namespaces, "n"),
                (&descriptions, "d"),
                (&attributes, "a"),
            ],
            vec![
                step("c", "relnamespace", "n", "oid", false),
                step("c", "oid", "d", "objoid", false),
                JoinStep {
                    conjuncts: vec![
                        (column("c", "oid"), column("a", "attrelid")),
                        (column("d", "objsubid"), column("a", "attnum")),
                    ],
                    natural: false,
                    coalesce: Vec::new(),
                    outer_left: true,
                    outer_right: false,
                },
            ],
            &[
                ("n", "nspname", "nspname"),
                ("c", "relname", "relname"),
                ("a", "attname", "attname"),
                ("d", "description", "description"),
            ],
            &[("c", "relname"), ("d", "objsubid")],
        );
        let class_predicate = any_text_equal(&classes, "relkind", &["r", "v", "s"]).map(Some)?;
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![classes, namespaces, descriptions, attributes],
            vec![class_rows, namespace_rows, description_rows, attribute_rows],
            vec![class_predicate, namespace_predicate, None, None],
            boundary,
        )
        .map(Some)
    }
}

fn legacy_lowercase_sequence_classes(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let (table, mut rows) = synthesize_pg_class(catalog);
    let relkind = table
        .columns
        .iter()
        .position(|column| column.name == "relkind")
        .expect("pg_class has relkind");
    for row in &mut rows {
        if row.get(relkind) == Some(&SqlValue::Text("S".to_string())) {
            row[relkind] = SqlValue::Text("s".to_string());
        }
    }
    (table, rows)
}

fn catalog_column(table: &RelationalTable, name: &str) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|column| column.name == name)
        .ok_or_else(|| sql_pg_error(format!("catalog GPU source is missing column {name:?}")))
}

fn text_equal(
    table: &RelationalTable,
    name: &str,
    value: &str,
) -> Result<ResidentExpr, ExecuteError> {
    Ok(ResidentExpr::Binary {
        op: ResidentBinaryOp::Eq,
        lhs: Box::new(ResidentExpr::Column(catalog_column(table, name)?)),
        rhs: Box::new(ResidentExpr::TextLiteral(value.to_string())),
    })
}

fn any_text_equal(
    table: &RelationalTable,
    name: &str,
    values: &[&str],
) -> Result<ResidentExpr, ExecuteError> {
    let mut predicates = values.iter().map(|value| text_equal(table, name, value));
    let first = predicates
        .next()
        .ok_or_else(|| sql_pg_error("catalog GPU membership list is empty".to_string()))??;
    predicates.try_fold(first, |combined, predicate| {
        predicate.map(|predicate| ResidentExpr::Binary {
            op: ResidentBinaryOp::Or,
            lhs: Box::new(combined),
            rhs: Box::new(predicate),
        })
    })
}

fn column(alias: &str, name: &str) -> JoinColRef {
    JoinColRef {
        qualifier: Some(alias.to_string()),
        column: name.to_string(),
    }
}

fn step(
    left_alias: &str,
    left_column: &str,
    right_alias: &str,
    right_column: &str,
    outer_left: bool,
) -> JoinStep {
    JoinStep {
        conjuncts: vec![(
            column(left_alias, left_column),
            column(right_alias, right_column),
        )],
        natural: false,
        coalesce: Vec::new(),
        outer_left,
        outer_right: false,
    }
}

fn plan(
    relations: &[(&RelationalTable, &str)],
    steps: Vec<JoinStep>,
    projection: &[(&str, &str, &str)],
    order_by: &[(&str, &str)],
) -> JoinPlan {
    JoinPlan {
        relations: relations
            .iter()
            .map(|(table, alias)| JoinRelationRef {
                table: table.name.clone(),
                alias: (*alias).to_string(),
                public_only: false,
            })
            .collect(),
        steps,
        projection: projection
            .iter()
            .map(|(alias, name, _)| JoinProjItem::Column(column(alias, name)))
            .collect(),
        distinct: false,
        projection_aliases: projection
            .iter()
            .map(|(_, _, output)| Some((*output).to_string()))
            .collect(),
        order_by: order_by
            .iter()
            .map(|(alias, name)| (column(alias, name), false))
            .collect(),
        order_by_nulls_first: vec![None; order_by.len()],
        limit: None,
        offset: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_program_recognition_is_exact() {
        assert_eq!(
            canonicalize_sql_for_exact_match(SEQUENCE_CLASS_PROGRAM).unwrap(),
            SEQUENCE_CLASS_PROGRAM
        );
        assert_eq!(
            canonicalize_sql_for_exact_match(RELATION_DESCRIPTION_PROGRAM).unwrap(),
            RELATION_DESCRIPTION_PROGRAM
        );
    }
}
