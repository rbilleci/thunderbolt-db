//! Exact PostgreSQL 16 information-schema presentation quirks on the GPU catalog path.

use super::*;

const KEY_COLUMN_USAGE_PROGRAM: &str = "select table_schema, table_name, column_name, \
    constraint_name, ordinal_position from information_schema.key_column_usage where \
    table_schema = 'public' order by table_name, ordinal_position";

impl Engine {
    pub(super) fn execute_psql_information_schema_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if canonical != KEY_COLUMN_USAGE_PROGRAM {
            return Ok(None);
        }

        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (mut table, mut rows) = synthesize_information_schema_key_column_usage(&catalog);
        table.columns.push(RelationalColumn {
            id: 0,
            table_oid: 0,
            attnum: (table.columns.len() + 1) as i16,
            name: "__legacy_constraint_order".to_string(),
            ty: SqlType::Int4,
            domain: None,
            default: None,
            type_oid: SqlType::Int4.postgres_oid(),
            type_size: SqlType::Int4.type_size(),
        });
        for row in &mut rows {
            let is_foreign_key = !matches!(row.get(8), Some(SqlValue::Null));
            row.push(SqlValue::Int4(i32::from(is_foreign_key)));
        }
        let table_schema = table
            .columns
            .iter()
            .position(|column| column.name == "table_schema")
            .ok_or_else(|| {
                sql_pg_error("key_column_usage GPU source is missing table_schema".to_string())
            })?;
        let predicate = ResidentExpr::Binary {
            op: ResidentBinaryOp::Eq,
            lhs: Box::new(ResidentExpr::Column(table_schema)),
            rhs: Box::new(ResidentExpr::TextLiteral("public".to_string())),
        };
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(
                [
                    "table_schema",
                    "table_name",
                    "column_name",
                    "constraint_name",
                    "ordinal_position",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            Some(predicate),
            &[
                "table_name",
                "ordinal_position",
                "__legacy_constraint_order",
                "constraint_name",
            ],
            boundary,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_column_usage_program_recognition_is_exact() {
        assert_eq!(
            canonicalize_sql_for_exact_match(KEY_COLUMN_USAGE_PROGRAM).unwrap(),
            KEY_COLUMN_USAGE_PROGRAM
        );
    }
}
