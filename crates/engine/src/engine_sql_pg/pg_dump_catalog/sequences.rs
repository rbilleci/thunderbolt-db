//! PostgreSQL 16 dump sequence configuration and state metadata.

use super::*;

pub(super) fn pg16_dump_sequence_metadata_oid(canonical: &str) -> Option<u32> {
    let oid = canonical.strip_prefix(
        "select format_type(seqtypid, null), seqstart, seqincrement, seqmax, seqmin, seqcache, seqcycle from pg_catalog.pg_sequence where seqrelid = '",
    )?;
    let oid = oid.strip_suffix("'::oid")?;
    oid.parse().ok()
}

pub(super) fn pg16_dump_sequence_state_name(sql: &str) -> Option<String> {
    let Command::Select(select) = parse_command_allowing_catalog(sql).ok()? else {
        return None;
    };
    if !select.public_only
        || select.distinct
        || select.projection
            != SelectProjection::Columns(vec!["last_value".to_string(), "is_called".to_string()])
        || select.group_by.is_some()
        || !select.having_groups.is_empty()
        || select.filter.is_some()
        || !select.filters.is_empty()
        || !select.filter_groups.is_empty()
        || !select.order_by.is_empty()
        || select.limit.is_some()
        || select.offset.is_some()
    {
        return None;
    }
    Some(select.table)
}

impl Engine {
    pub(super) fn execute_pg16_dump_sequence_metadata(
        &self,
        sequence_oid: u32,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "pg_sequence",
            &[
                ("format_type", SqlType::Text),
                ("seqstart", SqlType::Int8),
                ("seqincrement", SqlType::Int8),
                ("seqmax", SqlType::Int8),
                ("seqmin", SqlType::Int8),
                ("seqcache", SqlType::Int8),
                ("seqcycle", SqlType::Bool),
                ("__seqrelid", SqlType::Int4),
            ],
        );
        let rows = catalog
            .relational_sequences
            .values()
            .map(|sequence| {
                vec![
                    SqlValue::Text("bigint".to_string()),
                    SqlValue::Int8(1),
                    SqlValue::Int8(1),
                    SqlValue::Int8(i64::MAX),
                    SqlValue::Int8(1),
                    SqlValue::Int8(1),
                    SqlValue::Bool(false),
                    SqlValue::Int4(sequence.oid as i32),
                ]
            })
            .collect();
        let predicate = int4_comparison(
            &table,
            "__seqrelid",
            ResidentBinaryOp::Eq,
            sequence_oid as i32,
        )?;
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(
                [
                    "format_type",
                    "seqstart",
                    "seqincrement",
                    "seqmax",
                    "seqmin",
                    "seqcache",
                    "seqcycle",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            Some(predicate),
            &[],
            boundary,
        )
    }

    pub(super) fn execute_pg16_dump_sequence_state(
        &self,
        sequence_name: &str,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let table = catalog_relation_table(
            "pg_catalog",
            "__pg16_dump_sequence_state",
            &[("last_value", SqlType::Int8), ("is_called", SqlType::Bool)],
        );
        let rows = catalog
            .relational_sequences
            .get(sequence_name)
            .map(|sequence| {
                vec![vec![
                    SqlValue::Int8(sequence.last_value),
                    SqlValue::Bool(sequence.is_called),
                ]]
            })
            .unwrap_or_default();
        self.execute_pg_dump_transient_relation(table, rows, &[], boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_state_matcher_is_exact_and_retains_the_bound_relation() {
        assert_eq!(
            pg16_dump_sequence_state_name("SELECT last_value, is_called FROM public.account_seq",),
            Some("account_seq".to_string()),
        );
        assert_eq!(
            pg16_dump_sequence_state_name(
                "SELECT last_value, is_called, 1 FROM public.account_seq",
            ),
            None,
        );
    }
}
