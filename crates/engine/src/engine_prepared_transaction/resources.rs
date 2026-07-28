use super::*;

// Two fragment-kind fields (4), the resolved-operation wrapper (20), the transaction-claim body
// (100), and the canonical terminal outcome (92). Header/digests/physical framing are deliberately
// excluded by `canonical_logical_intent_outcome_bytes`; the binary transaction payload is added
// separately below.
const CANONICAL_TRANSACTION_LOGICAL_OVERHEAD: u64 = 216;
const BINARY_TRANSACTION_HEADER: u64 = 3 + 8 + 4 + 4;

pub(super) struct PreparedResourceEstimate {
    pub(super) resources: TransactionResources,
    pub(super) fast_eligible: bool,
}

impl Engine {
    pub(super) fn estimate_bound_prepared_resources(
        &self,
        operations: &[gpu_db_sql::ParsedCommand],
        proof: &PreparedRouteProof,
    ) -> Result<PreparedResourceEstimate, ExecuteError> {
        let operation_count = u32::try_from(operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "prepared transaction operation count exceeds u32 framing".to_string(),
            )
        })?;
        let mutation_count = u32::try_from(
            operations
                .iter()
                .filter(|operation| is_prepared_mutation(operation.command()))
                .count(),
        )
        .map_err(|_| {
            ExecuteError::Unsupported(
                "prepared transaction mutation count exceeds u32 framing".to_string(),
            )
        })?;
        let touched_tables = u32::try_from(proof.tables.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "prepared transaction touched-table count exceeds u32 framing".to_string(),
            )
        })?;

        let mut fast_eligible = proof.indexed_execution_eligible;
        let mut mutated_keys = BTreeSet::new();
        let mut binary_bytes = BINARY_TRANSACTION_HEADER;
        let mut post_image_bytes = 0_u64;
        let mut maintained_index_fanout = 0_u32;
        let mut result_bytes = 0_u64;
        let mut resolved_mutations = 0_u32;

        for operation in operations {
            match operation.command() {
                Command::Select(select) => {
                    let table = prepared_table(proof, &select.table)?;
                    fast_eligible &= plain_prepared_select(select);
                    let bound = bind_relational_select(table, select)?;
                    result_bytes = checked_add(
                        result_bytes,
                        fixed_result_bytes(&bound.selected_columns).unwrap_or_else(|| {
                            fast_eligible = false;
                            0
                        }),
                        "result bytes",
                    )?;
                    let groups = normalized_select_groups(&bound);
                    let Some(_key) = exact_unique_group_key(table, &groups) else {
                        fast_eligible = false;
                        continue;
                    };
                }
                Command::Insert(insert) => {
                    let table = prepared_table(proof, &insert.table)?;
                    if table
                        .columns
                        .iter()
                        .any(|column| column.ty == SqlType::Text)
                        || insert.rows.len() != 1
                    {
                        fast_eligible = false;
                        continue;
                    }
                    let Some(row) = fixed_insert_row(table, insert)? else {
                        fast_eligible = false;
                        continue;
                    };
                    let Some(key) = primary_row_key(table, &row) else {
                        fast_eligible = false;
                        continue;
                    };
                    mutated_keys.insert((table.name.clone(), key));
                    add_mutation_fanout(&mut maintained_index_fanout, table)?;
                    let encoded = encode_relational_row(&row);
                    binary_bytes = checked_add(
                        binary_bytes,
                        binary_insert_bytes(&table.name, encoded.len())?,
                        "binary WAL bytes",
                    )?;
                    post_image_bytes =
                        checked_add(post_image_bytes, encoded.len() as u64, "post-image bytes")?;
                    resolved_mutations += 1;
                    result_bytes = checked_add(
                        result_bytes,
                        fixed_named_result_bytes(table, &insert.returning).unwrap_or_else(|| {
                            fast_eligible = false;
                            0
                        }),
                        "result bytes",
                    )?;
                }
                Command::Update(update) => {
                    let table = prepared_table(proof, &update.table)?;
                    if table
                        .columns
                        .iter()
                        .any(|column| column.ty == SqlType::Text)
                    {
                        fast_eligible = false;
                        continue;
                    }
                    let groups = bind_delete_filter_groups(
                        table,
                        &Delete {
                            table: update.table.clone(),
                            filter: update.filter.clone(),
                            filters: update.filters.clone(),
                            filter_groups: update.filter_groups.clone(),
                            returning: Vec::new(),
                        },
                    )?;
                    let Some(key) = exact_unique_group_key(table, &groups) else {
                        fast_eligible = false;
                        continue;
                    };
                    if !mutated_keys.insert((table.name.clone(), key)) {
                        fast_eligible = false;
                        continue;
                    }
                    let _assignments = bind_update_assignments(table, update)?;
                    let row_bytes = max_fixed_encoded_row_bytes(table).unwrap_or_else(|| {
                        fast_eligible = false;
                        0
                    });
                    binary_bytes = checked_add(
                        binary_bytes,
                        binary_update_bytes(&table.name, row_bytes, row_bytes)?,
                        "binary WAL bytes",
                    )?;
                    post_image_bytes =
                        checked_add(post_image_bytes, row_bytes as u64, "post-image bytes")?;
                    resolved_mutations += 1;
                    add_mutation_fanout(&mut maintained_index_fanout, table)?;
                    result_bytes = checked_add(
                        result_bytes,
                        fixed_named_result_bytes(table, &update.returning).unwrap_or_else(|| {
                            fast_eligible = false;
                            0
                        }),
                        "result bytes",
                    )?;
                }
                Command::Delete(delete) => {
                    let table = prepared_table(proof, &delete.table)?;
                    if table
                        .columns
                        .iter()
                        .any(|column| column.ty == SqlType::Text)
                    {
                        fast_eligible = false;
                        continue;
                    }
                    let groups = bind_delete_filter_groups(table, delete)?;
                    let Some(key) = exact_unique_group_key(table, &groups) else {
                        fast_eligible = false;
                        continue;
                    };
                    if !mutated_keys.insert((table.name.clone(), key)) {
                        fast_eligible = false;
                        continue;
                    }
                    let row_bytes = max_fixed_encoded_row_bytes(table).unwrap_or_else(|| {
                        fast_eligible = false;
                        0
                    });
                    binary_bytes = checked_add(
                        binary_bytes,
                        binary_delete_bytes(&table.name, row_bytes)?,
                        "binary WAL bytes",
                    )?;
                    resolved_mutations += 1;
                    add_mutation_fanout(&mut maintained_index_fanout, table)?;
                    result_bytes = checked_add(
                        result_bytes,
                        fixed_named_result_bytes(table, &delete.returning).unwrap_or_else(|| {
                            fast_eligible = false;
                            0
                        }),
                        "result bytes",
                    )?;
                }
                _ => fast_eligible = false,
            }
        }

        let logical_wal = if resolved_mutations == 0 {
            0
        } else {
            checked_add(
                binary_bytes,
                CANONICAL_TRANSACTION_LOGICAL_OVERHEAD,
                "logical WAL bytes",
            )?
        };
        Ok(PreparedResourceEstimate {
            resources: TransactionResources {
                operations: operation_count,
                mutations: mutation_count,
                post_image_and_wal_bytes: checked_add(
                    post_image_bytes,
                    logical_wal,
                    "post-image plus logical-WAL bytes",
                )?,
                maintained_index_fanout,
                touched_tables,
                cold_accesses: 0,
                result_bytes,
            },
            fast_eligible,
        })
    }
}

fn prepared_table<'a>(
    proof: &'a PreparedRouteProof,
    name: &str,
) -> Result<&'a RelationalTable, ExecuteError> {
    proof
        .tables
        .get(name)
        .map(|requirement| &requirement.table)
        .ok_or_else(|| ExecuteError::UndefinedRelation(name.to_string()))
}

fn plain_prepared_select(select: &Select) -> bool {
    !select.distinct
        && select.group_by.is_none()
        && select.having_groups.is_empty()
        && select.order_by.is_empty()
        && select.limit.is_none()
        && select.offset.is_none()
        && matches!(
            select.projection,
            SelectProjection::All | SelectProjection::Columns(_)
        )
}

fn exact_unique_group_key(
    table: &RelationalTable,
    groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
) -> Option<Vec<SqlValue>> {
    let [group] = groups else { return None };
    table
        .indexes
        .iter()
        .filter(|index| index.unique)
        .find_map(|index| {
            let positions = crate::engine_residency::index_key_column_positions(table, index)?;
            positions
                .iter()
                .map(|position| {
                    group
                        .iter()
                        .find(|(column, op, _)| column == position && *op == SelectFilterOp::Eq)
                        .map(|(_, _, value)| value.clone())
                })
                .collect::<Option<Vec<_>>>()
        })
}

fn primary_row_key(table: &RelationalTable, row: &[SqlValue]) -> Option<Vec<SqlValue>> {
    let index = table
        .indexes
        .iter()
        .find(|index| index.primary_key)
        .or_else(|| table.indexes.iter().find(|index| index.unique))?;
    crate::engine_residency::index_key_column_positions(table, index)?
        .into_iter()
        .map(|position| row.get(position).cloned())
        .collect()
}

fn fixed_insert_row(
    table: &RelationalTable,
    insert: &Insert,
) -> Result<Option<Vec<SqlValue>>, ExecuteError> {
    let Some(source) = insert.rows.first() else {
        return Ok(None);
    };
    let positions = if insert.columns.is_empty() {
        (0..table.columns.len()).collect::<Vec<_>>()
    } else {
        insert
            .columns
            .iter()
            .map(|name| relational_column_index(table, name))
            .collect::<Result<Vec<_>, _>>()?
    };
    if source.len() != positions.len() {
        return Err(prepared_route_error(
            "prepared INSERT row does not match its target columns",
        ));
    }
    let mut row = vec![None; table.columns.len()];
    for (cell, position) in source.iter().zip(positions) {
        let Some(value) = cell.value() else {
            return Ok(None);
        };
        row[position] = Some(
            coerce_insert_value(
                value.clone(),
                table.columns[position].ty,
                &table.columns[position].name,
            )
            .map_err(ExecuteError::Engine)?,
        );
    }
    for (position, value) in row.iter_mut().enumerate() {
        if value.is_some() {
            continue;
        }
        *value = match &table.columns[position].default {
            Some(ColumnDefault::Literal(value)) => Some(value.clone()),
            Some(ColumnDefault::DeferredScalar { .. })
            | Some(ColumnDefault::SequenceNextVal { .. })
            | None => return Ok(None),
        };
    }
    Ok(Some(
        row.into_iter().collect::<Option<Vec<_>>>().expect("filled"),
    ))
}

fn fixed_result_bytes(columns: &[RelationalColumn]) -> Option<u64> {
    columns.iter().try_fold(0_u64, |total, column| {
        total.checked_add(match column.ty {
            SqlType::Int2 => 2,
            SqlType::Int4 | SqlType::Date => 4,
            SqlType::Int8 | SqlType::Timestamp => 8,
            SqlType::Numeric { .. } | SqlType::Uuid => 16,
            SqlType::Bool => 1,
            SqlType::Text => return None,
        })
    })
}

/// Maximum length of the canonical textual row image for a fixed-width table. Unlike a value
/// preflight, this bound is stable across READ COMMITTED refreshes between admission and the
/// statement snapshot. TEXT is deliberately unbounded and therefore not a latency-class shape.
fn max_fixed_encoded_row_bytes(table: &RelationalTable) -> Option<usize> {
    table
        .columns
        .iter()
        .enumerate()
        .try_fold(0_usize, |total, (ordinal, column)| {
            let cell = match column.ty {
                SqlType::Int2 => 9,            // i2:-32768
                SqlType::Int4 => 13,           // i:-2147483648
                SqlType::Int8 => 22,           // n:-9223372036854775808
                SqlType::Numeric { .. } => 46, // d:<i128>:<u8 scale>
                SqlType::Bool => 4,            // max(NULL_TOKEN, b:t)
                SqlType::Text => return None,
                SqlType::Date => 16,      // date:-2147483648
                SqlType::Timestamp => 23, // ts:-9223372036854775808
                SqlType::Uuid => 41,      // uuid:xxxxxxxx-....
            };
            total
                .checked_add(usize::from(ordinal != 0))?
                .checked_add(cell)
        })
}

fn fixed_named_result_bytes(table: &RelationalTable, names: &[String]) -> Option<u64> {
    let columns = names
        .iter()
        .map(|name| {
            table
                .columns
                .iter()
                .find(|column| &column.name == name)
                .cloned()
        })
        .collect::<Option<Vec<_>>>()?;
    fixed_result_bytes(&columns)
}

fn add_mutation_fanout(total: &mut u32, table: &RelationalTable) -> Result<(), ExecuteError> {
    *total = total
        .checked_add(u32::try_from(table.indexes.len()).map_err(|_| {
            ExecuteError::Unsupported("prepared index fanout exceeds u32 framing".to_string())
        })?)
        .ok_or_else(|| {
            ExecuteError::Unsupported("prepared index fanout count overflowed".to_string())
        })?;
    Ok(())
}

fn binary_insert_bytes(table: &str, row: usize) -> Result<u64, ExecuteError> {
    binary_mutation_bytes(table, &[row])
}

fn binary_update_bytes(table: &str, old: usize, new: usize) -> Result<u64, ExecuteError> {
    binary_mutation_bytes(table, &[old, new])
}

fn binary_delete_bytes(table: &str, old: usize) -> Result<u64, ExecuteError> {
    binary_mutation_bytes(table, &[old])
}

fn binary_mutation_bytes(table: &str, rows: &[usize]) -> Result<u64, ExecuteError> {
    let table = u64::try_from(table.len())
        .map_err(|_| ExecuteError::Unsupported("table name length overflowed".to_string()))?;
    rows.iter().try_fold(1 + 2 + table + 8, |total, row| {
        total
            .checked_add(4)
            .and_then(|value| value.checked_add(*row as u64))
            .ok_or_else(|| {
                ExecuteError::Unsupported("binary WAL byte count overflowed".to_string())
            })
    })
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, ExecuteError> {
    left.checked_add(right).ok_or_else(|| {
        ExecuteError::Unsupported(format!("prepared transaction {label} overflowed"))
    })
}

fn is_prepared_mutation(command: &Command) -> bool {
    matches!(
        command,
        Command::Insert(_) | Command::Update(_) | Command::Delete(_)
    )
}
