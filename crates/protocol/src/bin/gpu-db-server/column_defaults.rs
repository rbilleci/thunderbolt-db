// Legacy host-backed column-default semantics. This is parity/bootstrap debt, not a product path.

use super::{
    bool_text, next_sequence_value, sequence_target_error, sql_value_matches_type, ColumnDefault,
    ErrorField, Session, SqlValue,
};

pub(super) fn column_default_matches_type(
    value: &ColumnDefault,
    ty: gpu_db_protocol::SqlType,
) -> bool {
    match value {
        ColumnDefault::Literal(value) => sql_value_matches_type(value, ty),
        ColumnDefault::SequenceNextVal { .. } => ty == gpu_db_protocol::SqlType::Int4,
    }
}

fn format_default_expr(value: &SqlValue) -> String {
    match value {
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) => format!("{value}::smallint"),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Text(value) => format!("'{}'::text", value.replace('\'', "''")),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => bool_text(*value),
        SqlValue::Date(value) => {
            format!("'{}'::date", gpu_db_protocol::datetime::format_date(*value))
        }
        SqlValue::Timestamp(value) => format!(
            "'{}'::timestamp",
            gpu_db_protocol::datetime::format_timestamp(*value)
        ),
        SqlValue::Uuid(value) => format!("'{}'::uuid", gpu_db_protocol::uuid::format_uuid(value)),
        SqlValue::Parameter { .. } => {
            unreachable!("column defaults never contain prepared parameters")
        }
    }
}

pub(super) fn format_column_default_expr(value: &ColumnDefault) -> String {
    match value {
        ColumnDefault::Literal(value) => format_default_expr(value),
        ColumnDefault::SequenceNextVal { sequence, .. } => {
            format!("nextval('{}'::regclass)", sequence.replace('\'', "''"))
        }
    }
}

pub(super) fn preflight_column_default_target(
    session: &Session,
    default: &ColumnDefault,
) -> Option<ErrorField> {
    match default {
        ColumnDefault::Literal(_) => None,
        ColumnDefault::SequenceNextVal {
            sequence,
            create_if_missing: true,
        } => {
            if session.tables.contains_key(sequence)
                || session.views.contains_key(sequence)
                || session.materialized_views.contains_key(sequence)
                || session.sequences.contains_key(sequence)
            {
                Some(ErrorField {
                    code: "42P07",
                    message: "relation already exists",
                    position: None,
                })
            } else {
                None
            }
        }
        ColumnDefault::SequenceNextVal {
            sequence,
            create_if_missing: false,
        } => sequence_target_error(session, sequence),
    }
}

pub(super) fn resolve_column_domain_type(
    session: &Session,
    def: &mut gpu_db_protocol::ColumnDef,
) -> Result<u32, ErrorField> {
    if let Some(domain_name) = def.domain.as_ref() {
        let domain = session.domains.get(domain_name).ok_or(ErrorField {
            code: "42704",
            message: "type does not exist",
            position: None,
        })?;
        def.ty = domain.base_type;
        Ok(domain.oid)
    } else {
        Ok(def.ty.postgres_oid())
    }
}

pub(super) fn add_column_default_supported(default: &ColumnDefault) -> bool {
    match default {
        ColumnDefault::Literal(_) => true,
        ColumnDefault::SequenceNextVal {
            create_if_missing, ..
        } => !create_if_missing,
    }
}

pub(super) fn evaluate_column_default(
    session: &mut Session,
    default: &ColumnDefault,
) -> Result<SqlValue, ErrorField> {
    match default {
        ColumnDefault::Literal(value) => Ok(value.clone()),
        ColumnDefault::SequenceNextVal { sequence, .. } => {
            if let Some(error) = sequence_target_error(session, sequence) {
                return Err(error);
            }
            let sequence_state = session
                .sequences
                .get_mut(sequence)
                .expect("sequence target checked");
            let value = next_sequence_value(sequence_state)?;
            session.currval_sequences.insert(sequence.clone(), value);
            session.mark_sequence_dirty(sequence.clone());
            i32::try_from(value)
                .map(SqlValue::Int4)
                .map_err(|_| ErrorField {
                    code: "22003",
                    message: "sequence value is out of range for int4 default",
                    position: None,
                })
        }
    }
}
