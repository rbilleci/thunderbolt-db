//! Exact scalar-cast binding for the bounded empty-catalog query subset.

use super::*;

fn cast_error(source: Option<SqlType>, target: &str) -> ExecuteError {
    sql_pg_error(format!(
        "catalog cast from {source:?} to {target} is outside the typed subset"
    ))
}

fn literal(node: &Node) -> Option<&pg_query::protobuf::AConst> {
    match node.node.as_ref() {
        Some(NodeEnum::AConst(constant)) => Some(constant),
        Some(NodeEnum::TypeCast(cast)) => cast.arg.as_deref().and_then(literal),
        Some(NodeEnum::CollateClause(collate)) => collate.arg.as_deref().and_then(literal),
        _ => None,
    }
}

fn literal_text(node: &Node) -> Option<String> {
    let constant = literal(node)?;
    match &constant.val {
        Some(a_const::Val::Ival(value)) => Some(value.ival.to_string()),
        Some(a_const::Val::Fval(value)) => Some(value.fval.clone()),
        Some(a_const::Val::Sval(value)) => Some(value.sval.clone()),
        Some(a_const::Val::Boolval(value)) => Some(if value.boolval {
            "true".to_string()
        } else {
            "false".to_string()
        }),
        _ => None,
    }
}

fn validate_bool_literal(node: &Node) -> Result<(), ExecuteError> {
    let Some(value) = literal_text(node) else {
        return Ok(());
    };
    let value = value.trim().to_ascii_lowercase();
    if matches!(
        value.as_str(),
        "true"
            | "tru"
            | "tr"
            | "t"
            | "yes"
            | "ye"
            | "y"
            | "on"
            | "1"
            | "false"
            | "fals"
            | "fal"
            | "fa"
            | "f"
            | "no"
            | "n"
            | "off"
            | "of"
            | "0"
    ) {
        Ok(())
    } else {
        Err(sql_pg_error(format!(
            "invalid input syntax for type boolean: {:?}",
            value
        )))
    }
}

fn integer_literal(node: &Node) -> Option<Result<i128, ExecuteError>> {
    let value = literal_text(node)?;
    Some(
        value
            .trim()
            .parse::<i128>()
            .map_err(|_| sql_pg_error("invalid input syntax for an integer cast".to_string())),
    )
}

fn validate_integer_literal(node: &Node, target: &str) -> Result<(), ExecuteError> {
    let Some(value) = integer_literal(node) else {
        return Ok(());
    };
    let value = value?;
    let in_range = match target {
        "int2" => (i128::from(i16::MIN)..=i128::from(i16::MAX)).contains(&value),
        "int4" => (i128::from(i32::MIN)..=i128::from(i32::MAX)).contains(&value),
        "int8" => (i128::from(i64::MIN)..=i128::from(i64::MAX)).contains(&value),
        "oid" => (0..=i128::from(u32::MAX)).contains(&value),
        _ => false,
    };
    if in_range {
        Ok(())
    } else {
        Err(sql_pg_error(format!(
            "integer value {value} is out of range for type {target}"
        )))
    }
}

fn numeric_oid_literal(value: &str) -> bool {
    value.trim().parse::<u32>().is_ok()
}

fn regclass_name_exists(value: &str, catalog: &CatalogSnapshot) -> bool {
    let value = value.trim().to_ascii_lowercase();
    if numeric_oid_literal(&value) {
        return true;
    }
    if let Some(name) = value.strip_prefix("public.") {
        return catalog.relational_catalog.contains_key(name)
            || catalog.relational_views.contains_key(name)
            || catalog.relational_materialized_views.contains_key(name)
            || catalog.relational_sequences.contains_key(name);
    }
    if value.starts_with("pg_catalog.") || value.starts_with("information_schema.") {
        return synthesize_catalog_relation(&value, catalog).is_some();
    }
    if value.contains('.') {
        return false;
    }
    catalog.relational_catalog.contains_key(&value)
        || catalog.relational_views.contains_key(&value)
        || catalog.relational_materialized_views.contains_key(&value)
        || catalog.relational_sequences.contains_key(&value)
        || synthesize_catalog_relation(&value, catalog).is_some()
        || empty_binding_only_catalog_relation(&value).is_some()
}

fn regnamespace_name_exists(value: &str, catalog: &CatalogSnapshot) -> bool {
    if numeric_oid_literal(value) {
        return true;
    }
    match value.trim().to_ascii_lowercase().as_str() {
        "pg_catalog" | "information_schema" => true,
        "public" => catalog.relational_public_schema_exists,
        _ => false,
    }
}

fn regtype_name_exists(value: &str, catalog: &CatalogSnapshot) -> bool {
    let mut value = value.trim().to_ascii_lowercase();
    if numeric_oid_literal(&value) {
        return true;
    }
    if let Some(name) = value.strip_prefix("public.") {
        return catalog.relational_domains.contains_key(name);
    }
    if let Some(name) = value.strip_prefix("pg_catalog.") {
        value = name.to_string();
    } else if value.contains('.') {
        return false;
    }
    if matches!(
        value.as_str(),
        "bool"
            | "boolean"
            | "int2"
            | "smallint"
            | "int4"
            | "integer"
            | "int"
            | "int8"
            | "bigint"
            | "numeric"
            | "decimal"
            | "text"
            | "date"
            | "timestamp"
            | "timestamp without time zone"
            | "uuid"
            | "oid"
            | "regclass"
            | "regnamespace"
            | "regtype"
    ) {
        return true;
    }
    let (_, rows) = synthesize_pg_type(catalog);
    rows.iter()
        .any(|row| matches!(row.get(1), Some(SqlValue::Text(name)) if name == &value))
}

fn validate_reg_literal(
    node: &Node,
    target: &str,
    catalog: &CatalogSnapshot,
) -> Result<(), ExecuteError> {
    let Some(constant) = literal(node) else {
        return Ok(());
    };
    match &constant.val {
        Some(a_const::Val::Ival(value)) if value.ival >= 0 => Ok(()),
        Some(a_const::Val::Ival(value)) => Err(sql_pg_error(format!(
            "OID value {} is out of range for type {target}",
            value.ival
        ))),
        Some(a_const::Val::Sval(value)) => {
            let exists = match target {
                "regclass" => regclass_name_exists(&value.sval, catalog),
                "regnamespace" => regnamespace_name_exists(&value.sval, catalog),
                "regtype" => regtype_name_exists(&value.sval, catalog),
                _ => false,
            };
            if exists {
                Ok(())
            } else {
                Err(sql_pg_error(format!(
                    "object {:?} does not exist for {target} cast",
                    value.sval
                )))
            }
        }
        Some(a_const::Val::Fval(value)) => {
            if numeric_oid_literal(&value.fval) {
                Ok(())
            } else {
                Err(sql_pg_error(format!(
                    "object {:?} does not exist for {target} cast",
                    value.fval
                )))
            }
        }
        Some(a_const::Val::Boolval(value)) => Err(sql_pg_error(format!(
            "object {:?} does not exist for {target} cast",
            value.boolval
        ))),
        _ => Ok(()),
    }
}

pub(super) fn validate_catalog_scalar_cast(
    target: &str,
    argument: &Node,
    source: Option<SqlType>,
    catalog: &CatalogSnapshot,
) -> Result<(), ExecuteError> {
    let Some(source) = source else {
        return Ok(());
    };
    match target {
        "text" => Ok(()),
        "bool" if matches!(source, SqlType::Bool | SqlType::Text) => {
            validate_bool_literal(argument)
        }
        "int2" | "int4" | "int8" | "oid"
            if catalog_integer_type(source) || source == SqlType::Text =>
        {
            validate_integer_literal(argument, target)
        }
        "regclass" | "regnamespace" | "regtype"
            if catalog_integer_type(source) || source == SqlType::Text =>
        {
            validate_reg_literal(argument, target, catalog)
        }
        _ => Err(cast_error(Some(source), target)),
    }
}
