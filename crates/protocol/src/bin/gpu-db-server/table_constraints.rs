// Legacy host-backed table-constraint construction. This is parity/bootstrap debt, not a product path.

use super::{
    sql_value_matches_type, validate_check_constraints, validate_unique_indexes,
    CatalogCheckConstraint, CatalogIndex, ErrorField, SelectFilter, Session,
};

pub(super) fn add_primary_key_to_session(
    session: &mut Session,
    table_name: &str,
    constraint_name: String,
    column: String,
) -> Result<(), ErrorField> {
    if session
        .indexes
        .iter()
        .any(|index| index.name == constraint_name)
        || session.tables.contains_key(&constraint_name)
        || session.views.contains_key(&constraint_name)
        || session.materialized_views.contains_key(&constraint_name)
        || session.sequences.contains_key(&constraint_name)
    {
        return Err(ErrorField {
            code: "42P07",
            message: "relation already exists",
            position: None,
        });
    }
    if session
        .indexes
        .iter()
        .any(|index| index.table == table_name && index.primary_key)
    {
        return Err(ErrorField {
            code: "42P16",
            message: "multiple primary keys are not allowed",
            position: None,
        });
    }
    let Some(table) = session.tables.get(table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    if !table
        .columns
        .iter()
        .any(|candidate| candidate.def.name == column)
    {
        return Err(ErrorField {
            code: "42703",
            message: "column does not exist",
            position: None,
        });
    }
    let mut candidate_indexes = session.indexes.clone();
    candidate_indexes.push(CatalogIndex {
        name: constraint_name.clone(),
        table: table_name.to_string(),
        column: column.clone(),
        unique: true,
        primary_key: true,
        unique_constraint: false,
    });
    validate_unique_indexes(table, &candidate_indexes)?;
    session.indexes.push(CatalogIndex {
        name: constraint_name,
        table: table_name.to_string(),
        column,
        unique: true,
        primary_key: true,
        unique_constraint: false,
    });
    session.dirty_indexes = true;
    Ok(())
}

pub(super) fn add_unique_constraint_to_session(
    session: &mut Session,
    table_name: &str,
    constraint_name: String,
    column: String,
) -> Result<(), ErrorField> {
    if session
        .indexes
        .iter()
        .any(|index| index.name == constraint_name)
        || session.tables.contains_key(&constraint_name)
        || session.views.contains_key(&constraint_name)
        || session.materialized_views.contains_key(&constraint_name)
        || session.sequences.contains_key(&constraint_name)
    {
        return Err(ErrorField {
            code: "42P07",
            message: "relation already exists",
            position: None,
        });
    }
    let Some(table) = session.tables.get(table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    if !table
        .columns
        .iter()
        .any(|candidate| candidate.def.name == column)
    {
        return Err(ErrorField {
            code: "42703",
            message: "column does not exist",
            position: None,
        });
    }
    let mut candidate_indexes = session.indexes.clone();
    candidate_indexes.push(CatalogIndex {
        name: constraint_name.clone(),
        table: table_name.to_string(),
        column: column.clone(),
        unique: true,
        primary_key: false,
        unique_constraint: true,
    });
    validate_unique_indexes(table, &candidate_indexes)?;
    session.indexes.push(CatalogIndex {
        name: constraint_name,
        table: table_name.to_string(),
        column,
        unique: true,
        primary_key: false,
        unique_constraint: true,
    });
    session.dirty_indexes = true;
    Ok(())
}

pub(super) fn add_check_constraint_to_session(
    session: &mut Session,
    table_name: &str,
    constraint_name: String,
    filter: SelectFilter,
) -> Result<(), ErrorField> {
    if session
        .indexes
        .iter()
        .any(|index| index.name == constraint_name)
        || session.tables.values().any(|table| {
            table
                .check_constraints
                .iter()
                .any(|constraint| constraint.name == constraint_name)
                || table
                    .foreign_keys
                    .iter()
                    .any(|constraint| constraint.name == constraint_name)
        })
        || session.tables.contains_key(&constraint_name)
        || session.views.contains_key(&constraint_name)
        || session.materialized_views.contains_key(&constraint_name)
        || session.sequences.contains_key(&constraint_name)
    {
        return Err(ErrorField {
            code: "42710",
            message: "constraint already exists",
            position: None,
        });
    }
    let Some(table) = session.tables.get(table_name) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let Some(column) = table
        .columns
        .iter()
        .find(|column| column.def.name == filter.column)
    else {
        return Err(ErrorField {
            code: "42703",
            message: "column does not exist",
            position: None,
        });
    };
    if !sql_value_matches_type(&filter.value, column.def.ty) {
        return Err(ErrorField {
            code: "42804",
            message: "column type mismatch",
            position: None,
        });
    }
    let mut candidate = table.clone();
    candidate.check_constraints.push(CatalogCheckConstraint {
        name: constraint_name,
        table: table_name.to_string(),
        column: filter.column,
        op: filter.op,
        value: filter.value,
    });
    validate_check_constraints(&candidate)?;
    session
        .tables
        .get_mut(table_name)
        .expect("table existence checked")
        .check_constraints = candidate.check_constraints;
    session.mark_table_dirty(table_name.to_string());
    Ok(())
}
