// Legacy catalog-comment ownership. This is not a product execution path.

use super::{
    catalog_constraint_entries, catalog_constraint_oid, catalog_index_oid, database_exists,
    role_exists, shared_catalog, tablespace_exists, text_column, write_command_complete,
    write_error, write_single_row, CatalogCommentTarget, ErrorField, ReadWrite, Session,
    PG_EXTENSION_CLASS_OID, PG_PUBLICATION_CLASS_OID, PG_SUBSCRIPTION_CLASS_OID,
    PLPGSQL_EXTENSION_OID, PUBLIC_NAMESPACE_OID,
};
use gpu_db_protocol::{Command, CommentTarget};
use std::io;

pub(super) fn pg_dump_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Extension {
        extension: "plpgsql".to_string(),
    }) {
        rows.push(vec![
            Some(description.clone()),
            Some(PG_EXTENSION_CLASS_OID.to_string()),
            Some(PLPGSQL_EXTENSION_OID.to_string()),
            Some("0".to_string()),
        ]);
    }
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Schema {
        schema: "public".to_string(),
    }) {
        rows.push(vec![
            Some(description.clone()),
            Some("2615".to_string()),
            Some(PUBLIC_NAMESPACE_OID.to_string()),
            Some("0".to_string()),
        ]);
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
        for column in &table.columns {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Column {
                table: table.name.clone(),
                attnum: column.attnum,
            }) {
                rows.push(vec![
                    Some(description.clone()),
                    Some("1259".to_string()),
                    Some(table.oid.to_string()),
                    Some(column.attnum.to_string()),
                ]);
            }
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(sequence.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut domains = session.domains.values().collect::<Vec<_>>();
    domains.sort_by_key(|domain| domain.oid);
    for domain in domains {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Domain {
            domain: domain.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1247".to_string()),
                Some(domain.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut functions = session.functions.values().collect::<Vec<_>>();
    functions.sort_by(|left, right| left.name.cmp(&right.name));
    for function in functions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Function {
            function: function.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(function.name.clone()),
                Some("function".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut publications = session.publications.values().collect::<Vec<_>>();
    publications.sort_by_key(|publication| publication.oid);
    for publication in publications {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Publication {
            publication: publication.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some(PG_PUBLICATION_CLASS_OID.to_string()),
                Some(publication.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut subscriptions = session.subscriptions.values().collect::<Vec<_>>();
    subscriptions.sort_by_key(|subscription| subscription.oid);
    for subscription in subscriptions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Subscription {
            subscription: subscription.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some(PG_SUBSCRIPTION_CLASS_OID.to_string()),
                Some(subscription.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut index_comments = session
        .comments
        .iter()
        .filter_map(|(target, description)| match target {
            CatalogCommentTarget::Index { index } => Some((index, description)),
            CatalogCommentTarget::Database { .. }
            | CatalogCommentTarget::Role { .. }
            | CatalogCommentTarget::Schema { .. }
            | CatalogCommentTarget::Tablespace { .. }
            | CatalogCommentTarget::Table { .. }
            | CatalogCommentTarget::Column { .. }
            | CatalogCommentTarget::View { .. }
            | CatalogCommentTarget::MaterializedView { .. }
            | CatalogCommentTarget::Extension { .. }
            | CatalogCommentTarget::Function { .. }
            | CatalogCommentTarget::Sequence { .. }
            | CatalogCommentTarget::Domain { .. }
            | CatalogCommentTarget::Publication { .. }
            | CatalogCommentTarget::Subscription { .. }
            | CatalogCommentTarget::Constraint { .. } => None,
        })
        .collect::<Vec<_>>();
    index_comments.sort_by(|left, right| left.0.cmp(right.0));
    for (index, description) in index_comments {
        if let Some(oid) = catalog_index_oid(session, index) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut constraint_comments = session
        .comments
        .iter()
        .filter_map(|(target, description)| match target {
            CatalogCommentTarget::Constraint { table, constraint } => {
                Some((table, constraint, description))
            }
            CatalogCommentTarget::Database { .. }
            | CatalogCommentTarget::Role { .. }
            | CatalogCommentTarget::Schema { .. }
            | CatalogCommentTarget::Tablespace { .. }
            | CatalogCommentTarget::Table { .. }
            | CatalogCommentTarget::Column { .. }
            | CatalogCommentTarget::View { .. }
            | CatalogCommentTarget::MaterializedView { .. }
            | CatalogCommentTarget::Extension { .. }
            | CatalogCommentTarget::Function { .. }
            | CatalogCommentTarget::Sequence { .. }
            | CatalogCommentTarget::Domain { .. }
            | CatalogCommentTarget::Publication { .. }
            | CatalogCommentTarget::Subscription { .. }
            | CatalogCommentTarget::Index { .. } => None,
        })
        .collect::<Vec<_>>();
    constraint_comments
        .sort_by(|left, right| left.0.cmp(right.0).then_with(|| left.1.cmp(right.1)));
    for (table_name, constraint_name, description) in constraint_comments {
        if let Some(entry) = catalog_constraint_entries(session)
            .into_iter()
            .find(|entry| &entry.index.table == table_name && &entry.index.name == constraint_name)
        {
            rows.push(vec![
                Some(description.clone()),
                Some("2606".to_string()),
                Some(catalog_constraint_oid(&entry).to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| {
        let left_key = (
            left[1]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            left[2]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            left[3]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
        );
        let right_key = (
            right[1]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            right[2]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            right[3]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    rows
}

fn pg_catalog_schema_description_query() -> &'static str {
    "select n.nspname, pg_catalog.obj_description(n.oid, 'pg_namespace') as description from pg_catalog.pg_namespace n where n.nspname = 'public' order by n.nspname"
}

fn pg_catalog_schema_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        session
            .comments
            .get(&CatalogCommentTarget::Schema {
                schema: "public".to_string(),
            })
            .cloned(),
    ]]
}

pub(super) fn try_execute_schema_description_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != pg_catalog_schema_description_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &[text_column("nspname"), text_column("description")],
        &pg_catalog_schema_description_rows(session),
    ))
}

fn pg_catalog_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v') order by c.relname, d.objsubid"
}

fn pg_catalog_descriptions_with_sequences_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v','s') order by c.relname, d.objsubid"
}

fn pg_catalog_descriptions_with_materialized_views_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v','m','s') order by c.relname, d.objsubid"
}

fn pg_catalog_table_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind = 'r' order by c.relname, d.objsubid"
}

fn pg_catalog_constraint_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, con.conname, d.description from pg_catalog.pg_description d join pg_catalog.pg_constraint con on con.oid = d.objoid join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
}

fn pg_catalog_table_index_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_sequence_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v','s') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_sequence_matview_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v','m','s') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_descriptions_without_views_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
        for column in &table.columns {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Column {
                table: table.name.clone(),
                attnum: column.attnum,
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
    }
    rows
}

fn pg_catalog_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = pg_catalog_table_description_rows(session);
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    rows
}

fn pg_catalog_constraint_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for entry in catalog_constraint_entries(session) {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
            table: entry.index.table.clone(),
            constraint: entry.index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(entry.index.table),
                Some(entry.index.name),
                Some(description.clone()),
            ]);
        }
    }
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
                table: table.name.clone(),
                constraint: constraint.name.clone(),
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(constraint.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
        for constraint in &table.foreign_keys {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
                table: table.name.clone(),
                constraint: constraint.name.clone(),
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(constraint.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn pg_catalog_table_index_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut indexes = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .collect::<Vec<_>>();
    indexes.sort_by(|left, right| left.name.cmp(&right.name));
    for index in indexes {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Index {
            index: index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(index.name.clone()),
                Some("i".to_string()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    for row in pg_catalog_description_rows(session) {
        let name = row[1].as_deref().unwrap_or_default();
        let relkind = if session.views.contains_key(name) {
            "v"
        } else if session.materialized_views.contains_key(name) {
            "m"
        } else if session.sequences.contains_key(name) {
            "s"
        } else {
            "r"
        };
        rows.push(vec![
            row[0].clone(),
            row[1].clone(),
            Some(relkind.to_string()),
            row[2].clone(),
            row[3].clone(),
        ]);
    }
    rows
}

fn pg_catalog_table_index_description_rows_without_views(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut indexes = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .collect::<Vec<_>>();
    indexes.sort_by(|left, right| left.name.cmp(&right.name));
    for index in indexes {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Index {
            index: index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(index.name.clone()),
                Some("i".to_string()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    for row in pg_catalog_table_description_rows(session) {
        rows.push(vec![
            row[0].clone(),
            row[1].clone(),
            Some("r".to_string()),
            row[2].clone(),
            row[3].clone(),
        ]);
    }
    rows
}

fn psql_object_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Extension {
        extension: "plpgsql".to_string(),
    }) {
        rows.push(vec![
            Some("pg_catalog".to_string()),
            Some("plpgsql".to_string()),
            Some("extension".to_string()),
            Some(description.clone()),
        ]);
    }
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Schema {
        schema: "public".to_string(),
    }) {
        rows.push(vec![
            Some("public".to_string()),
            Some("public".to_string()),
            Some("schema".to_string()),
            Some(description.clone()),
        ]);
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("view".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("materialized view".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("sequence".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut publications = session.publications.values().collect::<Vec<_>>();
    publications.sort_by(|left, right| left.name.cmp(&right.name));
    for publication in publications {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Publication {
            publication: publication.name.clone(),
        }) {
            rows.push(vec![
                None,
                Some(publication.name.clone()),
                Some("publication".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut subscriptions = session.subscriptions.values().collect::<Vec<_>>();
    subscriptions.sort_by(|left, right| left.name.cmp(&right.name));
    for subscription in subscriptions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Subscription {
            subscription: subscription.name.clone(),
        }) {
            rows.push(vec![
                None,
                Some(subscription.name.clone()),
                Some("subscription".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut constraints = catalog_constraint_entries(session);
    constraints.sort_by(|left, right| {
        left.index
            .table
            .cmp(&right.index.table)
            .then_with(|| left.index.name.cmp(&right.index.name))
    });
    for entry in constraints {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
            table: entry.index.table.clone(),
            constraint: entry.index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(entry.index.name),
                Some("table constraint".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    rows
}

fn psql_list_object_descriptions_query(canonical: &str) -> bool {
    canonical.starts_with(
        "select distinct tt.nspname as \"schema\", tt.name as \"name\", tt.object as \"object\", d.description as \"description\" from ( select pgc.oid as oid, pgc.tableoid as tableoid",
    ) && canonical.contains("cast('table constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('domain constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('operator class' as pg_catalog.text) as object")
        && canonical.contains("cast('operator family' as pg_catalog.text) as object")
        && canonical.contains("cast('rule' as pg_catalog.text) as object")
        && canonical.contains("cast('trigger' as pg_catalog.text) as object")
        && canonical.contains("join pg_catalog.pg_description d on (tt.oid = d.objoid and tt.tableoid = d.classoid and d.objsubid = 0)")
        && canonical.ends_with("order by 1, 2, 3")
}

pub(super) fn try_execute_relation_description_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    let (columns, rows) = if canonical == pg_catalog_descriptions_query()
        || canonical == pg_catalog_descriptions_with_sequences_query()
        || canonical == pg_catalog_descriptions_with_materialized_views_query()
    {
        (
            vec![
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("description"),
            ],
            pg_catalog_description_rows(session),
        )
    } else if canonical == pg_catalog_table_descriptions_query() {
        (
            vec![
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("description"),
            ],
            pg_catalog_table_description_rows(session),
        )
    } else if canonical == pg_catalog_constraint_descriptions_query() {
        (
            vec![
                text_column("nspname"),
                text_column("relname"),
                text_column("conname"),
                text_column("description"),
            ],
            pg_catalog_constraint_description_rows(session),
        )
    } else if canonical == pg_catalog_table_index_descriptions_query()
        || canonical == pg_catalog_table_index_sequence_descriptions_query()
        || canonical == pg_catalog_table_index_sequence_matview_descriptions_query()
    {
        (
            vec![
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("attname"),
                text_column("description"),
            ],
            pg_catalog_table_index_description_rows(session),
        )
    } else if canonical == pg_catalog_table_index_descriptions_without_views_query() {
        (
            vec![
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("attname"),
                text_column("description"),
            ],
            pg_catalog_table_index_description_rows_without_views(session),
        )
    } else if psql_list_object_descriptions_query(canonical) {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Object"),
                text_column("Description"),
            ],
            psql_object_description_rows(session),
        )
    } else {
        return None;
    };
    Some(write_single_row(stream, &columns, &rows))
}

#[cfg(test)]
pub(super) fn test_pg_dump_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_dump_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_schema_description_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    pg_catalog_schema_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_table_description_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    pg_catalog_table_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_constraint_description_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    pg_catalog_constraint_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_table_descriptions_query() -> &'static str {
    pg_catalog_table_descriptions_query()
}

#[cfg(test)]
pub(super) fn test_psql_object_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    psql_object_description_rows(session)
}

#[cfg(test)]
pub(super) fn test_psql_list_object_descriptions_query(canonical: &str) -> bool {
    psql_list_object_descriptions_query(canonical)
}

fn shared_catalog_contains_table(table: &str) -> bool {
    shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned")
        .tables
        .contains_key(table)
}

fn shared_catalog_contains_view(view: &str) -> bool {
    shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned")
        .views
        .contains_key(view)
}

fn shared_catalog_contains_sequence(sequence: &str) -> bool {
    shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned")
        .sequences
        .contains_key(sequence)
}

fn shared_catalog_contains_live_index(index: &str) -> bool {
    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog
        .indexes
        .iter()
        .any(|candidate| candidate.name == index && catalog.tables.contains_key(&candidate.table))
}

fn shared_catalog_contains_table_constraint(table: &str, constraint: &str) -> bool {
    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    (catalog.tables.contains_key(table)
        && catalog.indexes.iter().any(|candidate| {
            candidate.table == table
                && candidate.name == constraint
                && (candidate.primary_key || candidate.unique_constraint)
        }))
        || catalog.tables.get(table).is_some_and(|table| {
            table
                .check_constraints
                .iter()
                .any(|candidate| candidate.name == constraint)
        })
}

pub(super) fn execute_catalog_comment(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CommentOn(comment) => {
            let target = match comment.target {
                CommentTarget::Database { database } => {
                    if !database_exists(session, &database) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "3D000",
                                message: "database does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Database { database }
                }
                CommentTarget::Role { role } => {
                    if !role_exists(session, &role) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "role does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Role { role }
                }
                CommentTarget::Schema { schema } => {
                    if schema != "public" {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "3F000",
                                message: "schema does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Schema { schema }
                }
                CommentTarget::Tablespace { tablespace } => {
                    if !tablespace_exists(session, &tablespace) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "tablespace does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Tablespace { tablespace }
                }
                CommentTarget::Table { table } => {
                    if !session.tables.contains_key(&table) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Table { table }
                }
                CommentTarget::Column { table, column } => {
                    let Some(table_ref) = session.tables.get(&table) else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    };
                    let Some(column_ref) = table_ref
                        .columns
                        .iter()
                        .find(|candidate| candidate.def.name == column)
                    else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    CatalogCommentTarget::Column {
                        table,
                        attnum: column_ref.attnum,
                    }
                }
                CommentTarget::Index { index } => {
                    let exists = session.indexes.iter().any(|candidate| {
                        candidate.name == index && session.tables.contains_key(&candidate.table)
                    }) || (session.shared_catalog
                        && shared_catalog_contains_live_index(&index));
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "index does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Index { index }
                }
                CommentTarget::View { view } => {
                    let exists = session.views.contains_key(&view)
                        || (session.shared_catalog && shared_catalog_contains_view(&view));
                    if !exists {
                        if session.tables.contains_key(&view)
                            || (session.shared_catalog && shared_catalog_contains_table(&view))
                            || session.sequences.contains_key(&view)
                            || (session.shared_catalog && shared_catalog_contains_sequence(&view))
                        {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42809",
                                    message: "relation is not a view",
                                    position: None,
                                },
                            );
                        }
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "view does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::View { view }
                }
                CommentTarget::MaterializedView { materialized_view } => {
                    let exists = session.materialized_views.contains_key(&materialized_view);
                    if !exists {
                        if session.tables.contains_key(&materialized_view)
                            || session.views.contains_key(&materialized_view)
                            || session.sequences.contains_key(&materialized_view)
                        {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42809",
                                    message: "relation is not a materialized view",
                                    position: None,
                                },
                            );
                        }
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "materialized view does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::MaterializedView { materialized_view }
                }
                CommentTarget::Function { function } => {
                    if !session.functions.contains_key(&function) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42883",
                                message: "function does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Function { function }
                }
                CommentTarget::Extension { extension } => {
                    if extension != "plpgsql" {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "extension does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Extension { extension }
                }
                CommentTarget::Sequence { sequence } => {
                    let exists = session.sequences.contains_key(&sequence)
                        || (session.shared_catalog && shared_catalog_contains_sequence(&sequence));
                    if !exists {
                        if session.tables.contains_key(&sequence)
                            || session.views.contains_key(&sequence)
                            || (session.shared_catalog
                                && (shared_catalog_contains_table(&sequence)
                                    || shared_catalog_contains_view(&sequence)))
                        {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42809",
                                    message: "relation is not a sequence",
                                    position: None,
                                },
                            );
                        }
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "sequence does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Sequence { sequence }
                }
                CommentTarget::Domain { domain } => {
                    let exists = session.domains.contains_key(&domain);
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "domain does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Domain { domain }
                }
                CommentTarget::Publication { publication } => {
                    if !session.publications.contains_key(&publication) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "publication does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Publication { publication }
                }
                CommentTarget::Subscription { subscription } => {
                    if !session.subscriptions.contains_key(&subscription) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "subscription does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Subscription { subscription }
                }
                CommentTarget::Constraint { table, constraint } => {
                    let table_exists = session.tables.contains_key(&table)
                        || (session.shared_catalog && shared_catalog_contains_table(&table));
                    if !table_exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    }
                    let exists = session.indexes.iter().any(|candidate| {
                        candidate.table == table
                            && candidate.name == constraint
                            && (candidate.primary_key || candidate.unique_constraint)
                    }) || session.tables.get(&table).is_some_and(|table| {
                        table
                            .check_constraints
                            .iter()
                            .any(|candidate| candidate.name == constraint)
                            || table
                                .foreign_keys
                                .iter()
                                .any(|candidate| candidate.name == constraint)
                    }) || (session.shared_catalog
                        && shared_catalog_contains_table_constraint(&table, &constraint));
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "constraint does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Constraint { table, constraint }
                }
            };
            if let Some(value) = comment.comment {
                session.comments.insert(target.clone(), value);
            } else {
                session.comments.remove(&target);
            }
            session.mark_comment_dirty(target);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "COMMENT")
        }
        _ => unreachable!("catalog-comment executor received an unrelated command"),
    }
}
