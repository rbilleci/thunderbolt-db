// Legacy publication/subscription catalog ownership. This is not a product execution path.

use super::{
    bool_column, bool_text, int4_column, schema_permission_error, text_column,
    write_command_complete, write_error, write_single_row, CatalogCommentTarget, Column, Command,
    ErrorField, Publication, PublicationTarget, ReadWrite, SchemaPrivilege, Session, Subscription,
    PUBLIC_NAMESPACE_OID,
};
use std::collections::BTreeSet;
use std::io;

fn publication_table_target_error(session: &Session, table: &str) -> Option<ErrorField> {
    if session.views.contains_key(table)
        || session.materialized_views.contains_key(table)
        || session.sequences.contains_key(table)
    {
        return Some(ErrorField {
            code: "42809",
            message: "relation is not a table",
            position: None,
        });
    }
    if !session.tables.contains_key(table) {
        return Some(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    }
    None
}

fn create_publication(
    session: &mut Session,
    name: String,
    target: PublicationTarget,
) -> Result<(), ErrorField> {
    if session.publications.contains_key(&name) {
        return Err(ErrorField {
            code: "42710",
            message: "publication already exists",
            position: None,
        });
    }
    let (all_tables, tables) = match target {
        PublicationTarget::AllTables => (true, Vec::new()),
        PublicationTarget::Tables(tables) => {
            let mut seen = BTreeSet::new();
            for table in &tables {
                if !seen.insert(table.clone()) {
                    return Err(ErrorField {
                        code: "42710",
                        message: "publication table specified more than once",
                        position: None,
                    });
                }
                if let Some(error) = publication_table_target_error(session, table) {
                    return Err(error);
                }
            }
            (false, tables)
        }
    };
    let oid = session.next_relation_oid;
    session.next_relation_oid = session.next_relation_oid.checked_add(1).ok_or(ErrorField {
        code: "54000",
        message: "publication OID allocation exhausted",
        position: None,
    })?;
    session.publications.insert(
        name.clone(),
        Publication {
            oid,
            name: name.clone(),
            all_tables,
            tables,
        },
    );
    session.mark_publication_dirty(name);
    Ok(())
}

fn drop_publication(
    session: &mut Session,
    names: &[String],
    if_exists: bool,
) -> Result<(), ErrorField> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "publication specified more than once",
                position: None,
            });
        }
        if !if_exists && !session.publications.contains_key(name) {
            return Err(ErrorField {
                code: "42704",
                message: "publication does not exist",
                position: None,
            });
        }
    }
    for name in names {
        session.publications.remove(name);
        let target = CatalogCommentTarget::Publication {
            publication: name.clone(),
        };
        session.comments.remove(&target);
        session.mark_comment_dirty(target);
        session.mark_publication_dirty(name.clone());
    }
    Ok(())
}

fn create_subscription(
    session: &mut Session,
    name: String,
    connection: String,
    publications: Vec<String>,
) -> Result<(), ErrorField> {
    if session.subscriptions.contains_key(&name) {
        return Err(ErrorField {
            code: "42710",
            message: "subscription already exists",
            position: None,
        });
    }
    let mut seen = BTreeSet::new();
    for publication in &publications {
        if !seen.insert(publication.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "subscription publication specified more than once",
                position: None,
            });
        }
        if !session.publications.contains_key(publication) {
            return Err(ErrorField {
                code: "42704",
                message: "publication does not exist",
                position: None,
            });
        }
    }
    let oid = session.next_relation_oid;
    session.next_relation_oid = session.next_relation_oid.checked_add(1).ok_or(ErrorField {
        code: "54000",
        message: "subscription OID allocation exhausted",
        position: None,
    })?;
    session.subscriptions.insert(
        name.clone(),
        Subscription {
            oid,
            name: name.clone(),
            connection,
            publications,
            enabled: false,
        },
    );
    session.mark_subscription_dirty(name);
    Ok(())
}

fn drop_subscription(
    session: &mut Session,
    names: &[String],
    if_exists: bool,
) -> Result<(), ErrorField> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "subscription specified more than once",
                position: None,
            });
        }
        if !if_exists && !session.subscriptions.contains_key(name) {
            return Err(ErrorField {
                code: "42704",
                message: "subscription does not exist",
                position: None,
            });
        }
    }
    for name in names {
        session.subscriptions.remove(name);
        let target = CatalogCommentTarget::Subscription {
            subscription: name.clone(),
        };
        session.comments.remove(&target);
        session.mark_comment_dirty(target);
        session.mark_subscription_dirty(name.clone());
    }
    Ok(())
}

fn psql_list_publications_catalog_query() -> &'static str {
    "select pubname as \"name\", pg_catalog.pg_get_userbyid(pubowner) as \"owner\", puballtables as \"all tables\", pubinsert as \"inserts\", pubupdate as \"updates\", pubdelete as \"deletes\", pubtruncate as \"truncates\", pubviaroot as \"via root\" from pg_catalog.pg_publication order by 1"
}

fn psql_list_publications_verbose_catalog_query() -> &'static str {
    "select oid, pubname, pg_catalog.pg_get_userbyid(pubowner) as owner, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot from pg_catalog.pg_publication order by 2"
}

fn psql_list_subscriptions_catalog_query() -> &'static str {
    "select subname as \"name\" , pg_catalog.pg_get_userbyid(subowner) as \"owner\" , subenabled as \"enabled\" , subpublications as \"publication\" from pg_catalog.pg_subscription where subdbid = (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database())order by 1"
}

fn psql_describe_schema_publications_query() -> &'static str {
    "select pubname from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_namespace n on n.oid = pn.pnnspid where n.nspname = 'public' order by 1"
}

fn catalog_describe_publication_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pubname , null , null from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_class pc on pc.relnamespace = pn.pnnspid where pc.oid ='";
    let middle = "' and pg_catalog.pg_relation_is_publishable('";
    let suffix = "') union select pubname , pg_get_expr(pr.prqual, c.oid) , (case when pr.prattrs is not null then (select string_agg(attname, ', ') from pg_catalog.generate_series(0, pg_catalog.array_upper(pr.prattrs::pg_catalog.int2[], 1)) s, pg_catalog.pg_attribute where attrelid = pr.prrelid and attnum = prattrs[s]) else null end) from pg_catalog.pg_publication p join pg_catalog.pg_publication_rel pr on p.oid = pr.prpubid join pg_catalog.pg_class c on c.oid = pr.prrelid where pr.prrelid = '";
    let suffix_tail =
        "' union select pubname , null , null from pg_catalog.pg_publication p where p.puballtables and pg_catalog.pg_relation_is_publishable('";
    let final_suffix = "') order by 1";
    let rest = canonical.strip_prefix(prefix)?;
    let (first_oid, rest) = rest.split_once(middle)?;
    let (second_oid, rest) = rest.split_once(suffix)?;
    let (third_oid, rest) = rest.split_once(suffix_tail)?;
    let final_oid = rest.strip_suffix(final_suffix)?;
    if first_oid == second_oid && second_oid == third_oid && third_oid == final_oid {
        first_oid.parse().ok()
    } else {
        None
    }
}

fn pg_catalog_publication_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("pubname"),
        int4_column("pubowner"),
        bool_column("puballtables"),
        bool_column("pubinsert"),
        bool_column("pubupdate"),
        bool_column("pubdelete"),
        bool_column("pubtruncate"),
        bool_column("pubviaroot"),
    ]
}

fn pg_catalog_publication_rel_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("prpubid"),
        int4_column("prrelid"),
        text_column("prrelqual"),
        text_column("prattrs"),
    ]
}

fn pg_catalog_publication_namespace_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("pnpubid"),
        int4_column("pnnspid"),
    ]
}

fn catalog_psql_publication_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .map(|publication| {
            vec![
                Some(publication.name.clone()),
                Some("postgres".to_string()),
                Some(bool_text(publication.all_tables)),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect()
}

fn catalog_psql_publication_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .map(|publication| {
            vec![
                Some(publication.oid.to_string()),
                Some(publication.name.clone()),
                Some("postgres".to_string()),
                Some(bool_text(publication.all_tables)),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect()
}

fn catalog_publication_class_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .map(|publication| {
            vec![
                Some("6104".to_string()),
                Some(publication.oid.to_string()),
                Some(publication.name.clone()),
                Some("10".to_string()),
                Some(bool_text(publication.all_tables)),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect()
}

fn catalog_publication_direct_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .map(|publication| {
            vec![
                Some(publication.name.clone()),
                Some(bool_text(publication.all_tables)),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect()
}

fn catalog_publication_rel_direct_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for publication in session.publications.values() {
        if publication.all_tables {
            continue;
        }
        for table in &publication.tables {
            if session.tables.contains_key(table) {
                rows.push(vec![Some(publication.name.clone()), Some(table.clone())]);
            }
        }
    }
    rows.sort();
    rows
}

fn catalog_publication_rel_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for publication in session.publications.values() {
        if publication.all_tables {
            continue;
        }
        for table in &publication.tables {
            if let Some(table_state) = session.tables.get(table) {
                rows.push(vec![
                    Some("6106".to_string()),
                    Some(format!("{}{}", publication.oid, table_state.oid)),
                    Some(publication.oid.to_string()),
                    Some(table_state.oid.to_string()),
                    None,
                    None,
                ]);
            }
        }
    }
    rows
}

fn catalog_publication_namespace_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .filter(|publication| publication.all_tables)
        .map(|publication| {
            vec![
                Some("6237".to_string()),
                Some((publication.oid + 100_000).to_string()),
                Some(publication.oid.to_string()),
                Some(PUBLIC_NAMESPACE_OID.to_string()),
            ]
        })
        .collect()
}

fn catalog_describe_publication_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for publication in session.publications.values() {
        if publication.all_tables || publication.tables.iter().any(|name| name == &table.name) {
            rows.push(vec![Some(publication.name.clone()), None, None]);
        }
    }
    rows
}

fn catalog_psql_subscription_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .subscriptions
        .values()
        .map(|subscription| {
            vec![
                Some(subscription.name.clone()),
                Some("postgres".to_string()),
                Some(bool_text(subscription.enabled)),
                Some(subscription.publications.join(", ")),
            ]
        })
        .collect()
}

fn catalog_subscription_direct_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .subscriptions
        .values()
        .map(|subscription| {
            vec![
                Some(subscription.name.clone()),
                Some(bool_text(subscription.enabled)),
                Some(subscription.connection.clone()),
                Some(format!("{{{}}}", subscription.publications.join(","))),
            ]
        })
        .collect()
}

fn catalog_schema_publication_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .publications
        .values()
        .filter(|publication| publication.all_tables)
        .map(|publication| vec![Some(publication.name.clone())])
        .collect()
}

#[cfg(test)]
pub(super) fn test_psql_list_publications_catalog_query() -> &'static str {
    psql_list_publications_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_publications_verbose_catalog_query() -> &'static str {
    psql_list_publications_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_subscriptions_catalog_query() -> &'static str {
    psql_list_subscriptions_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_schema_publications_query() -> &'static str {
    psql_describe_schema_publications_query()
}

#[cfg(test)]
pub(super) fn test_catalog_describe_publication_query_oid(canonical: &str) -> Option<u32> {
    catalog_describe_publication_query_oid(canonical)
}

pub(super) fn try_execute_replication_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_list_publications_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("All tables"),
                bool_column("Inserts"),
                bool_column("Updates"),
                bool_column("Deletes"),
                bool_column("Truncates"),
                bool_column("Via root"),
            ],
            &catalog_psql_publication_rows(session),
        ));
    }
    if canonical == psql_list_publications_verbose_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("pubname"),
                text_column("owner"),
                bool_column("puballtables"),
                bool_column("pubinsert"),
                bool_column("pubupdate"),
                bool_column("pubdelete"),
                bool_column("pubtruncate"),
                bool_column("pubviaroot"),
            ],
            &catalog_psql_publication_verbose_rows(session),
        ));
    }
    if canonical
        == "select pubname, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot from pg_catalog.pg_publication order by pubname"
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("pubname"),
                bool_column("puballtables"),
                bool_column("pubinsert"),
                bool_column("pubupdate"),
                bool_column("pubdelete"),
                bool_column("pubtruncate"),
                bool_column("pubviaroot"),
            ],
            &catalog_publication_direct_rows(session),
        ));
    }
    if canonical
        == "select p.pubname, c.relname from pg_catalog.pg_publication p join pg_catalog.pg_publication_rel pr on pr.prpubid = p.oid join pg_catalog.pg_class c on c.oid = pr.prrelid order by p.pubname, c.relname"
    {
        return Some(write_single_row(
            stream,
            &[text_column("pubname"), text_column("relname")],
            &catalog_publication_rel_direct_rows(session),
        ));
    }
    if canonical == psql_list_subscriptions_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("Enabled"),
                text_column("Publication"),
            ],
            &catalog_psql_subscription_rows(session),
        ));
    }
    if canonical
        == "select subname, subenabled, subconninfo, subpublications from pg_catalog.pg_subscription order by subname"
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("subname"),
                bool_column("subenabled"),
                text_column("subconninfo"),
                text_column("subpublications"),
            ],
            &catalog_subscription_direct_rows(session),
        ));
    }
    if canonical == psql_describe_schema_publications_query() {
        return Some(write_single_row(
            stream,
            &[text_column("pubname")],
            &catalog_schema_publication_rows(session),
        ));
    }
    if let Some(oid) = catalog_describe_publication_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("pubname"),
                text_column("?column?"),
                text_column("?column?"),
            ],
            &catalog_describe_publication_rows(session, oid),
        ));
    }
    None
}

pub(super) fn try_execute_replication_pg_dump_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical.starts_with("select p.tableoid, p.oid, p.pubname")
        && canonical.contains("from pg_publication p")
    {
        return Some(write_single_row(
            stream,
            &pg_catalog_publication_columns(),
            &catalog_publication_class_rows(session),
        ));
    }
    if canonical.starts_with("select tableoid, oid, prpubid, prrelid")
        && canonical.contains("from pg_catalog.pg_publication_rel pr")
    {
        return Some(write_single_row(
            stream,
            &pg_catalog_publication_rel_columns(),
            &catalog_publication_rel_rows(session),
        ));
    }
    if canonical
        == "select tableoid, oid, pnpubid, pnnspid from pg_catalog.pg_publication_namespace"
    {
        return Some(write_single_row(
            stream,
            &pg_catalog_publication_namespace_columns(),
            &catalog_publication_namespace_rows(session),
        ));
    }
    None
}

pub(super) fn execute_replication_catalog_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreatePublication(create) => {
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if let Err(error) = create_publication(session, create.name, create.target) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE PUBLICATION")
        }
        Command::DropPublication(drop) => {
            if let Err(error) = drop_publication(session, &drop.names, drop.if_exists) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP PUBLICATION")
        }
        Command::CreateSubscription(create) => {
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if let Err(error) =
                create_subscription(session, create.name, create.connection, create.publications)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE SUBSCRIPTION")
        }
        Command::DropSubscription(drop) => {
            if let Err(error) = drop_subscription(session, &drop.names, drop.if_exists) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP SUBSCRIPTION")
        }
        _ => unreachable!("replication catalog executor called with an unrelated command"),
    }
}
