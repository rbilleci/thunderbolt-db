//! PostgreSQL 16 dump catalog programs executed as prepared GPU catalog routes.
//!
//! `pg_dump` emits stable, versioned catalog programs whose presentation expressions are wider
//! than the general SQL binder currently accepts. A recognized program encodes complete,
//! query-independent candidate relations from one pinned [`CatalogSnapshot`], then a typed plan runs
//! its filters, joins, aggregate, projection, and ordering through the ordinary GPU relational
//! executors. Host-side cardinality shortcuts are limited to relations proven authoritatively empty
//! by the modeled catalog state. Recognition is fail-closed and happens only after libpg_query has
//! accepted the statement. This module owns no session state, mutation admission, sequence
//! allocation, WAL claim, or publication.

mod comments;
mod database;
mod dependencies;
mod gpu_plans;
#[cfg(test)]
mod gpu_tests;
mod indexes;
mod prepared;
mod replication;
mod sequences;

use super::*;
use gpu_plans::*;
pub(crate) use prepared::pg16_prepared_catalog_program_table;

impl Engine {
    pub fn execute_prepared_catalog_program(
        &self,
        program: &PreparedCatalogProgram,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.execute_prepared_catalog_program_scoped(program)
    }

    pub fn execute_prepared_catalog_program_in_transaction(
        &self,
        txn_id: TxnId,
        program: &PreparedCatalogProgram,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let _scope = self.enter_transaction_read(snapshot);
        self.execute_prepared_catalog_program_scoped(program)
    }

    fn execute_prepared_catalog_program_scoped(
        &self,
        program: &PreparedCatalogProgram,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        self.execute_pg16_prepared_catalog_program_gpu(program, &catalog, boundary)
    }

    pub(super) fn execute_pg_dump_catalog_route_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        if is_pg16_dump_class_metadata_program(&canonical) {
            return self.execute_pg16_dump_class_metadata().map(Some);
        }
        if is_pg16_dump_function_metadata_program(&canonical) {
            return self.execute_pg16_dump_function_metadata().map(Some);
        }
        if is_pg16_dump_type_metadata_program(&canonical) {
            return self.execute_pg16_dump_type_metadata().map(Some);
        }
        if is_pg16_dump_language_metadata_program(&canonical) {
            return self.execute_pg16_dump_language_metadata().map(Some);
        }
        if is_pg16_dump_default_acl_program(&canonical) {
            return self.execute_pg16_dump_default_acl().map(Some);
        }
        if let Some(relation_oids) = pg16_dump_attribute_program_oids(&canonical) {
            return self
                .execute_pg16_dump_attribute_metadata(&relation_oids)
                .map(Some);
        }
        if let Some(relation_oids) = pg16_dump_attrdef_program_oids(&canonical) {
            return self
                .execute_pg16_dump_attrdef_metadata(&relation_oids)
                .map(Some);
        }
        if let Some(relation_oids) = indexes::pg16_dump_index_metadata_program_oids(&canonical) {
            return self
                .execute_pg16_dump_index_metadata(&relation_oids)
                .map(Some);
        }
        if let Some(relation_oids) =
            indexes::pg16_dump_foreign_key_metadata_program_oids(&canonical)
        {
            return self
                .execute_pg16_dump_foreign_key_metadata(&relation_oids)
                .map(Some);
        }
        if let Some(kind) = replication::pg16_dump_replication_program(&canonical) {
            return self.execute_pg16_dump_replication_metadata(kind).map(Some);
        }
        if is_pg16_dump_subscription_count_program(&canonical) {
            return self.execute_pg16_dump_subscription_count().map(Some);
        }
        if let Some(view_oid) = dependencies::pg16_dump_view_definition_oid(&canonical) {
            return self.execute_pg16_dump_view_definition(view_oid).map(Some);
        }
        if dependencies::is_pg16_dump_extension_membership_program(&canonical) {
            return self.execute_pg16_dump_extension_membership().map(Some);
        }
        if let Some(sequence_oid) = sequences::pg16_dump_sequence_metadata_oid(&canonical) {
            return self
                .execute_pg16_dump_sequence_metadata(sequence_oid)
                .map(Some);
        }
        if let Some(sequence) = sequences::pg16_dump_sequence_state_name(sql) {
            // The SQL shape alone is not authoritative: an ordinary user table may legitimately
            // expose columns named `last_value` and `is_called`. Resolve the object kind from the
            // catalog pinned by the surrounding statement/transaction scope; if it is not a
            // sequence, fall through to the ordinary GPU relational binder.
            let boundary = self.read_snapshot_boundary();
            let catalog = self.read_catalog_as_of(boundary);
            if catalog.relational_sequences.contains_key(&sequence) {
                return self.execute_pg16_dump_sequence_state(&sequence).map(Some);
            }
        }
        if dependencies::is_pg16_dump_dependency_program(&canonical) {
            return self.execute_pg16_dump_dependencies().map(Some);
        }
        if comments::is_pg16_dump_description_program(&canonical) {
            return self.execute_pg16_dump_descriptions().map(Some);
        }
        if database::is_pg16_dump_database_metadata_program(&canonical) {
            return self.execute_pg16_dump_database_metadata().map(Some);
        }
        if let Some(table) = pg16_dump_authoritatively_empty_relation(&canonical) {
            let boundary = self.read_snapshot_boundary();
            return self
                .execute_pg_dump_transient_relation(table, Vec::new(), &[], boundary)
                .map(Some);
        }
        Ok(None)
    }

    fn execute_pg16_dump_class_metadata(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_class_metadata_relation(&catalog);
        self.execute_pg16_dump_class_gpu_plan(table, rows, &catalog, boundary)
    }

    fn execute_pg16_dump_function_metadata(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_function_metadata_relation(&catalog);
        self.execute_pg16_dump_function_gpu_plan(table, rows, boundary)
    }

    fn execute_pg16_dump_type_metadata(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_type_metadata_relation(&catalog);
        self.execute_pg16_dump_type_gpu_plan(table, rows, &catalog, boundary)
    }

    fn execute_pg16_dump_language_metadata(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let (table, rows) = pg16_dump_language_metadata_relation();
        let predicate = bool_comparison(&table, "__lanispl", true)?;
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::Columns(
                [
                    "tableoid",
                    "oid",
                    "lanname",
                    "lanpltrusted",
                    "lanplcallfoid",
                    "laninline",
                    "lanvalidator",
                    "lanacl",
                    "acldefault",
                    "lanowner",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            Some(predicate),
            &["oid"],
            boundary,
        )
    }

    fn execute_pg16_dump_default_acl(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (table, rows) = pg16_dump_default_acl_relation(&catalog);
        self.execute_pg_dump_transient_relation(table, rows, &[], boundary)
    }

    fn execute_pg16_dump_attribute_metadata(
        &self,
        relation_oids: &[u32],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (source, source_rows) = oid_source_relation(relation_oids);
        let (attributes, attribute_rows) = pg16_dump_attribute_metadata_relation(&catalog);
        let (types, type_rows) = pg16_dump_attribute_type_relation(&catalog);
        let predicate = int4_comparison(&attributes, "attnum", ResidentBinaryOp::Gt, 0)?;
        let projection = vec![
            ("a", "attrelid", "attrelid"),
            ("a", "attnum", "attnum"),
            ("a", "attname", "attname"),
            ("a", "attstattarget", "attstattarget"),
            ("a", "attstorage", "attstorage"),
            ("t", "typstorage", "typstorage"),
            ("a", "attnotnull", "attnotnull"),
            ("a", "atthasdef", "atthasdef"),
            ("a", "attisdropped", "attisdropped"),
            ("a", "attlen", "attlen"),
            ("a", "attalign", "attalign"),
            ("a", "attislocal", "attislocal"),
            ("a", "atttypname", "atttypname"),
            ("a", "attoptions", "attoptions"),
            ("a", "attcollation", "attcollation"),
            ("a", "attfdwoptions", "attfdwoptions"),
            ("a", "attcompression", "attcompression"),
            ("a", "attidentity", "attidentity"),
            ("a", "attmissingval", "attmissingval"),
            ("a", "attgenerated", "attgenerated"),
        ];
        let plan = join_plan(
            &[(&source, "src"), (&attributes, "a"), (&types, "t")],
            vec![
                join_step("src", "tbloid", "a", "attrelid", false),
                join_step("a", "atttypid", "t", "oid", true),
            ],
            projection,
            vec![("a", "attrelid"), ("a", "attnum")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![source, attributes, types],
            vec![source_rows, attribute_rows, type_rows],
            vec![None, Some(predicate), None],
            boundary,
        )
    }

    fn execute_pg16_dump_attrdef_metadata(
        &self,
        relation_oids: &[u32],
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let (source, source_rows) = oid_source_relation(relation_oids);
        let (attributes, attribute_rows) = pg16_dump_attrdef_metadata_relation(&catalog)?;
        let projection = ["tableoid", "oid", "adrelid", "adnum", "adsrc"]
            .into_iter()
            .map(|column| ("a", column, column))
            .collect();
        let plan = join_plan(
            &[(&source, "src"), (&attributes, "a")],
            vec![join_step("src", "tbloid", "a", "adrelid", false)],
            projection,
            vec![("a", "adrelid"), ("a", "adnum")],
        );
        self.execute_pg_dump_gpu_join(
            &plan,
            vec![source, attributes],
            vec![source_rows, attribute_rows],
            vec![None, None],
            boundary,
        )
    }

    fn execute_pg16_dump_subscription_count(&self) -> Result<RelationalSelectResult, ExecuteError> {
        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        let database_oid = catalog
            .relational_databases
            .get("postgres")
            .map_or(5, |database| database.oid);
        let table = catalog_relation_table(
            "pg_catalog",
            "pg_subscription",
            &[("subdbid", SqlType::Int4)],
        );
        let rows = catalog
            .relational_subscriptions
            .values()
            .map(|_| vec![SqlValue::Int4(database_oid as i32)])
            .collect();
        self.execute_pg_dump_gpu_int4_equal_count(table, rows, "subdbid", database_oid as i32)
    }

    pub(super) fn execute_pg_dump_transient_relation(
        &self,
        table: RelationalTable,
        rows: Vec<Vec<SqlValue>>,
        order_by: &[&str],
        boundary: Index,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        self.execute_pg_dump_gpu_select(
            table,
            rows,
            SelectProjection::All,
            None,
            order_by,
            boundary,
        )
    }
}

pub(super) fn pg16_dump_oid_array_program_is_exact(
    canonical: &str,
    prefix: &str,
    suffix: &str,
) -> bool {
    pg16_dump_oid_array_program_values(canonical, prefix, suffix).is_some()
}

pub(super) fn pg16_dump_oid_array_program_values(
    canonical: &str,
    prefix: &str,
    suffix: &str,
) -> Option<Vec<u32>> {
    let body = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    if body.is_empty() {
        return Some(Vec::new());
    }
    body.split(',')
        .map(|raw| {
            if raw.is_empty() {
                return None;
            }
            raw.parse::<u32>().ok()
        })
        .collect()
}

pub(crate) fn is_pg16_dump_sequence_state_program(sql: &str) -> bool {
    sequences::pg16_dump_sequence_state_name(sql).is_some()
}

const PG16_DUMP_CLASS_METADATA_PROGRAM: &str = "select c.tableoid, c.oid, c.relname, c.relnamespace, c.relkind, c.reltype, c.relowner, c.relchecks, c.relhasindex, c.relhasrules, c.relpages, c.relhastriggers, c.relpersistence, c.reloftype, c.relacl, acldefault(case when c.relkind = 'S' then 's'::\"char\" else 'r'::\"char\" end, c.relowner) as acldefault, case when c.relkind = 'f' then (select ftserver from pg_catalog.pg_foreign_table where ftrelid = c.oid) else 0 end as foreignserver, c.relfrozenxid, tc.relfrozenxid as tfrozenxid, tc.oid as toid, tc.relpages as toastpages, tc.reloptions as toast_reloptions, d.refobjid as owning_tab, d.refobjsubid as owning_col, tsp.spcname as reltablespace, false as relhasoids, c.relispopulated, c.relreplident, c.relrowsecurity, c.relforcerowsecurity, c.relminmxid, tc.relminmxid as tminmxid, array_remove(array_remove(c.reloptions,'check_option=local'),'check_option=cascaded') as reloptions, case when 'check_option=local' = any (c.reloptions) then 'LOCAL'::text when 'check_option=cascaded' = any (c.reloptions) then 'CASCADED'::text else null end as checkoption, am.amname, (d.deptype = 'i') is true as is_identity_sequence, c.relispartition as ispartition from pg_class c left join pg_depend d on (c.relkind = 'S' and d.classid = 'pg_class'::regclass and d.objid = c.oid and d.objsubid = 0 and d.refclassid = 'pg_class'::regclass and d.deptype in ('a', 'i')) left join pg_tablespace tsp on (tsp.oid = c.reltablespace) left join pg_am am on (c.relam = am.oid) left join pg_class tc on (c.reltoastrelid = tc.oid and tc.relkind = 't' and c.relkind <> 'p') where c.relkind in ('r', 'S', 'v', 'c', 'm', 'f', 'p') order by c.oid";

fn is_pg16_dump_class_metadata_program(canonical: &str) -> bool {
    canonical == PG16_DUMP_CLASS_METADATA_PROGRAM
}

fn is_pg16_dump_function_metadata_program(canonical: &str) -> bool {
    canonical
        == "select p.tableoid, p.oid, p.proname, p.prolang, p.pronargs, p.proargtypes, p.prorettype, p.proacl, acldefault('f', p.proowner) as acldefault, p.pronamespace, p.proowner from pg_proc p left join pg_init_privs pip on (p.oid = pip.objoid and pip.classoid = 'pg_proc'::regclass and pip.objsubid = 0) where p.prokind <> 'a' and not exists (select 1 from pg_depend where classid = 'pg_proc'::regclass and objid = p.oid and deptype = 'i') and ( pronamespace != (select oid from pg_namespace where nspname = 'pg_catalog') or exists (select 1 from pg_cast where pg_cast.oid > 16383 and p.oid = pg_cast.castfunc) or exists (select 1 from pg_transform where pg_transform.oid > 16383 and (p.oid = pg_transform.trffromsql or p.oid = pg_transform.trftosql)) or p.proacl is distinct from pip.initprivs)"
}

fn is_pg16_dump_type_metadata_program(canonical: &str) -> bool {
    canonical
        == "select tableoid, oid, typname, typnamespace, typacl, acldefault('T', typowner) as \
            acldefault, typowner, typelem, typrelid, case when typrelid = 0 then ' '::\"char\" \
            else (select relkind from pg_class where oid = typrelid) end as typrelkind, typtype, \
            typisdefined, typname[0] = '_' and typelem != 0 and (select typarray from pg_type te \
            where oid = pg_type.typelem) = oid as isarray from pg_type"
}

fn is_pg16_dump_language_metadata_program(canonical: &str) -> bool {
    canonical
        == "select tableoid, oid, lanname, lanpltrusted, lanplcallfoid, laninline, lanvalidator, \
            lanacl, acldefault('l', lanowner) as acldefault, lanowner from pg_language where \
            lanispl order by oid"
}

fn is_pg16_dump_default_acl_program(canonical: &str) -> bool {
    canonical
        == "select oid, tableoid, defaclrole, defaclnamespace, defaclobjtype, defaclacl, case when defaclnamespace = 0 then acldefault(case when defaclobjtype = 'S' then 's'::\"char\" else defaclobjtype end, defaclrole) else '{}' end as acldefault from pg_default_acl"
}

fn pg16_dump_default_acl_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_default_acl",
        &[
            ("oid", SqlType::Int4),
            ("tableoid", SqlType::Int4),
            ("defaclrole", SqlType::Int4),
            ("defaclnamespace", SqlType::Int4),
            ("defaclobjtype", SqlType::Text),
            ("defaclacl", SqlType::Text),
            ("acldefault", SqlType::Text),
        ],
    );
    let acl = relation_default_acl_array(&catalog.relational_default_table_acl);
    let rows = if matches!(acl, SqlValue::Null) {
        Vec::new()
    } else {
        vec![vec![
            SqlValue::Int4(82_600),
            SqlValue::Int4(826),
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Text("r".to_string()),
            acl,
            SqlValue::Text("{}".to_string()),
        ]]
    };
    (table, rows)
}

fn is_pg16_dump_subscription_count_program(canonical: &str) -> bool {
    canonical
        == "select count(*) from pg_subscription where subdbid = (select oid from pg_database where datname = current_database())"
}

fn pg16_dump_attribute_program_oids(canonical: &str) -> Option<Vec<u32>> {
    let prefix = "select a.attrelid, a.attnum, a.attname, a.attstattarget, a.attstorage, t.typstorage, a.attnotnull, a.atthasdef, a.attisdropped, a.attlen, a.attalign, a.attislocal, pg_catalog.format_type(t.oid, a.atttypmod) as atttypname, array_to_string(a.attoptions, ', ') as attoptions, case when a.attcollation <> t.typcollation then a.attcollation else 0 end as attcollation, pg_catalog.array_to_string(array(select pg_catalog.quote_ident(option_name) || ' ' || pg_catalog.quote_literal(option_value) from pg_catalog.pg_options_to_table(attfdwoptions) order by option_name), e',\n    ') as attfdwoptions, a.attcompression as attcompression, a.attidentity, case when a.atthasmissing and not a.attisdropped then a.attmissingval else null end as attmissingval, a.attgenerated from unnest('{";
    let suffix = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_attribute a on (src.tbloid = a.attrelid) left join pg_catalog.pg_type t on (a.atttypid = t.oid) where a.attnum > 0::pg_catalog.int2 order by a.attrelid, a.attnum";
    let oids = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    if oids.trim().is_empty() {
        return Some(Vec::new());
    }
    oids.split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

fn pg16_dump_attribute_metadata_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_attribute_metadata",
        &[
            ("attrelid", SqlType::Int4),
            ("attnum", SqlType::Int4),
            ("attname", SqlType::Text),
            ("attstattarget", SqlType::Int4),
            ("attstorage", SqlType::Text),
            ("atttypid", SqlType::Int4),
            ("attnotnull", SqlType::Bool),
            ("atthasdef", SqlType::Bool),
            ("attisdropped", SqlType::Bool),
            ("attlen", SqlType::Int4),
            ("attalign", SqlType::Text),
            ("attislocal", SqlType::Bool),
            ("atttypname", SqlType::Text),
            ("attoptions", SqlType::Text),
            ("attcollation", SqlType::Int4),
            ("attfdwoptions", SqlType::Text),
            ("attcompression", SqlType::Text),
            ("attidentity", SqlType::Text),
            ("attmissingval", SqlType::Text),
            ("attgenerated", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        rows.extend(relation.columns.iter().map(|column| {
            pg16_dump_attribute_metadata_row(relation.oid, column, column.default.is_some())
        }));
    }
    for view in catalog.relational_views.values() {
        let Some(source) = catalog.relational_catalog.get(&view.query.table) else {
            continue;
        };
        let selected = match &view.query.projection {
            SelectProjection::All => source.columns.iter().collect::<Vec<_>>(),
            SelectProjection::Columns(names) => names
                .iter()
                .filter_map(|name| source.columns.iter().find(|column| column.name == *name))
                .collect(),
            _ => Vec::new(),
        };
        rows.extend(
            selected
                .into_iter()
                .enumerate()
                .filter_map(|(index, column)| {
                    let attnum = i16::try_from(index + 1).ok()?;
                    let mut projected = column.clone();
                    projected.attnum = attnum;
                    Some(pg16_dump_attribute_metadata_row(
                        view.oid, &projected, false,
                    ))
                }),
        );
    }
    for view in catalog.relational_materialized_views.values() {
        rows.extend(
            view.columns
                .iter()
                .map(|column| pg16_dump_attribute_metadata_row(view.oid, column, false)),
        );
    }
    (table, rows)
}

fn pg16_dump_attribute_type_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "pg_type",
        &[("oid", SqlType::Int4), ("typstorage", SqlType::Text)],
    );
    let mut rows = gpu_db_sql::SUPPORTED_SQL_TYPES
        .into_iter()
        .map(|ty| {
            vec![
                SqlValue::Int4(ty.postgres_oid() as i32),
                SqlValue::Text(pg_dump_type_storage(ty).to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.extend(catalog.relational_domains.values().map(|domain| {
        vec![
            SqlValue::Int4(domain.oid as i32),
            SqlValue::Text(pg_dump_type_storage(domain.base_type).to_string()),
        ]
    }));
    (table, rows)
}

fn pg16_dump_attribute_metadata_row(
    relation_oid: u32,
    column: &RelationalColumn,
    has_default: bool,
) -> Vec<SqlValue> {
    vec![
        SqlValue::Int4(relation_oid as i32),
        SqlValue::Int4(column.attnum as i32),
        SqlValue::Text(column.name.clone()),
        SqlValue::Int4(-1),
        SqlValue::Text(pg_dump_type_storage(column.ty).to_string()),
        SqlValue::Int4(column.type_oid as i32),
        SqlValue::Bool(false),
        SqlValue::Bool(has_default),
        SqlValue::Bool(false),
        SqlValue::Int4(column.ty.type_size() as i32),
        SqlValue::Text(pg_dump_type_alignment(column.ty).to_string()),
        SqlValue::Bool(true),
        SqlValue::Text(
            column
                .domain
                .clone()
                .unwrap_or_else(|| pg_dump_type_display(column.ty).to_string()),
        ),
        SqlValue::Null,
        SqlValue::Int4(0),
        SqlValue::Null,
        SqlValue::Text(String::new()),
        SqlValue::Text(String::new()),
        SqlValue::Null,
        SqlValue::Text(String::new()),
    ]
}

fn pg_dump_type_storage(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Numeric { .. } => "m",
        SqlType::Text => "x",
        _ => "p",
    }
}

fn pg_dump_type_alignment(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "s",
        SqlType::Int4 | SqlType::Numeric { .. } | SqlType::Text | SqlType::Date => "i",
        SqlType::Int8 | SqlType::Timestamp => "d",
        SqlType::Bool | SqlType::Uuid => "c",
    }
}

fn pg_dump_type_display(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "smallint",
        SqlType::Int4 => "integer",
        SqlType::Int8 => "bigint",
        SqlType::Numeric { .. } => "numeric",
        SqlType::Bool => "boolean",
        SqlType::Text => "text",
        SqlType::Date => "date",
        SqlType::Timestamp => "timestamp without time zone",
        SqlType::Uuid => "uuid",
    }
}

fn pg16_dump_attrdef_program_oids(canonical: &str) -> Option<Vec<u32>> {
    let prefix = "select a.tableoid, a.oid, adrelid, adnum, pg_catalog.pg_get_expr(adbin, adrelid) as adsrc from unnest('{";
    let suffix = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_attrdef a on (src.tbloid = a.adrelid) order by a.adrelid, a.adnum";
    let body = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let oids = body
        .split(',')
        .map(|raw| raw.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!oids.is_empty()).then_some(oids)
}

fn pg16_dump_attrdef_metadata_relation(
    catalog: &CatalogSnapshot,
) -> Result<(RelationalTable, Vec<Vec<SqlValue>>), ExecuteError> {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_attrdef_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("adrelid", SqlType::Int4),
            ("adnum", SqlType::Int4),
            ("adsrc", SqlType::Text),
        ],
    );
    let mut rows = Vec::new();
    for relation in catalog.relational_catalog.values() {
        for column in &relation.columns {
            let Some(default) = &column.default else {
                continue;
            };
            let expression = match default {
                ColumnDefault::Literal(value) => pg16_dump_default_expression(value)?,
                ColumnDefault::SequenceNextVal { sequence, .. } => {
                    format!("nextval('{}'::regclass)", sequence.replace('\'', "''"))
                }
            };
            rows.push(vec![
                SqlValue::Int4(2604),
                SqlValue::Int4((30_000_u32 + relation.oid + column.attnum as u32) as i32),
                SqlValue::Int4(relation.oid as i32),
                SqlValue::Int4(column.attnum as i32),
                SqlValue::Text(expression),
            ]);
        }
    }
    Ok((table, rows))
}

fn pg16_dump_default_expression(value: &SqlValue) -> Result<String, ExecuteError> {
    pg16_column_default_expression(value).map_err(ExecuteError::Engine)
}

fn pg16_dump_language_metadata_relation() -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_language_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("lanname", SqlType::Text),
            ("lanpltrusted", SqlType::Bool),
            ("lanplcallfoid", SqlType::Int4),
            ("laninline", SqlType::Int4),
            ("lanvalidator", SqlType::Int4),
            ("lanacl", SqlType::Text),
            ("acldefault", SqlType::Text),
            ("lanowner", SqlType::Int4),
            ("__lanispl", SqlType::Bool),
        ],
    );
    let rows = vec![
        vec![
            SqlValue::Int4(2612),
            SqlValue::Int4(14),
            SqlValue::Text("sql".to_string()),
            SqlValue::Bool(true),
            SqlValue::Int4(0),
            SqlValue::Int4(0),
            SqlValue::Int4(0),
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Bool(false),
        ],
        vec![
            SqlValue::Int4(2612),
            SqlValue::Int4(13_501),
            SqlValue::Text("plpgsql".to_string()),
            SqlValue::Bool(true),
            SqlValue::Int4(13_502),
            SqlValue::Int4(13_503),
            SqlValue::Int4(13_504),
            SqlValue::Null,
            SqlValue::Text("postgres=U/postgres".to_string()),
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Bool(true),
        ],
    ];
    (table, rows)
}

fn pg16_dump_authoritatively_empty_relation(canonical: &str) -> Option<RelationalTable> {
    let relation = |name: &str, columns: &[(&str, SqlType)]| {
        catalog_relation_table("pg_catalog", name, columns)
    };
    if canonical
        == "select conrelid, confrelid from pg_constraint join pg_depend on (objid = confrelid) where contype = 'f' and refclassid = 'pg_extension'::regclass and classid = 'pg_class'::regclass"
    {
        return Some(relation(
            "__pg16_dump_extension_foreign_keys",
            &[("conrelid", SqlType::Int4), ("confrelid", SqlType::Int4)],
        ));
    }
    if canonical
        .strip_prefix(
            "select unnest(setconfig) from pg_db_role_setting where setdatabase = 0 and setrole = (select oid from pg_roles where rolname = '",
        )
        .and_then(|role| role.strip_suffix("')"))
        .is_some_and(pg16_dump_plain_role_literal)
    {
        return Some(relation(
            "__pg16_dump_role_settings",
            &[("unnest", SqlType::Text)],
        ));
    }
    if canonical
        == "select ur.rolname as role, um.rolname as member, ug.rolname as grantor, a.roleid as roleid, a.member as memberid, a.grantor as grantorid, a.admin_option, a.inherit_option, a.set_option from pg_auth_members a left join pg_roles ur on ur.oid = a.roleid left join pg_roles um on um.oid = a.member left join pg_roles ug on ug.oid = a.grantor where not (ur.rolname ~ '^pg_' and um.rolname ~ '^pg_')order by 1,2,3"
    {
        return Some(relation(
            "__pg16_dump_role_memberships",
            &[
                ("role", SqlType::Text),
                ("member", SqlType::Text),
                ("grantor", SqlType::Text),
                ("roleid", SqlType::Int4),
                ("memberid", SqlType::Int4),
                ("grantorid", SqlType::Int4),
                ("admin_option", SqlType::Bool),
                ("inherit_option", SqlType::Bool),
                ("set_option", SqlType::Bool),
            ],
        ));
    }
    if canonical
        == "select parname, pg_catalog.pg_get_userbyid(10) as parowner, paracl, pg_catalog.acldefault('p', 10) as acldefault from pg_catalog.pg_parameter_acl order by 1"
    {
        return Some(relation(
            "__pg16_dump_parameter_acls",
            &[
                ("parname", SqlType::Text),
                ("parowner", SqlType::Text),
                ("paracl", SqlType::Text),
                ("acldefault", SqlType::Text),
            ],
        ));
    }
    if canonical
        .strip_prefix(
            "select unnest(setconfig) from pg_db_role_setting where setrole = 0 and setdatabase = '",
        )
        .and_then(|oid| oid.strip_suffix("'::oid"))
        .is_some_and(|oid| oid.parse::<u32>().is_ok())
    {
        return Some(relation(
            "__pg16_dump_database_role_settings",
            &[("unnest", SqlType::Text)],
        ));
    }
    if canonical
        .strip_prefix(
            "select rolname, unnest(setconfig) from pg_db_role_setting s, pg_roles r where setrole = r.oid and setdatabase = '",
        )
        .and_then(|oid| oid.strip_suffix("'::oid"))
        .is_some_and(|oid| oid.parse::<u32>().is_ok())
    {
        return Some(relation(
            "__pg16_dump_database_role_user_settings",
            &[("rolname", SqlType::Text), ("unnest", SqlType::Text)],
        ));
    }
    if canonical
        == "select p.tableoid, p.oid, p.proname as aggname, p.pronamespace as aggnamespace, p.pronargs, p.proargtypes, p.proowner, p.proacl as aggacl, acldefault('f', p.proowner) as acldefault from pg_proc p left join pg_init_privs pip on (p.oid = pip.objoid and pip.classoid = 'pg_proc'::regclass and pip.objsubid = 0) where p.prokind = 'a' and (p.pronamespace != (select oid from pg_namespace where nspname = 'pg_catalog') or p.proacl is distinct from pip.initprivs)"
    {
        return Some(relation(
            "__pg16_dump_aggregates",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("aggname", SqlType::Text),
                ("aggnamespace", SqlType::Int4),
                ("pronargs", SqlType::Int4),
                ("proargtypes", SqlType::Text),
                ("proowner", SqlType::Int4),
                ("aggacl", SqlType::Text),
                ("acldefault", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, amname, amtype, amhandler::pg_catalog.regproc as amhandler from pg_am"
    {
        return Some(relation(
            "__pg16_dump_access_methods",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("amname", SqlType::Text),
                ("amtype", SqlType::Text),
                ("amhandler", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, oprname, oprnamespace, oprowner, oprkind, oprleft, oprright, oprcode::oid as oprcode from pg_operator"
    {
        return Some(relation(
            "__pg16_dump_operators",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("oprname", SqlType::Text),
                ("oprnamespace", SqlType::Int4),
                ("oprowner", SqlType::Int4),
                ("oprkind", SqlType::Text),
                ("oprleft", SqlType::Int4),
                ("oprright", SqlType::Int4),
                ("oprcode", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, collname, collnamespace, collowner, collencoding from pg_collation"
    {
        return Some(relation(
            "__pg16_dump_collations",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("collname", SqlType::Text),
                ("collnamespace", SqlType::Int4),
                ("collowner", SqlType::Int4),
                ("collencoding", SqlType::Int4),
            ],
        ));
    }
    if canonical == "select tableoid, oid, conname, connamespace, conowner from pg_conversion" {
        return Some(relation(
            "__pg16_dump_conversions",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("conname", SqlType::Text),
                ("connamespace", SqlType::Int4),
                ("conowner", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, castsource, casttarget, castfunc, castcontext, castmethod from pg_cast c where not exists ( select 1 from pg_range r where c.castsource = r.rngtypid and c.casttarget = r.rngmultitypid ) order by 3,4"
    {
        return Some(relation(
            "__pg16_dump_casts",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("castsource", SqlType::Int4),
                ("casttarget", SqlType::Int4),
                ("castfunc", SqlType::Int4),
                ("castcontext", SqlType::Text),
                ("castmethod", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, opcmethod, opcname, opcnamespace, opcowner from pg_opclass"
    {
        return Some(relation(
            "__pg16_dump_opclasses",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("opcmethod", SqlType::Int4),
                ("opcname", SqlType::Text),
                ("opcnamespace", SqlType::Int4),
                ("opcowner", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, opfmethod, opfname, opfnamespace, opfowner from pg_opfamily"
    {
        return Some(relation(
            "__pg16_dump_opfamilies",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("opfmethod", SqlType::Int4),
                ("opfname", SqlType::Text),
                ("opfnamespace", SqlType::Int4),
                ("opfowner", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, prsname, prsnamespace, prsstart::oid, prstoken::oid, prsend::oid, prsheadline::oid, prslextype::oid from pg_ts_parser"
    {
        return Some(relation(
            "__pg16_dump_ts_parsers",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("prsname", SqlType::Text),
                ("prsnamespace", SqlType::Int4),
                ("prsstart", SqlType::Int4),
                ("prstoken", SqlType::Int4),
                ("prsend", SqlType::Int4),
                ("prsheadline", SqlType::Int4),
                ("prslextype", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, tmplname, tmplnamespace, tmplinit::oid, tmpllexize::oid from pg_ts_template"
    {
        return Some(relation(
            "__pg16_dump_ts_templates",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("tmplname", SqlType::Text),
                ("tmplnamespace", SqlType::Int4),
                ("tmplinit", SqlType::Int4),
                ("tmpllexize", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, dictname, dictnamespace, dictowner, dicttemplate, dictinitoption from pg_ts_dict"
    {
        return Some(relation(
            "__pg16_dump_ts_dictionaries",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("dictname", SqlType::Text),
                ("dictnamespace", SqlType::Int4),
                ("dictowner", SqlType::Int4),
                ("dicttemplate", SqlType::Int4),
                ("dictinitoption", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, cfgname, cfgnamespace, cfgowner, cfgparser from pg_ts_config"
    {
        return Some(relation(
            "__pg16_dump_ts_configs",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("cfgname", SqlType::Text),
                ("cfgnamespace", SqlType::Int4),
                ("cfgowner", SqlType::Int4),
                ("cfgparser", SqlType::Int4),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, fdwname, fdwowner, fdwhandler::pg_catalog.regproc, fdwvalidator::pg_catalog.regproc, fdwacl, acldefault('F', fdwowner) as acldefault, array_to_string(array(select quote_ident(option_name) || ' ' || quote_literal(option_value) from pg_options_to_table(fdwoptions) order by option_name), e',\n    ') as fdwoptions from pg_foreign_data_wrapper"
    {
        return Some(relation(
            "__pg16_dump_fdws",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("fdwname", SqlType::Text),
                ("fdwowner", SqlType::Int4),
                ("fdwhandler", SqlType::Text),
                ("fdwvalidator", SqlType::Text),
                ("fdwacl", SqlType::Text),
                ("acldefault", SqlType::Text),
                ("fdwoptions", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, srvname, srvowner, srvfdw, srvtype, srvversion, srvacl, acldefault('S', srvowner) as acldefault, array_to_string(array(select quote_ident(option_name) || ' ' || quote_literal(option_value) from pg_options_to_table(srvoptions) order by option_name), e',\n    ') as srvoptions from pg_foreign_server"
    {
        return Some(relation(
            "__pg16_dump_foreign_servers",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("srvname", SqlType::Text),
                ("srvowner", SqlType::Int4),
                ("srvfdw", SqlType::Int4),
                ("srvtype", SqlType::Text),
                ("srvversion", SqlType::Text),
                ("srvacl", SqlType::Text),
                ("acldefault", SqlType::Text),
                ("srvoptions", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, trftype, trflang, trffromsql::oid, trftosql::oid from pg_transform order by 3,4"
    {
        return Some(relation(
            "__pg16_dump_transforms",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("trftype", SqlType::Int4),
                ("trflang", SqlType::Int4),
                ("trffromsql", SqlType::Int4),
                ("trftosql", SqlType::Int4),
            ],
        ));
    }
    if canonical == "select inhrelid, inhparent from pg_inherits" {
        return Some(relation(
            "__pg16_dump_inherits",
            &[("inhrelid", SqlType::Int4), ("inhparent", SqlType::Int4)],
        ));
    }
    if canonical
        == "select partrelid from pg_partitioned_table where (select c.oid from pg_opclass c join pg_am a on c.opcmethod = a.oid where opcname = 'enum_ops' and opcnamespace = 'pg_catalog'::regnamespace and amname = 'hash') = any(partclass)"
    {
        return Some(relation(
            "__pg16_dump_partitioned_tables",
            &[("partrelid", SqlType::Int4)],
        ));
    }
    if canonical
        == "select tableoid, oid, stxname, stxnamespace, stxowner, stxrelid, stxstattarget from pg_catalog.pg_statistic_ext"
    {
        return Some(relation(
            "__pg16_dump_statistics",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("stxname", SqlType::Text),
                ("stxnamespace", SqlType::Int4),
                ("stxowner", SqlType::Int4),
                ("stxrelid", SqlType::Int4),
                ("stxstattarget", SqlType::Int4),
            ],
        ));
    }
    if pg16_dump_oid_array_program_is_exact(
        canonical,
        "select t.tgrelid, t.tgname, t.tgfoid::pg_catalog.regproc as tgfname, pg_catalog.pg_get_triggerdef(t.oid, false) as tgdef, t.tgenabled, t.tableoid, t.oid, t.tgparentid <> 0 as tgispartition from unnest('{",
        "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_trigger t on (src.tbloid = t.tgrelid) left join pg_catalog.pg_trigger u on (u.oid = t.tgparentid) where ((not t.tgisinternal and t.tgparentid = 0) or t.tgenabled != u.tgenabled) order by t.tgrelid, t.tgname",
    ) {
        return Some(relation(
            "__pg16_dump_triggers",
            &[
                ("tgrelid", SqlType::Int4),
                ("tgname", SqlType::Text),
                ("tgfname", SqlType::Text),
                ("tgdef", SqlType::Text),
                ("tgenabled", SqlType::Text),
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("tgispartition", SqlType::Bool),
            ],
        ));
    }
    if canonical
        == "select tableoid, oid, rulename, ev_class as ruletable, ev_type, is_instead, ev_enabled from pg_rewrite order by oid"
    {
        return Some(relation(
            "__pg16_dump_rules",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("rulename", SqlType::Text),
                ("ruletable", SqlType::Int4),
                ("ev_type", SqlType::Text),
                ("is_instead", SqlType::Bool),
                ("ev_enabled", SqlType::Text),
            ],
        ));
    }
    if pg16_dump_oid_array_program_is_exact(
        canonical,
        "select pol.oid, pol.tableoid, pol.polrelid, pol.polname, pol.polcmd, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select pg_catalog.quote_ident(rolname) from pg_catalog.pg_roles where oid = any(pol.polroles)), ', ') end as polroles, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid) as polqual, pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid) as polwithcheck from unnest('{",
        "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_policy pol on (src.tbloid = pol.polrelid)",
    ) {
        return Some(relation(
            "__pg16_dump_policies",
            &[
                ("oid", SqlType::Int4),
                ("tableoid", SqlType::Int4),
                ("polrelid", SqlType::Int4),
                ("polname", SqlType::Text),
                ("polcmd", SqlType::Text),
                ("polpermissive", SqlType::Bool),
                ("polroles", SqlType::Text),
                ("polqual", SqlType::Text),
                ("polwithcheck", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select e.tableoid, e.oid, evtname, evtenabled, evtevent, evtowner, array_to_string(array(select quote_literal(x) from unnest(evttags) as t(x)), ', ') as evttags, e.evtfoid::regproc as evtfname from pg_event_trigger e order by e.oid"
    {
        return Some(relation(
            "__pg16_dump_event_triggers",
            &[
                ("tableoid", SqlType::Int4),
                ("oid", SqlType::Int4),
                ("evtname", SqlType::Text),
                ("evtenabled", SqlType::Text),
                ("evtevent", SqlType::Text),
                ("evtowner", SqlType::Int4),
                ("evttags", SqlType::Text),
                ("evtfname", SqlType::Text),
            ],
        ));
    }
    if canonical == "select distinct attrelid from pg_attribute where attacl is not null" {
        return Some(relation(
            "__pg16_dump_attribute_acls",
            &[("attrelid", SqlType::Int4)],
        ));
    }
    if canonical == "select objoid, classoid, objsubid, privtype, initprivs from pg_init_privs" {
        return Some(relation(
            "__pg16_dump_initial_privileges",
            &[
                ("objoid", SqlType::Int4),
                ("classoid", SqlType::Int4),
                ("objsubid", SqlType::Int4),
                ("privtype", SqlType::Text),
                ("initprivs", SqlType::Text),
            ],
        ));
    }
    if canonical
        == "select label, provider, classoid, objoid, objsubid from pg_catalog.pg_seclabels order by classoid, objoid, objsubid"
    {
        return Some(relation(
            "__pg16_dump_security_labels",
            &[
                ("label", SqlType::Text),
                ("provider", SqlType::Text),
                ("classoid", SqlType::Int4),
                ("objoid", SqlType::Int4),
                ("objsubid", SqlType::Int4),
            ],
        ));
    }
    if pg16_dump_shared_security_label_program_is_exact(canonical) {
        return Some(relation(
            "__pg16_dump_shared_security_labels",
            &[("provider", SqlType::Text), ("label", SqlType::Text)],
        ));
    }
    None
}

fn pg16_dump_plain_role_literal(role: &str) -> bool {
    !role.is_empty()
        && role
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn pg16_dump_shared_security_label_program_is_exact(canonical: &str) -> bool {
    let Some(rest) = canonical.strip_prefix(
        "select provider, label from pg_catalog.pg_shseclabel where classoid = 'pg_catalog.",
    ) else {
        return false;
    };
    let Some((class, oid)) = rest.split_once("'::pg_catalog.regclass and objoid = '") else {
        return false;
    };
    matches!(class, "pg_authid" | "pg_database" | "pg_tablespace")
        && oid
            .strip_suffix('\'')
            .is_some_and(|oid| oid.parse::<u32>().is_ok())
}

fn pg16_dump_type_metadata_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_type_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("typname", SqlType::Text),
            ("typnamespace", SqlType::Int4),
            ("typacl", SqlType::Text),
            ("acldefault", SqlType::Text),
            ("typowner", SqlType::Int4),
            ("typelem", SqlType::Int4),
            ("typrelid", SqlType::Int4),
            ("typrelkind", SqlType::Text),
            ("typtype", SqlType::Text),
            ("typisdefined", SqlType::Bool),
            ("isarray", SqlType::Bool),
        ],
    );
    let mut rows = gpu_db_sql::SUPPORTED_SQL_TYPES
        .into_iter()
        .map(|ty| {
            vec![
                SqlValue::Int4(1247),
                SqlValue::Int4(ty.postgres_oid() as i32),
                SqlValue::Text(ty.catalog_name().to_string()),
                SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
                SqlValue::Null,
                SqlValue::Null,
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Int4(0),
                SqlValue::Int4(0),
                SqlValue::Text(" ".to_string()),
                SqlValue::Text("b".to_string()),
                SqlValue::Bool(true),
                SqlValue::Bool(false),
            ]
        })
        .collect::<Vec<_>>();
    rows.extend(catalog.relational_domains.values().map(|domain| {
        vec![
            SqlValue::Int4(1247),
            SqlValue::Int4(domain.oid as i32),
            SqlValue::Text(domain.name.clone()),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
            SqlValue::Null,
            SqlValue::Null,
            SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
            SqlValue::Int4(0),
            SqlValue::Int4(0),
            SqlValue::Text(" ".to_string()),
            SqlValue::Text("d".to_string()),
            SqlValue::Bool(true),
            SqlValue::Bool(false),
        ]
    }));
    (table, rows)
}

fn pg16_dump_function_metadata_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_function_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("proname", SqlType::Text),
            ("prolang", SqlType::Int4),
            ("pronargs", SqlType::Int4),
            ("proargtypes", SqlType::Text),
            ("prorettype", SqlType::Int4),
            ("proacl", SqlType::Text),
            ("acldefault", SqlType::Text),
            ("pronamespace", SqlType::Int4),
            ("proowner", SqlType::Int4),
            ("__prokind", SqlType::Text),
        ],
    );
    let rows = catalog
        .relational_functions
        .values()
        .map(|function| {
            vec![
                SqlValue::Int4(1255),
                SqlValue::Int4(function.oid as i32),
                SqlValue::Text(function.name.clone()),
                SqlValue::Int4(14),
                SqlValue::Int4(0),
                SqlValue::Text(String::new()),
                SqlValue::Int4(function.return_type.postgres_oid() as i32),
                function_acl_array(&function.acl),
                SqlValue::Text("{=X/postgres,postgres=X/postgres}".to_string()),
                SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
                SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
                SqlValue::Text("f".to_string()),
            ]
        })
        .collect();
    (table, rows)
}

fn pg16_dump_class_metadata_relation(
    catalog: &CatalogSnapshot,
) -> (RelationalTable, Vec<Vec<SqlValue>>) {
    let table = catalog_relation_table(
        "pg_catalog",
        "__pg16_dump_class_metadata",
        &[
            ("tableoid", SqlType::Int4),
            ("oid", SqlType::Int4),
            ("relname", SqlType::Text),
            ("relnamespace", SqlType::Int4),
            ("relkind", SqlType::Text),
            ("reltype", SqlType::Int4),
            ("relowner", SqlType::Int4),
            ("relchecks", SqlType::Int4),
            ("relhasindex", SqlType::Bool),
            ("relhasrules", SqlType::Bool),
            ("relpages", SqlType::Int4),
            ("relhastriggers", SqlType::Bool),
            ("relpersistence", SqlType::Text),
            ("reloftype", SqlType::Int4),
            ("relacl", SqlType::Text),
            ("acldefault", SqlType::Text),
            ("foreignserver", SqlType::Int4),
            ("relfrozenxid", SqlType::Text),
            ("tfrozenxid", SqlType::Text),
            ("toid", SqlType::Int4),
            ("toastpages", SqlType::Int4),
            ("toast_reloptions", SqlType::Text),
            ("owning_tab", SqlType::Int4),
            ("owning_col", SqlType::Int4),
            ("reltablespace", SqlType::Text),
            ("relhasoids", SqlType::Bool),
            ("relispopulated", SqlType::Bool),
            ("relreplident", SqlType::Text),
            ("relrowsecurity", SqlType::Bool),
            ("relforcerowsecurity", SqlType::Bool),
            ("relminmxid", SqlType::Text),
            ("tminmxid", SqlType::Text),
            ("reloptions", SqlType::Text),
            ("checkoption", SqlType::Text),
            ("amname", SqlType::Text),
            ("is_identity_sequence", SqlType::Bool),
            ("ispartition", SqlType::Bool),
            ("__relam", SqlType::Int4),
            ("__reltablespace_oid", SqlType::Int4),
            ("__reltoastrelid", SqlType::Int4),
        ],
    );
    let mut rows = Vec::with_capacity(
        catalog.relational_catalog.len()
            + catalog.relational_views.len()
            + catalog.relational_materialized_views.len()
            + catalog.relational_sequences.len(),
    );
    rows.extend(catalog.relational_catalog.values().map(|relation| {
        pg16_dump_class_metadata_row(DumpClassRow {
            oid: relation.oid,
            name: &relation.name,
            kind: "r",
            checks: relation.check_constraints.len(),
            has_index: !relation.indexes.is_empty(),
            has_rules: false,
            populated: true,
            access_method: Some("heap"),
            acl: &relation.acl,
        })
    }));
    rows.extend(catalog.relational_views.values().map(|relation| {
        pg16_dump_class_metadata_row(DumpClassRow {
            oid: relation.oid,
            name: &relation.name,
            kind: "v",
            checks: 0,
            has_index: false,
            has_rules: true,
            populated: true,
            access_method: None,
            acl: &relation.acl,
        })
    }));
    rows.extend(
        catalog
            .relational_materialized_views
            .values()
            .map(|relation| {
                pg16_dump_class_metadata_row(DumpClassRow {
                    oid: relation.oid,
                    name: &relation.name,
                    kind: "m",
                    checks: 0,
                    has_index: false,
                    has_rules: true,
                    populated: true,
                    access_method: Some("heap"),
                    acl: &relation.acl,
                })
            }),
    );
    rows.extend(catalog.relational_sequences.values().map(|relation| {
        pg16_dump_class_metadata_row(DumpClassRow {
            oid: relation.oid,
            name: &relation.name,
            kind: "S",
            checks: 0,
            has_index: false,
            has_rules: false,
            populated: true,
            access_method: None,
            acl: &relation.acl,
        })
    }));
    let empty_acl = BTreeMap::new();
    rows.extend(catalog_index_entries(catalog).into_iter().map(|entry| {
        pg16_dump_class_metadata_row(DumpClassRow {
            oid: entry.index_oid,
            name: &entry.index.name,
            kind: "i",
            checks: 0,
            has_index: false,
            has_rules: false,
            populated: true,
            access_method: Some("btree"),
            acl: &empty_acl,
        })
    }));
    (table, rows)
}

struct DumpClassRow<'a> {
    oid: u32,
    name: &'a str,
    kind: &'a str,
    checks: usize,
    has_index: bool,
    has_rules: bool,
    populated: bool,
    access_method: Option<&'a str>,
    acl: &'a BTreeMap<String, BTreeSet<TablePrivilege>>,
}

fn pg16_dump_class_metadata_row(metadata: DumpClassRow<'_>) -> Vec<SqlValue> {
    vec![
        SqlValue::Int4(1259),
        SqlValue::Int4(metadata.oid as i32),
        SqlValue::Text(metadata.name.to_string()),
        SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
        SqlValue::Text(metadata.kind.to_string()),
        SqlValue::Int4(0),
        SqlValue::Int4(PG_BOOTSTRAP_OWNER_OID),
        SqlValue::Int4(metadata.checks as i32),
        SqlValue::Bool(metadata.has_index),
        SqlValue::Bool(metadata.has_rules),
        SqlValue::Int4(0),
        SqlValue::Bool(false),
        SqlValue::Text("p".to_string()),
        SqlValue::Int4(0),
        relation_acl_array(metadata.acl, metadata.kind),
        SqlValue::Text(if metadata.kind == "S" {
            "{postgres=rwU/postgres}".to_string()
        } else {
            "{postgres=arwdDxt/postgres}".to_string()
        }),
        SqlValue::Int4(0),
        SqlValue::Text("0".to_string()),
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Bool(false),
        SqlValue::Bool(metadata.populated),
        SqlValue::Text("d".to_string()),
        SqlValue::Bool(false),
        SqlValue::Bool(false),
        SqlValue::Text("0".to_string()),
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        metadata
            .access_method
            .map_or(SqlValue::Null, |value| SqlValue::Text(value.to_string())),
        SqlValue::Bool(false),
        SqlValue::Bool(false),
        SqlValue::Int4(if matches!(metadata.kind, "r" | "m") {
            PG_HEAP_AM_OID
        } else {
            0
        }),
        SqlValue::Int4(0),
        SqlValue::Int4(0),
    ]
}

fn relation_acl_entries(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> Vec<String> {
    acl.iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let mut letters = String::new();
            for (privilege, letter) in [
                (TablePrivilege::Insert, 'a'),
                (TablePrivilege::Select, 'r'),
                (TablePrivilege::Update, 'w'),
                (TablePrivilege::Delete, 'd'),
            ] {
                if privileges.contains(&privilege) {
                    letters.push(letter);
                }
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!("{grantee}={letters}/postgres"))
        })
        .collect()
}

fn acl_entries_value(entries: Vec<String>) -> SqlValue {
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(format!("{{{}}}", entries.join(",")))
    }
}

fn relation_acl_array(
    acl: &BTreeMap<String, BTreeSet<TablePrivilege>>,
    relation_kind: &str,
) -> SqlValue {
    let mut entries = relation_acl_entries(acl);
    if !entries.is_empty() {
        entries.insert(
            0,
            if relation_kind == "S" {
                "postgres=rwU/postgres"
            } else {
                "postgres=arwdDxt/postgres"
            }
            .to_string(),
        );
    }
    acl_entries_value(entries)
}

fn relation_default_acl_array(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> SqlValue {
    acl_entries_value(relation_acl_entries(acl))
}

fn function_acl_array(acl: &BTreeMap<String, BTreeSet<FunctionPrivilege>>) -> SqlValue {
    let entries = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if !privileges.contains(&FunctionPrivilege::Execute) {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!("{grantee}=X/postgres"))
        })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        SqlValue::Null
    } else {
        SqlValue::Text(format!("{{{}}}", entries.join(",")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pg16_relation_acl_uses_the_relation_kind_owner_baseline() {
        let acl = BTreeMap::from([(
            "dump_reader".to_string(),
            BTreeSet::from([TablePrivilege::Select, TablePrivilege::Update]),
        )]);
        assert_eq!(
            relation_acl_array(&acl, "S"),
            SqlValue::Text("{postgres=rwU/postgres,dump_reader=rw/postgres}".to_string())
        );
        assert_eq!(
            relation_acl_array(&acl, "r"),
            SqlValue::Text("{postgres=arwdDxt/postgres,dump_reader=rw/postgres}".to_string())
        );
        assert_eq!(relation_acl_array(&BTreeMap::new(), "S"), SqlValue::Null);
    }

    #[test]
    fn pg16_class_program_recognition_rejects_shape_shortcuts() {
        let canonical = "select c.tableoid, c.oid, c.relname, c.relnamespace, c.relkind, \
            c.reltype, c.relowner, c.relchecks, c.relhasindex, c.relhasrules, c.relpages, \
            c.relhastriggers, c.relpersistence, c.reloftype, c.relacl, acldefault('r', c.relowner) \
            from pg_class c left join pg_depend d on true \
            where c.relkind in ('r', 's', 'v', 'c', 'm', 'f', 'p') order by c.oid";
        assert!(!is_pg16_dump_class_metadata_program(canonical));
        assert!(!is_pg16_dump_class_metadata_program(
            &canonical.replace("order by c.oid", "order by c.relname")
        ));
        assert!(!is_pg16_dump_class_metadata_program(
            &canonical.replace("c.relacl", "null as relacl")
        ));
        assert!(is_pg16_dump_class_metadata_program(
            PG16_DUMP_CLASS_METADATA_PROGRAM
        ));
        assert!(!is_pg16_dump_class_metadata_program(
            &PG16_DUMP_CLASS_METADATA_PROGRAM.replacen("relkind = 'S'", "relkind = 's'", 1)
        ));
        assert!(!is_pg16_dump_class_metadata_program(
            &PG16_DUMP_CLASS_METADATA_PROGRAM.replacen("::\"char\"", "::\"CHAR\"", 1)
        ));
    }

    #[test]
    fn pg16_empty_program_recognition_is_exact_and_literal_bounded() {
        let extension_foreign_keys = "select conrelid, confrelid from pg_constraint join \
            pg_depend on (objid = confrelid) where contype = 'f' and refclassid = \
            'pg_extension'::regclass and classid = 'pg_class'::regclass";
        assert!(pg16_dump_authoritatively_empty_relation(extension_foreign_keys).is_some());
        assert!(pg16_dump_authoritatively_empty_relation(
            &extension_foreign_keys.replace("contype = 'f'", "contype != 'f'")
        )
        .is_none());

        let transform = "select tableoid, oid, trftype, trflang, trffromsql::oid, \
            trftosql::oid from pg_transform order by 3,4";
        assert!(pg16_dump_authoritatively_empty_relation(transform).is_some());
        assert!(pg16_dump_authoritatively_empty_relation(
            &transform.replace("trflang", "trflang + 1")
        )
        .is_none());
        assert!(
            pg16_dump_authoritatively_empty_relation(&format!("{transform} where true")).is_none()
        );

        let shared_label = "select provider, label from pg_catalog.pg_shseclabel where \
            classoid = 'pg_catalog.pg_authid'::pg_catalog.regclass and objoid = '10'";
        assert!(pg16_dump_authoritatively_empty_relation(shared_label).is_some());
        assert!(pg16_dump_authoritatively_empty_relation(
            &shared_label.replace("'10'", "'10 or 1=1'")
        )
        .is_none());
        assert!(pg16_dump_authoritatively_empty_relation(
            &shared_label.replace("pg_authid", "pg_attribute")
        )
        .is_none());

        let role_setting =
            "select unnest(setconfig) from pg_db_role_setting where setdatabase = 0 \
            and setrole = (select oid from pg_roles where rolname = 'dump_reader')";
        assert!(pg16_dump_authoritatively_empty_relation(role_setting).is_some());
        assert!(pg16_dump_authoritatively_empty_relation(
            &role_setting.replace("dump_reader", "dump_reader' or '1'='1")
        )
        .is_none());

        assert!(pg16_dump_oid_array_program_is_exact(
            "prefix{1,2,3}suffix",
            "prefix{",
            "}suffix"
        ));
        assert!(pg16_dump_oid_array_program_is_exact(
            "prefix{}suffix",
            "prefix{",
            "}suffix"
        ));
        assert!(!pg16_dump_oid_array_program_is_exact(
            "prefix{1,2 union select 3}suffix",
            "prefix{",
            "}suffix"
        ));
    }
}
