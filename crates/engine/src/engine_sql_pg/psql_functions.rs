//! PostgreSQL 16 `psql \df[+]` through one bounded terminal GPU catalog operator.
//!
//! The host only encodes complete, snapshot-pinned `pg_proc`/catalog facts.  Namespace, result
//! type, owner, language, comment, ACL display, name filtering, ordering, and final projection
//! all execute in `gpu_db_catalog_function_list`; its final frame is decoded once here.

use super::*;
use gpu_db_execution::{
    DeviceFunctionListDescription, DeviceFunctionListFunction, DeviceFunctionListGrant,
    DeviceFunctionListRequest, DeviceFunctionListRow, DeviceFunctionListTextCandidate,
    DeviceFunctionListVerdict, FUNCTION_LIST_COLUMN_COUNT, MAX_FUNCTION_LIST_TEXT,
};
#[cfg(test)]
use std::sync::Mutex;

const FUNCTION_PRIVILEGE_EXECUTE: u32 = 1;

impl Engine {
    pub(super) fn execute_psql_functions_if_applicable(
        &self,
        sql: &str,
    ) -> Result<Option<RelationalSelectResult>, ExecuteError> {
        let Ok(canonical) = canonicalize_sql_for_exact_match(sql) else {
            return Ok(None);
        };
        let Some(program) = pg16_function_list_program(&canonical) else {
            return Ok(None);
        };

        let boundary = self.read_snapshot_boundary();
        let catalog = self.read_catalog_as_of(boundary);
        self.execute_psql_function_list_gpu(
            &catalog,
            program.name_filter.as_deref(),
            program.verbose,
        )
        .map(Some)
    }

    fn execute_psql_function_list_gpu(
        &self,
        catalog: &CatalogSnapshot,
        name_filter: Option<&str>,
        verbose: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (input, request) = function_list_input(catalog, name_filter, verbose)?;
        self.execute_psql_function_list_input_gpu(input, request, verbose)
    }

    fn execute_psql_function_list_input_gpu(
        &self,
        input: DeviceFunctionListInput,
        request: DeviceFunctionListRequest,
        verbose: bool,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        // This resident anchor supplies the CUDA context and pooled stream.  It is not a source
        // relation for the result: the function-list operator owns the complete final projection.
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__psql_function_list_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) =
            self.build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])?;
        let (verdict, rows) = memory
            .catalog_function_list(
                request,
                &input.functions,
                &input.namespaces,
                &input.types,
                &input.owners,
                &input.languages,
                &input.descriptions,
                &input.grants,
                &input.bytes,
            )
            .map_err(|error| psql_function_device_error(error.to_string()))?;
        #[cfg(test)]
        let verdict = if consume_function_list_sabotage(
            self,
            &input.bytes,
            request.name_filter_offset,
            request.name_filter_len,
            request.verbose,
        )? {
            // The CUDA operator already consumed the complete raw candidate relation.  Model a
            // corrupted terminal verdict rather than substituting a host-side formatter.
            DeviceFunctionListVerdict::InvalidInput
        } else {
            verdict
        };
        match verdict {
            DeviceFunctionListVerdict::Complete => {
                psql_function_list_result(rows, verbose, memory.metadata().gpu_id)
            }
            DeviceFunctionListVerdict::InvalidInput => Err(psql_function_device_error(
                "terminal catalog operator rejected raw function candidates",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn sabotage_next_psql_function_list_verdict(
        &self,
        name_filter: Option<&str>,
        verbose: bool,
    ) {
        *DEVICE_FUNCTION_LIST_SABOTAGE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(DeviceFunctionListSabotage {
            engine_identity: self as *const Self as usize,
            name_filter: name_filter.map(str::to_string),
            verbose,
        });
    }
}

#[cfg(test)]
struct DeviceFunctionListSabotage {
    engine_identity: usize,
    name_filter: Option<String>,
    verbose: bool,
}

#[cfg(test)]
static DEVICE_FUNCTION_LIST_SABOTAGE: Mutex<Option<DeviceFunctionListSabotage>> = Mutex::new(None);
#[cfg(test)]
static DEVICE_FUNCTION_LIST_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn consume_function_list_sabotage(
    engine: &Engine,
    bytes: &[u8],
    offset: u32,
    len: u32,
    verbose: u32,
) -> Result<bool, ExecuteError> {
    let start = usize::try_from(offset)
        .map_err(|_| psql_function_device_error("sabotage request offset is invalid"))?;
    let end = start
        .checked_add(
            usize::try_from(len)
                .map_err(|_| psql_function_device_error("sabotage request length is invalid"))?,
        )
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| psql_function_device_error("sabotage request text is invalid"))?;
    let name_filter = std::str::from_utf8(&bytes[start..end])
        .map_err(|_| psql_function_device_error("sabotage request text is not UTF-8"))?;
    let mut sabotage = DEVICE_FUNCTION_LIST_SABOTAGE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let matches = sabotage.as_ref().is_some_and(|request| {
        request.engine_identity == engine as *const Engine as usize
            && request.name_filter.as_deref().unwrap_or("") == name_filter
            && request.verbose == (verbose != 0)
    });
    if matches {
        *sabotage = None;
    }
    Ok(matches)
}

fn function_list_input(
    catalog: &CatalogSnapshot,
    name_filter: Option<&str>,
    verbose: bool,
) -> Result<(DeviceFunctionListInput, DeviceFunctionListRequest), ExecuteError> {
    let mut input = DeviceFunctionListInput::default();
    let (_functions, function_rows) = synthesize_catalog_relation("pg_catalog.pg_proc", catalog)
        .expect("pg_proc is a modeled GPU catalog relation");
    let (_namespaces, namespace_rows) =
        synthesize_catalog_relation("pg_catalog.pg_namespace", catalog)
            .expect("pg_namespace is a modeled GPU catalog relation");
    let (_owners, owner_rows) = synthesize_catalog_relation("pg_catalog.pg_roles", catalog)
        .expect("pg_roles is a modeled GPU catalog relation");
    let (_languages, language_rows) =
        synthesize_catalog_relation("pg_catalog.pg_language", catalog)
            .expect("pg_language is a modeled GPU catalog relation");
    let (_descriptions, description_rows) =
        synthesize_catalog_relation("pg_catalog.pg_description", catalog)
            .expect("pg_description is a modeled GPU catalog relation");

    input.functions = function_rows
        .iter()
        .map(|row| input.function(row))
        .collect::<Result<Vec<_>, _>>()?;
    input.namespaces = namespace_rows
        .iter()
        .map(|row| input.text_candidate(row, 1, 2, "pg_namespace"))
        .collect::<Result<Vec<_>, _>>()?;
    input.owners = owner_rows
        .iter()
        .map(|row| input.text_candidate(row, 0, 1, "pg_roles"))
        .collect::<Result<Vec<_>, _>>()?;
    input.languages = language_rows
        .iter()
        .map(|row| input.text_candidate(row, 0, 1, "pg_language"))
        .collect::<Result<Vec<_>, _>>()?;
    input.descriptions = description_rows
        .iter()
        .map(|row| input.description(row))
        .collect::<Result<Vec<_>, _>>()?;
    input.types = function_return_type_candidates(catalog)
        .into_iter()
        .map(|(oid, display)| input.text_candidate_from(oid, &display))
        .collect::<Result<Vec<_>, _>>()?;
    // Encode every raw grant from the immutable function catalog.  The terminal operator,
    // rather than this staging loop, applies the EXECUTE predicate and emits PostgreSQL's
    // newline-separated ACL display.
    for function in catalog.relational_functions.values() {
        for (grantee, privileges) in &function.acl {
            for privilege in privileges {
                let (grantee_offset, grantee_len) = input.text(grantee)?;
                input.grants.push(DeviceFunctionListGrant {
                    function_oid: function.oid,
                    privilege: match privilege {
                        FunctionPrivilege::Execute => FUNCTION_PRIVILEGE_EXECUTE,
                    },
                    grantee_offset,
                    grantee_len,
                    grantee_is_public: u32::from(grantee == "public"),
                });
            }
        }
    }
    let (name_filter_offset, name_filter_len) = input.text(name_filter.unwrap_or(""))?;
    let request = DeviceFunctionListRequest {
        name_filter_offset,
        name_filter_len,
        verbose: u32::from(verbose),
    };
    Ok((input, request))
}

#[derive(Clone, Default)]
struct DeviceFunctionListInput {
    bytes: Vec<u8>,
    functions: Vec<DeviceFunctionListFunction>,
    namespaces: Vec<DeviceFunctionListTextCandidate>,
    types: Vec<DeviceFunctionListTextCandidate>,
    owners: Vec<DeviceFunctionListTextCandidate>,
    languages: Vec<DeviceFunctionListTextCandidate>,
    descriptions: Vec<DeviceFunctionListDescription>,
    grants: Vec<DeviceFunctionListGrant>,
}

impl DeviceFunctionListInput {
    fn text(&mut self, value: &str) -> Result<(u32, u32), ExecuteError> {
        let offset = u32::try_from(self.bytes.len()).map_err(|_| {
            psql_function_device_error("catalog function text staging exceeds device range")
        })?;
        let len = u32::try_from(value.len()).map_err(|_| {
            psql_function_device_error("catalog function text staging exceeds device range")
        })?;
        if value.len() > MAX_FUNCTION_LIST_TEXT {
            return Err(psql_function_device_error(
                "catalog function text candidate exceeds terminal device cell",
            ));
        }
        self.bytes.extend_from_slice(value.as_bytes());
        Ok((offset, len))
    }

    fn text_candidate_from(
        &mut self,
        oid: i32,
        value: &str,
    ) -> Result<DeviceFunctionListTextCandidate, ExecuteError> {
        let (text_offset, text_len) = self.text(value)?;
        Ok(DeviceFunctionListTextCandidate {
            oid: nonnegative_u32(oid, "catalog text candidate OID")?,
            text_offset,
            text_len,
        })
    }

    fn text_candidate(
        &mut self,
        row: &[SqlValue],
        oid_index: usize,
        text_index: usize,
        relation: &str,
    ) -> Result<DeviceFunctionListTextCandidate, ExecuteError> {
        self.text_candidate_from(
            sql_int4(row, oid_index, relation)?,
            sql_text(row, text_index, relation)?,
        )
    }

    fn function(&mut self, row: &[SqlValue]) -> Result<DeviceFunctionListFunction, ExecuteError> {
        let name = sql_text(row, 2, "pg_proc")?;
        let (name_offset, name_len) = self.text(name)?;
        Ok(DeviceFunctionListFunction {
            oid: nonnegative_u32(sql_int4(row, 1, "pg_proc")?, "pg_proc oid")?,
            namespace_oid: nonnegative_u32(sql_int4(row, 3, "pg_proc")?, "pg_proc namespace OID")?,
            owner_oid: nonnegative_u32(sql_int4(row, 4, "pg_proc")?, "pg_proc owner OID")?,
            return_type_oid: sql_int4(row, 5, "pg_proc")?,
            language_oid: nonnegative_u32(sql_int4(row, 7, "pg_proc")?, "pg_proc language OID")?,
            name_offset,
            name_len,
            prokind: sql_single_byte(row, 8, "pg_proc")?,
            provolatile: sql_single_byte(row, 9, "pg_proc")?,
            proparallel: sql_single_byte(row, 10, "pg_proc")?,
            prosecdef: u32::from(sql_bool(row, 11, "pg_proc")?),
        })
    }

    fn description(
        &mut self,
        row: &[SqlValue],
    ) -> Result<DeviceFunctionListDescription, ExecuteError> {
        let (text_offset, text_len) = self.text(sql_text(row, 0, "pg_description")?)?;
        Ok(DeviceFunctionListDescription {
            class_oid: sql_int4(row, 1, "pg_description")?,
            object_oid: nonnegative_u32(
                sql_int4(row, 2, "pg_description")?,
                "pg_description object OID",
            )?,
            object_sub_id: sql_int4(row, 3, "pg_description")?,
            text_offset,
            text_len,
        })
    }
}

fn function_return_type_candidates(catalog: &CatalogSnapshot) -> Vec<(i32, String)> {
    let mut rows = vec![
        (16, "boolean".to_string()),
        (20, "bigint".to_string()),
        (21, "smallint".to_string()),
        (23, "integer".to_string()),
        (25, "text".to_string()),
        (1082, "date".to_string()),
        (1114, "timestamp without time zone".to_string()),
        (1700, "numeric".to_string()),
        (2950, "uuid".to_string()),
    ];
    rows.extend(
        catalog
            .relational_domains
            .values()
            .map(|domain| (domain.oid as i32, domain.name.clone())),
    );
    rows
}

fn psql_function_list_result(
    rows: Vec<DeviceFunctionListRow>,
    verbose: bool,
    gpu_id: u16,
) -> Result<RelationalSelectResult, ExecuteError> {
    let names = function_output_columns(verbose);
    let table = catalog_relation_table("pg_catalog", "__psql_function_list", &names);
    let field_count = names.len();
    let rows = rows
        .into_iter()
        .enumerate()
        .map(|(ordinal, row)| {
            if row.ordinal != ordinal as u32 || field_count > FUNCTION_LIST_COLUMN_COUNT {
                return Err(psql_function_device_error(
                    "terminal catalog operator returned an invalid row ordinal",
                ));
            }
            (0..field_count)
                .map(|field| {
                    psql_function_device_cell(&row, field).map_err(|error| {
                        psql_function_device_error(format!(
                            "terminal catalog field {field} is malformed: {error}"
                        ))
                    })
                })
                .collect()
        })
        .collect::<Result<Vec<Vec<SqlValue>>, _>>()?;
    Ok(RelationalSelectResult {
        columns: Arc::new(table.columns),
        rows: rows.into(),
        planned_target: DeviceTarget::Gpu(gpu_id),
        executed_target: DeviceTarget::Gpu(gpu_id),
        fallback_reason: None,
        access_path: Arc::new(RelationalAccessPath::FullTableScan),
    })
}

fn function_output_columns(verbose: bool) -> Vec<(&'static str, SqlType)> {
    let mut columns = vec![
        ("Schema", SqlType::Text),
        ("Name", SqlType::Text),
        ("Result data type", SqlType::Text),
        ("Argument data types", SqlType::Text),
        ("Type", SqlType::Text),
    ];
    if verbose {
        columns.extend([
            ("Volatility", SqlType::Text),
            ("Parallel", SqlType::Text),
            ("Owner", SqlType::Text),
            ("Security", SqlType::Text),
            ("Access privileges", SqlType::Text),
            ("Language", SqlType::Text),
            ("Internal name", SqlType::Text),
            ("Description", SqlType::Text),
        ]);
    }
    columns
}

fn psql_function_device_cell(
    row: &DeviceFunctionListRow,
    field: usize,
) -> Result<SqlValue, ExecuteError> {
    let cell = &row.cells[field];
    let len = cell.len;
    if len == u32::MAX {
        return Ok(SqlValue::Null);
    }
    let len = usize::try_from(len)
        .ok()
        .filter(|len| *len <= MAX_FUNCTION_LIST_TEXT)
        .ok_or_else(|| psql_function_device_error("terminal catalog cell length is invalid"))?;
    std::str::from_utf8(&cell.bytes[..len])
        .map(|value| SqlValue::Text(value.to_string()))
        .map_err(|_| psql_function_device_error("terminal catalog cell is not UTF-8"))
}

fn sql_int4(row: &[SqlValue], index: usize, relation: &str) -> Result<i32, ExecuteError> {
    match row.get(index) {
        Some(SqlValue::Int4(value)) => Ok(*value),
        _ => Err(psql_function_device_error(format!(
            "raw {relation} candidate has no int4 field {index}"
        ))),
    }
}

fn sql_text<'a>(
    row: &'a [SqlValue],
    index: usize,
    relation: &str,
) -> Result<&'a str, ExecuteError> {
    match row.get(index) {
        Some(SqlValue::Text(value)) => Ok(value),
        _ => Err(psql_function_device_error(format!(
            "raw {relation} candidate has no text field {index}"
        ))),
    }
}

fn sql_bool(row: &[SqlValue], index: usize, relation: &str) -> Result<bool, ExecuteError> {
    match row.get(index) {
        Some(SqlValue::Bool(value)) => Ok(*value),
        _ => Err(psql_function_device_error(format!(
            "raw {relation} candidate has no bool field {index}"
        ))),
    }
}

fn sql_single_byte(row: &[SqlValue], index: usize, relation: &str) -> Result<u32, ExecuteError> {
    let value = sql_text(row, index, relation)?;
    match value.as_bytes() {
        [value] => Ok(u32::from(*value)),
        _ => Err(psql_function_device_error(format!(
            "raw {relation} candidate has no one-byte field {index}"
        ))),
    }
}

fn nonnegative_u32(value: i32, field: &str) -> Result<u32, ExecuteError> {
    u32::try_from(value)
        .map_err(|_| psql_function_device_error(format!("{field} is outside device range")))
}

fn psql_function_device_error(detail: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(format!(
        "psql function-list device stage failed closed: {}",
        detail.into()
    )))
}

const PG16_FUNCTION_SELECT: &str = concat!(
    "select n.nspname as \"Schema\", p.proname as \"Name\", ",
    "pg_catalog.pg_get_function_result(p.oid) as \"Result data type\", ",
    "pg_catalog.pg_get_function_arguments(p.oid) as \"Argument data types\", ",
    "case p.prokind when 'a' then 'agg' when 'w' then 'window' when 'p' then 'proc' ",
    "else 'func' end as \"Type\""
);
const PG16_FUNCTION_VERBOSE_SELECT: &str = concat!(
    ", case when p.provolatile = 'i' then 'immutable' when p.provolatile = 's' then 'stable' ",
    "when p.provolatile = 'v' then 'volatile' end as \"Volatility\", ",
    "case when p.proparallel = 'r' then 'restricted' when p.proparallel = 's' then 'safe' ",
    "when p.proparallel = 'u' then 'unsafe' end as \"Parallel\", ",
    "pg_catalog.pg_get_userbyid(p.proowner) as \"Owner\", ",
    "case when prosecdef then 'definer' else 'invoker' end as \"Security\", ",
    "pg_catalog.array_to_string(p.proacl, e'\\n') as \"Access privileges\", ",
    "l.lanname as \"Language\", ",
    "case when l.lanname in ('internal', 'c') then p.prosrc end as \"Internal name\", ",
    "pg_catalog.obj_description(p.oid, 'pg_proc') as \"Description\""
);
const PG16_FUNCTION_FROM: &str = concat!(
    " from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n ",
    "on n.oid = p.pronamespace"
);
const PG16_FUNCTION_LANGUAGE_JOIN: &str =
    " left join pg_catalog.pg_language l on l.oid = p.prolang";
const PG16_FUNCTION_UNFILTERED_WHERE: &str = concat!(
    "pg_catalog.pg_function_is_visible(p.oid) and n.nspname <> 'pg_catalog' ",
    "and n.nspname <> 'information_schema'"
);
const PG16_FUNCTION_ORDER: &str = " order by 1, 2, 4";

#[derive(Debug, PartialEq, Eq)]
struct Pg16FunctionListProgram {
    name_filter: Option<String>,
    verbose: bool,
}

/// Recognize the complete normalized PostgreSQL 16 `psql \df[+]` program.
///
/// The terminal GPU operator implements this exact projection, joins, predicate, ordering, and
/// one-identifier filter grammar.  Near matches must fall through to the general GPU binder so
/// the fixed route never substitutes its own semantics for caller-supplied SQL.
fn pg16_function_list_program(canonical: &str) -> Option<Pg16FunctionListProgram> {
    for (verbose, select_extension, language_join) in [
        (false, "", ""),
        (
            true,
            PG16_FUNCTION_VERBOSE_SELECT,
            PG16_FUNCTION_LANGUAGE_JOIN,
        ),
    ] {
        let prefix = format!(
            "{PG16_FUNCTION_SELECT}{select_extension}{PG16_FUNCTION_FROM}{language_join} where "
        );
        let Some(where_clause) = canonical
            .strip_prefix(&prefix)
            .and_then(|remainder| remainder.strip_suffix(PG16_FUNCTION_ORDER))
        else {
            continue;
        };

        if where_clause == PG16_FUNCTION_UNFILTERED_WHERE {
            return Some(Pg16FunctionListProgram {
                name_filter: None,
                verbose,
            });
        }
        if let Some(name_filter) = pg16_function_name_filter(where_clause) {
            return Some(Pg16FunctionListProgram {
                name_filter: Some(name_filter),
                verbose,
            });
        }
    }
    None
}

/// Extract the exact one-identifier `WHERE` grammar emitted by PostgreSQL 16 `psql` for
/// `\df[+] name`. Broader regex syntax deliberately falls through to the general GPU binder.
fn pg16_function_name_filter(where_clause: &str) -> Option<String> {
    let prefix = "p.proname operator(pg_catalog.~) '^(";
    let name = where_clause.strip_prefix(prefix)?.strip_suffix(
        ")$' collate pg_catalog.default and pg_catalog.pg_function_is_visible(p.oid)",
    )?;
    (!name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'))
    .then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pg16_function_list_sql(verbose: bool, where_clause: &str) -> String {
        format!(
            "{PG16_FUNCTION_SELECT}{}{PG16_FUNCTION_FROM}{} where {where_clause}{PG16_FUNCTION_ORDER}",
            if verbose {
                PG16_FUNCTION_VERBOSE_SELECT
            } else {
                ""
            },
            if verbose {
                PG16_FUNCTION_LANGUAGE_JOIN
            } else {
                ""
            },
        )
    }

    fn pg16_function_name_where(name: &str) -> String {
        format!(
            "p.proname operator(pg_catalog.~) '^({name})$' collate pg_catalog.default \
             and pg_catalog.pg_function_is_visible(p.oid)"
        )
    }

    #[test]
    fn pg16_function_list_matcher_admits_only_exact_supported_programs() {
        let normal = pg16_function_list_sql(false, PG16_FUNCTION_UNFILTERED_WHERE);
        assert_eq!(
            pg16_function_list_program(&normal),
            Some(Pg16FunctionListProgram {
                name_filter: None,
                verbose: false,
            })
        );

        let verbose = pg16_function_list_sql(true, PG16_FUNCTION_UNFILTERED_WHERE);
        assert_eq!(
            pg16_function_list_program(&verbose),
            Some(Pg16FunctionListProgram {
                name_filter: None,
                verbose: true,
            })
        );

        let named = pg16_function_list_sql(true, &pg16_function_name_where("function_alpha"));
        assert_eq!(
            pg16_function_list_program(&named),
            Some(Pg16FunctionListProgram {
                name_filter: Some("function_alpha".to_string()),
                verbose: true,
            })
        );

        let before_order = normal.strip_suffix(PG16_FUNCTION_ORDER).unwrap();
        for near_match in [
            format!("{before_order} and 1=0{PG16_FUNCTION_ORDER}"),
            format!("{normal} limit 1"),
            normal.replacen(
                "left join pg_catalog.pg_namespace n",
                "join pg_catalog.pg_namespace n",
                1,
            ),
            normal.replacen("end as \"Type\"", "end as \"Type\", p.oid", 1),
            pg16_function_list_sql(
                false,
                &format!("{} and 1=0", pg16_function_name_where("function_alpha")),
            ),
        ] {
            assert_eq!(
                pg16_function_list_program(&near_match),
                None,
                "near-match must not enter the fixed function-list route: {near_match}"
            );
        }
    }

    #[test]
    fn function_list_near_match_never_enters_terminal_route() {
        let engine = Engine::new_local_test_engine();
        let normal = pg16_function_list_sql(false, PG16_FUNCTION_UNFILTERED_WHERE);
        let before_order = normal.strip_suffix(PG16_FUNCTION_ORDER).unwrap();
        for near_match in [
            format!("{before_order} and 1=0{PG16_FUNCTION_ORDER}"),
            format!("{normal} limit 1"),
            pg16_function_list_sql(
                false,
                &format!("{} and 1=0", pg16_function_name_where("function_alpha")),
            ),
        ] {
            assert!(
                engine
                    .execute_psql_functions_if_applicable(&near_match)
                    .unwrap()
                    .is_none(),
                "near-match must fall through instead of producing fixed route rows: {near_match}"
            );
        }
    }

    #[test]
    fn function_list_raw_candidates_keep_acl_grants_separate_from_pg_proc() {
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (
                1,
                "CREATE FUNCTION function_candidate_target() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
            ),
            (
                2,
                "CREATE FUNCTION function_candidate_decoy() RETURNS text LANGUAGE sql AS 'SELECT ''x'''",
            ),
            (3, "GRANT EXECUTE ON FUNCTION function_candidate_target() TO PUBLIC"),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let routed = engine
            .execute_psql_functions_if_applicable(&pg16_function_list_sql(
                true,
                &pg16_function_name_where("function_candidate_target"),
            ))
            .unwrap()
            .expect("exact PG16 function-list program must enter the terminal GPU route");
        assert_eq!(routed.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(routed.rows.len(), 1);
        assert_eq!(
            routed.rows[0][1],
            SqlValue::Text("function_candidate_target".to_string())
        );

        let catalog = engine.catalog_snapshot();
        let function = &catalog.relational_functions["function_candidate_target"];
        let mut input = DeviceFunctionListInput::default();
        for (grantee, privileges) in &function.acl {
            for privilege in privileges {
                let (grantee_offset, grantee_len) = input.text(grantee).unwrap();
                input.grants.push(DeviceFunctionListGrant {
                    function_oid: function.oid,
                    privilege: match privilege {
                        FunctionPrivilege::Execute => FUNCTION_PRIVILEGE_EXECUTE,
                    },
                    grantee_offset,
                    grantee_len,
                    grantee_is_public: u32::from(grantee == "public"),
                });
            }
        }
        assert_eq!(input.grants.len(), 1);
        assert_eq!(input.grants[0].function_oid, function.oid);
        assert_eq!(input.grants[0].privilege, FUNCTION_PRIVILEGE_EXECUTE);
        assert_eq!(input.grants[0].grantee_is_public, 1);
        assert!(
            input.functions.is_empty(),
            "ACL grants are a separate raw relation"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn function_list_joins_formats_and_filters_on_the_gpu() {
        let _gpu_test = DEVICE_FUNCTION_LIST_TEST_LOCK.lock().unwrap();
        let engine = Engine::new_local_test_engine();
        for (txn_id, sql) in [
            (
                1,
                "CREATE FUNCTION function_zeta() RETURNS text LANGUAGE sql AS 'SELECT ''z'''",
            ),
            (
                2,
                "CREATE FUNCTION function_alpha() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
            ),
            (
                3,
                "CREATE FUNCTION function_decoy() RETURNS boolean LANGUAGE sql AS 'SELECT true'",
            ),
            (4, "CREATE ROLE function_acl_zeta"),
            (5, "CREATE ROLE function_acl_alpha"),
            (
                6,
                "GRANT EXECUTE ON FUNCTION function_alpha() TO function_acl_zeta",
            ),
            (7, "GRANT EXECUTE ON FUNCTION function_alpha() TO PUBLIC"),
            (
                8,
                "GRANT EXECUTE ON FUNCTION function_alpha() TO function_acl_alpha",
            ),
            (
                9,
                "COMMENT ON FUNCTION function_alpha() IS 'selected function comment'",
            ),
        ] {
            engine.execute_text(txn_id, sql).unwrap();
        }
        let catalog = engine.catalog_snapshot();
        let (mut input, request) =
            function_list_input(&catalog, Some("function_alpha"), true).unwrap();
        // Reverse the complete raw relation and add a grant for a decoy function.  The terminal
        // operator, not BTreeMap staging order or a host ACL formatter, must sort/filter the
        // selected function's display.
        input.grants.reverse();
        let decoy_oid = catalog.relational_functions["function_decoy"].oid;
        let mut decoy = input.grants[0];
        decoy.function_oid = decoy_oid;
        input.grants.push(decoy);
        let result = engine
            .execute_psql_function_list_input_gpu(input.clone(), request, true)
            .unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0],
            vec![
                SqlValue::Text("public".to_string()),
                SqlValue::Text("function_alpha".to_string()),
                SqlValue::Text("integer".to_string()),
                SqlValue::Text(String::new()),
                SqlValue::Text("func".to_string()),
                SqlValue::Text("volatile".to_string()),
                SqlValue::Text("unsafe".to_string()),
                SqlValue::Text("postgres".to_string()),
                SqlValue::Text("invoker".to_string()),
                SqlValue::Text(
                    "=X/postgres\nfunction_acl_alpha=X/postgres\nfunction_acl_zeta=X/postgres"
                        .to_string(),
                ),
                SqlValue::Text("sql".to_string()),
                SqlValue::Null,
                SqlValue::Text("selected function comment".to_string()),
            ]
        );

        let ordered = engine
            .execute_psql_functions_if_applicable(&pg16_function_list_sql(
                false,
                PG16_FUNCTION_UNFILTERED_WHERE,
            ))
            .unwrap();
        let ordered = ordered.expect("exact PG16 \\df program must enter the terminal GPU route");
        assert_eq!(ordered.executed_target, DeviceTarget::Gpu(0));
        let names = ordered
            .rows
            .iter()
            .map(|row| row[1].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                SqlValue::Text("function_alpha".to_string()),
                SqlValue::Text("function_decoy".to_string()),
                SqlValue::Text("function_zeta".to_string()),
            ]
        );

        // A duplicate raw matching grant is an invalid device input; it cannot become a repeated
        // visible ACL line or trigger a host-side DISTINCT substitute.
        let sabotage_input = input.clone();
        let duplicate = input.grants[0];
        input.grants.push(duplicate);
        let duplicate_error = engine
            .execute_psql_function_list_input_gpu(input.clone(), request, true)
            .expect_err("duplicate grant candidates must fail closed on the terminal GPU operator");
        assert!(duplicate_error.to_string().contains("device"));

        engine.sabotage_next_psql_function_list_verdict(Some("function_alpha"), true);
        let sabotage_error = engine
            .execute_psql_function_list_input_gpu(sabotage_input, request, true)
            .expect_err("terminal verdict sabotage must not fall back to host ACL formatting");
        assert!(sabotage_error.to_string().contains("device"));
    }
}
