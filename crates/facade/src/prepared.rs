//! Protocol-neutral prepared statement ownership.

use gpu_db_engine::Engine;
use gpu_db_sql::{
    Command, CopyToStdout, ParsedCommand, PreparedCommand, Select, SelectProjection, SqlType,
};

use super::{
    map_column, map_db_value, map_execute_error, map_logical_type, map_parse_error, ColumnMeta,
    DbError, DbValue, ErrorCategory, LogicalType, QueryOutcome, SharedEngine, SharedSession,
};

/// A parsed typed command template with direct AST parameter slots.
///
/// Wire adapters keep OIDs and format codes outside this owner and pass decoded [`DbValue`]
/// instances only at Bind/Execute time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedStatement {
    command: Option<PreparedCommand>,
    description: Option<PreparedDescription>,
}

/// One prepared command after Bind has replaced every typed AST parameter slot.
///
/// The parsed command stays opaque to protocol adapters; Execute can clone this neutral owner for
/// the engine without reconstructing or reparsing SQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundPreparedStatement {
    parsed: Option<ParsedCommand>,
    template: Option<PreparedCommand>,
    description: Option<PreparedDescription>,
}

struct ValidatedPreparedExecution {
    parsed: Option<ParsedCommand>,
    catalog_version: Option<u64>,
    expected_columns: Option<Vec<ColumnMeta>>,
}

impl BoundPreparedStatement {
    pub fn is_empty(&self) -> bool {
        self.parsed.is_none()
    }

    fn is_transaction_exit(&self) -> bool {
        self.parsed.as_ref().is_some_and(|parsed| {
            matches!(
                parsed.command(),
                gpu_db_sql::Command::Commit { .. } | gpu_db_sql::Command::Rollback { .. }
            )
        })
    }

    /// Prove that this opaque bound owner is exactly the parameter-free SELECT synthesized for
    /// one COPY TO statement. This inspects the retained AST directly; it never reparses SQL.
    pub(super) fn is_exact_copy_to_select(&self, copy: &CopyToStdout) -> bool {
        let expected = Select {
            table: copy.table.clone(),
            distinct: false,
            projection: copy
                .columns
                .as_ref()
                .map_or(SelectProjection::All, |columns| {
                    SelectProjection::Columns(columns.clone())
                }),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        self.parsed
            .as_ref()
            .is_some_and(|parsed| parsed.command() == &Command::Select(expected))
    }

    fn validate_for_execution(
        &self,
        engine: &Engine,
        txn_id: Option<u64>,
    ) -> Result<ValidatedPreparedExecution, DbError> {
        if self.parsed.is_none() {
            return Ok(ValidatedPreparedExecution {
                parsed: None,
                catalog_version: None,
                expected_columns: Some(Vec::new()),
            });
        }
        let Some(expected) = &self.description else {
            return Ok(ValidatedPreparedExecution {
                parsed: self.parsed.clone(),
                catalog_version: None,
                expected_columns: None,
            });
        };
        let hints = expected
            .parameter_types
            .iter()
            .copied()
            .map(sql_type)
            .map(Some)
            .collect::<Vec<_>>();
        let current = describe_prepared_command(
            engine,
            txn_id,
            self.template
                .as_ref()
                .expect("a non-empty bound command retains its template"),
            &hints,
        )?;
        let current_parameter_types = current
            .parameter_types
            .into_iter()
            .map(map_logical_type)
            .collect::<Vec<_>>();
        let current_result_columns = current
            .result_columns
            .iter()
            .map(map_column)
            .collect::<Vec<_>>();
        if current_parameter_types != expected.parameter_types {
            return Err(DbError {
                category: ErrorCategory::DatatypeMismatch,
                message: "cached prepared-command parameter types changed; re-Parse is required"
                    .to_string(),
            });
        }
        if current_result_columns != expected.result_columns {
            return Err(cached_result_type_changed());
        }
        Ok(ValidatedPreparedExecution {
            parsed: self.parsed.clone(),
            catalog_version: Some(current.catalog_version),
            expected_columns: Some(expected.result_columns.clone()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedDescription {
    parameter_types: Vec<LogicalType>,
    result_columns: Vec<ColumnMeta>,
}

impl PreparedStatement {
    pub fn is_empty(&self) -> bool {
        self.command.is_none()
    }

    pub fn is_transaction_exit(&self) -> bool {
        self.command.as_ref().is_some_and(|prepared| {
            matches!(
                prepared.command(),
                gpu_db_sql::Command::Commit { .. } | gpu_db_sql::Command::Rollback { .. }
            )
        })
    }

    /// Whether this statement directly owns a SQL transaction boundary. Pgwire simple-query
    /// batching uses this neutral classification to avoid wrapping an explicit BEGIN/COMMIT chain
    /// in a second implicit transaction; the adapter never inspects the engine-facing AST.
    pub fn is_transaction_control(&self) -> bool {
        self.command.as_ref().is_some_and(|prepared| {
            matches!(
                prepared.command(),
                gpu_db_sql::Command::Begin { .. }
                    | gpu_db_sql::Command::Commit { .. }
                    | gpu_db_sql::Command::Rollback { .. }
            )
        })
    }

    pub fn parse(sql: &str) -> Result<Self, DbError> {
        match PreparedCommand::parse(sql) {
            Ok(command) => Ok(Self {
                command: Some(command),
                description: None,
            }),
            Err(gpu_db_sql::ParseError::Empty) => Ok(Self {
                command: None,
                description: None,
            }),
            Err(error) => Err(map_parse_error(error)),
        }
    }

    fn describe(
        mut self,
        engine: &Engine,
        txn_id: Option<u64>,
        parameter_type_hints: &[Option<LogicalType>],
    ) -> Result<Self, DbError> {
        let Some(command) = &self.command else {
            let parameter_types = parameter_type_hints
                .iter()
                .enumerate()
                .map(|(index, hint)| {
                    hint.ok_or_else(|| DbError {
                        category: ErrorCategory::IndeterminateDatatype,
                        message: format!(
                            "could not determine data type of unused parameter ${}",
                            index + 1
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.description = Some(PreparedDescription {
                parameter_types,
                result_columns: Vec::new(),
            });
            return Ok(self);
        };
        let parameter_type_hints = parameter_type_hints
            .iter()
            .map(|hint| hint.map(sql_type))
            .collect::<Vec<_>>();
        let description =
            describe_prepared_command(engine, txn_id, command, &parameter_type_hints)?;
        self.description = Some(PreparedDescription {
            parameter_types: description
                .parameter_types
                .into_iter()
                .map(map_logical_type)
                .collect(),
            result_columns: description.result_columns.iter().map(map_column).collect(),
        });
        Ok(self)
    }

    fn revalidate_description(&self, engine: &Engine, txn_id: Option<u64>) -> Result<(), DbError> {
        let expected = self.description.as_ref().ok_or_else(|| DbError {
            category: ErrorCategory::Internal,
            message: "prepared statement was not catalog-described".to_string(),
        })?;
        let Some(command) = &self.command else {
            return Ok(());
        };
        let hints = expected
            .parameter_types
            .iter()
            .copied()
            .map(sql_type)
            .map(Some)
            .collect::<Vec<_>>();
        let current = describe_prepared_command(engine, txn_id, command, &hints)?;
        let current_parameter_types = current
            .parameter_types
            .into_iter()
            .map(map_logical_type)
            .collect::<Vec<_>>();
        if current_parameter_types != expected.parameter_types {
            return Err(DbError {
                category: ErrorCategory::DatatypeMismatch,
                message: "cached prepared-command parameter types changed; re-Parse is required"
                    .to_string(),
            });
        }
        let current_result_columns = current
            .result_columns
            .iter()
            .map(map_column)
            .collect::<Vec<_>>();
        if current_result_columns != expected.result_columns {
            return Err(cached_result_type_changed());
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.description.as_ref().map_or_else(
            || {
                self.command
                    .as_ref()
                    .map_or(0, PreparedCommand::parameter_count)
            },
            |description| description.parameter_types.len(),
        )
    }

    /// Catalog-resolved neutral parameter types. Available on statements prepared through an
    /// engine/facade owner; syntax-only [`Self::parse`] statements return `None`.
    pub fn parameter_types(&self) -> Option<&[LogicalType]> {
        self.description
            .as_ref()
            .map(|description| description.parameter_types.as_slice())
    }

    /// Catalog-resolved neutral result columns. An empty slice means the command has no row result.
    pub fn result_columns(&self) -> Option<&[ColumnMeta]> {
        self.description
            .as_ref()
            .map(|description| description.result_columns.as_slice())
    }

    pub fn bind_values(&self, params: &[DbValue]) -> Result<BoundPreparedStatement, DbError> {
        self.bind(params).map(|parsed| BoundPreparedStatement {
            parsed,
            template: self.command.clone(),
            description: self.description.clone(),
        })
    }

    pub(crate) fn bind(&self, params: &[DbValue]) -> Result<Option<ParsedCommand>, DbError> {
        let Some(command) = &self.command else {
            let expected = self
                .description
                .as_ref()
                .map_or(0, |description| description.parameter_types.len());
            if params.len() == expected {
                return Ok(None);
            }
            return Err(DbError {
                category: ErrorCategory::InvalidRequest,
                message: format!(
                    "empty prepared statement expects {expected} declared parameters, got {}",
                    params.len()
                ),
            });
        };
        let expected = self.description.as_ref().map_or_else(
            || command.parameter_count(),
            |description| description.parameter_types.len(),
        );
        if params.len() != expected {
            return Err(map_parse_error(
                gpu_db_sql::ParseError::InvalidParameterCount {
                    expected,
                    actual: params.len(),
                },
            ));
        }
        let params = params.iter().map(map_db_value).collect::<Vec<_>>();
        command
            .bind(&params[..command.parameter_count()])
            .map(Some)
            .map_err(map_parse_error)
    }
}

impl SharedEngine {
    /// Parse and describe one prepared command against this session's current catalog snapshot,
    /// including a transaction-private catalog overlay when present.
    pub fn prepare_statement(
        &self,
        session: &SharedSession,
        sql: &str,
        parameter_type_hints: &[Option<LogicalType>],
    ) -> Result<PreparedStatement, DbError> {
        self.ensure_session_owner(session)?;
        let prepared = PreparedStatement::parse(sql)?;
        validate_description_session_state(
            session,
            prepared.is_empty(),
            prepared.is_transaction_exit(),
        )?;
        prepared.describe(
            &self.engine,
            session.description_txn_id(),
            parameter_type_hints,
        )
    }

    /// Describe a syntax-validated prepared AST without reparsing its SQL text, using the same
    /// session-private catalog boundary as [`Self::prepare_statement`].
    pub fn describe_prepared_statement(
        &self,
        session: &SharedSession,
        prepared: PreparedStatement,
        parameter_type_hints: &[Option<LogicalType>],
    ) -> Result<PreparedStatement, DbError> {
        self.ensure_session_owner(session)?;
        validate_description_session_state(
            session,
            prepared.is_empty(),
            prepared.is_transaction_exit(),
        )?;
        prepared.describe(
            &self.engine,
            session.description_txn_id(),
            parameter_type_hints,
        )
    }

    /// Revalidate cached statement metadata against the current catalog before a protocol adapter
    /// emits ParameterDescription or RowDescription.
    pub fn revalidate_prepared_description(
        &self,
        session: &SharedSession,
        prepared: &PreparedStatement,
    ) -> Result<(), DbError> {
        self.ensure_session_owner(session)?;
        validate_description_session_state(
            session,
            prepared.is_empty(),
            prepared.is_transaction_exit(),
        )?;
        prepared.revalidate_description(&self.engine, session.description_txn_id())
    }

    /// Revalidate a bound portal's cached metadata without executing its command.
    pub fn revalidate_bound_description(
        &self,
        session: &SharedSession,
        bound: &BoundPreparedStatement,
    ) -> Result<(), DbError> {
        self.ensure_session_owner(session)?;
        validate_description_session_state(session, bound.is_empty(), bound.is_transaction_exit())?;
        bound
            .validate_for_execution(&self.engine, session.description_txn_id())
            .map(|_| ())
    }
}

fn validate_description_session_state(
    session: &SharedSession,
    is_empty: bool,
    is_transaction_exit: bool,
) -> Result<(), DbError> {
    if session.transaction_status() == super::SessionTransactionStatus::FailedTransaction
        && !is_empty
        && !is_transaction_exit
    {
        return Err(super::in_failed_transaction_error());
    }
    Ok(())
}

/// Execute the opaque AST produced at Bind through the one shared-session prepared boundary.
pub(super) fn submit_prepared_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_status() == super::SessionTransactionStatus::FailedTransaction
        && !bound.is_transaction_exit()
        && !bound.is_empty()
    {
        return Err(super::in_failed_transaction_error());
    }
    let validated = match bound.validate_for_execution(&shared.engine, session.description_txn_id())
    {
        Ok(validated) => validated,
        Err(error) => {
            session.mark_transaction_failed();
            return Err(error);
        }
    };
    let outcome = match validated.parsed {
        Some(parsed) => {
            super::submit_parsed_with_catalog(shared, session, parsed, validated.catalog_version)?
        }
        None => QueryOutcome::Empty,
    };
    let result = validate_result_columns(outcome, validated.expected_columns.as_deref());
    if result.is_err() {
        session.mark_transaction_failed();
    }
    result
}

fn describe_prepared_command(
    engine: &Engine,
    txn_id: Option<u64>,
    command: &PreparedCommand,
    parameter_type_hints: &[Option<SqlType>],
) -> Result<gpu_db_engine::PreparedCommandDescription, DbError> {
    match txn_id {
        Some(txn_id) => {
            engine.describe_prepared_command_in_transaction(txn_id, command, parameter_type_hints)
        }
        None => engine.describe_prepared_command(command, parameter_type_hints),
    }
    .map_err(map_execute_error)
}

fn validate_result_columns(
    outcome: QueryOutcome,
    expected: Option<&[ColumnMeta]>,
) -> Result<QueryOutcome, DbError> {
    let Some(expected) = expected else {
        return Ok(outcome);
    };
    let actual = match &outcome {
        QueryOutcome::Rows { columns, .. } | QueryOutcome::Returning { columns, .. } => {
            columns.as_slice()
        }
        _ => &[],
    };
    if actual == expected {
        Ok(outcome)
    } else {
        Err(cached_result_type_changed())
    }
}

fn cached_result_type_changed() -> DbError {
    DbError {
        category: ErrorCategory::Unsupported,
        message: "cached prepared-command result type changed; re-Parse is required".to_string(),
    }
}

fn sql_type(ty: LogicalType) -> SqlType {
    match ty {
        LogicalType::Int2 => SqlType::Int2,
        LogicalType::Int4 => SqlType::Int4,
        LogicalType::Int8 => SqlType::Int8,
        LogicalType::Numeric => SqlType::Numeric {
            precision: gpu_db_sql::NUMERIC_DEFAULT_PRECISION,
            scale: gpu_db_sql::NUMERIC_DEFAULT_SCALE,
        },
        LogicalType::Bool => SqlType::Bool,
        LogicalType::Text => SqlType::Text,
        LogicalType::Date => SqlType::Date,
        LogicalType::Timestamp => SqlType::Timestamp,
        LogicalType::Uuid => SqlType::Uuid,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ColumnMeta, CommandTag, DbError, LogicalType, QueryOutcome, SessionTransactionStatus,
        SharedEngine, SharedSession, SubmissionRequest,
    };
    use super::*;

    fn submit_text(
        engine: &SharedEngine,
        session: &mut SharedSession,
        sql: &str,
    ) -> Result<QueryOutcome, DbError> {
        engine
            .submit(session, SubmissionRequest::Text(sql))
            .into_immediate()
    }

    fn submit_prepared(
        engine: &SharedEngine,
        session: &mut SharedSession,
        bound: &BoundPreparedStatement,
    ) -> Result<QueryOutcome, DbError> {
        engine
            .submit(session, SubmissionRequest::Prepared(bound))
            .into_immediate()
    }

    #[test]
    fn empty_prepared_execute_is_allowed_in_a_failed_transaction() {
        let facade = SharedEngine::new();
        let mut session = facade.open_session();
        submit_text(&facade, &mut session, "BEGIN").unwrap();
        assert!(submit_text(&facade, &mut session, "not valid sql").is_err());
        assert_eq!(
            session.transaction_status(),
            SessionTransactionStatus::FailedTransaction
        );

        let empty = facade.prepare_statement(&session, " ; ", &[]).unwrap();
        let bound = empty.bind_values(&[]).unwrap();
        assert_eq!(
            submit_prepared(&facade, &mut session, &bound).unwrap(),
            QueryOutcome::Empty
        );
    }

    #[test]
    fn every_session_api_rejects_a_session_from_another_shared_engine() {
        let first = SharedEngine::new();
        let second = SharedEngine::new();
        let mut first_session = first.open_session();
        let mut second_session = second.open_session();
        submit_text(&first, &mut first_session, "BEGIN").unwrap();
        submit_text(&second, &mut second_session, "BEGIN").unwrap();

        let prepared = second
            .prepare_statement(
                &second_session,
                "CREATE TABLE cross_engine_target (id int4)",
                &[],
            )
            .unwrap();
        let bound = prepared.bind_values(&[]).unwrap();
        for error in [
            second
                .prepare_statement(&first_session, "CREATE TABLE rejected (id int4)", &[])
                .unwrap_err(),
            second
                .describe_prepared_statement(
                    &first_session,
                    PreparedStatement::parse("CREATE TABLE rejected (id int4)").unwrap(),
                    &[],
                )
                .unwrap_err(),
            second
                .revalidate_prepared_description(&first_session, &prepared)
                .unwrap_err(),
            second
                .revalidate_bound_description(&first_session, &bound)
                .unwrap_err(),
        ] {
            assert_eq!(error.category, ErrorCategory::InvalidRequest);
            assert!(error.message.contains("different SharedEngine"));
        }
        let submit_error = second
            .submit(&mut first_session, SubmissionRequest::Prepared(&bound))
            .into_immediate()
            .unwrap_err();
        assert_eq!(submit_error.category, ErrorCategory::InvalidRequest);
        let close_error = second
            .submit(&mut first_session, SubmissionRequest::CloseSession)
            .into_immediate()
            .unwrap_err();
        assert_eq!(close_error.category, ErrorCategory::InvalidRequest);
        assert_eq!(
            first_session.transaction_status(),
            SessionTransactionStatus::InTransaction
        );
        assert_eq!(
            second_session.transaction_status(),
            SessionTransactionStatus::InTransaction
        );
        assert_eq!(
            second
                .prepare_statement(&second_session, "SELECT id FROM cross_engine_target", &[],)
                .unwrap_err()
                .category,
            ErrorCategory::UndefinedRelation,
            "rejected cross-engine submit must not stage into the colliding transaction id"
        );

        submit_prepared(&second, &mut second_session, &bound).unwrap();
        second
            .prepare_statement(&second_session, "SELECT id FROM cross_engine_target", &[])
            .unwrap();
        submit_text(&first, &mut first_session, "ROLLBACK").unwrap();
        submit_text(&second, &mut second_session, "ROLLBACK").unwrap();
    }

    #[test]
    fn prepare_and_revalidate_follow_the_session_private_catalog() {
        let facade = SharedEngine::new();
        let mut creator = facade.open_session();
        let observer = facade.open_session();
        submit_text(&facade, &mut creator, "BEGIN").unwrap();
        submit_text(
            &facade,
            &mut creator,
            "CREATE TABLE private_prepare (id int4 PRIMARY KEY, value text)",
        )
        .unwrap();

        let prepared = facade
            .prepare_statement(
                &creator,
                "INSERT INTO private_prepare VALUES ($1, $2) RETURNING value",
                &[],
            )
            .unwrap();
        assert_eq!(
            prepared.parameter_types(),
            Some([LogicalType::Int4, LogicalType::Text].as_slice())
        );
        assert_eq!(
            prepared.result_columns(),
            Some(
                [ColumnMeta {
                    name: "value".to_string(),
                    logical_type: LogicalType::Text,
                }]
                .as_slice()
            )
        );
        facade
            .revalidate_prepared_description(&creator, &prepared)
            .unwrap();
        assert_eq!(
            facade
                .prepare_statement(
                    &observer,
                    "SELECT value FROM private_prepare WHERE id = $1",
                    &[],
                )
                .unwrap_err()
                .category,
            ErrorCategory::UndefinedRelation
        );

        submit_text(&facade, &mut creator, "ROLLBACK").unwrap();
        assert_eq!(
            facade
                .revalidate_prepared_description(&creator, &prepared)
                .unwrap_err()
                .category,
            ErrorCategory::UndefinedRelation
        );
    }

    #[test]
    fn facade_description_apis_enforce_failed_transaction_precedence() {
        let facade = SharedEngine::new();
        let mut session = facade.open_session();
        submit_text(&facade, &mut session, "BEGIN").unwrap();
        submit_text(
            &facade,
            &mut session,
            "CREATE TABLE failed_private_description (id int4)",
        )
        .unwrap();
        let prepared = facade
            .prepare_statement(
                &session,
                "INSERT INTO failed_private_description VALUES ($1)",
                &[],
            )
            .unwrap();
        let bound = prepared.bind_values(&[DbValue::Int4(1)]).unwrap();
        assert!(submit_text(&facade, &mut session, "not valid sql").is_err());
        assert_eq!(
            session.transaction_status(),
            SessionTransactionStatus::FailedTransaction
        );

        for error in [
            facade
                .prepare_statement(&session, "SELECT id FROM failed_private_description", &[])
                .unwrap_err(),
            facade
                .describe_prepared_statement(
                    &session,
                    PreparedStatement::parse("SELECT id FROM failed_private_description").unwrap(),
                    &[],
                )
                .unwrap_err(),
            facade
                .revalidate_prepared_description(&session, &prepared)
                .unwrap_err(),
            facade
                .revalidate_bound_description(&session, &bound)
                .unwrap_err(),
        ] {
            assert_eq!(error.category, ErrorCategory::InFailedTransaction);
        }
        assert!(facade.prepare_statement(&session, " ; ", &[]).is_ok());
        assert!(facade.prepare_statement(&session, "ROLLBACK", &[]).is_ok());
        submit_text(&facade, &mut session, "ROLLBACK").unwrap();
    }

    #[test]
    fn committed_private_prepared_description_revalidates_against_publication() {
        let facade = SharedEngine::new();
        let mut session = facade.open_session();
        submit_text(&facade, &mut session, "BEGIN").unwrap();
        submit_text(
            &facade,
            &mut session,
            "CREATE TABLE committed_prepare (id int4, value text)",
        )
        .unwrap();
        let prepared = facade
            .prepare_statement(
                &session,
                "SELECT value FROM committed_prepare WHERE id = $1",
                &[],
            )
            .unwrap();

        submit_text(&facade, &mut session, "COMMIT").unwrap();

        facade
            .revalidate_prepared_description(&session, &prepared)
            .unwrap();
        assert_eq!(
            prepared.parameter_types(),
            Some([LogicalType::Int4].as_slice())
        );
        assert_eq!(
            prepared.result_columns(),
            Some(
                [ColumnMeta {
                    name: "value".to_string(),
                    logical_type: LogicalType::Text,
                }]
                .as_slice()
            )
        );
    }

    #[test]
    fn prepared_description_uses_transaction_isolation_snapshot_rules() {
        let facade = SharedEngine::new();
        let mut reader = facade.open_session();
        let mut writer = facade.open_session();

        submit_text(&facade, &mut reader, "BEGIN").unwrap();
        submit_text(&facade, &mut writer, "CREATE TABLE rc_visible (id int4)").unwrap();
        facade
            .prepare_statement(&reader, "SELECT id FROM rc_visible", &[])
            .unwrap();
        submit_text(&facade, &mut reader, "ROLLBACK").unwrap();

        submit_text(&facade, &mut writer, "CREATE TABLE rr_anchor (id int4)").unwrap();
        submit_text(
            &facade,
            &mut reader,
            "BEGIN ISOLATION LEVEL REPEATABLE READ",
        )
        .unwrap();
        facade
            .prepare_statement(&reader, "SELECT id FROM rr_anchor", &[])
            .unwrap();
        submit_text(&facade, &mut writer, "CREATE TABLE rr_hidden (id int4)").unwrap();
        assert_eq!(
            facade
                .prepare_statement(&reader, "SELECT id FROM rr_hidden", &[])
                .unwrap_err()
                .category,
            ErrorCategory::UndefinedRelation
        );
        submit_text(&facade, &mut reader, "ROLLBACK").unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn explicit_cast_w1_has_replayable_identity_and_exact_returning() {
        let facade = SharedEngine::new();
        facade.engine.set_auto_admit_on_commit(true);
        let mut session = facade.open_session();
        submit_text(
            &facade,
            &mut session,
            "CREATE TABLE cast_accounts (id int4 PRIMARY KEY, balance int8)",
        )
        .unwrap();
        let insert = facade
            .prepare_statement(
                &session,
                "INSERT INTO cast_accounts VALUES ($1::int4, $2::int8)",
                &[],
            )
            .unwrap()
            .bind_values(&[DbValue::Int4(7), DbValue::Int8(100)])
            .unwrap();
        submit_prepared(&facade, &mut session, &insert).unwrap();
        let update = facade
            .prepare_statement(
                &session,
                "UPDATE cast_accounts SET balance = balance + $2::int8 \
                 WHERE id = $1::int4 RETURNING balance",
                &[],
            )
            .unwrap()
            .bind_values(&[DbValue::Int4(7), DbValue::Int8(-9)])
            .unwrap();
        let outcome = submit_prepared(&facade, &mut session, &update).unwrap();
        assert_eq!(
            outcome,
            QueryOutcome::Returning {
                tag: CommandTag::Update,
                columns: vec![ColumnMeta {
                    name: "balance".to_string(),
                    logical_type: LogicalType::Int8,
                }],
                rows: vec![vec![DbValue::Int8(91)]],
                rows_affected: 1,
            }
        );
    }
}
