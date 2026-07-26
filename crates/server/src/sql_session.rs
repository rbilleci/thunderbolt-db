//! Connection-local SQL PREPARE/EXECUTE and cursor execution over the canonical facade session.
//!
//! Syntax remains in the `sql_prepared` and `sql_cursor` leaves. This bridge owns installation,
//! lookup, cancellation, and blocking/async parity for those connection-local objects; it never
//! owns relational execution, WAL, or publication.

use std::sync::{Arc, Mutex};

use gpu_db_facade::{
    BoundPreparedStatement, CommandTag, DbError, ErrorCategory, PreparedStatement, QueryOutcome,
    SessionTransactionStatus, SharedEngine, SharedSession, SubmissionRequest,
};

use crate::async_submit::{
    analyze_sql_prepare_cancellable,
    execute_prepared_cancellable as execute_prepared_shared_session_blocking_cancellable,
    execute_text_cancellable as execute_shared_session_blocking_cancellable,
};
use crate::cancellation::ActiveRequest;
use crate::extended::{ExtendedError, ExtendedSession};
use crate::sql_cursor::SqlCursorAction;
use crate::sql_prepared::{
    bind_sql_execute, classify_sql_prepared_statement, sql_prepare_name, SqlPreparedAction,
};
use crate::wire_response::{cancellation_checked_outcome, cancellation_error};
use crate::{shared_session_transaction_status, submit_text_cancellable};

fn submit_prepared(
    engine: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    engine
        .submit(session, SubmissionRequest::Prepared(bound))
        .into_immediate()
}

pub(super) fn submit_prepared_cancellable(
    engine: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
    active: &ActiveRequest,
) -> Result<QueryOutcome, DbError> {
    if active.is_cancelled() {
        return Err(cancellation_error());
    }
    cancellation_checked_outcome(active, submit_prepared(engine, session, bound))
}

fn command_outcome(tag: &str) -> QueryOutcome {
    QueryOutcome::Command {
        tag: CommandTag::Other(tag.to_string()),
        rows_affected: None,
    }
}

fn deallocate_command_tag(target: &crate::sql_prepared::SqlDeallocateTarget) -> &'static str {
    match target {
        crate::sql_prepared::SqlDeallocateTarget::All => "DEALLOCATE ALL",
        crate::sql_prepared::SqlDeallocateTarget::Named(_) => "DEALLOCATE",
    }
}

fn extended_error_as_db(error: ExtendedError) -> DbError {
    DbError {
        category: ErrorCategory::InvalidRequest,
        message: error.message,
    }
}

pub(crate) fn cursor_error_is_connection_local(error: &DbError) -> bool {
    matches!(
        error.message.as_str(),
        "cursor already exists" | "cursor does not exist"
    )
}

pub(crate) fn outcome_error_poison_transaction(outcome: &Result<QueryOutcome, DbError>) -> bool {
    outcome
        .as_ref()
        .is_err_and(|error| !cursor_error_is_connection_local(error))
}

fn validate_cursor_query(query: &str) -> Result<(), DbError> {
    match PreparedStatement::parse(query) {
        Ok(prepared) if prepared.parameter_count() > 0 => Err(DbError {
            category: ErrorCategory::Unsupported,
            message: "parameterized cursor declarations are not supported".to_string(),
        }),
        Err(error)
            if matches!(
                error.message.as_str(),
                "LIMIT must not be negative" | "OFFSET must not be negative"
            ) =>
        {
            Err(error)
        }
        Ok(_) | Err(_) => Ok(()),
    }
}

fn bind_cursor_sql_execute(
    extended: &ExtendedSession,
    query: &str,
) -> Result<Option<BoundPreparedStatement>, DbError> {
    let Some(action) = classify_sql_prepared_statement(query)? else {
        return Ok(None);
    };
    let SqlPreparedAction::Execute { name, arguments } = action else {
        return Err(DbError {
            category: ErrorCategory::Unsupported,
            message: "cursor declarations only support relational SELECT or SQL EXECUTE"
                .to_string(),
        });
    };
    let plan = extended
        .sql_prepared_plan(&name)
        .map_err(extended_error_as_db)?;
    if let Some(error) = plan.deferred_execution_error {
        return Err(error);
    }
    bind_sql_execute(&plan.prepared, &arguments).map(Some)
}

fn in_failed_transaction_error() -> DbError {
    DbError {
        category: ErrorCategory::InFailedTransaction,
        message: "current transaction is aborted, commands ignored until end of transaction block"
            .to_string(),
    }
}

pub(super) fn classify_prepared_action(
    extended: &ExtendedSession,
    statement: &str,
    transaction_status: SessionTransactionStatus,
) -> Result<Option<SqlPreparedAction>, DbError> {
    if transaction_status != SessionTransactionStatus::FailedTransaction {
        if let Some(name) = sql_prepare_name(statement)? {
            if extended.has_sql_prepared(&name) {
                return Err(DbError {
                    category: ErrorCategory::InvalidRequest,
                    message: "prepared statement already exists".to_string(),
                });
            }
        }
    }
    classify_sql_prepared_statement(statement)
}

pub(super) fn execute_prepared_action_blocking(
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    action: SqlPreparedAction,
    active: &ActiveRequest,
) -> Result<QueryOutcome, DbError> {
    if active.is_cancelled() {
        return Err(cancellation_error());
    }
    if session.transaction_status() == SessionTransactionStatus::FailedTransaction {
        return Err(in_failed_transaction_error());
    }
    match action {
        SqlPreparedAction::Prepare {
            name,
            query,
            parameter_hints,
        } => {
            let plan = ExtendedSession::analyze_sql_prepare_plan(
                engine,
                session,
                &query,
                &parameter_hints,
            )?;
            if active.is_cancelled() {
                return Err(cancellation_error());
            }
            extended
                .install_sql_prepared_plan(name, plan)
                .map_err(extended_error_as_db)?;
            Ok(command_outcome("PREPARE"))
        }
        SqlPreparedAction::Execute { name, arguments } => {
            let plan = extended
                .sql_prepared_plan(&name)
                .map_err(extended_error_as_db)?;
            if let Some(error) = plan.deferred_execution_error {
                return Err(error);
            }
            let bound = bind_sql_execute(&plan.prepared, &arguments)?;
            submit_prepared_cancellable(engine, session, &bound, active)
        }
        SqlPreparedAction::Deallocate(target) => {
            let tag = deallocate_command_tag(&target);
            extended
                .deallocate_sql_prepared(target)
                .map_err(extended_error_as_db)?;
            Ok(command_outcome(tag))
        }
    }
}

pub(super) async fn execute_prepared_action_async(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    action: SqlPreparedAction,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, DbError>, String> {
    if active.is_cancelled() {
        return Ok(Err(cancellation_error()));
    }
    if shared_session_transaction_status(&session) == SessionTransactionStatus::FailedTransaction {
        return Ok(Err(in_failed_transaction_error()));
    }
    match action {
        SqlPreparedAction::Prepare {
            name,
            query,
            parameter_hints,
        } => {
            let prepared = analyze_sql_prepare_cancellable(
                engine,
                session,
                executor,
                query,
                parameter_hints,
                active,
            )
            .await?;
            match prepared {
                Ok(plan) => {
                    if active.is_cancelled() {
                        return Ok(Err(cancellation_error()));
                    }
                    Ok(extended
                        .install_sql_prepared_plan(name, plan)
                        .map(|()| command_outcome("PREPARE"))
                        .map_err(extended_error_as_db))
                }
                Err(error) => Ok(Err(error)),
            }
        }
        SqlPreparedAction::Execute { name, arguments } => {
            let plan = match extended.sql_prepared_plan(&name) {
                Ok(plan) => plan,
                Err(error) => return Ok(Err(extended_error_as_db(error))),
            };
            if let Some(error) = plan.deferred_execution_error {
                return Ok(Err(error));
            }
            let bound = match bind_sql_execute(&plan.prepared, &arguments) {
                Ok(bound) => bound,
                Err(error) => return Ok(Err(error)),
            };
            execute_prepared_shared_session_blocking_cancellable(
                engine, session, executor, bound, active,
            )
            .await
        }
        SqlPreparedAction::Deallocate(target) => {
            let tag = deallocate_command_tag(&target);
            Ok(extended
                .deallocate_sql_prepared(target)
                .map(|()| command_outcome(tag))
                .map_err(extended_error_as_db))
        }
    }
}

pub(super) fn execute_cursor_action_blocking(
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    action: SqlCursorAction,
    active: &ActiveRequest,
) -> Result<QueryOutcome, DbError> {
    if active.is_cancelled() {
        return Err(cancellation_error());
    }
    if session.transaction_status() == SessionTransactionStatus::FailedTransaction {
        return Err(in_failed_transaction_error());
    }
    match action {
        SqlCursorAction::Declare { name, query } => {
            extended
                .ensure_sql_cursor_name_available(&name)
                .map_err(extended_error_as_db)?;
            validate_cursor_query(&query)?;
            let transaction_bound = session.transaction_status() != SessionTransactionStatus::Idle;
            let outcome = match bind_cursor_sql_execute(extended, &query)? {
                Some(bound) => submit_prepared_cancellable(engine, session, &bound, active)?,
                None => submit_text_cancellable(engine, session, &query, active)?,
            };
            if active.is_cancelled() {
                return Err(cancellation_error());
            }
            extended
                .install_sql_cursor(name, outcome, transaction_bound)
                .map_err(extended_error_as_db)?;
            Ok(command_outcome("DECLARE CURSOR"))
        }
        SqlCursorAction::Fetch { name, count } => extended
            .fetch_sql_cursor(&name, count)
            .map_err(extended_error_as_db),
        SqlCursorAction::Move { name, count } => extended
            .move_sql_cursor(&name, count)
            .map_err(extended_error_as_db),
        SqlCursorAction::Close(target) => {
            extended
                .close_sql_cursor(target)
                .map_err(extended_error_as_db)?;
            Ok(command_outcome("CLOSE CURSOR"))
        }
    }
}

pub(super) async fn execute_cursor_action_async(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    action: SqlCursorAction,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, DbError>, String> {
    if active.is_cancelled() {
        return Ok(Err(cancellation_error()));
    }
    if shared_session_transaction_status(&session) == SessionTransactionStatus::FailedTransaction {
        return Ok(Err(in_failed_transaction_error()));
    }
    match action {
        SqlCursorAction::Declare { name, query } => {
            if let Err(error) = extended.ensure_sql_cursor_name_available(&name) {
                return Ok(Err(extended_error_as_db(error)));
            }
            if let Err(error) = validate_cursor_query(&query) {
                return Ok(Err(error));
            }
            let transaction_bound =
                shared_session_transaction_status(&session) != SessionTransactionStatus::Idle;
            let outcome = match bind_cursor_sql_execute(extended, &query) {
                Err(error) => return Ok(Err(error)),
                Ok(Some(bound)) => {
                    execute_prepared_shared_session_blocking_cancellable(
                        engine, session, executor, bound, active,
                    )
                    .await?
                }
                Ok(None) => {
                    execute_shared_session_blocking_cancellable(
                        engine, session, executor, query, active,
                    )
                    .await?
                }
            };
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(error) => return Ok(Err(error)),
            };
            if active.is_cancelled() {
                return Ok(Err(cancellation_error()));
            }
            Ok(extended
                .install_sql_cursor(name, outcome, transaction_bound)
                .map(|()| command_outcome("DECLARE CURSOR"))
                .map_err(extended_error_as_db))
        }
        SqlCursorAction::Fetch { name, count } => Ok(extended
            .fetch_sql_cursor(&name, count)
            .map_err(extended_error_as_db)),
        SqlCursorAction::Move { name, count } => Ok(extended
            .move_sql_cursor(&name, count)
            .map_err(extended_error_as_db)),
        SqlCursorAction::Close(target) => Ok(extended
            .close_sql_cursor(target)
            .map(|()| command_outcome("CLOSE CURSOR"))
            .map_err(extended_error_as_db)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationRegistry;
    use crate::sql_cursor::SqlCursorCloseTarget;
    use crate::sql_prepared::SqlDeallocateTarget;
    use gpu_db_facade::{ColumnMeta, DbValue, LogicalType};

    #[test]
    fn deallocate_all_uses_the_distinct_postgresql_command_tag() {
        assert_eq!(
            deallocate_command_tag(&SqlDeallocateTarget::All),
            "DEALLOCATE ALL"
        );
        assert_eq!(
            deallocate_command_tag(&SqlDeallocateTarget::Named("kept".to_string())),
            "DEALLOCATE"
        );
    }

    #[test]
    fn only_connection_local_cursor_state_errors_preserve_a_transaction() {
        for message in ["cursor already exists", "cursor does not exist"] {
            let outcome = Err(DbError {
                category: ErrorCategory::InvalidRequest,
                message: message.to_string(),
            });
            assert!(!outcome_error_poison_transaction(&outcome));
        }
        let query_error = Err(DbError {
            category: ErrorCategory::Syntax,
            message: "LIMIT must not be negative".to_string(),
        });
        assert!(outcome_error_poison_transaction(&query_error));
    }

    #[test]
    fn cursor_query_preflight_preserves_bounded_frozen_diagnostics() {
        let parameterized =
            validate_cursor_query("SELECT id FROM accounts WHERE id = $1").unwrap_err();
        assert_eq!(parameterized.category, ErrorCategory::Unsupported);
        assert_eq!(
            parameterized.message,
            "parameterized cursor declarations are not supported"
        );

        for (query, message) in [
            (
                "SELECT id FROM accounts LIMIT -1",
                "LIMIT must not be negative",
            ),
            (
                "SELECT id FROM accounts OFFSET -1",
                "OFFSET must not be negative",
            ),
        ] {
            let error = validate_cursor_query(query).unwrap_err();
            assert_eq!(error.category, ErrorCategory::Syntax);
            assert_eq!(error.message, message);
        }
    }

    #[test]
    fn duplicate_sql_prepare_name_precedes_unsupported_type_analysis() {
        let mut extended = ExtendedSession::default();
        extended
            .install_sql_prepared(
                "duplicate_name".to_string(),
                gpu_db_facade::PreparedStatement::parse("SELECT 1").unwrap(),
            )
            .unwrap();
        let error = classify_prepared_action(
            &extended,
            "PREPARE duplicate_name(jsonb) AS SELECT $1",
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
        assert_eq!(error.category, ErrorCategory::InvalidRequest);
        assert_eq!(error.message, "prepared statement already exists");

        let failed_error = classify_prepared_action(
            &extended,
            "PREPARE duplicate_name(jsonb) AS SELECT $1",
            SessionTransactionStatus::FailedTransaction,
        )
        .unwrap_err();
        assert_ne!(failed_error.message, "prepared statement already exists");
    }

    fn install_local_objects(
        engine: &SharedEngine,
        session: &mut SharedSession,
        extended: &mut ExtendedSession,
        active: &ActiveRequest,
    ) {
        execute_prepared_action_blocking(
            engine,
            session,
            extended,
            SqlPreparedAction::Prepare {
                name: "kept".to_string(),
                query: "SELECT 1 AS one".to_string(),
                parameter_hints: Vec::new(),
            },
            active,
        )
        .unwrap();
        extended
            .install_sql_cursor(
                "kept_cursor".to_string(),
                QueryOutcome::Rows {
                    columns: vec![ColumnMeta {
                        name: "one".to_string(),
                        logical_type: LogicalType::Int4,
                        numeric_typmod: None,
                    }],
                    rows: vec![vec![DbValue::Int4(1)]],
                },
                true,
            )
            .unwrap();
    }

    fn enter_failed_transaction(engine: &SharedEngine, session: &mut SharedSession) {
        engine
            .submit(session, SubmissionRequest::Text("BEGIN"))
            .into_immediate()
            .unwrap();
        engine
            .submit(
                session,
                SubmissionRequest::Text("SELECT id FROM missing_failed_state_relation"),
            )
            .into_immediate()
            .unwrap_err();
        assert_eq!(
            session.transaction_status(),
            SessionTransactionStatus::FailedTransaction
        );
    }

    #[test]
    fn blocking_local_prepared_and_cursor_actions_obey_failed_transaction_precedence() {
        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        let mut extended = ExtendedSession::default();
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        install_local_objects(&engine, &mut session, &mut extended, &active);
        enter_failed_transaction(&engine, &mut session);

        for result in [
            execute_cursor_action_blocking(
                &engine,
                &mut session,
                &mut extended,
                SqlCursorAction::Fetch {
                    name: "kept_cursor".to_string(),
                    count: Some(1),
                },
                &active,
            ),
            execute_cursor_action_blocking(
                &engine,
                &mut session,
                &mut extended,
                SqlCursorAction::Close(SqlCursorCloseTarget::Named("kept_cursor".to_string())),
                &active,
            ),
            execute_prepared_action_blocking(
                &engine,
                &mut session,
                &mut extended,
                SqlPreparedAction::Deallocate(SqlDeallocateTarget::Named("kept".to_string())),
                &active,
            ),
        ] {
            assert_eq!(
                result.unwrap_err().category,
                ErrorCategory::InFailedTransaction
            );
        }

        engine
            .submit(&mut session, SubmissionRequest::Text("ROLLBACK"))
            .into_immediate()
            .unwrap();
        execute_prepared_action_blocking(
            &engine,
            &mut session,
            &mut extended,
            SqlPreparedAction::Deallocate(SqlDeallocateTarget::Named("kept".to_string())),
            &active,
        )
        .expect("failed-state DEALLOCATE must not remove the prepared statement");
    }

    #[tokio::test]
    async fn async_local_prepared_and_cursor_actions_obey_failed_transaction_precedence() {
        let engine = Arc::new(SharedEngine::new());
        let mut session = engine.open_session();
        let mut extended = ExtendedSession::default();
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        install_local_objects(&engine, &mut session, &mut extended, &active);
        enter_failed_transaction(&engine, &mut session);
        let session = Arc::new(Mutex::new(session));
        let executor = Arc::new(tokio::sync::Semaphore::new(1));

        let fetch = execute_cursor_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            &mut extended,
            SqlCursorAction::Fetch {
                name: "kept_cursor".to_string(),
                count: Some(1),
            },
            &active,
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(fetch.category, ErrorCategory::InFailedTransaction);

        let close = execute_cursor_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            &mut extended,
            SqlCursorAction::Close(SqlCursorCloseTarget::Named("kept_cursor".to_string())),
            &active,
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(close.category, ErrorCategory::InFailedTransaction);

        let deallocate = execute_prepared_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            &mut extended,
            SqlPreparedAction::Deallocate(SqlDeallocateTarget::Named("kept".to_string())),
            &active,
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(deallocate.category, ErrorCategory::InFailedTransaction);

        engine
            .submit(
                &mut session.lock().unwrap(),
                SubmissionRequest::Text("ROLLBACK"),
            )
            .into_immediate()
            .unwrap();
        execute_prepared_action_blocking(
            &engine,
            &mut session.lock().unwrap(),
            &mut extended,
            SqlPreparedAction::Deallocate(SqlDeallocateTarget::Named("kept".to_string())),
            &active,
        )
        .expect("failed-state async DEALLOCATE must not remove the prepared statement");
    }
}
