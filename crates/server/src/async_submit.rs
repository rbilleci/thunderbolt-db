//! Bounded Tokio-to-blocking facade submission helpers.
//!
//! Runtime tasks never hold a synchronous session lock or execute the blocking engine directly.
//! Cancellation can win while a request is queued for a permit and is checked again inside the
//! worker immediately before the facade boundary.

use std::sync::{Arc, Mutex};

use gpu_db_facade::{
    BoundPreparedStatement, DbError, PointLookupBatcher, QueryOutcome, SharedEngine, SharedSession,
    SubmissionDispatch, SubmissionRequest,
};

use crate::cancellation::{cancel_effect_free_success, ActiveRequest};
use crate::extended::{
    BindCompletion, BindRequest, DescriptionOwner, ExtendedError, ExtendedSession, PrepareAnalysis,
    PrepareRequest, TransactionAction,
};
use crate::wire_response::{
    cancellation_checked_outcome, cancellation_error, cancelled_extended_error,
    encode_outcome_messages,
};

pub(crate) async fn execute_text(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    sql: String,
) -> Result<Result<QueryOutcome, DbError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        engine
            .submit(&mut session, SubmissionRequest::Text(&sql))
            .into_immediate()
    })
    .await
    .map_err(|error| error.to_string())
}

pub(crate) async fn execute_text_cancellable(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    sql: String,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, DbError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(cancellation_error()));
    };
    let cancellation = active.token();
    let outcome = tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        engine
            .submit(&mut session, SubmissionRequest::Text(&sql))
            .into_immediate()
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancellation_checked_outcome(active, outcome))
}

pub(crate) async fn execute_prepared_cancellable(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    bound: BoundPreparedStatement,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, DbError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(cancellation_error()));
    };
    let cancellation = active.token();
    let outcome = tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        engine
            .submit(&mut session, SubmissionRequest::Prepared(&bound))
            .into_immediate()
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancellation_checked_outcome(active, outcome))
}

/// Run effect-free Parse catalog analysis without allowing a cancellation received while queued
/// to install a prepared statement or emit ParseComplete afterward.
pub(crate) async fn analyze_prepare_cancellable(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    request: PrepareRequest,
    active: &ActiveRequest,
) -> Result<Result<PrepareAnalysis, DbError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(cancellation_error()));
    };
    let cancellation = active.token();
    let analysis = tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(cancellation_error());
        }
        ExtendedSession::analyze_prepare(&engine, &mut session, &request)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        analysis,
        cancellation_error,
    ))
}

/// Decode and bind parameters off the runtime without allowing a queued cancellation to install
/// a portal or emit BindComplete afterward. Bind is connection-local and effect-free until the
/// caller installs its returned completion.
pub(crate) async fn bind_cancellable(
    executor: &Arc<tokio::sync::Semaphore>,
    request: BindRequest,
    active: &ActiveRequest,
) -> Result<Result<BindCompletion, ExtendedError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(cancelled_extended_error()));
    };
    let cancellation = active.token();
    let completion = tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(cancelled_extended_error());
        }
        ExtendedSession::bind_request(request)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        completion,
        cancelled_extended_error,
    ))
}

/// Revalidate cached Describe ownership without allowing cancellation during the bounded-worker
/// wait to emit stale/effect-free metadata afterward.
pub(crate) async fn revalidate_description_cancellable(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    owner: DescriptionOwner,
    active: &ActiveRequest,
) -> Result<Result<(), ExtendedError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(cancelled_extended_error()));
    };
    let cancellation = active.token();
    let validation = tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(cancelled_extended_error());
        }
        let session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(cancelled_extended_error());
        }
        ExtendedSession::revalidate_description_owner(&engine, &session, owner)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        validation,
        cancelled_extended_error,
    ))
}

/// Complete one simple-query implicit transaction without allowing cancellation while queued to
/// admit COMMIT. A pre-admission cancellation becomes the original 57014 error and uses the
/// uncancellable cleanup path to roll back; once COMMIT crosses the facade, its result wins.
pub(crate) async fn complete_simple_query_action(
    engine: Arc<SharedEngine>,
    session: Arc<Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    active: &ActiveRequest,
    outcome: &Result<QueryOutcome, DbError>,
    response: &mut Vec<u8>,
) -> Result<(), String> {
    if outcome.is_err() {
        session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mark_transaction_failed();
    }
    let action = extended.simple_query_completion_action(outcome);
    let Some(sql) = action.sql() else {
        extended.complete_transaction_action(action, true);
        return Ok(());
    };
    let completion = if matches!(action, TransactionAction::CommitImplicit) {
        execute_text_cancellable(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            sql.to_string(),
            active,
        )
        .await?
    } else {
        execute_text(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            sql.to_string(),
        )
        .await?
    };
    match completion {
        Ok(_) => extended.complete_transaction_action(action, true),
        Err(error) => {
            response.extend_from_slice(
                &encode_outcome_messages(Err(error)).map_err(|error| error.to_string())?,
            );
            if let Some(cleanup) = action.failure_cleanup().sql() {
                if execute_text(engine, session, executor, cleanup.to_string())
                    .await?
                    .is_ok()
                {
                    extended.complete_transaction_action(action.failure_cleanup(), true);
                }
            }
        }
    }
    Ok(())
}

/// Classify one idle single-statement request through the batching facade without allowing a
/// cancellation observed while waiting for the session mutex to cross a fallback mutation path.
pub(crate) async fn execute_batchable_or_fallback(
    engine: Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Arc<PointLookupBatcher>,
    session: Arc<Mutex<SharedSession>>,
    sql: String,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, DbError>, String> {
    execute_batchable_or_fallback_with_hook(engine, executor, batcher, session, sql, active, || {})
        .await
}

async fn execute_batchable_or_fallback_with_hook<F>(
    engine: Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Arc<PointLookupBatcher>,
    session: Arc<Mutex<SharedSession>>,
    sql: String,
    active: &ActiveRequest,
    after_initial_check: F,
) -> Result<Result<QueryOutcome, DbError>, String>
where
    F: FnOnce() + Send + 'static,
{
    let dispatch = {
        let Some(_permit) = active
            .acquire_permit(executor)
            .await
            .map_err(|error| error.to_string())?
        else {
            return Ok(Err(cancellation_error()));
        };
        let cancellation = active.token();
        tokio::task::spawn_blocking(move || {
            if cancellation.is_cancelled() {
                return DispatchOut::Immediate(Err(cancellation_error()));
            }
            after_initial_check();
            let mut session = session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cancellation.is_cancelled() {
                return DispatchOut::Immediate(Err(cancellation_error()));
            }
            match engine.submit(
                &mut session,
                SubmissionRequest::BatchedText {
                    sql: &sql,
                    batcher: &batcher,
                },
            ) {
                SubmissionDispatch::Immediate(result) => DispatchOut::Immediate(result),
                SubmissionDispatch::Batched(receiver) => DispatchOut::Batched(receiver),
            }
        })
        .await
        .map_err(|error| error.to_string())?
    };
    match dispatch {
        DispatchOut::Immediate(result) => Ok(result),
        DispatchOut::Batched(receiver) => Ok(await_batched_result(receiver, active).await),
    }
}

/// Prefer an already-ready batch result over simultaneous cancellation so its original error can
/// never be relabelled. If cancellation becomes ready first, dropping the read-only receiver is a
/// pre-effect cancellation and the batcher may finish harmlessly for no consumer.
pub(crate) async fn await_batched_result(
    mut receiver: tokio::sync::oneshot::Receiver<Result<QueryOutcome, DbError>>,
    active: &ActiveRequest,
) -> Result<QueryOutcome, DbError> {
    tokio::select! {
        biased;
        result = &mut receiver => match result {
            Ok(result) => result,
            Err(_) => Err(DbError {
                category: gpu_db_facade::ErrorCategory::Internal,
                message: "batched point-lookup did not produce a response (coalescer unavailable)"
                    .to_string(),
            }),
        },
        _ = active.cancelled() => Err(cancellation_error()),
    }
}

enum DispatchOut {
    Immediate(Result<QueryOutcome, DbError>),
    Batched(tokio::sync::oneshot::Receiver<Result<QueryOutcome, DbError>>),
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::time::Duration;

    use super::*;
    use crate::cancellation::CancellationRegistry;
    use gpu_db_facade::{ErrorCategory, PreparedStatement, SessionTransactionStatus};

    #[tokio::test]
    async fn cancellation_while_waiting_for_a_permit_is_pre_effect() {
        let engine = Arc::new(SharedEngine::new());
        let session = Arc::new(Mutex::new(engine.open_session()));
        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();

        let execute = execute_text_cancellable(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            "CREATE TABLE queued_cancel (id int4)".to_string(),
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (outcome, ()) = tokio::join!(execute, cancel);
        assert_eq!(
            outcome.unwrap().unwrap_err().category,
            ErrorCategory::Cancelled
        );
        drop(active);

        // The canceled worker never crossed the facade boundary: the same CREATE remains valid.
        executor.add_permits(1);
        execute_text(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            "CREATE TABLE queued_cancel (id int4)".to_string(),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn batched_fallback_rechecks_cancellation_after_the_session_lock() {
        let engine = Arc::new(SharedEngine::new());
        let mut initial_session = engine.open_session();
        engine
            .submit(
                &mut initial_session,
                SubmissionRequest::Text(
                    "CREATE TABLE cancelled_batched_fallback (id int4 PRIMARY KEY)",
                ),
            )
            .into_immediate()
            .unwrap();
        let session = Arc::new(Mutex::new(initial_session));
        let batcher = Arc::new(PointLookupBatcher::with_triggers(
            Arc::clone(&engine),
            1,
            Duration::ZERO,
        ));
        let executor = Arc::new(tokio::sync::Semaphore::new(1));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let process_id = connection.backend_key().process_id();
        let secret = connection.backend_key().secret_key_bytes();
        let active = connection.begin_request().unwrap();
        let rendezvous = Arc::new(Barrier::new(2));

        // Hold the exact mutex the worker needs. The hook runs only after the worker has consumed
        // its permit and passed its first cancellation check, so leaving the barrier forces it to
        // wait at the session lock before the final pre-facade check.
        let session_guard = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let worker = {
            let engine = Arc::clone(&engine);
            let executor = Arc::clone(&executor);
            let batcher = Arc::clone(&batcher);
            let session = Arc::clone(&session);
            let rendezvous = Arc::clone(&rendezvous);
            tokio::spawn(async move {
                let result = execute_batchable_or_fallback_with_hook(
                    engine,
                    &executor,
                    batcher,
                    session,
                    "INSERT INTO cancelled_batched_fallback VALUES (1)".to_string(),
                    &active,
                    move || {
                        rendezvous.wait();
                    },
                )
                .await;
                (result, active)
            })
        };
        rendezvous.wait();
        assert_eq!(executor.available_permits(), 0);
        assert!(registry.cancel(process_id, &secret));
        drop(session_guard);

        let (result, active) = worker.await.unwrap();
        let error = result.unwrap().unwrap_err();
        assert_eq!(error.category, ErrorCategory::Cancelled);
        assert_eq!(
            gpu_db_facade::pg_adapter::error_sqlstate(error.category),
            "57014"
        );
        drop(active);

        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("INSERT INTO cancelled_batched_fallback VALUES (1)"),
            )
            .into_immediate()
            .expect("cancelled BatchedText fallback published the primary key");
    }

    #[tokio::test]
    async fn queued_implicit_begin_cancels_without_transaction_state() {
        let engine = Arc::new(SharedEngine::new());
        let session = Arc::new(Mutex::new(engine.open_session()));
        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();

        let active = connection.begin_request().unwrap();
        let begin = execute_text_cancellable(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            "BEGIN".to_string(),
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (begin, ()) = tokio::join!(begin, cancel);
        assert_eq!(
            begin.unwrap().unwrap_err().category,
            ErrorCategory::Cancelled
        );
        assert_eq!(
            session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .transaction_status(),
            SessionTransactionStatus::Idle,
            "a cancelled queued implicit BEGIN must not leak a transaction"
        );
        drop(active);
    }

    #[tokio::test]
    async fn queued_parse_analysis_uses_a_fresh_zero_permit_executor() {
        let engine = Arc::new(SharedEngine::new());
        let session = Arc::new(Mutex::new(engine.open_session()));
        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        let request = PrepareRequest {
            statement_name: "queued".to_string(),
            parsed: PreparedStatement::parse("SELECT 1 AS value").unwrap(),
            copy: None,
            query: "SELECT 1 AS value".to_string(),
            parameter_type_hints: Vec::new(),
        };
        let analysis = analyze_prepare_cancellable(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            request,
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (analysis, ()) = tokio::join!(analysis, cancel);
        let Err(error) = analysis.unwrap() else {
            panic!("queued Parse analysis unexpectedly completed")
        };
        assert_eq!(
            error.category,
            ErrorCategory::Cancelled,
            "queued Parse analysis must not cross the facade boundary"
        );
        assert_eq!(executor.available_permits(), 0);

        let Err(error) = cancel_effect_free_success(
            &active,
            Err::<PrepareAnalysis, _>(DbError {
                category: ErrorCategory::Engine,
                message: "original Parse facade error".to_string(),
            }),
            cancellation_error,
        ) else {
            panic!("metadata error unexpectedly became success")
        };
        assert_eq!(error.category, ErrorCategory::Engine);
        assert_eq!(error.message, "original Parse facade error");
        let Err(error) = cancel_effect_free_success(
            &active,
            Err::<BindCompletion, _>(ExtendedError::new("XX000", "original Bind metadata error")),
            cancelled_extended_error,
        ) else {
            panic!("Bind metadata error unexpectedly became success")
        };
        assert_eq!(error.code, "XX000");
        assert_eq!(error.message, "original Bind metadata error");
    }
}
