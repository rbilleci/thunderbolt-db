//! PostgreSQL BackendKeyData registration and exact active-request cancellation.
//!
//! This owner is deliberately server-local. It can interrupt protocol work before facade
//! admission or suppress an effect-free result, but it is not an execution, transaction, WAL, or
//! publication boundary. A successful mutation is never rewritten as cancelled after the fact.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use rand::{rngs::OsRng, RngCore};
use subtle::ConstantTimeEq;

const MAX_PROCESS_ID: u32 = i32::MAX as u32;

/// Process/key pair encoded in PostgreSQL BackendKeyData. The secret intentionally has no Debug
/// implementation and is compared in constant time when a CancelRequest arrives.
pub(crate) struct BackendKey {
    process_id: u32,
    secret_key: [u8; 4],
}

impl BackendKey {
    pub(crate) fn process_id_i32(&self) -> i32 {
        self.process_id as i32
    }

    pub(crate) fn secret_key_i32(&self) -> i32 {
        i32::from_be_bytes(self.secret_key)
    }

    #[cfg(test)]
    pub(crate) fn process_id(&self) -> u32 {
        self.process_id
    }

    #[cfg(test)]
    pub(crate) fn secret_key_bytes(&self) -> [u8; 4] {
        self.secret_key
    }
}

struct RegisteredTarget {
    secret_key: [u8; 4],
    state: Weak<RequestState>,
}

/// One registry is shared by every listener worker. CancelRequest connections have no session and
/// do nothing except try this exact lookup, request cancellation, and close without a response.
pub(crate) struct CancellationRegistry {
    next_process_id: AtomicU32,
    targets: Mutex<HashMap<u32, RegisteredTarget>>,
}

impl CancellationRegistry {
    pub(crate) fn new() -> Self {
        let mut seed = [0_u8; 4];
        OsRng.fill_bytes(&mut seed);
        let seed = (u32::from_ne_bytes(seed) & MAX_PROCESS_ID).max(1);
        Self {
            next_process_id: AtomicU32::new(seed),
            targets: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn register(self: &Arc<Self>) -> ConnectionCancellation {
        let state = Arc::new(RequestState::new());
        let mut targets = self
            .targets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let process_id = loop {
            let candidate = self.next_process_id.fetch_add(1, Ordering::Relaxed) & MAX_PROCESS_ID;
            if candidate != 0 && !targets.contains_key(&candidate) {
                break candidate;
            }
        };
        let secret_key = loop {
            let mut candidate = [0_u8; 4];
            OsRng.fill_bytes(&mut candidate);
            if targets
                .values()
                .all(|target| !bool::from(target.secret_key.ct_eq(candidate.as_slice())))
            {
                break candidate;
            }
        };
        targets.insert(
            process_id,
            RegisteredTarget {
                secret_key,
                state: Arc::downgrade(&state),
            },
        );
        ConnectionCancellation {
            key: BackendKey {
                process_id,
                secret_key,
            },
            state,
            registry: Arc::downgrade(self),
        }
    }

    /// Resolve one raw startup CancelRequest. PostgreSQL sends exactly four secret bytes; any
    /// other shape, unknown process id, stale registration, or wrong secret is a silent no-op.
    pub(crate) fn cancel(&self, process_id: u32, secret_key: &[u8]) -> bool {
        let Ok(secret_key): Result<&[u8; 4], _> = secret_key.try_into() else {
            return false;
        };
        let target = {
            let targets = self
                .targets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(target) = targets.get(&process_id) else {
                return false;
            };
            if !bool::from(target.secret_key.ct_eq(secret_key)) {
                return false;
            }
            target.state.upgrade()
        };
        target.is_some_and(|state| state.request_cancel())
    }

    #[cfg(test)]
    pub(crate) fn request_is_active(&self, process_id: u32) -> bool {
        self.targets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&process_id)
            .and_then(|target| target.state.upgrade())
            .is_some_and(|state| state.active_generation.load(Ordering::Acquire) != 0)
    }
}

impl Default for CancellationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection registration lease. Dropping it removes the key before a process id can ever be
/// reused, while a weak-pointer identity check prevents an old lease from deleting a newer entry.
pub(crate) struct ConnectionCancellation {
    key: BackendKey,
    state: Arc<RequestState>,
    registry: Weak<CancellationRegistry>,
}

impl ConnectionCancellation {
    pub(crate) fn backend_key(&self) -> &BackendKey {
        &self.key
    }

    pub(crate) fn begin_request(&self) -> io::Result<ActiveRequest> {
        self.state.begin()
    }
}

impl Drop for ConnectionCancellation {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut targets = registry
            .targets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if targets
            .get(&self.key.process_id)
            .is_some_and(|target| target.state.as_ptr() == Arc::as_ptr(&self.state))
        {
            targets.remove(&self.key.process_id);
        }
    }
}

struct RequestState {
    next_generation: AtomicU64,
    active_generation: AtomicU64,
    cancelled_generation: AtomicU64,
    notification: tokio::sync::Notify,
}

impl RequestState {
    fn new() -> Self {
        Self {
            next_generation: AtomicU64::new(1),
            active_generation: AtomicU64::new(0),
            cancelled_generation: AtomicU64::new(0),
            notification: tokio::sync::Notify::new(),
        }
    }

    fn begin(self: &Arc<Self>) -> io::Result<ActiveRequest> {
        let generation = loop {
            let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
            if generation != 0 {
                break generation;
            }
        };
        self.active_generation
            .compare_exchange(0, generation, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| io::Error::other("connection attempted overlapping frontend requests"))?;
        Ok(ActiveRequest {
            state: Arc::clone(self),
            generation,
        })
    }

    fn request_cancel(&self) -> bool {
        let generation = self.active_generation.load(Ordering::Acquire);
        if generation == 0 {
            return false;
        }
        self.cancelled_generation
            .store(generation, Ordering::Release);
        // There is at most one waiter for the one active request. `notify_one` retains a permit
        // when cancellation races between the waiter's flag check and its first poll; the
        // non-buffering `notify_waiters` form could lose that wakeup.
        self.notification.notify_one();
        true
    }
}

/// Exact active frontend-request lease. Generation matching prevents a late CancelRequest for one
/// command from cancelling the next command on a reused connection.
pub(crate) struct ActiveRequest {
    state: Arc<RequestState>,
    generation: u64,
}

impl ActiveRequest {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.state.cancelled_generation.load(Ordering::Acquire) == self.generation
    }

    pub(crate) async fn cancelled(&self) {
        loop {
            let notified = self.state.notification.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }

    /// Wait for one bounded blocking-executor slot without allowing a cancellation received
    /// while queued to cross the facade boundary afterward.
    pub(crate) async fn acquire_permit<'a>(
        &self,
        semaphore: &'a tokio::sync::Semaphore,
    ) -> Result<Option<tokio::sync::SemaphorePermit<'a>>, tokio::sync::AcquireError> {
        tokio::select! {
            biased;
            _ = self.cancelled() => Ok(None),
            permit = semaphore.acquire() => permit.map(Some),
        }
    }

    pub(crate) fn token(&self) -> RequestCancellation {
        RequestCancellation {
            state: Arc::clone(&self.state),
            generation: self.generation,
        }
    }
}

/// Replace only an effect-free successful result after cancellation. An existing error always
/// wins unchanged: it may carry stronger protocol precedence or describe an indeterminate
/// post-durable facade outcome. Callers must use this only when every `Ok(T)` is suppressible.
pub(crate) fn cancel_effect_free_success<T, E>(
    active: &ActiveRequest,
    result: Result<T, E>,
    cancellation_error: impl FnOnce() -> E,
) -> Result<T, E> {
    if active.is_cancelled() && result.is_ok() {
        Err(cancellation_error())
    } else {
        result
    }
}

/// Cloneable observation token for a blocking worker. It deliberately owns no active-request
/// lease, so dropping a queued/finished worker cannot clear the connection's generation early.
#[derive(Clone)]
pub(crate) struct RequestCancellation {
    state: Arc<RequestState>,
    generation: u64,
}

impl RequestCancellation {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.state.cancelled_generation.load(Ordering::Acquire) == self.generation
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        if self
            .state
            .active_generation
            .compare_exchange(self.generation, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.state.cancelled_generation.compare_exchange(
                self.generation,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_db_facade::{
        ColumnMeta, CommandTag, DbError, DbValue, ErrorCategory, LogicalType, QueryOutcome,
        SessionTransactionStatus, SharedEngine, SubmissionRequest,
    };

    fn backend_messages(mut bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let mut messages = Vec::new();
        while !bytes.is_empty() {
            assert!(bytes.len() >= 5, "truncated backend frame");
            let length = u32::from_be_bytes(bytes[1..5].try_into().unwrap()) as usize;
            let frame_length = 1 + length;
            assert!(length >= 4 && bytes.len() >= frame_length);
            messages.push((bytes[0], bytes[5..frame_length].to_vec()));
            bytes = &bytes[frame_length..];
        }
        messages
    }

    fn assert_error_and_ready(messages: &[(u8, Vec<u8>)], sqlstate: &[u8], status: u8) {
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'E', b'Z']
        );
        assert!(messages[0]
            .1
            .windows(sqlstate.len())
            .any(|window| window == sqlstate));
        assert_eq!(messages[1].1, vec![status]);
    }

    #[test]
    fn exact_key_cancels_only_the_active_generation() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let process_id = connection.backend_key().process_id();
        let secret = connection.backend_key().secret_key_bytes();

        assert!(
            !registry.cancel(process_id, &secret),
            "idle cancel must be a no-op"
        );
        let first = connection.begin_request().unwrap();
        assert!(!registry.cancel(process_id.wrapping_add(1), &secret));
        assert!(!registry.cancel(process_id, &[0_u8; 4]));
        assert!(!registry.cancel(process_id, &[0_u8; 3]));
        assert!(!first.is_cancelled());
        assert!(registry.cancel(process_id, &secret));
        assert!(first.is_cancelled());
        drop(first);

        let second = connection.begin_request().unwrap();
        assert!(
            !second.is_cancelled(),
            "cancel leaked into the next generation"
        );
    }

    #[tokio::test]
    async fn notification_and_stale_registration_are_exact() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let process_id = connection.backend_key().process_id();
        let secret = connection.backend_key().secret_key_bytes();
        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(process_id, &secret));
        active.cancelled().await;
        drop(active);
        drop(connection);
        assert!(!registry.cancel(process_id, &secret));
    }

    #[tokio::test]
    async fn ready_batched_error_beats_simultaneous_cancellation() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        sender
            .send(Err(DbError {
                category: ErrorCategory::Engine,
                message: "indeterminate-like batched facade error".to_string(),
            }))
            .unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        let error = crate::async_submit::await_batched_result(receiver, &active)
            .await
            .unwrap_err();
        assert_eq!(error.category, ErrorCategory::Engine);
        assert_eq!(error.message, "indeterminate-like batched facade error");
    }

    #[test]
    fn registrations_have_distinct_live_process_ids_and_secrets() {
        let registry = Arc::new(CancellationRegistry::new());
        let first = registry.register();
        let second = registry.register();
        assert_ne!(
            first.backend_key().process_id(),
            second.backend_key().process_id()
        );
        assert_ne!(
            first.backend_key().secret_key_bytes(),
            second.backend_key().secret_key_bytes()
        );
    }

    #[test]
    fn cancellation_is_pre_effect_and_never_relabels_a_successful_mutation() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));

        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        let error = crate::submit_text_cancellable(
            &engine,
            &mut session,
            "CREATE TABLE cancelled_before_effect (id int4)",
            &active,
        )
        .unwrap_err();
        assert_eq!(error.category, ErrorCategory::Cancelled);
        drop(active);

        // The same CREATE succeeds afterward, proving the cancelled attempt never reached the
        // facade mutation boundary.
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("CREATE TABLE cancelled_before_effect (id int4)"),
            )
            .into_immediate()
            .unwrap();

        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        let published = QueryOutcome::Command {
            tag: CommandTag::Insert,
            rows_affected: Some(1),
        };
        assert_eq!(
            crate::cancellation_checked_outcome(&active, Ok(published.clone())).unwrap(),
            published,
            "a successful mutation must not be falsely reported as cancelled"
        );
        assert_eq!(
            crate::cancellation_checked_outcome(
                &active,
                Ok(QueryOutcome::Rows {
                    columns: Vec::new(),
                    rows: Vec::new(),
                }),
            )
            .unwrap_err()
            .category,
            ErrorCategory::Cancelled,
        );
    }

    #[test]
    fn cancellation_suppresses_only_effect_free_encoded_results() {
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));

        let mut rows = Ok(QueryOutcome::Rows {
            columns: vec![ColumnMeta {
                name: "value".to_string(),
                logical_type: LogicalType::Int4,
            }],
            rows: vec![vec![DbValue::Int4(7)]],
        });
        let encoded = crate::encode_cancellable_outcome_messages(&active, &mut rows).unwrap();
        assert_eq!(encoded.first(), Some(&b'E'));
        assert!(encoded
            .windows(b"C57014\0".len())
            .any(|window| window == b"C57014\0"));
        assert!(matches!(
            rows,
            Err(DbError {
                category: ErrorCategory::Cancelled,
                ..
            })
        ));

        let published = QueryOutcome::Command {
            tag: CommandTag::Insert,
            rows_affected: Some(1),
        };
        let mut command = Ok(published.clone());
        let encoded = crate::encode_cancellable_outcome_messages(&active, &mut command).unwrap();
        assert_eq!(encoded.first(), Some(&b'C'));
        assert_eq!(command.unwrap(), published);

        // The facade maps engine post-durable `ExecuteError::Indeterminate` failures to an Engine
        // DbError. Such an error may describe a WAL-claimed mutation that recovery will publish;
        // cancellation must preserve it instead of inventing a false 57014 result.
        let mut indeterminate = Err(DbError {
            category: ErrorCategory::Engine,
            message: "indeterminate post-durable apply; recovery owns publication".to_string(),
        });
        let encoded =
            crate::encode_cancellable_outcome_messages(&active, &mut indeterminate).unwrap();
        assert_eq!(encoded.first(), Some(&b'E'));
        assert!(!encoded
            .windows(b"C57014\0".len())
            .any(|window| window == b"C57014\0"));
        let error = indeterminate.unwrap_err();
        assert_eq!(error.category, ErrorCategory::Engine);
        assert!(error.message.contains("recovery owns publication"));

        let preserved = cancel_effect_free_success(
            &active,
            Err::<(), _>(DbError {
                category: ErrorCategory::Engine,
                message: "metadata/COPY facade error".to_string(),
            }),
            crate::cancellation_error,
        )
        .unwrap_err();
        assert_eq!(preserved.category, ErrorCategory::Engine);
        assert_eq!(preserved.message, "metadata/COPY facade error");
    }

    #[test]
    fn blocking_cancelled_simple_query_fails_explicit_transaction_until_rollback() {
        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        for sql in [
            "CREATE TABLE blocking_cancelled_tx (id int4 PRIMARY KEY)",
            "BEGIN",
        ] {
            engine
                .submit(&mut session, SubmissionRequest::Text(sql))
                .into_immediate()
                .unwrap();
        }
        assert_eq!(
            session.transaction_status(),
            SessionTransactionStatus::InTransaction
        );

        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let mut extended = crate::extended::ExtendedSession::default();
        let mut copy_in = None;

        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        let cancelled = crate::execute_simple_query_blocking(
            &engine,
            &mut session,
            &mut extended,
            &mut copy_in,
            "INSERT INTO blocking_cancelled_tx VALUES (1)",
            &active,
        )
        .unwrap();
        drop(active);
        assert_error_and_ready(&backend_messages(&cancelled), b"C57014\0", b'E');

        let active = connection.begin_request().unwrap();
        let blocked = crate::execute_simple_query_blocking(
            &engine,
            &mut session,
            &mut extended,
            &mut copy_in,
            "INSERT INTO blocking_cancelled_tx VALUES (2)",
            &active,
        )
        .unwrap();
        drop(active);
        assert_error_and_ready(&backend_messages(&blocked), b"C25P02\0", b'E');

        let active = connection.begin_request().unwrap();
        let rollback = crate::execute_simple_query_blocking(
            &engine,
            &mut session,
            &mut extended,
            &mut copy_in,
            "ROLLBACK",
            &active,
        )
        .unwrap();
        drop(active);
        assert_eq!(
            backend_messages(&rollback),
            vec![(b'C', b"ROLLBACK\0".to_vec()), (b'Z', vec![b'I'])]
        );

        let active = connection.begin_request().unwrap();
        let count = crate::execute_simple_query_blocking(
            &engine,
            &mut session,
            &mut extended,
            &mut copy_in,
            "SELECT COUNT(*) FROM blocking_cancelled_tx",
            &active,
        )
        .unwrap();
        drop(active);
        let count = backend_messages(&count);
        assert_eq!(
            count.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'T', b'D', b'C', b'Z']
        );
        assert!(
            count[1].1.ends_with(b"0"),
            "cancelled row published: {count:?}"
        );

        let active = connection.begin_request().unwrap();
        let reused = crate::execute_simple_query_blocking(
            &engine,
            &mut session,
            &mut extended,
            &mut copy_in,
            "INSERT INTO blocking_cancelled_tx VALUES (3)",
            &active,
        )
        .unwrap();
        assert_eq!(
            backend_messages(&reused)
                .iter()
                .map(|(tag, _)| *tag)
                .collect::<Vec<_>>(),
            vec![b'C', b'Z']
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_cancelled_simple_query_fails_explicit_transaction_until_rollback() {
        let engine = Arc::new(SharedEngine::new());
        let mut initial_session = engine.open_session();
        for sql in [
            "CREATE TABLE async_cancelled_tx (id int4 PRIMARY KEY)",
            "BEGIN",
        ] {
            engine
                .submit(&mut initial_session, SubmissionRequest::Text(sql))
                .into_immediate()
                .unwrap();
        }
        let session = Arc::new(Mutex::new(initial_session));
        let executor = Arc::new(tokio::sync::Semaphore::new(1));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let mut extended = crate::extended::ExtendedSession::default();
        let mut copy_in = None;

        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        let cancelled = crate::execute_simple_query_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            None,
            &mut extended,
            &mut copy_in,
            "INSERT INTO async_cancelled_tx VALUES (1)".to_string(),
            &active,
        )
        .await
        .unwrap();
        drop(active);
        assert_error_and_ready(&backend_messages(&cancelled), b"C57014\0", b'E');

        let active = connection.begin_request().unwrap();
        let blocked = crate::execute_simple_query_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            None,
            &mut extended,
            &mut copy_in,
            "INSERT INTO async_cancelled_tx VALUES (2)".to_string(),
            &active,
        )
        .await
        .unwrap();
        drop(active);
        assert_error_and_ready(&backend_messages(&blocked), b"C25P02\0", b'E');

        let active = connection.begin_request().unwrap();
        let rollback = crate::execute_simple_query_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            None,
            &mut extended,
            &mut copy_in,
            "ROLLBACK".to_string(),
            &active,
        )
        .await
        .unwrap();
        drop(active);
        assert_eq!(
            backend_messages(&rollback),
            vec![(b'C', b"ROLLBACK\0".to_vec()), (b'Z', vec![b'I'])]
        );

        let active = connection.begin_request().unwrap();
        let count = crate::execute_simple_query_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            None,
            &mut extended,
            &mut copy_in,
            "SELECT COUNT(*) FROM async_cancelled_tx".to_string(),
            &active,
        )
        .await
        .unwrap();
        drop(active);
        let count = backend_messages(&count);
        assert_eq!(
            count.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'T', b'D', b'C', b'Z']
        );
        assert!(
            count[1].1.ends_with(b"0"),
            "cancelled row published: {count:?}"
        );

        let active = connection.begin_request().unwrap();
        let reused = crate::execute_simple_query_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            None,
            &mut extended,
            &mut copy_in,
            "INSERT INTO async_cancelled_tx VALUES (3)".to_string(),
            &active,
        )
        .await
        .unwrap();
        assert_eq!(
            backend_messages(&reused)
                .iter()
                .map(|(tag, _)| *tag)
                .collect::<Vec<_>>(),
            vec![b'C', b'Z']
        );
    }

    #[tokio::test]
    async fn queued_simple_query_implicit_commit_cancels_to_rollback_without_publication() {
        let engine = Arc::new(SharedEngine::new());
        let mut initial_session = engine.open_session();
        for sql in [
            "CREATE TABLE queued_simple_commit (id int4 PRIMARY KEY)",
            "BEGIN",
            "INSERT INTO queued_simple_commit VALUES (1)",
        ] {
            engine
                .submit(&mut initial_session, SubmissionRequest::Text(sql))
                .into_immediate()
                .unwrap();
        }
        let session = Arc::new(Mutex::new(initial_session));
        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        let mut extended = crate::extended::ExtendedSession::default();
        extended
            .complete_transaction_action(crate::extended::TransactionAction::BeginImplicit, true);
        let outcome = Ok(QueryOutcome::Command {
            tag: CommandTag::Insert,
            rows_affected: Some(1),
        });
        let mut response = Vec::new();

        let mut completion = Box::pin(crate::complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            &mut extended,
            &active,
            &outcome,
            &mut response,
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), completion.as_mut())
                .await
                .is_err(),
            "implicit COMMIT did not queue at the zero-permit boundary"
        );
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        executor.add_permits(1);
        completion.await.unwrap();
        drop(active);

        let messages = backend_messages(&response);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'E']
        );
        assert!(messages[0]
            .1
            .windows(b"C57014\0".len())
            .any(|window| window == b"C57014\0"));
        assert!(!extended.has_implicit_transaction());
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(session.transaction_status(), SessionTransactionStatus::Idle);
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("INSERT INTO queued_simple_commit VALUES (1)"),
            )
            .into_immediate()
            .expect("cancelled queued COMMIT published the staged primary key");
    }

    #[test]
    fn cached_portal_rows_cancel_but_returning_results_do_not() {
        let engine = SharedEngine::new();
        let session = engine.open_session();
        let mut extended = crate::extended::ExtendedSession::default();
        extended
            .parse(
                "query".to_string(),
                "SELECT 1 AS value",
                &[],
                |sql, hints| engine.prepare_statement(&session, sql, hints),
            )
            .unwrap();
        extended
            .bind("rows".to_string(), "query", &[], &[], &[])
            .unwrap();
        extended
            .set_execution_outcome(
                "rows",
                Ok(QueryOutcome::Rows {
                    columns: vec![ColumnMeta {
                        name: "value".to_string(),
                        logical_type: LogicalType::Int4,
                    }],
                    rows: vec![vec![DbValue::Int4(1)]],
                }),
            )
            .unwrap();
        extended
            .bind("returning".to_string(), "query", &[], &[], &[])
            .unwrap();
        extended
            .set_execution_outcome(
                "returning",
                Ok(QueryOutcome::Returning {
                    tag: CommandTag::Insert,
                    columns: vec![ColumnMeta {
                        name: "value".to_string(),
                        logical_type: LogicalType::Int4,
                    }],
                    rows: vec![vec![DbValue::Int4(1)]],
                    rows_affected: 1,
                }),
            )
            .unwrap();
        extended
            .bind("error".to_string(), "query", &[], &[], &[])
            .unwrap();
        extended
            .set_execution_outcome(
                "error",
                Err(DbError {
                    category: ErrorCategory::Engine,
                    message: "indeterminate post-durable portal outcome".to_string(),
                }),
            )
            .unwrap();

        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        assert!(registry.cancel(
            connection.backend_key().process_id(),
            &connection.backend_key().secret_key_bytes(),
        ));
        assert_eq!(
            crate::encode_execute_cancellable(&mut extended, "rows", 0, &active)
                .unwrap_err()
                .code,
            "57014"
        );
        let encoded =
            crate::encode_execute_cancellable(&mut extended, "returning", 0, &active).unwrap();
        assert_eq!(encoded.first(), Some(&b'D'));
        assert!(encoded.windows(1).any(|window| window == b"C"));
        let error =
            crate::encode_execute_cancellable(&mut extended, "error", 0, &active).unwrap_err();
        assert_ne!(error.code, "57014");
        assert_eq!(error.message, "indeterminate post-durable portal outcome");
    }
}
