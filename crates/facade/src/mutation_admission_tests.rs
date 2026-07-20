use super::*;

fn submit_text(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    shared
        .submit(session, SubmissionRequest::Text(sql))
        .into_immediate()
}

#[test]
fn canonical_session_submission_owns_transaction_control() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(&shared, &mut session, "BEGIN").unwrap();
    assert!(session.in_transaction());
    submit_text(&shared, &mut session, "ROLLBACK").unwrap();
    assert!(!session.in_transaction());
}

#[test]
fn compatibility_reads_stay_outside_mutation_admission() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(&shared, &mut session, "SET answer=forty-two").unwrap();
    let (visible_before, wal_before) = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };

    for sql in ["GET answer", "SELECT bounded_fn()", "SELECT currval('seq')"] {
        assert!(matches!(
            submit_text(&shared, &mut session, sql).unwrap(),
            QueryOutcome::Command { .. }
        ));
    }

    let engine = shared.read_engine().unwrap();
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}

#[test]
fn nonconcurrent_returning_remains_unsupported_and_pre_effect() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let (visible_before, wal_before) = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = submit_text(
        &shared,
        &mut session,
        "INSERT INTO missing VALUES (1) RETURNING id",
    )
    .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Unsupported);
    assert_eq!(
        error.message,
        "DML RETURNING requires the GPU-native concurrent mutation path"
    );
    let engine = shared.read_engine().unwrap();
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}
