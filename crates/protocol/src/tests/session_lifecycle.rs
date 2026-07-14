//! Session lifecycle and ready-loop state tests.

use super::*;

#[test]
fn session_lifecycle_follows_startup_auth_and_transaction_flow() {
    let mut session = SessionLifecycle::default();
    assert_eq!(session.state(), SessionState::Startup);

    assert_eq!(
        session.apply(SessionEvent::StartupAccepted).unwrap(),
        SessionState::Authenticating
    );
    assert_eq!(
        session
            .apply(SessionEvent::AuthenticationSucceeded)
            .unwrap(),
        SessionState::Ready
    );
    assert_eq!(
        session.apply(SessionEvent::Begin).unwrap(),
        SessionState::InTransaction
    );
    assert_eq!(
        session.apply(SessionEvent::Commit).unwrap(),
        SessionState::Ready
    );
    assert_eq!(
        session.apply(SessionEvent::TerminateRequested).unwrap(),
        SessionState::Terminating
    );
    assert_eq!(
        session.apply(SessionEvent::ConnectionClosed).unwrap(),
        SessionState::Closed
    );
}

#[test]
fn session_lifecycle_rejects_invalid_transitions() {
    let mut session = SessionLifecycle::default();
    assert_eq!(
        session.apply(SessionEvent::Begin).unwrap_err(),
        SessionTransitionError::InvalidTransition {
            from: SessionState::Startup,
            event: SessionEvent::Begin,
        }
    );

    session.apply(SessionEvent::StartupAccepted).unwrap();
    session
        .apply(SessionEvent::AuthenticationSucceeded)
        .unwrap();
    assert_eq!(
        session.apply(SessionEvent::Commit).unwrap_err(),
        SessionTransitionError::InvalidTransition {
            from: SessionState::Ready,
            event: SessionEvent::Commit,
        }
    );
}

#[test]
fn ready_loop_state_tracks_sync_recovery_and_transaction_status() {
    let mut ready_loop = ReadyLoopState::default();
    assert!(ready_loop.should_dispatch_extended_message());
    assert!(!ready_loop.in_transaction());

    ready_loop.set_transaction_status(TransactionStatus::InTransaction);
    assert!(ready_loop.in_transaction());

    ready_loop.mark_extended_error();
    assert!(ready_loop.skip_until_sync());
    assert!(!ready_loop.should_dispatch_extended_message());

    assert!(!ready_loop.clear_extended_error_on_sync(true));
    assert!(ready_loop.skip_until_sync());

    assert!(ready_loop.clear_extended_error_on_sync(false));
    assert!(!ready_loop.skip_until_sync());
    assert!(ready_loop.should_dispatch_extended_message());
    assert!(ready_loop.in_transaction());

    let from_flags = ReadyLoopState::from_flags(false, true);
    assert!(!from_flags.in_transaction());
    assert!(from_flags.skip_until_sync());
}
