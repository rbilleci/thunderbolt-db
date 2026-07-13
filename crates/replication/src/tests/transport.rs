#[test]
fn append_entries_transport_request_reports_follower_response() {
    let mut follower = RaftReplicator::new(3);
    let accepted = AppendEntriesRequest {
        leader_term: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![LogEntry {
            term: 2,
            index: 1,
            payload: b"replicated".to_vec().into(),
        }],
        leader_commit: 1,
    }
    .apply_to(&mut follower);

    assert!(accepted.accepted);
    assert_eq!(accepted.follower_term, 2);
    assert_eq!(accepted.follower_commit_index, 1);
    assert_eq!(accepted.follower_applied_index, 0);
    assert_eq!(accepted.error, None);

    let rejected = AppendEntriesRequest {
        leader_term: 1,
        prev_log_index: 1,
        prev_log_term: 2,
        entries: vec![],
        leader_commit: 1,
    }
    .apply_to(&mut follower);

    assert!(!rejected.accepted);
    assert_eq!(rejected.follower_term, 2);
    assert_eq!(rejected.follower_commit_index, 1);
    assert_eq!(rejected.follower_applied_index, 0);
    assert_eq!(
        rejected.error.as_deref(),
        Some("proposal failed: stale leader term 1 (local term 2)")
    );
}

#[test]
fn append_entries_transport_tcp_loopback_round_trips_frame() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut follower = RaftReplicator::new(3);
        let response = serve_append_entries_once(
            &listener,
            &mut follower,
            std::time::Duration::from_millis(250),
        )
        .unwrap();
        assert!(response.accepted);
        follower.commit_index()
    });

    let request = AppendEntriesRequest {
        leader_term: 3,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![LogEntry {
            term: 3,
            index: 1,
            payload: b"replicated over loopback".to_vec().into(),
        }],
        leader_commit: 1,
    };
    let response =
        send_append_entries_once(addr, &request, std::time::Duration::from_millis(250)).unwrap();
    assert!(response.accepted);
    assert_eq!(response.follower_term, 3);
    assert_eq!(response.follower_commit_index, 1);
    assert_eq!(server.join().unwrap(), 1);
}
