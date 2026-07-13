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
fn append_entries_transport_frame_round_trips_request_and_response() {
    let request = AppendEntriesRequest {
        leader_term: 7,
        prev_log_index: 3,
        prev_log_term: 6,
        entries: vec![
            LogEntry {
                term: 7,
                index: 4,
                payload: b"set a=1".to_vec().into(),
            },
            LogEntry {
                term: 7,
                index: 5,
                payload: b"set b=2".to_vec().into(),
            },
        ],
        leader_commit: 5,
    };
    let decoded = AppendEntriesRequest::decode_frame(&request.encode_frame()).unwrap();
    assert_eq!(decoded, request);

    let response = AppendEntriesResponse {
        accepted: false,
        follower_term: 8,
        follower_commit_index: 4,
        follower_applied_index: 3,
        error: Some("stale leader term".to_string()),
    };
    let decoded = AppendEntriesResponse::decode_frame(&response.encode_frame()).unwrap();
    assert_eq!(decoded, response);
}

#[test]
fn append_entries_transport_frame_rejects_truncated_payload() {
    let request = AppendEntriesRequest {
        leader_term: 1,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![LogEntry {
            term: 1,
            index: 1,
            payload: b"payload".to_vec().into(),
        }],
        leader_commit: 0,
    };
    let mut frame = request.encode_frame();
    frame.pop();

    let err = AppendEntriesRequest::decode_frame(&frame).unwrap_err();
    assert!(err
        .to_string()
        .contains("append entries frame decode failed"));
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
