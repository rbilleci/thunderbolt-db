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
