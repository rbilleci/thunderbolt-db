//! Startup packet framing and parser coverage.

use super::*;

fn with_length_prefix(mut payload: Vec<u8>) -> Vec<u8> {
    let len = (payload.len() + 4) as u32;
    let mut frame = len.to_be_bytes().to_vec();
    frame.append(&mut payload);
    frame
}

#[test]
fn parses_pg_v3_startup_packet_with_params() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"user\0postgres\0database\0gpu\0application_name\0\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("user".to_string(), "postgres".to_string()),
                ("database".to_string(), "gpu".to_string()),
                ("application_name".to_string(), "".to_string())
            ],
        }
    );
}

#[test]
fn parses_ssl_gssenc_and_cancel_requests() {
    let ssl = with_length_prefix(PG_SSL_REQUEST_CODE.to_be_bytes().to_vec());
    assert_eq!(
        parse_startup_packet(&ssl).unwrap(),
        StartupPacket::SslRequest
    );

    let gssenc = with_length_prefix(PG_GSSENC_REQUEST_CODE.to_be_bytes().to_vec());
    assert_eq!(
        parse_startup_packet(&gssenc).unwrap(),
        StartupPacket::GssEncRequest
    );

    let mut cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
    cancel_payload.extend_from_slice(&123u32.to_be_bytes());
    cancel_payload.extend_from_slice(&456u32.to_be_bytes());
    let cancel = with_length_prefix(cancel_payload);
    assert_eq!(
        parse_startup_packet(&cancel).unwrap(),
        StartupPacket::CancelRequest {
            process_id: 123,
            secret_key: 456u32.to_be_bytes().to_vec(),
        }
    );

    let mut extended_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
    extended_cancel_payload.extend_from_slice(&321u32.to_be_bytes());
    extended_cancel_payload.extend_from_slice(b"longer-secret-key");
    let extended_cancel = with_length_prefix(extended_cancel_payload);
    assert_eq!(
        parse_startup_packet(&extended_cancel).unwrap(),
        StartupPacket::CancelRequest {
            process_id: 321,
            secret_key: b"longer-secret-key".to_vec(),
        }
    );

    let mut max_length_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
    max_length_cancel_payload.extend_from_slice(&777u32.to_be_bytes());
    max_length_cancel_payload.extend_from_slice(&vec![0xCD; PG_CANCEL_SECRET_KEY_MAX_BYTES]);
    let max_length_cancel = with_length_prefix(max_length_cancel_payload);
    assert_eq!(
        parse_startup_packet(&max_length_cancel).unwrap(),
        StartupPacket::CancelRequest {
            process_id: 777,
            secret_key: vec![0xCD; PG_CANCEL_SECRET_KEY_MAX_BYTES],
        }
    );

    let mut zero_process_id_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
    zero_process_id_cancel_payload.extend_from_slice(&0u32.to_be_bytes());
    zero_process_id_cancel_payload.extend_from_slice(b"zero-key");
    let zero_process_id_cancel = with_length_prefix(zero_process_id_cancel_payload);
    assert_eq!(
        parse_startup_packet(&zero_process_id_cancel).unwrap(),
        StartupPacket::CancelRequest {
            process_id: 0,
            secret_key: b"zero-key".to_vec(),
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_empty_params() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.push(0);
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_extra_trailing_terminator() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"user\0postgres\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![("user".to_string(), "postgres".to_string())],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_empty_parameter_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"options\0\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![("options".to_string(), String::new())],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_multiple_params() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"user\0postgres\0database\0analytics\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("user".to_string(), "postgres".to_string()),
                ("database".to_string(), "analytics".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_duplicate_keys() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"options\0-a\0options\0-b\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("options".to_string(), "-a".to_string()),
                ("options".to_string(), "-b".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_mixed_empty_and_non_empty_values() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice(b"application_name\0gpu-db\0options\0\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("application_name".to_string(), "gpu-db".to_string()),
                ("options".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_params() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("cliënt\0möde\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![("cliënt".to_string(), "möde".to_string())],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0påth=a\0möde\0påth=b\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), "påth=a".to_string()),
                ("möde".to_string(), "påth=b".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_empty_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![("möde".to_string(), String::new())],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_multiple_utf8_params() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("cliënt\0möde\0rôle\0anályst\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("cliënt".to_string(), "möde".to_string()),
                ("rôle".to_string(), "anályst".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys_and_empty_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0påth=a\0möde\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), "påth=a".to_string()),
                ("möde".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_empty_value_before_utf8_param() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0rôle\0anályst\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), String::new()),
                ("rôle".to_string(), "anályst".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_multiple_utf8_params_and_middle_empty_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("cliënt\0möde\0rôle\0\0tëam\0dév\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("cliënt".to_string(), "möde".to_string()),
                ("rôle".to_string(), String::new()),
                ("tëam".to_string(), "dév".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_multiple_utf8_params_and_trailing_empty_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("cliënt\0möde\0tëam\0dév\0rôle\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("cliënt".to_string(), "möde".to_string()),
                ("tëam".to_string(), "dév".to_string()),
                ("rôle".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_multiple_utf8_empty_values() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0rôle\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), String::new()),
                ("rôle".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_utf8_duplicate_keys_and_multiple_empty_values() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0möde\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), String::new()),
                ("möde".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_three_utf8_duplicate_keys() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0påth=a\0möde\0\0möde\0påth=c\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), "påth=a".to_string()),
                ("möde".to_string(), String::new()),
                ("möde".to_string(), "påth=c".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0påth=a\0rôle\0anályst\0möde\0påth=b\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), "påth=a".to_string()),
                ("rôle".to_string(), "anályst".to_string()),
                ("möde".to_string(), "påth=b".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys_and_empty_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0påth=a\0rôle\0anályst\0möde\0\0tëam\0dév\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), "påth=a".to_string()),
                ("rôle".to_string(), "anályst".to_string()),
                ("möde".to_string(), String::new()),
                ("tëam".to_string(), "dév".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_interleaved_utf8_duplicate_keys_and_both_empty_values() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0rôle\0anályst\0möde\0\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), String::new()),
                ("rôle".to_string(), "anályst".to_string()),
                ("möde".to_string(), String::new()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_startup_packet_with_interleaved_utf8_empty_duplicate_before_value() {
    let mut payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    payload.extend_from_slice("möde\0\0rôle\0anályst\0möde\0påth=b\0\0".as_bytes());
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version: PG_PROTOCOL_V3,
            params: vec![
                ("möde".to_string(), String::new()),
                ("rôle".to_string(), "anályst".to_string()),
                ("möde".to_string(), "påth=b".to_string()),
            ],
        }
    );
}

#[test]
fn parses_pg_v3_minor_version_startup_packet() {
    let protocol_version = (PG_PROTOCOL_MAJOR_V3 << 16) | 2;
    let mut payload = protocol_version.to_be_bytes().to_vec();
    payload.extend_from_slice(b"user\0postgres\0\0");
    let frame = with_length_prefix(payload);

    let packet = parse_startup_packet(&frame).unwrap();
    assert_eq!(
        packet,
        StartupPacket::Startup {
            protocol_version,
            params: vec![("user".to_string(), "postgres".to_string())],
        }
    );
}

#[test]
fn rejects_ssl_gssenc_and_cancel_requests_with_invalid_lengths() {
    let mut ssl_payload = PG_SSL_REQUEST_CODE.to_be_bytes().to_vec();
    ssl_payload.push(0);
    let ssl = with_length_prefix(ssl_payload);
    assert_eq!(
        parse_startup_packet(&ssl).unwrap_err(),
        StartupPacketError::LengthMismatch {
            expected: 8,
            actual: 9,
        }
    );

    let mut gssenc_payload = PG_GSSENC_REQUEST_CODE.to_be_bytes().to_vec();
    gssenc_payload.push(0);
    let gssenc = with_length_prefix(gssenc_payload);
    assert_eq!(
        parse_startup_packet(&gssenc).unwrap_err(),
        StartupPacketError::LengthMismatch {
            expected: 8,
            actual: 9,
        }
    );

    let cancel = with_length_prefix(PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec());
    assert_eq!(
        parse_startup_packet(&cancel).unwrap_err(),
        StartupPacketError::LengthMismatch {
            expected: 16,
            actual: 8,
        }
    );

    let mut oversized_cancel_payload = PG_CANCEL_REQUEST_CODE.to_be_bytes().to_vec();
    oversized_cancel_payload.extend_from_slice(&123u32.to_be_bytes());
    oversized_cancel_payload.extend_from_slice(&vec![0xAB; PG_CANCEL_SECRET_KEY_MAX_BYTES + 1]);
    let oversized_cancel = with_length_prefix(oversized_cancel_payload);
    assert_eq!(
        parse_startup_packet(&oversized_cancel).unwrap_err(),
        StartupPacketError::InvalidCancelKeyLength {
            actual: PG_CANCEL_SECRET_KEY_MAX_BYTES + 1,
            max: PG_CANCEL_SECRET_KEY_MAX_BYTES,
        }
    );
}

#[test]
fn rejects_startup_packet_with_length_or_parameter_errors() {
    let short = vec![0, 0, 0, 8, 0, 3, 0];
    assert_eq!(
        parse_startup_packet(&short).unwrap_err(),
        StartupPacketError::TooShort
    );

    let bad_len = vec![0, 0, 0, 10, 0, 3, 0, 0, 0];
    assert_eq!(
        parse_startup_packet(&bad_len).unwrap_err(),
        StartupPacketError::LengthMismatch {
            expected: 10,
            actual: 9,
        }
    );

    let invalid_length_field = vec![0, 0, 0, 4, 0, 3, 0, 0];
    assert_eq!(
        parse_startup_packet(&invalid_length_field).unwrap_err(),
        StartupPacketError::InvalidLengthField { declared: 4 }
    );

    let unsupported_protocol_code = vec![0, 0, 0, 8, 0x12, 0x34, 0x56, 0x78];
    assert_eq!(
        parse_startup_packet(&unsupported_protocol_code).unwrap_err(),
        StartupPacketError::UnsupportedProtocolCode(0x1234_5678)
    );

    let empty_param_payload = with_length_prefix(PG_PROTOCOL_V3.to_be_bytes().to_vec());
    assert_eq!(
        parse_startup_packet(&empty_param_payload).unwrap_err(),
        StartupPacketError::UnterminatedParameterPayload
    );

    let mut double_null_empty_param_payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    double_null_empty_param_payload.extend_from_slice(b"\0\0");
    let double_null_empty_param_payload = with_length_prefix(double_null_empty_param_payload);
    assert_eq!(
        parse_startup_packet(&double_null_empty_param_payload).unwrap_err(),
        StartupPacketError::InvalidParameterPairing
    );

    let mut bad_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    bad_params.extend_from_slice(b"user\0postgres");
    let bad_params = with_length_prefix(bad_params);
    assert_eq!(
        parse_startup_packet(&bad_params).unwrap_err(),
        StartupPacketError::UnterminatedParameterPayload
    );

    let mut extra_trailing_terminator_payload = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    extra_trailing_terminator_payload.extend_from_slice(b"user\0postgres\0\0\0");
    let extra_trailing_terminator_payload = with_length_prefix(extra_trailing_terminator_payload);
    assert_eq!(
        parse_startup_packet(&extra_trailing_terminator_payload).unwrap_err(),
        StartupPacketError::InvalidParameterPairing
    );

    let mut invalid_utf8_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    invalid_utf8_params.extend_from_slice(&[0xFF, 0, b'v', 0, 0]);
    let invalid_utf8_params = with_length_prefix(invalid_utf8_params);
    assert_eq!(
        parse_startup_packet(&invalid_utf8_params).unwrap_err(),
        StartupPacketError::InvalidUtf8
    );

    let mut invalid_utf8_value_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    invalid_utf8_value_params.extend_from_slice(b"user\0");
    invalid_utf8_value_params.extend_from_slice(&[0xFF, 0, 0]);
    let invalid_utf8_value_params = with_length_prefix(invalid_utf8_value_params);
    assert_eq!(
        parse_startup_packet(&invalid_utf8_value_params).unwrap_err(),
        StartupPacketError::InvalidUtf8
    );

    let mut empty_key_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    empty_key_params.extend_from_slice(b"\0value\0\0");
    let empty_key_params = with_length_prefix(empty_key_params);
    assert_eq!(
        parse_startup_packet(&empty_key_params).unwrap_err(),
        StartupPacketError::InvalidParameterPairing
    );

    let mut dangling_key_params = PG_PROTOCOL_V3.to_be_bytes().to_vec();
    dangling_key_params.extend_from_slice(b"user\0postgres\0database\0\0");
    let dangling_key_params = with_length_prefix(dangling_key_params);
    assert_eq!(
        parse_startup_packet(&dangling_key_params).unwrap_err(),
        StartupPacketError::InvalidParameterPairing
    );
}
