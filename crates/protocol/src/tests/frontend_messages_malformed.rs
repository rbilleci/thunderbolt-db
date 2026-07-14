//! Malformed PostgreSQL frontend-message framing and payload coverage.

use super::*;

#[test]
fn rejects_malformed_frontend_frames() {
    assert_eq!(
        parse_frontend_message(&[b'Q', 0, 0, 0]).unwrap_err(),
        FrontendMessageError::TooShort
    );

    let bad_len = vec![b'Q', 0, 0, 0, 7, b';', 0];
    assert_eq!(
        parse_frontend_message(&bad_len).unwrap_err(),
        FrontendMessageError::LengthMismatch {
            expected: 8,
            actual: 7,
        }
    );

    let invalid_length_field = vec![b'Q', 0, 0, 0, 3];
    assert_eq!(
        parse_frontend_message(&invalid_length_field).unwrap_err(),
        FrontendMessageError::InvalidLengthField { declared: 3 }
    );

    let unterminated = frontend_frame(b'Q', b"SELECT 1;");
    assert_eq!(
        parse_frontend_message(&unterminated).unwrap_err(),
        FrontendMessageError::UnterminatedSimpleQuery
    );

    let query_with_embedded_null = frontend_frame(b'Q', b"SELECT\0 1;\0");
    assert_eq!(
        parse_frontend_message(&query_with_embedded_null).unwrap_err(),
        FrontendMessageError::UnterminatedSimpleQuery
    );

    let password_like_sasl_response = frontend_frame(b'p', b"secret");
    assert_eq!(
        parse_frontend_message(&password_like_sasl_response).unwrap(),
        FrontendMessage::SaslResponse(b"secret".to_vec())
    );

    let binary_sasl_response = frontend_frame(b'p', &[0xFF, 0xFE]);
    assert_eq!(
        parse_frontend_message(&binary_sasl_response).unwrap(),
        FrontendMessage::SaslResponse(vec![0xFF, 0xFE])
    );

    let mixed_utf8_binary_sasl_response = frontend_frame(b'p', &[b'p', b'r', 0xC3, 0xB8, 0xFF]);
    assert_eq!(
        parse_frontend_message(&mixed_utf8_binary_sasl_response).unwrap(),
        FrontendMessage::SaslResponse(vec![b'p', b'r', 0xC3, 0xB8, 0xFF])
    );

    let unsupported_frontend_message = frontend_frame(b'Z', b"");
    assert_eq!(
        parse_frontend_message(&unsupported_frontend_message).unwrap_err(),
        FrontendMessageError::UnsupportedTag(b'Z')
    );

    let malformed_sasl_initial = frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let empty_mechanism_sasl_initial = frontend_frame(b'p', b"\0\xff\xff\xff\xff");
    assert_eq!(
        parse_frontend_message(&empty_mechanism_sasl_initial).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let invalid_utf8_mechanism_sasl_initial = frontend_frame(b'p', &[0xFF, 0, 0, 0, 0, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_mechanism_sasl_initial).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let malformed_sasl_initial_negative_len = frontend_frame(
        b'p',
        &[b'S', b'C', b'R', b'A', b'M', 0, 0xFF, 0xFF, 0xFF, 0xFE],
    );
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_negative_len).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_sasl_initial_null_len_with_trailing_payload =
        frontend_frame(b'p', b"SCRAM-SHA-256\0\xFF\xFF\xFF\xFFx");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_null_len_with_trailing_payload).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_sasl_initial_declared_len_mismatch =
        frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0\0\x03xy");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_declared_len_mismatch).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_sasl_initial_negative_non_null_len =
        frontend_frame(b'p', b"SCRAM-SHA-256\0\xFF\xFF\xFF\xFE");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_negative_non_null_len).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_sasl_initial_empty_mechanism = frontend_frame(b'p', b"\0\0\0\0\0");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_empty_mechanism).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_sasl_initial_zero_len_with_trailing_payload =
        frontend_frame(b'p', b"SCRAM-SHA-256\0\0\0\0\0x");
    assert_eq!(
        parse_frontend_message(&malformed_sasl_initial_zero_len_with_trailing_payload).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let malformed_parse = frontend_frame(b'P', b"stmt\0SELECT 1\0\0\x01");
    assert_eq!(
        parse_frontend_message(&malformed_parse).unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let unterminated_parse_statement_name = frontend_frame(b'P', b"stmt");
    assert_eq!(
        parse_frontend_message(&unterminated_parse_statement_name).unwrap_err(),
        FrontendMessageError::UnterminatedParseStatementName
    );

    let unterminated_parse_query = frontend_frame(b'P', b"stmt\0SELECT 1");
    assert_eq!(
        parse_frontend_message(&unterminated_parse_query).unwrap_err(),
        FrontendMessageError::UnterminatedParseQuery
    );

    let truncated_parse_type_count_field = frontend_frame(b'P', b"stmt\0SELECT 1\0\0");
    assert_eq!(
        parse_frontend_message(&truncated_parse_type_count_field).unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let mut truncated_parse_parameter_oid_field_payload = Vec::new();
    truncated_parse_parameter_oid_field_payload.extend_from_slice(b"stmt\0SELECT 1\0");
    truncated_parse_parameter_oid_field_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_parse_parameter_oid_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
    let truncated_parse_parameter_oid_field =
        frontend_frame(b'P', &truncated_parse_parameter_oid_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_parse_parameter_oid_field).unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let mut malformed_bind_with_truncated_result_format_codes_payload = Vec::new();
    malformed_bind_with_truncated_result_format_codes_payload.extend_from_slice(b"\0\0");
    malformed_bind_with_truncated_result_format_codes_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_truncated_result_format_codes_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_truncated_result_format_codes_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    malformed_bind_with_truncated_result_format_codes_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let malformed_bind_with_truncated_result_format_codes = frontend_frame(
        b'B',
        &malformed_bind_with_truncated_result_format_codes_payload,
    );
    assert_eq!(
        parse_frontend_message(&malformed_bind_with_truncated_result_format_codes).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut malformed_bind_with_zero_params_and_invalid_shared_format_payload = Vec::new();
    malformed_bind_with_zero_params_and_invalid_shared_format_payload.extend_from_slice(b"\0\0");
    malformed_bind_with_zero_params_and_invalid_shared_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    malformed_bind_with_zero_params_and_invalid_shared_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    malformed_bind_with_zero_params_and_invalid_shared_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_zero_params_and_invalid_shared_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let malformed_bind_with_zero_params_and_invalid_shared_format = frontend_frame(
        b'B',
        &malformed_bind_with_zero_params_and_invalid_shared_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&malformed_bind_with_zero_params_and_invalid_shared_format)
            .unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let invalid_utf8_parse_statement_name = frontend_frame(
        b'P',
        &[
            0xFF, 0, b'S', b'E', b'L', b'E', b'C', b'T', b' ', b'1', 0, 0, 0,
        ],
    );
    assert_eq!(
        parse_frontend_message(&invalid_utf8_parse_statement_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_parse_query =
        frontend_frame(b'P', &[b's', b't', b'm', b't', 0, 0xFF, 0, 0, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_parse_query).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let malformed_parse_negative_type_count = frontend_frame(
        b'P',
        &[
            b's', b't', b'm', b't', 0, b'S', b'E', b'L', b'E', b'C', b'T', b' ', b'1', 0, 0xFF,
            0xFF,
        ],
    );
    assert_eq!(
        parse_frontend_message(&malformed_parse_negative_type_count).unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let malformed_parse_with_zero_type_count_and_trailing_bytes =
        frontend_frame(b'P', b"stmt\0SELECT 1\0\0\0\xFF");
    assert_eq!(
        parse_frontend_message(&malformed_parse_with_zero_type_count_and_trailing_bytes)
            .unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let mut malformed_parse_with_trailing_oid_bytes_payload = Vec::new();
    malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(b"stmt\0SELECT 1\0");
    malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(&1_i16.to_be_bytes());
    malformed_parse_with_trailing_oid_bytes_payload.extend_from_slice(&23_u32.to_be_bytes());
    malformed_parse_with_trailing_oid_bytes_payload.push(0xFF);
    let malformed_parse_with_trailing_oid_bytes =
        frontend_frame(b'P', &malformed_parse_with_trailing_oid_bytes_payload);
    assert_eq!(
        parse_frontend_message(&malformed_parse_with_trailing_oid_bytes).unwrap_err(),
        FrontendMessageError::InvalidParseParameterPayload
    );

    let empty_describe = frontend_frame(b'D', b"");
    assert_eq!(
        parse_frontend_message(&empty_describe).unwrap_err(),
        FrontendMessageError::TooShort
    );

    let truncated_describe_name = frontend_frame(b'D', b"S");
    assert_eq!(
        parse_frontend_message(&truncated_describe_name).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let truncated_describe_name_lowercase_target = frontend_frame(b'D', b"pportal");
    assert_eq!(
        parse_frontend_message(&truncated_describe_name_lowercase_target).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let invalid_describe_target = frontend_frame(b'D', b"Xstmt\0");
    assert_eq!(
        parse_frontend_message(&invalid_describe_target).unwrap_err(),
        FrontendMessageError::InvalidDescribeTarget
    );

    let describe_with_embedded_null = frontend_frame(b'D', b"Sstmt\0extra\0");
    assert_eq!(
        parse_frontend_message(&describe_with_embedded_null).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let invalid_utf8_describe_name = frontend_frame(b'D', b"S\xFF\0");
    assert_eq!(
        parse_frontend_message(&invalid_utf8_describe_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_describe_name_lowercase_target = frontend_frame(b'D', b"p\xFF\0");
    assert_eq!(
        parse_frontend_message(&invalid_utf8_describe_name_lowercase_target).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let describe_with_trailing_bytes_after_name = frontend_frame(b'D', b"Sstmt\0\xFF");
    assert_eq!(
        parse_frontend_message(&describe_with_trailing_bytes_after_name).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let describe_unnamed_with_trailing_bytes_after_name = frontend_frame(b'D', b"p\0\xFF");
    assert_eq!(
        parse_frontend_message(&describe_unnamed_with_trailing_bytes_after_name).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let describe_unnamed_with_extra_null_segment = frontend_frame(b'D', b"p\0\0");
    assert_eq!(
        parse_frontend_message(&describe_unnamed_with_extra_null_segment).unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let describe_unnamed_uppercase_with_trailing_bytes_after_name =
        frontend_frame(b'D', b"S\0\xFF");
    assert_eq!(
        parse_frontend_message(&describe_unnamed_uppercase_with_trailing_bytes_after_name)
            .unwrap_err(),
        FrontendMessageError::UnterminatedDescribeName
    );

    let empty_close = frontend_frame(b'C', b"");
    assert_eq!(
        parse_frontend_message(&empty_close).unwrap_err(),
        FrontendMessageError::TooShort
    );

    let truncated_close_name = frontend_frame(b'C', b"P");
    assert_eq!(
        parse_frontend_message(&truncated_close_name).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let truncated_close_name_lowercase_target = frontend_frame(b'C', b"sstmt");
    assert_eq!(
        parse_frontend_message(&truncated_close_name_lowercase_target).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let invalid_close_target = frontend_frame(b'C', b"Xstmt\0");
    assert_eq!(
        parse_frontend_message(&invalid_close_target).unwrap_err(),
        FrontendMessageError::InvalidCloseTarget
    );

    let close_with_embedded_null = frontend_frame(b'C', b"Sstmt\0extra\0");
    assert_eq!(
        parse_frontend_message(&close_with_embedded_null).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let invalid_utf8_close_name = frontend_frame(b'C', b"S\xFF\0");
    assert_eq!(
        parse_frontend_message(&invalid_utf8_close_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_close_name_lowercase_target = frontend_frame(b'C', b"s\xFF\0");
    assert_eq!(
        parse_frontend_message(&invalid_utf8_close_name_lowercase_target).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let close_with_trailing_bytes_after_name = frontend_frame(b'C', b"Pportal\0\xFF");
    assert_eq!(
        parse_frontend_message(&close_with_trailing_bytes_after_name).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let close_unnamed_with_trailing_bytes_after_name = frontend_frame(b'C', b"s\0\xFF");
    assert_eq!(
        parse_frontend_message(&close_unnamed_with_trailing_bytes_after_name).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let close_unnamed_with_extra_null_segment = frontend_frame(b'C', b"s\0\0");
    assert_eq!(
        parse_frontend_message(&close_unnamed_with_extra_null_segment).unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let close_unnamed_uppercase_with_trailing_bytes_after_name = frontend_frame(b'C', b"P\0\xFF");
    assert_eq!(
        parse_frontend_message(&close_unnamed_uppercase_with_trailing_bytes_after_name)
            .unwrap_err(),
        FrontendMessageError::UnterminatedCloseName
    );

    let malformed_execute = frontend_frame(b'E', b"portal\0\0\0");
    assert_eq!(
        parse_frontend_message(&malformed_execute).unwrap_err(),
        FrontendMessageError::InvalidExecutePayload
    );

    let unterminated_execute_portal_name = frontend_frame(b'E', b"portal");
    assert_eq!(
        parse_frontend_message(&unterminated_execute_portal_name).unwrap_err(),
        FrontendMessageError::UnterminatedExecutePortalName
    );

    let execute_with_short_max_rows_payload = frontend_frame(b'E', b"portal\0\0\0\x01");
    assert_eq!(
        parse_frontend_message(&execute_with_short_max_rows_payload).unwrap_err(),
        FrontendMessageError::InvalidExecutePayload
    );

    let invalid_utf8_execute_portal_name = frontend_frame(b'E', &[0xFF, 0, 0, 0, 0, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_execute_portal_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let execute_with_embedded_null_portal_name = frontend_frame(b'E', b"portal\0extra\0\0\0\0");
    assert_eq!(
        parse_frontend_message(&execute_with_embedded_null_portal_name).unwrap_err(),
        FrontendMessageError::InvalidExecutePayload
    );

    let mut negative_max_rows_execute_payload = Vec::new();
    negative_max_rows_execute_payload.extend_from_slice(b"portal\0");
    negative_max_rows_execute_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    let negative_max_rows_execute = frontend_frame(b'E', &negative_max_rows_execute_payload);
    assert_eq!(
        parse_frontend_message(&negative_max_rows_execute).unwrap_err(),
        FrontendMessageError::InvalidExecutePayload
    );

    let mut malformed_execute_with_trailing_bytes_payload = Vec::new();
    malformed_execute_with_trailing_bytes_payload.extend_from_slice(b"portal\0");
    malformed_execute_with_trailing_bytes_payload.extend_from_slice(&5_i32.to_be_bytes());
    malformed_execute_with_trailing_bytes_payload.push(0xAA);
    let malformed_execute_with_trailing_bytes =
        frontend_frame(b'E', &malformed_execute_with_trailing_bytes_payload);
    assert_eq!(
        parse_frontend_message(&malformed_execute_with_trailing_bytes).unwrap_err(),
        FrontendMessageError::InvalidExecutePayload
    );

    let malformed_bind = frontend_frame(b'B', b"portal\0stmt\0\0\x01");
    assert_eq!(
        parse_frontend_message(&malformed_bind).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let unterminated_bind_portal_name = frontend_frame(b'B', b"portal");
    assert_eq!(
        parse_frontend_message(&unterminated_bind_portal_name).unwrap_err(),
        FrontendMessageError::UnterminatedBindPortalName
    );

    let unterminated_bind_statement_name = frontend_frame(b'B', b"portal\0stmt");
    assert_eq!(
        parse_frontend_message(&unterminated_bind_statement_name).unwrap_err(),
        FrontendMessageError::UnterminatedBindStatementName
    );

    let mut negative_bind_result_format_code_payload = Vec::new();
    negative_bind_result_format_code_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    negative_bind_result_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_bind_result_format_code =
        frontend_frame(b'B', &negative_bind_result_format_code_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_result_format_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut negative_bind_result_format_count_payload = Vec::new();
    negative_bind_result_format_count_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_bind_result_format_count =
        frontend_frame(b'B', &negative_bind_result_format_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_result_format_count).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let invalid_utf8_bind_portal_name =
        frontend_frame(b'B', &[0xFF, 0, b's', b't', b'm', b't', 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_bind_portal_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_bind_statement_name = frontend_frame(
        b'B',
        &[b'p', b'o', b'r', b't', b'a', b'l', 0, 0xFF, 0, 0, 0, 0, 0],
    );
    assert_eq!(
        parse_frontend_message(&invalid_utf8_bind_statement_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let mut negative_bind_format_count_payload = Vec::new();
    negative_bind_format_count_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_bind_format_count = frontend_frame(b'B', &negative_bind_format_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_format_count).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut invalid_bind_format_count_payload = Vec::new();
    invalid_bind_format_count_payload.extend_from_slice(b"portal\0stmt\0");
    invalid_bind_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_bind_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_bind_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_bind_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_bind_format_count_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    invalid_bind_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_bind_format_count = frontend_frame(b'B', &invalid_bind_format_count_payload);
    assert_eq!(
        parse_frontend_message(&invalid_bind_format_count).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut invalid_bind_format_count_with_zero_parameters_payload = Vec::new();
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(b"portal\0stmt\0");
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_bind_format_count_with_zero_parameters_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_bind_format_count_with_zero_parameters = frontend_frame(
        b'B',
        &invalid_bind_format_count_with_zero_parameters_payload,
    );
    assert_eq!(
        parse_frontend_message(&invalid_bind_format_count_with_zero_parameters).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut invalid_bind_format_code_payload = Vec::new();
    invalid_bind_format_code_payload.extend_from_slice(b"portal\0stmt\0");
    invalid_bind_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_bind_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_bind_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_bind_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_bind_format_code = frontend_frame(b'B', &invalid_bind_format_code_payload);
    assert_eq!(
        parse_frontend_message(&invalid_bind_format_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut invalid_bind_shared_format_with_multiple_parameters_payload = Vec::new();
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(b"portal\0stmt\0");
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&42_i32.to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    invalid_bind_shared_format_with_multiple_parameters_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let invalid_bind_shared_format_with_multiple_parameters = frontend_frame(
        b'B',
        &invalid_bind_shared_format_with_multiple_parameters_payload,
    );
    assert_eq!(
        parse_frontend_message(&invalid_bind_shared_format_with_multiple_parameters).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut bind_with_multiple_result_formats_payload = Vec::new();
    bind_with_multiple_result_formats_payload.extend_from_slice(b"portal\0stmt\0");
    bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    bind_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_multiple_result_formats =
        frontend_frame(b'B', &bind_with_multiple_result_formats_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_multiple_result_formats).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal".to_string(),
            statement_name: "stmt".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_with_zero_parameters_and_single_format_payload = Vec::new();
    bind_with_zero_parameters_and_single_format_payload.extend_from_slice(b"portal\0stmt\0");
    bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_zero_parameters_and_single_format =
        frontend_frame(b'B', &bind_with_zero_parameters_and_single_format_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_zero_parameters_and_single_format).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal".to_string(),
            statement_name: "stmt".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![],
        }
    );

    let mut bind_with_zero_parameters_and_single_text_format_payload = Vec::new();
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(b"portal_text\0stmt_text\0");
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_parameters_and_single_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_zero_parameters_and_single_text_format = frontend_frame(
        b'B',
        &bind_with_zero_parameters_and_single_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_with_zero_parameters_and_single_text_format).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_text".to_string(),
            statement_name: "stmt_text".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload = Vec::new();
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(b"\0\0");
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(b"\x00\x00\x00\x2a");
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_shared_parameter_format_for_multiple_parameters = frontend_frame(
        b'B',
        &bind_unnamed_with_shared_parameter_format_for_multiple_parameters_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_shared_parameter_format_for_multiple_parameters)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_shared_parameter_format_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice("pörtal_shared\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(b"\x00\x00\x00\x2a");
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_shared_parameter_format = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_shared_parameter_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_shared_parameter_format,)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_shared".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_shared_parameter_format_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(b"\0stmt_shared_named\0");
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(b"\x00\x00\x00\x2a");
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_shared_parameter_format = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_shared_parameter_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_shared_parameter_format,)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_shared_named".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_named_portal_with_named_statement_shared_parameter_format_payload = Vec::new();
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice("pörtal_shared_pair\0stmt_shared_pair\0".as_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(b"\x00\x00\x00\x2a");
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_parameter_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_shared_parameter_format = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_shared_parameter_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_shared_parameter_format,)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_shared_pair".to_string(),
            statement_name: "stmt_shared_pair".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_unnamed_with_shared_text_format_for_multiple_parameters_payload = Vec::new();
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice("héllo".as_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_shared_text_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_shared_text_format_for_multiple_parameters = frontend_frame(
        b'B',
        &bind_unnamed_with_shared_text_format_for_multiple_parameters_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_shared_text_format_for_multiple_parameters)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec()), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_shared_text_format_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice("pörtal_shared_text\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice("héllo".as_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_shared_text_format = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_shared_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_shared_text_format)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_shared_text".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec()), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_shared_text_format_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(b"\0stmt_shared_text_named\0");
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice("héllo".as_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_shared_text_format = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_shared_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_shared_text_format)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_shared_text_named".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec()), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_named_portal_with_named_statement_shared_text_format_payload = Vec::new();
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice("pörtal_shared_text_pair\0stmt_shared_text_pair\0".as_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice("héllo".as_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_shared_text_format = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_shared_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_shared_text_format).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_shared_text_pair".to_string(),
            statement_name: "stmt_shared_text_pair".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec()), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_with_shared_parameter_format_for_multiple_parameters_payload = Vec::new();
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(b"portal_shared\0stmt_shared\0");
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(b"\x00\x00\x00\x2a");
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_shared_parameter_format_for_multiple_parameters_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_shared_parameter_format_for_multiple_parameters = frontend_frame(
        b'B',
        &bind_with_shared_parameter_format_for_multiple_parameters_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_with_shared_parameter_format_for_multiple_parameters).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_shared".to_string(),
            statement_name: "stmt_shared".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0x00, 0x00, 0x2a]), None],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_with_multiple_parameter_formats_and_invalid_code_payload = Vec::new();
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(b"portal\0stmt\0");
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_parameter_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_multiple_parameter_formats_and_invalid_code = frontend_frame(
        b'B',
        &bind_with_multiple_parameter_formats_and_invalid_code_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_with_multiple_parameter_formats_and_invalid_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut bind_with_negative_parameter_format_code_payload = Vec::new();
    bind_with_negative_parameter_format_code_payload.extend_from_slice(b"portal\0stmt\0");
    bind_with_negative_parameter_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_negative_parameter_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    bind_with_negative_parameter_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_negative_parameter_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_negative_parameter_format_code =
        frontend_frame(b'B', &bind_with_negative_parameter_format_code_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_negative_parameter_format_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut bind_with_invalid_single_result_format_code_payload = Vec::new();
    bind_with_invalid_single_result_format_code_payload.extend_from_slice(b"portal\0stmt\0");
    bind_with_invalid_single_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_invalid_single_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_invalid_single_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_invalid_single_result_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    let bind_with_invalid_single_result_format_code =
        frontend_frame(b'B', &bind_with_invalid_single_result_format_code_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_invalid_single_result_format_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut bind_with_multiple_result_formats_and_invalid_code_payload = Vec::new();
    bind_with_multiple_result_formats_and_invalid_code_payload.extend_from_slice(b"portal\0stmt\0");
    bind_with_multiple_result_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_and_invalid_code_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_with_multiple_result_formats_and_invalid_code_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiple_result_formats_and_invalid_code_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    let bind_with_multiple_result_formats_and_invalid_code = frontend_frame(
        b'B',
        &bind_with_multiple_result_formats_and_invalid_code_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_with_multiple_result_formats_and_invalid_code).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_result_format_payload = Vec::new();
    truncated_bind_result_format_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    let truncated_bind_result_format = frontend_frame(b'B', &truncated_bind_result_format_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_result_format).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_result_format_codes_payload = Vec::new();
    truncated_bind_result_format_codes_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
    truncated_bind_result_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_codes_payload.push(0x00);
    let truncated_bind_result_format_codes =
        frontend_frame(b'B', &truncated_bind_result_format_codes_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_result_format_codes).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_format_count_field_payload = Vec::new();
    truncated_bind_format_count_field_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_format_count_field_payload.push(0x00);
    let truncated_bind_format_count_field =
        frontend_frame(b'B', &truncated_bind_format_count_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_format_count_field).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_parameter_format_codes_payload = Vec::new();
    truncated_bind_parameter_format_codes_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_parameter_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
    truncated_bind_parameter_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
    let truncated_bind_parameter_format_codes =
        frontend_frame(b'B', &truncated_bind_parameter_format_codes_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_parameter_format_codes).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_parameter_count_field_payload = Vec::new();
    truncated_bind_parameter_count_field_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_parameter_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_parameter_count_field_payload.push(0x00);
    let truncated_bind_parameter_count_field =
        frontend_frame(b'B', &truncated_bind_parameter_count_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_parameter_count_field).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut negative_bind_parameter_count_payload = Vec::new();
    negative_bind_parameter_count_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_parameter_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_parameter_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_bind_parameter_count =
        frontend_frame(b'B', &negative_bind_parameter_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_parameter_count).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut negative_bind_parameter_length_payload = Vec::new();
    negative_bind_parameter_length_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_parameter_length_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_parameter_length_payload.extend_from_slice(&1_i16.to_be_bytes());
    negative_bind_parameter_length_payload.extend_from_slice(&(-2_i32).to_be_bytes());
    negative_bind_parameter_length_payload.extend_from_slice(&0_i16.to_be_bytes());
    let negative_bind_parameter_length =
        frontend_frame(b'B', &negative_bind_parameter_length_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_parameter_length).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_parameter_length_field_payload = Vec::new();
    truncated_bind_parameter_length_field_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_parameter_length_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_parameter_length_field_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_bind_parameter_length_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
    let truncated_bind_parameter_length_field =
        frontend_frame(b'B', &truncated_bind_parameter_length_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_parameter_length_field).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_parameter_value_payload = Vec::new();
    truncated_bind_parameter_value_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_parameter_value_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_parameter_value_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_bind_parameter_value_payload.extend_from_slice(&3_i32.to_be_bytes());
    truncated_bind_parameter_value_payload.extend_from_slice(b"ab");
    let truncated_bind_parameter_value =
        frontend_frame(b'B', &truncated_bind_parameter_value_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_parameter_value).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut truncated_bind_result_format_count_field_payload = Vec::new();
    truncated_bind_result_format_count_field_payload.extend_from_slice(b"portal\0stmt\0");
    truncated_bind_result_format_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_bind_result_format_count_field_payload.push(0x00);
    let truncated_bind_result_format_count_field =
        frontend_frame(b'B', &truncated_bind_result_format_count_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_bind_result_format_count_field).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut negative_bind_result_format_count_payload = Vec::new();
    negative_bind_result_format_count_payload.extend_from_slice(b"portal\0stmt\0");
    negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_bind_result_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_bind_result_format_count =
        frontend_frame(b'B', &negative_bind_result_format_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_bind_result_format_count).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let mut malformed_bind_with_trailing_bytes_payload = Vec::new();
    malformed_bind_with_trailing_bytes_payload.extend_from_slice(b"portal\0stmt\0");
    malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_bind_with_trailing_bytes_payload.push(0x7F);
    let malformed_bind_with_trailing_bytes =
        frontend_frame(b'B', &malformed_bind_with_trailing_bytes_payload);
    assert_eq!(
        parse_frontend_message(&malformed_bind_with_trailing_bytes).unwrap_err(),
        FrontendMessageError::InvalidBindPayload
    );

    let truncated_function_call_oid_field = frontend_frame(b'F', &[0x00, 0x00, 0x00]);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_oid_field).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let malformed_function_call = frontend_frame(b'F', b"\0\0\0*");
    assert_eq!(
        parse_frontend_message(&malformed_function_call).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_result_format_payload = Vec::new();
    truncated_function_call_result_format_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_result_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    let truncated_function_call_result_format =
        frontend_frame(b'F', &truncated_function_call_result_format_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_result_format).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_result_format_count_field_payload = Vec::new();
    truncated_function_call_result_format_count_field_payload
        .extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_result_format_count_field_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_result_format_count_field_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_result_format_count_field_payload.push(0x00);
    let truncated_function_call_result_format_count_field = frontend_frame(
        b'F',
        &truncated_function_call_result_format_count_field_payload,
    );
    assert_eq!(
        parse_frontend_message(&truncated_function_call_result_format_count_field).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_format_count_payload = Vec::new();
    negative_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_function_call_format_count =
        frontend_frame(b'F', &negative_function_call_format_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_format_count).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_arg_count_field_payload = Vec::new();
    truncated_function_call_arg_count_field_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_arg_count_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_arg_count_field_payload.push(0x00);
    let truncated_function_call_arg_count_field =
        frontend_frame(b'F', &truncated_function_call_arg_count_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_arg_count_field).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_shared_format_code_payload = Vec::new();
    negative_function_call_shared_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_shared_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    negative_function_call_shared_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    negative_function_call_shared_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_shared_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    let negative_function_call_shared_format_code =
        frontend_frame(b'F', &negative_function_call_shared_format_code_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_shared_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut invalid_function_call_format_code_payload = Vec::new();
    invalid_function_call_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    invalid_function_call_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_function_call_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_function_call_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_function_call_format_code =
        frontend_frame(b'F', &invalid_function_call_format_code_payload);
    assert_eq!(
        parse_frontend_message(&invalid_function_call_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut mismatched_function_call_format_count_payload = Vec::new();
    mismatched_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&4_i32.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&7_i32.to_be_bytes());
    mismatched_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    let mismatched_function_call_format_count =
        frontend_frame(b'F', &mismatched_function_call_format_count_payload);
    assert_eq!(
        parse_frontend_message(&mismatched_function_call_format_count).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_format_codes_payload = Vec::new();
    truncated_function_call_format_codes_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_format_codes_payload.extend_from_slice(&2_i16.to_be_bytes());
    truncated_function_call_format_codes_payload.extend_from_slice(&0_i16.to_be_bytes());
    let truncated_function_call_format_codes =
        frontend_frame(b'F', &truncated_function_call_format_codes_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_format_codes).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_argument_value_payload = Vec::new();
    truncated_function_call_argument_value_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_argument_value_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_argument_value_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_function_call_argument_value_payload.extend_from_slice(&3_i32.to_be_bytes());
    truncated_function_call_argument_value_payload.extend_from_slice(b"ab");
    let truncated_function_call_argument_value =
        frontend_frame(b'F', &truncated_function_call_argument_value_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_argument_value).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut invalid_function_call_multi_format_code_payload = Vec::new();
    invalid_function_call_multi_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_multi_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_function_call_multi_format_code =
        frontend_frame(b'F', &invalid_function_call_multi_format_code_payload);
    assert_eq!(
        parse_frontend_message(&invalid_function_call_multi_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut invalid_function_call_format_count_payload = Vec::new();
    invalid_function_call_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    invalid_function_call_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_function_call_format_count =
        frontend_frame(b'F', &invalid_function_call_format_count_payload);
    assert_eq!(
        parse_frontend_message(&invalid_function_call_format_count).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut invalid_function_call_zero_arg_multi_format_payload = Vec::new();
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&42_u32.to_be_bytes());
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&2_i16.to_be_bytes());
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_zero_arg_multi_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    let invalid_function_call_zero_arg_multi_format =
        frontend_frame(b'F', &invalid_function_call_zero_arg_multi_format_payload);
    assert_eq!(
        parse_frontend_message(&invalid_function_call_zero_arg_multi_format).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_arg_count_payload = Vec::new();
    negative_function_call_arg_count_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_arg_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_arg_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_function_call_arg_count =
        frontend_frame(b'F', &negative_function_call_arg_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_arg_count).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_arg_length_payload = Vec::new();
    negative_function_call_arg_length_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_arg_length_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_arg_length_payload.extend_from_slice(&1_i16.to_be_bytes());
    negative_function_call_arg_length_payload.extend_from_slice(&(-2_i32).to_be_bytes());
    negative_function_call_arg_length_payload.extend_from_slice(&0_i16.to_be_bytes());
    let negative_function_call_arg_length =
        frontend_frame(b'F', &negative_function_call_arg_length_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_arg_length).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_arg_length_field_payload = Vec::new();
    truncated_function_call_arg_length_field_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_arg_length_field_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_arg_length_field_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_function_call_arg_length_field_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
    let truncated_function_call_arg_length_field =
        frontend_frame(b'F', &truncated_function_call_arg_length_field_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_arg_length_field).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_arg_value_payload = Vec::new();
    truncated_function_call_arg_value_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_arg_value_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_arg_value_payload.extend_from_slice(&1_i16.to_be_bytes());
    truncated_function_call_arg_value_payload.extend_from_slice(&3_i32.to_be_bytes());
    truncated_function_call_arg_value_payload.extend_from_slice(b"ab");
    truncated_function_call_arg_value_payload.extend_from_slice(&0_i16.to_be_bytes());
    let truncated_function_call_arg_value =
        frontend_frame(b'F', &truncated_function_call_arg_value_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_arg_value).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut malformed_function_call_with_trailing_bytes_payload = Vec::new();
    malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&42_u32.to_be_bytes());
    malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_function_call_with_trailing_bytes_payload.extend_from_slice(&0_i16.to_be_bytes());
    malformed_function_call_with_trailing_bytes_payload.push(0xAA);
    let malformed_function_call_with_trailing_bytes =
        frontend_frame(b'F', &malformed_function_call_with_trailing_bytes_payload);
    assert_eq!(
        parse_frontend_message(&malformed_function_call_with_trailing_bytes).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_result_format_count_payload = Vec::new();
    negative_function_call_result_format_count_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_result_format_count_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_result_format_count_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_function_call_result_format_count =
        frontend_frame(b'F', &negative_function_call_result_format_count_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_result_format_count).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut negative_function_call_result_format_code_payload = Vec::new();
    negative_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    negative_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    negative_function_call_result_format_code_payload.extend_from_slice(&1_i16.to_be_bytes());
    negative_function_call_result_format_code_payload.extend_from_slice(&(-1_i16).to_be_bytes());
    let negative_function_call_result_format_code =
        frontend_frame(b'F', &negative_function_call_result_format_code_payload);
    assert_eq!(
        parse_frontend_message(&negative_function_call_result_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut truncated_function_call_result_format_code_payload = Vec::new();
    truncated_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    truncated_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    truncated_function_call_result_format_code_payload.push(0x00);
    let truncated_function_call_result_format_code =
        frontend_frame(b'F', &truncated_function_call_result_format_code_payload);
    assert_eq!(
        parse_frontend_message(&truncated_function_call_result_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let mut invalid_function_call_result_format_code_payload = Vec::new();
    invalid_function_call_result_format_code_payload.extend_from_slice(&42_u32.to_be_bytes());
    invalid_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_result_format_code_payload.extend_from_slice(&0_i16.to_be_bytes());
    invalid_function_call_result_format_code_payload.extend_from_slice(&2_i16.to_be_bytes());
    let invalid_function_call_result_format_code =
        frontend_frame(b'F', &invalid_function_call_result_format_code_payload);
    assert_eq!(
        parse_frontend_message(&invalid_function_call_result_format_code).unwrap_err(),
        FrontendMessageError::InvalidFunctionCallPayload
    );

    let malformed_copy_done = frontend_frame(b'c', &[0]);
    assert_eq!(
        parse_frontend_message(&malformed_copy_done).unwrap_err(),
        FrontendMessageError::LengthMismatch {
            expected: 5,
            actual: 6,
        }
    );

    let malformed_terminate = frontend_frame(b'X', &[0]);
    assert_eq!(
        parse_frontend_message(&malformed_terminate).unwrap_err(),
        FrontendMessageError::LengthMismatch {
            expected: 5,
            actual: 6,
        }
    );

    let malformed_sync = frontend_frame(b'S', &[0]);
    assert_eq!(
        parse_frontend_message(&malformed_sync).unwrap_err(),
        FrontendMessageError::LengthMismatch {
            expected: 5,
            actual: 6,
        }
    );

    let malformed_flush = frontend_frame(b'H', &[0]);
    assert_eq!(
        parse_frontend_message(&malformed_flush).unwrap_err(),
        FrontendMessageError::LengthMismatch {
            expected: 5,
            actual: 6,
        }
    );

    let unterminated_copy_fail = frontend_frame(b'f', b"bad row");
    assert_eq!(
        parse_frontend_message(&unterminated_copy_fail).unwrap_err(),
        FrontendMessageError::UnterminatedCopyFail
    );

    let copy_fail_with_embedded_null = frontend_frame(b'f', b"bad\0row\0");
    assert_eq!(
        parse_frontend_message(&copy_fail_with_embedded_null).unwrap_err(),
        FrontendMessageError::UnterminatedCopyFail
    );

    let invalid_utf8_copy_fail = frontend_frame(b'f', &[0xFF, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_copy_fail).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_query = frontend_frame(b'Q', &[0xFF, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_query).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let password_with_embedded_null_and_terminator = frontend_frame(b'p', b"secret\0extra\0");
    assert_eq!(
        parse_frontend_message(&password_with_embedded_null_and_terminator).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let invalid_utf8_password = frontend_frame(b'p', &[0xFF, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_password).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let binary_sasl_response_with_embedded_null = frontend_frame(b'p', b"proof\0chunk");
    assert_eq!(
        parse_frontend_message(&binary_sasl_response_with_embedded_null).unwrap_err(),
        FrontendMessageError::InvalidSaslInitialResponsePayload
    );

    let invalid_utf8_parse_statement = frontend_frame(b'P', &[0xFF, 0, b'S', 0, 0, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_parse_statement).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let invalid_utf8_describe_name = frontend_frame(b'D', &[b'S', 0xFF, 0]);
    assert_eq!(
        parse_frontend_message(&invalid_utf8_describe_name).unwrap_err(),
        FrontendMessageError::InvalidUtf8
    );

    let unsupported = frontend_frame(b'V', &[]);
    assert_eq!(
        parse_frontend_message(&unsupported).unwrap_err(),
        FrontendMessageError::UnsupportedTag(b'V')
    );
}
