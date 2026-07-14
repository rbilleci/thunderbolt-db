//! Valid PostgreSQL frontend-message framing and payload coverage.

use super::*;

#[test]
fn parses_simple_query_and_control_frontend_messages() {
    let query = frontend_frame(b'Q', b"SELECT 1;\0");
    assert_eq!(
        parse_frontend_message(&query).unwrap(),
        FrontendMessage::SimpleQuery("SELECT 1;".to_string())
    );

    let empty_query = frontend_frame(b'Q', b"\0");
    assert_eq!(
        parse_frontend_message(&empty_query).unwrap(),
        FrontendMessage::SimpleQuery(String::new())
    );

    let utf8_query = frontend_frame(b'Q', "SELECT 'héllo';\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&utf8_query).unwrap(),
        FrontendMessage::SimpleQuery("SELECT 'héllo';".to_string())
    );

    let multiline_utf8_query = frontend_frame(b'Q', "SELECT 'héllo'\nFROM tést;\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&multiline_utf8_query).unwrap(),
        FrontendMessage::SimpleQuery("SELECT 'héllo'\nFROM tést;".to_string())
    );

    let password = frontend_frame(b'p', b"secret\0");
    assert_eq!(
        parse_frontend_message(&password).unwrap(),
        FrontendMessage::PasswordMessage("secret".to_string())
    );

    let empty_password = frontend_frame(b'p', b"\0");
    assert_eq!(
        parse_frontend_message(&empty_password).unwrap(),
        FrontendMessage::PasswordMessage(String::new())
    );

    let utf8_password = frontend_frame(b'p', "påsswörd\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&utf8_password).unwrap(),
        FrontendMessage::PasswordMessage("påsswörd".to_string())
    );

    let multiline_utf8_password = frontend_frame(b'p', "påss\nwördΔ\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&multiline_utf8_password).unwrap(),
        FrontendMessage::PasswordMessage("påss\nwördΔ".to_string())
    );

    let mut sasl_initial_payload = Vec::new();
    sasl_initial_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_payload.extend_from_slice(&5_i32.to_be_bytes());
    sasl_initial_payload.extend_from_slice(b"n,,r=");
    let sasl_initial = frontend_frame(b'p', &sasl_initial_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some(b"n,,r=".to_vec()),
        }
    );

    let mut sasl_initial_without_data_payload = Vec::new();
    sasl_initial_without_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_without_data_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    let sasl_initial_without_data = frontend_frame(b'p', &sasl_initial_without_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_without_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: None,
        }
    );

    let mut sasl_initial_with_empty_data_payload = Vec::new();
    sasl_initial_with_empty_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_with_empty_data_payload.extend_from_slice(&0_i32.to_be_bytes());
    let sasl_initial_with_empty_data = frontend_frame(b'p', &sasl_initial_with_empty_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_empty_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some(Vec::new()),
        }
    );

    let mut sasl_initial_with_binary_data_payload = Vec::new();
    sasl_initial_with_binary_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_with_binary_data_payload.extend_from_slice(&3_i32.to_be_bytes());
    sasl_initial_with_binary_data_payload.extend_from_slice(&[0x00, 0xFF, 0x00]);
    let sasl_initial_with_binary_data =
        frontend_frame(b'p', &sasl_initial_with_binary_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_binary_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some(vec![0x00, 0xFF, 0x00]),
        }
    );

    let mut sasl_initial_with_utf8_data_payload = Vec::new();
    sasl_initial_with_utf8_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_with_utf8_data_payload
        .extend_from_slice(&("n,,r=nönce".len() as i32).to_be_bytes());
    sasl_initial_with_utf8_data_payload.extend_from_slice("n,,r=nönce".as_bytes());
    let sasl_initial_with_utf8_data = frontend_frame(b'p', &sasl_initial_with_utf8_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some("n,,r=nönce".as_bytes().to_vec()),
        }
    );

    let mut sasl_initial_with_multiline_utf8_data_payload = Vec::new();
    sasl_initial_with_multiline_utf8_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_with_multiline_utf8_data_payload
        .extend_from_slice(&("n,,r=nönce\nΔetail".len() as i32).to_be_bytes());
    sasl_initial_with_multiline_utf8_data_payload
        .extend_from_slice("n,,r=nönce\nΔetail".as_bytes());
    let sasl_initial_with_multiline_utf8_data =
        frontend_frame(b'p', &sasl_initial_with_multiline_utf8_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_multiline_utf8_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some("n,,r=nönce\nΔetail".as_bytes().to_vec()),
        }
    );

    let mut sasl_initial_with_mixed_utf8_binary_data_payload = Vec::new();
    sasl_initial_with_mixed_utf8_binary_data_payload.extend_from_slice(b"SCRAM-SHA-256\0");
    sasl_initial_with_mixed_utf8_binary_data_payload.extend_from_slice(&5_i32.to_be_bytes());
    sasl_initial_with_mixed_utf8_binary_data_payload
        .extend_from_slice(&[b'n', 0xC3, 0xB8, 0xFF, b'!']);
    let sasl_initial_with_mixed_utf8_binary_data =
        frontend_frame(b'p', &sasl_initial_with_mixed_utf8_binary_data_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_mixed_utf8_binary_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some(vec![b'n', 0xC3, 0xB8, 0xFF, b'!']),
        }
    );

    let mut sasl_initial_with_utf8_mechanism_payload = Vec::new();
    sasl_initial_with_utf8_mechanism_payload.extend_from_slice("SCRÄM\0".as_bytes());
    sasl_initial_with_utf8_mechanism_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    let sasl_initial_with_utf8_mechanism =
        frontend_frame(b'p', &sasl_initial_with_utf8_mechanism_payload);
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_mechanism).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRÄM".to_string(),
            initial_response: None,
        }
    );

    let mut sasl_initial_with_utf8_mechanism_and_binary_data_payload = Vec::new();
    sasl_initial_with_utf8_mechanism_and_binary_data_payload
        .extend_from_slice("SCRÄM\0".as_bytes());
    sasl_initial_with_utf8_mechanism_and_binary_data_payload
        .extend_from_slice(&3_i32.to_be_bytes());
    sasl_initial_with_utf8_mechanism_and_binary_data_payload.extend_from_slice(&[0x00, 0xFF, 0x7F]);
    let sasl_initial_with_utf8_mechanism_and_binary_data = frontend_frame(
        b'p',
        &sasl_initial_with_utf8_mechanism_and_binary_data_payload,
    );
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_binary_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRÄM".to_string(),
            initial_response: Some(vec![0x00, 0xFF, 0x7F]),
        }
    );

    let mut sasl_initial_with_utf8_mechanism_and_empty_data_payload = Vec::new();
    sasl_initial_with_utf8_mechanism_and_empty_data_payload.extend_from_slice("SCRÄM\0".as_bytes());
    sasl_initial_with_utf8_mechanism_and_empty_data_payload.extend_from_slice(&0_i32.to_be_bytes());
    let sasl_initial_with_utf8_mechanism_and_empty_data = frontend_frame(
        b'p',
        &sasl_initial_with_utf8_mechanism_and_empty_data_payload,
    );
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_empty_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRÄM".to_string(),
            initial_response: Some(Vec::new()),
        }
    );

    let mut sasl_initial_with_utf8_mechanism_and_utf8_data_payload = Vec::new();
    sasl_initial_with_utf8_mechanism_and_utf8_data_payload.extend_from_slice("SCRÄM\0".as_bytes());
    sasl_initial_with_utf8_mechanism_and_utf8_data_payload
        .extend_from_slice(&("n,,r=nönce".len() as i32).to_be_bytes());
    sasl_initial_with_utf8_mechanism_and_utf8_data_payload
        .extend_from_slice("n,,r=nönce".as_bytes());
    let sasl_initial_with_utf8_mechanism_and_utf8_data = frontend_frame(
        b'p',
        &sasl_initial_with_utf8_mechanism_and_utf8_data_payload,
    );
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_utf8_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRÄM".to_string(),
            initial_response: Some("n,,r=nönce".as_bytes().to_vec()),
        }
    );

    let mut sasl_initial_with_utf8_mechanism_and_mixed_data_payload = Vec::new();
    sasl_initial_with_utf8_mechanism_and_mixed_data_payload.extend_from_slice("SCRÄM\0".as_bytes());
    sasl_initial_with_utf8_mechanism_and_mixed_data_payload.extend_from_slice(&5_i32.to_be_bytes());
    sasl_initial_with_utf8_mechanism_and_mixed_data_payload
        .extend_from_slice(&[b'n', 0xC3, 0xB8, 0xFF, b'!']);
    let sasl_initial_with_utf8_mechanism_and_mixed_data = frontend_frame(
        b'p',
        &sasl_initial_with_utf8_mechanism_and_mixed_data_payload,
    );
    assert_eq!(
        parse_frontend_message(&sasl_initial_with_utf8_mechanism_and_mixed_data).unwrap(),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRÄM".to_string(),
            initial_response: Some(vec![b'n', 0xC3, 0xB8, 0xFF, b'!']),
        }
    );

    let sasl_response = frontend_frame(b'p', b"c=biws,r=nonce,p=proof");
    assert_eq!(
        parse_frontend_message(&sasl_response).unwrap(),
        FrontendMessage::SaslResponse(b"c=biws,r=nonce,p=proof".to_vec())
    );

    let empty_sasl_response = frontend_frame(b'p', b"");
    assert_eq!(
        parse_frontend_message(&empty_sasl_response).unwrap(),
        FrontendMessage::SaslResponse(Vec::new())
    );

    let utf8_sasl_response = frontend_frame(b'p', "prøöf".as_bytes());
    assert_eq!(
        parse_frontend_message(&utf8_sasl_response).unwrap(),
        FrontendMessage::SaslResponse("prøöf".as_bytes().to_vec())
    );

    let multiline_utf8_sasl_response = frontend_frame(b'p', "c=biws\nr=noncé\np=prøöf".as_bytes());
    assert_eq!(
        parse_frontend_message(&multiline_utf8_sasl_response).unwrap(),
        FrontendMessage::SaslResponse("c=biws\nr=noncé\np=prøöf".as_bytes().to_vec())
    );

    let mut parse_payload = Vec::new();
    parse_payload.extend_from_slice(b"stmt1\0SELECT $1::int4\0");
    parse_payload.extend_from_slice(&1_i16.to_be_bytes());
    parse_payload.extend_from_slice(&23_u32.to_be_bytes());
    let parse = frontend_frame(b'P', &parse_payload);
    assert_eq!(
        parse_frontend_message(&parse).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt1".to_string(),
            query: "SELECT $1::int4".to_string(),
            parameter_type_oids: vec![23],
        }
    );

    let mut parse_unnamed_statement_payload = Vec::new();
    parse_unnamed_statement_payload.extend_from_slice(b"\0SELECT 1\0");
    parse_unnamed_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_unnamed_statement = frontend_frame(b'P', &parse_unnamed_statement_payload);
    assert_eq!(
        parse_frontend_message(&parse_unnamed_statement).unwrap(),
        FrontendMessage::Parse {
            statement_name: String::new(),
            query: "SELECT 1".to_string(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_unnamed_statement_with_parameter_oid_payload = Vec::new();
    parse_unnamed_statement_with_parameter_oid_payload.extend_from_slice(b"\0SELECT $1::text\0");
    parse_unnamed_statement_with_parameter_oid_payload.extend_from_slice(&1_i16.to_be_bytes());
    parse_unnamed_statement_with_parameter_oid_payload.extend_from_slice(&25_u32.to_be_bytes());
    let parse_unnamed_statement_with_parameter_oid =
        frontend_frame(b'P', &parse_unnamed_statement_with_parameter_oid_payload);
    assert_eq!(
        parse_frontend_message(&parse_unnamed_statement_with_parameter_oid).unwrap(),
        FrontendMessage::Parse {
            statement_name: String::new(),
            query: "SELECT $1::text".to_string(),
            parameter_type_oids: vec![25],
        }
    );

    let mut parse_utf8_query_payload = Vec::new();
    parse_utf8_query_payload.extend_from_slice("stmt_utf8\0SELECT 'héllo'\0".as_bytes());
    parse_utf8_query_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_utf8_query = frontend_frame(b'P', &parse_utf8_query_payload);
    assert_eq!(
        parse_frontend_message(&parse_utf8_query).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt_utf8".to_string(),
            query: "SELECT 'héllo'".to_string(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_empty_query_payload = Vec::new();
    parse_empty_query_payload.extend_from_slice(b"stmt_empty\0\0");
    parse_empty_query_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_empty_query = frontend_frame(b'P', &parse_empty_query_payload);
    assert_eq!(
        parse_frontend_message(&parse_empty_query).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt_empty".to_string(),
            query: String::new(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_unnamed_empty_query_payload = Vec::new();
    parse_unnamed_empty_query_payload.extend_from_slice(b"\0\0");
    parse_unnamed_empty_query_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_unnamed_empty_query = frontend_frame(b'P', &parse_unnamed_empty_query_payload);
    assert_eq!(
        parse_frontend_message(&parse_unnamed_empty_query).unwrap(),
        FrontendMessage::Parse {
            statement_name: String::new(),
            query: String::new(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_empty_query_with_parameter_oid_payload = Vec::new();
    parse_empty_query_with_parameter_oid_payload.extend_from_slice(b"stmt_empty_typed\0\0");
    parse_empty_query_with_parameter_oid_payload.extend_from_slice(&1_i16.to_be_bytes());
    parse_empty_query_with_parameter_oid_payload.extend_from_slice(&25_u32.to_be_bytes());
    let parse_empty_query_with_parameter_oid =
        frontend_frame(b'P', &parse_empty_query_with_parameter_oid_payload);
    assert_eq!(
        parse_frontend_message(&parse_empty_query_with_parameter_oid).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt_empty_typed".to_string(),
            query: String::new(),
            parameter_type_oids: vec![25],
        }
    );

    let mut parse_unnamed_empty_query_with_parameter_oid_payload = Vec::new();
    parse_unnamed_empty_query_with_parameter_oid_payload.extend_from_slice(b"\0\0");
    parse_unnamed_empty_query_with_parameter_oid_payload.extend_from_slice(&1_i16.to_be_bytes());
    parse_unnamed_empty_query_with_parameter_oid_payload.extend_from_slice(&25_u32.to_be_bytes());
    let parse_unnamed_empty_query_with_parameter_oid =
        frontend_frame(b'P', &parse_unnamed_empty_query_with_parameter_oid_payload);
    assert_eq!(
        parse_frontend_message(&parse_unnamed_empty_query_with_parameter_oid).unwrap(),
        FrontendMessage::Parse {
            statement_name: String::new(),
            query: String::new(),
            parameter_type_oids: vec![25],
        }
    );

    let mut parse_multiline_utf8_query_payload = Vec::new();
    parse_multiline_utf8_query_payload
        .extend_from_slice("stmt_utf8_multi\0SELECT 'héllo'\nFROM tést\0".as_bytes());
    parse_multiline_utf8_query_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_multiline_utf8_query = frontend_frame(b'P', &parse_multiline_utf8_query_payload);
    assert_eq!(
        parse_frontend_message(&parse_multiline_utf8_query).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt_utf8_multi".to_string(),
            query: "SELECT 'héllo'\nFROM tést".to_string(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_utf8_statement_name_payload = Vec::new();
    parse_utf8_statement_name_payload.extend_from_slice("stmté\0SELECT 42\0".as_bytes());
    parse_utf8_statement_name_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_utf8_statement_name = frontend_frame(b'P', &parse_utf8_statement_name_payload);
    assert_eq!(
        parse_frontend_message(&parse_utf8_statement_name).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmté".to_string(),
            query: "SELECT 42".to_string(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_multiline_utf8_statement_name_payload = Vec::new();
    parse_multiline_utf8_statement_name_payload
        .extend_from_slice("stmté\nΔetail\0SELECT 42\0".as_bytes());
    parse_multiline_utf8_statement_name_payload.extend_from_slice(&0_i16.to_be_bytes());
    let parse_multiline_utf8_statement_name =
        frontend_frame(b'P', &parse_multiline_utf8_statement_name_payload);
    assert_eq!(
        parse_frontend_message(&parse_multiline_utf8_statement_name).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmté\nΔetail".to_string(),
            query: "SELECT 42".to_string(),
            parameter_type_oids: vec![],
        }
    );

    let mut parse_utf8_statement_with_parameter_oids_payload = Vec::new();
    parse_utf8_statement_with_parameter_oids_payload
        .extend_from_slice("stmté\0SELECT $1::text, $2::bytea\0".as_bytes());
    parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&2_i16.to_be_bytes());
    parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&25_u32.to_be_bytes());
    parse_utf8_statement_with_parameter_oids_payload.extend_from_slice(&17_u32.to_be_bytes());
    let parse_utf8_statement_with_parameter_oids =
        frontend_frame(b'P', &parse_utf8_statement_with_parameter_oids_payload);
    assert_eq!(
        parse_frontend_message(&parse_utf8_statement_with_parameter_oids).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmté".to_string(),
            query: "SELECT $1::text, $2::bytea".to_string(),
            parameter_type_oids: vec![25, 17],
        }
    );

    let mut parse_utf8_query_with_parameter_oids_payload = Vec::new();
    parse_utf8_query_with_parameter_oids_payload
        .extend_from_slice("stmt_utf8\0SELECT 'héllo', $1::text\0".as_bytes());
    parse_utf8_query_with_parameter_oids_payload.extend_from_slice(&1_i16.to_be_bytes());
    parse_utf8_query_with_parameter_oids_payload.extend_from_slice(&25_u32.to_be_bytes());
    let parse_utf8_query_with_parameter_oids =
        frontend_frame(b'P', &parse_utf8_query_with_parameter_oids_payload);
    assert_eq!(
        parse_frontend_message(&parse_utf8_query_with_parameter_oids).unwrap(),
        FrontendMessage::Parse {
            statement_name: "stmt_utf8".to_string(),
            query: "SELECT 'héllo', $1::text".to_string(),
            parameter_type_oids: vec![25],
        }
    );

    let mut bind_payload = Vec::new();
    bind_payload.extend_from_slice(b"portal1\0stmt1\0");
    bind_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_payload.extend_from_slice(&2_i16.to_be_bytes());
    bind_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_payload.extend_from_slice(&42_i32.to_be_bytes());
    bind_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    bind_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind = frontend_frame(b'B', &bind_payload);
    assert_eq!(
        parse_frontend_message(&bind).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal1".to_string(),
            statement_name: "stmt1".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(42_i32.to_be_bytes().to_vec()), None],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_default_formats_payload = Vec::new();
    bind_unnamed_with_default_formats_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_default_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_with_default_formats =
        frontend_frame(b'B', &bind_unnamed_with_default_formats_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_default_formats).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(b"\0stmt_named\0");
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice("héllo".as_bytes());
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement =
        frontend_frame(b'B', &bind_unnamed_portal_with_named_statement_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_named".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_payload
        .extend_from_slice("pörtal_named\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE]);
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement =
        frontend_frame(b'B', &bind_named_portal_with_unnamed_statement_payload);
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_named".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0xCA, 0xFE, 0xBA, 0xBE])],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_text_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_text_payload
        .extend_from_slice("pörtal_text\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_text_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice("héllo".as_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_text_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_text =
        frontend_frame(b'B', &bind_named_portal_with_unnamed_statement_text_payload);
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_text".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_empty_text_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice("pörtal_empty\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_empty_text = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_empty_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_empty_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_empty".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_empty_binary_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice("pörtal_empty_bin\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_empty_binary = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_empty_binary_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_empty_binary).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_empty_bin".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_binary_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_binary_payload
        .extend_from_slice(b"\0stmt_bin_named\0");
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_portal_with_named_statement_binary_payload
        .extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_binary_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_binary = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_binary_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_binary).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_bin_named".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0xDE, 0xAD, 0xBE, 0xEF])],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_empty_text_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(b"\0stmt_empty_named\0");
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_empty_text = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_empty_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_empty_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_empty_named".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_empty_binary_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(b"\0stmt_empty_bin_named\0");
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_empty_binary = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_empty_binary_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_empty_binary).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_empty_bin_named".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_named_statement_empty_binary_payload = Vec::new();
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice("pörtal_pair\0stmt_pair\0".as_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_binary_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_empty_binary = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_empty_binary_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_empty_binary).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_pair".to_string(),
            statement_name: "stmt_pair".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_named_statement_empty_text_payload = Vec::new();
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice("pörtal_pair_text\0stmt_pair_text\0".as_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i32.to_be_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_empty_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_empty_text = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_empty_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_empty_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_pair_text".to_string(),
            statement_name: "stmt_pair_text".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_zero_params_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice("pörtal_zero\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_zero_params = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_zero_params_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_zero_params).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_zero".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_zero_params_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(b"\0stmt_zero_named\0");
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_zero_params = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_zero_params_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_zero_params).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_zero_named".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_named_statement_zero_params_payload = Vec::new();
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice("pörtal_zero_pair\0stmt_zero_pair\0".as_bytes());
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_zero_params = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_zero_params_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_zero_params).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_zero_pair".to_string(),
            statement_name: "stmt_zero_pair".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_zero_params_text_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice("pörtal_zero_text\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_zero_params_text = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_zero_params_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_zero_params_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_zero_text".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_zero_params_text_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(b"\0stmt_zero_text_named\0");
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_zero_params_text = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_zero_params_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_zero_params_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_zero_text_named".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![],
            result_format_codes: vec![0],
        }
    );

    let mut bind_named_portal_with_named_statement_zero_params_text_payload = Vec::new();
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice("pörtal_zero_text_pair\0stmt_zero_text_pair\0".as_bytes());
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_zero_params_text_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_zero_params_text = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_zero_params_text_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_zero_params_text).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_zero_text_pair".to_string(),
            statement_name: "stmt_zero_text_pair".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_zero_params_and_single_shared_text_format_payload = Vec::new();
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_with_zero_params_and_single_shared_text_format = frontend_frame(
        b'B',
        &bind_unnamed_with_zero_params_and_single_shared_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_zero_params_and_single_shared_text_format)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_zero_params_and_single_shared_binary_format_payload = Vec::new();
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(b"\0\0");
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_zero_params_and_single_shared_binary_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_zero_params_and_single_shared_binary_format = frontend_frame(
        b'B',
        &bind_unnamed_with_zero_params_and_single_shared_binary_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_zero_params_and_single_shared_binary_format)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_with_empty_text_parameter_payload = Vec::new();
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_with_empty_text_parameter =
        frontend_frame(b'B', &bind_unnamed_with_empty_text_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_empty_text_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_empty_binary_parameter_payload = Vec::new();
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_empty_binary_parameter =
        frontend_frame(b'B', &bind_unnamed_with_empty_binary_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_empty_binary_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_with_text_parameter_payload = Vec::new();
    bind_unnamed_with_text_parameter_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_text_parameter_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_unnamed_with_text_parameter_payload.extend_from_slice("héllo".as_bytes());
    bind_unnamed_with_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_unnamed_with_text_parameter =
        frontend_frame(b'B', &bind_unnamed_with_text_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_text_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_unnamed_with_binary_parameter_payload = Vec::new();
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_binary_parameter =
        frontend_frame(b'B', &bind_unnamed_with_binary_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_binary_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
            result_format_codes: vec![1],
        }
    );

    let mut bind_with_zero_params_and_single_shared_format_payload = Vec::new();
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(b"portal3\0stmt3\0");
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_zero_params_and_single_shared_format_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_zero_params_and_single_shared_format = frontend_frame(
        b'B',
        &bind_with_zero_params_and_single_shared_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_with_zero_params_and_single_shared_format).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal3".to_string(),
            statement_name: "stmt3".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_with_multiple_result_formats_payload = Vec::new();
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_multiple_result_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_multiple_result_formats =
        frontend_frame(b'B', &bind_unnamed_with_multiple_result_formats_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_multiple_result_formats).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_multiple_result_formats_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice("pörtal_results\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_multiple_result_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_multiple_result_formats = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_multiple_result_formats_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_unnamed_statement_multiple_result_formats,)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_results".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_multiple_result_formats_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(b"\0stmt_results_named\0");
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_multiple_result_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_multiple_result_formats = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_multiple_result_formats_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_unnamed_portal_with_named_statement_multiple_result_formats,)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_results_named".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![0, 1],
        }
    );

    let mut bind_unnamed_with_default_parameter_formats_payload = Vec::new();
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(b"\0\0");
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(b"text");
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_with_default_parameter_formats =
        frontend_frame(b'B', &bind_unnamed_with_default_parameter_formats_payload);
    assert_eq!(
        parse_frontend_message(&bind_unnamed_with_default_parameter_formats).unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: String::new(),
            parameter_format_codes: vec![],
            parameters: vec![Some(b"text".to_vec()), None],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_unnamed_statement_default_parameter_formats_payload = Vec::new();
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice("pörtal_default\0\0".as_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(b"text");
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_unnamed_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_unnamed_statement_default_parameter_formats = frontend_frame(
        b'B',
        &bind_named_portal_with_unnamed_statement_default_parameter_formats_payload,
    );
    assert_eq!(
        parse_frontend_message(
            &bind_named_portal_with_unnamed_statement_default_parameter_formats,
        )
        .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_default".to_string(),
            statement_name: String::new(),
            parameter_format_codes: vec![],
            parameters: vec![Some(b"text".to_vec()), None],
            result_format_codes: vec![1],
        }
    );

    let mut bind_unnamed_portal_with_named_statement_default_parameter_formats_payload = Vec::new();
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(b"\0stmt_default_named\0");
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(b"text");
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_unnamed_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_unnamed_portal_with_named_statement_default_parameter_formats = frontend_frame(
        b'B',
        &bind_unnamed_portal_with_named_statement_default_parameter_formats_payload,
    );
    assert_eq!(
        parse_frontend_message(
            &bind_unnamed_portal_with_named_statement_default_parameter_formats,
        )
        .unwrap(),
        FrontendMessage::Bind {
            portal_name: String::new(),
            statement_name: "stmt_default_named".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![Some(b"text".to_vec()), None],
            result_format_codes: vec![1],
        }
    );

    let mut bind_named_portal_with_named_statement_default_parameter_formats_payload = Vec::new();
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice("pörtal_default_pair\0stmt_default_pair\0".as_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&2_i16.to_be_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&4_i32.to_be_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(b"text");
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&(-1_i32).to_be_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    bind_named_portal_with_named_statement_default_parameter_formats_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let bind_named_portal_with_named_statement_default_parameter_formats = frontend_frame(
        b'B',
        &bind_named_portal_with_named_statement_default_parameter_formats_payload,
    );
    assert_eq!(
        parse_frontend_message(&bind_named_portal_with_named_statement_default_parameter_formats)
            .unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal_default_pair".to_string(),
            statement_name: "stmt_default_pair".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![Some(b"text".to_vec()), None],
            result_format_codes: vec![1],
        }
    );

    let mut bind_with_default_parameter_formats_payload = Vec::new();
    bind_with_default_parameter_formats_payload.extend_from_slice(b"portal2\0stmt2\0");
    bind_with_default_parameter_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_default_parameter_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    bind_with_default_parameter_formats_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_with_default_parameter_formats_payload.extend_from_slice(b"text");
    bind_with_default_parameter_formats_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    bind_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_default_parameter_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_default_parameter_formats =
        frontend_frame(b'B', &bind_with_default_parameter_formats_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_default_parameter_formats).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal2".to_string(),
            statement_name: "stmt2".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![Some(b"text".to_vec()), None],
            result_format_codes: vec![1],
        }
    );

    let mut bind_utf8_names_payload = Vec::new();
    bind_utf8_names_payload.extend_from_slice("pörtal\0stmté\0".as_bytes());
    bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_utf8_names = frontend_frame(b'B', &bind_utf8_names_payload);
    assert_eq!(
        parse_frontend_message(&bind_utf8_names).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal".to_string(),
            statement_name: "stmté".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![],
        }
    );

    let mut bind_multiline_utf8_names_payload = Vec::new();
    bind_multiline_utf8_names_payload
        .extend_from_slice("pörtal\nΔetail\0stmté\nΔetail\0".as_bytes());
    bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_multiline_utf8_names_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_multiline_utf8_names = frontend_frame(b'B', &bind_multiline_utf8_names_payload);
    assert_eq!(
        parse_frontend_message(&bind_multiline_utf8_names).unwrap(),
        FrontendMessage::Bind {
            portal_name: "pörtal\nΔetail".to_string(),
            statement_name: "stmté\nΔetail".to_string(),
            parameter_format_codes: vec![],
            parameters: vec![],
            result_format_codes: vec![],
        }
    );

    let mut bind_with_utf8_text_parameter_payload = Vec::new();
    bind_with_utf8_text_parameter_payload.extend_from_slice(b"portal_utf8\0stmt_utf8\0");
    bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice("héllo".as_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_utf8_text_parameter =
        frontend_frame(b'B', &bind_with_utf8_text_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_utf8_text_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_utf8".to_string(),
            statement_name: "stmt_utf8".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo".as_bytes().to_vec())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_with_multiline_utf8_text_parameter_payload = Vec::new();
    bind_with_multiline_utf8_text_parameter_payload
        .extend_from_slice(b"portal_utf8_multi\0stmt_utf8_multi\0");
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_multiline_utf8_text_parameter_payload
        .extend_from_slice(&("héllo\nΔetail".len() as i32).to_be_bytes());
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice("héllo\nΔetail".as_bytes());
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_multiline_utf8_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_multiline_utf8_text_parameter =
        frontend_frame(b'B', &bind_with_multiline_utf8_text_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_multiline_utf8_text_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_utf8_multi".to_string(),
            statement_name: "stmt_utf8_multi".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some("héllo\nΔetail".as_bytes().to_vec())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_with_empty_text_parameter_payload = Vec::new();
    bind_with_empty_text_parameter_payload
        .extend_from_slice(b"portal_empty_text\0stmt_empty_text\0");
    bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_text_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
    bind_with_empty_text_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_text_parameter_payload.extend_from_slice(&0_i16.to_be_bytes());
    let bind_with_empty_text_parameter =
        frontend_frame(b'B', &bind_with_empty_text_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_empty_text_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_empty_text".to_string(),
            statement_name: "stmt_empty_text".to_string(),
            parameter_format_codes: vec![0],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![0],
        }
    );

    let mut bind_with_binary_parameter_payload = Vec::new();
    bind_with_binary_parameter_payload.extend_from_slice(b"portal_bin\0stmt_bin\0");
    bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_binary_parameter_payload.extend_from_slice(&4_i32.to_be_bytes());
    bind_with_binary_parameter_payload.extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
    bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_binary_parameter = frontend_frame(b'B', &bind_with_binary_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_binary_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_bin".to_string(),
            statement_name: "stmt_bin".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
            result_format_codes: vec![1],
        }
    );

    let mut bind_with_empty_binary_parameter_payload = Vec::new();
    bind_with_empty_binary_parameter_payload.extend_from_slice(b"portal_empty\0stmt_empty\0");
    bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_binary_parameter_payload.extend_from_slice(&0_i32.to_be_bytes());
    bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    bind_with_empty_binary_parameter_payload.extend_from_slice(&1_i16.to_be_bytes());
    let bind_with_empty_binary_parameter =
        frontend_frame(b'B', &bind_with_empty_binary_parameter_payload);
    assert_eq!(
        parse_frontend_message(&bind_with_empty_binary_parameter).unwrap(),
        FrontendMessage::Bind {
            portal_name: "portal_empty".to_string(),
            statement_name: "stmt_empty".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(Vec::new())],
            result_format_codes: vec![1],
        }
    );

    let describe_stmt = frontend_frame(b'D', b"Sstmt1\0");
    assert_eq!(
        parse_frontend_message(&describe_stmt).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "stmt1".to_string(),
        }
    );

    let describe_portal = frontend_frame(b'D', b"Pportal1\0");
    assert_eq!(
        parse_frontend_message(&describe_portal).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "portal1".to_string(),
        }
    );

    let describe_stmt_lowercase = frontend_frame(b'D', b"sstmt1\0");
    assert_eq!(
        parse_frontend_message(&describe_stmt_lowercase).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "stmt1".to_string(),
        }
    );

    let describe_portal_lowercase = frontend_frame(b'D', b"pportal1\0");
    assert_eq!(
        parse_frontend_message(&describe_portal_lowercase).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "portal1".to_string(),
        }
    );

    let describe_utf8_portal = frontend_frame(b'D', "Ppörtal\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&describe_utf8_portal).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "pörtal".to_string(),
        }
    );

    let describe_utf8_statement = frontend_frame(b'D', "Sstmté\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&describe_utf8_statement).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "stmté".to_string(),
        }
    );

    let describe_multiline_utf8_portal = frontend_frame(b'D', "Ppörtal\nΔetail\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&describe_multiline_utf8_portal).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "pörtal\nΔetail".to_string(),
        }
    );

    let describe_multiline_utf8_statement = frontend_frame(b'D', "Sstmté\nΔetail\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&describe_multiline_utf8_statement).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "stmté\nΔetail".to_string(),
        }
    );

    let describe_unnamed_statement = frontend_frame(b'D', b"S\0");
    assert_eq!(
        parse_frontend_message(&describe_unnamed_statement).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: String::new(),
        }
    );

    let describe_unnamed_portal_lowercase = frontend_frame(b'D', b"p\0");
    assert_eq!(
        parse_frontend_message(&describe_unnamed_portal_lowercase).unwrap(),
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: String::new(),
        }
    );

    let close_stmt = frontend_frame(b'C', b"Sstmt1\0");
    assert_eq!(
        parse_frontend_message(&close_stmt).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Statement,
            name: "stmt1".to_string(),
        }
    );

    let close_portal = frontend_frame(b'C', b"Pportal1\0");
    assert_eq!(
        parse_frontend_message(&close_portal).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "portal1".to_string(),
        }
    );

    let close_stmt_lowercase = frontend_frame(b'C', b"sstmt1\0");
    assert_eq!(
        parse_frontend_message(&close_stmt_lowercase).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Statement,
            name: "stmt1".to_string(),
        }
    );

    let close_portal_lowercase = frontend_frame(b'C', b"pportal1\0");
    assert_eq!(
        parse_frontend_message(&close_portal_lowercase).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "portal1".to_string(),
        }
    );

    let close_utf8_portal = frontend_frame(b'C', "Ppörtal\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&close_utf8_portal).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "pörtal".to_string(),
        }
    );

    let close_utf8_statement = frontend_frame(b'C', "Sstmté\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&close_utf8_statement).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Statement,
            name: "stmté".to_string(),
        }
    );

    let close_multiline_utf8_portal = frontend_frame(b'C', "Ppörtal\nΔetail\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&close_multiline_utf8_portal).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "pörtal\nΔetail".to_string(),
        }
    );

    let close_multiline_utf8_statement = frontend_frame(b'C', "Sstmté\nΔetail\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&close_multiline_utf8_statement).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Statement,
            name: "stmté\nΔetail".to_string(),
        }
    );

    let close_unnamed_statement = frontend_frame(b'C', b"S\0");
    assert_eq!(
        parse_frontend_message(&close_unnamed_statement).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Statement,
            name: String::new(),
        }
    );

    let close_unnamed_portal_lowercase = frontend_frame(b'C', b"p\0");
    assert_eq!(
        parse_frontend_message(&close_unnamed_portal_lowercase).unwrap(),
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: String::new(),
        }
    );

    let mut execute_payload = Vec::new();
    execute_payload.extend_from_slice(b"portal1\0");
    execute_payload.extend_from_slice(&128_u32.to_be_bytes());
    let execute = frontend_frame(b'E', &execute_payload);
    assert_eq!(
        parse_frontend_message(&execute).unwrap(),
        FrontendMessage::Execute {
            portal_name: "portal1".to_string(),
            max_rows: 128,
        }
    );

    let mut execute_unnamed_portal_unlimited_rows_payload = Vec::new();
    execute_unnamed_portal_unlimited_rows_payload.extend_from_slice(b"\0");
    execute_unnamed_portal_unlimited_rows_payload.extend_from_slice(&0_u32.to_be_bytes());
    let execute_unnamed_portal_unlimited_rows =
        frontend_frame(b'E', &execute_unnamed_portal_unlimited_rows_payload);
    assert_eq!(
        parse_frontend_message(&execute_unnamed_portal_unlimited_rows).unwrap(),
        FrontendMessage::Execute {
            portal_name: String::new(),
            max_rows: 0,
        }
    );

    let mut execute_utf8_portal_payload = Vec::new();
    execute_utf8_portal_payload.extend_from_slice("pörtal\0".as_bytes());
    execute_utf8_portal_payload.extend_from_slice(&64_u32.to_be_bytes());
    let execute_utf8_portal = frontend_frame(b'E', &execute_utf8_portal_payload);
    assert_eq!(
        parse_frontend_message(&execute_utf8_portal).unwrap(),
        FrontendMessage::Execute {
            portal_name: "pörtal".to_string(),
            max_rows: 64,
        }
    );

    let mut execute_multiline_utf8_portal_payload = Vec::new();
    execute_multiline_utf8_portal_payload.extend_from_slice("pörtal\nΔetail\0".as_bytes());
    execute_multiline_utf8_portal_payload.extend_from_slice(&32_u32.to_be_bytes());
    let execute_multiline_utf8_portal =
        frontend_frame(b'E', &execute_multiline_utf8_portal_payload);
    assert_eq!(
        parse_frontend_message(&execute_multiline_utf8_portal).unwrap(),
        FrontendMessage::Execute {
            portal_name: "pörtal\nΔetail".to_string(),
            max_rows: 32,
        }
    );

    let mut function_call_payload = Vec::new();
    function_call_payload.extend_from_slice(&42_u32.to_be_bytes());
    function_call_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_payload.extend_from_slice(&2_i16.to_be_bytes());
    function_call_payload.extend_from_slice(&4_i32.to_be_bytes());
    function_call_payload.extend_from_slice(&7_i32.to_be_bytes());
    function_call_payload.extend_from_slice(&(-1_i32).to_be_bytes());
    function_call_payload.extend_from_slice(&0_i16.to_be_bytes());
    let function_call = frontend_frame(b'F', &function_call_payload);
    assert_eq!(
        parse_frontend_message(&function_call).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 42,
            argument_format_codes: vec![1],
            arguments: vec![Some(7_i32.to_be_bytes().to_vec()), None],
            result_format_code: 0,
        }
    );

    let mut function_call_with_multiple_argument_formats_payload = Vec::new();
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&7_u32.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&2_i16.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&3_i32.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(b"foo");
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&2_i32.to_be_bytes());
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&[0xCA, 0xFE]);
    function_call_with_multiple_argument_formats_payload.extend_from_slice(&1_i16.to_be_bytes());
    let function_call_with_multiple_argument_formats =
        frontend_frame(b'F', &function_call_with_multiple_argument_formats_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_multiple_argument_formats).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 7,
            argument_format_codes: vec![0, 1],
            arguments: vec![Some(b"foo".to_vec()), Some(vec![0xCA, 0xFE])],
            result_format_code: 1,
        }
    );

    let mut function_call_with_zero_args_and_single_shared_format_payload = Vec::new();
    function_call_with_zero_args_and_single_shared_format_payload
        .extend_from_slice(&99_u32.to_be_bytes());
    function_call_with_zero_args_and_single_shared_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    let function_call_with_zero_args_and_single_shared_format = frontend_frame(
        b'F',
        &function_call_with_zero_args_and_single_shared_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&function_call_with_zero_args_and_single_shared_format).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 99,
            argument_format_codes: vec![1],
            arguments: vec![],
            result_format_code: 1,
        }
    );

    let mut function_call_with_zero_args_and_single_shared_text_format_payload = Vec::new();
    function_call_with_zero_args_and_single_shared_text_format_payload
        .extend_from_slice(&100_u32.to_be_bytes());
    function_call_with_zero_args_and_single_shared_text_format_payload
        .extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_zero_args_and_single_shared_text_format_payload
        .extend_from_slice(&0_i16.to_be_bytes());
    let function_call_with_zero_args_and_single_shared_text_format = frontend_frame(
        b'F',
        &function_call_with_zero_args_and_single_shared_text_format_payload,
    );
    assert_eq!(
        parse_frontend_message(&function_call_with_zero_args_and_single_shared_text_format)
            .unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 100,
            argument_format_codes: vec![0],
            arguments: vec![],
            result_format_code: 0,
        }
    );

    let mut function_call_with_embedded_null_binary_arg_payload = Vec::new();
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&101_u32.to_be_bytes());
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&4_i32.to_be_bytes());
    function_call_with_embedded_null_binary_arg_payload
        .extend_from_slice(&[0x00, 0xCA, 0x00, 0xFE]);
    function_call_with_embedded_null_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    let function_call_with_embedded_null_binary_arg =
        frontend_frame(b'F', &function_call_with_embedded_null_binary_arg_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_embedded_null_binary_arg).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 101,
            argument_format_codes: vec![1],
            arguments: vec![Some(vec![0x00, 0xCA, 0x00, 0xFE])],
            result_format_code: 1,
        }
    );

    let mut function_call_with_utf8_text_arg_payload = Vec::new();
    function_call_with_utf8_text_arg_payload.extend_from_slice(&103_u32.to_be_bytes());
    function_call_with_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_utf8_text_arg_payload
        .extend_from_slice(&("héllo".len() as i32).to_be_bytes());
    function_call_with_utf8_text_arg_payload.extend_from_slice("héllo".as_bytes());
    function_call_with_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    let function_call_with_utf8_text_arg =
        frontend_frame(b'F', &function_call_with_utf8_text_arg_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_utf8_text_arg).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 103,
            argument_format_codes: vec![0],
            arguments: vec![Some("héllo".as_bytes().to_vec())],
            result_format_code: 0,
        }
    );

    let mut function_call_with_multiline_utf8_text_arg_payload = Vec::new();
    function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&105_u32.to_be_bytes());
    function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_multiline_utf8_text_arg_payload
        .extend_from_slice(&("héllo\nΔetail".len() as i32).to_be_bytes());
    function_call_with_multiline_utf8_text_arg_payload
        .extend_from_slice("héllo\nΔetail".as_bytes());
    function_call_with_multiline_utf8_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    let function_call_with_multiline_utf8_text_arg =
        frontend_frame(b'F', &function_call_with_multiline_utf8_text_arg_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_multiline_utf8_text_arg).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 105,
            argument_format_codes: vec![0],
            arguments: vec![Some("héllo\nΔetail".as_bytes().to_vec())],
            result_format_code: 0,
        }
    );

    let mut function_call_with_empty_text_arg_payload = Vec::new();
    function_call_with_empty_text_arg_payload.extend_from_slice(&104_u32.to_be_bytes());
    function_call_with_empty_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_empty_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    function_call_with_empty_text_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_empty_text_arg_payload.extend_from_slice(&0_i32.to_be_bytes());
    function_call_with_empty_text_arg_payload.extend_from_slice(&0_i16.to_be_bytes());
    let function_call_with_empty_text_arg =
        frontend_frame(b'F', &function_call_with_empty_text_arg_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_empty_text_arg).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 104,
            argument_format_codes: vec![0],
            arguments: vec![Some(Vec::new())],
            result_format_code: 0,
        }
    );

    let mut function_call_with_empty_binary_arg_payload = Vec::new();
    function_call_with_empty_binary_arg_payload.extend_from_slice(&102_u32.to_be_bytes());
    function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    function_call_with_empty_binary_arg_payload.extend_from_slice(&0_i32.to_be_bytes());
    function_call_with_empty_binary_arg_payload.extend_from_slice(&1_i16.to_be_bytes());
    let function_call_with_empty_binary_arg =
        frontend_frame(b'F', &function_call_with_empty_binary_arg_payload);
    assert_eq!(
        parse_frontend_message(&function_call_with_empty_binary_arg).unwrap(),
        FrontendMessage::FunctionCall {
            function_oid: 102,
            argument_format_codes: vec![1],
            arguments: vec![Some(Vec::new())],
            result_format_code: 1,
        }
    );

    let copy_data = frontend_frame(b'd', &[0, 1, 2, 3]);
    assert_eq!(
        parse_frontend_message(&copy_data).unwrap(),
        FrontendMessage::CopyData(vec![0, 1, 2, 3])
    );

    let empty_copy_data = frontend_frame(b'd', &[]);
    assert_eq!(
        parse_frontend_message(&empty_copy_data).unwrap(),
        FrontendMessage::CopyData(Vec::new())
    );

    let binary_copy_data = frontend_frame(b'd', b"row\0chunk\xff");
    assert_eq!(
        parse_frontend_message(&binary_copy_data).unwrap(),
        FrontendMessage::CopyData(b"row\0chunk\xff".to_vec())
    );

    let utf8_copy_data = frontend_frame(b'd', "röwΔ".as_bytes());
    assert_eq!(
        parse_frontend_message(&utf8_copy_data).unwrap(),
        FrontendMessage::CopyData("röwΔ".as_bytes().to_vec())
    );

    let mixed_utf8_binary_copy_data = frontend_frame(b'd', &[b'r', 0xC3, 0xB8, 0xFF, b'!']);
    assert_eq!(
        parse_frontend_message(&mixed_utf8_binary_copy_data).unwrap(),
        FrontendMessage::CopyData(vec![b'r', 0xC3, 0xB8, 0xFF, b'!'])
    );

    let copy_done = frontend_frame(b'c', &[]);
    assert_eq!(
        parse_frontend_message(&copy_done).unwrap(),
        FrontendMessage::CopyDone
    );

    let copy_fail = frontend_frame(b'f', b"bad row\0");
    assert_eq!(
        parse_frontend_message(&copy_fail).unwrap(),
        FrontendMessage::CopyFail("bad row".to_string())
    );

    let empty_copy_fail = frontend_frame(b'f', b"\0");
    assert_eq!(
        parse_frontend_message(&empty_copy_fail).unwrap(),
        FrontendMessage::CopyFail(String::new())
    );

    let utf8_copy_fail = frontend_frame(b'f', "röw mismatch\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&utf8_copy_fail).unwrap(),
        FrontendMessage::CopyFail("röw mismatch".to_string())
    );

    let multiline_utf8_copy_fail = frontend_frame(b'f', "röw mismatch\nΔetail\0".as_bytes());
    assert_eq!(
        parse_frontend_message(&multiline_utf8_copy_fail).unwrap(),
        FrontendMessage::CopyFail("röw mismatch\nΔetail".to_string())
    );

    let terminate = frontend_frame(b'X', &[]);
    assert_eq!(
        parse_frontend_message(&terminate).unwrap(),
        FrontendMessage::Terminate
    );

    let sync = frontend_frame(b'S', &[]);
    assert_eq!(
        parse_frontend_message(&sync).unwrap(),
        FrontendMessage::Sync
    );

    let flush = frontend_frame(b'H', &[]);
    assert_eq!(
        parse_frontend_message(&flush).unwrap(),
        FrontendMessage::Flush
    );
}
