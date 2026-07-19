use super::{validate_resident_arith_program, ExprStep, ExprTerminal, ResidentElemType};

fn validate(
    bytes: u64,
    program: &[ExprStep],
    needles: &[Vec<u8>],
    rows: u64,
    elem: ResidentElemType,
    terminal: ExprTerminal,
) -> bool {
    validate_resident_arith_program(bytes, program, needles, rows, elem, terminal).is_ok()
}

#[test]
fn expression_preflight_checks_typed_stack_and_fixed_windows() {
    let valid = [
        ExprStep::LoadColumn { byte_offset: 16 },
        ExprStep::CompareScalar {
            cmp: 0,
            scalar: 7,
            scalar_on_left: false,
        },
    ];
    assert!(validate(
        32,
        &valid,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    assert!(!validate(
        31,
        &valid,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));

    let wrong_width = [
        ExprStep::LoadColumn { byte_offset: 0 },
        ExprStep::CompareScalarI64 {
            cmp: 0,
            scalar: 7,
            scalar_on_left: false,
        },
    ];
    assert!(!validate(
        64,
        &wrong_width,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));

    let mask_as_value = [
        ExprStep::ConstMask { value: true },
        ExprStep::ScalarBinary {
            op: 0,
            scalar: 1,
            scalar_on_left: false,
        },
    ];
    assert!(!validate(
        64,
        &mask_as_value,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Value
    ));
    assert!(!validate(
        64,
        &[ExprStep::ConstMask { value: true }],
        &[],
        4,
        ResidentElemType::I64,
        ExprTerminal::Value,
    ));

    let two_values = [
        ExprStep::LoadColumn { byte_offset: 0 },
        ExprStep::LoadColumn { byte_offset: 16 },
    ];
    assert!(validate(
        32,
        &two_values,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::TwoValues
    ));
    assert!(!validate(
        32,
        &two_values,
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Value
    ));
    assert!(!validate(
        16,
        &two_values[..1],
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Mask,
    ));
    assert!(!validate(
        32,
        &[ExprStep::LoadColumn { byte_offset: 1 }],
        &[],
        4,
        ResidentElemType::I32,
        ExprTerminal::Value,
    ));
}

#[test]
fn indexed_load_preflight_must_cover_the_full_source_extent() {
    let program = [ExprStep::LoadColumn { byte_offset: 12 }];

    assert!(validate(
        16,
        &program,
        &[],
        1,
        ResidentElemType::I32,
        ExprTerminal::Value,
    ));
    assert!(!validate(
        16,
        &program,
        &[],
        2,
        ResidentElemType::I32,
        ExprTerminal::Value,
    ));
}

#[test]
fn expression_preflight_checks_bitmap_uuid_and_opcode_contracts() {
    let bool_mask = [ExprStep::BoolMask {
        bitmap_byte_offset: 8,
        negate: false,
    }];
    assert!(!validate(
        12,
        &bool_mask,
        &[],
        33,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    assert!(validate(
        16,
        &bool_mask,
        &[],
        33,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));

    let uuid = [ExprStep::UuidCmpMask {
        byte_offset: 16,
        needle_idx: 0,
        scalar_on_left: false,
        cmp: 5,
    }];
    assert!(validate(
        48,
        &uuid,
        &[vec![0; 16]],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    assert!(!validate(
        47,
        &uuid,
        &[vec![0; 16]],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    let bad_cmp = [ExprStep::CompareScalar {
        cmp: 6,
        scalar: 0,
        scalar_on_left: false,
    }];
    assert!(!validate(
        48,
        &bad_cmp,
        &[],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));

    let text_columns = [ExprStep::TextCmpColumnsMask {
        a_offsets_byte_offset: 0,
        a_bytes_byte_offset: 48,
        a_bytes_len: 4,
        b_offsets_byte_offset: 24,
        b_bytes_byte_offset: 52,
        b_bytes_len: 4,
        cmp: 0,
    }];
    assert!(validate(
        56,
        &text_columns,
        &[],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    assert!(!validate(
        55,
        &text_columns,
        &[],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
    let misaligned_text = [ExprStep::TextCmpColumnsMask {
        a_offsets_byte_offset: 4,
        a_bytes_byte_offset: 48,
        a_bytes_len: 4,
        b_offsets_byte_offset: 24,
        b_bytes_byte_offset: 52,
        b_bytes_len: 4,
        cmp: 0,
    }];
    assert!(!validate(
        64,
        &misaligned_text,
        &[],
        2,
        ResidentElemType::I32,
        ExprTerminal::Mask
    ));
}
