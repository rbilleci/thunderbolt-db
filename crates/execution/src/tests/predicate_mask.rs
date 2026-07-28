use super::*;

#[test]
fn first_true_row_terminal_stays_a_single_word_device_readback_without_coordinates() {
    let source = include_str!("../predicate_mask.rs");
    let terminal = source
        .split("pub fn first_true_row")
        .nth(1)
        .and_then(|source| source.split("pub(super) fn device_ptr").next())
        .expect("first-true terminal and its one-word reducer exist");
    assert!(terminal.contains("atom.global.min.u32"));
    assert!(terminal.contains("u32::MAX"));
    assert_eq!(terminal.matches("dtoh(").count(), 1);
    assert!(!terminal.contains("Vec<u32>"));
    assert!(!terminal.contains("coordinates"));
}

#[test]
fn predicate_mask_terminal_ptx_keeps_the_u32_max_domain_in_u64_grid_stride_arithmetic() {
    let source = include_str!("../predicate_mask.rs");
    let ptx_for = |entry| {
        source
            .split(entry)
            .nth(1)
            .and_then(|ptx| ptx.split("\"#;").next())
            .expect("terminal PTX exists")
    };
    for entry in [
        "gpu_db_predicate_mask_any_true(",
        "gpu_db_predicate_mask_first_true_row(",
    ] {
        let ptx = ptx_for(entry);
        for required in [
            "mul.wide.u32 %rd3, %r3, %r4",
            "cvt.u64.u32 %rd0, %r2",
            "add.u64 %rd3, %rd3, %rd0",
            "mul.wide.u32 %rd4, %r5, %r4",
            "cvt.u64.u32",
            "setp.ge.u64",
            "mul.lo.u64",
            "add.u64",
        ] {
            assert!(ptx.contains(required), "{entry} lacks {required}");
        }
        for forbidden in [
            "mad.wide.u32",
            "mad.lo.u32",
            "mul.lo.u32",
            "setp.ge.u32",
            "add.u32",
        ] {
            assert!(!ptx.contains(forbidden), "{entry} still has {forbidden}");
        }
    }
    let first_row = ptx_for("gpu_db_predicate_mask_first_true_row(");
    assert!(first_row.contains("cvt.u32.u64 %r8, %rd3"));
    assert_eq!(u64::from(u32::MAX), 4_294_967_295);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_predicate_mask_first_true_row_is_minimal_and_repeatable() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let resident = runtime
        .retain_device_memory_copy(0, &0_u64.to_le_bytes())
        .expect("resident source");

    let no_true = resident
        .row_range_mask_u32(8, 0, 0)
        .expect("all-false mask");
    assert_eq!(no_true.first_true_row().unwrap(), None);

    let row_zero = resident.row_range_mask_u32(8, 0, 4).expect("row-zero mask");
    assert_eq!(row_zero.first_true_row().unwrap(), Some(0));

    let later = resident
        .row_range_mask_u32(8, 3, 8)
        .expect("later-row mask");
    for _ in 0..8 {
        assert_eq!(later.first_true_row().unwrap(), Some(3));
    }
}
