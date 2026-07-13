#[test]
fn resident_text_prefix_ptx_is_pure_ascii() {
    let ptx = include_bytes!("../resident_text_prefix.ptx");
    assert!(
        ptx.is_ascii(),
        "the runtime JIT rejects non-ASCII PTX even when ptxas accepts it"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cuda_resident_text_prefix_count_reduces_on_device_and_fails_malformed_closed() {
    let runtime = CudaDriverRuntime::probe().expect("requires a local NVIDIA driver and GPU");
    let rows = ["alpha", "alphabet", "beta", "", "alpine", "omega"];
    let row_count = rows.len() as u64;
    let offsets_off = 16_u64;
    let bytes_off = offsets_off + (row_count + 1) * 8;
    let mut offsets = Vec::new();
    let mut blob = Vec::new();
    offsets.extend_from_slice(&0_u64.to_le_bytes());
    for row in rows {
        blob.extend_from_slice(row.as_bytes());
        offsets.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    let resident = std::sync::Arc::new(
        runtime
            .retain_device_memory_chunks(
                0,
                bytes_off + blob.len() as u64,
                &[
                    CudaDeviceMemoryChunk {
                        byte_offset: offsets_off,
                        bytes: &offsets,
                    },
                    CudaDeviceMemoryChunk {
                        byte_offset: bytes_off,
                        bytes: &blob,
                    },
                ],
            )
            .expect("resident text payload"),
    );

    let reader = std::sync::Arc::clone(&resident);
    let blob_len = blob.len() as u64;
    assert_eq!(
        std::thread::spawn(move || {
            reader.count_text_prefix_from_payload(
                offsets_off,
                bytes_off,
                blob_len,
                row_count,
                b"alpha",
            )
        })
        .join()
        .expect("fresh reader thread")
        .expect("the API binds its primary context before module and buffer work"),
        2
    );

    for (prefix, expected) in [
        (b"alpha".as_slice(), 2),
        (b"al".as_slice(), 3),
        (b"beta".as_slice(), 1),
        (b"missing".as_slice(), 0),
        (b"".as_slice(), row_count),
    ] {
        assert_eq!(
            resident
                .count_text_prefix_from_payload(
                    offsets_off,
                    bytes_off,
                    blob.len() as u64,
                    row_count,
                    prefix,
                )
                .expect("device prefix reduction"),
            expected,
            "prefix {:?}",
            String::from_utf8_lossy(prefix)
        );
    }

    assert!(
        resident
            .count_text_prefix_from_payload(
                offsets_off + 4,
                bytes_off,
                blob.len() as u64,
                row_count,
                b"al",
            )
            .is_err(),
        "misaligned offsets must fail before launch"
    );

    let malformed_offsets = [0_u64, 5, u64::MAX]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let malformed_bytes_off = 8 + 3 * 8;
    let malformed_blob = b"alpha";
    let malformed = runtime
        .retain_device_memory_chunks(
            0,
            malformed_bytes_off + malformed_blob.len() as u64,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 8,
                    bytes: &malformed_offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: malformed_bytes_off,
                    bytes: malformed_blob,
                },
            ],
        )
        .expect("malformed resident text payload");
    assert!(
        malformed
            .count_text_prefix_from_payload(
                8,
                malformed_bytes_off,
                malformed_blob.len() as u64,
                2,
                b"alpha",
            )
            .is_err(),
        "device-reported malformed offsets must fail closed"
    );

    let nonzero_first_offsets = [1_u64, 1]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let nonzero_first_bytes_off = 8 + 2 * 8;
    let nonzero_first = runtime
        .retain_device_memory_chunks(
            0,
            nonzero_first_bytes_off + 1,
            &[
                CudaDeviceMemoryChunk {
                    byte_offset: 8,
                    bytes: &nonzero_first_offsets,
                },
                CudaDeviceMemoryChunk {
                    byte_offset: nonzero_first_bytes_off,
                    bytes: b"x",
                },
            ],
        )
        .expect("nonzero-first resident text payload");
    assert!(
        nonzero_first
            .count_text_prefix_from_payload(8, nonzero_first_bytes_off, 1, 1, b"")
            .is_err(),
        "the canonical offsets vector must begin at zero"
    );

    let zero_offset = 0_u64.to_le_bytes();
    let empty = runtime
        .retain_device_memory_chunks(
            0,
            16,
            &[CudaDeviceMemoryChunk {
                byte_offset: 8,
                bytes: &zero_offset,
            }],
        )
        .expect("empty resident text payload");
    assert_eq!(
        empty
            .count_text_prefix_from_payload(8, 16, 0, 0, b"anything")
            .expect("zero-row device prefix reduction"),
        0
    );

    assert_eq!(
        resident
            .count_text_prefix_from_payload(
                offsets_off,
                bytes_off,
                blob.len() as u64,
                row_count,
                b"al",
            )
            .expect("context reusable after rejected and device-reported inputs"),
        3
    );
}
