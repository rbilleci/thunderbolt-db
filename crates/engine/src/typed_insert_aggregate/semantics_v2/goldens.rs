//! Test-only byte goldens for the frozen codec-5 semantics-v2 profile.
//!
//! This module is deliberately a fixture constructor, not a semantics-v2 encoder.  It has no
//! production module declaration, WAL constructor, recovery conversion, apply operation, device
//! work, or publication capability.  The only future module edge is `#[cfg(test)]` from the
//! private semantics-v2 facade.

// Q1's two non-minimal freeze shapes and their repaired-inner-root hostile cases are separate
// test-only owners.  Keeping them out of this small baseline fixture prevents the literal
// minimum abort from becoming a catch-all vector builder.
#[path = "goldens/q1_sabotage.rs"]
mod q1_sabotage;
#[path = "goldens/q1_vectors.rs"]
mod q1_vectors;
#[path = "goldens/q2_guard_sabotage.rs"]
mod q2_guard_sabotage;
#[path = "goldens/q2_reencode.rs"]
mod q2_reencode;
#[path = "goldens/q2_witnesses.rs"]
mod q2_witnesses;
#[path = "goldens/q3_sequence.rs"]
mod q3_sequence;
#[path = "goldens/s8_literals.rs"]
mod s8_literals;
#[path = "goldens/s8_sabotage.rs"]
mod s8_sabotage;
#[path = "goldens/s8_vectors.rs"]
mod s8_vectors;

use super::{
    close_canonical_semantics_v2_for_test, fail_retained_source_copy_at_for_test,
    fill_canonical_semantics_v2_for_test, measure_canonical_semantics_v2,
    validate_canonical_semantics_v2_guards_for_test, validate_dependency_token_digest_for_test,
};
use crate::typed_insert_aggregate::{
    encode_status_v2, TypedInsertStatusV2, AGGREGATE_CHUNK_FLAG_FIRST, AGGREGATE_CHUNK_FLAG_LAST,
    AGGREGATE_CHUNK_HEADER_BYTES, AGGREGATE_CHUNK_MAGIC, AGGREGATE_FORMAT_VERSION,
    AGGREGATE_ROOT_TRAILER_BYTES, AGGREGATE_SECTION_COUNT, AGGREGATE_SECTION_HEADER_BYTES,
    AGGREGATE_STATUS_V2_BYTES, AGGREGATE_STREAM_MAGIC,
    ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE, OUTER_CONTENT_ROW,
    OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
};
use crate::typed_insert_batch::{
    copy_decoded_canonical_typed_insert_published_only_after_measure,
    measure_decoded_canonical_typed_insert_published_only_from_source, CanonicalTypedInsertReadAt,
};
use crate::EngineError;
use sha2::{Digest, Sha256};
use std::cell::Cell;

const ABSENT_U32: u32 = u32::MAX;
const STABLE_TRANSACTION_ID: u64 = 77;
const COMMIT_SEQUENCE: u64 = 17;
const CATALOG_EPOCH: u64 = 7;
const TABLE_ID: u64 = 101;
const TABLE_GENERATION: u64 = 11;
const ROW_ID: u64 = 100;
const TERMINAL_CONSTRAINT_ID: u64 = 301;
const OUTER_FLAGS: u32 = OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW;
const AGGREGATE_FLAGS: u32 = 1; // autocommit

const MINIMAL_ABORT_NULL_S2_TYPED_INSERT_HEX: &str = "47505544425459504544494e53310000010001000100010000000000080000005c01000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e44929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b010000004a000000060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c14610000000001000000010000000200000046000000010000000000000002000000696401000000010002000000170000000400010000000000010100000000000000000001000000020200000000010100000004000000000000000300000047000000010000000000000001060000007075626c69630c000000636f6465635f676f6c64656e00400000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c1461040000000400000000000000050000000400000000000000060000000400000000000000070000003400000001000000000000000000000000000000929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b0000000008000000050000000000000000";
const MINIMAL_ABORT_S7_HEX: &str = "475055444253374f5645524c415932000100020080020000000000000e000100da080000000000000100000001000000010000000200000002000000000000000000000000000000000000000000000000000000010000000000000000000000da000000000000008002000000000000800100000000000000040000000000002000000000000000200400000000000040010000000000006005000000000000c001000000000000200700000000000040000000000000006007000000000000000000000000000060070000000000000000000000000000600700000000000000000000000000006007000000000000000000000000000060070000000000000000000000000000600700000000000000000000000000006007000000000000a000000000000000000800000000000000000000000000000008000000000000da0000000000000007000000000000000700000000000000333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333334444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444455555555555555555555555555555555555555555555555555555555555555557276ad2d1a97797b8546de45a871b4d0366e7518709c2410f1bde8b17112a2ba5cb8ab23d90583e5636d29f9ecc2515347e188ecfedbde62db8422373d1947c672b5879d0651955fea53f059f8ea5039ed76c8dd50b371742b1ba9b2dc65bcb50000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006500000000000000004000000000000007000000000000000b000000000000000b000000000000006400000000000000650000000000000003000000000000000300000000000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000000000000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c14612222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222291fab6431c4dfa2710b5ec5c77bc0af9020b366883af4197881298be540aa8efb6b834870004be7608033e1d7060227592bdc5d901247bf83fb50168a21d360a543ddcda4aee9f51f6608840488874cc77fb384b7ce673e299bd395e1ee3ee57a3b8d74c4f15e358ce67655f569eccd12b16672422f4980647a4be19adc0eb4b21f7a199ad4f41bc29238c002bec95e71711df5977a55cb11059417df979ec6e00000000000000006400000000000000000000000000000003000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000020000000000000000000000010000000000000000000000000000000900000000000000c001000001000000000000000000000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e4426fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e44d8e7d13151369fecd867f58b18da8fd1bd6d4c6f34a2a2145dbda342d9c74eac929636fcb7deeb65c1fb78c22eaf21ec34f445564e81d66062ceb0685321291b55555555555555555555555555555555555555555555555555555555555555557276ad2d1a97797b8546de45a871b4d0366e7518709c2410f1bde8b17112a2ba1aae9e34636708b1979c0b3fbb8d46528538ea61b7cb342a92f4629670d244d50000000001030000650000000000000000400000000000000b000000000000000900000000000000ffffffffffffffff07000000000000000000000000000000e30dae616be5c3b69d3a30e1f3152ded1b23f1e69c85994a18ee5400eb8c146122222222222222222222222222222222222222222222222222222222222222228c34eeadf027fa10b94b864808dcfc7225d3ae64ecbe51aad400163dc9c4f064011832795927b066872c951aeda2f0f632807ef92a7bfcb7d6cb2b653342020cdce820d0e1e3cf0bcccad0e3c1bbf8b77dcbface61af57322607cb9fba272c8b01000000090202002d0100000000000000000000000000000c000000000000000900000000000000ffffffffffffffff070000000000000000000000000000002323232323232323232323232323232323232323232323232323232323232323242424242424242424242424242424242424242424242424242424242424242494533896b541eac6f48cb828d8707f75ddb88c994de96f41d094e53e19b993f5fdd903c9bc214b376f490b6af4c24ad5f6d939867167e4bc26a43d09b2946ff0324bed41517ef07c4d8d0ef17ec9c68bef99667ee711dadefa490655cdcffb7f00000000000000000100000000000000ffffffffffffffff000000000000000000000000010000000900000000000000ffffffffffffffff00000000000000000000000000000000010000000000000000000000010000000000000000000000da0000000000000000000000000000000000000000000000000000000000000091fab6431c4dfa2710b5ec5c77bc0af9020b366883af4197881298be540aa8efb6b834870004be7608033e1d7060227592bdc5d901247bf83fb50168a21d360ad55f99e7ded7eef597710b4bb09af7ad13d6f31710f66ea0e88e04e7ab75274147505544425459504544494d41474532020070000100000000000000010000000000000000000000600000000000000000000000000000000a0000000000000091fab6431c4dfa2710b5ec5c77bc0af9020b366883af4197881298be540aa8ef00000000000000000000000000000000000000000000000001000000000000000100000002000000170000000400000000000000000000000000000000000000d0000000000000000a00000000000000d5440a3977b9477b8d287446b3439f8f2afe384e02709dabe6f81eba73cfa63a00010000000000000000";

// Independent freeze values. S1, S2, and S7 use their dedicated complete literals below; the
// empty slots preserve aggregate-section numbering. They deliberately do not call the fixture
// builder: any builder drift changes the observed bytes rather than the expected result.
const MINIMAL_ABORT_SECTION_HEX: [&str; AGGREGATE_SECTION_COUNT] = [
    "",
    "",
    "",
    "000000000000000064000000000000000300000000000000ffffffff0000000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e44",
    "",
    "00000000000000000100000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e440301000000000000000000002d0100000000000032333530320000007276ad2d1a97797b8546de45a871b4d0366e7518709c2410f1bde8b17112a2ba0000000000000000000000000000000000000000000000000000000000000000",
    MINIMAL_ABORT_S7_HEX,
    "",
];

const MINIMAL_ABORT_AGGREGATE_HEADER_HEX: &str = "475055444254584e414747310000000001000200010001000100000008000000760c0000000000004d00000000000000010000000100000001000000000000000000000000000000000000000000000000000000000000000100000000000000";
const MINIMAL_ABORT_SECTION_HEADER_HEX: [&str; AGGREGATE_SECTION_COUNT] = [
    "01000000010000009000000000000000",
    "0200000001000000c401000000000000",
    "03000000000000000000000000000000",
    "04000000010000004000000000000000",
    "05000000000000000000000000000000",
    "06000000010000008800000000000000",
    "0700000001000000da08000000000000",
    "08000000000000000000000000000000",
];
const MINIMAL_ABORT_CHUNK_HEADER_HEX: &str = "47505544424f503105010300f60c00000000000000000000010000000000000000000000f60c000000000000c27ddfce53e60737e8f053442f37fbc120d19bafa687e36a2c06fd5d7638c2e6";
const MINIMAL_ABORT_STATUS_HEX: &str = "475055444253544154555332a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a24d00000000000000b0a02afa81b630cf1f799049326f878c22b7863b58b88849aa881ca963230df2010100000000000000000000000000000100000000000000fcabfbaaee638af1b3dba21332170c867f989160f24377caca74aaa222e2180e0000000000000000000000000000000000000000000000000000000000000000c27ddfce53e60737e8f053442f37fbc120d19bafa687e36a2c06fd5d7638c2e6";
const MINIMAL_ABORT_SECTION_ROOT_HEX: [&str; AGGREGATE_SECTION_COUNT] = [
    "f676b0ec41eefc32b6f3c148299855912124a475bfb70c87b942c0849a806a82",
    "a93802e6cc5f14afb0ccee5ac4649612eb3f47708167eaccade99587a80dd4d4",
    "948e2fa2ca884a0eaff824acde6511fe215a4c9f3590f713310568357668dcb3",
    "cbd1df698c22fa43375b7c974569f8babafdeddef845b14f399dacc80e609fe0",
    "dbb754af51c0b8dd7972ea10958c25955c6ce5f0f6c8708e51fac2191984ae58",
    "fcabfbaaee638af1b3dba21332170c867f989160f24377caca74aaa222e2180e",
    "731cccfc4d591f544906f7fc2a9805c9b215787680625247d7fc375aaf131bd3",
    "d843ed617704dd47e1759f53d860964e00207de7380561ba9e61455c5283aec1",
];
const MINIMAL_ABORT_AGGREGATE_ROOT_HEX: &str =
    "c27ddfce53e60737e8f053442f37fbc120d19bafa687e36a2c06fd5d7638c2e6";
const MINIMAL_ABORT_REQUEST_DIGEST_HEX: &str =
    "b0a02afa81b630cf1f799049326f878c22b7863b58b88849aa881ca963230df2";
const MINIMAL_ABORT_ROOT_DESCRIPTOR_HEX: &str =
    "5cb8ab23d90583e5636d29f9ecc2515347e188ecfedbde62db8422373d1947c6";
const MINIMAL_ABORT_S7_PAYLOAD_DIGEST_HEX: &str =
    "72b5879d0651955fea53f059f8ea5039ed76c8dd50b371742b1ba9b2dc65bcb5";
const MINIMAL_ABORT_TABLE_MANIFEST_HEX: &str =
    "21f7a199ad4f41bc29238c002bec95e71711df5977a55cb11059417df979ec6e";
const MINIMAL_ABORT_TARGET_TOKEN_HEX: &str =
    "dce820d0e1e3cf0bcccad0e3c1bbf8b77dcbface61af57322607cb9fba272c8b";
const MINIMAL_ABORT_NOT_NULL_TOKEN_HEX: &str =
    "324bed41517ef07c4d8d0ef17ec9c68bef99667ee711dadefa490655cdcffb7f";
const MINIMAL_ABORT_IMAGE_CONTENT_HEX: &str =
    "b6b834870004be7608033e1d7060227592bdc5d901247bf83fb50168a21d360a";
const MINIMAL_ABORT_OVERLAY_AFTER_HEX: &str =
    "7276ad2d1a97797b8546de45a871b4d0366e7518709c2410f1bde8b17112a2ba";

const MINIMAL_ABORT_S1_HEX: &str = "0000000000000000010000000100000026fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e4426fae30b557b8e22e36a2894e497466e572b572a9fd22c2623a08a334f649e4455555555555555555555555555555555555555555555555555555555555555557276ad2d1a97797b8546de45a871b4d0366e7518709c2410f1bde8b17112a2ba";

// The complete S7 stream is one independently frozen literal. No fixture constructor or digest
// helper participates in the expected-byte comparison below.

struct MinimalAbortFixture {
    sections: [Vec<u8>; AGGREGATE_SECTION_COUNT],
    stream: Vec<u8>,
    fragment_body: Vec<u8>,
    status: Vec<u8>,
    outer: gpu_db_wal::CanonicalPreApplyHeader,
    outcome: gpu_db_wal::CanonicalOutcome,
    aggregate_root: [u8; 32],
    request_digest: [u8; 32],
    root_descriptor_digest: [u8; 32],
    s7_payload_digest: [u8; 32],
    table_manifest_digest: [u8; 32],
    target_dependency_digest: [u8; 32],
    terminal_dependency_digest: [u8; 32],
    image_content_digest: [u8; 32],
    overlay_after: [u8; 32],
    section_roots: [[u8; 32]; AGGREGATE_SECTION_COUNT],
}

struct SwitchableCanonicalSource<'a> {
    before: &'a [u8],
    after: &'a [u8],
    use_after: Cell<bool>,
}

impl CanonicalTypedInsertReadAt for SwitchableCanonicalSource<'_> {
    fn len(&self) -> u64 {
        self.before.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let source = if self.use_after.get() {
            self.after
        } else {
            self.before
        };
        let start = usize::try_from(offset)
            .map_err(|_| EngineError::Durability("test source offset is not addressable".into()))?;
        let end = start
            .checked_add(out.len())
            .filter(|end| *end <= source.len())
            .ok_or_else(|| EngineError::Durability("test source read exceeds its bytes".into()))?;
        out.copy_from_slice(&source[start..end]);
        Ok(())
    }
}

fn minimal_abort_fixture() -> MinimalAbortFixture {
    let s2_record = decode_hex(MINIMAL_ABORT_NULL_S2_TYPED_INSERT_HEX);
    assert_eq!(s2_record.len(), 448, "the fixed S2 fixture length drifted");
    let decoded_s2 = crate::typed_insert_batch::decode_canonical_typed_insert_record(&s2_record)
        .expect("the fixed S2 fixture decodes before deriving its S7 bindings");
    let s2_facts = decoded_s2.facts();
    let table_oid = s2_facts.target.oid;
    let schema_digest = s2_facts.target.schema_digest;
    let returning_layout_digest = s2_facts.returning.digest;
    let typed_statement_digest: [u8; 32] = s2_record[36..68]
        .try_into()
        .expect("canonical S2 statement digest width");

    let catalog_digest = [0x33; 32];
    let initial_table_root = [0x22; 32];
    let initial_database_root = [0x44; 32];
    let initial_overlay = [0x55; 32];

    let final_image = build_zero_row_final_image();
    let image_layout_digest: [u8; 32] = final_image[64..96]
        .try_into()
        .expect("final image layout digest width");
    let image_content_digest = v2_digest(
        b"gpu-db/write001/s7-image-content/v2",
        &[&(final_image.len() as u64).to_le_bytes(), &final_image],
    );
    let mut image_descriptor = vec![0; 160];
    write_u32(&mut image_descriptor, 0, 0);
    write_u32(&mut image_descriptor, 4, 0);
    write_u32(&mut image_descriptor, 8, 1);
    write_u32(&mut image_descriptor, 12, 0);
    write_u32(&mut image_descriptor, 16, 0);
    write_u32(&mut image_descriptor, 20, 1);
    write_u64(&mut image_descriptor, 24, 0);
    write_u64(&mut image_descriptor, 32, final_image.len() as u64);
    image_descriptor[64..96].copy_from_slice(&image_layout_digest);
    image_descriptor[96..128].copy_from_slice(&image_content_digest);
    let image_descriptor_digest = v2_digest(
        b"gpu-db/write001/s7-image-descriptor/v2",
        &[&image_descriptor[..128], &[0; 32]],
    );
    image_descriptor[128..160].copy_from_slice(&image_descriptor_digest);

    let target_name = qualified_name_digest(b"public", b"codec_golden");
    let target_identity = v2_digest(
        b"gpu-db/write001/s7-table-object/v2",
        &[
            &[1],
            &TABLE_ID.to_le_bytes(),
            &table_oid.to_le_bytes(),
            &CATALOG_EPOCH.to_le_bytes(),
            &TABLE_GENERATION.to_le_bytes(),
            &schema_digest,
            &initial_table_root,
            &target_name,
        ],
    );
    let mut target_dependency = vec![0; 224];
    write_u32(&mut target_dependency, 0, 0);
    target_dependency[4] = 1;
    target_dependency[5] = 3;
    write_u64(&mut target_dependency, 8, TABLE_ID);
    write_u32(&mut target_dependency, 16, table_oid);
    write_u32(&mut target_dependency, 20, 0);
    write_u64(&mut target_dependency, 24, TABLE_GENERATION);
    write_u64(&mut target_dependency, 32, 9);
    write_u32(&mut target_dependency, 40, ABSENT_U32);
    write_u32(&mut target_dependency, 44, ABSENT_U32);
    write_u64(&mut target_dependency, 48, CATALOG_EPOCH);
    target_dependency[64..96].copy_from_slice(&schema_digest);
    target_dependency[96..128].copy_from_slice(&initial_table_root);
    target_dependency[128..160].copy_from_slice(&target_name);
    target_dependency[160..192].copy_from_slice(&target_identity);
    let target_dependency_digest = dependency_digest(&target_dependency);
    target_dependency[192..224].copy_from_slice(&target_dependency_digest);

    let terminal_name = synthesized_not_null_name_digest(1, TABLE_ID, 0);
    let terminal_shape = [0x23; 32];
    let terminal_root = [0x24; 32];
    let terminal_identity = v2_digest(
        b"gpu-db/write001/s7-constraint-object/v2",
        &[
            &[9],
            &TERMINAL_CONSTRAINT_ID.to_le_bytes(),
            &0_u32.to_le_bytes(),
            &TABLE_ID.to_le_bytes(),
            &CATALOG_EPOCH.to_le_bytes(),
            &12_u64.to_le_bytes(),
            &terminal_shape,
            &terminal_root,
            &terminal_name,
        ],
    );
    let mut terminal_dependency = vec![0; 224];
    write_u32(&mut terminal_dependency, 0, 1);
    terminal_dependency[4] = 9;
    terminal_dependency[5] = 2;
    write_u16(&mut terminal_dependency, 6, 2);
    write_u64(&mut terminal_dependency, 8, TERMINAL_CONSTRAINT_ID);
    write_u32(&mut terminal_dependency, 20, 0);
    write_u64(&mut terminal_dependency, 24, 12);
    write_u64(&mut terminal_dependency, 32, 9);
    write_u32(&mut terminal_dependency, 40, ABSENT_U32);
    write_u32(&mut terminal_dependency, 44, ABSENT_U32);
    write_u64(&mut terminal_dependency, 48, CATALOG_EPOCH);
    terminal_dependency[64..96].copy_from_slice(&terminal_shape);
    terminal_dependency[96..128].copy_from_slice(&terminal_root);
    terminal_dependency[128..160].copy_from_slice(&terminal_name);
    terminal_dependency[160..192].copy_from_slice(&terminal_identity);
    let terminal_dependency_digest = dependency_digest(&terminal_dependency);
    terminal_dependency[192..224].copy_from_slice(&terminal_dependency_digest);

    let mut target_use = vec![0; 32];
    write_u32(&mut target_use, 0, 0);
    write_u32(&mut target_use, 4, 0);
    write_u16(&mut target_use, 8, 1);
    write_u32(&mut target_use, 12, 0);
    write_u32(&mut target_use, 16, ABSENT_U32);
    write_u32(&mut target_use, 20, ABSENT_U32);

    let mut terminal_use = vec![0; 32];
    write_u32(&mut terminal_use, 0, 0);
    write_u32(&mut terminal_use, 4, 1);
    write_u16(&mut terminal_use, 8, 9);
    write_u32(&mut terminal_use, 12, 0);
    write_u32(&mut terminal_use, 16, ABSENT_U32);
    write_u32(&mut terminal_use, 20, ABSENT_U32);
    let dependency_root = v2_digest(
        b"gpu-db/write001/s7-statement-dependencies/v2",
        &[
            &0_u32.to_le_bytes(),
            &2_u32.to_le_bytes(),
            &target_use,
            &target_dependency_digest,
            &terminal_use,
            &terminal_dependency_digest,
        ],
    );

    let mut s4 = vec![0; 64];
    write_u64(&mut s4, 8, ROW_ID);
    s4[16] = 3;
    write_u32(&mut s4, 20, 0);
    write_u32(&mut s4, 24, ABSENT_U32);
    s4[32..64].copy_from_slice(&typed_statement_digest);
    let disposition_root = v2_digest(
        b"gpu-db/write001/s7-statement-dispositions/v2",
        &[&0_u32.to_le_bytes(), &1_u32.to_le_bytes(), &s4],
    );
    let sequence_root = v2_digest(
        b"gpu-db/write001/s7-statement-sequences/v2",
        &[&0_u32.to_le_bytes(), &0_u32.to_le_bytes()],
    );
    let projection_root = v2_digest(
        b"gpu-db/write001/s7-statement-projections/v2",
        &[&0_u32.to_le_bytes(), &0_u32.to_le_bytes()],
    );
    let s2_digest = v2_digest(
        b"gpu-db/write001/s7-s2-record/v2",
        &[&(s2_record.len() as u32).to_le_bytes(), &s2_record],
    );
    let overlay_after = v2_digest(
        b"gpu-db/write001/s7-statement-overlay-root/v2",
        &[
            &initial_overlay,
            &0_u32.to_le_bytes(),
            &typed_statement_digest,
            &s2_digest,
            &disposition_root,
            &sequence_root,
            &dependency_root,
            &projection_root,
        ],
    );

    let mut s1 = vec![0; 144];
    s1[8] = 1;
    write_u32(&mut s1, 12, 1);
    s1[16..48].copy_from_slice(&typed_statement_digest);
    s1[48..80].copy_from_slice(&typed_statement_digest);
    s1[80..112].copy_from_slice(&initial_overlay);
    s1[112..144].copy_from_slice(&overlay_after);

    let statement_outcome = canonical_abort_outcome(overlay_after);
    let mut s6 = vec![0; 136];
    write_u16(&mut s6, 8, 1);
    s6[12..44].copy_from_slice(&typed_statement_digest);
    s6[44..136].copy_from_slice(&statement_outcome);
    let s6_digest = v2_digest(b"gpu-db/write001/s7-s6-entry/v2", &[&s6]);

    let mut table_disposition = vec![0; 32];
    write_u64(&mut table_disposition, 8, ROW_ID);
    table_disposition[24] = 3;

    let mut resolution = vec![0; 320];
    write_u32(&mut resolution, 16, 0);
    write_u32(&mut resolution, 28, 1);
    write_u32(&mut resolution, 44, 2);
    write_u32(&mut resolution, 56, 1);
    write_u64(&mut resolution, 72, 9);
    write_u32(&mut resolution, 80, s2_record.len() as u32);
    write_u32(&mut resolution, 84, 1);
    resolution[96..128].copy_from_slice(&typed_statement_digest);
    resolution[128..160].copy_from_slice(&typed_statement_digest);
    resolution[160..192].copy_from_slice(&s2_digest);
    resolution[192..224].copy_from_slice(&returning_layout_digest);
    resolution[224..256].copy_from_slice(&initial_overlay);
    resolution[256..288].copy_from_slice(&overlay_after);
    resolution[288..320].copy_from_slice(&s6_digest);

    let transition_root = v2_digest(
        b"gpu-db/write001/s7-table-transition-root/v2",
        &[&TABLE_ID.to_le_bytes(), &0_u32.to_le_bytes()],
    );
    let index_effect_root = v2_digest(
        b"gpu-db/write001/s7-table-index-effect-root/v2",
        &[&TABLE_ID.to_le_bytes(), &0_u32.to_le_bytes()],
    );
    let mut table = vec![0; 384];
    write_u64(&mut table, 8, TABLE_ID);
    write_u32(&mut table, 16, table_oid);
    write_u64(&mut table, 24, CATALOG_EPOCH);
    write_u64(&mut table, 32, TABLE_GENERATION);
    write_u64(&mut table, 40, TABLE_GENERATION);
    write_u64(&mut table, 48, ROW_ID);
    write_u64(&mut table, 56, ROW_ID + 1);
    write_u64(&mut table, 64, 3);
    write_u64(&mut table, 72, 3);
    write_u32(&mut table, 84, 1);
    write_u32(&mut table, 112, 0);
    write_u32(&mut table, 116, 1);
    table[128..160].copy_from_slice(&schema_digest);
    table[160..192].copy_from_slice(&initial_table_root);
    table[192..224].copy_from_slice(&initial_table_root);
    table[224..256].copy_from_slice(&image_layout_digest);
    table[256..288].copy_from_slice(&image_content_digest);
    table[288..320].copy_from_slice(&transition_root);
    table[320..352].copy_from_slice(&index_effect_root);
    let table_manifest_digest = v2_digest(
        b"gpu-db/write001/s7-table-manifest/v2",
        &[
            &table[..352],
            &[0; 32],
            &target_dependency_digest,
            &table_disposition,
            &image_descriptor_digest,
        ],
    );
    table[352..384].copy_from_slice(&table_manifest_digest);

    let counts = [1_u32, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1];
    let fixed_widths = [384_u64, 32, 320, 224, 32, 384, 112, 192, 192, 128, 128, 160];
    let count_index = [0_usize, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut directory_offsets = [(0_u64, 0_u64); 14];
    let mut cursor = 640_u64;
    for index in 0..14 {
        let bytes = if index < 12 {
            u64::from(counts[count_index[index]]) * fixed_widths[index]
        } else if index == 12 {
            0
        } else {
            final_image.len() as u64
        };
        directory_offsets[index] = (cursor, bytes);
        cursor += bytes;
    }
    assert_eq!(cursor, 2_266, "minimal S7 geometry is frozen");

    let mut s7 = vec![0; 640];
    s7[..16].copy_from_slice(b"GPUDBS7OVERLAY2\0");
    write_u16(&mut s7, 16, 1);
    write_u16(&mut s7, 18, 2);
    write_u32(&mut s7, 20, 640);
    write_u16(&mut s7, 28, 14);
    write_u16(&mut s7, 30, 1);
    write_u64(&mut s7, 32, cursor);
    for (index, count) in counts.into_iter().enumerate() {
        write_u32(&mut s7, 40 + index * 4, count);
    }
    write_u64(&mut s7, 96, final_image.len() as u64);
    for (index, (offset, bytes)) in directory_offsets.into_iter().enumerate() {
        write_u64(&mut s7, 104 + index * 16, offset);
        write_u64(&mut s7, 112 + index * 16, bytes);
    }
    write_u64(&mut s7, 328, CATALOG_EPOCH);
    write_u64(&mut s7, 336, CATALOG_EPOCH);
    s7[344..376].copy_from_slice(&catalog_digest);
    s7[376..408].copy_from_slice(&catalog_digest);
    s7[408..440].copy_from_slice(&initial_database_root);
    s7[440..472].copy_from_slice(&initial_database_root);
    s7[472..504].copy_from_slice(&initial_overlay);
    s7[504..536].copy_from_slice(&overlay_after);
    let root_descriptor_digest = v2_digest(
        b"gpu-db/write001/s7-root-descriptor/v2",
        &[
            &1_u16.to_le_bytes(),
            &CATALOG_EPOCH.to_le_bytes(),
            &CATALOG_EPOCH.to_le_bytes(),
            &catalog_digest,
            &catalog_digest,
            &initial_database_root,
            &initial_database_root,
            &initial_overlay,
            &overlay_after,
            &1_u32.to_le_bytes(),
            &0_u32.to_le_bytes(),
            &table[4..8],
            &TABLE_ID.to_le_bytes(),
            &TABLE_GENERATION.to_le_bytes(),
            &TABLE_GENERATION.to_le_bytes(),
            &initial_table_root,
            &initial_table_root,
            &table_manifest_digest,
        ],
    );
    s7[536..568].copy_from_slice(&root_descriptor_digest);
    s7.extend_from_slice(&table);
    s7.extend_from_slice(&table_disposition);
    s7.extend_from_slice(&resolution);
    s7.extend_from_slice(&target_dependency);
    s7.extend_from_slice(&terminal_dependency);
    s7.extend_from_slice(&target_use);
    s7.extend_from_slice(&terminal_use);
    s7.extend_from_slice(&image_descriptor);
    s7.extend_from_slice(&final_image);
    assert_eq!(s7.len(), cursor as usize, "minimal S7 directory coverage");
    let s7_payload_digest = v2_digest(
        b"gpu-db/write001/s7-payload/v2",
        &[
            &(s7.len() as u64).to_le_bytes(),
            &s7[..568],
            &[0; 32],
            &s7[600..640],
            &s7[640..],
        ],
    );
    s7[568..600].copy_from_slice(&s7_payload_digest);

    let request_digest = v2_digest(
        b"gpu-db/write001/aggregate-request/v2",
        &[
            &[1],
            &1_u32.to_le_bytes(),
            &0_u32.to_le_bytes(),
            &typed_statement_digest,
            &0_u32.to_le_bytes(),
        ],
    );
    let sections = [
        s1,
        [
            u32::try_from(s2_record.len())
                .expect("S2 test record fits u32")
                .to_le_bytes()
                .as_slice(),
            s2_record.as_slice(),
        ]
        .concat(),
        Vec::new(),
        s4,
        Vec::new(),
        s6,
        s7,
        Vec::new(),
    ];
    let section_entries = [1_u32, 1, 0, 1, 0, 1, 1, 0];
    let section_headers: [[u8; 16]; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|index| {
        let mut header = [0; 16];
        header[..2].copy_from_slice(&u16::try_from(index + 1).unwrap().to_le_bytes());
        header[4..8].copy_from_slice(&section_entries[index].to_le_bytes());
        header[8..16].copy_from_slice(&(sections[index].len() as u64).to_le_bytes());
        header
    });
    let section_region_bytes = section_headers
        .iter()
        .zip(sections.iter())
        .map(|(_, payload)| 16_u64 + payload.len() as u64)
        .sum::<u64>();
    let mut aggregate_header = [0; 96];
    aggregate_header[..16].copy_from_slice(AGGREGATE_STREAM_MAGIC);
    write_u16(&mut aggregate_header, 16, AGGREGATE_FORMAT_VERSION);
    write_u16(&mut aggregate_header, 18, 2);
    write_u16(&mut aggregate_header, 20, AGGREGATE_FORMAT_VERSION);
    write_u16(&mut aggregate_header, 22, AGGREGATE_FORMAT_VERSION);
    write_u32(&mut aggregate_header, 24, AGGREGATE_FLAGS);
    write_u16(&mut aggregate_header, 28, AGGREGATE_SECTION_COUNT as u16);
    write_u64(&mut aggregate_header, 32, section_region_bytes);
    write_u64(&mut aggregate_header, 40, STABLE_TRANSACTION_ID);
    write_u32(&mut aggregate_header, 48, 1);
    write_u32(&mut aggregate_header, 52, 1);
    write_u64(&mut aggregate_header, 56, 1);
    write_u32(&mut aggregate_header, 88, 1);
    let section_roots = std::array::from_fn(|index| {
        v1_digest(
            b"gpu-db/write001/aggregate-section/v1",
            &[&section_headers[index], &sections[index]],
        )
    });
    let aggregate_root = v1_digest(
        b"gpu-db/write001/aggregate-root/v1",
        &[
            &aggregate_header,
            &section_roots[0],
            &section_roots[1],
            &section_roots[2],
            &section_roots[3],
            &section_roots[4],
            &section_roots[5],
            &section_roots[6],
            &section_roots[7],
        ],
    );
    let mut stream = aggregate_header.to_vec();
    for (header, payload) in section_headers.iter().zip(sections.iter()) {
        stream.extend_from_slice(header);
        stream.extend_from_slice(payload);
    }
    stream.extend_from_slice(&aggregate_root);
    assert_eq!(
        stream.len(),
        3_318,
        "minimal aggregate stream geometry is frozen"
    );

    let mut fragment_body = vec![0; AGGREGATE_CHUNK_HEADER_BYTES as usize];
    fragment_body[..8].copy_from_slice(AGGREGATE_CHUNK_MAGIC);
    fragment_body[8] = ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE;
    fragment_body[9] = AGGREGATE_FORMAT_VERSION as u8;
    write_u16(
        &mut fragment_body,
        10,
        AGGREGATE_CHUNK_FLAG_FIRST | AGGREGATE_CHUNK_FLAG_LAST,
    );
    write_u64(&mut fragment_body, 12, stream.len() as u64);
    write_u32(&mut fragment_body, 24, 1);
    write_u32(&mut fragment_body, 36, stream.len() as u32);
    fragment_body[44..76].copy_from_slice(&aggregate_root);
    fragment_body.extend_from_slice(&stream);

    let status = TypedInsertStatusV2 {
        database_id: [0xa1; 16],
        timeline_id: [0xa2; 16],
        txn_id: STABLE_TRANSACTION_ID,
        request_digest,
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: 1,
        response_artifact_count: 0,
        statement_outcome_root: section_roots[5],
        response_root: [0; 32],
        aggregate_root,
    };
    let mut status_bytes = vec![0; AGGREGATE_STATUS_V2_BYTES as usize];
    encode_status_v2(&status, &mut status_bytes).expect("test-only STATUS2 fixture encodes");
    let outer = gpu_db_wal::CanonicalPreApplyHeader {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: [0xa1; 16],
            cluster_id: [0xa3; 16],
            timeline_id: [0xa2; 16],
            format_epoch: 5,
        },
        leader_epoch: 6,
        commit_seq: COMMIT_SEQUENCE,
        stable_transaction_id: STABLE_TRANSACTION_ID,
        request_digest,
        isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
        flags: OUTER_FLAGS,
        catalog_before_epoch: CATALOG_EPOCH,
        catalog_after_epoch: CATALOG_EPOCH,
        catalog_before_digest: catalog_digest,
        catalog_after_digest: catalog_digest,
        operation_count: 2,
        table_block_count: 1,
        allocator_high_water: 0,
    };
    let outcome = gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
        affected_rows: 0,
        sqlstate: Some(*b"23502"),
        constraint_id: TERMINAL_CONSTRAINT_ID,
        target_digest: aggregate_root,
        returning_digest: [0; 32],
    };
    MinimalAbortFixture {
        sections,
        stream,
        fragment_body,
        status: status_bytes,
        outer,
        outcome,
        aggregate_root,
        request_digest,
        root_descriptor_digest,
        s7_payload_digest,
        table_manifest_digest,
        target_dependency_digest,
        terminal_dependency_digest,
        image_content_digest,
        overlay_after,
        section_roots,
    }
}

fn build_zero_row_final_image() -> Vec<u8> {
    let storage = [2, 0, 0, 0];
    let validity = [0_u8];
    let values = [1, 0, 0, 0, 0, 0, 0, 0, 0];
    let vector_digest = v2_digest(
        b"gpu-db/write001/typed-vector/v2",
        &[&storage, &0_u32.to_le_bytes(), &validity, &values],
    );
    let mut descriptor = [0_u8; 96];
    write_u32(&mut descriptor, 0, 0);
    write_u32(&mut descriptor, 4, 0);
    write_u32(&mut descriptor, 8, 1);
    write_i16(&mut descriptor, 16, 1);
    descriptor[20..24].copy_from_slice(&storage);
    write_u32(&mut descriptor, 24, 23);
    write_i16(&mut descriptor, 28, 4);
    write_u64(&mut descriptor, 48, 208);
    write_u64(&mut descriptor, 56, 10);
    descriptor[64..96].copy_from_slice(&vector_digest);
    let layout_digest = v2_digest(
        b"gpu-db/write001/image-layout/v2",
        &[
            &0_u32.to_le_bytes(),
            &1_u32.to_le_bytes(),
            &0_u64.to_le_bytes(),
            &descriptor[..48],
            &0_u64.to_le_bytes(),
            &0_u64.to_le_bytes(),
            &[0; 32],
        ],
    );
    let mut image = vec![0; 112];
    image[..16].copy_from_slice(b"GPUDBTYPEDIMAGE2");
    write_u16(&mut image, 16, 2);
    write_u16(&mut image, 18, 112);
    write_u32(&mut image, 20, 1);
    write_u32(&mut image, 28, 1);
    write_u64(&mut image, 40, 96);
    write_u64(&mut image, 56, 10);
    image[64..96].copy_from_slice(&layout_digest);
    image.extend_from_slice(&descriptor);
    image.extend_from_slice(&validity);
    image.extend_from_slice(&values);
    image
}

fn canonical_abort_outcome(target_digest: [u8; 32]) -> [u8; 92] {
    let outcome = gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::AbortError,
        affected_rows: 0,
        sqlstate: Some(*b"23502"),
        constraint_id: TERMINAL_CONSTRAINT_ID,
        target_digest,
        returning_digest: [0; 32],
    };
    let mut bytes = [0; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
    gpu_db_wal::encode_canonical_outcome_into_exact(&outcome, &mut bytes)
        .expect("fixed test abort outcome encodes");
    bytes
}

fn qualified_name_digest(schema: &[u8], object: &[u8]) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/s7-qualified-name/v2",
        &[
            &(schema.len() as u32).to_le_bytes(),
            schema,
            &(object.len() as u32).to_le_bytes(),
            object,
        ],
    )
}

fn synthesized_not_null_name_digest(owner_kind: u8, owner: u64, source_ordinal: u32) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/s7-synthesized-not-null-name/v2",
        &[
            &[owner_kind],
            &owner.to_le_bytes(),
            &source_ordinal.to_le_bytes(),
        ],
    )
}

fn dependency_digest(bytes: &[u8]) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&bytes[..192], &[0; 32], &[0; 32], &[0; 32]],
    )
}

fn v2_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update(field);
    }
    hasher.finalize().into()
}

fn v1_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert!(
        value.len().is_multiple_of(2),
        "fixture hex stays byte-aligned"
    );
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("fixture hex is ASCII"), 16)
                .expect("fixture hex digit")
        })
        .collect()
}

fn write_u16(target: &mut [u8], offset: usize, value: u16) {
    target[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_i16(target: &mut [u8], offset: usize, value: i16) {
    target[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(target: &mut [u8], offset: usize, value: u32) {
    target[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(target: &mut [u8], offset: usize, value: u64) {
    target[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn minimal_abort_vector_is_a_single_chunk_bounded_v2_pass_zero_fixture() {
    let fixture = minimal_abort_fixture();
    assert_minimal_abort_literal_freeze(&fixture);
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ];
    let measure = measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
        .expect("minimal one-table terminal-abort fixture passes v2 raw proof");
    assert_eq!(
        measure.terminal_kind,
        gpu_db_wal::CanonicalOutcomeKind::AbortError
    );
    assert_eq!(measure.terminal_sqlstate, Some(*b"23502"));
    assert_eq!(measure.terminal_constraint_id, TERMINAL_CONSTRAINT_ID);
    fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        .expect("minimal abort fills its exact retained S1--S8 owners");
    close_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        .expect("minimal abort closes every witness-free S1--S8 fact");
    assert_eq!(fixture.sections[6].len(), 2_266);
    assert_eq!(fixture.stream.len(), 3_318);
    assert_eq!(fixture.fragment_body.len(), 3_394);
    assert_eq!(AGGREGATE_ROOT_TRAILER_BYTES, 32);
    assert_eq!(AGGREGATE_SECTION_HEADER_BYTES, 16);
}

#[test]
fn retained_source_copy_failure_drains_before_the_same_golden_can_retry() {
    let fixture = minimal_abort_fixture();
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ];
    // This golden has precisely one S2 and one image source copy. Each injected failure must
    // drop every already-reserved/direct decoded owner before the same bytes can be retried.
    for attempt in 1..=2 {
        let failed = fail_retained_source_copy_at_for_test(attempt, || {
            fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        });
        assert!(
            failed.is_err(),
            "injected source-copy failure {attempt} rejects"
        );
        fill_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
            .expect("the retry has no partial retained graph or leaked source owner");
    }
}

#[test]
fn dependency_token_digest_sabotage_rejects_after_the_bounded_source_is_proved() {
    let fixture = minimal_abort_fixture();
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ];
    // S7 fixed directory order: header, one table, one table-disposition, one resolution,
    // then the first dependency token.  The original source itself remains canonical; only the
    // candidate token proof is sabotaged, so this exercises the recomputation rather than a
    // surrounding framing/root failure.
    let dependency_start = 640 + 384 + 32 + 320;
    let mut token: [u8; 224] = fixture.sections[6][dependency_start..dependency_start + 224]
        .try_into()
        .expect("minimal golden has its first dependency token");
    validate_dependency_token_digest_for_test(&fixture.outer, &fragments, &token)
        .expect("original dependency token follows its exact digest preimage");
    token[192] ^= 1;
    assert!(
        validate_dependency_token_digest_for_test(&fixture.outer, &fragments, &token).is_err(),
        "one-byte dependency-token digest sabotage rejects without trusting the token bytes"
    );
}

#[test]
fn s2_source_fingerprint_refuses_a_changed_chunk_backed_source_after_measurement() {
    let fixture = minimal_abort_fixture();
    let record = &fixture.sections[1][4..];
    let mut changed = record.to_vec();
    changed[0] ^= 1;
    let source = SwitchableCanonicalSource {
        before: record,
        after: &changed,
        use_after: Cell::new(false),
    };
    let measure = measure_decoded_canonical_typed_insert_published_only_from_source(&source)
        .expect("original golden S2 source measures");
    source.use_after.set(true);
    let mut copy = vec![0; record.len()];
    assert!(
        copy_decoded_canonical_typed_insert_published_only_after_measure(
            &source, measure, &mut copy
        )
        .is_err(),
        "post-measure source mutation cannot reuse the S2 reservation/fingerprint"
    );
}

fn assert_minimal_abort_literal_freeze(fixture: &MinimalAbortFixture) {
    for (index, (actual, expected_hex)) in fixture
        .sections
        .iter()
        .zip(MINIMAL_ABORT_SECTION_HEX)
        .enumerate()
    {
        let expected = expected_minimal_abort_section(index, expected_hex);
        assert_eq!(
            actual.len(),
            expected.len(),
            "S{} literal length drifted",
            index + 1
        );
        let mismatch = actual
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual != expected);
        assert!(
            mismatch.is_none(),
            "S{} literal drifted first at byte {:?}: actual={:?} expected={:?}",
            index + 1,
            mismatch,
            mismatch.map(|offset| actual[offset]),
            mismatch.map(|offset| expected[offset]),
        );
    }

    let aggregate_header = decode_hex(MINIMAL_ABORT_AGGREGATE_HEADER_HEX);
    assert_eq!(
        aggregate_header.len(),
        96,
        "aggregate header geometry is frozen"
    );
    assert_eq!(
        &fixture.stream[..aggregate_header.len()],
        aggregate_header,
        "aggregate header literal drifted"
    );

    let mut cursor = aggregate_header.len();
    for (index, ((section, expected_header_hex), expected_section_hex)) in fixture
        .sections
        .iter()
        .zip(MINIMAL_ABORT_SECTION_HEADER_HEX)
        .zip(MINIMAL_ABORT_SECTION_HEX)
        .enumerate()
    {
        let expected_header = decode_hex(expected_header_hex);
        assert_eq!(
            expected_header.len(),
            AGGREGATE_SECTION_HEADER_BYTES as usize
        );
        let header_end = cursor + expected_header.len();
        assert_eq!(
            &fixture.stream[cursor..header_end],
            expected_header,
            "S{} aggregate section header literal drifted",
            index + 1
        );
        cursor = header_end;
        let expected_section = expected_minimal_abort_section(index, expected_section_hex);
        let section_end = cursor + section.len();
        assert_eq!(
            &fixture.stream[cursor..section_end],
            expected_section,
            "S{} aggregate payload literal drifted",
            index + 1
        );
        cursor = section_end;
    }
    let expected_root = digest_literal(MINIMAL_ABORT_AGGREGATE_ROOT_HEX);
    assert_eq!(
        &fixture.stream[cursor..],
        expected_root,
        "aggregate root trailer literal drifted"
    );
    assert_eq!(
        cursor + AGGREGATE_ROOT_TRAILER_BYTES as usize,
        fixture.stream.len()
    );

    let expected_chunk_header = decode_hex(MINIMAL_ABORT_CHUNK_HEADER_HEX);
    assert_eq!(
        &fixture.fragment_body[..expected_chunk_header.len()],
        expected_chunk_header,
        "chunk header literal drifted"
    );
    assert_eq!(
        &fixture.fragment_body[expected_chunk_header.len()..],
        &fixture.stream,
        "chunk payload must be the frozen aggregate stream"
    );
    assert_eq!(
        fixture.status,
        decode_hex(MINIMAL_ABORT_STATUS_HEX),
        "STATUS2 literal drifted"
    );

    for (index, (actual, expected_hex)) in fixture
        .section_roots
        .iter()
        .zip(MINIMAL_ABORT_SECTION_ROOT_HEX)
        .enumerate()
    {
        assert_eq!(
            actual,
            &digest_literal(expected_hex),
            "S{} section root literal drifted",
            index + 1
        );
    }
    assert_eq!(fixture.aggregate_root, expected_root);
    assert_eq!(
        fixture.request_digest,
        digest_literal(MINIMAL_ABORT_REQUEST_DIGEST_HEX)
    );
    assert_eq!(
        fixture.root_descriptor_digest,
        digest_literal(MINIMAL_ABORT_ROOT_DESCRIPTOR_HEX)
    );
    assert_eq!(
        fixture.s7_payload_digest,
        digest_literal(MINIMAL_ABORT_S7_PAYLOAD_DIGEST_HEX)
    );
    assert_eq!(
        fixture.table_manifest_digest,
        digest_literal(MINIMAL_ABORT_TABLE_MANIFEST_HEX)
    );
    assert_eq!(
        fixture.target_dependency_digest,
        digest_literal(MINIMAL_ABORT_TARGET_TOKEN_HEX)
    );
    assert_eq!(
        fixture.terminal_dependency_digest,
        digest_literal(MINIMAL_ABORT_NOT_NULL_TOKEN_HEX)
    );
    assert_eq!(
        fixture.image_content_digest,
        digest_literal(MINIMAL_ABORT_IMAGE_CONTENT_HEX)
    );
    assert_eq!(
        fixture.overlay_after,
        digest_literal(MINIMAL_ABORT_OVERLAY_AFTER_HEX)
    );
}

fn expected_minimal_abort_section(index: usize, expected_hex: &str) -> Vec<u8> {
    if index == 0 {
        return decode_hex(MINIMAL_ABORT_S1_HEX);
    }
    if index == 1 {
        let mut expected = decode_hex("c0010000");
        expected.extend_from_slice(&decode_hex(MINIMAL_ABORT_NULL_S2_TYPED_INSERT_HEX));
        return expected;
    }
    if index == 6 {
        let expected = decode_hex(MINIMAL_ABORT_S7_HEX);
        assert_eq!(expected.len(), 2_266, "minimal S7 literal geometry drifted");
        return expected;
    }
    decode_hex(expected_hex)
}

fn digest_literal(hex: &str) -> [u8; 32] {
    decode_hex(hex)
        .try_into()
        .expect("fixed digest literal has 32 bytes")
}

#[test]
fn golden_builder_stays_test_only_and_header_uses_closed_semantics_selector() {
    let facade = include_str!("../semantics_v2.rs");
    let codec = include_str!("../codec.rs");
    assert!(facade.contains("#[cfg(test)]"));
    assert!(facade.contains("semantics_v2/goldens.rs"));
    let encoder = codec
        .split("fn encoded_headers")
        .nth(1)
        .and_then(|tail| tail.split("fn aggregate_status_roots").next())
        .expect("codec keeps one bounded header encoder");
    assert!(encoder.contains("writer.u16(view.semantics.wire_version())?"));
    assert!(codec.contains("AGGREGATE_SEMANTICS_V1 | AGGREGATE_SEMANTICS_V2"));
    assert!(!encoder.contains("AGGREGATE_FLAG_RETAINED_RESPONSE"));
    assert!(!encoder.contains("S8"));
}
