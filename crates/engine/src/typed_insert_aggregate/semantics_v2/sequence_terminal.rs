//! Fixed-width S5 terminal sequence-state override.
//!
//! A sequence effect remains owned by its existing S2/S5 row/default entry. When a later
//! `ALTER SEQUENCE ... RESTART` is the final operation on that same sequence, this suffix binds
//! the overwritten terminal state to the transaction request without manufacturing another S5
//! effect, statement, sequence owner, or recovery action.

use sha2::{Digest, Sha256};

pub(super) const SEQUENCE_RESTART_TAIL_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SequenceRestartTail {
    pub(super) owner_operation_ordinal: u32,
    pub(super) sequence_oid: u32,
    pub(super) last_value: i64,
    pub(super) owner_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(super) descriptor_digest: gpu_db_wal::CanonicalDigest,
    pub(super) witness_digest: gpu_db_wal::CanonicalDigest,
}

pub(super) fn encode(
    request_digest: gpu_db_wal::CanonicalDigest,
    tail: SequenceRestartTail,
) -> Result<[u8; SEQUENCE_RESTART_TAIL_BYTES], crate::EngineError> {
    if request_digest == [0; 32]
        || tail.owner_operation_ordinal == u32::MAX
        || tail.sequence_oid == 0
        || tail.owner_statement_digest == [0; 32]
        || tail.descriptor_digest == [0; 32]
    {
        return Err(error("terminal sequence restart identity is invalid"));
    }
    let expected = witness_digest(
        request_digest,
        tail.owner_operation_ordinal,
        tail.sequence_oid,
        tail.last_value,
        tail.owner_statement_digest,
        tail.descriptor_digest,
    );
    if tail.witness_digest != expected {
        return Err(error("terminal sequence restart witness drifted"));
    }
    Ok(encode_retained(tail))
}

pub(super) fn encode_retained(tail: SequenceRestartTail) -> [u8; SEQUENCE_RESTART_TAIL_BYTES] {
    let mut bytes = [0_u8; SEQUENCE_RESTART_TAIL_BYTES];
    bytes[0] = 1;
    bytes[1] = 2;
    bytes[2] = 1;
    bytes[4..8].copy_from_slice(&tail.owner_operation_ordinal.to_le_bytes());
    bytes[8..12].copy_from_slice(&tail.sequence_oid.to_le_bytes());
    bytes[16..24].copy_from_slice(&tail.last_value.to_le_bytes());
    bytes[32..64].copy_from_slice(&tail.owner_statement_digest);
    bytes[64..96].copy_from_slice(&tail.descriptor_digest);
    bytes[96..128].copy_from_slice(&tail.witness_digest);
    bytes
}

pub(super) fn decode(bytes: &[u8]) -> Result<SequenceRestartTail, crate::EngineError> {
    if bytes.len() != SEQUENCE_RESTART_TAIL_BYTES
        || bytes[0] != 1
        || bytes[1] != 2
        || bytes[2] != 1
        || bytes[3] != 0
        || bytes[12..16] != [0; 4]
        || bytes[24..32] != [0; 8]
    {
        return Err(error("terminal sequence restart body is not canonical"));
    }
    let tail = SequenceRestartTail {
        owner_operation_ordinal: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        sequence_oid: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        last_value: i64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        owner_statement_digest: bytes[32..64].try_into().unwrap(),
        descriptor_digest: bytes[64..96].try_into().unwrap(),
        witness_digest: bytes[96..128].try_into().unwrap(),
    };
    if tail.owner_operation_ordinal == u32::MAX
        || tail.sequence_oid == 0
        || tail.owner_statement_digest == [0; 32]
        || tail.descriptor_digest == [0; 32]
        || tail.witness_digest == [0; 32]
    {
        return Err(error("terminal sequence restart fields are invalid"));
    }
    Ok(tail)
}

pub(super) fn validate(
    request_digest: gpu_db_wal::CanonicalDigest,
    tail: SequenceRestartTail,
) -> Result<(), crate::EngineError> {
    if tail.witness_digest
        != witness_digest(
            request_digest,
            tail.owner_operation_ordinal,
            tail.sequence_oid,
            tail.last_value,
            tail.owner_statement_digest,
            tail.descriptor_digest,
        )
    {
        return Err(error(
            "terminal sequence restart does not bind the transaction request",
        ));
    }
    Ok(())
}

pub(super) fn witness_digest(
    request_digest: gpu_db_wal::CanonicalDigest,
    owner_operation_ordinal: u32,
    sequence_oid: u32,
    last_value: i64,
    owner_statement_digest: gpu_db_wal::CanonicalDigest,
    descriptor_digest: gpu_db_wal::CanonicalDigest,
) -> gpu_db_wal::CanonicalDigest {
    let domain = b"gpu-db/write001/s5-sequence-restart-tail/v2";
    let owner_operation_ordinal = owner_operation_ordinal.to_le_bytes();
    let sequence_oid = sequence_oid.to_le_bytes();
    let last_value = last_value.to_le_bytes();
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    for part in [
        request_digest.as_slice(),
        owner_operation_ordinal.as_slice(),
        sequence_oid.as_slice(),
        last_value.as_slice(),
        owner_statement_digest.as_slice(),
        descriptor_digest.as_slice(),
    ] {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    digest.finalize().into()
}

fn error(message: &str) -> crate::EngineError {
    crate::EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 S5 terminal: {message}"
    ))
}
