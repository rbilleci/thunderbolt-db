//! Exact `GPUDBSTATUS2` fragment for codec-5 aggregate transactions.

use super::*;
use crate::EngineError;

pub(crate) const STATUS_STATE_DURABLE_PENDING_PUBLICATION: u8 = 1;
const STATUS_V2_KNOWN_FLAGS: u16 = 0;

/// The immutable durable-pending status carried immediately after the aggregate chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypedInsertStatusV2 {
    pub(crate) database_id: [u8; 16],
    pub(crate) timeline_id: [u8; 16],
    pub(crate) txn_id: u64,
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) isolation: u8,
    pub(crate) flags: u16,
    pub(crate) retention_deadline: u64,
    pub(crate) statement_count: u32,
    pub(crate) response_artifact_count: u32,
    pub(crate) statement_outcome_root: gpu_db_wal::CanonicalDigest,
    pub(crate) response_root: gpu_db_wal::CanonicalDigest,
    pub(crate) aggregate_root: gpu_db_wal::CanonicalDigest,
}

pub(crate) fn encode_status_v2(
    status: &TypedInsertStatusV2,
    out: &mut [u8],
) -> Result<(), EngineError> {
    validate(status)?;
    if out.len() != AGGREGATE_STATUS_V2_BYTES as usize {
        return Err(error("STATUS2 output length is not exactly 204 bytes"));
    }
    let mut writer = Writer::new(out);
    writer.bytes(AGGREGATE_STATUS_MAGIC)?;
    writer.bytes(&status.database_id)?;
    writer.bytes(&status.timeline_id)?;
    writer.u64(status.txn_id)?;
    writer.bytes(&status.request_digest)?;
    writer.u8(STATUS_STATE_DURABLE_PENDING_PUBLICATION)?;
    writer.u8(status.isolation)?;
    writer.u16(status.flags)?;
    writer.u32(0)?;
    writer.u64(status.retention_deadline)?;
    writer.u32(status.statement_count)?;
    writer.u32(status.response_artifact_count)?;
    writer.bytes(&status.statement_outcome_root)?;
    writer.bytes(&status.response_root)?;
    writer.bytes(&status.aggregate_root)?;
    writer.finish()
}

pub(crate) fn decode_status_v2(bytes: &[u8]) -> Result<TypedInsertStatusV2, EngineError> {
    if bytes.len() != AGGREGATE_STATUS_V2_BYTES as usize {
        return Err(error("STATUS2 input length is not exactly 204 bytes"));
    }
    let mut reader = Reader::new(bytes);
    if reader.exact::<12>()? != *AGGREGATE_STATUS_MAGIC {
        return Err(error("STATUS2 magic is invalid"));
    }
    let database_id = reader.exact::<16>()?;
    let timeline_id = reader.exact::<16>()?;
    let txn_id = reader.u64()?;
    let request_digest = reader.exact::<32>()?;
    if reader.u8()? != STATUS_STATE_DURABLE_PENDING_PUBLICATION {
        return Err(error("STATUS2 state is not durable-pending-publication"));
    }
    let isolation = reader.u8()?;
    let flags = reader.u16()?;
    if reader.u32()? != 0 {
        return Err(error("STATUS2 reserved field is nonzero"));
    }
    let retention_deadline = reader.u64()?;
    let statement_count = reader.u32()?;
    let response_artifact_count = reader.u32()?;
    let statement_outcome_root = reader.exact::<32>()?;
    let response_root = reader.exact::<32>()?;
    let aggregate_root = reader.exact::<32>()?;
    reader.finish()?;
    let status = TypedInsertStatusV2 {
        database_id,
        timeline_id,
        txn_id,
        request_digest,
        isolation,
        flags,
        retention_deadline,
        statement_count,
        response_artifact_count,
        statement_outcome_root,
        response_root,
        aggregate_root,
    };
    validate(&status)?;
    Ok(status)
}

fn validate(status: &TypedInsertStatusV2) -> Result<(), EngineError> {
    if status.database_id == [0; 16]
        || status.timeline_id == [0; 16]
        || status.txn_id == 0
        || status.statement_count == 0
        || status.aggregate_root == [0; 32]
        || status.request_digest == [0; 32]
        || status.statement_outcome_root == [0; 32]
        || !matches!(status.isolation, 1..=3)
        || status.flags & !STATUS_V2_KNOWN_FLAGS != 0
    {
        return Err(error("STATUS2 identity or root is invalid"));
    }
    if status.response_artifact_count > status.statement_count {
        return Err(error(
            "STATUS2 response artifact count exceeds statement count",
        ));
    }
    Ok(())
}

struct Writer<'a> {
    bytes: &'a mut [u8],
    offset: usize,
}

impl<'a> Writer<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), EngineError> {
        let end = self
            .offset
            .checked_add(value.len())
            .ok_or_else(|| error("STATUS2 output offset overflows"))?;
        let target = self
            .bytes
            .get_mut(self.offset..end)
            .ok_or_else(|| error("STATUS2 output is short"))?;
        target.copy_from_slice(value);
        self.offset = end;
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), EngineError> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), EngineError> {
        self.bytes(&value.to_le_bytes())
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.offset != self.bytes.len() {
            return Err(error("STATUS2 output has surplus bytes"));
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| error("STATUS2 input offset overflows"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| error("STATUS2 input is truncated"))?;
        self.offset = end;
        Ok(value)
    }

    fn exact<const N: usize>(&mut self) -> Result<[u8; N], EngineError> {
        self.take(N)?
            .try_into()
            .map_err(|_| error("STATUS2 fixed field length drifted"))
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.exact::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.exact()?))
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.exact()?))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.exact()?))
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.offset != self.bytes.len() {
            return Err(error("STATUS2 has trailing bytes"));
        }
        Ok(())
    }
}

fn error(message: &str) -> EngineError {
    EngineError::Durability(format!("typed INSERT aggregate STATUS2: {message}"))
}
