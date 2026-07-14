use serde::{Deserialize, Serialize};

use crate::CudaRuntimeProbeError;

/// Device-transfer representation for a snapshot-resolved MVCC row batch.
///
/// Offsets and metadata remain structure-of-arrays so CUDA visibility and length kernels can
/// consume each section directly. This type owns encoding and validation only; relational
/// visibility decisions remain on-device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaMvccRowBatch {
    pub row_count: u32,
    pub key_offsets: Vec<u32>,
    pub key_bytes: Vec<u8>,
    pub value_offsets: Vec<u32>,
    pub value_bytes: Vec<u8>,
    pub begin_txn_ids: Vec<u64>,
    pub end_txn_ids: Vec<u64>,
    pub provenance_handles: Vec<u32>,
}

impl CudaMvccRowBatch {
    pub fn from_key_values<I, K, V>(rows: I) -> Result<Self, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        Self::from_key_values_with_metadata(
            rows.into_iter()
                .map(|(key, value)| (key, value, 0_u64, u64::MAX, None)),
        )
    }

    pub fn from_key_values_with_metadata<I, K, V>(rows: I) -> Result<Self, CudaRuntimeProbeError>
    where
        I: IntoIterator<Item = (K, V, u64, u64, Option<u32>)>,
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        let mut row_count = 0_u32;
        let mut key_offsets = vec![0_u32];
        let mut key_bytes = Vec::new();
        let mut value_offsets = vec![0_u32];
        let mut value_bytes = Vec::new();
        let mut begin_txn_ids = Vec::new();
        let mut end_txn_ids = Vec::new();
        let mut provenance_handles = Vec::new();

        for (key, value, begin_txn_id, end_txn_id, provenance_handle) in rows {
            let provenance_handle = provenance_handle.unwrap_or(row_count);
            row_count = row_count
                .checked_add(1)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

            key_bytes.extend_from_slice(key.as_ref());
            key_offsets.push(
                u32::try_from(key_bytes.len())
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(key_bytes.len()))?,
            );

            value_bytes.extend_from_slice(value.as_ref());
            value_offsets.push(
                u32::try_from(value_bytes.len())
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(value_bytes.len()))?,
            );
            begin_txn_ids.push(begin_txn_id);
            end_txn_ids.push(end_txn_id);
            provenance_handles.push(provenance_handle);
        }

        Ok(Self {
            row_count,
            key_offsets,
            key_bytes,
            value_offsets,
            value_bytes,
            begin_txn_ids,
            end_txn_ids,
            provenance_handles,
        })
    }

    pub fn transfer_bytes(&self) -> usize {
        (self.key_offsets.len() + self.value_offsets.len()) * std::mem::size_of::<u32>()
            + self.key_bytes.len()
            + self.value_bytes.len()
            + self.begin_txn_ids.len() * std::mem::size_of::<u64>()
            + self.end_txn_ids.len() * std::mem::size_of::<u64>()
            + self.provenance_handles.len() * std::mem::size_of::<u32>()
    }

    pub fn key_len(&self, row_index: usize) -> Option<u32> {
        row_segment_len(&self.key_offsets, row_index)
    }

    pub fn value_len(&self, row_index: usize) -> Option<u32> {
        row_segment_len(&self.value_offsets, row_index)
    }

    pub fn validate(&self) -> Result<(), CudaRuntimeProbeError> {
        let row_count = self.row_count as usize;
        if self.key_offsets.len() != row_count + 1 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.key_offsets.len(),
            ));
        }
        if self.value_offsets.len() != row_count + 1 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.value_offsets.len(),
            ));
        }
        if self.begin_txn_ids.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.begin_txn_ids.len(),
            ));
        }
        if self.end_txn_ids.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.end_txn_ids.len(),
            ));
        }
        if self.provenance_handles.len() != row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                self.provenance_handles.len(),
            ));
        }
        validate_offsets(&self.key_offsets, self.key_bytes.len())?;
        validate_offsets(&self.value_offsets, self.value_bytes.len())?;
        Ok(())
    }
}

fn row_segment_len(offsets: &[u32], row_index: usize) -> Option<u32> {
    let start = *offsets.get(row_index)?;
    let end = *offsets.get(row_index + 1)?;
    end.checked_sub(start)
}

fn validate_offsets(offsets: &[u32], bytes_len: usize) -> Result<(), CudaRuntimeProbeError> {
    if offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes_len));
    }
    let mut previous = 0_u32;
    for offset in offsets.iter().copied().skip(1) {
        if offset < previous {
            return Err(CudaRuntimeProbeError::InvalidInputLength(offset as usize));
        }
        previous = offset;
    }
    if previous as usize != bytes_len {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes_len));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_offsets_metadata_and_transfer_size() {
        let batch = CudaMvccRowBatch::from_key_values_with_metadata([
            (b"acct:1".as_slice(), b"open".as_slice(), 3, 9, Some(11)),
            (b"acct:22".as_slice(), b"".as_slice(), 4, u64::MAX, Some(12)),
            (b"".as_slice(), b"closed".as_slice(), 5, 8, Some(13)),
        ])
        .unwrap();

        assert_eq!(batch.row_count, 3);
        assert_eq!(batch.key_offsets, vec![0, 6, 13, 13]);
        assert_eq!(batch.key_bytes, b"acct:1acct:22");
        assert_eq!(batch.value_offsets, vec![0, 4, 4, 10]);
        assert_eq!(batch.value_bytes, b"openclosed");
        assert_eq!(batch.begin_txn_ids, vec![3, 4, 5]);
        assert_eq!(batch.end_txn_ids, vec![9, u64::MAX, 8]);
        assert_eq!(batch.provenance_handles, vec![11, 12, 13]);
        assert_eq!(batch.key_len(0), Some(6));
        assert_eq!(batch.key_len(2), Some(0));
        assert_eq!(batch.value_len(1), Some(0));
        assert_eq!(batch.value_len(3), None);
        assert_eq!(
            batch.transfer_bytes(),
            (batch.key_offsets.len() + batch.value_offsets.len()) * std::mem::size_of::<u32>()
                + batch.key_bytes.len()
                + batch.value_bytes.len()
                + batch.begin_txn_ids.len() * std::mem::size_of::<u64>()
                + batch.end_txn_ids.len() * std::mem::size_of::<u64>()
                + batch.provenance_handles.len() * std::mem::size_of::<u32>()
        );
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn rejects_incoherent_offsets() {
        let mut batch =
            CudaMvccRowBatch::from_key_values([(b"acct:1".as_slice(), b"open".as_slice())])
                .unwrap();
        batch.key_offsets[1] = 99;

        assert_eq!(
            batch.validate(),
            Err(CudaRuntimeProbeError::InvalidInputLength(
                batch.key_bytes.len()
            ))
        );
    }
}
