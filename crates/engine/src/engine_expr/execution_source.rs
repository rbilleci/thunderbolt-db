//! Resident device-source handles and on-device MVCC visibility-program contracts.

use std::sync::Arc;

use crate::relational_model::RelationalResidencySnapshot;
use gpu_db_execution::{CudaResidentDeviceMemory, ExprStep};

/// The table payload `execute_resident_expr_select_with_binding` reads its rows from: the published
/// device buffer + the matching catalog/GPU descriptor (whose `row_count` + `resident_device_*` section
/// vectors define every column byte-offset) + the row count. Passing `None` makes the executor look these
/// up by table name in the SINGLE resident store (the whole-table buffer), byte-identical to before;
/// passing `Some(src)` INJECTS them so the SAME executor can serve one resident shard slice (S10c) -- the
/// injected descriptor's `row_count` + section vectors describe that slice. Whole-table callers pass `None`.
pub(crate) struct ResidentExecSource {
    pub(crate) descriptor: Arc<RelationalResidencySnapshot>,
    pub(crate) device_memory: Arc<CudaResidentDeviceMemory>,
    pub(crate) row_count: u64,
}

/// A shard-resident table's unified execution source: the recompacted whole-table device buffer
/// (int4 + version columns + null bitmaps) as an injectable [`ResidentExecSource`], its SV3b/SV6 MVCC
/// visibility, and the GPU it lives on. Built by `Engine::build_sharded_unified_exec_source`; consumed by
/// the sharded shape bridge AND the SQL->Expr PG path (so every general shape serves sharded tables).
pub(crate) struct ShardedUnifiedExecSource {
    pub(crate) src: ResidentExecSource,
    pub(crate) visibility: Option<ResidentVisibility>,
    pub(crate) gpu_id: u16,
}

/// SV3b/SV6: the on-device MVCC visibility descriptor for a VERSIONED (unified) buffer. A row is visible
/// iff `deleted_by > read_txn_id` (SV3b upper bound — tombstoned-at-or-before-my-snapshot rows are hidden)
/// AND `created_by <= read_txn_id` (SV6 lower bound — versions appended by a commit newer than my snapshot
/// are hidden; the SV5 UPDATE double-read flip-gate). Each bound is present only when SOME gathered shard
/// carries the corresponding on-demand region (the sparse-versioning property: an UPDATE can tombstone in
/// one shard and append into another, so the two offsets are independent); at least one is `Some` by
/// construction — a buffer with neither offset takes the unversioned `None` visibility path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentVisibility {
    /// The read snapshot (`committed_seq` bound at read start), as the signed i64 the s64 kernels compare.
    pub(crate) read_txn_id: i64,
    /// Byte offset of the co-resident dense i64 `deleted_by` column in the unified buffer, if gathered.
    pub(crate) deleted_by_offset: Option<u64>,
    /// Byte offset of the co-resident dense i64 `created_by` column in the unified buffer, if gathered.
    pub(crate) created_by_offset: Option<u64>,
}

impl ResidentVisibility {
    /// Append this visibility's conjunct(s) to a mask-VM `program`. Each conjunct is a `LoadColumnI64`
    /// IMMEDIATELY consumed by its `CompareScalarI64` (the mixed-width VM caller contract: an i64 compare
    /// must pop an 8-byte buffer), producing a 0/1 mask. `and_onto_existing_mask` = a WHERE mask is already
    /// on the VM stack, so EVERY conjunct is ANDed onto it; otherwise the FIRST conjunct becomes the mask
    /// and only a second conjunct ANDs (`MaskBinary` op 0). `deleted_by > read_txn_id` is cmp 3 (Gt);
    /// `created_by <= read_txn_id` is cmp 2 (Le) — see the PTX cmp code table
    /// (`gpu_db_buffer_i64_compare_scalar_to_mask`: 0=eq/1=lt/2=le/3=gt/4=ge/5=ne).
    pub(crate) fn push_conjuncts(&self, program: &mut Vec<ExprStep>, and_onto_existing_mask: bool) {
        let mut have_mask = and_onto_existing_mask;
        for (byte_offset, cmp) in [
            (self.deleted_by_offset, 3_u32), // visible: deleted_by > read_txn_id
            (self.created_by_offset, 2_u32), // visible: created_by <= read_txn_id
        ] {
            let Some(byte_offset) = byte_offset else {
                continue;
            };
            program.push(ExprStep::LoadColumnI64 { byte_offset });
            program.push(ExprStep::CompareScalarI64 {
                cmp,
                scalar: self.read_txn_id,
                scalar_on_left: false,
            });
            if have_mask {
                program.push(ExprStep::MaskBinary { op: 0 });
            }
            have_mask = true;
        }
    }
}
