//! Join-side source resolution for resident, sharded-unified, and transient catalog relations.
//! Join predicate, execution, projection, and coordinate ownership remains elsewhere.

use super::join_source::{JoinDeviceMemory, JoinExecSide};
use crate::relational_model::{RelationalResidencyEntry, RelationalTable};
use crate::{Engine, ExecuteError};
use gpu_db_sql::SqlValue;
use gpu_db_types::{EngineError, Index};

impl Engine {
    /// Resolve one join relation to its (residency entry, device memory, row count): a RESIDENT user
    /// table uses its published snapshot + retained device memory; a SYNTHESIZED catalog relation
    /// (`rows = Some`, M5 J5) is uploaded as a TRANSIENT device payload + descriptor that lives only for
    /// this query. Either way the join runs the SAME GPU pre-filter + key-projection + hash-join kernels
    /// over the result -- no CPU relational join (charter).
    pub(crate) fn resolve_join_side(
        &self,
        name: &str,
        table: &RelationalTable,
        rows: Option<Vec<Vec<SqlValue>>>,
        copin_s: Index,
    ) -> Result<JoinExecSide, ExecuteError> {
        match rows {
            None => {
                // THE FLIP: a SHARD-resident relation (no single-buffer entry) joins via its UNIFIED
                // exec source — the same GPU pre-filter/key-projection/hash-join kernels run over the
                // recompacted (or zero-copy single-shard) buffer. A VERSIONED sharded relation
                // clean-errors: the join kernels do not thread the visibility conjuncts (never a
                // tombstone leak). No host rows ride the entry — the join is GPU-only (charter).
                let shards = self.read_residency_shards();
                if self.relational_residency_entry(name).is_none()
                    && shards.get(name).is_some_and(|shards| !shards.is_empty())
                {
                    let unified = self.build_sharded_unified_exec_source(table, None, copin_s)?;
                    let row_count = unified.src.descriptor.row_count;
                    let entry = RelationalResidencyEntry::new(unified.src.descriptor);
                    return Ok((
                        entry,
                        JoinDeviceMemory::Resident(unified.src.device_memory),
                        row_count,
                        unified.visibility,
                    ));
                }
                let entry = self.relational_residency_entry(name).ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" has no resident snapshot (the join path is GPU-only)"
                    )))
                })?;
                if !entry.descriptor.is_valid() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" resident snapshot is invalid"
                    ))));
                }
                let row_count = entry.descriptor.row_count;
                let memory = entry.device_memory.clone().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{name}\" has no retained resident device memory"
                    )))
                })?;
                Ok((entry, JoinDeviceMemory::Resident(memory), row_count, None))
            }
            Some(rows) => {
                let _transient_scope = gpu_db_execution::Probe::scope("join_transient_side_upload");
                let (snapshot, memory) = self.build_transient_relation_residency(table, &rows)?;
                let row_count = rows.len();
                let entry = RelationalResidencyEntry::new(std::sync::Arc::new(snapshot));
                Ok((entry, JoinDeviceMemory::Transient(memory), row_count, None))
            }
        }
    }
}
