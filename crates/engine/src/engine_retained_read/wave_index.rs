use super::{
    build_int4_pk_hash_table_host, Arc, CudaResidentDeviceMemory, Engine, WaveResidentIndex,
};

impl Engine {
    /// ADR-009 R1: fetch (building + caching on demand) the GPU hash index over resident int4 key column
    /// `filter_idx` of the resident buffer `device_memory`. `Some((index, table_mask, hash_shift))` drives
    /// the index-probe route; `None` means "use the scan" — the column is non-unique / un-buildable. The
    /// cache is keyed by table and validated by `(column_idx, resident_device_ptr)`: the index is built
    /// from + used with the SAME device buffer (its address pinned via `_resident_guard`, so a re-admission
    /// allocates a new buffer with a new address → cache miss → rebuild against the live bytes). This makes
    /// the index inherently consistent with the buffer the probe gathers from — no host-rows cross-map race.
    pub(super) fn wave_resident_int4_index(
        &self,
        table_name: &str,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        filter_idx: usize,
        row_count: u64,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        let resident_device_ptr = device_memory.device_ptr();
        {
            let cache = self
                .read_state
                .residency
                .wave_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(existing) = cache.get(table_name) {
                if existing.column_idx == filter_idx
                    && existing.resident_device_ptr == resident_device_ptr
                {
                    return existing.index_memory.as_ref().map(|memory| {
                        (Arc::clone(memory), existing.table_mask, existing.hash_shift)
                    });
                }
            }
        }
        // Miss / different column / different buffer: build OUTSIDE the lock (a DtoH of the key column + a
        // host hash pass + one HtoD upload), then publish. A concurrent builder for the same buffer merely
        // rebuilds + overwrites — rare (once per residency generation) and harmless: each index is
        // self-contained and its in-flight kernels pin their own `Arc`, so a replaced entry's index buffer
        // is freed only once no submission still holds it.
        // Serialize the budget check, device allocation, and cache publication with admission.
        // A cap decline is intentionally just an index miss: the caller executes the same query
        // with the resident GPU scan, never with host relational execution.
        let _budget_allocation = self
            .read_state
            .residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let built = self.build_wave_resident_int4_index(device_memory, filter_offset, row_count);
        let (index_memory, table_mask, hash_shift) = match &built {
            Some((memory, mask, shift)) => (Some(Arc::clone(memory)), *mask, *shift),
            None => (None, 0, 0),
        };
        let mut cache = self
            .read_state
            .residency
            .wave_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(
            table_name.to_string(),
            WaveResidentIndex {
                column_idx: filter_idx,
                resident_device_ptr,
                _resident_guard: Arc::clone(device_memory),
                index_memory,
                table_mask,
                hash_shift,
            },
        );
        built
    }

    /// ADR-009 R1: build the open-addressing GPU hash index (`(key<<32)|(row+1)`, 0 = empty; Fibonacci
    /// `(key*0x9E3779B1)>>hash_shift` + linear probe) over int4 column at `filter_offset` of `device_memory`,
    /// then upload it once (HtoD). The keys are DtoH-read from the resident buffer ITSELF — the same bytes
    /// the scan reads and the probe gathers — so the row→value mapping is inherently consistent with the
    /// buffer and a NULL int4 (materialized as `0`) is indexed exactly as the scan matches it (no skip, so
    /// `WHERE col = 0` agrees on both routes). `None` (→ caller scans) when: the table is empty / too large
    /// to pack a row index into 32 bits; the DtoH fails; a key would exceed the kernel's 256-probe cap; or
    /// — critically for correctness — the column has DUPLICATE keys (incl. multiple NULLs-as-0): a hash
    /// index holds one row per key but the scan returns EVERY match, so a duplicate makes the index decline.
    fn build_wave_resident_int4_index(
        &self,
        device_memory: &Arc<CudaResidentDeviceMemory>,
        filter_offset: u64,
        row_count: u64,
    ) -> Option<(Arc<CudaResidentDeviceMemory>, u32, u32)> {
        // `row + 1` is packed into the low 32 bits, so the row index must fit in u32.
        if row_count == 0 || row_count >= (u32::MAX as u64) {
            return None;
        }
        let row_count_usize = usize::try_from(row_count).ok()?;
        let keys = device_memory
            .read_resident_i32_column(filter_offset, row_count_usize)
            .ok()?;
        if keys.len() != row_count_usize {
            return None;
        }
        let (index, table_mask, hash_shift) = build_int4_pk_hash_table_host(&keys, row_count)?;
        let index_bytes: Vec<u8> = index.iter().flat_map(|entry| entry.to_le_bytes()).collect();
        let runtime = self.cuda_driver_probe_runtime();
        let gpu_id = device_memory.metadata().gpu_id;
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(index_bytes.len() as u64)
                    > budget
            })
        {
            return None;
        }
        let memory = runtime
            .retain_device_memory_copy(gpu_id, &index_bytes)
            .ok()?;
        if self
            .relational_residency_budget_bytes(gpu_id)
            .is_some_and(|budget| {
                self.relational_resident_bytes_for_gpu(gpu_id)
                    .saturating_add(memory.metadata().allocated_bytes)
                    > budget
            })
        {
            return None;
        }
        Some((Arc::new(memory), table_mask, hash_shift))
    }
}
