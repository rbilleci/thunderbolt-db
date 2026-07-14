//! Shared cold-input admission for streaming joins and rank windows.

use super::*;

impl Engine {
    pub(crate) fn ensure_streaming_join_cold(
        &self,
        table: &RelationalTable,
        copin_s: Index,
        gpu_id: u16,
        input_cap: u64,
    ) -> Option<Arc<ColdTableChunks>> {
        // Row-byte target is intentionally half the payload reservation: headers, text offset
        // arrays, validity, and section alignment also occupy the retained descriptor bytes.
        let chunk_target = (input_cap / 2).max(1);
        if let Some(cold) = self.load_streaming_cold(&table.name, table, chunk_target, copin_s) {
            return Some(cold);
        }
        // A class entry is the record of truth and cannot be rescanned/re-tiled from its frozen
        // store. It joins only when its existing chunks already satisfy the two-input target.
        if self.table_chunk_authoritative(&table.name).is_some() {
            let cold = self
                .read_state
                .residency
                .streaming_cold_chunks
                .load()
                .get(&table.name)
                .cloned()?;
            return (cold.chunk_target_bytes <= chunk_target).then_some(cold);
        }
        let count_select = Select {
            table: table.name.clone(),
            distinct: false,
            projection: SelectProjection::CountAll,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        };
        let bound = bind_relational_select(table, &count_select).ok()?;
        // The reduction fold halves its supplied budget for the row-byte target. Descriptor overhead
        // must still fit `input_cap`; the caller validates the completed retained bytes before launch.
        self.run_streaming_reduction_fold(
            &count_select,
            table,
            &bound,
            None,
            copin_s,
            StreamAgg::Count,
            gpu_id,
            input_cap.max(1),
        )
        .ok()?;
        self.load_streaming_cold(&table.name, table, chunk_target, copin_s)
    }
}
