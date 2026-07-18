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
        self.transition_oversized_device_table_to_streaming_repair(&table.name)
            .ok()?;
        // Row-byte target is intentionally half the payload reservation: headers, text offset
        // arrays, validity, and section alignment also occupy the retained descriptor bytes.
        let chunk_target = (input_cap / 2).max(1);
        if let Some(cold) = self.load_streaming_cold(&table.name, table, chunk_target, copin_s) {
            return Some(cold);
        }
        // A class entry is the record of truth. Re-tile its encoded payloads directly when a
        // two-input operator needs a smaller target; never de-authoritize into host execution.
        if self.table_chunk_authoritative(&table.name).is_some() {
            return self.rechunk_streaming_cold_class(table, chunk_target);
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
        self.maybe_enter_chunk_class_from_cold(&table.name);
        self.load_streaming_cold(&table.name, table, chunk_target, copin_s)
    }
}
