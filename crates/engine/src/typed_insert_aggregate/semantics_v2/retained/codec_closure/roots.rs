//! Bottom-up transition, manifest, and root-descriptor closure.

use super::error;
use super::statements::{begin, exact, range};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedTable,
};
use crate::EngineError;
use sha2::Digest;

pub(super) fn validate(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    validate_statement_overlay_chain(graph)?;
    for table in &graph.tables {
        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "table transition root",
        )?;
        let effects = range(
            &graph.key_effects,
            table.key_effect_start,
            table.key_effect_count,
            "table effect root",
        )?;
        let transition_root = digest_list(
            b"gpu-db/write001/s7-table-transition-root/v2",
            table.stable_table_id,
            transitions.iter().map(|entry| entry.transition_digest),
        );
        let effect_root = digest_list(
            b"gpu-db/write001/s7-table-index-effect-root/v2",
            table.stable_table_id,
            effects.iter().map(|entry| entry.effect_digest),
        );
        if table.transition_root != transition_root || table.index_effect_root != effect_root {
            return Err(error("table transition/index-effect root is invalid"));
        }
        if table.manifest_digest != manifest_digest(graph, table)? {
            return Err(error("table manifest digest is invalid"));
        }
    }
    let has_transition = graph.tables.iter().any(|table| table.transition_count != 0);
    if has_transition == (graph.header.final_database_root == graph.header.initial_database_root) {
        return Err(error(
            "database root does not exactly reflect whether any table transitioned",
        ));
    }
    if graph.header.root_descriptor != root_descriptor(graph)? {
        return Err(error("root descriptor digest is invalid"));
    }
    Ok(())
}

fn validate_statement_overlay_chain(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let mut prior = graph.header.initial_overlay_root;
    for resolution in &graph.resolutions {
        let statement = graph
            .statements
            .get(resolution.statement_ordinal as usize)
            .ok_or_else(|| error("overlay S1 statement is absent"))?;
        let disposition_root = statement_disposition_root(graph, resolution)?;
        let sequence_root = statement_sequence_root(graph, resolution)?;
        let dependency_root = statement_dependency_root(graph, resolution)?;
        let projection_root = statement_projection_root(graph, resolution)?;
        let expected = exact(
            b"gpu-db/write001/s7-statement-overlay-root/v2",
            &[
                &prior,
                &resolution.statement_ordinal.to_le_bytes(),
                &resolution.typed_statement_digest,
                &resolution.record_digest,
                &disposition_root,
                &sequence_root,
                &dependency_root,
                &projection_root,
            ],
        );
        if statement.overlay_before != prior
            || statement.overlay_after != expected
            || resolution.overlay_before != prior
            || resolution.overlay_after != expected
        {
            return Err(error("statement overlay-root chain is invalid"));
        }
        prior = expected;
    }
    if prior != graph.header.final_overlay_root {
        return Err(error("final overlay root is invalid"));
    }
    Ok(())
}

fn statement_disposition_root(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
) -> Result<[u8; 32], EngineError> {
    let entries = range(
        &graph.dispositions,
        resolution.s4_start,
        resolution.s4_count,
        "overlay S4",
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-statement-dispositions/v2");
    digest.update(resolution.statement_ordinal.to_le_bytes());
    digest.update(resolution.s4_count.to_le_bytes());
    for entry in entries {
        digest.update(s4_bytes(entry));
    }
    Ok(digest.finalize().into())
}

fn statement_sequence_root(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
) -> Result<[u8; 32], EngineError> {
    let entries = range(
        &graph.sequence_effects,
        resolution.s5_start,
        resolution.s5_count,
        "overlay S5",
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-statement-sequences/v2");
    digest.update(resolution.statement_ordinal.to_le_bytes());
    digest.update(resolution.s5_count.to_le_bytes());
    for entry in entries {
        let mut prefix = [0_u8; 52];
        prefix[..4].copy_from_slice(&entry.statement_ordinal.to_le_bytes());
        prefix[4..8].copy_from_slice(&entry.effect_ordinal.to_le_bytes());
        prefix[8] = 1;
        prefix[9] = entry.flags;
        prefix[12..16].copy_from_slice(&entry.disposition_ref.to_le_bytes());
        prefix[16..20]
            .copy_from_slice(&(crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES as u32).to_le_bytes());
        prefix[20..52].copy_from_slice(&entry.body_digest);
        let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
        crate::encode_sequence_value_reference_into_exact(&entry.reference, &mut body)?;
        digest.update(prefix);
        digest.update(body);
    }
    Ok(digest.finalize().into())
}

fn statement_dependency_root(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
) -> Result<[u8; 32], EngineError> {
    let entries = range(
        &graph.dependency_uses,
        resolution.dependency_use_start,
        resolution.dependency_use_count,
        "overlay dependency use",
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-statement-dependencies/v2");
    digest.update(resolution.statement_ordinal.to_le_bytes());
    digest.update(resolution.dependency_use_count.to_le_bytes());
    for entry in entries {
        let token = graph
            .dependencies
            .get(entry.dependency_ref as usize)
            .ok_or_else(|| error("overlay dependency token is absent"))?;
        let mut raw = [0_u8; 32];
        raw[..4].copy_from_slice(&entry.statement_ordinal.to_le_bytes());
        raw[4..8].copy_from_slice(&entry.dependency_ref.to_le_bytes());
        raw[8..10].copy_from_slice(&entry.role.to_le_bytes());
        raw[12..16].copy_from_slice(&entry.source_ordinal.to_le_bytes());
        raw[16..20].copy_from_slice(&entry.transition_ref.to_le_bytes());
        raw[20..24].copy_from_slice(&entry.key_effect_ref.to_le_bytes());
        digest.update(raw);
        digest.update(token.token_digest);
    }
    Ok(digest.finalize().into())
}

fn statement_projection_root(
    graph: &ReservedSemanticsV2Graph,
    resolution: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedStatementResolution,
) -> Result<[u8; 32], EngineError> {
    let entries = range(
        &graph.projections,
        resolution.projection_start,
        resolution.projection_count,
        "overlay projection",
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-statement-projections/v2");
    digest.update(resolution.statement_ordinal.to_le_bytes());
    digest.update(resolution.projection_count.to_le_bytes());
    for entry in entries {
        digest.update(entry.projection_digest);
    }
    Ok(digest.finalize().into())
}

fn digest_list(
    domain: &[u8],
    stable_table_id: u64,
    values: impl ExactSizeIterator<Item = [u8; 32]>,
) -> [u8; 32] {
    let mut digest = begin(domain);
    digest.update(stable_table_id.to_le_bytes());
    digest.update((values.len() as u32).to_le_bytes());
    for value in values {
        digest.update(value);
    }
    digest.finalize().into()
}

fn manifest_digest(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
) -> Result<[u8; 32], EngineError> {
    let target = graph
        .dependencies
        .get(table.target_dependency_ref as usize)
        .ok_or_else(|| error("table target token is absent"))?;
    let dispositions = range(
        &graph.table_dispositions,
        table.disposition_start,
        table.disposition_count,
        "table manifest disposition",
    )?;
    let indexes = range(
        &graph.indexes,
        table.owned_index_start,
        table.owned_index_count,
        "table manifest index",
    )?;
    let transitions = range(
        &graph.transitions,
        table.transition_start,
        table.transition_count,
        "table manifest transition",
    )?;
    let effects = range(
        &graph.key_effects,
        table.key_effect_start,
        table.key_effect_count,
        "table manifest effect",
    )?;
    let mut digest = begin(b"gpu-db/write001/s7-table-manifest/v2");
    digest.update(table_bytes_before_manifest(table));
    digest.update([0; 32]);
    digest.update(target.token_digest);
    for entry in dispositions {
        digest.update(table_disposition_bytes(entry));
    }
    for index in indexes {
        digest.update(index.descriptor_digest);
    }
    for transition in transitions {
        digest.update(transition.transition_digest);
    }
    for effect in effects {
        digest.update(effect.effect_digest);
    }
    digest.update(table.image_descriptor_digest);
    Ok(digest.finalize().into())
}

fn root_descriptor(graph: &ReservedSemanticsV2Graph) -> Result<[u8; 32], EngineError> {
    let header = &graph.header;
    let mut digest = begin(b"gpu-db/write001/s7-root-descriptor/v2");
    digest.update(header.root_descriptor_version.to_le_bytes());
    digest.update(header.catalog_before_epoch.to_le_bytes());
    digest.update(header.catalog_after_epoch.to_le_bytes());
    digest.update(header.catalog_before_digest);
    digest.update(header.catalog_after_digest);
    digest.update(header.initial_database_root);
    digest.update(header.final_database_root);
    digest.update(header.initial_overlay_root);
    digest.update(header.final_overlay_root);
    digest.update(
        u32::try_from(graph.tables.len())
            .map_err(|_| error("root table count overflows"))?
            .to_le_bytes(),
    );
    for (ordinal, table) in graph.tables.iter().enumerate() {
        if table.table_ref != ordinal as u32 {
            return Err(error("root descriptor table order is invalid"));
        }
        digest.update(table.table_ref.to_le_bytes());
        digest.update(table.stable_table_id.to_le_bytes());
        digest.update(table.data_generation_before.to_le_bytes());
        digest.update(table.data_generation_after.to_le_bytes());
        digest.update(table.initial_table_root);
        digest.update(table.final_table_root);
        digest.update(table.manifest_digest);
    }
    Ok(digest.finalize().into())
}

fn table_bytes_before_manifest(table: &RetainedTable) -> [u8; 352] {
    let mut raw = [0_u8; 352];
    raw[..4].copy_from_slice(&table.table_ref.to_le_bytes());
    raw[8..16].copy_from_slice(&table.stable_table_id.to_le_bytes());
    raw[16..20].copy_from_slice(&table.display_oid.to_le_bytes());
    raw[20..24].copy_from_slice(&table.target_dependency_ref.to_le_bytes());
    raw[24..32].copy_from_slice(&table.catalog_epoch.to_le_bytes());
    raw[32..40].copy_from_slice(&table.data_generation_before.to_le_bytes());
    raw[40..48].copy_from_slice(&table.data_generation_after.to_le_bytes());
    raw[48..56].copy_from_slice(&table.row_allocator_before.to_le_bytes());
    raw[56..64].copy_from_slice(&table.row_allocator_high_water.to_le_bytes());
    raw[64..72].copy_from_slice(&table.initial_logical_row_count.to_le_bytes());
    raw[72..80].copy_from_slice(&table.final_logical_row_count.to_le_bytes());
    raw[80..84].copy_from_slice(&table.disposition_start.to_le_bytes());
    raw[84..88].copy_from_slice(&table.disposition_count.to_le_bytes());
    raw[88..92].copy_from_slice(&table.transition_start.to_le_bytes());
    raw[92..96].copy_from_slice(&table.transition_count.to_le_bytes());
    raw[96..100].copy_from_slice(&table.owned_index_start.to_le_bytes());
    raw[100..104].copy_from_slice(&table.owned_index_count.to_le_bytes());
    raw[104..108].copy_from_slice(&table.key_effect_start.to_le_bytes());
    raw[108..112].copy_from_slice(&table.key_effect_count.to_le_bytes());
    raw[112..116].copy_from_slice(&table.image_ref.to_le_bytes());
    raw[116..120].copy_from_slice(&table.catalog_column_count.to_le_bytes());
    raw[128..160].copy_from_slice(&table.schema_digest);
    raw[160..192].copy_from_slice(&table.initial_table_root);
    raw[192..224].copy_from_slice(&table.final_table_root);
    raw[224..256].copy_from_slice(&table.image_layout_digest);
    raw[256..288].copy_from_slice(&table.image_content_digest);
    raw[288..320].copy_from_slice(&table.transition_root);
    raw[320..352].copy_from_slice(&table.index_effect_root);
    raw
}

fn table_disposition_bytes(
    entry: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedTableDisposition,
) -> [u8; 32] {
    let mut raw = [0_u8; 32];
    raw[..4].copy_from_slice(&entry.table_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&entry.disposition_ref.to_le_bytes());
    raw[8..16].copy_from_slice(&entry.stable_row_id.to_le_bytes());
    raw[16..20].copy_from_slice(&entry.statement_ordinal.to_le_bytes());
    raw[20..24].copy_from_slice(&entry.source_row_ordinal.to_le_bytes());
    raw[24] = entry.disposition;
    raw
}

fn s4_bytes(
    entry: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedDisposition,
) -> [u8; 64] {
    let mut raw = [0_u8; 64];
    raw[..4].copy_from_slice(&entry.statement_ordinal.to_le_bytes());
    raw[4..8].copy_from_slice(&entry.source_row_ordinal.to_le_bytes());
    raw[8..16].copy_from_slice(&entry.stable_row_id.to_le_bytes());
    raw[16] = entry.disposition;
    raw[20..24].copy_from_slice(&entry.table_ref.to_le_bytes());
    raw[24..28].copy_from_slice(&entry.transition_ref.to_le_bytes());
    raw[32..64].copy_from_slice(&entry.typed_statement_digest);
    raw
}
