//! Witness-free S1--S8 closure for the semantics-v2 retained graph.
//!
//! Q0 proved the hostile raw grammar before allocation. This owner deliberately replays the
//! semantic joins from the strict S2/image owners without consulting a catalog, allocator, or
//! generation witness. The resulting typestate is `RetentionAuthorityPending`; this inert slice
//! defines no production successor for external checks.

#[path = "codec_closure/dependencies.rs"]
mod dependencies;
#[path = "codec_closure/indexes.rs"]
mod indexes;
#[path = "codec_closure/response.rs"]
mod response;
#[path = "codec_closure/roots.rs"]
mod roots;
#[path = "codec_closure/rows.rs"]
mod rows;
#[path = "codec_closure/sequences.rs"]
mod sequences;
#[path = "codec_closure/statements.rs"]
mod statements;

use super::{
    graph::{self, ReservedSemanticsV2Graph},
    SemanticsV2BoundIdentity,
};
use crate::EngineError;

pub(super) fn validate(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
) -> Result<(), EngineError> {
    statements::validate(graph)?;
    rows::validate(graph)?;
    indexes::validate(graph)?;
    dependencies::validate(graph, catalog_composition)?;
    sequences::validate(identity, graph)?;
    statements::validate_returning(graph)?;
    response::validate(identity, graph)?;
    roots::validate(graph)
}

/// Shared exact S3 proof for the one place a terminal private-sequence rename can update the
/// final table default after the final sealed S2 record. It admits no independent replay action.
pub(super) fn terminal_s3_sequence_rename_closes_table_schema(
    table: &graph::RetainedTable,
    final_record: Option<&crate::typed_insert_batch::DecodedTypedInsertRecord>,
    catalog_composition: Option<&crate::wal_binary::BinaryTransactionRecord>,
) -> bool {
    dependencies::terminal_s3_sequence_rename_closes_table_schema(
        table,
        final_record,
        catalog_composition,
    )
}

pub(super) fn error(message: impl AsRef<str>) -> EngineError {
    super::retained_error(&format!("codec closure: {}", message.as_ref()))
}
