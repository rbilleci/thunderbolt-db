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

use super::{graph::ReservedSemanticsV2Graph, SemanticsV2BoundIdentity};
use crate::EngineError;

pub(super) fn validate(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    statements::validate(graph)?;
    rows::validate(graph)?;
    indexes::validate(graph)?;
    dependencies::validate(graph)?;
    sequences::validate(identity, graph)?;
    statements::validate_returning(graph)?;
    response::validate(identity, graph)?;
    roots::validate(graph)
}

pub(super) fn error(message: impl AsRef<str>) -> EngineError {
    super::retained_error(&format!("codec closure: {}", message.as_ref()))
}
