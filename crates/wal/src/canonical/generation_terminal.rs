//! Root-format-v1 canonical terminal-marker extension.
//!
//! This codec owns only the exact durable descriptor grammar accepted by the runtime logical
//! generation publication design.  It neither constructs data roots nor makes an envelope
//! publishable by itself.  In particular, the existing 92-byte outcome marker remains the only
//! form emitted by current writers, including frozen codec-5 INSERT.

use super::*;

/// Exact extension magic following a 92-byte canonical outcome.
pub(crate) const GENERATION_TERMINAL_EXTENSION_MAGIC: [u8; 16] = *b"GPUDBGENROOT1\0\0\0";
/// The sole root-format-v1 terminal descriptor version.
pub(crate) const GENERATION_TERMINAL_EXTENSION_VERSION: u16 = 1;

const GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES: usize = 24;
const GENERATION_TERMINAL_DESCRIPTOR_FIXED_BYTES: usize = 204;
const GENERATION_TABLE_TRANSITION_FIXED_BYTES: usize = 120;
const GENERATION_INDEX_MANIFEST_BYTES: usize = 80;
const RUNTIME_TERMINAL_DOMAIN: &[u8] = b"gpu-db/runtime-generation/terminal-envelope/v1";

fn descriptor_error(message: impl Into<String>) -> EngineError {
    durability(format!("runtime generation terminal: {}", message.into()))
}

fn nonzero_digest(value: &[u8; 32], label: &str) -> Result<(), EngineError> {
    if *value == [0; 32] {
        return Err(descriptor_error(format!("{label} must not be zero")));
    }
    Ok(())
}

fn nonzero_id(value: u64, label: &str) -> Result<(), EngineError> {
    if value == 0 {
        return Err(descriptor_error(format!("{label} zero is reserved")));
    }
    Ok(())
}

/// The only two terminal-marker publication forms for root-format version one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum GenerationPublicationFormV1 {
    /// A resolved/preflight operation seals final identities before the one canonical append.
    PrebuiltResolved = 1,
    /// A deterministic operation fences root-free fragments first, then seals roots in its one
    /// complete terminal marker after hidden GPU apply.
    WalFirstTerminalRoots = 2,
}

impl GenerationPublicationFormV1 {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::PrebuiltResolved),
            2 => Ok(Self::WalFirstTerminalRoots),
            other => Err(descriptor_error(format!(
                "unsupported publication form {other}"
            ))),
        }
    }
}

/// The closed table-transition tags carried by a v1 generation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum GenerationTableDeltaKindV1 {
    RowSet = 1,
    CreateEmpty = 2,
    Drop = 3,
    ResetEmpty = 4,
    Rebuild = 5,
}

impl GenerationTableDeltaKindV1 {
    fn decode(value: u8) -> Result<Self, EngineError> {
        match value {
            1 => Ok(Self::RowSet),
            2 => Ok(Self::CreateEmpty),
            3 => Ok(Self::Drop),
            4 => Ok(Self::ResetEmpty),
            5 => Ok(Self::Rebuild),
            other => Err(descriptor_error(format!(
                "unsupported table transition kind {other}"
            ))),
        }
    }
}

/// One present table side.  Absence is represented only by `None` on its parent transition and
/// serializes as all-zero scalar fields with a zero index count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenerationTableSideV1 {
    pub data_generation: u64,
    pub table_root: CanonicalDigest,
    pub logical_row_count: u64,
    pub indexes: Vec<GenerationIndexManifestV1>,
}

/// One ordered index manifest on either side of a table transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenerationIndexManifestV1 {
    pub stable_index_id: u64,
    pub index_shape_root: CanonicalDigest,
    pub index_generation: u64,
    pub index_root: CanonicalDigest,
}

impl GenerationIndexManifestV1 {
    fn validate(&self) -> Result<(), EngineError> {
        nonzero_id(self.stable_index_id, "stable index id")?;
        nonzero_id(self.index_generation, "index generation")?;
        nonzero_digest(&self.index_shape_root, "index shape root")?;
        nonzero_digest(&self.index_root, "index root")
    }
}

/// One complete before/after logical table transition.  The vectors on each side are independent:
/// an index CREATE/DROP is observable without inventing a union vector or a host-side shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenerationTableTransitionV1 {
    pub kind: GenerationTableDeltaKindV1,
    pub stable_table_id: u64,
    pub before: Option<GenerationTableSideV1>,
    pub after: Option<GenerationTableSideV1>,
}

/// The versioned root-format-v1 terminal descriptor.  Root bytes are durable comparison facts;
/// this WAL codec has no API for creating them from host rows or physical placement metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenerationTerminalDescriptorV1 {
    pub publication_form: GenerationPublicationFormV1,
    pub initial_database_root: CanonicalDigest,
    pub catalog_before_epoch: u64,
    pub catalog_before_digest: CanonicalDigest,
    pub catalog_after_epoch: u64,
    pub catalog_after_digest: CanonicalDigest,
    pub stable_transaction_id: u64,
    pub commit_sequence: u64,
    pub runtime_generation_input_digest: CanonicalDigest,
    pub final_database_root: CanonicalDigest,
    pub table_transitions: Vec<GenerationTableTransitionV1>,
}

/// Exact marker representation.  The legacy variant remains byte-identical to the historical
/// 92-byte marker, while the v1 variant appends one authenticated root descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CanonicalTerminalMarker {
    Legacy(CanonicalOutcome),
    GenerationV1 {
        outcome: CanonicalOutcome,
        descriptor: Box<GenerationTerminalDescriptorV1>,
    },
}

/// Exact measured marker geometry.  Callers reserve this many bytes before the terminal marker
/// is made durable; this small codec does not allocate during encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CanonicalTerminalMarkerMeasure {
    pub marker_bytes: usize,
    pub descriptor_bytes: Option<u32>,
}

impl CanonicalTerminalMarker {
    /// The decoded terminal outcome shared by legacy and extension forms.
    pub(crate) fn outcome(&self) -> &CanonicalOutcome {
        match self {
            Self::Legacy(outcome) | Self::GenerationV1 { outcome, .. } => outcome,
        }
    }

    fn validate(&self) -> Result<(), EngineError> {
        match self {
            Self::Legacy(outcome) => outcome.validate(),
            Self::GenerationV1 {
                outcome,
                descriptor,
            } => {
                outcome.validate()?;
                validate_generation_outcome_shape(outcome)?;
                descriptor.validate_for_outcome(outcome)
            }
        }
    }

    /// Validate the terminal descriptor against the immutable canonical envelope header.  The
    /// marker body is intentionally decodable without a header so WAL prefix scanning can retain
    /// it, but a complete envelope/replay path must close these duplicated identity fields before
    /// it uses the descriptor as publication evidence.
    pub(crate) fn validate_for_header(
        &self,
        header: &CanonicalPreApplyHeader,
    ) -> Result<(), EngineError> {
        self.validate()?;
        if let Self::GenerationV1 { descriptor, .. } = self {
            if descriptor.catalog_before_epoch != header.catalog_before_epoch
                || descriptor.catalog_before_digest != header.catalog_before_digest
                || descriptor.catalog_after_epoch != header.catalog_after_epoch
                || descriptor.catalog_after_digest != header.catalog_after_digest
                || descriptor.stable_transaction_id != header.stable_transaction_id
                || descriptor.commit_sequence != header.commit_seq
            {
                return Err(descriptor_error(
                    "terminal descriptor does not match canonical pre-apply header",
                ));
            }
        }
        Ok(())
    }
}

/// V1 terminal-outcome restrictions intentionally mirror the root-free generation input.  They
/// live here rather than in `CanonicalOutcome::validate` so exact legacy markers retain their
/// historical grammar and byte compatibility.
fn validate_generation_outcome_shape(outcome: &CanonicalOutcome) -> Result<(), EngineError> {
    match outcome.kind {
        CanonicalOutcomeKind::CommitSuccess if outcome.constraint_id == 0 => Ok(()),
        CanonicalOutcomeKind::CommitNoOp
            if outcome.constraint_id == 0 && outcome.affected_rows == 0 =>
        {
            Ok(())
        }
        CanonicalOutcomeKind::AbortError => {
            let Some(sqlstate) = outcome.sqlstate else {
                return Err(descriptor_error("AbortError SQLSTATE"));
            };
            if sqlstate
                .iter()
                .any(|byte| !byte.is_ascii_uppercase() && !byte.is_ascii_digit())
            {
                return Err(descriptor_error("AbortError SQLSTATE bytes"));
            }
            if outcome.affected_rows != 0 {
                return Err(descriptor_error("AbortError affected rows"));
            }
            Ok(())
        }
        CanonicalOutcomeKind::CommitSuccess | CanonicalOutcomeKind::CommitNoOp => {
            Err(descriptor_error("commit outcome shape"))
        }
    }
}

impl GenerationTerminalDescriptorV1 {
    /// Return the exact descriptor size without allocating or serializing it.
    pub(crate) fn encoded_len(&self) -> Result<u32, EngineError> {
        self.validate()?;
        let transitions = self.table_transitions.iter().try_fold(
            GENERATION_TERMINAL_DESCRIPTOR_FIXED_BYTES,
            |total, transition| {
                transition.encoded_len().and_then(|len| {
                    total
                        .checked_add(len)
                        .ok_or_else(|| descriptor_error("descriptor byte length overflow"))
                })
            },
        )?;
        u32::try_from(transitions)
            .map_err(|_| descriptor_error("descriptor byte length exceeds u32"))
    }

    fn validate(&self) -> Result<(), EngineError> {
        nonzero_digest(&self.initial_database_root, "initial database root")?;
        nonzero_digest(&self.final_database_root, "final database root")?;
        nonzero_digest(&self.catalog_before_digest, "catalog before digest")?;
        nonzero_digest(&self.catalog_after_digest, "catalog after digest")?;
        nonzero_digest(
            &self.runtime_generation_input_digest,
            "runtime generation input digest",
        )?;
        nonzero_id(self.stable_transaction_id, "stable transaction id")?;
        if self.commit_sequence == 0 || self.commit_sequence == u64::MAX {
            return Err(descriptor_error("commit sequence outside v1 domain"));
        }
        if self.catalog_after_epoch < self.catalog_before_epoch {
            return Err(descriptor_error("catalog epoch regressed"));
        }
        let mut previous = None;
        for transition in &self.table_transitions {
            transition.validate(self.commit_sequence)?;
            if previous >= Some(transition.stable_table_id) {
                return Err(descriptor_error(
                    "table transitions are not strictly stable-table-id ascending",
                ));
            }
            previous = Some(transition.stable_table_id);
        }
        if self.table_transitions.is_empty()
            && self.initial_database_root != self.final_database_root
        {
            return Err(descriptor_error(
                "zero-transition descriptor changes database data root",
            ));
        }
        Ok(())
    }

    fn validate_for_outcome(&self, outcome: &CanonicalOutcome) -> Result<(), EngineError> {
        self.validate()?;
        if outcome.kind == CanonicalOutcomeKind::CommitSuccess {
            return Ok(());
        }
        if self.initial_database_root != self.final_database_root {
            return Err(descriptor_error(
                "non-success descriptor changes database data root",
            ));
        }
        if self.catalog_before_epoch != self.catalog_after_epoch
            || self.catalog_before_digest != self.catalog_after_digest
        {
            return Err(descriptor_error(
                "non-success descriptor changes catalog identity",
            ));
        }
        if !self.table_transitions.is_empty() {
            return Err(descriptor_error(
                "non-success descriptor has table transitions",
            ));
        }
        Ok(())
    }

    fn encode_into(&self, out: &mut [u8]) -> Result<(), EngineError> {
        self.validate()?;
        let expected = usize::try_from(self.encoded_len()?)
            .map_err(|_| descriptor_error("descriptor length exceeds usize"))?;
        if out.len() != expected {
            return Err(descriptor_error(format!(
                "descriptor buffer must be exactly {expected} bytes, got {}",
                out.len()
            )));
        }
        let mut writer = DescriptorWriter::new(out);
        writer.u16(GENERATION_TERMINAL_EXTENSION_VERSION)?;
        writer.u8(self.publication_form as u8)?;
        writer.bytes(&[0; 5])?;
        writer.bytes(&self.initial_database_root)?;
        writer.u64(self.catalog_before_epoch)?;
        writer.bytes(&self.catalog_before_digest)?;
        writer.u64(self.catalog_after_epoch)?;
        writer.bytes(&self.catalog_after_digest)?;
        writer.u64(self.stable_transaction_id)?;
        writer.u64(self.commit_sequence)?;
        writer.bytes(&self.runtime_generation_input_digest)?;
        writer.bytes(&self.final_database_root)?;
        writer.u32(
            u32::try_from(self.table_transitions.len())
                .map_err(|_| descriptor_error("table transition count exceeds u32"))?,
        )?;
        for transition in &self.table_transitions {
            transition.encode_into(&mut writer)?;
        }
        writer.finish()
    }

    fn decode(bytes: &[u8]) -> Result<Self, EngineError> {
        let mut reader = DescriptorReader::new(bytes);
        let version = reader.u16()?;
        if version != GENERATION_TERMINAL_EXTENSION_VERSION {
            return Err(descriptor_error(format!(
                "unsupported descriptor version {version}"
            )));
        }
        let publication_form = GenerationPublicationFormV1::decode(reader.u8()?)?;
        if reader.take(5)? != [0; 5] {
            return Err(descriptor_error("non-zero descriptor reserved bytes"));
        }
        let initial_database_root = reader.array()?;
        let catalog_before_epoch = reader.u64()?;
        let catalog_before_digest = reader.array()?;
        let catalog_after_epoch = reader.u64()?;
        let catalog_after_digest = reader.array()?;
        let stable_transaction_id = reader.u64()?;
        let commit_sequence = reader.u64()?;
        let runtime_generation_input_digest = reader.array()?;
        let final_database_root = reader.array()?;
        let count = usize::try_from(reader.u32()?)
            .map_err(|_| descriptor_error("table transition count exceeds usize"))?;
        if count > reader.remaining() / GENERATION_TABLE_TRANSITION_FIXED_BYTES {
            return Err(descriptor_error(
                "table transition count exceeds remaining descriptor geometry",
            ));
        }
        let mut table_transitions = Vec::new();
        table_transitions
            .try_reserve_exact(count)
            .map_err(|_| descriptor_error("unable to reserve decoded table transitions"))?;
        for _ in 0..count {
            table_transitions.push(GenerationTableTransitionV1::decode(&mut reader)?);
        }
        reader.finish()?;
        let descriptor = Self {
            publication_form,
            initial_database_root,
            catalog_before_epoch,
            catalog_before_digest,
            catalog_after_epoch,
            catalog_after_digest,
            stable_transaction_id,
            commit_sequence,
            runtime_generation_input_digest,
            final_database_root,
            table_transitions,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
}

impl GenerationTableTransitionV1 {
    fn encoded_len(&self) -> Result<usize, EngineError> {
        self.validate_basic()?;
        self.before
            .as_ref()
            .into_iter()
            .chain(self.after.as_ref())
            .try_fold(GENERATION_TABLE_TRANSITION_FIXED_BYTES, |total, side| {
                side.indexes
                    .len()
                    .checked_mul(GENERATION_INDEX_MANIFEST_BYTES)
                    .and_then(|indexes| total.checked_add(indexes))
                    .ok_or_else(|| descriptor_error("table transition byte length overflow"))
            })
    }

    fn validate(&self, commit_sequence: u64) -> Result<(), EngineError> {
        self.validate_basic()?;
        match self.kind {
            GenerationTableDeltaKindV1::RowSet => {
                let (before, after) = required_sides(self)?;
                validate_row_set_index_shape_vector(before, after)?;
                generation_matches_commit(
                    after.data_generation,
                    commit_sequence,
                    "RowSet final table generation",
                )?;
                if before.table_root == after.table_root {
                    return Err(descriptor_error("RowSet table root"));
                }
            }
            GenerationTableDeltaKindV1::CreateEmpty => {
                let Some(after) = self.after.as_ref() else {
                    return Err(descriptor_error("CreateEmpty table sides"));
                };
                if self.before.is_some() {
                    return Err(descriptor_error("CreateEmpty table sides"));
                }
                if after.logical_row_count != 0 {
                    return Err(descriptor_error("CreateEmpty final row count"));
                }
            }
            GenerationTableDeltaKindV1::Drop => {
                if self.before.is_none() || self.after.is_some() {
                    return Err(descriptor_error("Drop table sides"));
                }
            }
            GenerationTableDeltaKindV1::ResetEmpty => {
                let (before, after) = required_sides(self)?;
                if before.logical_row_count == 0 || after.logical_row_count != 0 {
                    return Err(descriptor_error("ResetEmpty row counts"));
                }
            }
            GenerationTableDeltaKindV1::Rebuild => {
                let Some(after) = self.after.as_ref() else {
                    return Err(descriptor_error("Rebuild missing final table"));
                };
                if after.logical_row_count == 0 {
                    return Err(descriptor_error("Rebuild final row count"));
                }
            }
        }
        self.validate_generation_evolution(commit_sequence)
    }

    fn encode_into(&self, writer: &mut DescriptorWriter<'_>) -> Result<(), EngineError> {
        self.validate_basic()?;
        writer.u8(self.kind as u8)?;
        writer.u64(self.stable_table_id)?;
        writer.u8(u8::from(self.before.is_some()))?;
        writer.u8(u8::from(self.after.is_some()))?;
        writer.bytes(&[0; 5])?;
        write_table_side_scalars(writer, self.before.as_ref())?;
        write_table_side_scalars(writer, self.after.as_ref())?;
        write_table_indexes(writer, self.before.as_ref())?;
        write_table_indexes(writer, self.after.as_ref())
    }

    fn decode(reader: &mut DescriptorReader<'_>) -> Result<Self, EngineError> {
        let kind = GenerationTableDeltaKindV1::decode(reader.u8()?)?;
        let stable_table_id = reader.u64()?;
        let before_present = reader.flag("before table presence")?;
        let after_present = reader.flag("after table presence")?;
        if reader.take(5)? != [0; 5] {
            return Err(descriptor_error("non-zero table transition reserved bytes"));
        }
        let mut before = read_table_side_scalars(reader, before_present)?;
        let mut after = read_table_side_scalars(reader, after_present)?;
        read_table_indexes(reader, &mut before)?;
        read_table_indexes(reader, &mut after)?;
        let result = Self {
            kind,
            stable_table_id,
            before,
            after,
        };
        // Enclosing descriptor validation closes the transition's kind/presence relation after
        // all index vectors have been decoded; validate only scalar/index well-formedness here.
        result.validate_basic()?;
        Ok(result)
    }

    fn validate_basic(&self) -> Result<(), EngineError> {
        nonzero_id(self.stable_table_id, "stable table id")?;
        for side in [&self.before, &self.after].into_iter().flatten() {
            side.validate()?;
        }
        Ok(())
    }

    fn validate_generation_evolution(&self, commit_sequence: u64) -> Result<(), EngineError> {
        if let Some(before) = self.before.as_ref() {
            generation_precedes_commit(
                before.data_generation,
                commit_sequence,
                "prior table generation",
            )?;
            for index in &before.indexes {
                generation_precedes_commit(
                    index.index_generation,
                    commit_sequence,
                    "prior index generation",
                )?;
            }
        }

        let Some(after) = self.after.as_ref() else {
            return Ok(());
        };
        match self.before.as_ref() {
            None => generation_matches_commit(
                after.data_generation,
                commit_sequence,
                "new table generation",
            )?,
            Some(before) if table_manifest_facts_unchanged(before, after) => {
                if after.data_generation != before.data_generation {
                    return Err(descriptor_error("unchanged table generation"));
                }
            }
            Some(_) => generation_matches_commit(
                after.data_generation,
                commit_sequence,
                "changed table generation",
            )?,
        }

        let mut before_index = self
            .before
            .as_ref()
            .map_or(&[][..], |before| before.indexes.as_slice())
            .iter()
            .peekable();
        for index in &after.indexes {
            while before_index
                .peek()
                .is_some_and(|before| before.stable_index_id < index.stable_index_id)
            {
                before_index.next();
            }
            let Some(before) =
                before_index.next_if(|before| before.stable_index_id == index.stable_index_id)
            else {
                generation_matches_commit(
                    index.index_generation,
                    commit_sequence,
                    "new index generation",
                )?;
                continue;
            };
            if before.index_root == index.index_root
                && before.index_shape_root == index.index_shape_root
            {
                if index.index_generation != before.index_generation {
                    return Err(descriptor_error("unchanged index generation"));
                }
            } else {
                generation_matches_commit(
                    index.index_generation,
                    commit_sequence,
                    "changed index generation",
                )?;
            }
        }
        Ok(())
    }
}

impl GenerationTableSideV1 {
    fn validate(&self) -> Result<(), EngineError> {
        nonzero_id(self.data_generation, "table data generation")?;
        nonzero_digest(&self.table_root, "table root")?;
        let mut previous = None;
        for index in &self.indexes {
            index.validate()?;
            if previous >= Some(index.stable_index_id) {
                return Err(descriptor_error(
                    "table indexes are not strictly stable-index-id ascending",
                ));
            }
            previous = Some(index.stable_index_id);
        }
        Ok(())
    }
}

fn required_sides(
    transition: &GenerationTableTransitionV1,
) -> Result<(&GenerationTableSideV1, &GenerationTableSideV1), EngineError> {
    match (&transition.before, &transition.after) {
        (Some(before), Some(after)) => Ok((before, after)),
        _ => Err(descriptor_error("table transition sides")),
    }
}

fn validate_row_set_index_shape_vector(
    before: &GenerationTableSideV1,
    after: &GenerationTableSideV1,
) -> Result<(), EngineError> {
    if before.indexes.len() != after.indexes.len()
        || before
            .indexes
            .iter()
            .zip(&after.indexes)
            .any(|(before, after)| {
                before.stable_index_id != after.stable_index_id
                    || before.index_shape_root != after.index_shape_root
            })
    {
        return Err(descriptor_error("RowSet index ID/shape vector"));
    }
    Ok(())
}

/// This is a local comparison of descriptor assertions, not a root recomputation. A retained
/// table generation is valid only when every V1 table-manifest fact available to this codec is
/// retained exactly; any visible row or index manifest mutation instead consumes the commit
/// generation.
fn table_manifest_facts_unchanged(
    before: &GenerationTableSideV1,
    after: &GenerationTableSideV1,
) -> bool {
    before.table_root == after.table_root
        && before.logical_row_count == after.logical_row_count
        && before.indexes.len() == after.indexes.len()
        && before
            .indexes
            .iter()
            .zip(&after.indexes)
            .all(|(before, after)| {
                before.stable_index_id == after.stable_index_id
                    && before.index_shape_root == after.index_shape_root
                    && before.index_generation == after.index_generation
                    && before.index_root == after.index_root
            })
}

fn generation_precedes_commit(
    generation: u64,
    commit_sequence: u64,
    label: &str,
) -> Result<(), EngineError> {
    if generation >= commit_sequence {
        return Err(descriptor_error(format!(
            "{label} must precede commit sequence"
        )));
    }
    Ok(())
}

fn generation_matches_commit(
    generation: u64,
    commit_sequence: u64,
    label: &str,
) -> Result<(), EngineError> {
    if generation != commit_sequence {
        return Err(descriptor_error(format!(
            "{label} must equal commit sequence"
        )));
    }
    Ok(())
}

fn write_table_side_scalars(
    writer: &mut DescriptorWriter<'_>,
    side: Option<&GenerationTableSideV1>,
) -> Result<(), EngineError> {
    match side {
        Some(side) => {
            side.validate()?;
            writer.u64(side.data_generation)?;
            writer.bytes(&side.table_root)?;
            writer.u64(side.logical_row_count)?;
        }
        None => {
            writer.u64(0)?;
            writer.bytes(&[0; 32])?;
            writer.u64(0)?;
        }
    }
    Ok(())
}

fn write_table_indexes(
    writer: &mut DescriptorWriter<'_>,
    side: Option<&GenerationTableSideV1>,
) -> Result<(), EngineError> {
    let indexes = side.map_or(&[][..], |side| side.indexes.as_slice());
    writer.u32(
        u32::try_from(indexes.len()).map_err(|_| descriptor_error("index count exceeds u32"))?,
    )?;
    for index in indexes {
        writer.u64(index.stable_index_id)?;
        writer.bytes(&index.index_shape_root)?;
        writer.u64(index.index_generation)?;
        writer.bytes(&index.index_root)?;
    }
    Ok(())
}

fn read_table_side_scalars(
    reader: &mut DescriptorReader<'_>,
    present: bool,
) -> Result<Option<GenerationTableSideV1>, EngineError> {
    let data_generation = reader.u64()?;
    let table_root = reader.array()?;
    let logical_row_count = reader.u64()?;
    if !present {
        if data_generation != 0 || table_root != [0; 32] || logical_row_count != 0 {
            return Err(descriptor_error("absent table side carries data"));
        }
        return Ok(None);
    }
    Ok(Some(GenerationTableSideV1 {
        data_generation,
        table_root,
        logical_row_count,
        indexes: Vec::new(),
    }))
}

fn read_table_indexes(
    reader: &mut DescriptorReader<'_>,
    side: &mut Option<GenerationTableSideV1>,
) -> Result<(), EngineError> {
    let index_count = usize::try_from(reader.u32()?)
        .map_err(|_| descriptor_error("index count exceeds usize"))?;
    if side.is_none() {
        if index_count != 0 {
            return Err(descriptor_error("absent table side carries indexes"));
        }
        return Ok(());
    }
    if index_count > reader.remaining() / GENERATION_INDEX_MANIFEST_BYTES {
        return Err(descriptor_error(
            "index count exceeds remaining descriptor geometry",
        ));
    }
    let mut indexes = Vec::new();
    indexes
        .try_reserve_exact(index_count)
        .map_err(|_| descriptor_error("unable to reserve decoded indexes"))?;
    for _ in 0..index_count {
        indexes.push(GenerationIndexManifestV1 {
            stable_index_id: reader.u64()?,
            index_shape_root: reader.array()?,
            index_generation: reader.u64()?,
            index_root: reader.array()?,
        });
    }
    let side = side.as_mut().expect("present side checked above");
    side.indexes = indexes;
    side.validate()
}

/// Measure the exact terminal-marker body size without allocating it.
pub(crate) fn measure_canonical_terminal_marker(
    marker: &CanonicalTerminalMarker,
) -> Result<CanonicalTerminalMarkerMeasure, EngineError> {
    marker.validate()?;
    match marker {
        CanonicalTerminalMarker::Legacy(_) => Ok(CanonicalTerminalMarkerMeasure {
            marker_bytes: CANONICAL_OUTCOME_BYTES,
            descriptor_bytes: None,
        }),
        CanonicalTerminalMarker::GenerationV1 { descriptor, .. } => {
            let descriptor_bytes = descriptor.encoded_len()?;
            let marker_bytes = CANONICAL_OUTCOME_BYTES
                .checked_add(GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES)
                .and_then(|bytes| bytes.checked_add(usize::try_from(descriptor_bytes).ok()?))
                .ok_or_else(|| descriptor_error("terminal marker byte length overflow"))?;
            Ok(CanonicalTerminalMarkerMeasure {
                marker_bytes,
                descriptor_bytes: Some(descriptor_bytes),
            })
        }
    }
}

/// Encode an exact legacy or v1 generation terminal-marker body into caller-owned bytes.
pub(crate) fn encode_canonical_terminal_marker_into(
    marker: &CanonicalTerminalMarker,
    out: &mut [u8],
) -> Result<(), EngineError> {
    let measure = measure_canonical_terminal_marker(marker)?;
    if out.len() != measure.marker_bytes {
        return Err(descriptor_error(format!(
            "terminal marker buffer must be exactly {} bytes, got {}",
            measure.marker_bytes,
            out.len()
        )));
    }
    let mut outcome = [0; CANONICAL_OUTCOME_BYTES];
    encode_canonical_outcome_into_exact(marker.outcome(), &mut outcome)?;
    out[..CANONICAL_OUTCOME_BYTES].copy_from_slice(&outcome);
    if let CanonicalTerminalMarker::GenerationV1 { descriptor, .. } = marker {
        let mut at = CANONICAL_OUTCOME_BYTES;
        out[at..at + GENERATION_TERMINAL_EXTENSION_MAGIC.len()]
            .copy_from_slice(&GENERATION_TERMINAL_EXTENSION_MAGIC);
        at += GENERATION_TERMINAL_EXTENSION_MAGIC.len();
        out[at..at + 2].copy_from_slice(&GENERATION_TERMINAL_EXTENSION_VERSION.to_le_bytes());
        at += 2;
        out[at..at + 2].fill(0);
        at += 2;
        let descriptor_bytes = measure.descriptor_bytes.expect("generation marker measure");
        out[at..at + 4].copy_from_slice(&descriptor_bytes.to_le_bytes());
        at += 4;
        descriptor.encode_into(&mut out[at..])?;
    }
    Ok(())
}

/// Decode a canonical marker body.  A body of exactly 92 bytes is legacy; any longer body must
/// use the complete root-format-v1 extension grammar with no trailing byte.
pub(crate) fn decode_canonical_terminal_marker(
    bytes: &[u8],
) -> Result<CanonicalTerminalMarker, EngineError> {
    if bytes.len() < CANONICAL_OUTCOME_BYTES {
        return Err(descriptor_error(
            "terminal marker truncates canonical outcome",
        ));
    }
    let outcome = decode_canonical_outcome_exact(&bytes[..CANONICAL_OUTCOME_BYTES])?;
    if bytes.len() == CANONICAL_OUTCOME_BYTES {
        return Ok(CanonicalTerminalMarker::Legacy(outcome));
    }
    let extension = &bytes[CANONICAL_OUTCOME_BYTES..];
    if extension.len() < GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES {
        return Err(descriptor_error("terminal extension prefix is truncated"));
    }
    let mut reader = DescriptorReader::new(extension);
    if reader.take(GENERATION_TERMINAL_EXTENSION_MAGIC.len())?
        != GENERATION_TERMINAL_EXTENSION_MAGIC
    {
        return Err(descriptor_error("terminal extension magic"));
    }
    let version = reader.u16()?;
    if version != GENERATION_TERMINAL_EXTENSION_VERSION {
        return Err(descriptor_error(format!(
            "unsupported terminal extension version {version}"
        )));
    }
    if reader.take(2)? != [0; 2] {
        return Err(descriptor_error(
            "non-zero terminal extension reserved bytes",
        ));
    }
    let descriptor_bytes = usize::try_from(reader.u32()?)
        .map_err(|_| descriptor_error("descriptor byte length exceeds usize"))?;
    if reader.remaining() != descriptor_bytes {
        return Err(descriptor_error("terminal extension descriptor length"));
    }
    let descriptor = GenerationTerminalDescriptorV1::decode(reader.take(descriptor_bytes)?)?;
    reader.finish()?;
    let marker = CanonicalTerminalMarker::GenerationV1 {
        outcome,
        descriptor: Box::new(descriptor),
    };
    marker.validate()?;
    Ok(marker)
}

/// Return the authenticated terminal-envelope digest for either marker form.  Legacy markers
/// preserve the historical digest domain exactly; only the v1 extension uses its new domain.
#[cfg(test)]
fn canonical_terminal_marker_digest(
    header: &CanonicalPreApplyHeader,
    ordered_fragment_root: CanonicalDigest,
    marker: &CanonicalTerminalMarker,
) -> Result<CanonicalDigest, EngineError> {
    marker.validate_for_header(header)?;
    let header_bytes = header.encode()?;
    let measure = measure_canonical_terminal_marker(marker)?;
    let mut marker_bytes = vec![0; measure.marker_bytes];
    encode_canonical_terminal_marker_into(marker, &mut marker_bytes)?;
    canonical_terminal_marker_digest_from_encoded(
        &header_bytes,
        ordered_fragment_root,
        marker,
        &marker_bytes,
    )
}

/// Hash the exact marker bytes that an enclosing encoder has already measured and emitted.  This
/// deliberately takes the validated marker only to select its format domain: callers use it only
/// on a just-encoded body or on the exact body just decoded from an authenticated frame.
pub(super) fn canonical_terminal_marker_digest_from_encoded(
    header_bytes: &[u8],
    ordered_fragment_root: CanonicalDigest,
    marker: &CanonicalTerminalMarker,
    marker_bytes: &[u8],
) -> Result<CanonicalDigest, EngineError> {
    let measure = measure_canonical_terminal_marker(marker)?;
    if marker_bytes.len() != measure.marker_bytes {
        return Err(descriptor_error(
            "terminal marker bytes do not match measured geometry",
        ));
    }
    let domain = match marker {
        CanonicalTerminalMarker::Legacy(_) => FINAL_DOMAIN,
        CanonicalTerminalMarker::GenerationV1 { .. } => RUNTIME_TERMINAL_DOMAIN,
    };
    Ok(digest_parts(
        domain,
        &[header_bytes, &ordered_fragment_root, marker_bytes],
    ))
}

struct DescriptorWriter<'a> {
    bytes: &'a mut [u8],
    at: usize,
}

impl<'a> DescriptorWriter<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn u8(&mut self, value: u8) -> Result<(), EngineError> {
        self.bytes(value.to_le_bytes().as_slice())
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

    fn bytes(&mut self, value: &[u8]) -> Result<(), EngineError> {
        let end = self
            .at
            .checked_add(value.len())
            .ok_or_else(|| descriptor_error("descriptor write offset overflow"))?;
        let destination = self
            .bytes
            .get_mut(self.at..end)
            .ok_or_else(|| descriptor_error("descriptor write exceeds buffer"))?;
        destination.copy_from_slice(value);
        self.at = end;
        Ok(())
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(descriptor_error("descriptor did not fill exact buffer"))
        }
    }
}

struct DescriptorReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> DescriptorReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .at
            .checked_add(count)
            .ok_or_else(|| descriptor_error("descriptor read offset overflow"))?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| descriptor_error("descriptor is truncated"))?;
        self.at = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("exact slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("exact slice"),
        ))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("exact slice"),
        ))
    }

    fn array(&mut self) -> Result<[u8; 32], EngineError> {
        Ok(self.take(32)?.try_into().expect("exact slice"))
    }

    fn flag(&mut self, label: &str) -> Result<bool, EngineError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(descriptor_error(format!("invalid {label} flag {other}"))),
        }
    }

    fn finish(self) -> Result<(), EngineError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(descriptor_error("descriptor has trailing bytes"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> CanonicalDigest {
        [seed; 32]
    }

    fn header() -> CanonicalPreApplyHeader {
        CanonicalPreApplyHeader {
            identity: CanonicalIdentity {
                database_id: [1; 16],
                cluster_id: [2; 16],
                timeline_id: [3; 16],
                format_epoch: 4,
            },
            leader_epoch: 5,
            commit_seq: 9,
            stable_transaction_id: 8,
            request_digest: digest(6),
            isolation: CanonicalIsolation::ReadCommitted,
            flags: 7,
            catalog_before_epoch: 10,
            catalog_after_epoch: 11,
            catalog_before_digest: digest(12),
            catalog_after_digest: digest(13),
            operation_count: 1,
            table_block_count: 1,
            allocator_high_water: 14,
        }
    }

    fn outcome() -> CanonicalOutcome {
        CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 2,
            sqlstate: None,
            constraint_id: 0,
            target_digest: digest(15),
            returning_digest: digest(16),
        }
    }

    fn physical() -> CanonicalPhysicalRange {
        CanonicalPhysicalRange {
            log_epoch: 5,
            lane_id: 3,
            segment_id: 4,
            first_frame_ordinal: 7,
        }
    }

    fn index(id: u64, generation: u64, shape: u8, root: u8) -> GenerationIndexManifestV1 {
        GenerationIndexManifestV1 {
            stable_index_id: id,
            index_shape_root: digest(shape),
            index_generation: generation,
            index_root: digest(root),
        }
    }

    fn side(
        generation: u64,
        root: u8,
        rows: u64,
        indexes: Vec<GenerationIndexManifestV1>,
    ) -> GenerationTableSideV1 {
        GenerationTableSideV1 {
            data_generation: generation,
            table_root: digest(root),
            logical_row_count: rows,
            indexes,
        }
    }

    fn descriptor() -> GenerationTerminalDescriptorV1 {
        GenerationTerminalDescriptorV1 {
            publication_form: GenerationPublicationFormV1::PrebuiltResolved,
            initial_database_root: digest(20),
            catalog_before_epoch: 10,
            catalog_before_digest: digest(12),
            catalog_after_epoch: 11,
            catalog_after_digest: digest(13),
            stable_transaction_id: 8,
            commit_sequence: 9,
            runtime_generation_input_digest: digest(21),
            final_database_root: digest(22),
            table_transitions: vec![GenerationTableTransitionV1 {
                kind: GenerationTableDeltaKindV1::RowSet,
                stable_table_id: 4,
                before: Some(side(6, 23, 1, vec![index(3, 6, 24, 25)])),
                after: Some(side(9, 25, 2, vec![index(3, 9, 24, 26)])),
            }],
        }
    }

    fn success_marker(descriptor: GenerationTerminalDescriptorV1) -> CanonicalTerminalMarker {
        CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(descriptor),
        }
    }

    fn commit_no_op_outcome() -> CanonicalOutcome {
        CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitNoOp,
            affected_rows: 0,
            sqlstate: None,
            constraint_id: 0,
            target_digest: digest(15),
            returning_digest: digest(16),
        }
    }

    fn abort_outcome() -> CanonicalOutcome {
        CanonicalOutcome {
            kind: CanonicalOutcomeKind::AbortError,
            affected_rows: 0,
            sqlstate: Some(*b"23505"),
            constraint_id: 3,
            target_digest: digest(15),
            returning_digest: digest(16),
        }
    }

    fn non_success_descriptor() -> GenerationTerminalDescriptorV1 {
        let mut descriptor = descriptor();
        descriptor.final_database_root = descriptor.initial_database_root;
        descriptor.catalog_after_epoch = descriptor.catalog_before_epoch;
        descriptor.catalog_after_digest = descriptor.catalog_before_digest;
        descriptor.table_transitions.clear();
        descriptor
    }

    fn marker_with(
        outcome: CanonicalOutcome,
        descriptor: GenerationTerminalDescriptorV1,
    ) -> CanonicalTerminalMarker {
        CanonicalTerminalMarker::GenerationV1 {
            outcome,
            descriptor: Box::new(descriptor),
        }
    }

    fn encode_marker(marker: &CanonicalTerminalMarker) -> Vec<u8> {
        let measure = measure_canonical_terminal_marker(marker).expect("valid marker measure");
        let mut bytes = vec![0; measure.marker_bytes];
        encode_canonical_terminal_marker_into(marker, &mut bytes).expect("valid marker encode");
        bytes
    }

    fn assert_encode_rejects(marker: &CanonicalTerminalMarker, label: &str) {
        let error = encode_canonical_terminal_marker_into(marker, &mut [])
            .expect_err("semantic-invalid marker must reject during encode");
        assert!(
            error.to_string().contains(label),
            "expected {label:?}, got {error}"
        );
    }

    fn assert_decode_rejects(bytes: &[u8], label: &str) {
        let error = decode_canonical_terminal_marker(bytes)
            .expect_err("semantic-invalid marker must reject during decode");
        assert!(
            error.to_string().contains(label),
            "expected {label:?}, got {error}"
        );
    }

    fn assert_descriptor_rejected_on_encode_and_decode(
        invalid: GenerationTerminalDescriptorV1,
        valid: GenerationTerminalDescriptorV1,
        mutate_encoded: impl FnOnce(&mut [u8]),
        label: &str,
    ) {
        assert_encode_rejects(&success_marker(invalid), label);
        let mut bytes = encode_marker(&success_marker(valid));
        mutate_encoded(&mut bytes);
        assert_decode_rejects(&bytes, label);
    }

    #[test]
    fn legacy_marker_is_exact_92_byte_outcome_without_extension() {
        let marker = CanonicalTerminalMarker::Legacy(outcome());
        let measure = measure_canonical_terminal_marker(&marker).expect("legacy marker measure");
        assert_eq!(measure.marker_bytes, CANONICAL_OUTCOME_BYTES);
        assert_eq!(measure.descriptor_bytes, None);
        let mut bytes = vec![0; measure.marker_bytes];
        encode_canonical_terminal_marker_into(&marker, &mut bytes).expect("legacy encode");
        assert_eq!(decode_canonical_terminal_marker(&bytes).unwrap(), marker);
        assert_eq!(decode_canonical_outcome_exact(&bytes).unwrap(), outcome());

        let mut historical_abort = outcome();
        historical_abort.kind = CanonicalOutcomeKind::AbortError;
        historical_abort.affected_rows = 1;
        historical_abort.sqlstate = Some(*b"23a05");
        historical_abort.constraint_id = 3;
        let historical_marker = CanonicalTerminalMarker::Legacy(historical_abort.clone());
        let bytes = encode_marker(&historical_marker);
        assert_eq!(
            decode_canonical_terminal_marker(&bytes).unwrap(),
            historical_marker,
            "V1-only outcome shape restrictions must not alter the legacy marker grammar"
        );
    }

    #[test]
    fn generation_marker_round_trips_exact_descriptor_and_binds_extension_bytes() {
        let marker = CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(descriptor()),
        };
        let measure = measure_canonical_terminal_marker(&marker).expect("generation measure");
        assert_eq!(measure.descriptor_bytes, Some(484));
        assert_eq!(measure.marker_bytes, 92 + 24 + 484);
        let mut bytes = vec![0; measure.marker_bytes];
        encode_canonical_terminal_marker_into(&marker, &mut bytes).expect("generation encode");
        assert_eq!(&bytes[92..108], GENERATION_TERMINAL_EXTENSION_MAGIC);
        assert_eq!(&bytes[108..110], &1_u16.to_le_bytes());
        assert_eq!(&bytes[110..112], &[0; 2]);
        assert_eq!(&bytes[112..116], &484_u32.to_le_bytes());

        // The descriptor is intentionally a literal field-order codec.  These byte offsets pin
        // the specified transition grammar: both scalar tuples precede either index vector.
        const DESCRIPTOR: usize = 116;
        const TRANSITION: usize = DESCRIPTOR + 204;
        assert_eq!(&bytes[DESCRIPTOR..DESCRIPTOR + 2], &1_u16.to_le_bytes());
        assert_eq!(bytes[DESCRIPTOR + 2], 1);
        assert_eq!(&bytes[DESCRIPTOR + 3..DESCRIPTOR + 8], &[0; 5]);
        assert_eq!(&bytes[DESCRIPTOR + 8..DESCRIPTOR + 40], &digest(20));
        assert_eq!(
            &bytes[DESCRIPTOR + 40..DESCRIPTOR + 48],
            &10_u64.to_le_bytes()
        );
        assert_eq!(&bytes[DESCRIPTOR + 48..DESCRIPTOR + 80], &digest(12));
        assert_eq!(
            &bytes[DESCRIPTOR + 80..DESCRIPTOR + 88],
            &11_u64.to_le_bytes()
        );
        assert_eq!(&bytes[DESCRIPTOR + 88..DESCRIPTOR + 120], &digest(13));
        assert_eq!(
            &bytes[DESCRIPTOR + 120..DESCRIPTOR + 128],
            &8_u64.to_le_bytes()
        );
        assert_eq!(
            &bytes[DESCRIPTOR + 128..DESCRIPTOR + 136],
            &9_u64.to_le_bytes()
        );
        assert_eq!(&bytes[DESCRIPTOR + 136..DESCRIPTOR + 168], &digest(21));
        assert_eq!(&bytes[DESCRIPTOR + 168..DESCRIPTOR + 200], &digest(22));
        assert_eq!(
            &bytes[DESCRIPTOR + 200..DESCRIPTOR + 204],
            &1_u32.to_le_bytes()
        );
        assert_eq!(bytes[TRANSITION], GenerationTableDeltaKindV1::RowSet as u8);
        assert_eq!(&bytes[TRANSITION + 1..TRANSITION + 9], &4_u64.to_le_bytes());
        assert_eq!(&bytes[TRANSITION + 9..TRANSITION + 11], &[1, 1]);
        assert_eq!(&bytes[TRANSITION + 11..TRANSITION + 16], &[0; 5]);
        assert_eq!(
            &bytes[TRANSITION + 16..TRANSITION + 24],
            &6_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 24..TRANSITION + 56], &digest(23));
        assert_eq!(
            &bytes[TRANSITION + 56..TRANSITION + 64],
            &1_u64.to_le_bytes()
        );
        assert_eq!(
            &bytes[TRANSITION + 64..TRANSITION + 72],
            &9_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 72..TRANSITION + 104], &digest(25));
        assert_eq!(
            &bytes[TRANSITION + 104..TRANSITION + 112],
            &2_u64.to_le_bytes()
        );
        assert_eq!(
            &bytes[TRANSITION + 112..TRANSITION + 116],
            &1_u32.to_le_bytes()
        );
        assert_eq!(
            &bytes[TRANSITION + 116..TRANSITION + 124],
            &3_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 124..TRANSITION + 156], &digest(24));
        assert_eq!(
            &bytes[TRANSITION + 156..TRANSITION + 164],
            &6_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 164..TRANSITION + 196], &digest(25));
        assert_eq!(
            &bytes[TRANSITION + 196..TRANSITION + 200],
            &1_u32.to_le_bytes()
        );
        assert_eq!(
            &bytes[TRANSITION + 200..TRANSITION + 208],
            &3_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 208..TRANSITION + 240], &digest(24));
        assert_eq!(
            &bytes[TRANSITION + 240..TRANSITION + 248],
            &9_u64.to_le_bytes()
        );
        assert_eq!(&bytes[TRANSITION + 248..TRANSITION + 280], &digest(26));
        assert_eq!(decode_canonical_terminal_marker(&bytes).unwrap(), marker);

        let root = digest(27);
        let generation_digest = canonical_terminal_marker_digest(&header(), root, &marker).unwrap();
        let legacy_digest = canonical_terminal_marker_digest(
            &header(),
            root,
            &CanonicalTerminalMarker::Legacy(outcome()),
        )
        .unwrap();
        assert_ne!(generation_digest, legacy_digest);
        bytes[116 + 168..116 + 200].fill(0);
        assert!(
            decode_canonical_terminal_marker(&bytes).is_err(),
            "descriptor substitutions fail before a marker can be accepted"
        );
    }

    #[test]
    fn terminal_marker_rejects_extension_length_reserved_presence_and_shape_sabotage() {
        let marker = CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(descriptor()),
        };
        let mut bytes = vec![
            0;
            measure_canonical_terminal_marker(&marker)
                .unwrap()
                .marker_bytes
        ];
        encode_canonical_terminal_marker_into(&marker, &mut bytes).unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[92] ^= 1;
        assert!(decode_canonical_terminal_marker(&bad_magic).is_err());
        let mut bad_reserved = bytes.clone();
        bad_reserved[110] = 1;
        assert!(decode_canonical_terminal_marker(&bad_reserved).is_err());
        let mut bad_length = bytes.clone();
        bad_length[112..116].copy_from_slice(&399_u32.to_le_bytes());
        assert!(decode_canonical_terminal_marker(&bad_length).is_err());
        assert!(decode_canonical_terminal_marker(&bytes[..bytes.len() - 1]).is_err());
        let mut surplus = bytes.clone();
        surplus.push(0);
        assert!(decode_canonical_terminal_marker(&surplus).is_err());

        let mut impossible_transition_count = bytes.clone();
        // The descriptor decoder bounds counts against remaining fixed-width geometry before it
        // reserves a vector, so a tiny hostile marker cannot request a u32-sized allocation.
        impossible_transition_count[116 + 200..116 + 204].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_canonical_terminal_marker(&impossible_transition_count).is_err());

        let mut impossible_index_count = bytes.clone();
        impossible_index_count[116 + 204 + 112..116 + 204 + 116]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_canonical_terminal_marker(&impossible_index_count).is_err());

        let mut absent_side_data = bytes;
        // Descriptor fixed fields (204) followed by table kind/id/presence/reserved.  Clear the
        // before-present flag while preserving its serialized before side.
        absent_side_data[116 + 204 + 9] = 0;
        assert!(decode_canonical_terminal_marker(&absent_side_data).is_err());
    }

    #[test]
    fn v1_outcome_shape_rejects_invalid_commit_noop_and_abort_facts_on_encode_and_decode() {
        const CONSTRAINT_ID: usize = 12;
        const AFFECTED_ROWS: usize = 4;
        const SQLSTATE_PRESENT: usize = 1;
        const SQLSTATE: usize = 20;

        let mut invalid_success = outcome();
        invalid_success.constraint_id = 1;
        assert_encode_rejects(
            &marker_with(invalid_success, descriptor()),
            "commit outcome shape",
        );
        let mut bytes = encode_marker(&success_marker(descriptor()));
        bytes[CONSTRAINT_ID..CONSTRAINT_ID + 8].copy_from_slice(&1_u64.to_le_bytes());
        assert_decode_rejects(&bytes, "commit outcome shape");

        let mut invalid_no_op = commit_no_op_outcome();
        invalid_no_op.constraint_id = 1;
        assert_encode_rejects(
            &marker_with(invalid_no_op, non_success_descriptor()),
            "commit outcome shape",
        );
        let mut bytes = encode_marker(&marker_with(
            commit_no_op_outcome(),
            non_success_descriptor(),
        ));
        bytes[CONSTRAINT_ID..CONSTRAINT_ID + 8].copy_from_slice(&1_u64.to_le_bytes());
        assert_decode_rejects(&bytes, "commit outcome shape");

        let mut invalid_no_op = commit_no_op_outcome();
        invalid_no_op.affected_rows = 1;
        assert_encode_rejects(
            &marker_with(invalid_no_op, non_success_descriptor()),
            "commit outcome shape",
        );
        let mut bytes = encode_marker(&marker_with(
            commit_no_op_outcome(),
            non_success_descriptor(),
        ));
        bytes[AFFECTED_ROWS..AFFECTED_ROWS + 8].copy_from_slice(&1_u64.to_le_bytes());
        assert_decode_rejects(&bytes, "commit outcome shape");

        let mut invalid_abort = abort_outcome();
        invalid_abort.affected_rows = 1;
        assert_encode_rejects(
            &marker_with(invalid_abort, non_success_descriptor()),
            "AbortError affected rows",
        );
        let mut bytes = encode_marker(&marker_with(abort_outcome(), non_success_descriptor()));
        bytes[AFFECTED_ROWS..AFFECTED_ROWS + 8].copy_from_slice(&1_u64.to_le_bytes());
        assert_decode_rejects(&bytes, "AbortError affected rows");

        let mut invalid_abort = abort_outcome();
        invalid_abort.sqlstate = Some(*b"23a05");
        assert_encode_rejects(
            &marker_with(invalid_abort, non_success_descriptor()),
            "AbortError SQLSTATE bytes",
        );
        let mut bytes = encode_marker(&marker_with(abort_outcome(), non_success_descriptor()));
        bytes[SQLSTATE_PRESENT] = 1;
        bytes[SQLSTATE..SQLSTATE + 5].copy_from_slice(b"23a05");
        assert_decode_rejects(&bytes, "AbortError SQLSTATE bytes");
    }

    #[test]
    fn v1_non_success_and_catalog_only_closure_rejects_descriptor_contradictions() {
        const DESCRIPTOR: usize =
            CANONICAL_OUTCOME_BYTES + GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES;
        const INITIAL_DATABASE_ROOT: usize = DESCRIPTOR + 8;
        const CATALOG_BEFORE_DIGEST: usize = DESCRIPTOR + 48;
        const CATALOG_AFTER_EPOCH: usize = DESCRIPTOR + 80;
        const CATALOG_AFTER_DIGEST: usize = DESCRIPTOR + 88;
        const FINAL_DATABASE_ROOT: usize = DESCRIPTOR + 168;

        let mut valid_with_transition = non_success_descriptor();
        valid_with_transition.table_transitions = descriptor().table_transitions;
        let mut invalid = valid_with_transition.clone();
        invalid.final_database_root = digest(22);
        assert_encode_rejects(
            &marker_with(commit_no_op_outcome(), invalid),
            "non-success descriptor changes database data root",
        );
        let mut bytes = encode_marker(&success_marker(valid_with_transition));
        bytes[FINAL_DATABASE_ROOT..FINAL_DATABASE_ROOT + 32].copy_from_slice(&digest(22));
        bytes[0] = CanonicalOutcomeKind::CommitNoOp as u8;
        bytes[4..12].fill(0);
        assert_decode_rejects(&bytes, "non-success descriptor changes database data root");

        let mut invalid = non_success_descriptor();
        invalid.catalog_after_epoch += 1;
        assert_encode_rejects(
            &marker_with(commit_no_op_outcome(), invalid),
            "non-success descriptor changes catalog identity",
        );
        let mut bytes = encode_marker(&marker_with(
            commit_no_op_outcome(),
            non_success_descriptor(),
        ));
        bytes[CATALOG_AFTER_EPOCH..CATALOG_AFTER_EPOCH + 8].copy_from_slice(&11_u64.to_le_bytes());
        assert_decode_rejects(&bytes, "non-success descriptor changes catalog identity");

        let mut invalid = non_success_descriptor();
        invalid.catalog_after_digest = digest(13);
        assert_encode_rejects(
            &marker_with(commit_no_op_outcome(), invalid),
            "non-success descriptor changes catalog identity",
        );
        let mut bytes = encode_marker(&marker_with(
            commit_no_op_outcome(),
            non_success_descriptor(),
        ));
        bytes[CATALOG_AFTER_DIGEST..CATALOG_AFTER_DIGEST + 32].copy_from_slice(&digest(13));
        assert_decode_rejects(&bytes, "non-success descriptor changes catalog identity");

        let mut invalid = non_success_descriptor();
        invalid.table_transitions = descriptor().table_transitions;
        assert_encode_rejects(
            &marker_with(commit_no_op_outcome(), invalid.clone()),
            "non-success descriptor has table transitions",
        );
        let transition_marker = success_marker(invalid);
        let mut bytes = encode_marker(&transition_marker);
        bytes[0] = CanonicalOutcomeKind::CommitNoOp as u8;
        bytes[4..12].fill(0);
        assert_decode_rejects(&bytes, "non-success descriptor has table transitions");

        let mut invalid = non_success_descriptor();
        invalid.final_database_root = digest(22);
        assert_encode_rejects(
            &success_marker(invalid),
            "zero-transition descriptor changes database data root",
        );
        let mut bytes = encode_marker(&success_marker(non_success_descriptor()));
        bytes[INITIAL_DATABASE_ROOT..INITIAL_DATABASE_ROOT + 32].copy_from_slice(&digest(22));
        assert_decode_rejects(
            &bytes,
            "zero-transition descriptor changes database data root",
        );

        let mut invalid = descriptor();
        invalid.catalog_before_digest = [0; 32];
        assert_encode_rejects(
            &success_marker(invalid),
            "catalog before digest must not be zero",
        );
        let mut bytes = encode_marker(&success_marker(descriptor()));
        bytes[CATALOG_BEFORE_DIGEST..CATALOG_BEFORE_DIGEST + 32].fill(0);
        assert_decode_rejects(&bytes, "catalog before digest must not be zero");

        let mut invalid = descriptor();
        invalid.catalog_after_digest = [0; 32];
        assert_encode_rejects(
            &success_marker(invalid),
            "catalog after digest must not be zero",
        );
        let mut bytes = encode_marker(&success_marker(descriptor()));
        bytes[CATALOG_AFTER_DIGEST..CATALOG_AFTER_DIGEST + 32].fill(0);
        assert_decode_rejects(&bytes, "catalog after digest must not be zero");
    }

    #[test]
    fn v1_transition_kind_locals_reject_noncanonical_rows_and_rowset_index_shape_vectors() {
        const DESCRIPTOR: usize =
            CANONICAL_OUTCOME_BYTES + GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES;
        const TRANSITION: usize = DESCRIPTOR + GENERATION_TERMINAL_DESCRIPTOR_FIXED_BYTES;
        const BEFORE_ROWS: usize = TRANSITION + 56;
        const BEFORE_TABLE_ROOT: usize = TRANSITION + 24;
        const AFTER_TABLE_GENERATION: usize = TRANSITION + 64;
        const AFTER_TABLE_ROOT: usize = TRANSITION + 72;
        const AFTER_ROWS: usize = TRANSITION + 104;
        const AFTER_INDEX_ID: usize = TRANSITION + 200;
        const AFTER_INDEX_SHAPE: usize = TRANSITION + 208;

        let mut valid = descriptor();
        valid.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::CreateEmpty,
            stable_table_id: 4,
            before: None,
            after: Some(side(9, 25, 0, vec![])),
        }];
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("CreateEmpty final table")
            .logical_row_count = 1;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| bytes[AFTER_ROWS..AFTER_ROWS + 8].copy_from_slice(&1_u64.to_le_bytes()),
            "CreateEmpty final row count",
        );

        let mut valid = descriptor();
        valid.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::ResetEmpty,
            stable_table_id: 4,
            before: Some(side(6, 23, 1, vec![])),
            after: Some(side(9, 25, 0, vec![])),
        }];
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .before
            .as_mut()
            .expect("ResetEmpty prior table")
            .logical_row_count = 0;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| bytes[BEFORE_ROWS..BEFORE_ROWS + 8].fill(0),
            "ResetEmpty row counts",
        );
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("ResetEmpty final table")
            .logical_row_count = 1;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| bytes[AFTER_ROWS..AFTER_ROWS + 8].copy_from_slice(&1_u64.to_le_bytes()),
            "ResetEmpty row counts",
        );

        let mut valid = descriptor();
        valid.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::Rebuild,
            stable_table_id: 4,
            before: None,
            after: Some(side(9, 25, 1, vec![])),
        }];
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("Rebuild final table")
            .logical_row_count = 0;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| bytes[AFTER_ROWS..AFTER_ROWS + 8].fill(0),
            "Rebuild final row count",
        );

        let valid = descriptor();
        let mut invalid = valid.clone();
        let after = invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("RowSet final table");
        after.data_generation = 6;
        after.table_root = digest(23);
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| {
                bytes[AFTER_TABLE_GENERATION..AFTER_TABLE_GENERATION + 8]
                    .copy_from_slice(&6_u64.to_le_bytes());
                bytes[AFTER_TABLE_ROOT..AFTER_TABLE_ROOT + 32].copy_from_slice(&digest(23));
            },
            "RowSet final table generation must equal commit sequence",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("RowSet final table")
            .table_root = digest(23);
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| {
                bytes[BEFORE_TABLE_ROOT..BEFORE_TABLE_ROOT + 32].copy_from_slice(&digest(23));
                bytes[AFTER_TABLE_ROOT..AFTER_TABLE_ROOT + 32].copy_from_slice(&digest(23));
            },
            "RowSet table root",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("RowSet final table")
            .indexes[0]
            .stable_index_id = 4;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| bytes[AFTER_INDEX_ID..AFTER_INDEX_ID + 8].copy_from_slice(&4_u64.to_le_bytes()),
            "RowSet index ID/shape vector",
        );
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("RowSet final table")
            .indexes[0]
            .index_shape_root = digest(27);
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| bytes[AFTER_INDEX_SHAPE..AFTER_INDEX_SHAPE + 32].copy_from_slice(&digest(27)),
            "RowSet index ID/shape vector",
        );
    }

    #[test]
    fn v1_generation_evolution_locals_reject_noncanonical_table_and_index_generations() {
        const DESCRIPTOR: usize =
            CANONICAL_OUTCOME_BYTES + GENERATION_TERMINAL_EXTENSION_PREFIX_BYTES;
        const TRANSITION: usize = DESCRIPTOR + GENERATION_TERMINAL_DESCRIPTOR_FIXED_BYTES;
        const BEFORE_TABLE_GENERATION: usize = TRANSITION + 16;
        const AFTER_TABLE_GENERATION: usize = TRANSITION + 64;
        const AFTER_TABLE_ROWS: usize = TRANSITION + 104;
        const BEFORE_INDEX_GENERATION: usize = TRANSITION + 156;
        const AFTER_INDEX_SHAPE: usize = TRANSITION + 208;
        const AFTER_INDEX_GENERATION: usize = TRANSITION + 240;
        const AFTER_INDEX_ROOT: usize = TRANSITION + 248;

        let mut valid = descriptor();
        valid.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::CreateEmpty,
            stable_table_id: 4,
            before: None,
            after: Some(side(9, 25, 0, vec![])),
        }];
        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("new final table")
            .data_generation = 8;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| {
                bytes[AFTER_TABLE_GENERATION..AFTER_TABLE_GENERATION + 8]
                    .copy_from_slice(&8_u64.to_le_bytes())
            },
            "new table generation must equal commit sequence",
        );

        let mut retained = descriptor();
        retained.table_transitions[0].kind = GenerationTableDeltaKindV1::Rebuild;
        let retained_after = retained.table_transitions[0]
            .after
            .as_mut()
            .expect("retained final table");
        retained_after.data_generation = 6;
        retained_after.table_root = digest(23);
        retained_after.logical_row_count = 1;
        retained_after.indexes[0].index_generation = 6;
        retained_after.indexes[0].index_root = digest(25);
        assert!(
            measure_canonical_terminal_marker(&success_marker(retained.clone())).is_ok(),
            "a physical Rebuild may retain all table and index manifest facts and generation"
        );

        let mut invalid = retained.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("matched final table")
            .data_generation = 9;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            retained.clone(),
            |bytes| {
                bytes[AFTER_TABLE_GENERATION..AFTER_TABLE_GENERATION + 8]
                    .copy_from_slice(&9_u64.to_le_bytes())
            },
            "unchanged table generation",
        );

        let mut invalid = retained.clone();
        let after = invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("index-mutated final table");
        after.indexes[0].index_generation = 9;
        after.indexes[0].index_root = digest(26);
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            retained.clone(),
            |bytes| {
                bytes[AFTER_INDEX_GENERATION..AFTER_INDEX_GENERATION + 8]
                    .copy_from_slice(&9_u64.to_le_bytes());
                bytes[AFTER_INDEX_ROOT..AFTER_INDEX_ROOT + 32].copy_from_slice(&digest(26));
            },
            "changed table generation must equal commit sequence",
        );

        let mut invalid = retained.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("row-mutated final table")
            .logical_row_count = 2;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            retained,
            |bytes| {
                bytes[AFTER_TABLE_ROWS..AFTER_TABLE_ROWS + 8].copy_from_slice(&2_u64.to_le_bytes())
            },
            "changed table generation must equal commit sequence",
        );

        let valid = descriptor();

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("changed final table")
            .data_generation = 8;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| {
                bytes[AFTER_TABLE_GENERATION..AFTER_TABLE_GENERATION + 8]
                    .copy_from_slice(&8_u64.to_le_bytes())
            },
            "RowSet final table generation must equal commit sequence",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .before
            .as_mut()
            .expect("prior table")
            .data_generation = 9;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| {
                bytes[BEFORE_TABLE_GENERATION..BEFORE_TABLE_GENERATION + 8]
                    .copy_from_slice(&9_u64.to_le_bytes())
            },
            "prior table generation must precede commit sequence",
        );

        let mut valid_new_index = descriptor();
        valid_new_index.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::Rebuild,
            stable_table_id: 4,
            before: Some(side(6, 23, 1, vec![])),
            after: Some(side(9, 25, 2, vec![index(3, 9, 24, 26)])),
        }];
        let mut invalid = valid_new_index.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("new-index final table")
            .indexes[0]
            .index_generation = 8;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid_new_index,
            |bytes| {
                let after_new_index_generation = TRANSITION + 160;
                bytes[after_new_index_generation..after_new_index_generation + 8]
                    .copy_from_slice(&8_u64.to_le_bytes())
            },
            "new index generation must equal commit sequence",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("matched final index")
            .indexes[0]
            .index_root = digest(25);
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| bytes[AFTER_INDEX_ROOT..AFTER_INDEX_ROOT + 32].copy_from_slice(&digest(25)),
            "unchanged index generation",
        );

        let mut valid_rebuild_shape = descriptor();
        valid_rebuild_shape.table_transitions = vec![GenerationTableTransitionV1 {
            kind: GenerationTableDeltaKindV1::Rebuild,
            stable_table_id: 4,
            before: Some(side(6, 23, 1, vec![index(3, 6, 24, 25)])),
            after: Some(side(9, 25, 1, vec![index(3, 9, 27, 25)])),
        }];
        let mut invalid = valid_rebuild_shape.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("shape-changed final index")
            .indexes[0]
            .index_generation = 6;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid_rebuild_shape,
            |bytes| {
                bytes[AFTER_INDEX_SHAPE..AFTER_INDEX_SHAPE + 32].copy_from_slice(&digest(27));
                bytes[AFTER_INDEX_GENERATION..AFTER_INDEX_GENERATION + 8]
                    .copy_from_slice(&6_u64.to_le_bytes());
            },
            "changed index generation must equal commit sequence",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .after
            .as_mut()
            .expect("changed final index")
            .indexes[0]
            .index_generation = 8;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid.clone(),
            |bytes| {
                bytes[AFTER_INDEX_GENERATION..AFTER_INDEX_GENERATION + 8]
                    .copy_from_slice(&8_u64.to_le_bytes())
            },
            "changed index generation must equal commit sequence",
        );

        let mut invalid = valid.clone();
        invalid.table_transitions[0]
            .before
            .as_mut()
            .expect("prior index")
            .indexes[0]
            .index_generation = 9;
        assert_descriptor_rejected_on_encode_and_decode(
            invalid,
            valid,
            |bytes| {
                bytes[BEFORE_INDEX_GENERATION..BEFORE_INDEX_GENERATION + 8]
                    .copy_from_slice(&9_u64.to_le_bytes())
            },
            "prior index generation must precede commit sequence",
        );
    }

    #[test]
    fn terminal_descriptor_must_repeat_the_immutable_header_identity() {
        let marker = CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(descriptor()),
        };
        assert!(canonical_terminal_marker_digest(&header(), digest(28), &marker).is_ok());
        let mut mismatched = descriptor();
        mismatched.commit_sequence += 1;
        let mismatched = CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(mismatched),
        };
        assert!(canonical_terminal_marker_digest(&header(), digest(28), &mismatched).is_err());
    }

    #[test]
    fn generic_envelope_round_trips_only_the_explicit_extended_marker_form() {
        let marker = CanonicalTerminalMarker::GenerationV1 {
            outcome: outcome(),
            descriptor: Box::new(descriptor()),
        };
        let fragments = [CanonicalFragment {
            kind: CanonicalFragmentKind::RowMutation,
            body: b"root-free mutation".to_vec(),
        }];
        let marker_measure = measure_canonical_terminal_marker(&marker).unwrap();
        let footprint = canonical_wal_footprint_by_index_with_marker(
            1,
            |_| Ok(u64::try_from(fragments[0].body.len()).unwrap()),
            marker_measure.marker_bytes,
        )
        .expect("extended marker footprint");
        let encoded = encode_canonical_envelope_with_terminal_marker(
            physical(),
            &header(),
            &fragments,
            &marker,
        )
        .expect("extended envelope encodes");
        let packed = pack_canonical_record_payload(&encoded).expect("extended record packs");
        assert_eq!(packed.len() as u64, footprint.packed_record_bytes);
        assert!(
            decode_canonical_record_payload(&packed).is_err(),
            "the legacy recovery decoder must not admit an unowned root-format-v1 marker"
        );
        let decoded = decode_canonical_envelope_with_terminal_marker(&encoded.frames)
            .expect("private extended record decodes");
        assert_eq!(decoded.terminal_marker, marker);
        assert_eq!(decoded.outcome, outcome());
        assert_eq!(decoded.final_digest, encoded.final_digest);
        assert_ne!(
            decoded.final_digest,
            canonical_terminal_marker_digest(
                &header(),
                decoded.ordered_fragment_root,
                &CanonicalTerminalMarker::Legacy(outcome()),
            )
            .unwrap(),
        );
    }
}
