//! General transaction writer for codec-5 semantics v2.
//!
//! It closes typed statements, catalog dependencies, and durable sequence receipts. Row traversal
//! is type-neutral, and generation roots are accepted only as authenticated CUDA commitments.

use super::super::{
    encode_reserved_typed_insert_canonical_envelope,
    measure_and_prepare_typed_insert_aggregate_encoding, reserve_typed_insert_aggregate_bodies,
    reserve_typed_insert_canonical_envelope, EncodedTypedInsertCanonicalEnvelope,
    TypedInsertAggregateSectionView, TypedInsertAggregateSemantics, TypedInsertAggregateView,
    TypedInsertStatusV2, AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_CATALOG,
    AGGREGATE_FLAG_EXPLICIT, AGGREGATE_FLAG_OPERATION_COMPOSITION, AGGREGATE_FLAG_PRIVATE_SEQUENCE,
    AGGREGATE_FLAG_PUBLISHED_SEQUENCE, AGGREGATE_FLAG_RETURNING, AGGREGATE_SECTION_COUNT,
    OUTER_CONTENT_CATALOG, OUTER_CONTENT_OPERATION_COMPOSITION, OUTER_CONTENT_PRIVATE_SEQUENCE,
    OUTER_CONTENT_PUBLISHED_SEQUENCE, OUTER_CONTENT_RETURNING, OUTER_CONTENT_ROW,
    OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
};
#[cfg(test)]
use crate::typed_insert_batch::PreparedResidentRuntimeGenerationView;
use crate::typed_insert_batch::{
    decode_canonical_typed_insert_record, parse_canonical_typed_insert_record_prefix,
    DecodedTypedInsertRecord, DecodedTypedValueFacts, CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES,
};
use crate::EngineError;
use sha2::{Digest, Sha256};

#[path = "writer/domain_closure.rs"]
mod domain_closure;
#[path = "writer/foreign_key_closure.rs"]
mod foreign_key_closure;
#[path = "writer/format.rs"]
mod format;

use format::*;
pub(crate) use format::{
    live_autocommit_request_digest, live_explicit_request_digest,
    live_explicit_request_digest_with_catalog, write001_identifier_digest,
};

const ABSENT_U32: u32 = u32::MAX;
const S1_BYTES: usize = 144;
const S4_BYTES: usize = 64;
const S6_BYTES: usize = 136;
const S7_HEADER_BYTES: usize = 640;
const S7_FIXED_WIDTHS: [usize; 12] = [384, 32, 320, 224, 32, 384, 112, 192, 192, 128, 128, 160];
const S7_COUNT_TO_DIRECTORY: [usize; 12] = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const S7_MAGIC: &[u8; 16] = b"GPUDBS7OVERLAY2\0";
const S7_TABLE_FLAG_RESETS_EXISTING_ROWS: u32 = 1;
const S7_TABLE_FLAG_INITIAL_TABLE_ABSENT: u32 = 1 << 1;

#[derive(Clone, Copy)]
pub(crate) enum LiveTypedInsertMode {
    Autocommit,
    Explicit,
}

impl LiveTypedInsertMode {
    const fn aggregate_flag(self) -> u32 {
        match self {
            Self::Autocommit => AGGREGATE_FLAG_AUTOCOMMIT,
            Self::Explicit => AGGREGATE_FLAG_EXPLICIT,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertIdentity {
    pub(crate) physical: gpu_db_wal::CanonicalPhysicalRange,
    pub(crate) canonical: gpu_db_wal::CanonicalIdentity,
    pub(crate) leader_epoch: u64,
    pub(crate) commit_sequence: u64,
    pub(crate) stable_transaction_id: u64,
    pub(crate) mode: LiveTypedInsertMode,
    /// The generic transaction owner binds retry/status identity before this ephemeral codec
    /// preparation leaf runs. The codec must carry that exact identity rather than minting a
    /// statement-specific request authority.
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) isolation: gpu_db_wal::CanonicalIsolation,
    pub(crate) catalog_epoch: u64,
    pub(crate) catalog_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) catalog_after_epoch: u64,
    pub(crate) catalog_after_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) dependency_validation_floor: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertTableGeneration<'a> {
    pub(crate) schema: &'a str,
    pub(crate) name: &'a str,
    pub(crate) stable_table_id: u64,
    pub(crate) display_oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
    /// The one S7 image-content digest computed from this exact immutable final image before
    /// CUDA generation. Both the image directory and table closure consume that same fact.
    pub(crate) image_content_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) data_generation_before: u64,
    pub(crate) data_generation_after: u64,
    pub(crate) row_allocator_before: u64,
    pub(crate) row_allocator_high_water: u64,
    pub(crate) initial_logical_row_count: u64,
    pub(crate) final_logical_row_count: u64,
    pub(crate) initial_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) initial_database_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_database_root: gpu_db_wal::CanonicalDigest,
    pub(crate) resets_existing_rows: bool,
    /// S7 records the real absent predecessor for transactional CREATE TABLE followed by its
    /// first INSERT. The GPU derives the first root; this flag is only the durable closure that
    /// prevents replay from relabelling that CREATE as a normal append or reset.
    pub(crate) initial_table_absent: bool,
    /// The catalog-owned named index identities and exact GPU predecessor/successor roots for
    /// this same table generation.  This is metadata for S7 closure only; it cannot publish an
    /// index or select another device plan.
    pub(crate) indexes: &'a [LiveTypedInsertIndexGeneration<'a>],
    /// S3-proven absent-to-present indexes for an existing table.  An S2 statement may omit
    /// exactly these indexes only when it precedes their ordered catalog operation.
    pub(crate) created_indexes_on_existing_table: &'a [LiveTypedInsertCreatedIndex],
}

/// One catalog index represented by the canonical one-table S7 closure and device generation.
#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertIndexGeneration<'a> {
    pub(crate) catalog: &'a crate::RelationalIndex,
    pub(crate) base_generation: u64,
    pub(crate) base_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_generation: u64,
    pub(crate) final_root: gpu_db_wal::CanonicalDigest,
}

/// One exact S3-proven CREATE INDEX transition over a table that already had a published typed
/// generation.  This is live assembly evidence only: S3 remains the catalog carrier, and the
/// zero base root plus the S7 descriptor remain the durable/replay closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LiveTypedInsertCreatedIndex {
    pub(crate) stable_index_id: u64,
    pub(crate) operation_ordinal: u32,
}

/// Immutable parent roots used only to close an already-completed device FK verdict in S7.
#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertForeignIndexGeneration<'a> {
    pub(crate) parent: &'a crate::RelationalTable,
    pub(crate) parent_schema_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) parent_generation: u64,
    pub(crate) parent_root: gpu_db_wal::CanonicalDigest,
    pub(crate) catalog: &'a crate::RelationalIndex,
    pub(crate) index_generation: u64,
    pub(crate) index_root: gpu_db_wal::CanonicalDigest,
}

pub(crate) struct LiveTypedInsertView<'a> {
    pub(crate) identity: LiveTypedInsertIdentity,
    pub(crate) table: LiveTypedInsertTableGeneration<'a>,
    pub(crate) statements: &'a [LiveTypedInsertStatementView<'a>],
    pub(crate) final_image: &'a [u8],
    /// Deduplicated immutable parent indexes used by the sealed S2 foreign-key bindings. They
    /// authenticate the existing GPU verdict without becoming target-owned index generations.
    pub(crate) foreign_indexes: &'a [LiveTypedInsertForeignIndexGeneration<'a>],
    /// Already-durable sequence transitions serialized as S5/S7 closure facts.
    pub(crate) published_sequence_references: &'a [crate::BinarySequenceValueReference],
    /// Exact S7 final-row closure emitted by the already-completed generic CUDA generation.
    /// This prevents the live writer from rehashing every typed cell on the CPU; recovery still
    /// independently recomputes the same digest through the retained decoder.
    pub(crate) final_row_digests: Option<&'a dyn LiveFinalRowDigestSource>,
    /// Table-row-ordered final-writer bindings. Ordinary INSERT rows keep the source statement
    /// ordinal and a zero digest, preserving the frozen codec bytes. A transaction-private
    /// rewrite carries its later operation ordinal and exact statement digest in the already
    /// reserved transition fields.
    pub(crate) final_writers: &'a [LiveTypedInsertFinalWriter],
    /// Scalar geometry from the already-completed generic CUDA generation. It is retained as
    /// S7's source-shape proof without rebuilding a second host view over the same sealed cells.
    pub(crate) source_geometry: Option<gpu_db_execution::RuntimeTypedInsertGenerationGeometry>,
}

#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertFinalWriter {
    pub(crate) stable_row_id: u64,
    pub(crate) source_statement_ordinal: u32,
    pub(crate) source_row_ordinal: u32,
    pub(crate) final_writer_statement_ordinal: u32,
    pub(crate) final_writer_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) survives: bool,
}

/// One independently parsed INSERT statement inside the shared transaction-final aggregate.
/// Its S2 bytes and identity remain distinct even though all rows are composed into one final
/// table image, CUDA generation, device plan, WAL record, and publication lifecycle.
#[derive(Clone, Copy)]
pub(crate) struct LiveTypedInsertStatementView<'a> {
    pub(crate) statement_ordinal: u32,
    /// Position in the complete ordered transaction program. Typed statement ordinals remain
    /// dense while catalog lifecycle operations retain their own intervening positions.
    pub(crate) operation_ordinal: u32,
    /// Stable-table-ordered S7 table reference. One-table callers use zero; a database-scoped
    /// transaction binds every statement to exactly one entry in the shared table directory.
    pub(crate) table_ref: u32,
    /// The statement's own catalog cut. A later private sequence rename may update the final
    /// table default and therefore S7's final-table digest, but must not relabel this sealed S2
    /// record as having been prepared against that later schema.
    pub(crate) table_schema_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) record: &'a [u8],
    /// Production binds this to the exact immutable owner of `record`. Test/fixture callers
    /// retain `None`, which deliberately selects the strict decoder. This is provenance proof,
    /// not a second S2 carrier or a cache of decoded semantics.
    pub(crate) sealed_source: Option<&'a crate::typed_insert_batch::SealedTypedInsertCodec5Sources>,
}

/// One later `ALTER SEQUENCE ... RESTART` whose final state overwrites the last S2-owned
/// sequence effect in this same transaction. It is serialized only as a suffix of that existing
/// S5 effect; it is not another sequence entry or publication authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LiveTypedInsertSequenceRestart {
    pub(crate) owner_operation_ordinal: u32,
    pub(crate) owner_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) sequence_oid: u32,
    pub(crate) sequence_name: String,
    pub(crate) last_value: i64,
}

/// Canonical semantics-v2 WAL authority emitted by the constrained live writer.
///
/// This owner is closed by construction: the writer derives every S1--S8 field from the sealed
/// typed source and authenticated generation commitments, then the shared codec and outer-WAL
/// encoders prove exact geometry, roots, status, and outcome binding.  Fresh recovery deliberately
/// does not trust this marker; it independently decodes and closes the durable bytes through the
/// retained semantics-v2 authority.
pub(crate) struct ClosedLiveTypedInsert {
    envelope: EncodedTypedInsertCanonicalEnvelope,
    #[cfg(feature = "probe-timing")]
    probe_timing_nanos: [u64; 11],
}

impl ClosedLiveTypedInsert {
    pub(crate) fn header(&self) -> &gpu_db_wal::CanonicalPreApplyHeader {
        self.envelope.header()
    }

    pub(crate) fn packed_payload(&self) -> &[u8] {
        self.envelope.packed_payload()
    }

    pub(crate) fn packed_payload_authority(&self) -> std::sync::Arc<[u8]> {
        std::sync::Arc::clone(self.envelope.prepared_payload_authority())
    }

    pub(crate) fn into_prepared_record(self) -> gpu_db_wal::PreparedCanonicalWalRecord {
        self.envelope.into_prepared_record()
    }

    #[cfg(feature = "probe-timing")]
    pub(crate) fn probe_timing_nanos(&self) -> [u64; 11] {
        self.probe_timing_nanos
    }
}

/// Narrow handoff for device-completed S7 row identities; it exposes no generic digest iterator.
pub(crate) trait LiveFinalRowDigestSource {
    /// Copy the device-produced final-row commitment for a serialized transition.
    fn copy_write001_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError>;

    /// Copy the indexed S7 final-row commitment; its transition closes over emitted effects.
    fn copy_write001_s7_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError>;

    /// Copy the exact unindexed device row and transition commitments.
    fn copy_write001_transition_digests_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
        transition_destination: &mut [u8; 32],
    ) -> Result<(), EngineError>;
}

impl LiveFinalRowDigestSource for gpu_db_execution::RuntimeTypedInsertGenerationLogicalCompletion {
    fn copy_write001_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        self.copy_write001_s7_final_row_digest_for_serialized_transition_into(
            row_ordinal,
            stable_row_id,
            final_row_destination,
        )
        .map_err(|_| error("CUDA S7 final-row digest identity is absent"))
    }

    fn copy_write001_s7_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        self.copy_write001_s7_final_row_digest_into(
            row_ordinal,
            stable_row_id,
            typed_statement_digest,
            final_row_destination,
        )
        .map_err(|_| error("CUDA S7 final-row digest identity is absent"))
    }

    fn copy_write001_transition_digests_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
        transition_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        self.copy_write001_s7_transition_digests_into(
            row_ordinal,
            stable_row_id,
            typed_statement_digest,
            final_row_destination,
            transition_destination,
        )
        .map_err(|_| error("CUDA S7 transition digest identity is absent"))
    }
}

/// Canonical S7 control-plane serialization for the bounded named-index vertical.
///
/// The device remains the sole producer of final-row, table, database, and named-index successor
/// roots.  S7 nevertheless has to carry every durable index descriptor, typed key component, and
/// dependency record.  Schema/key identity comes from the sealed S2 record; key values come from
/// the already-bound final table image so a later UPDATE of a provisional row cannot resurrect the
/// original INSERT image.  This is WAL grammar closure, not a CPU index, lookup, apply, or
/// successor-root authority.
struct IndexedS7Closure {
    dependencies: Vec<[u8; 224]>,
    dependency_uses: Vec<[u8; 32]>,
    indexes: Vec<[u8; 384]>,
    index_digests: Vec<[u8; 32]>,
    index_keys: Vec<[u8; 112]>,
    transitions: Vec<[u8; 192]>,
    effects: Vec<[u8; 192]>,
    effect_digests: Vec<[u8; 32]>,
    components: Vec<[u8; 128]>,
    values: Vec<u8>,
    owned_index_start: u32,
    owned_index_count: u32,
}

#[derive(Clone, Copy, Default)]
struct IndexedS7TableRange {
    owned_index_start: u32,
    owned_index_count: u32,
    effect_start: u32,
    effect_count: u32,
}

#[derive(Clone, Copy)]
struct IndexedS7Bases {
    table_ref: u32,
    index_ref: u32,
    key_start: u32,
    effect_start: u32,
    component_start: u32,
    value_offset: u64,
    dependency_start: u32,
}

struct PublishedSequenceClosure {
    s5: Vec<u8>,
    statement_ranges: Vec<std::ops::Range<usize>>,
    published_count: u32,
    private_count: u32,
    dependencies: Vec<[u8; 224]>,
    dependency_uses: Vec<[u8; 32]>,
}

struct ReturningClosure {
    projections: Vec<[u8; 128]>,
    logical_result_digest: [u8; 32],
}

#[derive(Clone, Copy)]
struct SequenceEntry {
    statement_ordinal: u32,
    source_operation_ordinal: u32,
    effect_ordinal: u32,
    disposition_ref: u32,
    table_ref: u32,
    transition_ref: u32,
    sequence_oid: u32,
    terminal_restart: Option<super::sequence_terminal::SequenceRestartTail>,
    kind: SequenceEntryKind,
}

#[derive(Clone, Copy)]
enum SequenceEntryKind {
    Published {
        stable_sequence_id: u64,
        sequence_oid: u32,
        transition_txn_id: u64,
        name_digest: [u8; 32],
        reference_body_digest: [u8; 32],
        body: [u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES],
        final_value_overwritten: bool,
    },
    Private {
        final_value_overwritten: bool,
    },
}

#[derive(Clone)]
struct IndexedS7Key {
    raw: [u8; 112],
    digest: [u8; 32],
    catalog_column_ordinal: u32,
    stable_column_id: u32,
    attnum: i16,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
}

/// An initial table's owned descriptor is already in the one S7 directory when a later table in
/// the same transaction needs it as an FK parent. Reusing this exact descriptor keeps one index
/// identity across the creation and FK proofs; it is writer-local assembly state, not a second
/// durable carrier or recovery authority.
#[derive(Clone)]
struct SharedInitialForeignIndexDescriptor {
    owner_stable_table_id: u64,
    owner_display_oid: u32,
    stable_index_id: u64,
    display_oid: u32,
    raw: [u8; 384],
    digest: [u8; 32],
    keys: Vec<IndexedS7Key>,
}

fn shared_initial_foreign_descriptors(
    input: &LiveTypedInsertView<'_>,
    closure: &IndexedS7Closure,
) -> Result<Vec<SharedInitialForeignIndexDescriptor>, EngineError> {
    if !input.table.initial_table_absent {
        return Ok(Vec::new());
    }
    if closure.indexes.len() != closure.index_digests.len() {
        return Err(error("S7 shared initial-index inventory is incomplete"));
    }
    let table_ref = closure
        .indexes
        .iter()
        .find_map(|index| {
            (u64::from_le_bytes(index[64..72].try_into().expect("fixed index owner id"))
                == input.table.stable_table_id)
                .then(|| {
                    u32::from_le_bytes(index[8..12].try_into().expect("fixed index owner ref"))
                })
        })
        .unwrap_or(ABSENT_U32);
    let mut shared = Vec::new();
    for (index, digest) in closure.indexes.iter().zip(&closure.index_digests) {
        let owner_ref = u32::from_le_bytes(index[8..12].try_into().expect("fixed index owner ref"));
        let owner_stable_table_id =
            u64::from_le_bytes(index[64..72].try_into().expect("fixed index owner id"));
        let owner_display_oid =
            u32::from_le_bytes(index[336..340].try_into().expect("fixed index owner oid"));
        if owner_ref == ABSENT_U32
            || owner_ref != table_ref
            || owner_stable_table_id != input.table.stable_table_id
            || owner_display_oid != input.table.display_oid
        {
            continue;
        }
        let index_ref = u32::from_le_bytes(index[..4].try_into().expect("fixed index ref"));
        let keys = closure
            .index_keys
            .iter()
            .filter(|key| {
                u32::from_le_bytes(key[4..8].try_into().expect("fixed key index ref")) == index_ref
            })
            .map(|raw| IndexedS7Key {
                raw: *raw,
                digest: raw[72..104].try_into().expect("fixed index key digest"),
                catalog_column_ordinal: u32::from_le_bytes(
                    raw[12..16].try_into().expect("fixed index key column"),
                ),
                stable_column_id: u32::from_le_bytes(
                    raw[16..20].try_into().expect("fixed index key column id"),
                ),
                attnum: i16::from_le_bytes(raw[24..26].try_into().expect("fixed index key attnum")),
                storage: raw[28..32].try_into().expect("fixed index key storage"),
                declared_type_oid: u32::from_le_bytes(
                    raw[32..36].try_into().expect("fixed index key type oid"),
                ),
                signed_type_size: i16::from_le_bytes(
                    raw[36..38].try_into().expect("fixed index key type size"),
                ),
            })
            .collect::<Vec<_>>();
        let expected_key_count =
            u32::from_le_bytes(index[44..48].try_into().expect("fixed index key count"));
        if usize::try_from(expected_key_count).ok() != Some(keys.len()) {
            return Err(error("S7 shared initial-index key range is incomplete"));
        }
        shared.push(SharedInitialForeignIndexDescriptor {
            owner_stable_table_id,
            owner_display_oid,
            stable_index_id: u64::from_le_bytes(
                index[16..24].try_into().expect("fixed stable index id"),
            ),
            display_oid: u32::from_le_bytes(index[24..28].try_into().expect("fixed index oid")),
            raw: *index,
            digest: *digest,
            keys,
        });
    }
    Ok(shared)
}

impl IndexedS7Closure {
    fn from_sealed_records(
        input: &LiveTypedInsertView<'_>,
        records: &[DecodedTypedInsertRecord],
        rows: &[BoundRow],
        final_row_digests: &[[u8; 32]],
        bases: IndexedS7Bases,
    ) -> Result<Self, EngineError> {
        if input.table.indexes.is_empty() || final_row_digests.len() != rows.len() {
            return Err(error("indexed S7 closure has no exact owned-index/row set"));
        }
        let final_image = crate::typed_insert_batch::decode_typed_image(input.final_image)?;
        if final_image.facts().role != crate::typed_insert_batch::TypedImageRole::FinalTableImage
            || (!rows.is_empty()
                && usize::try_from(final_image.facts().rows).ok() != Some(rows.len()))
        {
            return Err(error(
                "indexed S7 closure final image differs from its surviving row set",
            ));
        }
        let terminal_statement = input
            .statements
            .last()
            .ok_or_else(|| error("indexed S7 closure has no staged statement"))?;
        let record = records
            .get(terminal_statement.statement_ordinal as usize)
            .ok_or_else(|| error("indexed S7 closure has no sealed statement record"))?;
        let created_indexes_are_exact = input
            .table
            .created_indexes_on_existing_table
            .iter()
            .enumerate()
            .all(|(ordinal, created)| {
                created.stable_index_id != 0
                    && created.stable_index_id != u64::MAX
                    && (ordinal == 0
                        || input.table.created_indexes_on_existing_table[ordinal - 1]
                            .stable_index_id
                            < created.stable_index_id)
                    && input.table.indexes.iter().any(|generation| {
                        u64::from(generation.catalog.oid) == created.stable_index_id
                            && generation.base_generation == 0
                            && generation.base_root == [0; 32]
                    })
            });
        if !created_indexes_are_exact {
            return Err(error(
                "indexed S7 closure has an invalid S3-created index proof",
            ));
        }
        for statement in input.statements {
            let record = records
                .get(statement.statement_ordinal as usize)
                .ok_or_else(|| error("indexed S7 statement record is absent"))?;
            let sources = record.indexes().collect::<Vec<_>>();
            let target = record.target_identity();
            let expected_indexes = input
                .table
                .indexes
                .iter()
                .filter(|generation| {
                    input
                        .table
                        .created_indexes_on_existing_table
                        .iter()
                        .find(|created| {
                            created.stable_index_id == u64::from(generation.catalog.oid)
                        })
                        .is_none_or(|created| {
                            created.operation_ordinal < statement.operation_ordinal
                        })
                })
                .collect::<Vec<_>>();
            if sources.len() != expected_indexes.len()
                || record.facts().statement_ordinal.as_u32() != statement.statement_ordinal
                || record.facts().typed_statement_digest != statement.typed_statement_digest
                || target.schema != input.table.schema
                || target.name != input.table.name
                || target.oid != input.table.display_oid
                || target.schema_digest != statement.table_schema_digest
            {
                return Err(error(&format!(
                    "sealed S2 statement target/index cardinality differs from the live catalog \
                     (table={}, statement={}, s2_indexes={}, live_indexes={}, \
                     s2_ordinal={}, target={}.{}#{})",
                    input.table.name,
                    statement.statement_ordinal,
                    sources.len(),
                    expected_indexes.len(),
                    record.facts().statement_ordinal.as_u32(),
                    target.schema,
                    target.name,
                    target.oid,
                )));
            }
            for (raw_ordinal, (source, generation)) in
                sources.iter().zip(expected_indexes).enumerate()
            {
                let catalog = generation.catalog;
                if usize::try_from(source.raw_ordinal).ok() != Some(raw_ordinal)
                    || source.oid != catalog.oid
                    || source.name != catalog.name
                    || source.table_name != input.table.name
                    || source.unique != catalog.unique
                    || source.primary_key != catalog.primary_key
                    || source.unique_constraint != catalog.unique_constraint
                    || usize::try_from(source.key_count).ok() != Some(catalog.key_columns.len())
                {
                    return Err(error("sealed S2 named index differs from the live catalog"));
                }
            }
        }
        struct OwnedIndex {
            raw_ordinal: u32,
            index_ref: u32,
            static_dependency_ref: u32,
            unique: bool,
            keys: Vec<IndexedS7Key>,
            descriptor: [u8; 384],
            descriptor_digest: [u8; 32],
        }

        let index_count = u32::try_from(input.table.indexes.len())
            .map_err(|_| error("S7 owned-index count exceeds u32"))?;
        let owner_name_digest = qualified_name_digest(input.table.schema, input.table.name)?;
        let sources = record.indexes().collect::<Vec<_>>();
        if sources.len() != input.table.indexes.len() {
            return Err(error(
                "terminal indexed S2 statement does not carry the final S3-composed directory",
            ));
        }
        let mut owned = Vec::with_capacity(input.table.indexes.len());
        let mut flat_keys = Vec::new();
        let mut prior_stable_index_id = None;
        for (raw_ordinal, (source, generation)) in
            sources.iter().zip(input.table.indexes).enumerate()
        {
            let catalog = generation.catalog;
            let absent_predecessor = (input.table.resets_existing_rows
                || input.table.initial_table_absent
                || input
                    .table
                    .created_indexes_on_existing_table
                    .binary_search_by_key(&u64::from(catalog.oid), |created| {
                        created.stable_index_id
                    })
                    .is_ok())
                && generation.base_generation == 0
                && generation.base_root == [0; 32];
            if catalog.oid == 0
                || prior_stable_index_id.is_some_and(|prior| u64::from(catalog.oid) <= prior)
                || (catalog.primary_key && !catalog.unique)
                || (catalog.unique_constraint && !catalog.unique)
                || catalog.key_columns.is_empty()
                || catalog.key_columns.len() > 32
                || ((generation.base_generation == 0) != (generation.base_root == [0; 32]))
                || (!absent_predecessor && generation.base_generation == 0)
                || generation.final_generation != input.table.data_generation_after
                || generation.final_root == [0; 32]
                || generation.final_root == generation.base_root
            {
                return Err(error(
                    "indexed S7 closure has an unsupported named-index predecessor",
                ));
            }
            prior_stable_index_id = Some(u64::from(catalog.oid));
            let index_ref = bases
                .index_ref
                .checked_add(
                    u32::try_from(raw_ordinal)
                        .map_err(|_| error("S7 index ordinal exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 index reference overflows"))?;
            let key_start = bases
                .key_start
                .checked_add(
                    u32::try_from(flat_keys.len())
                        .map_err(|_| error("S7 index key count exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 index key reference overflows"))?;
            let mut keys = Vec::with_capacity(catalog.key_columns.len());
            for (ordinal, binding) in record.index_key_columns(source.raw_ordinal)?.enumerate() {
                if catalog.key_columns.get(ordinal).map(String::as_str) != Some(binding.name) {
                    return Err(error(
                        "sealed S2 index key order differs from the live catalog",
                    ));
                }
                let key_ref = key_start
                    .checked_add(
                        u32::try_from(ordinal)
                            .map_err(|_| error("S7 index key ordinal exceeds u32"))?,
                    )
                    .ok_or_else(|| error("S7 index key reference overflows"))?;
                let mut raw = [0_u8; 112];
                put_u32(&mut raw, 0, key_ref);
                put_u32(&mut raw, 4, index_ref);
                put_u32(
                    &mut raw,
                    8,
                    u32::try_from(ordinal)
                        .map_err(|_| error("S7 index key ordinal exceeds u32"))?,
                );
                put_u32(&mut raw, 12, binding.catalog_column_ordinal);
                put_u32(&mut raw, 16, binding.column_id);
                put_u32(&mut raw, 20, input.table.display_oid);
                put_i16(&mut raw, 24, binding.attnum);
                let storage = crate::typed_insert_batch::typed_image_sql_storage(binding.ty);
                raw[28..32].copy_from_slice(&storage);
                put_u32(&mut raw, 32, binding.type_oid);
                put_i16(&mut raw, 36, binding.type_size);
                put_digest(&mut raw, 40, write001_identifier_digest(binding.name)?);
                let digest = v2_digest(
                    b"gpu-db/write001/s7-index-key-column/v2",
                    &[&raw[..72], &[0; 32], &raw[104..]],
                );
                put_digest(&mut raw, 72, digest);
                keys.push(IndexedS7Key {
                    raw,
                    digest,
                    catalog_column_ordinal: binding.catalog_column_ordinal,
                    stable_column_id: binding.column_id,
                    attnum: binding.attnum,
                    storage,
                    declared_type_oid: binding.type_oid,
                    signed_type_size: binding.type_size,
                });
            }
            if keys.len() != catalog.key_columns.len() {
                return Err(error(
                    "sealed S2 named-index key count differs from its descriptor",
                ));
            }

            let index_name_digest = qualified_name_digest(input.table.schema, &catalog.name)?;
            let constraint_backed = catalog.primary_key || catalog.unique_constraint;
            let mut descriptor = [0_u8; 384];
            put_u32(&mut descriptor, 0, index_ref);
            put_u32(
                &mut descriptor,
                4,
                u32::from(catalog.unique)
                    | (u32::from(catalog.primary_key) << 1)
                    | (u32::from(catalog.unique_constraint) << 2)
                    | (1 << 3),
            );
            put_u32(&mut descriptor, 8, bases.table_ref);
            put_u32(&mut descriptor, 12, source.raw_ordinal);
            put_u64(&mut descriptor, 16, u64::from(catalog.oid));
            put_u32(&mut descriptor, 24, catalog.oid);
            put_u32(
                &mut descriptor,
                28,
                if constraint_backed { catalog.oid } else { 0 },
            );
            put_u64(
                &mut descriptor,
                32,
                if constraint_backed {
                    u64::from(catalog.oid)
                } else {
                    u64::MAX
                },
            );
            put_u32(&mut descriptor, 40, key_start);
            put_u32(
                &mut descriptor,
                44,
                u32::try_from(keys.len()).map_err(|_| error("S7 index key count exceeds u32"))?,
            );
            descriptor[48] = 1;
            put_u64(&mut descriptor, 64, input.table.stable_table_id);
            put_u64(&mut descriptor, 72, input.identity.catalog_epoch);
            put_digest(&mut descriptor, 80, input.table.schema_digest);
            put_digest(&mut descriptor, 112, owner_name_digest);
            put_digest(&mut descriptor, 144, index_name_digest);
            if constraint_backed {
                put_digest(&mut descriptor, 176, index_name_digest);
            }
            put_digest(&mut descriptor, 208, input.table.initial_table_root);
            put_digest(&mut descriptor, 240, generation.base_root);
            put_digest(&mut descriptor, 272, generation.final_root);
            put_u32(&mut descriptor, 336, input.table.display_oid);
            put_u64(&mut descriptor, 344, input.table.data_generation_before);
            put_u64(&mut descriptor, 352, generation.base_generation);
            put_u64(&mut descriptor, 360, generation.final_generation);
            let descriptor_digest = index_descriptor_digest(&descriptor, &keys);
            put_digest(&mut descriptor, 304, descriptor_digest);
            let static_dependency_ref = bases
                .dependency_start
                .checked_add(
                    u32::try_from(raw_ordinal)
                        .map_err(|_| error("S7 index dependency ordinal exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 index dependency reference overflows"))?;
            flat_keys.extend(keys.iter().map(|key| key.raw));
            owned.push(OwnedIndex {
                raw_ordinal: source.raw_ordinal,
                index_ref,
                static_dependency_ref,
                unique: catalog.unique,
                keys,
                descriptor,
                descriptor_digest,
            });
        }

        let effects_per_row = owned
            .iter()
            .try_fold(0_usize, |count, index| {
                count.checked_add(1 + usize::from(index.unique))
            })
            .ok_or_else(|| error("S7 per-row index-effect count overflows"))?;
        let component_capacity = owned
            .iter()
            .try_fold(0_usize, |count, index| {
                count.checked_add(index.keys.len() * (1 + usize::from(index.unique)))
            })
            .ok_or_else(|| error("S7 per-row index-component count overflows"))?;
        let mut components = Vec::with_capacity(rows.len().saturating_mul(component_capacity));
        let mut effects = Vec::with_capacity(rows.len().saturating_mul(effects_per_row));
        let mut effect_digests = Vec::with_capacity(rows.len().saturating_mul(effects_per_row));
        let mut values = Vec::new();
        let mut row_effect_ranges = Vec::with_capacity(rows.len());
        let mut unique_bindings = Vec::new();
        let mut unique_dependency_refs = vec![vec![None; owned.len()]; rows.len()];
        let mut next_unique_dependency_ref = bases
            .dependency_start
            .checked_add(index_count)
            .ok_or_else(|| error("S7 unique dependency reference overflows"))?;
        for (index_ordinal, index) in owned.iter().enumerate() {
            if !index.unique {
                continue;
            }
            for (row_ordinal, row) in rows.iter().enumerate() {
                let image_row = row
                    .image_row
                    .ok_or_else(|| error("indexed row has no final image coordinate"))?;
                if index_effect_participates(&final_image, &index.keys, image_row)? {
                    unique_dependency_refs[row_ordinal][index_ordinal] =
                        Some(next_unique_dependency_ref);
                    next_unique_dependency_ref = next_unique_dependency_ref
                        .checked_add(1)
                        .ok_or_else(|| error("S7 unique dependency reference overflows"))?;
                }
            }
        }
        for (row_ordinal, row) in rows.iter().enumerate() {
            let record = records
                .get(row.statement_index)
                .ok_or_else(|| error("indexed row statement record is absent"))?;
            let image_row = row
                .image_row
                .ok_or_else(|| error("indexed row has no final image coordinate"))?;
            let statement_ordinal = record.facts().statement_ordinal.as_u32();
            let effect_start = bases
                .effect_start
                .checked_add(
                    u32::try_from(effects.len())
                        .map_err(|_| error("S7 effect count exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 effect reference overflows"))?;
            for index in &owned {
                append_index_effect(
                    &final_image,
                    &index.keys,
                    index.descriptor_digest,
                    row.transition_ref,
                    image_row,
                    1,
                    index.static_dependency_ref,
                    index.index_ref,
                    index.raw_ordinal,
                    bases.effect_start,
                    bases.component_start,
                    bases.value_offset,
                    &mut components,
                    &mut values,
                    &mut effects,
                    &mut effect_digests,
                )?;
            }
            for (index_ordinal, index) in owned.iter().enumerate() {
                if !index.unique {
                    continue;
                }
                let dependency_ref =
                    unique_dependency_refs[row_ordinal][index_ordinal].unwrap_or(ABSENT_U32);
                let (effect_ref, participates) = append_index_effect(
                    &final_image,
                    &index.keys,
                    index.descriptor_digest,
                    row.transition_ref,
                    image_row,
                    2,
                    dependency_ref,
                    index.index_ref,
                    index.raw_ordinal,
                    bases.effect_start,
                    bases.component_start,
                    bases.value_offset,
                    &mut components,
                    &mut values,
                    &mut effects,
                    &mut effect_digests,
                )?;
                if participates {
                    if dependency_ref == ABSENT_U32 {
                        return Err(error(
                            "S7 unique effect participation changed during construction",
                        ));
                    }
                    unique_bindings.push((
                        dependency_ref,
                        effect_ref,
                        row.transition_ref,
                        statement_ordinal,
                        index.index_ref,
                        index.raw_ordinal,
                    ));
                } else if dependency_ref != ABSENT_U32 {
                    return Err(error(
                        "S7 unique effect participation changed during construction",
                    ));
                }
            }
            let effect_end = bases
                .effect_start
                .checked_add(
                    u32::try_from(effects.len())
                        .map_err(|_| error("S7 effect count exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 effect reference overflows"))?;
            let effect_count = effect_end
                .checked_sub(effect_start)
                .ok_or_else(|| error("S7 row effect range underflows"))?;
            row_effect_ranges.push((effect_start, effect_count));
        }

        let mut transitions = Vec::with_capacity(rows.len());
        for ((row, final_row_digest), (effect_start, effect_count)) in
            rows.iter().zip(final_row_digests).zip(&row_effect_ranges)
        {
            let statement = input
                .statements
                .iter()
                .find(|statement| statement.statement_ordinal as usize == row.statement_index)
                .ok_or_else(|| error("indexed transition statement is absent"))?;
            let mut transition = transition_bytes(
                row.transition_ref,
                row.table_ref,
                row.stable_row_id,
                row.s4_ref,
                statement.statement_ordinal,
                row.source_row,
                row.table_ref,
                row.image_row.expect("indexed row has a final image"),
                *effect_start,
                *effect_count,
                row.final_writer_statement_ordinal,
                statement.typed_statement_digest,
                *final_row_digest,
                [0; 32],
                row.final_writer_statement_digest,
            );
            let effect_end = effect_start
                .checked_add(*effect_count)
                .ok_or_else(|| error("S7 transition effect range overflows"))?;
            let transition_effect_digests = effect_digests
                .get(
                    effect_start
                        .checked_sub(bases.effect_start)
                        .ok_or_else(|| error("S7 transition effect range underflows"))?
                        as usize
                        ..effect_end
                            .checked_sub(bases.effect_start)
                            .ok_or_else(|| error("S7 transition effect range underflows"))?
                            as usize,
                )
                .ok_or_else(|| error("S7 transition effect range is absent"))?;
            let transition_digest = v2_digest(
                b"gpu-db/write001/s7-transition/v2",
                &[
                    &transition[..128],
                    &[0; 32],
                    &transition[160..],
                    &transition_effect_digests.concat(),
                ],
            );
            put_digest(&mut transition, 128, transition_digest);
            transitions.push(transition);
        }

        unique_bindings.sort_unstable_by_key(|binding| binding.0);
        let mut dependencies = Vec::with_capacity(owned.len() + unique_bindings.len());
        for index in &owned {
            dependencies.push(index_dependency(
                index.static_dependency_ref,
                3,
                &index.descriptor,
                index.descriptor_digest,
                input.identity.dependency_validation_floor,
                ABSENT_U32,
                [0; 32],
            ));
        }
        for (dependency_ref, effect_ref, _, _, index_ref, _) in &unique_bindings {
            let local_effect_ref = effect_ref
                .checked_sub(bases.effect_start)
                .ok_or_else(|| error("S7 unique effect reference underflows"))?;
            let index = owned
                .iter()
                .find(|index| index.index_ref == *index_ref)
                .ok_or_else(|| error("S7 unique dependency lost its index descriptor"))?;
            dependencies.push(index_dependency(
                *dependency_ref,
                4,
                &index.descriptor,
                index.descriptor_digest,
                input.identity.dependency_validation_floor,
                *effect_ref,
                effect_digests[local_effect_ref as usize],
            ));
        }
        let mut dependency_uses =
            Vec::with_capacity(input.statements.len() * owned.len() + unique_bindings.len());
        for statement in input.statements {
            for index in &owned {
                dependency_uses.push(dependency_use(
                    statement.statement_ordinal,
                    index.static_dependency_ref,
                    2,
                    index.raw_ordinal,
                    ABSENT_U32,
                    ABSENT_U32,
                ));
            }
        }
        for (dependency_ref, effect_ref, transition_ref, statement_ordinal, _, raw_ordinal) in
            unique_bindings
        {
            dependency_uses.push(dependency_use(
                statement_ordinal,
                dependency_ref,
                3,
                raw_ordinal,
                transition_ref,
                effect_ref,
            ));
        }
        Ok(Self {
            dependencies,
            dependency_uses,
            indexes: owned.iter().map(|index| index.descriptor).collect(),
            index_digests: owned.iter().map(|index| index.descriptor_digest).collect(),
            index_keys: flat_keys,
            transitions,
            effects,
            effect_digests,
            components,
            values,
            owned_index_start: bases.index_ref,
            owned_index_count: index_count,
        })
    }
}

fn index_effect_participates(
    image: &crate::typed_insert_batch::DecodedTypedImage,
    keys: &[IndexedS7Key],
    image_row: u32,
) -> Result<bool, EngineError> {
    for key in keys {
        let (valid, _, _) = typed_key_value_from_image(image, image_row, key)?;
        if !valid {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn append_index_effect(
    image: &crate::typed_insert_batch::DecodedTypedImage,
    keys: &[IndexedS7Key],
    descriptor_digest: [u8; 32],
    transition_ref: u32,
    image_row: u32,
    role: u8,
    dependency_ref: u32,
    index_ref: u32,
    source_catalog_ordinal: u32,
    effect_ref_base: u32,
    component_ref_base: u32,
    value_offset_base: u64,
    components: &mut Vec<[u8; 128]>,
    values: &mut Vec<u8>,
    effects: &mut Vec<[u8; 192]>,
    effect_digests: &mut Vec<[u8; 32]>,
) -> Result<(u32, bool), EngineError> {
    if !matches!(role, 1 | 2) {
        return Err(error("live target-index effect role is invalid"));
    }
    let effect_ref = effect_ref_base
        .checked_add(
            u32::try_from(effects.len()).map_err(|_| error("S7 effect count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 effect reference overflows"))?;
    let component_start = component_ref_base
        .checked_add(
            u32::try_from(components.len()).map_err(|_| error("S7 component count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 component reference overflows"))?;
    let mut contains_null = false;
    let mut component_digests = Vec::with_capacity(keys.len());
    for (ordinal, key) in keys.iter().enumerate() {
        let (valid, typed_value_digest, bytes) = typed_key_value_from_image(image, image_row, key)?;
        contains_null |= !valid;
        let component_ref = component_ref_base
            .checked_add(
                u32::try_from(components.len())
                    .map_err(|_| error("S7 component reference exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 component reference overflows"))?;
        let mut component = [0_u8; 128];
        put_u32(&mut component, 0, component_ref);
        put_u32(&mut component, 4, effect_ref);
        component[8] = 2;
        component[9] = u8::from(!valid);
        put_u32(
            &mut component,
            12,
            u32::try_from(ordinal).map_err(|_| error("S7 component ordinal exceeds u32"))?,
        );
        put_u32(
            &mut component,
            16,
            u32::from_le_bytes(
                key.raw[..4]
                    .try_into()
                    .expect("fixed S7 index-key reference"),
            ),
        );
        put_u32(&mut component, 20, key.catalog_column_ordinal);
        put_u64(
            &mut component,
            24,
            value_offset_base
                .checked_add(
                    u64::try_from(values.len())
                        .map_err(|_| error("S7 key value arena exceeds u64"))?,
                )
                .ok_or_else(|| error("S7 key value arena overflows"))?,
        );
        put_u32(
            &mut component,
            32,
            u32::try_from(bytes.len()).map_err(|_| error("S7 key value exceeds u32"))?,
        );
        component[36..40].copy_from_slice(&key.storage);
        put_u32(&mut component, 40, key.declared_type_oid);
        put_i16(&mut component, 44, key.signed_type_size);
        put_digest(&mut component, 48, typed_value_digest);
        let component_digest = v2_digest(
            b"gpu-db/write001/s7-typed-key-component/v2",
            &[&component[..80], &[0; 32], &component[112..], &key.digest],
        );
        put_digest(&mut component, 80, component_digest);
        values.extend_from_slice(&bytes);
        component_digests.push(component_digest);
        components.push(component);
    }
    let participates = role == 1 || !contains_null;
    let bound_dependency_ref = if participates {
        dependency_ref
    } else {
        ABSENT_U32
    };
    let effect_digest = key_effect_digest(
        effect_ref,
        role,
        transition_ref,
        index_ref,
        source_catalog_ordinal,
        component_start,
        &component_digests,
        descriptor_digest,
        contains_null,
        participates,
    );
    let mut effect = [0_u8; 192];
    put_u32(&mut effect, 0, effect_ref);
    effect[4] = role;
    effect[5] = role;
    put_u32(&mut effect, 8, transition_ref);
    put_u32(&mut effect, 12, index_ref);
    put_u32(&mut effect, 16, bound_dependency_ref);
    put_u32(&mut effect, 20, ABSENT_U32);
    put_u32(&mut effect, 28, component_start);
    let component_count =
        u32::try_from(keys.len()).map_err(|_| error("S7 effect component count exceeds u32"))?;
    put_u32(&mut effect, 32, component_count);
    put_u32(&mut effect, 36, component_count);
    effect[41] = 1;
    effect[42] = 1;
    effect[43] = u8::from(participates);
    effect[44] = u8::from(contains_null);
    put_u32(&mut effect, 48, source_catalog_ordinal);
    put_digest(
        &mut effect,
        96,
        typed_key_digest(effect_ref, &component_digests),
    );
    put_digest(&mut effect, 128, effect_digest);
    effects.push(effect);
    effect_digests.push(effect_digest);
    Ok((effect_ref, participates))
}

fn typed_key_value_from_image(
    image: &crate::typed_insert_batch::DecodedTypedImage,
    image_row: u32,
    key: &IndexedS7Key,
) -> Result<(bool, [u8; 32], Vec<u8>), EngineError> {
    let column = image
        .columns()
        .find(|column| column.catalog_column_ordinal == key.catalog_column_ordinal)
        .ok_or_else(|| error("indexed S7 final-image key column is absent"))?;
    if column.stable_column_id != key.stable_column_id
        || column.attnum != key.attnum
        || crate::typed_insert_batch::typed_image_sql_storage(column.ty) != key.storage
        || column.type_oid != key.declared_type_oid
        || column.type_size != key.signed_type_size
    {
        return Err(error(
            "indexed S7 final-image key metadata differs from its descriptor",
        ));
    }
    let row = usize::try_from(image_row)
        .map_err(|_| error("indexed S7 final-image row exceeds addressability"))?;
    let (is_null, bytes) =
        column.with_logical_cell_at(row, |is_null, bytes| (is_null, bytes.to_vec()))?;
    let valid = !is_null;
    let digest = typed_key_value_digest_from_bytes(
        valid,
        &bytes,
        key.storage,
        key.declared_type_oid,
        key.signed_type_size,
    );
    Ok((valid, digest, bytes))
}

impl PublishedSequenceClosure {
    #[allow(clippy::too_many_arguments)] // sealed S7 slices are independently validated inputs
    fn from_sealed_records(
        inputs: &[LiveTypedInsertView<'_>],
        statements: &[BoundStatement<'_>],
        records: &[DecodedTypedInsertRecord],
        statement_row_starts: &[u32],
        statement_table_row_starts: &[u32],
        statement_row_counts: &[u32],
        transition_starts: &[u32],
        first_dependency_ref: u32,
        terminal_restarts: &[LiveTypedInsertSequenceRestart],
    ) -> Result<Self, EngineError> {
        let mut matched_references = inputs
            .iter()
            .map(|input| vec![false; input.published_sequence_references.len()])
            .collect::<Vec<_>>();
        let mut entries = Vec::new();
        for (statement_index, record) in records.iter().enumerate() {
            let bound = statements.get(statement_index).ok_or_else(|| {
                error("published sequence record has no staged statement identity")
            })?;
            let statement = bound.statement;
            let input = bound.input;
            let row_start = *statement_row_starts
                .get(statement_index)
                .ok_or_else(|| error("published sequence statement has no S4 start"))?;
            let table_row_start = *statement_table_row_starts
                .get(statement_index)
                .ok_or_else(|| error("published sequence statement has no table-row start"))?;
            let row_count = *statement_row_counts
                .get(statement_index)
                .ok_or_else(|| error("published sequence statement has no S4 count"))?;
            let parent = record.sequence_parent();
            for source in record.sequence_effects() {
                let mut bindings = record
                    .sequence_bindings()
                    .filter(|binding| binding.effect_ordinal == source.request.effect_ordinal);
                let binding = bindings
                    .next()
                    .ok_or_else(|| error("sequence effect lacks its S2 binding"))?;
                if bindings.next().is_some() || binding.request != source.request {
                    return Err(error("sequence effect has a noncanonical S2 binding"));
                }
                let disposition_ref = row_start
                    .checked_add(source.request.row_ordinal)
                    .ok_or_else(|| error("sequence source disposition reference overflows"))?;
                let table_row = table_row_start
                    .checked_add(source.request.row_ordinal)
                    .ok_or_else(|| error("sequence source table-row identity overflows"))?;
                let final_writer = input
                    .final_writers
                    .get(table_row as usize)
                    .ok_or_else(|| error("sequence source has no final disposition"))?;
                let stable_row_id = final_writer.stable_row_id;
                let transition_ref = if final_writer.survives {
                    let surviving_before = input.final_writers[..table_row as usize]
                        .iter()
                        .filter(|writer| writer.survives)
                        .count();
                    transition_starts
                        .get(bound.table_ref as usize)
                        .copied()
                        .ok_or_else(|| error("sequence source table has no transition range"))?
                        .checked_add(
                            u32::try_from(surviving_before)
                                .map_err(|_| error("sequence survivor ordinal exceeds u32"))?,
                        )
                        .ok_or_else(|| error("sequence source transition reference overflows"))?
                } else {
                    ABSENT_U32
                };
                let source_parent =
                    parent.ok_or_else(|| error("sequence effect lacks its S2 parent"))?;
                let target = record.target_identity();
                if source.request.effect_ordinal == ABSENT_U32
                    || source.request.row_ordinal >= row_count
                    || source.request.statement_ordinal.as_u32() != statement.statement_ordinal
                    || source.request.target_table_oid != input.table.display_oid
                    || target.oid != input.table.display_oid
                    || target.schema != input.table.schema
                    || source_parent.txn_id != input.identity.stable_transaction_id
                    || source_parent.statement_ordinal.as_u32() != statement.statement_ordinal
                    || source_parent.autocommit
                        != matches!(input.identity.mode, LiveTypedInsertMode::Autocommit)
                {
                    return Err(error(
                        "sequence effect does not close its S1/S2/S4 identity",
                    ));
                }
                let kind = match source.kind {
                    crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Published {
                        transition_txn_id,
                        input_digest,
                        returned_value,
                    } => {
                        let reference_index = input
                            .published_sequence_references
                            .iter()
                            .enumerate()
                            .find_map(|(index, reference)| {
                                (!matched_references[bound.table_ref as usize][index]
                                    && reference.transition_txn_id == transition_txn_id
                                    && reference.sequence_oid == source.request.sequence_oid)
                                    .then_some(index)
                            })
                            .ok_or_else(|| {
                                error("published S2 sequence effect lacks its durable receipt")
                            })?;
                        matched_references[bound.table_ref as usize][reference_index] = true;
                        let reference = &input.published_sequence_references[reference_index];
                        if reference.parent_txn_id != input.identity.stable_transaction_id
                            || reference.statement_ordinal != statement.statement_ordinal
                            || reference.expression_ordinal
                                != source.request.absolute_expression_ordinal
                            || reference.sequence_oid != source.request.sequence_oid
                            || reference.returned_value != returned_value
                            || reference.input_digest != input_digest
                            || reference.table_oid != source.request.target_table_oid
                            || reference.column_id != source.request.column_id
                            || reference.row_id != stable_row_id
                            || !reference.default_expression
                        {
                            return Err(error(
                                "published sequence receipt does not close its S2/S4 identity",
                            ));
                        }
                        let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
                        crate::encode_sequence_value_reference_into_exact(reference, &mut body)?;
                        SequenceEntryKind::Published {
                            // Persistent catalog OIDs are the current stable sequence identities.
                            stable_sequence_id: u64::from(source.request.sequence_oid),
                            sequence_oid: source.request.sequence_oid,
                            transition_txn_id,
                            name_digest: qualified_name_digest(
                                input.table.schema,
                                binding.effective_name,
                            )?,
                            reference_body_digest: gpu_db_wal::canonical_request_digest(&body),
                            body,
                            final_value_overwritten: reference.final_value_overwritten,
                        }
                    }
                    crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Private {
                        ..
                    } => {
                        if source.resolved_value == 0 {
                            return Err(error("private sequence effect resolved to zero"));
                        }
                        SequenceEntryKind::Private {
                            final_value_overwritten: !final_writer.survives,
                        }
                    }
                };
                entries.push(SequenceEntry {
                    statement_ordinal: statement.statement_ordinal,
                    source_operation_ordinal: statement.operation_ordinal,
                    effect_ordinal: source.request.effect_ordinal,
                    disposition_ref,
                    table_ref: bound.table_ref,
                    transition_ref,
                    sequence_oid: source.request.sequence_oid,
                    terminal_restart: None,
                    kind,
                });
            }
        }
        let reference_count = inputs.iter().try_fold(0_usize, |count, input| {
            count
                .checked_add(input.published_sequence_references.len())
                .ok_or_else(|| error("durable sequence receipt inventory overflows"))
        })?;
        let published_count = entries
            .iter()
            .filter(|entry| matches!(entry.kind, SequenceEntryKind::Published { .. }))
            .count();
        if published_count != reference_count
            || matched_references
                .iter()
                .any(|table| table.iter().any(|matched| !matched))
        {
            return Err(error(
                "durable sequence receipt inventory differs from published S2 effects",
            ));
        }
        let mut terminal_sequences = std::collections::BTreeSet::new();
        let request_digest = inputs[0].identity.request_digest;
        for restart in terminal_restarts {
            if restart.owner_operation_ordinal == u32::MAX
                || restart.owner_statement_digest == [0; 32]
                || restart.sequence_oid == 0
                || restart.sequence_name.is_empty()
                || !terminal_sequences.insert(restart.sequence_oid)
                || entries.iter().any(|entry| {
                    entry.sequence_oid == restart.sequence_oid
                        && entry.source_operation_ordinal >= restart.owner_operation_ordinal
                })
            {
                return Err(error(
                    "terminal sequence restart is not strictly after one exact S5 chain",
                ));
            }
            let entry = entries
                .iter_mut()
                .rev()
                .find(|entry| entry.sequence_oid == restart.sequence_oid)
                .ok_or_else(|| error("terminal sequence restart has no preceding S5 effect"))?;
            let descriptor_digest =
                crate::sequence_descriptor_digest(restart.sequence_oid, &restart.sequence_name);
            let tail = super::sequence_terminal::SequenceRestartTail {
                owner_operation_ordinal: restart.owner_operation_ordinal,
                sequence_oid: restart.sequence_oid,
                last_value: restart.last_value,
                owner_statement_digest: restart.owner_statement_digest,
                descriptor_digest,
                witness_digest: super::sequence_terminal::witness_digest(
                    request_digest,
                    restart.owner_operation_ordinal,
                    restart.sequence_oid,
                    restart.last_value,
                    restart.owner_statement_digest,
                    descriptor_digest,
                ),
            };
            super::sequence_terminal::encode(request_digest, tail)?;
            entry.terminal_restart = Some(tail);
        }
        let mut prior_transition = None;
        let mut s5 = Vec::with_capacity(
            entries.len()
                * (52
                    + crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES
                    + super::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES),
        );
        let mut statement_ranges = Vec::with_capacity(records.len());
        for statement in statements {
            let start = s5.len();
            for entry in entries
                .iter()
                .filter(|entry| entry.statement_ordinal == statement.statement.statement_ordinal)
            {
                put_u32_vec(&mut s5, entry.statement_ordinal);
                put_u32_vec(&mut s5, entry.effect_ordinal);
                match entry.kind {
                    SequenceEntryKind::Published {
                        transition_txn_id,
                        body,
                        final_value_overwritten,
                        ..
                    } => {
                        if prior_transition.is_some_and(|prior| prior >= transition_txn_id) {
                            return Err(error(
                                "published sequence receipts are not strictly transition ordered",
                            ));
                        }
                        s5.push(1);
                        s5.push(1 | if final_value_overwritten { 2 } else { 0 });
                        s5.extend_from_slice(&0_u16.to_le_bytes());
                        put_u32_vec(&mut s5, entry.disposition_ref);
                        let mut encoded_body = Vec::with_capacity(
                            crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES
                                + super::sequence_terminal::SEQUENCE_RESTART_TAIL_BYTES,
                        );
                        encoded_body.extend_from_slice(&body);
                        if let Some(tail) = entry.terminal_restart {
                            encoded_body.extend_from_slice(&super::sequence_terminal::encode(
                                request_digest,
                                tail,
                            )?);
                        }
                        put_u32_vec(
                            &mut s5,
                            u32::try_from(encoded_body.len())
                                .map_err(|_| error("S5 sequence body exceeds u32"))?,
                        );
                        s5.extend_from_slice(&gpu_db_wal::canonical_request_digest(&encoded_body));
                        s5.extend_from_slice(&encoded_body);
                        prior_transition = Some(transition_txn_id);
                    }
                    SequenceEntryKind::Private {
                        final_value_overwritten,
                    } => {
                        s5.push(2);
                        s5.push(1 | if final_value_overwritten { 2 } else { 0 });
                        s5.extend_from_slice(&0_u16.to_le_bytes());
                        put_u32_vec(&mut s5, entry.disposition_ref);
                        if let Some(tail) = entry.terminal_restart {
                            let body = super::sequence_terminal::encode(request_digest, tail)?;
                            put_u32_vec(&mut s5, body.len() as u32);
                            s5.extend_from_slice(&gpu_db_wal::canonical_request_digest(&body));
                            s5.extend_from_slice(&body);
                        } else {
                            put_u32_vec(&mut s5, 0);
                            s5.extend_from_slice(&[0; 32]);
                        }
                    }
                }
            }
            statement_ranges.push(start..s5.len());
        }

        let mut published_entries = entries
            .iter()
            .filter(|entry| matches!(entry.kind, SequenceEntryKind::Published { .. }))
            .copied()
            .collect::<Vec<_>>();
        published_entries.sort_by_key(|entry| match entry.kind {
            SequenceEntryKind::Published {
                stable_sequence_id,
                sequence_oid,
                transition_txn_id,
                ..
            } => (stable_sequence_id, sequence_oid, transition_txn_id),
            SequenceEntryKind::Private { .. } => {
                unreachable!("filtered published sequence entry")
            }
        });
        let mut dependencies = Vec::with_capacity(published_entries.len());
        let mut dependency_uses = Vec::with_capacity(published_entries.len());
        for (ordinal, entry) in published_entries.iter().enumerate() {
            let dependency_ref = first_dependency_ref
                .checked_add(
                    u32::try_from(ordinal)
                        .map_err(|_| error("published sequence dependency count exceeds u32"))?,
                )
                .ok_or_else(|| error("published sequence dependency reference overflows"))?;
            dependencies.push(published_sequence_dependency(
                dependency_ref,
                inputs[0].identity.catalog_epoch,
                entry,
            ));
            dependency_uses.push(dependency_use(
                entry.statement_ordinal,
                dependency_ref,
                8,
                entry.effect_ordinal,
                entry.transition_ref,
                ABSENT_U32,
            ));
        }
        Ok(Self {
            s5,
            statement_ranges,
            published_count: u32::try_from(published_count)
                .map_err(|_| error("published sequence count exceeds u32"))?,
            private_count: u32::try_from(entries.len() - published_count)
                .map_err(|_| error("private sequence count exceeds u32"))?,
            dependencies,
            dependency_uses,
        })
    }
}

impl ReturningClosure {
    fn from_sealed_record(
        record: &DecodedTypedInsertRecord,
        row_count: u32,
        statement_ordinal: u32,
        table_ref: u32,
        projection_start: u32,
    ) -> Result<Self, EngineError> {
        let facts = record.facts();
        let sources = record.returning_projections().collect::<Vec<_>>();
        if sources.is_empty() {
            if facts.returning.column_count != 0 || facts.returning.cell_count != 0 {
                return Err(error(
                    "empty RETURNING projection inventory has nonempty S2 geometry",
                ));
            }
            return Ok(Self {
                projections: Vec::new(),
                logical_result_digest: [0; 32],
            });
        }
        let projection_count = u32::try_from(sources.len())
            .map_err(|_| error("RETURNING projection count exceeds u32"))?;
        if facts.returning.column_count != projection_count
            || facts.returning.row_count != row_count
            || facts.returning.cell_count != u64::from(row_count) * u64::from(projection_count)
        {
            return Err(error(
                "RETURNING projection inventory differs from its S2 geometry",
            ));
        }
        let mut projections = Vec::with_capacity(sources.len());
        for (ordinal, source) in sources.iter().enumerate() {
            let ordinal = u32::try_from(ordinal)
                .map_err(|_| error("RETURNING projection ordinal exceeds u32"))?;
            let mut raw = [0_u8; 128];
            put_u32(
                &mut raw,
                0,
                projection_start
                    .checked_add(ordinal)
                    .ok_or_else(|| error("RETURNING projection reference overflows"))?,
            );
            put_u32(&mut raw, 4, statement_ordinal);
            put_u32(&mut raw, 8, ordinal);
            put_u32(&mut raw, 12, source.catalog_column_ordinal);
            put_u32(&mut raw, 16, source.column_id);
            put_u32(&mut raw, 20, table_ref);
            put_i16(&mut raw, 24, source.attnum);
            raw[28..32].copy_from_slice(&crate::typed_insert_batch::typed_image_sql_storage(
                source.ty,
            ));
            put_u32(&mut raw, 32, source.type_oid);
            put_i16(&mut raw, 36, source.type_size);
            // Logical engine RETURNING is canonically text-format-neutral. Pgwire may encode the
            // already-returned values for a client after this digest is closed, but that transport
            // choice is not another durable result authority.
            put_u16(&mut raw, 38, 0);
            put_u32(&mut raw, 40, ordinal);
            put_digest(&mut raw, 64, write001_identifier_digest(source.name)?);
            let digest = v2_digest(b"gpu-db/write001/s7-projection/v2", &[&raw[..96], &[0; 32]]);
            put_digest(&mut raw, 96, digest);
            projections.push(raw);
        }
        let logical_result_digest =
            logical_returning_digest(record, statement_ordinal, row_count, &projections)?;
        Ok(Self {
            projections,
            logical_result_digest,
        })
    }
}

// The pure writer unit test intentionally has no GPU requirement. Production has no source-view
// implementation, so a live canonical write cannot silently fall back to CPU final-row hashing.
#[cfg(test)]
impl LiveFinalRowDigestSource for PreparedResidentRuntimeGenerationView<'_> {
    fn copy_write001_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        *final_row_destination = self.final_row_digest(row_ordinal, 0, row_ordinal as u32)?;
        if stable_row_id == 0 {
            return Err(error("stable row identity is absent"));
        }
        Ok(())
    }

    fn copy_write001_s7_final_row_digest_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        _typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        self.copy_write001_final_row_digest_into(row_ordinal, stable_row_id, final_row_destination)
    }

    fn copy_write001_transition_digests_into(
        &self,
        row_ordinal: usize,
        stable_row_id: u64,
        typed_statement_digest: [u8; 32],
        final_row_destination: &mut [u8; 32],
        transition_destination: &mut [u8; 32],
    ) -> Result<(), EngineError> {
        self.copy_write001_s7_final_row_digest_into(
            row_ordinal,
            stable_row_id,
            typed_statement_digest,
            final_row_destination,
        )?;
        *transition_destination = transition_digest(
            row_ordinal as u32,
            0,
            stable_row_id,
            row_ordinal as u32,
            0,
            row_ordinal as u32,
            0,
            row_ordinal as u32,
            0,
            0,
            0,
            typed_statement_digest,
            *final_row_destination,
            [0; 32],
        );
        Ok(())
    }
}

pub(crate) fn encode_live_typed_insert(
    input: &LiveTypedInsertView<'_>,
) -> Result<ClosedLiveTypedInsert, EngineError> {
    encode_live_typed_insert_transaction(std::slice::from_ref(input), &[], None, false)
}

#[derive(Clone, Copy)]
struct BoundStatement<'a> {
    table_ref: u32,
    input: &'a LiveTypedInsertView<'a>,
    statement: &'a LiveTypedInsertStatementView<'a>,
}

#[derive(Clone, Copy)]
struct BoundRow {
    statement_index: usize,
    source_row: u32,
    s4_ref: u32,
    table_ref: u32,
    table_row: u32,
    image_row: Option<u32>,
    transition_ref: u32,
    stable_row_id: u64,
    final_writer_statement_ordinal: u32,
    final_writer_statement_digest: gpu_db_wal::CanonicalDigest,
}

fn same_transaction_identity(
    left: LiveTypedInsertIdentity,
    right: LiveTypedInsertIdentity,
) -> bool {
    left.physical == right.physical
        && left.canonical == right.canonical
        && left.leader_epoch == right.leader_epoch
        && left.commit_sequence == right.commit_sequence
        && left.stable_transaction_id == right.stable_transaction_id
        && matches!(
            (left.mode, right.mode),
            (
                LiveTypedInsertMode::Autocommit,
                LiveTypedInsertMode::Autocommit
            ) | (LiveTypedInsertMode::Explicit, LiveTypedInsertMode::Explicit)
        )
        && left.request_digest == right.request_digest
        && left.isolation == right.isolation
        && left.catalog_epoch == right.catalog_epoch
        && left.catalog_digest == right.catalog_digest
        && left.catalog_after_epoch == right.catalog_after_epoch
        && left.catalog_after_digest == right.catalog_after_digest
        && left.dependency_validation_floor == right.dependency_validation_floor
}

/// Close every table touched by one transaction into the existing plural semantics-v2 grammar.
/// Tables are stable-id ordered, statements remain transaction ordered, and their table-local
/// rows are bound through S4 to table-ordered transitions/images. This is one encoder and one
/// envelope authority; the one-table entry point above is only its compatibility-shaped view.
pub(crate) fn encode_live_typed_insert_transaction(
    inputs: &[LiveTypedInsertView<'_>],
    terminal_restarts: &[LiveTypedInsertSequenceRestart],
    operation_body: Option<&[u8]>,
    operation_changes_catalog: bool,
) -> Result<ClosedLiveTypedInsert, EngineError> {
    #[cfg(feature = "probe-timing")]
    let probe_s7_build_started = std::time::Instant::now();
    let first = inputs
        .first()
        .ok_or_else(|| error("live typed INSERT transaction has no table"))?;
    let identity = first.identity;
    let table_count = u32::try_from(inputs.len()).map_err(|_| error("table count exceeds u32"))?;
    let mut prior_table_id = None;
    let mut prior_database_root = None;
    let mut statements = Vec::new();
    for (table_ref, input) in inputs.iter().enumerate() {
        validate_input(input)?;
        let table_ref =
            u32::try_from(table_ref).map_err(|_| error("table reference exceeds u32"))?;
        if !same_transaction_identity(identity, input.identity)
            || prior_table_id.is_some_and(|prior| prior >= input.table.stable_table_id)
            || prior_database_root.is_some_and(|root| root != input.table.initial_database_root)
        {
            return Err(error(
                "live typed INSERT tables are not one identity/root chain in stable-id order",
            ));
        }
        for statement in input.statements {
            if statement.table_ref != table_ref {
                return Err(error(
                    "typed statement table reference differs from its table scope",
                ));
            }
            statements.push(BoundStatement {
                table_ref,
                input,
                statement,
            });
        }
        prior_table_id = Some(input.table.stable_table_id);
        prior_database_root = Some(input.table.final_database_root);
    }
    statements.sort_unstable_by_key(|bound| bound.statement.statement_ordinal);
    if statements.is_empty()
        || statements.iter().enumerate().any(|(ordinal, bound)| {
            bound.statement.statement_ordinal as usize != ordinal
                || bound.statement.typed_statement_digest == [0; 32]
                || bound.statement.record.is_empty()
        })
    {
        return Err(error(
            "live typed INSERT statements are not an exact transaction-ordered set",
        ));
    }
    let row_count = inputs.iter().try_fold(0_u32, |total, input| {
        total
            .checked_add(
                u32::try_from(input.final_writers.len())
                    .map_err(|_| error("table row count exceeds u32"))?,
            )
            .ok_or_else(|| error("transaction row count overflows u32"))
    })?;
    let statement_count =
        u32::try_from(statements.len()).map_err(|_| error("statement count exceeds u32"))?;
    let initial_database_root = first.table.initial_database_root;
    let final_database_root = inputs
        .last()
        .expect("nonempty checked table scopes")
        .table
        .final_database_root;
    // This is the one live-only elision: source sealing already proved the scalar row geometry
    // and the complete absence of every closure the strict decoder supplies below. The source
    // pointer must still be the exact immutable owner of the S2 bytes. Recovery, fixtures, and
    // any RETURNING/sequence/domain/index/FK shape retain the full strict decoder.
    let feature_free_live_closure = terminal_restarts.is_empty()
        && inputs.iter().all(|input| {
            input.table.indexes.is_empty()
                && input.foreign_indexes.is_empty()
                && input.published_sequence_references.is_empty()
        })
        && statements.iter().all(|bound| {
            let statement = bound.statement;
            statement.sealed_source.is_some_and(|source| {
                source
                    .feature_free_live_row_count_for(statement.statement_ordinal)
                    .is_some()
                    && source.record().len() == statement.record.len()
                    && std::ptr::eq(source.record().as_ptr(), statement.record.as_ptr())
            })
        });
    let mut decoded_records = Vec::with_capacity(statements.len());
    let mut record_prefixes = Vec::with_capacity(statements.len());
    let mut record_bytes = Vec::with_capacity(statements.len());
    let mut statement_row_starts = Vec::with_capacity(statements.len());
    let mut statement_table_row_starts = Vec::with_capacity(statements.len());
    let mut statement_row_counts = Vec::with_capacity(statements.len());
    let mut table_rows_seen = vec![0_u32; inputs.len()];
    let mut returnings = Vec::with_capacity(statements.len());
    let mut total_statement_rows = 0_u32;
    let mut total_returning_projections = 0_u32;
    #[cfg(feature = "probe-timing")]
    let mut probe_s2_decode_nanos = 0_u64;
    for (ordinal, bound) in statements.iter().enumerate() {
        let statement = bound.statement;
        let ordinal = u32::try_from(ordinal).map_err(|_| error("statement ordinal exceeds u32"))?;
        let bytes =
            u32::try_from(statement.record.len()).map_err(|_| error("S2 record exceeds u32"))?;
        let prefix: &[u8; CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES] = statement
            .record
            .get(..CANONICAL_TYPED_INSERT_RECORD_HEADER_BYTES)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| error("S2 record is shorter than its fixed header"))?;
        let record_prefix =
            parse_canonical_typed_insert_record_prefix(prefix, statement.record.len())?;
        let (statement_rows, returning) = if feature_free_live_closure {
            let rows = statement
                .sealed_source
                .and_then(|source| source.feature_free_live_row_count_for(ordinal))
                .expect("feature-free source was proved above");
            if statement.statement_ordinal != ordinal
                || statement.typed_statement_digest == [0; 32]
                || record_prefix.typed_statement_digest != statement.typed_statement_digest
                || rows == 0
            {
                return Err(error(
                    "provenance-sealed S2 statement differs from its live closure identity",
                ));
            }
            (
                rows,
                ReturningClosure {
                    projections: Vec::new(),
                    logical_result_digest: [0; 32],
                },
            )
        } else {
            #[cfg(feature = "probe-timing")]
            let probe_s2_decode_started = std::time::Instant::now();
            let decoded = decode_canonical_typed_insert_record(statement.record)?;
            #[cfg(feature = "probe-timing")]
            {
                probe_s2_decode_nanos = probe_s2_decode_nanos
                    .saturating_add(probe_s2_decode_started.elapsed().as_nanos() as u64);
            }
            let facts = decoded.facts();
            if statement.statement_ordinal != ordinal
                || facts.statement_ordinal.as_u32() != ordinal
                || statement.typed_statement_digest == [0; 32]
                || record_prefix.typed_statement_digest != statement.typed_statement_digest
                || facts.typed_statement_digest != statement.typed_statement_digest
                || facts.row_count == 0
            {
                return Err(error(
                    "S2 statement ordinal, digest, or row geometry differs from the staged source",
                ));
            }
            let returning = ReturningClosure::from_sealed_record(
                &decoded,
                facts.row_count,
                ordinal,
                bound.table_ref,
                total_returning_projections,
            )?;
            decoded_records.push(decoded);
            (facts.row_count, returning)
        };
        statement_row_starts.push(total_statement_rows);
        statement_table_row_starts.push(table_rows_seen[bound.table_ref as usize]);
        statement_row_counts.push(statement_rows);
        table_rows_seen[bound.table_ref as usize] = table_rows_seen[bound.table_ref as usize]
            .checked_add(statement_rows)
            .ok_or_else(|| error("table statement row count overflows"))?;
        total_statement_rows = total_statement_rows
            .checked_add(statement_rows)
            .ok_or_else(|| error("combined statement row count overflows"))?;
        total_returning_projections = total_returning_projections
            .checked_add(
                u32::try_from(returning.projections.len())
                    .map_err(|_| error("RETURNING projection count exceeds u32"))?,
            )
            .ok_or_else(|| error("RETURNING projection count overflows"))?;
        returnings.push(returning);
        record_prefixes.push(record_prefix);
        record_bytes.push(bytes);
    }
    if total_statement_rows != row_count {
        return Err(error(
            "combined S2 statement rows differ from the final image",
        ));
    }
    if !feature_free_live_closure {
        crate::typed_insert_batch::validate_canonical_typed_insert_private_sequence_chains(
            &decoded_records,
        )?;
    }
    if inputs.iter().enumerate().any(|(table_ref, input)| {
        usize::try_from(table_rows_seen[table_ref]).ok() != Some(input.final_writers.len())
    }) {
        return Err(error(
            "table-scoped S2 statement rows differ from their final images",
        ));
    }
    let projection_count = returnings.iter().try_fold(0_u32, |count, returning| {
        count
            .checked_add(
                u32::try_from(returning.projections.len())
                    .map_err(|_| error("RETURNING projection count exceeds u32"))?,
            )
            .ok_or_else(|| error("RETURNING projection count overflows"))
    })?;
    let has_returning = projection_count != 0;

    let initial_overlay = initial_overlay_root(identity, initial_database_root);
    let mut s4 = Vec::with_capacity(row_count as usize * S4_BYTES);
    let mut table_dispositions = Vec::with_capacity(row_count as usize * 32);
    let mut table_rows = vec![Vec::<BoundRow>::new(); inputs.len()];
    let mut transition_starts = Vec::with_capacity(inputs.len());
    let mut next_transition = 0_u32;
    for input in inputs {
        transition_starts.push(next_transition);
        next_transition = next_transition
            .checked_add(
                u32::try_from(
                    input
                        .final_writers
                        .iter()
                        .filter(|writer| writer.survives)
                        .count(),
                )
                .map_err(|_| error("table survivor count exceeds u32"))?,
            )
            .ok_or_else(|| error("transition range overflows"))?;
    }
    let mut table_image_rows = vec![0_u32; inputs.len()];
    for (statement_index, bound) in statements.iter().enumerate() {
        let statement = bound.statement;
        let statement_rows = statement_row_counts[statement_index];
        let s4_start = statement_row_starts[statement_index];
        let table_row_start = statement_table_row_starts[statement_index];
        for source_row in 0..statement_rows {
            let s4_ref = s4_start
                .checked_add(source_row)
                .ok_or_else(|| error("combined row ordinal overflows"))?;
            let table_row = table_row_start
                .checked_add(source_row)
                .ok_or_else(|| error("table row ordinal overflows"))?;
            let final_writer = bound
                .input
                .final_writers
                .get(table_row as usize)
                .ok_or_else(|| error("table row has no final-writer binding"))?;
            let stable_row_id = final_writer.stable_row_id;
            if final_writer.source_statement_ordinal != statement.statement_ordinal
                || final_writer.source_row_ordinal != source_row
            {
                return Err(error(
                    "table-row final writer differs from its exact S2 source",
                ));
            }
            let image_row = final_writer
                .survives
                .then_some(table_image_rows[bound.table_ref as usize]);
            let transition_ref = if let Some(image_row) = image_row {
                table_image_rows[bound.table_ref as usize] = image_row
                    .checked_add(1)
                    .ok_or_else(|| error("table image row ordinal overflows"))?;
                transition_starts[bound.table_ref as usize]
                    .checked_add(image_row)
                    .ok_or_else(|| error("transition reference overflows"))?
            } else {
                ABSENT_U32
            };
            append_s4(
                &mut s4,
                statement.statement_ordinal,
                source_row,
                bound.table_ref,
                transition_ref,
                stable_row_id,
                statement.typed_statement_digest,
                final_writer,
            );
            table_rows[bound.table_ref as usize].push(BoundRow {
                statement_index,
                source_row,
                s4_ref,
                table_ref: bound.table_ref,
                table_row,
                image_row,
                transition_ref,
                stable_row_id,
                final_writer_statement_ordinal: final_writer.final_writer_statement_ordinal,
                final_writer_statement_digest: final_writer.final_writer_statement_digest,
            });
        }
    }
    let mut final_row_digests = Vec::with_capacity(inputs.len());
    let mut device_transition_digests = Vec::with_capacity(inputs.len());
    for (table_ref, rows) in table_rows.iter().enumerate() {
        let input = &inputs[table_ref];
        let survivor_count = rows.iter().filter(|row| row.image_row.is_some()).count();
        let mut table_final_digests = Vec::with_capacity(survivor_count);
        let mut table_device_transition_digests = Vec::with_capacity(survivor_count);
        let mut disposition_rows = rows.iter().collect::<Vec<_>>();
        disposition_rows.sort_unstable_by_key(|row| row.stable_row_id);
        for row in disposition_rows {
            let statement = statements[row.statement_index].statement;
            append_table_disposition(
                &mut table_dispositions,
                row.table_ref,
                row.s4_ref,
                statement.statement_ordinal,
                row.source_row,
                row.stable_row_id,
                if row.image_row.is_some() { 1 } else { 2 },
            );
        }
        for row in rows {
            let statement = statements[row.statement_index].statement;
            let Some(image_row) = row.image_row else {
                continue;
            };
            let mut final_row_digest = [0_u8; 32];
            let mut transition_digest = [0_u8; 32];
            if statement_count == 1
                && input.table.indexes.is_empty()
                && input.foreign_indexes.is_empty()
                && row.final_writer_statement_digest == [0; 32]
            {
                input
                    .final_row_digests
                    .ok_or_else(|| error("surviving row has no GPU final-row digest source"))?
                    .copy_write001_transition_digests_into(
                        image_row as usize,
                        row.stable_row_id,
                        statement.typed_statement_digest,
                        &mut final_row_digest,
                        &mut transition_digest,
                    )?;
            } else {
                input
                    .final_row_digests
                    .ok_or_else(|| error("surviving row has no GPU final-row digest source"))?
                    .copy_write001_final_row_digest_into(
                        image_row as usize,
                        row.stable_row_id,
                        &mut final_row_digest,
                    )?;
            }
            table_final_digests.push(final_row_digest);
            table_device_transition_digests.push(transition_digest);
        }
        final_row_digests.push(table_final_digests);
        device_transition_digests.push(table_device_transition_digests);
    }

    let mut table_s7_ranges = vec![IndexedS7TableRange::default(); inputs.len()];
    let mut indexed = {
        let mut closure = IndexedS7Closure {
            dependencies: Vec::new(),
            dependency_uses: Vec::new(),
            indexes: Vec::new(),
            index_digests: Vec::new(),
            index_keys: Vec::new(),
            transitions: Vec::with_capacity(row_count as usize),
            effects: Vec::new(),
            effect_digests: Vec::new(),
            components: Vec::new(),
            values: Vec::new(),
            owned_index_start: 0,
            owned_index_count: 0,
        };
        let mut shared_initial_foreign_indexes = Vec::new();
        for (table_ref, rows) in table_rows.iter().enumerate() {
            let surviving_rows = rows
                .iter()
                .filter(|row| row.image_row.is_some())
                .copied()
                .collect::<Vec<_>>();
            let effect_start = u32::try_from(closure.effects.len())
                .map_err(|_| error("S7 effect count exceeds u32"))?;
            if !inputs[table_ref].foreign_indexes.is_empty()
                || !inputs[table_ref].table.indexes.is_empty()
            {
                let bases = IndexedS7Bases {
                    table_ref: u32::try_from(table_ref)
                        .map_err(|_| error("S7 table reference exceeds u32"))?,
                    index_ref: u32::try_from(closure.indexes.len())
                        .map_err(|_| error("S7 index reference exceeds u32"))?,
                    key_start: u32::try_from(closure.index_keys.len())
                        .map_err(|_| error("S7 index-key reference exceeds u32"))?,
                    effect_start,
                    component_start: u32::try_from(closure.components.len())
                        .map_err(|_| error("S7 component reference exceeds u32"))?,
                    value_offset: u64::try_from(closure.values.len())
                        .map_err(|_| error("S7 value offset exceeds u64"))?,
                    dependency_start: table_count
                        .checked_add(
                            u32::try_from(closure.dependencies.len())
                                .map_err(|_| error("S7 dependency count exceeds u32"))?,
                        )
                        .ok_or_else(|| error("S7 dependency reference overflows"))?,
                };
                let table_closure = if !inputs[table_ref].foreign_indexes.is_empty() {
                    foreign_key_closure::from_sealed_records(
                        &inputs[table_ref],
                        &decoded_records,
                        &surviving_rows,
                        &final_row_digests[table_ref],
                        bases,
                        &shared_initial_foreign_indexes,
                    )?
                } else {
                    IndexedS7Closure::from_sealed_records(
                        &inputs[table_ref],
                        &decoded_records,
                        &surviving_rows,
                        &final_row_digests[table_ref],
                        bases,
                    )?
                };
                let effect_count = u32::try_from(table_closure.effects.len())
                    .map_err(|_| error("S7 effect count exceeds u32"))?;
                table_s7_ranges[table_ref] = IndexedS7TableRange {
                    owned_index_start: table_closure.owned_index_start,
                    owned_index_count: table_closure.owned_index_count,
                    effect_start,
                    effect_count,
                };
                shared_initial_foreign_indexes.extend(shared_initial_foreign_descriptors(
                    &inputs[table_ref],
                    &table_closure,
                )?);
                closure.dependencies.extend(table_closure.dependencies);
                closure
                    .dependency_uses
                    .extend(table_closure.dependency_uses);
                closure.indexes.extend(table_closure.indexes);
                closure.index_digests.extend(table_closure.index_digests);
                closure.index_keys.extend(table_closure.index_keys);
                closure.transitions.extend(table_closure.transitions);
                closure.effects.extend(table_closure.effects);
                closure.effect_digests.extend(table_closure.effect_digests);
                closure.components.extend(table_closure.components);
                closure.values.extend(table_closure.values);
            } else {
                table_s7_ranges[table_ref] = IndexedS7TableRange {
                    owned_index_start: u32::try_from(closure.indexes.len())
                        .map_err(|_| error("S7 owned-index lower bound exceeds u32"))?,
                    effect_start,
                    ..IndexedS7TableRange::default()
                };
                for row in &surviving_rows {
                    let statement = statements[row.statement_index].statement;
                    let image_row = row.image_row.expect("filtered surviving row has an image");
                    let final_row_digest = final_row_digests[table_ref][image_row as usize];
                    let digest =
                        if statement_count == 1 && row.final_writer_statement_digest == [0; 32] {
                            device_transition_digests[table_ref][image_row as usize]
                        } else {
                            transition_digest(
                                row.transition_ref,
                                row.table_ref,
                                row.stable_row_id,
                                row.s4_ref,
                                statement.statement_ordinal,
                                row.source_row,
                                row.table_ref,
                                image_row,
                                effect_start,
                                0,
                                row.final_writer_statement_ordinal,
                                statement.typed_statement_digest,
                                final_row_digest,
                                row.final_writer_statement_digest,
                            )
                        };
                    closure.transitions.push(transition_bytes(
                        row.transition_ref,
                        row.table_ref,
                        row.stable_row_id,
                        row.s4_ref,
                        statement.statement_ordinal,
                        row.source_row,
                        row.table_ref,
                        image_row,
                        effect_start,
                        0,
                        row.final_writer_statement_ordinal,
                        statement.typed_statement_digest,
                        final_row_digest,
                        digest,
                        row.final_writer_statement_digest,
                    ));
                }
            }
        }
        closure
    };
    let first_domain_dependency_ref = table_count
        .checked_add(
            u32::try_from(indexed.dependencies.len())
                .map_err(|_| error("S7 dependency count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 dependency count exceeds u32"))?;
    let mut domains = domain_closure::DomainClosure {
        dependencies: Vec::new(),
        dependency_uses: Vec::new(),
    };
    if !feature_free_live_closure {
        for (table_ref, input) in inputs.iter().enumerate() {
            let table_first_dependency_ref = first_domain_dependency_ref
                .checked_add(
                    u32::try_from(domains.dependencies.len())
                        .map_err(|_| error("S7 domain dependency count exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 domain dependency reference overflows"))?;
            let table_domains = domain_closure::from_sealed_records(
                input,
                &decoded_records,
                u32::try_from(table_ref).map_err(|_| error("S7 table reference exceeds u32"))?,
                table_first_dependency_ref,
            )?;
            domains.dependencies.extend(table_domains.dependencies);
            domains
                .dependency_uses
                .extend(table_domains.dependency_uses);
        }
    }
    let first_sequence_dependency_ref = first_domain_dependency_ref
        .checked_add(
            u32::try_from(domains.dependencies.len())
                .map_err(|_| error("S7 dependency count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 dependency count exceeds u32"))?;
    let sequences = if feature_free_live_closure {
        PublishedSequenceClosure {
            s5: Vec::new(),
            statement_ranges: vec![0..0; statements.len()],
            published_count: 0,
            private_count: 0,
            dependencies: Vec::new(),
            dependency_uses: Vec::new(),
        }
    } else {
        PublishedSequenceClosure::from_sealed_records(
            inputs,
            &statements,
            &decoded_records,
            &statement_row_starts,
            &statement_table_row_starts,
            &statement_row_counts,
            &transition_starts,
            first_sequence_dependency_ref,
            terminal_restarts,
        )?
    };
    let sequence_count = sequences
        .published_count
        .checked_add(sequences.private_count)
        .ok_or_else(|| error("S5 sequence count overflows"))?;
    let mut dependencies = Vec::with_capacity(
        inputs.len()
            + indexed.dependencies.len()
            + domains.dependencies.len()
            + sequences.dependencies.len(),
    );
    for (table_ref, input) in inputs.iter().enumerate() {
        dependencies.push(target_dependency(
            input,
            u32::try_from(table_ref).map_err(|_| error("target dependency ref exceeds u32"))?,
        ));
    }
    dependencies.extend_from_slice(&indexed.dependencies);
    dependencies.extend_from_slice(&domains.dependencies);
    dependencies.extend_from_slice(&sequences.dependencies);
    let mut dependency_uses = Vec::with_capacity(
        statements.len()
            + indexed.dependency_uses.len()
            + domains.dependency_uses.len()
            + sequences.dependency_uses.len(),
    );
    for bound in &statements {
        dependency_uses.push(target_dependency_use(
            bound.statement.statement_ordinal,
            bound.table_ref,
        ));
    }
    dependency_uses.extend_from_slice(&indexed.dependency_uses);
    dependency_uses.extend_from_slice(&domains.dependency_uses);
    dependency_uses.extend_from_slice(&sequences.dependency_uses);
    let target_dependency_refs = canonicalize_dependency_references(
        &mut dependencies,
        &mut dependency_uses,
        &mut indexed.effects,
        &indexed.indexes,
    )?;
    dependency_uses.sort_unstable_by_key(|usage| {
        (
            u32::from_le_bytes(usage[0..4].try_into().expect("fixed statement ordinal")),
            u16::from_le_bytes(usage[8..10].try_into().expect("fixed dependency role")),
            u32::from_le_bytes(usage[12..16].try_into().expect("fixed source ordinal")),
            u32::from_le_bytes(usage[16..20].try_into().expect("fixed transition ref")),
            u32::from_le_bytes(usage[20..24].try_into().expect("fixed key-effect ref")),
            u32::from_le_bytes(usage[4..8].try_into().expect("fixed dependency ref")),
        )
    });

    let mut s1 = Vec::with_capacity(statements.len() * S1_BYTES);
    let mut s2 = Vec::new();
    let mut s6 = Vec::with_capacity(statements.len() * S6_BYTES);
    let mut resolutions = Vec::with_capacity(statements.len() * 320);
    let mut all_projections = Vec::new();
    let mut overlay_before = initial_overlay;
    let mut dependency_use_start = 0_u32;
    let mut sequence_start = 0_u32;
    let mut projection_start = 0_u32;
    for (statement_index, bound) in statements.iter().enumerate() {
        let statement = bound.statement;
        let ordinal = statement.statement_ordinal;
        let statement_rows = statement_row_counts[statement_index];
        let row_start = statement_row_starts[statement_index];
        let record_digest = v2_digest(
            b"gpu-db/write001/s7-s2-record/v2",
            &[
                &record_bytes[statement_index].to_le_bytes(),
                statement.record,
            ],
        );
        let statement_s4_start = row_start as usize * S4_BYTES;
        let statement_s4_end = statement_s4_start + statement_rows as usize * S4_BYTES;
        let surviving_rows = s4[statement_s4_start..statement_s4_end]
            .chunks_exact(S4_BYTES)
            .filter(|row| row[16] == 1)
            .count();
        let surviving_rows = u32::try_from(surviving_rows)
            .map_err(|_| error("statement survivor count exceeds u32"))?;
        let disposition_root = v2_digest(
            b"gpu-db/write001/s7-statement-dispositions/v2",
            &[
                &ordinal.to_le_bytes(),
                &statement_rows.to_le_bytes(),
                &s4[statement_s4_start..statement_s4_end],
            ],
        );
        let statement_sequence_count = if feature_free_live_closure {
            0
        } else {
            decoded_records[statement_index]
                .facts()
                .sequence_effect_count
        };
        let statement_sequence_range = sequences
            .statement_ranges
            .get(statement_index)
            .ok_or_else(|| error("statement has no S5 byte range"))?;
        let statement_sequence_bytes = sequences
            .s5
            .get(statement_sequence_range.clone())
            .ok_or_else(|| error("statement S5 range exceeds the sequence directory"))?;
        let sequence_root = v2_digest(
            b"gpu-db/write001/s7-statement-sequences/v2",
            &[
                &ordinal.to_le_bytes(),
                &statement_sequence_count.to_le_bytes(),
                statement_sequence_bytes,
            ],
        );
        let statement_uses = dependency_uses
            .iter()
            .copied()
            .filter(|usage| u32::from_le_bytes(usage[..4].try_into().unwrap()) == ordinal)
            .collect::<Vec<_>>();
        let statement_dependency_count = u32::try_from(statement_uses.len())
            .map_err(|_| error("statement dependency-use count exceeds u32"))?;
        let dependency_root = statement_dependency_root(ordinal, &dependencies, &statement_uses)?;
        let returning = &returnings[statement_index];
        let statement_projection_count = u32::try_from(returning.projections.len())
            .map_err(|_| error("statement projection count exceeds u32"))?;
        let projection_root = statement_projection_root(ordinal, &returning.projections)?;
        let overlay_after = v2_digest(
            b"gpu-db/write001/s7-statement-overlay-root/v2",
            &[
                &overlay_before,
                &ordinal.to_le_bytes(),
                &statement.typed_statement_digest,
                &record_digest,
                &disposition_root,
                &sequence_root,
                &dependency_root,
                &projection_root,
            ],
        );

        let mut s1_entry = [0_u8; S1_BYTES];
        put_u32(&mut s1_entry, 0, ordinal);
        put_u32(&mut s1_entry, 4, ordinal);
        s1_entry[8] = 1;
        put_u32(&mut s1_entry, 12, statement_rows);
        put_digest(&mut s1_entry, 16, statement.typed_statement_digest);
        put_digest(&mut s1_entry, 48, statement.typed_statement_digest);
        put_digest(&mut s1_entry, 80, overlay_before);
        put_digest(&mut s1_entry, 112, overlay_after);
        s1.extend_from_slice(&s1_entry);
        s2.extend_from_slice(&record_bytes[statement_index].to_le_bytes());
        s2.extend_from_slice(statement.record);

        let statement_has_returning = statement_projection_count != 0;
        let statement_outcome = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows: u64::from(statement_rows),
            sqlstate: None,
            constraint_id: 0,
            target_digest: overlay_after,
            returning_digest: returning.logical_result_digest,
        };
        let mut s6_entry = [0_u8; S6_BYTES];
        put_u32(&mut s6_entry, 0, ordinal);
        put_u32(&mut s6_entry, 4, ordinal);
        put_u16(&mut s6_entry, 8, 1);
        put_u16(&mut s6_entry, 10, u16::from(statement_has_returning));
        put_digest(&mut s6_entry, 12, statement.typed_statement_digest);
        let encoded_outcome: &mut [u8; gpu_db_wal::CANONICAL_OUTCOME_BYTES] = (&mut s6_entry
            [44..136])
            .try_into()
            .expect("S6 outcome slot has fixed width");
        gpu_db_wal::encode_canonical_outcome_into_exact(&statement_outcome, encoded_outcome)?;
        let s6_digest = v2_digest(b"gpu-db/write001/s7-s6-entry/v2", &[&s6_entry]);
        s6.extend_from_slice(&s6_entry);

        let mut resolution = [0_u8; 320];
        put_u32(&mut resolution, 0, ordinal);
        put_u32(&mut resolution, 4, ordinal);
        put_u32(&mut resolution, 8, ordinal);
        put_u32(&mut resolution, 12, ordinal);
        put_u32(&mut resolution, 16, bound.table_ref);
        put_u32(&mut resolution, 20, u32::from(statement_has_returning));
        put_u32(&mut resolution, 24, row_start);
        put_u32(&mut resolution, 28, statement_rows);
        put_u32(&mut resolution, 32, sequence_start);
        put_u32(&mut resolution, 36, statement_sequence_count);
        put_u32(&mut resolution, 40, dependency_use_start);
        put_u32(&mut resolution, 44, statement_dependency_count);
        put_u32(&mut resolution, 48, projection_start);
        put_u32(&mut resolution, 52, statement_projection_count);
        put_u32(&mut resolution, 56, statement_rows);
        put_u32(&mut resolution, 60, surviving_rows);
        put_u64(&mut resolution, 64, u64::from(statement_rows));
        put_u64(&mut resolution, 72, identity.dependency_validation_floor);
        put_u32(&mut resolution, 80, record_bytes[statement_index]);
        put_u32(&mut resolution, 84, ABSENT_U32);
        put_u32(&mut resolution, 88, ABSENT_U32);
        put_u32(&mut resolution, 92, ABSENT_U32);
        put_digest(&mut resolution, 96, statement.typed_statement_digest);
        put_digest(&mut resolution, 128, statement.typed_statement_digest);
        put_digest(&mut resolution, 160, record_digest);
        put_digest(
            &mut resolution,
            192,
            record_prefixes[statement_index].returning_digest,
        );
        put_digest(&mut resolution, 224, overlay_before);
        put_digest(&mut resolution, 256, overlay_after);
        put_digest(&mut resolution, 288, s6_digest);
        resolutions.extend_from_slice(&resolution);
        all_projections.extend_from_slice(&returning.projections);
        dependency_use_start = dependency_use_start
            .checked_add(statement_dependency_count)
            .ok_or_else(|| error("dependency-use range overflows"))?;
        sequence_start = sequence_start
            .checked_add(statement_sequence_count)
            .ok_or_else(|| error("sequence range overflows"))?;
        projection_start = projection_start
            .checked_add(statement_projection_count)
            .ok_or_else(|| error("projection range overflows"))?;
        overlay_before = overlay_after;
    }
    let final_overlay = overlay_before;

    let mut image_offset = 0_u64;
    let mut image_descriptors = Vec::with_capacity(inputs.len());
    for (image_ref, input) in inputs.iter().enumerate() {
        image_descriptors.push(image_descriptor(
            input,
            u32::try_from(image_ref).map_err(|_| error("image ref exceeds u32"))?,
            image_offset,
            input.table.image_content_digest,
        )?);
        image_offset = image_offset
            .checked_add(
                u64::try_from(input.final_image.len())
                    .map_err(|_| error("final image length exceeds u64"))?,
            )
            .ok_or_else(|| error("final image arena overflows"))?;
    }
    let mut tables = Vec::with_capacity(inputs.len());
    let mut disposition_start = 0_u32;
    for (table_ref, input) in inputs.iter().enumerate() {
        let table_ref = u32::try_from(table_ref).map_err(|_| error("table ref exceeds u32"))?;
        let rows = table_rows_seen[table_ref as usize];
        let surviving_rows = u32::try_from(
            input
                .final_writers
                .iter()
                .filter(|writer| writer.survives)
                .count(),
        )
        .map_err(|_| error("table survivor count exceeds u32"))?;
        let transition_start = transition_starts[table_ref as usize];
        let transition_end = transition_start
            .checked_add(surviving_rows)
            .ok_or_else(|| error("table transition range overflows"))?;
        let transitions = indexed
            .transitions
            .get(transition_start as usize..transition_end as usize)
            .ok_or_else(|| error("table transition range is absent"))?;
        let IndexedS7TableRange {
            owned_index_start,
            owned_index_count,
            effect_start,
            effect_count,
        } = table_s7_ranges[table_ref as usize];
        let transition_root = digest_transition_records(
            b"gpu-db/write001/s7-table-transition-root/v2",
            input.table.stable_table_id,
            transitions,
        )?;
        let effect_digests = indexed
            .effect_digests
            .get(effect_start as usize..(effect_start + effect_count) as usize)
            .ok_or_else(|| error("table effect digest range is absent"))?;
        let index_effect_root = digest_list(
            b"gpu-db/write001/s7-table-index-effect-root/v2",
            input.table.stable_table_id,
            effect_digests,
        );
        let target_dependency_ref = *target_dependency_refs
            .get(table_ref as usize)
            .ok_or_else(|| error("S7 target dependency remap is absent"))?;
        let mut table = table_block(
            input,
            table_ref,
            target_dependency_ref,
            rows,
            surviving_rows,
            disposition_start,
            transition_start,
            owned_index_start,
            owned_index_count,
            effect_start,
            effect_count,
            transition_root,
            index_effect_root,
            input.table.image_content_digest,
        );
        let disposition_end = disposition_start
            .checked_add(rows)
            .ok_or_else(|| error("table disposition range overflows"))?;
        let dispositions = table_dispositions
            .get(disposition_start as usize * 32..disposition_end as usize * 32)
            .ok_or_else(|| error("table disposition range is absent"))?;
        let index_digests = indexed
            .index_digests
            .get(
                owned_index_start as usize
                    ..owned_index_start
                        .checked_add(owned_index_count)
                        .ok_or_else(|| error("S7 owned-index range overflows"))?
                        as usize,
            )
            .ok_or_else(|| error("S7 owned-index range is absent"))?;
        let target_digest: [u8; 32] = dependencies[target_dependency_ref as usize][192..224]
            .try_into()
            .expect("target dependency digest has fixed width");
        let image_descriptor_digest: [u8; 32] = image_descriptors[table_ref as usize][128..160]
            .try_into()
            .expect("image descriptor digest has fixed width");
        let manifest = table_manifest_digest(
            &table,
            target_digest,
            dispositions,
            index_digests,
            transitions,
            effect_digests,
            image_descriptor_digest,
        )?;
        put_digest(&mut table, 352, manifest);
        tables.push(table);
        disposition_start = disposition_end;
    }
    let root_descriptor = root_descriptor(
        identity,
        initial_database_root,
        final_database_root,
        initial_overlay,
        final_overlay,
        &tables,
    );
    let counts = [
        table_count,
        statement_count,
        row_count,
        u32::try_from(dependencies.len()).map_err(|_| error("S7 dependency count exceeds u32"))?,
        u32::try_from(dependency_uses.len())
            .map_err(|_| error("S7 dependency-use count exceeds u32"))?,
        u32::try_from(indexed.indexes.len()).map_err(|_| error("S7 index count exceeds u32"))?,
        u32::try_from(indexed.index_keys.len())
            .map_err(|_| error("S7 index-key count exceeds u32"))?,
        u32::try_from(indexed.transitions.len())
            .map_err(|_| error("S7 transition count exceeds u32"))?,
        u32::try_from(indexed.effects.len()).map_err(|_| error("S7 effect count exceeds u32"))?,
        u32::try_from(indexed.components.len())
            .map_err(|_| error("S7 component count exceeds u32"))?,
        projection_count,
        table_count,
    ];
    let s7 = encode_s7(
        identity,
        initial_database_root,
        final_database_root,
        counts,
        root_descriptor,
        initial_overlay,
        final_overlay,
        &tables,
        &table_dispositions,
        &resolutions,
        &dependencies,
        &dependency_uses,
        &indexed.indexes,
        &indexed.index_keys,
        &indexed.transitions,
        &indexed.effects,
        &indexed.components,
        &all_projections,
        &indexed.values,
        &image_descriptors,
        &inputs
            .iter()
            .map(|input| input.final_image)
            .collect::<Vec<_>>(),
    )?;
    let has_operation_composition = operation_body.is_some();
    if operation_changes_catalog && !has_operation_composition {
        return Err(error(
            "catalog-changing codec-5 operation has no S3 composition body",
        ));
    }
    let has_catalog = operation_changes_catalog;
    let expected_catalog_after_epoch = if has_catalog {
        identity
            .catalog_epoch
            .checked_add(1)
            .ok_or_else(|| error("catalog epoch overflows"))?
    } else {
        identity.catalog_epoch
    };
    let expected_catalog_after_digest = if has_catalog {
        let body = operation_body.expect("catalog-changing operation owns its S3 body");
        crate::Engine::canonical_catalog_transition(
            identity.catalog_digest,
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            body,
        )
    } else {
        identity.catalog_digest
    };
    if identity.catalog_after_epoch != expected_catalog_after_epoch
        || identity.catalog_after_digest != expected_catalog_after_digest
    {
        return Err(error(
            "catalog section does not match the canonical catalog boundary transition",
        ));
    }
    let sections = [
        s1,
        s2,
        operation_body.map_or_else(Vec::new, <[u8]>::to_vec),
        s4,
        sequences.s5,
        s6,
        s7,
        Vec::new(),
    ];
    let request_digest = identity.request_digest;
    let section_views: [TypedInsertAggregateSectionView<'_>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: [
                statement_count,
                statement_count,
                u32::from(has_operation_composition),
                row_count,
                sequence_count,
                statement_count,
                1,
                0,
            ][index],
            payload: &sections[index],
        });
    let has_published_sequences = sequences.published_count != 0;
    let has_private_sequences = sequences.private_count != 0;
    let aggregate = TypedInsertAggregateView {
        semantics: TypedInsertAggregateSemantics::V2,
        flags: identity.mode.aggregate_flag()
            | if has_catalog {
                AGGREGATE_FLAG_CATALOG
            } else {
                0
            }
            | if has_operation_composition {
                AGGREGATE_FLAG_OPERATION_COMPOSITION
            } else {
                0
            }
            | if has_published_sequences {
                AGGREGATE_FLAG_PUBLISHED_SEQUENCE
            } else {
                0
            }
            | if has_private_sequences {
                AGGREGATE_FLAG_PRIVATE_SEQUENCE
            } else {
                0
            }
            | if has_returning {
                AGGREGATE_FLAG_RETURNING
            } else {
                0
            },
        // Bit 30 authenticated the retired three-record writer protocol. Clearing it selects
        // the generic one-record codec-5 recovery arm; bit 31 retains codec-5 framing.
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1
            | OUTER_CONTENT_ROW
            | if has_catalog {
                OUTER_CONTENT_CATALOG
            } else {
                0
            }
            | if has_operation_composition {
                OUTER_CONTENT_OPERATION_COMPOSITION
            } else {
                0
            }
            | if has_published_sequences {
                OUTER_CONTENT_PUBLISHED_SEQUENCE
            } else {
                0
            }
            | if has_private_sequences {
                OUTER_CONTENT_PRIVATE_SEQUENCE
            } else {
                0
            }
            | if has_returning {
                OUTER_CONTENT_RETURNING
            } else {
                0
            },
        stable_transaction_id: identity.stable_transaction_id,
        statement_count,
        insert_statement_count: statement_count,
        original_inserted_row_count: u64::from(row_count),
        final_row_transition_count: u64::from(next_transition),
        allocator_before: 0,
        allocator_high_water: 0,
        table_block_count: table_count,
        sections: section_views,
    };
    #[cfg(feature = "probe-timing")]
    let probe_s7_build_nanos = probe_s7_build_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_prepare_started = std::time::Instant::now();
    let (layout, prepared) = measure_and_prepare_typed_insert_aggregate_encoding(aggregate)?;
    let roots = prepared.roots();
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_prepare_nanos = probe_aggregate_prepare_started.elapsed().as_nanos() as u64;
    let status = TypedInsertStatusV2 {
        database_id: identity.canonical.database_id,
        timeline_id: identity.canonical.timeline_id,
        txn_id: identity.stable_transaction_id,
        request_digest,
        isolation: identity.isolation as u8,
        flags: 0,
        retention_deadline: 0,
        statement_count,
        response_artifact_count: 0,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    };
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_encode_started = std::time::Instant::now();
    let bodies = prepared.encode(&status, reserve_typed_insert_aggregate_bodies(layout)?)?;
    #[cfg(feature = "probe-timing")]
    let probe_aggregate_encode_nanos = probe_aggregate_encode_started.elapsed().as_nanos() as u64;
    let header = gpu_db_wal::CanonicalPreApplyHeader {
        identity: identity.canonical,
        leader_epoch: identity.leader_epoch,
        commit_seq: identity.commit_sequence,
        stable_transaction_id: identity.stable_transaction_id,
        request_digest,
        isolation: identity.isolation,
        flags: aggregate.outer_flags,
        catalog_before_epoch: identity.catalog_epoch,
        catalog_after_epoch: identity.catalog_after_epoch,
        catalog_before_digest: identity.catalog_digest,
        catalog_after_digest: identity.catalog_after_digest,
        operation_count: bodies.layout().fragment_count,
        table_block_count: table_count,
        allocator_high_water: 0,
    };
    let outcome = gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
        affected_rows: u64::from(row_count),
        sqlstate: None,
        constraint_id: 0,
        target_digest: roots.aggregate_root,
        returning_digest: roots.response_root,
    };
    #[cfg(feature = "probe-timing")]
    let probe_outer_reserve_started = std::time::Instant::now();
    let reserved =
        reserve_typed_insert_canonical_envelope(bodies, identity.physical, header, outcome)?;
    #[cfg(feature = "probe-timing")]
    let probe_outer_reserve_nanos = probe_outer_reserve_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_outer_encode_started = std::time::Instant::now();
    let envelope = encode_reserved_typed_insert_canonical_envelope(reserved)?;
    #[cfg(feature = "probe-timing")]
    let probe_outer_encode_nanos = probe_outer_encode_started.elapsed().as_nanos() as u64;
    #[cfg(feature = "probe-timing")]
    let probe_outer_phase_nanos = envelope.encoding().probe_timing_nanos;
    Ok(ClosedLiveTypedInsert {
        envelope,
        #[cfg(feature = "probe-timing")]
        probe_timing_nanos: [
            probe_s7_build_nanos,
            probe_s2_decode_nanos,
            probe_aggregate_prepare_nanos,
            probe_aggregate_encode_nanos,
            probe_outer_reserve_nanos,
            probe_outer_encode_nanos,
            probe_outer_phase_nanos[0],
            probe_outer_phase_nanos[1],
            probe_outer_phase_nanos[2],
            probe_outer_phase_nanos[3],
            probe_outer_phase_nanos[4],
        ],
    })
}

#[cfg(test)]
#[path = "writer/tests.rs"]
mod tests;
