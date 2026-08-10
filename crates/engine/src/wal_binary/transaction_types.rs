//! Binary transaction opcodes and durable record vocabulary.

use super::*;

/// Binary-record lead byte (invalid UTF-8 on purpose — see the parent module documentation).
pub(crate) const WAL_BINARY_TAG: u8 = 0xFF;
/// Format version for forward evolution; bump on layout change.
pub(super) const WAL_BINARY_VERSION: u8 = 1;
/// Op codes.
pub(super) const OP_INSERT: u8 = 1;
/// W5b: a covered lane DELETE, logged BY KEY — replay re-resolves the key against the replayed
/// state (deterministic: all ops on a key are lane-serialized in seq order, cross-key ops
/// commute). WAL-FIRST: a 0-row delete DOES reach the WAL (the locate moved to apply), and its
/// replay re-resolve to 0 rows is a legal no-op (not corruption).
pub(super) const OP_DELETE_BY_KEY: u8 = 2;
/// U2 (W5b): a covered lane UPDATE, logged BY KEY + the new row image + the new version's row id.
/// Replay re-resolves the key: a visible old version → tombstone it + append the new image at
/// the old version's stable entity id. The legacy `new_row_id` field remains a consumed allocator
/// reservation so v1 logs and allocator high-water replay stay compatible; it is not replacement identity.
pub(super) const OP_UPDATE_BY_KEY: u8 = 3;
/// R3-003: one explicit transaction's ordered, resolved row mutations. Every operation carries
/// stable entity identity plus the row image(s), so replay never re-evaluates SQL predicates.
pub(super) const OP_TRANSACTION: u8 = 4;
/// PRODUCT-001: the same resolved explicit transaction, prefixed by typed catalog mutations. A
/// distinct opcode preserves the exact v1 row-only layout and lets old durable records replay.
pub(super) const OP_COMPOSITE_TRANSACTION: u8 = 5;
/// PRODUCT-001: a resolved transaction containing one or more typed table-root resets. Its
/// reset-prefixed layout is distinct so opcodes 4/5 retain byte-for-byte compatibility.
pub(super) const OP_TABLE_RESET_TRANSACTION: u8 = 6;
/// ADR-014: current row transactions bind every mutation table to stable OID + schema identity.
/// Separate opcodes preserve byte-for-byte replay of legacy name-bound transaction records.
pub(super) const OP_IDENTITY_TRANSACTION: u8 = 7;
pub(super) const OP_IDENTITY_COMPOSITE_TRANSACTION: u8 = 8;
pub(super) const OP_IDENTITY_TABLE_RESET_TRANSACTION: u8 = 9;
/// PRODUCT-001 ordered catalog envelope. Unlike opcodes 5/8, this layout carries the global
/// statement ordinal of every catalog operation, its complete created-table identity set, and an
/// optional ordered table-reset block. Old composite records remain byte-for-byte decodable.
pub(super) const OP_ORDERED_CATALOG_TRANSACTION: u8 = 10;
pub(super) const OP_IDENTITY_ORDERED_CATALOG_TRANSACTION: u8 = 11;
/// PRODUCT-001 transactional view envelope. Opcodes 10/11 remain byte-for-byte CREATE-TABLE-only;
/// these additive variants carry one exact preimage/dependency/postimage proof per CREATE VIEW.
pub(super) const OP_ORDERED_CATALOG_VIEW_TRANSACTION: u8 = 12;
pub(super) const OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION: u8 = 13;
/// PRODUCT-001 complete stored-view lifecycle. Opcodes 12/13 remain byte-for-byte CREATE-only;
/// these variants type CREATE/RENAME/multi-target DROP preimages, dependencies, and postimages.
pub(super) const OP_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION: u8 = 14;
pub(super) const OP_IDENTITY_ORDERED_CATALOG_VIEW_LIFECYCLE_TRANSACTION: u8 = 15;

pub(super) const TXN_INSERT: u8 = 1;
pub(super) const TXN_UPDATE: u8 = 2;
pub(super) const TXN_DELETE: u8 = 3;

pub(super) const TXN_OPERATION_CATALOG: u8 = 1;
pub(super) const TXN_OPERATION_INSERT: u8 = 2;
pub(super) const TXN_OPERATION_UPDATE: u8 = 3;
pub(super) const TXN_OPERATION_DELETE: u8 = 4;
pub(super) const TXN_OPERATION_TABLE_RESET: u8 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BinaryTransactionMutation {
    Insert {
        table: String,
        row_id: u64,
        row_encoded: String,
    },
    Update {
        table: String,
        row_id: u64,
        old_row_encoded: String,
        new_row_encoded: String,
    },
    Delete {
        table: String,
        row_id: u64,
        old_row_encoded: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionTableReset {
    pub(crate) ordinal: u32,
    pub(crate) table: String,
    pub(crate) table_oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
    /// Last canonical publication that changed this table before the root replacement.
    pub(crate) source_commit_seq: Index,
    pub(crate) before_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) expected_rows: u64,
    pub(crate) after_empty_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) dependency_identities: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionTableIdentity {
    pub(crate) table_oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

/// A durable user-envelope reference to an already-published ordinary sequence transition.
///
/// The value transition has its own canonical transaction identity and WAL record.  Keeping the
/// exact returned value and input digest here makes the later INSERT envelope self-verifying
/// without asking replay to re-evaluate `nextval`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinarySequenceValueReference {
    pub(crate) transition_txn_id: TxnId,
    pub(crate) parent_txn_id: TxnId,
    pub(crate) statement_ordinal: u32,
    pub(crate) expression_ordinal: u32,
    pub(crate) sequence_oid: u32,
    pub(crate) returned_value: i64,
    pub(crate) input_digest: gpu_db_wal::CanonicalDigest,
    /// Stable relation/column binding for a materialized default. Explicit sequence calls encode
    /// zero in both fields; their durable `row_id` and transient staging ordinal are also zero.
    pub(crate) table_oid: u32,
    pub(crate) column_id: u32,
    /// Statement-local row position used only while binding the prepared INSERT entity. It is
    /// cleared before WAL framing and deliberately is not serialized; `row_id` is the durable
    /// identity.
    pub(crate) staging_row_ordinal: u32,
    /// Stable entity identity assigned by the enclosing user envelope. During statement staging
    /// this is the transaction-private provisional id; WAL binding rewrites it to the final
    /// globally claimed id. Explicit sequence calls encode zero.
    pub(crate) row_id: u64,
    /// Whether a later statement in the same transaction replaced the materialized default value
    /// or deleted its row. This preserves legal insert-then-update/delete programs while making
    /// the final durable row disposition independently checkable during replay.
    pub(crate) final_value_overwritten: bool,
    /// True when the returned value is embedded in an INSERT row. False identifies an explicit
    /// sequence call whose transition still belongs to this user transaction's retry/lifecycle
    /// closure but does not occupy an INSERT expression.
    pub(crate) default_expression: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionCatalogCommand {
    pub(crate) ordinal: u32,
    pub(crate) command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionCatalogOutput {
    /// Sole `pg_class.oid` high-water after applying the complete ordered catalog stream,
    /// including every implicit or explicit index.
    pub(crate) relational_next_oid: u32,
    pub(crate) relational_next_column_id: u32,
    /// Stable identities of implicit sequences created by admitted CREATE TABLE commands.
    pub(crate) created_sequence_oids: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryTransactionCatalogEpoch {
    /// Opcodes 4--15: indexes had no durable stable-OID/output closure.
    Legacy,
    /// Opcodes 16--19: exact shared pg_class/index identities are bound.
    IndexIdentityV1,
}

/// One durable explicit-transaction record. `allocator_high_water` is the row-id allocator value
/// after the transaction's insert identities were claimed. Apply uses an idempotent max operation,
/// so the live process (which preclaimed the ids before WAL encoding) and recovery converge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionRecord {
    /// Derived from the binary opcode on decode; live builders emit only `IndexIdentityV1` for a
    /// catalog-bearing record. This field is not separately serialized.
    pub(crate) catalog_epoch: BinaryTransactionCatalogEpoch,
    pub(crate) allocator_high_water: u64,
    /// Typed catalog operations and their positions in the complete transaction statement stream.
    /// The command vector itself is canonical statement order; ordinals bind its gaps to DML and
    /// table-reset operations without replaying SQL text.
    pub(crate) catalog_commands: Vec<BinaryTransactionCatalogCommand>,
    /// Stable output identity for every table created by `catalog_commands`. Current ordered
    /// records require exact coverage; legacy opcode 5/8 records decode this as empty.
    pub(crate) created_table_identities: BTreeMap<String, BinaryTransactionTableIdentity>,
    /// Exact ordered postimage of every implicit PRIMARY KEY / UNIQUE index created with each
    /// table. The table schema v1 digest intentionally predates stable index OIDs, so current
    /// opcodes bind these identities separately; an empty vector is itself an absence proof.
    pub(crate) created_table_index_identities: BTreeMap<String, Vec<BinaryCatalogIndexIdentity>>,
    /// Exact allocator post-state and implicit-sequence output closure for current ordered
    /// catalog records. Table schema identities alone do not carry generated sequence OIDs.
    pub(crate) catalog_output: Option<BinaryTransactionCatalogOutput>,
    /// Exact CREATE VIEW preimage, transitive source, and postimage closure in catalog-command
    /// order. Empty for every historical opcode and for CREATE-TABLE-only opcodes 10/11.
    pub(crate) view_operations: Vec<BinaryTransactionViewOperationIdentity>,
    /// Exact CREATE/RENAME/DROP VIEW lifecycle closure. Additive opcodes 14/15 use this field;
    /// opcodes 12/13 continue to decode only into `view_operations`.
    pub(crate) view_lifecycle_operations: Vec<BinaryTransactionViewLifecycleOperationIdentity>,
    /// Exact CREATE/RENAME/DROP INDEX target transitions. Empty for historical opcodes.
    pub(crate) index_lifecycle_operations: Vec<BinaryTransactionIndexLifecycleOperationIdentity>,
    /// Exact CREATE/RESTART/RENAME/DROP SEQUENCE target and column-default dependency transitions.
    /// Empty for opcodes 4--17.
    pub(crate) sequence_lifecycle_operations:
        Vec<BinaryTransactionSequenceLifecycleOperationIdentity>,
    pub(crate) sequence_reset_operations: Vec<BinaryTransactionSequenceResetOperationIdentity>,
    /// Final private value state keyed by stable sequence identity. Sequence-lifecycle opcodes use
    /// this instead of the legacy name-keyed map so rename and drop/recreate cannot redirect an
    /// earlier statement's value effect.
    pub(crate) sequence_advances_by_oid: BTreeMap<u32, (i64, bool)>,
    /// Complete statement order for current catalog-bearing records. Legacy transaction opcodes
    /// decode this as empty and retain their historical resolved-state semantics.
    pub(crate) operation_order: Vec<BinaryTransactionOperationIdentity>,
    /// Canonical typed/bound request identity at every position in `operation_order`. Effects may
    /// be coalesced or shadowed; request identity never is. Legacy opcodes decode this as empty.
    pub(crate) statement_digests: Vec<gpu_db_wal::CanonicalDigest>,
    /// Exact stable sequence identity consumed by an ordered catalog or INSERT statement, keyed
    /// by `(statement ordinal, sequence name)`. Catalog entries cover every sequence default;
    /// INSERT entries cover exactly the defaults that advanced `sequence_advances`.
    pub(crate) sequence_input_oids: BTreeMap<(u32, String), u32>,
    /// Ordinary published-sequence effects are separate durable transitions.  These references
    /// bind their already-materialized values into the user transaction without folding the
    /// sequence state into user rollback.
    pub(crate) sequence_value_references: Vec<BinarySequenceValueReference>,
    /// Surviving table-root barriers in canonical statement order. Old transaction opcodes decode
    /// this as empty; reset records use their own opcode and keep row bodies after the reset block.
    pub(crate) table_resets: Vec<BinaryTransactionTableReset>,
    /// Legacy final catalog post-state for each sequence consumed by a transaction-private
    /// default. Opcodes 4--17 retain this exact name-keyed layout; sequence-lifecycle opcodes
    /// require it to be empty and use `sequence_advances_by_oid`.
    pub(crate) sequence_advances: BTreeMap<String, (i64, bool)>,
    /// Stable binding for every table named by `mutations`. Empty only for legacy opcodes 4/5/6.
    pub(crate) table_identities: BTreeMap<String, BinaryTransactionTableIdentity>,
    pub(crate) mutations: Vec<BinaryTransactionMutation>,
}

impl BinaryTransactionRecord {
    /// Test fixture seed for an otherwise empty S3 catalog composition. Production composition
    /// now starts from the resolved pre-existing-row operation so it cannot accidentally mint a
    /// second empty authority.
    #[cfg(test)]
    pub(crate) fn catalog_composition_seed() -> Self {
        Self {
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water: 0,
            catalog_commands: Vec::new(),
            created_table_identities: BTreeMap::new(),
            created_table_index_identities: BTreeMap::new(),
            catalog_output: None,
            view_operations: Vec::new(),
            view_lifecycle_operations: Vec::new(),
            index_lifecycle_operations: Vec::new(),
            sequence_lifecycle_operations: Vec::new(),
            sequence_reset_operations: Vec::new(),
            sequence_advances_by_oid: BTreeMap::new(),
            operation_order: Vec::new(),
            statement_digests: Vec::new(),
            sequence_input_oids: BTreeMap::new(),
            sequence_value_references: Vec::new(),
            table_resets: Vec::new(),
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::new(),
            mutations: Vec::new(),
        }
    }
}

/// A decoded binary WAL record of any op (the tag dispatch for apply/replay consumers).
// Keep the transaction record inline: live apply and replay immediately move it into the sole
// transaction consumer, while boxing would add an allocation to every explicit transaction only
// to shrink this transient decode dispatch enum.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BinaryWalRecord {
    Insert(BinaryInsertRecord),
    DeleteByKey(BinaryDeleteByKeyRecord),
    UpdateByKey(BinaryUpdateByKeyRecord),
    Transaction(BinaryTransactionRecord),
    SequenceValueTransition(BinarySequenceValueTransitionRecord),
}
