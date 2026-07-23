//! W5a — BINARY row-op WAL records (the resolved-change-record format, assessment D5/R3).
//!
//! The WAL was a logical SQL-text log: replay re-parsed and re-executed every statement
//! (~30µs/record), the sequencer's covered INSERTs re-serialized their statement text, and every
//! checkpoint carried the full SQL. A binary record carries the RESOLVED mutation instead — the
//! table, the row ids, and the row images in the tuple store's canonical cell encoding — so
//! replay is decode + install (no parse, no re-resolve, no validation re-run: the record exists
//! only because the original commit validated it).
//!
//! FRAMING: the record rides the existing WAL framing (txn_id / len / FNV checksum) as an opaque
//! payload. `payload[0] == 0xFF` marks a binary record — 0xFF is an invalid UTF-8 leading byte,
//! so every SQL-text consumer's defensive `from_utf8 -> skip` arm (KvStateMachine, telemetry)
//! ignores binary records WITHOUT modification; the consumers that must apply them
//! (`apply_mvcc_entry`, `residency_invalidation_scope`) dispatch on the tag explicitly BEFORE
//! their UTF-8 checks. SQL text can never collide (it starts with printable ASCII).
//!
//! v1 (W5a) began with the covered single-table INSERT class; W5b added UPDATE/DELETE, and R3-003
//! added one resolved explicit-transaction operation carrying ordered row mutations plus atomic
//! sequence post-state. PRODUCT-001 extends that same operation with typed transaction-owned
//! catalog mutations; old row-only records remain byte-for-byte v1 compatible. Unsupported
//! autocommit shapes remain SQL text.

use super::*;

/// Binary-record lead byte (invalid UTF-8 on purpose — see the module doc).
pub(crate) const WAL_BINARY_TAG: u8 = 0xFF;
/// Format version for forward evolution; bump on layout change.
const WAL_BINARY_VERSION: u8 = 1;
/// Op codes.
const OP_INSERT: u8 = 1;
/// W5b: a covered lane DELETE, logged BY KEY — replay re-resolves the key against the replayed
/// state (deterministic: all ops on a key are lane-serialized in seq order, cross-key ops
/// commute). WAL-FIRST: a 0-row delete DOES reach the WAL (the locate moved to apply), and its
/// replay re-resolve to 0 rows is a legal no-op (not corruption).
const OP_DELETE_BY_KEY: u8 = 2;
/// U2 (W5b): a covered lane UPDATE, logged BY KEY + the new row image + the new version's row id.
/// Replay re-resolves the key: a visible old version → tombstone it + append the new image at
/// the old version's stable entity id. The legacy `new_row_id` field remains a consumed allocator
/// reservation so v1 logs and allocator high-water replay stay compatible; it is not replacement identity.
const OP_UPDATE_BY_KEY: u8 = 3;
/// R3-003: one explicit transaction's ordered, resolved row mutations. Every operation carries
/// stable entity identity plus the row image(s), so replay never re-evaluates SQL predicates.
const OP_TRANSACTION: u8 = 4;
/// PRODUCT-001: the same resolved explicit transaction, prefixed by typed catalog mutations. A
/// distinct opcode preserves the exact v1 row-only layout and lets old durable records replay.
const OP_COMPOSITE_TRANSACTION: u8 = 5;
/// PRODUCT-001: a resolved transaction containing one or more typed table-root resets. Its
/// reset-prefixed layout is distinct so opcodes 4/5 retain byte-for-byte compatibility.
const OP_TABLE_RESET_TRANSACTION: u8 = 6;
/// ADR-014: current row transactions bind every mutation table to stable OID + schema identity.
/// Separate opcodes preserve byte-for-byte replay of legacy name-bound transaction records.
const OP_IDENTITY_TRANSACTION: u8 = 7;
const OP_IDENTITY_COMPOSITE_TRANSACTION: u8 = 8;
const OP_IDENTITY_TABLE_RESET_TRANSACTION: u8 = 9;
/// PRODUCT-001 ordered catalog envelope. Unlike opcodes 5/8, this layout carries the global
/// statement ordinal of every catalog operation, its complete created-table identity set, and an
/// optional ordered table-reset block. Old composite records remain byte-for-byte decodable.
const OP_ORDERED_CATALOG_TRANSACTION: u8 = 10;
const OP_IDENTITY_ORDERED_CATALOG_TRANSACTION: u8 = 11;
/// PRODUCT-001 transactional view envelope. Opcodes 10/11 remain byte-for-byte CREATE-TABLE-only;
/// these additive variants carry one exact preimage/dependency/postimage proof per CREATE VIEW.
const OP_ORDERED_CATALOG_VIEW_TRANSACTION: u8 = 12;
const OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION: u8 = 13;

const TXN_INSERT: u8 = 1;
const TXN_UPDATE: u8 = 2;
const TXN_DELETE: u8 = 3;

const TXN_OPERATION_CATALOG: u8 = 1;
const TXN_OPERATION_INSERT: u8 = 2;
const TXN_OPERATION_UPDATE: u8 = 3;
const TXN_OPERATION_DELETE: u8 = 4;
const TXN_OPERATION_TABLE_RESET: u8 = 5;

const CATALOG_RELATION_TABLE: u8 = 1;
const CATALOG_RELATION_VIEW: u8 = 2;

/// The decoded form of a v1 binary INSERT record.
pub(crate) struct BinaryInsertRecord {
    pub(crate) table: String,
    /// `(row_id, encoded_row)` — the row image in `encode_relational_row`'s cell encoding
    /// (the tuple store's canonical format; decoded against the catalog at apply time).
    pub(crate) rows: Vec<(u64, String)>,
}

/// The decoded form of a W5b by-key DELETE record (U1).
pub(crate) struct BinaryDeleteByKeyRecord {
    pub(crate) table: String,
    /// The unique key column NAME (catalog identity survives replays; the covered route
    /// guarantees it is the single unique i32 index column).
    pub(crate) pk_column: String,
    pub(crate) pk_value: i32,
}

/// The decoded form of a W5b by-key UPDATE record (U2).
pub(crate) struct BinaryUpdateByKeyRecord {
    pub(crate) table: String,
    pub(crate) pk_column: String,
    pub(crate) pk_value: i32,
    /// Legacy v1 allocator reservation. ADR-014 replay derives replacement identity from the
    /// visible old version; this value is retained for framing and allocator high-water parity.
    pub(crate) new_row_id: u64,
    /// The new row image in `encode_relational_row`'s cell encoding (all columns, new values).
    pub(crate) new_row_encoded: String,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionCatalogCommand {
    pub(crate) ordinal: u32,
    pub(crate) command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionCatalogOutput {
    pub(crate) relational_next_oid: u32,
    pub(crate) relational_next_column_id: u32,
    /// Stable identities of implicit sequences created by admitted CREATE TABLE commands.
    pub(crate) created_sequence_oids: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryCatalogRelationKind {
    Table,
    View,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryCatalogRelationIdentity {
    pub(crate) kind: BinaryCatalogRelationKind,
    pub(crate) oid: u32,
    pub(crate) digest: gpu_db_wal::CanonicalDigest,
}

/// Per-command identity closure for CREATE VIEW / CREATE OR REPLACE VIEW. This is a vector rather
/// than a name map because one transaction may replace the same view more than once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionViewOperationIdentity {
    pub(crate) command_index: u32,
    pub(crate) ordinal: u32,
    pub(crate) target_before: Option<BinaryCatalogRelationIdentity>,
    pub(crate) dependencies: BTreeMap<String, BinaryCatalogRelationIdentity>,
    pub(crate) target_after: BinaryCatalogRelationIdentity,
}

fn valid_catalog_identity(identity: &BinaryCatalogRelationIdentity) -> bool {
    identity.oid != 0 && identity.digest != [0; 32]
}

fn valid_view_operation_identity(identity: &BinaryTransactionViewOperationIdentity) -> bool {
    identity.target_before.as_ref().is_none_or(|before| {
        before.kind == BinaryCatalogRelationKind::View && valid_catalog_identity(before)
    }) && identity.target_after.kind == BinaryCatalogRelationKind::View
        && valid_catalog_identity(&identity.target_after)
        && !identity.dependencies.is_empty()
        && identity.dependencies.len() <= u32::MAX as usize
        && identity.dependencies.iter().all(|(name, dependency)| {
            !name.is_empty()
                && name.len() <= u16::MAX as usize
                && valid_catalog_identity(dependency)
        })
}

fn encode_catalog_identity(out: &mut Vec<u8>, identity: &BinaryCatalogRelationIdentity) {
    out.push(match identity.kind {
        BinaryCatalogRelationKind::Table => CATALOG_RELATION_TABLE,
        BinaryCatalogRelationKind::View => CATALOG_RELATION_VIEW,
    });
    out.extend_from_slice(&identity.oid.to_le_bytes());
    out.extend_from_slice(&identity.digest);
}

fn encode_optional_catalog_identity(
    out: &mut Vec<u8>,
    identity: Option<&BinaryCatalogRelationIdentity>,
) {
    match identity {
        Some(identity) => {
            out.push(1);
            encode_catalog_identity(out, identity);
        }
        None => out.push(0),
    }
}

fn decode_catalog_identity<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
) -> Result<BinaryCatalogRelationIdentity, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    let kind = match take(1)?[0] {
        CATALOG_RELATION_TABLE => BinaryCatalogRelationKind::Table,
        CATALOG_RELATION_VIEW => BinaryCatalogRelationKind::View,
        other => return Err(fail(&format!("unsupported catalog relation kind {other}"))),
    };
    let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
    let digest = take(32)?.try_into().expect("32 bytes");
    let identity = BinaryCatalogRelationIdentity { kind, oid, digest };
    if !valid_catalog_identity(&identity) {
        return Err(fail("empty catalog relation identity"));
    }
    Ok(identity)
}

/// Stable identity of every admitted statement in an ordered catalog transaction. Row payloads
/// remain coalesced into `mutations`, but this vector preserves statements whose effects were
/// shadowed by a later reset or folded to no final row mutation. Its vector index is the one
/// transaction-wide ordinal authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BinaryTransactionOperationIdentity {
    Catalog { command_index: u32 },
    Insert { table: String },
    Update { table: String },
    Delete { table: String },
    TableReset { table: String },
}

impl BinaryTransactionOperationIdentity {
    pub(crate) fn table(&self) -> Option<&str> {
        match self {
            Self::Catalog { .. } => None,
            Self::Insert { table }
            | Self::Update { table }
            | Self::Delete { table }
            | Self::TableReset { table } => Some(table),
        }
    }

    pub(crate) fn matches_mutation(&self, mutation: &BinaryTransactionMutation) -> bool {
        matches!(
            (self, mutation),
            (
                Self::Insert { table: operation_table },
                BinaryTransactionMutation::Insert { table: mutation_table, .. }
            ) | (
                Self::Update { table: operation_table },
                BinaryTransactionMutation::Update { table: mutation_table, .. }
            ) | (
                Self::Delete { table: operation_table },
                BinaryTransactionMutation::Delete { table: mutation_table, .. }
            ) if operation_table == mutation_table
        )
    }
}

/// One durable explicit-transaction record. `allocator_high_water` is the row-id allocator value
/// after the transaction's insert identities were claimed. Apply uses an idempotent max operation,
/// so the live process (which preclaimed the ids before WAL encoding) and recovery converge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionRecord {
    pub(crate) allocator_high_water: u64,
    /// Typed catalog operations and their positions in the complete transaction statement stream.
    /// The command vector itself is canonical statement order; ordinals bind its gaps to DML and
    /// table-reset operations without replaying SQL text.
    pub(crate) catalog_commands: Vec<BinaryTransactionCatalogCommand>,
    /// Stable output identity for every table created by `catalog_commands`. Current ordered
    /// records require exact coverage; legacy opcode 5/8 records decode this as empty.
    pub(crate) created_table_identities: BTreeMap<String, BinaryTransactionTableIdentity>,
    /// Exact allocator post-state and implicit-sequence output closure for current ordered
    /// catalog records. Table schema identities alone do not carry generated sequence OIDs.
    pub(crate) catalog_output: Option<BinaryTransactionCatalogOutput>,
    /// Exact CREATE VIEW preimage, transitive source, and postimage closure in catalog-command
    /// order. Empty for every historical opcode and for CREATE-TABLE-only opcodes 10/11.
    pub(crate) view_operations: Vec<BinaryTransactionViewOperationIdentity>,
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
    /// Surviving table-root barriers in canonical statement order. Old transaction opcodes decode
    /// this as empty; reset records use their own opcode and keep row bodies after the reset block.
    pub(crate) table_resets: Vec<BinaryTransactionTableReset>,
    /// Final catalog post-state for each sequence consumed by a transaction-private default.
    /// Entries are sorted by name (`BTreeMap`) for deterministic WAL bytes.
    pub(crate) sequence_advances: BTreeMap<String, (i64, bool)>,
    /// Stable binding for every table named by `mutations`. Empty only for legacy opcodes 4/5/6.
    pub(crate) table_identities: BTreeMap<String, BinaryTransactionTableIdentity>,
    pub(crate) mutations: Vec<BinaryTransactionMutation>,
}

/// A decoded binary WAL record of any op (the tag dispatch for apply/replay consumers).
pub(crate) enum BinaryWalRecord {
    Insert(BinaryInsertRecord),
    DeleteByKey(BinaryDeleteByKeyRecord),
    UpdateByKey(BinaryUpdateByKeyRecord),
    Transaction(BinaryTransactionRecord),
}

/// Encode one resolved explicit transaction as ONE WAL payload. Width overflow is reported as
/// `None`; callers must fail the transaction rather than fall back to statement SQL records, which
/// would lose atomicity and predicate-resolution identity.
pub(crate) fn try_encode_binary_transaction(record: &BinaryTransactionRecord) -> Option<Vec<u8>> {
    let reset_names = record
        .table_resets
        .iter()
        .map(|reset| reset.table.as_str())
        .collect::<BTreeSet<_>>();
    let mutation_names = record
        .mutations
        .iter()
        .map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { table, .. }
            | BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
        })
        .collect::<BTreeSet<_>>();
    let identity_names = record
        .table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let identity_bound = !record.table_identities.is_empty();
    let mut catalog_names = BTreeMap::<&str, u32>::new();
    let mut view_commands = Vec::new();
    let mut created_sequence_names = BTreeSet::new();
    let mut prior_catalog_ordinal = None;
    for (command_index, operation) in record.catalog_commands.iter().enumerate() {
        if prior_catalog_ordinal.is_some_and(|prior| operation.ordinal <= prior) {
            return None;
        }
        prior_catalog_ordinal = Some(operation.ordinal);
        match &operation.command {
            Command::CreateTable(create) => {
                if catalog_names
                    .insert(create.table.as_str(), operation.ordinal)
                    .is_some()
                {
                    return None;
                }
                for sequence in create
                    .columns
                    .iter()
                    .filter_map(|column| match &column.default {
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) => Some(sequence.as_str()),
                        _ => None,
                    })
                {
                    if !created_sequence_names.insert(sequence) {
                        return None;
                    }
                }
            }
            Command::CreateView(create) => {
                view_commands.push((
                    u32::try_from(command_index).ok()?,
                    operation.ordinal,
                    create.name.as_str(),
                ));
            }
            _ => return None,
        }
    }
    let created_identity_names = record
        .created_table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let ordered_catalog = !record.catalog_commands.is_empty()
        && (!record.operation_order.is_empty()
            || record.catalog_commands.len() != 1
            || record.catalog_commands[0].ordinal != 0
            || !record.table_resets.is_empty()
            || !record.created_table_identities.is_empty()
            || record.catalog_output.is_some()
            || !record.view_operations.is_empty());
    let view_catalog = !view_commands.is_empty();
    if view_catalog != !record.view_operations.is_empty()
        || record.view_operations.len() != view_commands.len()
        || record.view_operations.iter().zip(&view_commands).any(
            |(identity, (command_index, ordinal, _))| {
                identity.command_index != *command_index
                    || identity.ordinal != *ordinal
                    || !valid_view_operation_identity(identity)
            },
        )
    {
        return None;
    }
    let mut sequence_names_by_ordinal = BTreeMap::<u32, BTreeSet<&str>>::new();
    for ((ordinal, sequence), oid) in &record.sequence_input_oids {
        if sequence.len() > u16::MAX as usize || *oid == 0 {
            return None;
        }
        sequence_names_by_ordinal
            .entry(*ordinal)
            .or_default()
            .insert(sequence);
    }
    let mut ordered_row_names = BTreeSet::new();
    let mut ordered_reset_names = BTreeSet::new();
    if ordered_catalog {
        if record.operation_order.is_empty()
            || record.statement_digests.len() != record.operation_order.len()
            || record.statement_digests.contains(&[0; 32])
        {
            return None;
        }
        let mut next_catalog_index = 0usize;
        for (ordinal, operation) in record.operation_order.iter().enumerate() {
            let ordinal_u32 = u32::try_from(ordinal).ok()?;
            let statement_digest = record.statement_digests.get(ordinal)?;
            match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    if usize::try_from(*command_index).ok() != Some(next_catalog_index)
                        || record
                            .catalog_commands
                            .get(next_catalog_index)
                            .is_none_or(|command| {
                                usize::try_from(command.ordinal).ok() != Some(ordinal)
                            })
                    {
                        return None;
                    }
                    let command = &record.catalog_commands.get(next_catalog_index)?.command;
                    if transaction_statement_digest(command).ok().as_ref() != Some(statement_digest)
                    {
                        return None;
                    }
                    let expected_sequences = match command {
                        Command::CreateTable(create) => create
                            .columns
                            .iter()
                            .filter_map(|column| match &column.default {
                                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                                    Some(sequence.as_str())
                                }
                                _ => None,
                            })
                            .collect::<BTreeSet<_>>(),
                        Command::CreateView(_) => BTreeSet::new(),
                        _ => return None,
                    };
                    if sequence_names_by_ordinal
                        .get(&ordinal_u32)
                        .cloned()
                        .unwrap_or_default()
                        != expected_sequences
                    {
                        return None;
                    }
                    next_catalog_index += 1;
                }
                BinaryTransactionOperationIdentity::Insert { table }
                | BinaryTransactionOperationIdentity::Update { table }
                | BinaryTransactionOperationIdentity::Delete { table } => {
                    if table.len() > u16::MAX as usize
                        || catalog_names
                            .get(table.as_str())
                            .is_some_and(|create_ordinal| {
                                usize::try_from(*create_ordinal)
                                    .ok()
                                    .is_none_or(|created| created >= ordinal)
                            })
                    {
                        return None;
                    }
                    ordered_row_names.insert(table.as_str());
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    if table.len() > u16::MAX as usize
                        || catalog_names
                            .get(table.as_str())
                            .is_some_and(|create_ordinal| {
                                usize::try_from(*create_ordinal)
                                    .ok()
                                    .is_none_or(|created| created >= ordinal)
                            })
                    {
                        return None;
                    }
                    let command = Command::TruncateTable(TruncateTable {
                        name: table.clone(),
                        restart_identity: false,
                    });
                    if transaction_statement_digest(&command).ok().as_ref()
                        != Some(statement_digest)
                    {
                        return None;
                    }
                    ordered_reset_names.insert(table.as_str());
                }
            }
            if !matches!(
                operation,
                BinaryTransactionOperationIdentity::Catalog { .. }
                    | BinaryTransactionOperationIdentity::Insert { .. }
            ) && sequence_names_by_ordinal.contains_key(&ordinal_u32)
            {
                return None;
            }
        }
        if next_catalog_index != record.catalog_commands.len() {
            return None;
        }
        if sequence_names_by_ordinal.keys().any(|ordinal| {
            usize::try_from(*ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= record.operation_order.len())
        }) {
            return None;
        }
        let insert_sequence_names = record
            .sequence_input_oids
            .keys()
            .filter_map(|(ordinal, sequence)| {
                usize::try_from(*ordinal)
                    .ok()
                    .and_then(|ordinal| record.operation_order.get(ordinal))
                    .and_then(|operation| {
                        matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                            .then_some(sequence.as_str())
                    })
            })
            .collect::<BTreeSet<_>>();
        if insert_sequence_names
            != record
                .sequence_advances
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        {
            return None;
        }
    } else if !record.statement_digests.is_empty() || !record.sequence_input_oids.is_empty() {
        return None;
    }
    let ordered_existing_row_names = ordered_row_names
        .difference(&catalog_names.keys().copied().collect::<BTreeSet<_>>())
        .copied()
        .collect::<BTreeSet<_>>();
    if record.catalog_commands.len() > u32::MAX as usize
        || record.created_table_identities.len() > u32::MAX as usize
        || record.catalog_output.as_ref().is_some_and(|output| {
            output.created_sequence_oids.len() > u32::MAX as usize
                || output
                    .created_sequence_oids
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    != created_sequence_names
        })
        || record.operation_order.len() > u32::MAX as usize
        || record.statement_digests.len() > u32::MAX as usize
        || record.sequence_input_oids.len() > u32::MAX as usize
        || record.table_resets.len() > u32::MAX as usize
        || record.view_operations.len() > u32::MAX as usize
        || record.sequence_advances.len() > u32::MAX as usize
        || record.mutations.len() > u32::MAX as usize
        || (record.catalog_commands.is_empty() && !record.created_table_identities.is_empty())
        || (record.catalog_commands.is_empty() && record.catalog_output.is_some())
        || (record.catalog_commands.is_empty() && !record.view_operations.is_empty())
        || (ordered_catalog
            && created_identity_names != catalog_names.keys().copied().collect::<BTreeSet<_>>())
        || (!ordered_catalog && !record.created_table_identities.is_empty())
        || (ordered_catalog != record.catalog_output.is_some())
        || (!ordered_catalog && !record.operation_order.is_empty())
        || reset_names.len() != record.table_resets.len()
        || (ordered_catalog
            && (ordered_reset_names != reset_names
                || ordered_existing_row_names != identity_names
                || !mutation_names.is_subset(&ordered_row_names)))
        || (!ordered_catalog && identity_bound && identity_names != mutation_names)
        || record.mutations.iter().any(|mutation| match mutation {
            BinaryTransactionMutation::Insert { .. } => false,
            BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => {
                reset_names.contains(table.as_str())
            }
        })
    {
        return None;
    }
    if ordered_catalog {
        for reset in &record.table_resets {
            let operation = usize::try_from(reset.ordinal)
                .ok()
                .and_then(|ordinal| record.operation_order.get(ordinal))?;
            if !matches!(operation, BinaryTransactionOperationIdentity::TableReset { table } if table == &reset.table)
            {
                return None;
            }
        }
        for mutation in &record.mutations {
            let table = match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table,
            };
            let last_reset = record
                .operation_order
                .iter()
                .rposition(|operation| matches!(operation, BinaryTransactionOperationIdentity::TableReset { table: reset_table } if reset_table == table));
            if !record
                .operation_order
                .iter()
                .enumerate()
                .any(|(ordinal, operation)| {
                    operation.matches_mutation(mutation)
                        && last_reset.is_none_or(|reset_ordinal| ordinal > reset_ordinal)
                })
            {
                return None;
            }
        }
    }
    let mut out = Vec::with_capacity(32 + record.mutations.len() * 96);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    let op = if ordered_catalog {
        if view_catalog {
            if identity_bound {
                OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
            } else {
                OP_ORDERED_CATALOG_VIEW_TRANSACTION
            }
        } else if identity_bound {
            OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
        } else {
            OP_ORDERED_CATALOG_TRANSACTION
        }
    } else {
        match (
            identity_bound,
            !record.table_resets.is_empty(),
            record.catalog_commands.is_empty(),
        ) {
            (true, true, _) => OP_IDENTITY_TABLE_RESET_TRANSACTION,
            (true, false, true) => OP_IDENTITY_TRANSACTION,
            (true, false, false) => OP_IDENTITY_COMPOSITE_TRANSACTION,
            (false, true, _) => OP_TABLE_RESET_TRANSACTION,
            (false, false, true) => OP_TRANSACTION,
            (false, false, false) => OP_COMPOSITE_TRANSACTION,
        }
    };
    out.push(op);
    if matches!(
        op,
        OP_COMPOSITE_TRANSACTION | OP_IDENTITY_COMPOSITE_TRANSACTION
    ) {
        out.extend_from_slice(&(record.catalog_commands.len() as u32).to_le_bytes());
        for operation in &record.catalog_commands {
            let encoded = serde_json::to_vec(&operation.command).ok()?;
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(&encoded);
        }
    }
    if matches!(
        op,
        OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    ) {
        out.extend_from_slice(&(record.catalog_commands.len() as u32).to_le_bytes());
        for operation in &record.catalog_commands {
            let encoded = serde_json::to_vec(&operation.command).ok()?;
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&operation.ordinal.to_le_bytes());
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(&encoded);
        }
        out.extend_from_slice(&(record.created_table_identities.len() as u32).to_le_bytes());
        for (table, identity) in &record.created_table_identities {
            if table.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
            out.extend_from_slice(&identity.table_oid.to_le_bytes());
            out.extend_from_slice(&identity.schema_digest);
        }
        let catalog_output = record.catalog_output.as_ref()?;
        out.extend_from_slice(&catalog_output.relational_next_oid.to_le_bytes());
        out.extend_from_slice(&catalog_output.relational_next_column_id.to_le_bytes());
        out.extend_from_slice(&(catalog_output.created_sequence_oids.len() as u32).to_le_bytes());
        for (sequence, oid) in &catalog_output.created_sequence_oids {
            if sequence.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
            out.extend_from_slice(sequence.as_bytes());
            out.extend_from_slice(&oid.to_le_bytes());
        }
        if matches!(
            op,
            OP_ORDERED_CATALOG_VIEW_TRANSACTION | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
        ) {
            out.extend_from_slice(&(record.view_operations.len() as u32).to_le_bytes());
            for identity in &record.view_operations {
                out.extend_from_slice(&identity.command_index.to_le_bytes());
                out.extend_from_slice(&identity.ordinal.to_le_bytes());
                encode_optional_catalog_identity(&mut out, identity.target_before.as_ref());
                out.extend_from_slice(&(identity.dependencies.len() as u32).to_le_bytes());
                for (name, dependency) in &identity.dependencies {
                    if name.len() > u16::MAX as usize {
                        return None;
                    }
                    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                    out.extend_from_slice(name.as_bytes());
                    encode_catalog_identity(&mut out, dependency);
                }
                encode_catalog_identity(&mut out, &identity.target_after);
            }
        }
        out.extend_from_slice(&(record.operation_order.len() as u32).to_le_bytes());
        for operation in &record.operation_order {
            let (kind, table) = match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    out.push(TXN_OPERATION_CATALOG);
                    out.extend_from_slice(&command_index.to_le_bytes());
                    continue;
                }
                BinaryTransactionOperationIdentity::Insert { table } => {
                    (TXN_OPERATION_INSERT, table)
                }
                BinaryTransactionOperationIdentity::Update { table } => {
                    (TXN_OPERATION_UPDATE, table)
                }
                BinaryTransactionOperationIdentity::Delete { table } => {
                    (TXN_OPERATION_DELETE, table)
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    (TXN_OPERATION_TABLE_RESET, table)
                }
            };
            out.push(kind);
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
        }
        out.extend_from_slice(&(record.statement_digests.len() as u32).to_le_bytes());
        for digest in &record.statement_digests {
            out.extend_from_slice(digest);
        }
        out.extend_from_slice(&(record.sequence_input_oids.len() as u32).to_le_bytes());
        for ((ordinal, sequence), oid) in &record.sequence_input_oids {
            out.extend_from_slice(&ordinal.to_le_bytes());
            out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
            out.extend_from_slice(sequence.as_bytes());
            out.extend_from_slice(&oid.to_le_bytes());
        }
    }
    if matches!(
        op,
        OP_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    ) {
        out.extend_from_slice(&(record.table_resets.len() as u32).to_le_bytes());
        let mut prior_ordinal = None;
        let mut tables = BTreeSet::new();
        for reset in &record.table_resets {
            if prior_ordinal.is_some_and(|prior| reset.ordinal <= prior)
                || !tables.insert(reset.table_oid)
                || reset.table.len() > u16::MAX as usize
                || reset.dependency_identities.len() > u32::MAX as usize
                || reset.dependency_identities.get(&reset.table) != Some(&reset.table_oid)
            {
                return None;
            }
            if catalog_names
                .get(reset.table.as_str())
                .is_some_and(|create_ordinal| *create_ordinal >= reset.ordinal)
            {
                return None;
            }
            prior_ordinal = Some(reset.ordinal);
            out.extend_from_slice(&reset.ordinal.to_le_bytes());
            out.extend_from_slice(&(reset.table.len() as u16).to_le_bytes());
            out.extend_from_slice(reset.table.as_bytes());
            out.extend_from_slice(&reset.table_oid.to_le_bytes());
            out.extend_from_slice(&reset.schema_digest);
            out.extend_from_slice(&reset.source_commit_seq.to_le_bytes());
            out.extend_from_slice(&reset.before_digest);
            out.extend_from_slice(&reset.expected_rows.to_le_bytes());
            out.extend_from_slice(&reset.after_empty_digest);
            out.extend_from_slice(&(reset.dependency_identities.len() as u32).to_le_bytes());
            for (name, oid) in &reset.dependency_identities {
                if name.len() > u16::MAX as usize {
                    return None;
                }
                out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.extend_from_slice(&oid.to_le_bytes());
            }
        }
    }
    if identity_bound {
        out.extend_from_slice(&(record.table_identities.len() as u32).to_le_bytes());
        for (table, identity) in &record.table_identities {
            if table.len() > u16::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(table.len() as u16).to_le_bytes());
            out.extend_from_slice(table.as_bytes());
            out.extend_from_slice(&identity.table_oid.to_le_bytes());
            out.extend_from_slice(&identity.schema_digest);
        }
    }
    out.extend_from_slice(&record.allocator_high_water.to_le_bytes());
    out.extend_from_slice(&(record.sequence_advances.len() as u32).to_le_bytes());
    for (sequence, (last_value, is_called)) in &record.sequence_advances {
        if sequence.len() > u16::MAX as usize {
            return None;
        }
        out.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
        out.extend_from_slice(sequence.as_bytes());
        out.extend_from_slice(&last_value.to_le_bytes());
        out.push(u8::from(*is_called));
    }
    out.extend_from_slice(&(record.mutations.len() as u32).to_le_bytes());
    for mutation in &record.mutations {
        let (kind, table, row_id) = match mutation {
            BinaryTransactionMutation::Insert { table, row_id, .. } => (TXN_INSERT, table, *row_id),
            BinaryTransactionMutation::Update { table, row_id, .. } => (TXN_UPDATE, table, *row_id),
            BinaryTransactionMutation::Delete { table, row_id, .. } => (TXN_DELETE, table, *row_id),
        };
        if table.len() > u16::MAX as usize {
            return None;
        }
        out.push(kind);
        out.extend_from_slice(&(table.len() as u16).to_le_bytes());
        out.extend_from_slice(table.as_bytes());
        out.extend_from_slice(&row_id.to_le_bytes());
        let mut push_row = |encoded: &str| -> Option<()> {
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(encoded.as_bytes());
            Some(())
        };
        match mutation {
            BinaryTransactionMutation::Insert { row_encoded, .. } => push_row(row_encoded)?,
            BinaryTransactionMutation::Update {
                old_row_encoded,
                new_row_encoded,
                ..
            } => {
                push_row(old_row_encoded)?;
                push_row(new_row_encoded)?;
            }
            BinaryTransactionMutation::Delete {
                old_row_encoded, ..
            } => push_row(old_row_encoded)?,
        }
    }
    out.shrink_to_fit();
    Some(out)
}

/// Encode a W5b by-key DELETE record. `None` on width-exceeding names (caller falls back to the
/// SQL-text record, same contract as the insert encoder).
pub(crate) fn encode_binary_delete_by_key(
    table: &str,
    pk_column: &str,
    pk_value: i32,
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || pk_column.len() > u16::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(3 + 2 + table.len() + 2 + pk_column.len() + 4);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_DELETE_BY_KEY);
    out.extend_from_slice(&(table.len() as u16).to_le_bytes());
    out.extend_from_slice(table.as_bytes());
    out.extend_from_slice(&(pk_column.len() as u16).to_le_bytes());
    out.extend_from_slice(pk_column.as_bytes());
    out.extend_from_slice(&pk_value.to_le_bytes());
    Some(out)
}

/// Encode a W5b by-key UPDATE record (U2): table + pk column/value + the new version's row id +
/// the new row image (all columns). `None` on width-exceeding shapes (caller falls back to SQL
/// text). `new_row` is the full post-image in catalog order.
pub(crate) fn encode_binary_update_by_key(
    table: &str,
    pk_column: &str,
    pk_value: i32,
    new_row_id: u64,
    new_row: &[SqlValue],
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || pk_column.len() > u16::MAX as usize {
        return None;
    }
    let encoded = encode_relational_row(new_row);
    let encoded_bytes = encoded.as_bytes();
    if encoded_bytes.len() > u32::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(
        3 + 2 + table.len() + 2 + pk_column.len() + 4 + 8 + 4 + encoded_bytes.len(),
    );
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_UPDATE_BY_KEY);
    out.extend_from_slice(&(table.len() as u16).to_le_bytes());
    out.extend_from_slice(table.as_bytes());
    out.extend_from_slice(&(pk_column.len() as u16).to_le_bytes());
    out.extend_from_slice(pk_column.as_bytes());
    out.extend_from_slice(&pk_value.to_le_bytes());
    out.extend_from_slice(&new_row_id.to_le_bytes());
    out.extend_from_slice(&(encoded_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(encoded_bytes);
    Some(out)
}

/// U2: the byte offset of the 8-byte `new_row_id` field inside an `OP_UPDATE_BY_KEY` record for
/// `(table, pk_column)` — the pump patches the row id it claims at wave formation here, exactly as
/// an INSERT patches its row id at [`CoveredInsertRoute`]'s `binary_row_id_offset`. Layout:
/// tag(1) + version(1) + op(1) + table_len(2) + table + pkcol_len(2) + pkcol + pk_value(4), then
/// the `new_row_id`. Stable given the encoding above; a `debug_assert` in the pump cross-checks it.
pub(crate) fn binary_update_new_row_id_offset(table: &str, pk_column: &str) -> usize {
    3 + 2 + table.len() + 2 + pk_column.len() + 4
}

/// Decode ANY binary record (op dispatch). Errors are LOUD (`Durability`) — a tagged record
/// that fails to decode is corruption-or-version-skew, never silently skipped.
pub(crate) fn decode_binary_record(payload: &[u8]) -> Result<BinaryWalRecord, EngineError> {
    if let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(payload)? {
        let operation = envelope
            .fragments
            .iter()
            .find(|fragment| {
                fragment.kind != gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
            })
            .ok_or_else(|| {
                EngineError::Durability(
                    "canonical binary WAL record has no operation fragment".to_string(),
                )
            })?;
        let inner = Engine::decode_engine_operation(&operation.body)?;
        return decode_binary_record(&inner);
    }
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    match payload.get(2) {
        Some(&OP_INSERT) => decode_binary_insert(payload).map(BinaryWalRecord::Insert),
        Some(&OP_TRANSACTION)
        | Some(&OP_COMPOSITE_TRANSACTION)
        | Some(&OP_TABLE_RESET_TRANSACTION)
        | Some(&OP_IDENTITY_TRANSACTION)
        | Some(&OP_IDENTITY_COMPOSITE_TRANSACTION)
        | Some(&OP_IDENTITY_TABLE_RESET_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_TRANSACTION)
        | Some(&OP_ORDERED_CATALOG_VIEW_TRANSACTION)
        | Some(&OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION) => {
            decode_binary_transaction(payload).map(BinaryWalRecord::Transaction)
        }
        Some(&OP_UPDATE_BY_KEY) => {
            let mut at = 0usize;
            let mut take = |n: usize| -> Result<&[u8], EngineError> {
                let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
                let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
                at = end;
                Ok(slice)
            };
            if take(1)?[0] != WAL_BINARY_TAG {
                return Err(fail("missing tag"));
            }
            if take(1)?[0] != WAL_BINARY_VERSION {
                return Err(fail("unsupported version"));
            }
            if take(1)?[0] != OP_UPDATE_BY_KEY {
                return Err(fail("op dispatch mismatch"));
            }
            let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(table_len)?)
                .map_err(|_| fail("non-utf8 table name"))?
                .to_string();
            let column_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let pk_column = std::str::from_utf8(take(column_len)?)
                .map_err(|_| fail("non-utf8 column name"))?
                .to_string();
            let pk_value = i32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let new_row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let enc_len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let new_row_encoded = std::str::from_utf8(take(enc_len)?)
                .map_err(|_| fail("non-utf8 row encoding"))?
                .to_string();
            if at != payload.len() {
                return Err(fail("trailing bytes"));
            }
            Ok(BinaryWalRecord::UpdateByKey(BinaryUpdateByKeyRecord {
                table,
                pk_column,
                pk_value,
                new_row_id,
                new_row_encoded,
            }))
        }
        Some(&OP_DELETE_BY_KEY) => {
            let mut at = 0usize;
            let mut take = |n: usize| -> Result<&[u8], EngineError> {
                let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
                let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
                at = end;
                Ok(slice)
            };
            if take(1)?[0] != WAL_BINARY_TAG {
                return Err(fail("missing tag"));
            }
            let version = take(1)?[0];
            if version != WAL_BINARY_VERSION {
                return Err(fail(&format!("unsupported version {version}")));
            }
            if take(1)?[0] != OP_DELETE_BY_KEY {
                return Err(fail("op dispatch mismatch"));
            }
            let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(table_len)?)
                .map_err(|_| fail("non-utf8 table name"))?
                .to_string();
            let column_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let pk_column = std::str::from_utf8(take(column_len)?)
                .map_err(|_| fail("non-utf8 column name"))?
                .to_string();
            let pk_value = i32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if at != payload.len() {
                return Err(fail("trailing bytes"));
            }
            Ok(BinaryWalRecord::DeleteByKey(BinaryDeleteByKeyRecord {
                table,
                pk_column,
                pk_value,
            }))
        }
        Some(op) => Err(fail(&format!("unsupported op {op}"))),
        None => Err(fail("truncated header")),
    }
}

fn decode_binary_transaction(payload: &[u8]) -> Result<BinaryTransactionRecord, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
        let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
        at = end;
        Ok(slice)
    };
    if take(1)?[0] != WAL_BINARY_TAG {
        return Err(fail("missing tag"));
    }
    if take(1)?[0] != WAL_BINARY_VERSION {
        return Err(fail("unsupported version"));
    }
    let op = take(1)?[0];
    if !matches!(
        op,
        OP_TRANSACTION
            | OP_COMPOSITE_TRANSACTION
            | OP_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_TRANSACTION
            | OP_IDENTITY_COMPOSITE_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    ) {
        return Err(fail("op dispatch mismatch"));
    }
    let mut catalog_commands = Vec::new();
    let ordered_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_ORDERED_CATALOG_VIEW_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    );
    let view_catalog = matches!(
        op,
        OP_ORDERED_CATALOG_VIEW_TRANSACTION | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    );
    if ordered_catalog
        || matches!(
            op,
            OP_COMPOSITE_TRANSACTION | OP_IDENTITY_COMPOSITE_TRANSACTION
        )
    {
        let command_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if (!ordered_catalog && command_count != 1) || (ordered_catalog && command_count == 0) {
            return Err(fail(
                "catalog transaction contains a non-canonical operation count",
            ));
        }
        catalog_commands.reserve(command_count.min(1024));
        let mut prior_ordinal = None;
        for _ in 0..command_count {
            let ordinal = if ordered_catalog {
                u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"))
            } else {
                0
            };
            if prior_ordinal.is_some_and(|prior| ordinal <= prior) {
                return Err(fail(
                    "catalog operations are not in canonical ordinal order",
                ));
            }
            prior_ordinal = Some(ordinal);
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let bytes = take(len)?;
            let command: Command = serde_json::from_slice(bytes)
                .map_err(|error| fail(&format!("typed catalog command decode failed: {error}")))?;
            if !matches!(command, Command::CreateTable(_))
                && !(view_catalog && matches!(command, Command::CreateView(_)))
            {
                return Err(fail("unsupported composite catalog command"));
            }
            if serde_json::to_vec(&command).ok().as_deref() != Some(bytes) {
                return Err(fail("non-canonical typed catalog command"));
            }
            catalog_commands.push(BinaryTransactionCatalogCommand { ordinal, command });
        }
    }
    let mut created_table_identities = BTreeMap::new();
    let mut catalog_output = None;
    let mut view_operations = Vec::new();
    let mut operation_order = Vec::new();
    let mut statement_digests = Vec::new();
    let mut sequence_input_oids = BTreeMap::new();
    if ordered_catalog {
        let identity_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        for _ in 0..identity_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 created-table identity name"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            if created_table_identities
                .insert(
                    table,
                    BinaryTransactionTableIdentity {
                        table_oid,
                        schema_digest,
                    },
                )
                .is_some()
            {
                return Err(fail("duplicate created-table identity"));
            }
        }
        let command_names = catalog_commands
            .iter()
            .filter_map(|operation| match &operation.command {
                Command::CreateTable(create) => Some(create.table.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let created_table_command_count = catalog_commands
            .iter()
            .filter(|operation| matches!(&operation.command, Command::CreateTable(_)))
            .count();
        let identity_names = created_table_identities
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if command_names.len() != created_table_command_count || command_names != identity_names {
            return Err(fail(
                "ordered catalog identities do not cover the exact created-table set",
            ));
        }
        let relational_next_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        let relational_next_column_id = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        let sequence_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let mut created_sequence_oids = BTreeMap::new();
        for _ in 0..sequence_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let sequence = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 created-sequence identity name"))?
                .to_string();
            let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if created_sequence_oids.insert(sequence, oid).is_some() {
                return Err(fail("duplicate created-sequence identity"));
            }
        }
        let expected_sequence_names = catalog_commands
            .iter()
            .flat_map(|operation| match &operation.command {
                Command::CreateTable(create) => create
                    .columns
                    .iter()
                    .filter_map(|column| match &column.default {
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) => Some(sequence.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<BTreeSet<_>>();
        if expected_sequence_names.len()
            != catalog_commands
                .iter()
                .flat_map(|operation| match &operation.command {
                    Command::CreateTable(create) => create
                        .columns
                        .iter()
                        .filter(|column| {
                            matches!(
                                &column.default,
                                Some(ColumnDefault::SequenceNextVal {
                                    create_if_missing: true,
                                    ..
                                })
                            )
                        })
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                })
                .count()
            || expected_sequence_names
                != created_sequence_oids
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
        {
            return Err(fail(
                "ordered catalog sequence identities do not cover the exact generated set",
            ));
        }
        catalog_output = Some(BinaryTransactionCatalogOutput {
            relational_next_oid,
            relational_next_column_id,
            created_sequence_oids,
        });
        if view_catalog {
            let view_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let expected_views = catalog_commands
                .iter()
                .enumerate()
                .filter_map(|(command_index, operation)| {
                    matches!(&operation.command, Command::CreateView(_))
                        .then_some((command_index, operation.ordinal))
                })
                .collect::<Vec<_>>();
            if view_count == 0 || view_count != expected_views.len() {
                return Err(fail(
                    "view identities do not cover the exact CREATE VIEW command set",
                ));
            }
            view_operations.reserve(view_count.min(1024));
            for (expected_command_index, expected_ordinal) in expected_views {
                let command_index = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if usize::try_from(command_index).ok() != Some(expected_command_index)
                    || ordinal != expected_ordinal
                {
                    return Err(fail(
                        "view identity does not match its catalog command position",
                    ));
                }
                let target_before = match take(1)?[0] {
                    0 => None,
                    1 => Some(decode_catalog_identity(&mut take)?),
                    _ => return Err(fail("invalid optional catalog identity flag")),
                };
                let dependency_count =
                    u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
                let mut dependencies = BTreeMap::new();
                let mut prior_name = None;
                for _ in 0..dependency_count {
                    let name_len =
                        u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                    let name = std::str::from_utf8(take(name_len)?)
                        .map_err(|_| fail("non-utf8 view dependency name"))?
                        .to_string();
                    let dependency = decode_catalog_identity(&mut take)?;
                    if name.is_empty()
                        || prior_name.as_ref().is_some_and(|prior| prior >= &name)
                        || dependencies.insert(name.clone(), dependency).is_some()
                    {
                        return Err(fail("non-canonical view dependency identity"));
                    }
                    prior_name = Some(name);
                }
                let target_after = decode_catalog_identity(&mut take)?;
                let identity = BinaryTransactionViewOperationIdentity {
                    command_index,
                    ordinal,
                    target_before,
                    dependencies,
                    target_after,
                };
                if !valid_view_operation_identity(&identity) {
                    return Err(fail("invalid CREATE VIEW identity closure"));
                }
                view_operations.push(identity);
            }
        }
        let operation_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if operation_count == 0 {
            return Err(fail(
                "ordered catalog transaction has an empty operation order",
            ));
        }
        operation_order.reserve(operation_count.min(64 * 1024));
        for _ in 0..operation_count {
            let kind = take(1)?[0];
            let operation = if kind == TXN_OPERATION_CATALOG {
                BinaryTransactionOperationIdentity::Catalog {
                    command_index: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
                }
            } else {
                let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                let table = std::str::from_utf8(take(name_len)?)
                    .map_err(|_| fail("non-utf8 ordered operation table"))?
                    .to_string();
                match kind {
                    TXN_OPERATION_INSERT => BinaryTransactionOperationIdentity::Insert { table },
                    TXN_OPERATION_UPDATE => BinaryTransactionOperationIdentity::Update { table },
                    TXN_OPERATION_DELETE => BinaryTransactionOperationIdentity::Delete { table },
                    TXN_OPERATION_TABLE_RESET => {
                        BinaryTransactionOperationIdentity::TableReset { table }
                    }
                    other => {
                        return Err(fail(&format!(
                            "unsupported ordered transaction operation {other}"
                        )))
                    }
                }
            };
            operation_order.push(operation);
        }
        let digest_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if digest_count != operation_count {
            return Err(fail(
                "ordered statement digests do not cover the operation order",
            ));
        }
        statement_digests.reserve(digest_count.min(64 * 1024));
        for _ in 0..digest_count {
            let digest: gpu_db_wal::CanonicalDigest = take(32)?.try_into().expect("32 bytes");
            if digest == [0; 32] {
                return Err(fail("ordered statement digest is empty"));
            }
            statement_digests.push(digest);
        }
        let sequence_input_count =
            u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let mut prior_sequence_key = None;
        for _ in 0..sequence_input_count {
            let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let sequence = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 ordered sequence input name"))?
                .to_string();
            let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let key = (ordinal, sequence);
            if oid == 0
                || prior_sequence_key
                    .as_ref()
                    .is_some_and(|prior| prior >= &key)
                || sequence_input_oids.insert(key.clone(), oid).is_some()
            {
                return Err(fail("non-canonical ordered sequence input identity"));
            }
            prior_sequence_key = Some(key);
        }
    }
    let mut table_resets = Vec::new();
    if ordered_catalog
        || matches!(
            op,
            OP_TABLE_RESET_TRANSACTION | OP_IDENTITY_TABLE_RESET_TRANSACTION
        )
    {
        let reset_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if !ordered_catalog && reset_count == 0 {
            return Err(fail("table-reset transaction requires at least one reset"));
        }
        table_resets.reserve(reset_count.min(1024));
        let mut prior_ordinal = None;
        let mut tables = BTreeSet::new();
        for _ in 0..reset_count {
            let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if prior_ordinal.is_some_and(|prior| ordinal <= prior) {
                return Err(fail("table resets are not in canonical ordinal order"));
            }
            prior_ordinal = Some(ordinal);
            let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(table_len)?)
                .map_err(|_| fail("non-utf8 reset table name"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if !tables.insert(table_oid) {
                return Err(fail("duplicate table reset identity"));
            }
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            let source_commit_seq = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let before_digest = take(32)?.try_into().expect("32 bytes");
            let expected_rows = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            let after_empty_digest = take(32)?.try_into().expect("32 bytes");
            let dependency_count =
                u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let mut dependency_identities = BTreeMap::new();
            for _ in 0..dependency_count {
                let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                let name = std::str::from_utf8(take(name_len)?)
                    .map_err(|_| fail("non-utf8 reset dependency name"))?
                    .to_string();
                let oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
                if dependency_identities.insert(name, oid).is_some() {
                    return Err(fail("duplicate table reset dependency"));
                }
            }
            if dependency_identities.get(&table) != Some(&table_oid) {
                return Err(fail(
                    "table reset dependency closure omits its target identity",
                ));
            }
            table_resets.push(BinaryTransactionTableReset {
                ordinal,
                table,
                table_oid,
                schema_digest,
                source_commit_seq,
                before_digest,
                expected_rows,
                after_empty_digest,
                dependency_identities,
            });
        }
    }
    let identity_bound = matches!(
        op,
        OP_IDENTITY_TRANSACTION
            | OP_IDENTITY_COMPOSITE_TRANSACTION
            | OP_IDENTITY_TABLE_RESET_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_TRANSACTION
            | OP_IDENTITY_ORDERED_CATALOG_VIEW_TRANSACTION
    );
    let mut table_identities = BTreeMap::new();
    if identity_bound {
        let identity_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        for _ in 0..identity_count {
            let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
            let table = std::str::from_utf8(take(name_len)?)
                .map_err(|_| fail("non-utf8 transaction identity table"))?
                .to_string();
            let table_oid = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            let schema_digest = take(32)?.try_into().expect("32 bytes");
            if table_identities
                .insert(
                    table,
                    BinaryTransactionTableIdentity {
                        table_oid,
                        schema_digest,
                    },
                )
                .is_some()
            {
                return Err(fail("duplicate transaction table identity"));
            }
        }
    }
    let allocator_high_water = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
    let sequence_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut sequence_advances = BTreeMap::new();
    for _ in 0..sequence_count {
        let name_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
        let name = std::str::from_utf8(take(name_len)?)
            .map_err(|_| fail("non-utf8 sequence name"))?
            .to_string();
        let last_value = i64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let is_called = match take(1)?[0] {
            0 => false,
            1 => true,
            _ => return Err(fail("invalid sequence called flag")),
        };
        if sequence_advances
            .insert(name, (last_value, is_called))
            .is_some()
        {
            return Err(fail("duplicate sequence advancement"));
        }
    }
    let mutation_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut mutations = Vec::with_capacity(mutation_count.min(64 * 1024));
    for _ in 0..mutation_count {
        let kind = take(1)?[0];
        let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
        let table = std::str::from_utf8(take(table_len)?)
            .map_err(|_| fail("non-utf8 table name"))?
            .to_string();
        let row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let mut take_row = || -> Result<String, EngineError> {
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            std::str::from_utf8(take(len)?)
                .map(str::to_string)
                .map_err(|_| fail("non-utf8 row encoding"))
        };
        let mutation = match kind {
            TXN_INSERT => BinaryTransactionMutation::Insert {
                table,
                row_id,
                row_encoded: take_row()?,
            },
            TXN_UPDATE => BinaryTransactionMutation::Update {
                table,
                row_id,
                old_row_encoded: take_row()?,
                new_row_encoded: take_row()?,
            },
            TXN_DELETE => BinaryTransactionMutation::Delete {
                table,
                row_id,
                old_row_encoded: take_row()?,
            },
            other => return Err(fail(&format!("unsupported transaction mutation {other}"))),
        };
        mutations.push(mutation);
    }
    if at != payload.len() {
        return Err(fail("trailing bytes"));
    }
    let reset_names = table_resets
        .iter()
        .map(|reset| reset.table.as_str())
        .collect::<BTreeSet<_>>();
    let mutation_names = mutations
        .iter()
        .map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { table, .. }
            | BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
        })
        .collect::<BTreeSet<_>>();
    let identity_names = table_identities
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let created_ordinals = catalog_commands
        .iter()
        .filter_map(|operation| match &operation.command {
            Command::CreateTable(create) => Some((create.table.as_str(), operation.ordinal)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut ordered_row_names = BTreeSet::new();
    let mut ordered_reset_names = BTreeSet::new();
    let mut next_catalog_index = 0usize;
    if ordered_catalog {
        for (ordinal, operation) in operation_order.iter().enumerate() {
            let ordinal_u32 = u32::try_from(ordinal)
                .map_err(|_| fail("ordered operation ordinal exceeds u32"))?;
            let statement_digest = statement_digests
                .get(ordinal)
                .ok_or_else(|| fail("ordered operation has no statement digest"))?;
            let input_sequence_names = sequence_input_oids
                .keys()
                .filter_map(|(input_ordinal, sequence)| {
                    (*input_ordinal == ordinal_u32).then_some(sequence.as_str())
                })
                .collect::<BTreeSet<_>>();
            match operation {
                BinaryTransactionOperationIdentity::Catalog { command_index } => {
                    if usize::try_from(*command_index).ok() != Some(next_catalog_index)
                        || catalog_commands
                            .get(next_catalog_index)
                            .is_none_or(|command| {
                                usize::try_from(command.ordinal).ok() != Some(ordinal)
                            })
                    {
                        return Err(fail(
                            "ordered catalog command does not match its operation position",
                        ));
                    }
                    let command = &catalog_commands[next_catalog_index].command;
                    if transaction_statement_digest(command)
                        .map_err(|error| fail(&error.to_string()))?
                        != *statement_digest
                    {
                        return Err(fail("ordered catalog statement digest mismatch"));
                    }
                    let expected_sequences = match command {
                        Command::CreateTable(create) => create
                            .columns
                            .iter()
                            .filter_map(|column| match &column.default {
                                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                                    Some(sequence.as_str())
                                }
                                _ => None,
                            })
                            .collect::<BTreeSet<_>>(),
                        Command::CreateView(_) => BTreeSet::new(),
                        _ => return Err(fail("unsupported ordered catalog command")),
                    };
                    if input_sequence_names != expected_sequences {
                        return Err(fail(
                            "ordered catalog sequence inputs do not cover its exact defaults",
                        ));
                    }
                    next_catalog_index += 1;
                }
                BinaryTransactionOperationIdentity::Insert { table }
                | BinaryTransactionOperationIdentity::Update { table }
                | BinaryTransactionOperationIdentity::Delete { table } => {
                    if created_ordinals.get(table.as_str()).is_some_and(|created| {
                        usize::try_from(*created)
                            .ok()
                            .is_none_or(|created| created >= ordinal)
                    }) {
                        return Err(fail(
                            "ordered row operation precedes its transaction-private relation",
                        ));
                    }
                    ordered_row_names.insert(table.as_str());
                }
                BinaryTransactionOperationIdentity::TableReset { table } => {
                    if created_ordinals.get(table.as_str()).is_some_and(|created| {
                        usize::try_from(*created)
                            .ok()
                            .is_none_or(|created| created >= ordinal)
                    }) {
                        return Err(fail(
                            "ordered table reset precedes its transaction-private relation",
                        ));
                    }
                    let command = Command::TruncateTable(TruncateTable {
                        name: table.clone(),
                        restart_identity: false,
                    });
                    if transaction_statement_digest(&command)
                        .map_err(|error| fail(&error.to_string()))?
                        != *statement_digest
                    {
                        return Err(fail("ordered table-reset statement digest mismatch"));
                    }
                    ordered_reset_names.insert(table.as_str());
                }
            }
            if !matches!(
                operation,
                BinaryTransactionOperationIdentity::Catalog { .. }
                    | BinaryTransactionOperationIdentity::Insert { .. }
            ) && !input_sequence_names.is_empty()
            {
                return Err(fail(
                    "non-INSERT ordered operation carries a sequence input",
                ));
            }
        }
        if next_catalog_index != catalog_commands.len() {
            return Err(fail("ordered operation vector omits a catalog command"));
        }
        if sequence_input_oids.keys().any(|(ordinal, _)| {
            usize::try_from(*ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= operation_order.len())
        }) {
            return Err(fail(
                "ordered sequence input ordinal exceeds operation order",
            ));
        }
        let insert_sequence_names = sequence_input_oids
            .keys()
            .filter_map(|(ordinal, sequence)| {
                usize::try_from(*ordinal)
                    .ok()
                    .and_then(|ordinal| operation_order.get(ordinal))
                    .and_then(|operation| {
                        matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                            .then_some(sequence.as_str())
                    })
            })
            .collect::<BTreeSet<_>>();
        if insert_sequence_names
            != sequence_advances
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        {
            return Err(fail(
                "ordered INSERT sequence identities do not close over sequence advances",
            ));
        }
    }
    let created_names = created_ordinals.keys().copied().collect::<BTreeSet<_>>();
    let ordered_existing_row_names = ordered_row_names
        .difference(&created_names)
        .copied()
        .collect::<BTreeSet<_>>();
    if reset_names.len() != table_resets.len()
        || (ordered_catalog
            && (ordered_reset_names != reset_names
                || ordered_existing_row_names != identity_names
                || !mutation_names.is_subset(&ordered_row_names)))
        || (!ordered_catalog && identity_bound && identity_names != mutation_names)
        || mutations.iter().any(|mutation| match mutation {
            BinaryTransactionMutation::Insert { .. } => false,
            BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. } => {
                reset_names.contains(table.as_str())
            }
        })
    {
        return Err(fail("table reset has non-canonical post-reset mutations"));
    }
    if ordered_catalog {
        for reset in &table_resets {
            let Some(operation) = usize::try_from(reset.ordinal)
                .ok()
                .and_then(|ordinal| operation_order.get(ordinal))
            else {
                return Err(fail("table reset ordinal exceeds the operation envelope"));
            };
            if !matches!(operation, BinaryTransactionOperationIdentity::TableReset { table } if table == &reset.table)
            {
                return Err(fail(
                    "table reset output does not match its ordered operation identity",
                ));
            }
        }
        for mutation in &mutations {
            let table = match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table,
            };
            let last_reset = operation_order.iter().rposition(
                |operation| matches!(operation, BinaryTransactionOperationIdentity::TableReset { table: reset_table } if reset_table == table),
            );
            if !operation_order
                .iter()
                .enumerate()
                .any(|(ordinal, operation)| {
                    operation.matches_mutation(mutation)
                        && last_reset.is_none_or(|reset_ordinal| ordinal > reset_ordinal)
                })
            {
                return Err(fail(
                    "resolved mutation has no ordered row operation after its last reset",
                ));
            }
        }
    }
    Ok(BinaryTransactionRecord {
        allocator_high_water,
        catalog_commands,
        created_table_identities,
        catalog_output,
        view_operations,
        operation_order,
        statement_digests,
        sequence_input_oids,
        table_resets,
        sequence_advances,
        table_identities,
        mutations,
    })
}

/// Encode a covered INSERT delta as a v1 binary record. `rows` are `(row_id, values)`.
/// Returns `None` on width-exceeding shapes (table name > u16, cell encoding > u32 — no
/// realistic statement, but the durability path must not silently wrap; the caller falls back
/// to the SQL-text record).
pub(crate) fn try_encode_binary_insert(
    table: &str,
    rows: &[(u64, &[SqlValue])],
) -> Option<Vec<u8>> {
    if table.len() > u16::MAX as usize || rows.len() > u32::MAX as usize {
        return None;
    }
    let mut out = encode_binary_insert_unchecked(table, rows)?;
    out.shrink_to_fit();
    Some(out)
}

fn encode_binary_insert_unchecked(table: &str, rows: &[(u64, &[SqlValue])]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(64 + rows.len() * 48);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(OP_INSERT);
    let table_bytes = table.as_bytes();
    out.extend_from_slice(&(table_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(table_bytes);
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (row_id, values) in rows {
        out.extend_from_slice(&row_id.to_le_bytes());
        let encoded = encode_relational_row(values);
        let encoded_bytes = encoded.as_bytes();
        if encoded_bytes.len() > u32::MAX as usize {
            return None;
        }
        out.extend_from_slice(&(encoded_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(encoded_bytes);
    }
    Some(out)
}

/// Whether `payload` is a binary WAL record (any version).
pub(crate) fn is_binary_wal_record(payload: &[u8]) -> bool {
    payload.first() == Some(&WAL_BINARY_TAG)
}

/// Decode a v1 binary record. Errors are LOUD (`Durability`): a tagged record that fails to
/// decode is corruption-or-version-skew — never silently skipped (the silent-skip discipline is
/// only for the UNTAGGED text consumers).
pub(crate) fn decode_binary_insert(payload: &[u8]) -> Result<BinaryInsertRecord, EngineError> {
    let fail = |what: &str| EngineError::Durability(format!("malformed binary WAL record: {what}"));
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or_else(|| fail("length overflow"))?;
        let slice = payload.get(at..end).ok_or_else(|| fail("truncated"))?;
        at = end;
        Ok(slice)
    };
    if take(1)?[0] != WAL_BINARY_TAG {
        return Err(fail("missing tag"));
    }
    let version = take(1)?[0];
    if version != WAL_BINARY_VERSION {
        return Err(fail(&format!("unsupported version {version}")));
    }
    if take(1)?[0] != OP_INSERT {
        return Err(fail("unsupported op"));
    }
    let table_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
    let table = std::str::from_utf8(take(table_len)?)
        .map_err(|_| fail("non-utf8 table name"))?
        .to_string();
    let row_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    // Capacity clamp (audit 21eddaa7 C): don't trust the record's count for the allocation —
    // a corrupt count fails loudly at `take` anyway; the clamp keeps the allocation bounded.
    let mut rows = Vec::with_capacity(row_count.min(64 * 1024));
    for _ in 0..row_count {
        let row_id = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        let encoded_len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let encoded = std::str::from_utf8(take(encoded_len)?)
            .map_err(|_| fail("non-utf8 row encoding"))?
            .to_string();
        rows.push((row_id, encoded));
    }
    if at != payload.len() {
        return Err(fail("trailing bytes"));
    }
    Ok(BinaryInsertRecord { table, rows })
}

#[cfg(test)]
#[path = "wal_binary/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "wal_binary/w5b_tests.rs"]
mod w5b_tests;
