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

const TXN_INSERT: u8 = 1;
const TXN_UPDATE: u8 = 2;
const TXN_DELETE: u8 = 3;

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

/// One durable explicit-transaction record. `allocator_high_water` is the row-id allocator value
/// after the transaction's insert identities were claimed. Apply uses an idempotent max operation,
/// so the live process (which preclaimed the ids before WAL encoding) and recovery converge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionRecord {
    pub(crate) allocator_high_water: u64,
    /// Typed catalog operations applied before the resolved final row mutations at the one atomic
    /// publication boundary. The first compatibility slice admits one `CREATE TABLE`; retaining a
    /// vector makes the durable format ready for ordered catalog expansion without SQL replay.
    pub(crate) catalog_commands: Vec<Command>,
    /// Final catalog post-state for each sequence consumed by a transaction-private default.
    /// Entries are sorted by name (`BTreeMap`) for deterministic WAL bytes.
    pub(crate) sequence_advances: BTreeMap<String, (i64, bool)>,
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
    if record.catalog_commands.len() > 1
        || record.sequence_advances.len() > u32::MAX as usize
        || record.mutations.len() > u32::MAX as usize
    {
        return None;
    }
    let mut out = Vec::with_capacity(32 + record.mutations.len() * 96);
    out.push(WAL_BINARY_TAG);
    out.push(WAL_BINARY_VERSION);
    out.push(if record.catalog_commands.is_empty() {
        OP_TRANSACTION
    } else {
        OP_COMPOSITE_TRANSACTION
    });
    if !record.catalog_commands.is_empty() {
        out.extend_from_slice(&(record.catalog_commands.len() as u32).to_le_bytes());
        for command in &record.catalog_commands {
            if !matches!(command, Command::CreateTable(_)) {
                return None;
            }
            let encoded = serde_json::to_vec(command).ok()?;
            if encoded.len() > u32::MAX as usize {
                return None;
            }
            out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            out.extend_from_slice(&encoded);
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
        Some(&OP_TRANSACTION) | Some(&OP_COMPOSITE_TRANSACTION) => {
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
    if !matches!(op, OP_TRANSACTION | OP_COMPOSITE_TRANSACTION) {
        return Err(fail("op dispatch mismatch"));
    }
    let mut catalog_commands = Vec::new();
    if op == OP_COMPOSITE_TRANSACTION {
        let command_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        if command_count != 1 {
            return Err(fail(
                "composite transaction v1 requires exactly one catalog command",
            ));
        }
        catalog_commands.reserve(command_count.min(1024));
        for _ in 0..command_count {
            let len = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let bytes = take(len)?;
            let command: Command = serde_json::from_slice(bytes)
                .map_err(|error| fail(&format!("typed catalog command decode failed: {error}")))?;
            if !matches!(command, Command::CreateTable(_)) {
                return Err(fail("unsupported composite catalog command"));
            }
            if serde_json::to_vec(&command).ok().as_deref() != Some(bytes) {
                return Err(fail("non-canonical typed catalog command"));
            }
            catalog_commands.push(command);
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
    Ok(BinaryTransactionRecord {
        allocator_high_water,
        catalog_commands,
        sequence_advances,
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
mod tests {
    use super::*;

    #[test]
    fn w5a_binary_insert_round_trips() {
        let rows_owned = [
            (
                7_u64,
                vec![SqlValue::Int4(1), SqlValue::Text("a|b\\c".into())],
            ),
            (8_u64, vec![SqlValue::Int4(2), SqlValue::Null]),
        ];
        let rows: Vec<(u64, &[SqlValue])> = rows_owned
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        let payload = try_encode_binary_insert("public_t", &rows).unwrap();
        assert!(is_binary_wal_record(&payload));
        assert!(
            std::str::from_utf8(&payload).is_err(),
            "0xFF tag must break UTF-8"
        );
        let decoded = decode_binary_insert(&payload).unwrap();
        assert_eq!(decoded.table, "public_t");
        assert_eq!(decoded.rows.len(), 2);
        assert_eq!(decoded.rows[0].0, 7);
        assert_eq!(decoded.rows[0].1, encode_relational_row(&rows_owned[0].1));
    }

    #[test]
    fn w5a_truncated_and_skewed_records_fail_loudly() {
        let payload = try_encode_binary_insert("t", &[(1, &[SqlValue::Int4(5)])]).unwrap();
        assert!(decode_binary_insert(&payload[..payload.len() - 1]).is_err());
        let mut skewed = payload.clone();
        skewed[1] = 99; // version
        assert!(decode_binary_insert(&skewed).is_err());
    }

    #[test]
    fn row_only_transaction_keeps_v1_opcode_and_composite_typed_catalog_round_trips() {
        let mut sequence_advances = BTreeMap::new();
        sequence_advances.insert("s".to_string(), (5, true));
        let row_only = BinaryTransactionRecord {
            allocator_high_water: 9,
            catalog_commands: Vec::new(),
            sequence_advances,
            mutations: vec![BinaryTransactionMutation::Insert {
                table: "t".to_string(),
                row_id: 8,
                row_encoded: "i:42".to_string(),
            }],
        };
        let old_payload = try_encode_binary_transaction(&row_only).unwrap();
        // Literal bytes captured from the pre-composite v1 row-transaction framing. Deliberately
        // do not use codec constants here: the fixture must detect an opcode/tag/version drift as
        // well as sequence and row-mutation layout drift.
        let pre_composite_fixture = vec![
            255, 1, 4, // tag, version, row-only transaction opcode
            9, 0, 0, 0, 0, 0, 0, 0, // allocator high-water
            1, 0, 0, 0, // one sequence advance
            1, 0, b's', // sequence name
            5, 0, 0, 0, 0, 0, 0, 0, 1, // sequence post-state + is_called
            1, 0, 0, 0, // one mutation
            1, // INSERT
            1, 0, b't', // table name
            8, 0, 0, 0, 0, 0, 0, 0, // stable row identity
            4, 0, 0, 0, b'i', b':', b'4', b'2', // encoded row image
        ];
        assert_eq!(old_payload, pre_composite_fixture);
        assert_eq!(
            old_payload[..3],
            [WAL_BINARY_TAG, WAL_BINARY_VERSION, OP_TRANSACTION]
        );
        assert!(matches!(
            decode_binary_record(&old_payload).unwrap(),
            BinaryWalRecord::Transaction(decoded) if decoded == row_only
        ));

        let command = parse_command("CREATE TABLE composite_codec (id int4)").unwrap();
        let composite = BinaryTransactionRecord {
            allocator_high_water: 8,
            catalog_commands: vec![command],
            sequence_advances: BTreeMap::new(),
            mutations: vec![BinaryTransactionMutation::Insert {
                table: "composite_codec".to_string(),
                row_id: 7,
                row_encoded: encode_relational_row(&[SqlValue::Int4(1)]),
            }],
        };
        let payload = try_encode_binary_transaction(&composite).unwrap();
        assert_eq!(
            payload[..3],
            [WAL_BINARY_TAG, WAL_BINARY_VERSION, OP_COMPOSITE_TRANSACTION]
        );
        assert!(matches!(
            decode_binary_record(&payload).unwrap(),
            BinaryWalRecord::Transaction(decoded) if decoded == composite
        ));
        assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());

        let two_commands = BinaryTransactionRecord {
            catalog_commands: vec![
                parse_command("CREATE TABLE composite_codec_a (id int4)").unwrap(),
                parse_command("CREATE TABLE composite_codec_b (id int4)").unwrap(),
            ],
            ..composite
        };
        assert!(try_encode_binary_transaction(&two_commands).is_none());
        let mut malformed = payload;
        malformed[3..7].copy_from_slice(&2_u32.to_le_bytes());
        assert!(decode_binary_record(&malformed).is_err());
    }
}

#[cfg(test)]
mod w5b_tests {
    use super::*;

    /// U1: the by-key DELETE record round-trips through the op-dispatch decoder, and a
    /// truncated/trailing-bytes record fails LOUDLY (never a silent skip).
    #[test]
    fn w5b_delete_by_key_round_trips_and_fails_loud() {
        let payload = encode_binary_delete_by_key("public_accounts", "id", -73).unwrap();
        assert!(is_binary_wal_record(&payload));
        match decode_binary_record(&payload).unwrap() {
            BinaryWalRecord::DeleteByKey(record) => {
                assert_eq!(record.table, "public_accounts");
                assert_eq!(record.pk_column, "id");
                assert_eq!(record.pk_value, -73);
            }
            _ => panic!("decoded the wrong op"),
        }
        assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
        let mut trailing = payload.clone();
        trailing.push(0);
        assert!(decode_binary_record(&trailing).is_err());
        // Unknown op byte is a loud version-skew error.
        let mut skewed = payload;
        skewed[2] = 99;
        assert!(decode_binary_record(&skewed).is_err());
    }

    /// U2 (W5b): the by-key UPDATE record round-trips (table + pk + new_row_id + new image)
    /// through the op-dispatch decoder; truncation/trailing bytes fail LOUDLY.
    #[test]
    fn w5b_update_by_key_round_trips_and_fails_loud() {
        let new_row = [SqlValue::Int4(42), SqlValue::Int4(999)];
        let payload =
            encode_binary_update_by_key("public_t", "id", 42, 7_000_001, &new_row).unwrap();
        assert!(is_binary_wal_record(&payload));
        assert!(std::str::from_utf8(&payload).is_err()); // 0xFF tag -> invalid UTF-8
        match decode_binary_record(&payload).unwrap() {
            BinaryWalRecord::UpdateByKey(record) => {
                assert_eq!(record.table, "public_t");
                assert_eq!(record.pk_column, "id");
                assert_eq!(record.pk_value, 42);
                assert_eq!(record.new_row_id, 7_000_001);
                assert_eq!(record.new_row_encoded, encode_relational_row(&new_row));
            }
            _ => panic!("decoded the wrong op"),
        }
        assert!(decode_binary_record(&payload[..payload.len() - 1]).is_err());
        let mut trailing = payload.clone();
        trailing.push(0);
        assert!(decode_binary_record(&trailing).is_err());

        // PUMP PATCH OFFSET (non-vacuous): the pump stamps its claimed row id at
        // `binary_update_new_row_id_offset` into a PLACEHOLDER-0 record; patching there must land
        // EXACTLY on the decoded new_row_id (a wrong offset = silent identity corruption).
        let placeholder = encode_binary_update_by_key("public_t", "id", 42, 0, &new_row).unwrap();
        assert_eq!(
            placeholder.len(),
            payload.len(),
            "placeholder is byte-width identical"
        );
        let off = binary_update_new_row_id_offset("public_t", "id");
        let mut patched = placeholder.clone();
        patched[off..off + 8].copy_from_slice(&7_000_001u64.to_le_bytes());
        assert_eq!(
            patched, payload,
            "patching the placeholder at the offset reproduces the fully-encoded record"
        );
        match decode_binary_record(&patched).unwrap() {
            BinaryWalRecord::UpdateByKey(record) => assert_eq!(record.new_row_id, 7_000_001),
            _ => panic!("decoded the wrong op"),
        }
    }
}
