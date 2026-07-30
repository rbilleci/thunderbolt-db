//! Resolved ordinary sequence-value transitions and transaction-reference framing.
//!
//! Opcode 20 is one independently durable `nextval`/default/`setval` transition. Opcode 21 wraps
//! any accepted transaction opcode (4--19) and appends references to transitions that were
//! published before the user envelope. The wrapper avoids cloning the already-versioned
//! transaction-layout matrix and keeps every historical byte layout intact.

use super::*;

pub(super) const OP_SEQUENCE_VALUE_TRANSITION: u8 = 20;
pub(super) const OP_SEQUENCE_REFERENCED_TRANSACTION: u8 = 21;

const SEQUENCE_VALUE_NEXTVAL: u8 = 1;
const SEQUENCE_VALUE_DEFAULT: u8 = 2;
const SEQUENCE_VALUE_SETVAL: u8 = 3;
const SEQUENCE_GUARD_SHARED_STABLE_OID: u8 = 1;
/// Exact durable width of one sequence-value reference in an opcode-21 transaction wrapper.
///
/// The wrapper owns framing and slice ordering; this subcodec owns precisely one reference.
pub(crate) const ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES: usize = 86;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinarySequenceValueOperation {
    NextVal,
    Default,
    SetVal { is_called: bool },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SequenceValueInput<'a> {
    pub(crate) parent_txn_id: TxnId,
    pub(crate) parent_autocommit: bool,
    pub(crate) statement_ordinal: u32,
    pub(crate) expression_ordinal: u32,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) source_name: &'a str,
    pub(crate) operation: BinarySequenceValueOperation,
    pub(crate) set_value: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinarySequenceValueTransitionRecord {
    pub(crate) transition_txn_id: TxnId,
    pub(crate) parent_txn_id: TxnId,
    /// True when the parent is one autocommit sequence/default statement rather than a
    /// multi-statement explicit transaction. Such a child permanently reserves the parent request
    /// identity even when later row validation fails and no user envelope is written.
    pub(crate) parent_autocommit: bool,
    pub(crate) statement_ordinal: u32,
    pub(crate) expression_ordinal: u32,
    pub(crate) parent_request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) input_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) sequence_oid: u32,
    pub(crate) source_name: String,
    pub(crate) effective_name: String,
    pub(crate) published_name: String,
    pub(crate) base_catalog_generation: Index,
    pub(crate) prior_last_value: i64,
    pub(crate) prior_is_called: bool,
    pub(crate) new_last_value: i64,
    pub(crate) new_is_called: bool,
    pub(crate) returned_value: i64,
    pub(crate) private_descriptor_digest: Option<gpu_db_wal::CanonicalDigest>,
    pub(crate) operation: BinarySequenceValueOperation,
}

pub(crate) fn sequence_descriptor_digest(
    sequence_oid: u32,
    effective_name: &str,
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(32 + effective_name.len());
    body.extend_from_slice(b"GPUDBSEQDESCRIPTOR1");
    body.extend_from_slice(&sequence_oid.to_le_bytes());
    body.extend_from_slice(&(effective_name.len() as u64).to_le_bytes());
    body.extend_from_slice(effective_name.as_bytes());
    gpu_db_wal::canonical_request_digest(&body)
}

pub(crate) fn sequence_value_input_digest(
    input: SequenceValueInput<'_>,
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(64 + input.source_name.len());
    body.extend_from_slice(b"GPUDBSEQVALUEINPUT1");
    body.extend_from_slice(&input.parent_txn_id.to_le_bytes());
    body.push(u8::from(input.parent_autocommit));
    body.extend_from_slice(&input.statement_ordinal.to_le_bytes());
    body.extend_from_slice(&input.expression_ordinal.to_le_bytes());
    body.extend_from_slice(&input.parent_request_digest);
    body.extend_from_slice(&(input.source_name.len() as u64).to_le_bytes());
    body.extend_from_slice(input.source_name.as_bytes());
    match input.operation {
        BinarySequenceValueOperation::NextVal => body.push(SEQUENCE_VALUE_NEXTVAL),
        BinarySequenceValueOperation::Default => body.push(SEQUENCE_VALUE_DEFAULT),
        BinarySequenceValueOperation::SetVal { is_called } => {
            body.push(SEQUENCE_VALUE_SETVAL);
            body.push(u8::from(is_called));
            body.extend_from_slice(
                &input
                    .set_value
                    .expect("setval input digest requires its requested value")
                    .to_le_bytes(),
            );
        }
    }
    gpu_db_wal::canonical_request_digest(&body)
}

pub(crate) fn valid_sequence_value_transition(
    record: &BinarySequenceValueTransitionRecord,
) -> bool {
    if record.transition_txn_id == 0
        || record.parent_txn_id == 0
        || record.sequence_oid == 0
        || record.source_name.is_empty()
        || record.source_name.len() > u16::MAX as usize
        || record.effective_name.is_empty()
        || record.effective_name.len() > u16::MAX as usize
        || record.published_name.is_empty()
        || record.published_name.len() > u16::MAX as usize
        || record.base_catalog_generation == 0
        || record.parent_request_digest == [0; 32]
        || record.input_digest == [0; 32]
        || record
            .private_descriptor_digest
            .is_some_and(|digest| digest == [0; 32])
        || record.private_descriptor_digest.is_some_and(|digest| {
            digest != sequence_descriptor_digest(record.sequence_oid, &record.effective_name)
        })
    {
        return false;
    }
    let requested_set_value = matches!(
        record.operation,
        BinarySequenceValueOperation::SetVal { .. }
    )
    .then_some(record.returned_value);
    if sequence_value_input_digest(SequenceValueInput {
        parent_txn_id: record.parent_txn_id,
        parent_autocommit: record.parent_autocommit,
        statement_ordinal: record.statement_ordinal,
        expression_ordinal: record.expression_ordinal,
        parent_request_digest: record.parent_request_digest,
        source_name: &record.source_name,
        operation: record.operation,
        set_value: requested_set_value,
    }) != record.input_digest
    {
        return false;
    }
    match record.operation {
        BinarySequenceValueOperation::NextVal | BinarySequenceValueOperation::Default => {
            let expected = if record.prior_is_called {
                record.prior_last_value.checked_add(1)
            } else {
                Some(record.prior_last_value)
            };
            expected.is_some_and(|expected| {
                record.new_last_value == expected
                    && record.returned_value == expected
                    && record.new_is_called
            })
        }
        BinarySequenceValueOperation::SetVal { is_called } => {
            record.new_last_value == record.returned_value && record.new_is_called == is_called
        }
    }
}

pub(crate) fn encode_sequence_value_transition(
    record: &BinarySequenceValueTransitionRecord,
) -> Option<Vec<u8>> {
    if !valid_sequence_value_transition(record) {
        return None;
    }
    let mut out = Vec::with_capacity(
        160 + record.source_name.len() + record.effective_name.len() + record.published_name.len(),
    );
    out.extend_from_slice(&[
        WAL_BINARY_TAG,
        WAL_BINARY_VERSION,
        OP_SEQUENCE_VALUE_TRANSITION,
    ]);
    out.extend_from_slice(&record.transition_txn_id.to_le_bytes());
    out.extend_from_slice(&record.parent_txn_id.to_le_bytes());
    out.push(u8::from(record.parent_autocommit));
    out.extend_from_slice(&record.statement_ordinal.to_le_bytes());
    out.extend_from_slice(&record.expression_ordinal.to_le_bytes());
    out.extend_from_slice(&record.parent_request_digest);
    out.extend_from_slice(&record.input_digest);
    out.extend_from_slice(&record.sequence_oid.to_le_bytes());
    put_string(&mut out, &record.source_name)?;
    put_string(&mut out, &record.effective_name)?;
    put_string(&mut out, &record.published_name)?;
    out.extend_from_slice(&record.base_catalog_generation.to_le_bytes());
    out.extend_from_slice(&record.prior_last_value.to_le_bytes());
    out.push(u8::from(record.prior_is_called));
    out.extend_from_slice(&record.new_last_value.to_le_bytes());
    out.push(u8::from(record.new_is_called));
    out.extend_from_slice(&record.returned_value.to_le_bytes());
    out.push(match record.operation {
        BinarySequenceValueOperation::NextVal => SEQUENCE_VALUE_NEXTVAL,
        BinarySequenceValueOperation::Default => SEQUENCE_VALUE_DEFAULT,
        BinarySequenceValueOperation::SetVal { .. } => SEQUENCE_VALUE_SETVAL,
    });
    out.push(SEQUENCE_GUARD_SHARED_STABLE_OID);
    match record.private_descriptor_digest {
        Some(digest) => {
            out.push(1);
            out.extend_from_slice(&digest);
        }
        None => out.push(0),
    }
    Some(out)
}

pub(crate) fn decode_sequence_value_transition(
    payload: &[u8],
) -> Result<BinarySequenceValueTransitionRecord, EngineError> {
    let mut decoder = SequenceDecoder::new(payload);
    decoder.header(OP_SEQUENCE_VALUE_TRANSITION)?;
    let transition_txn_id = decoder.u64()?;
    let parent_txn_id = decoder.u64()?;
    let parent_autocommit = decoder.boolean()?;
    let statement_ordinal = decoder.u32()?;
    let expression_ordinal = decoder.u32()?;
    let parent_request_digest = decoder.digest()?;
    let input_digest = decoder.digest()?;
    let sequence_oid = decoder.u32()?;
    let source_name = decoder.string()?;
    let effective_name = decoder.string()?;
    let published_name = decoder.string()?;
    let base_catalog_generation = decoder.u64()?;
    let prior_last_value = decoder.i64()?;
    let prior_is_called = decoder.boolean()?;
    let new_last_value = decoder.i64()?;
    let new_is_called = decoder.boolean()?;
    let returned_value = decoder.i64()?;
    let operation = match decoder.byte()? {
        SEQUENCE_VALUE_NEXTVAL => BinarySequenceValueOperation::NextVal,
        SEQUENCE_VALUE_DEFAULT => BinarySequenceValueOperation::Default,
        SEQUENCE_VALUE_SETVAL => BinarySequenceValueOperation::SetVal {
            is_called: new_is_called,
        },
        _ => return Err(decoder.fail("unknown sequence operation")),
    };
    if decoder.byte()? != SEQUENCE_GUARD_SHARED_STABLE_OID {
        return Err(decoder.fail("unsupported sequence guard mode"));
    }
    let private_descriptor_digest = match decoder.byte()? {
        0 => None,
        1 => Some(decoder.digest()?),
        _ => return Err(decoder.fail("invalid private descriptor marker")),
    };
    decoder.finish()?;
    let record = BinarySequenceValueTransitionRecord {
        transition_txn_id,
        parent_txn_id,
        parent_autocommit,
        statement_ordinal,
        expression_ordinal,
        parent_request_digest,
        input_digest,
        sequence_oid,
        source_name,
        effective_name,
        published_name,
        base_catalog_generation,
        prior_last_value,
        prior_is_called,
        new_last_value,
        new_is_called,
        returned_value,
        private_descriptor_digest,
        operation,
    };
    if !valid_sequence_value_transition(&record) {
        return Err(decoder.fail("noncanonical sequence transition"));
    }
    Ok(record)
}

pub(crate) fn encode_sequence_referenced_transaction(
    base: &[u8],
    references: &[BinarySequenceValueReference],
) -> Option<Vec<u8>> {
    if base.get(0..3).is_none()
        || base[0] != WAL_BINARY_TAG
        || base[1] != WAL_BINARY_VERSION
        || base[2] == OP_SEQUENCE_REFERENCED_TRANSACTION
        || !valid_sequence_value_reference_closure(references)
        || base.len() > u64::MAX as usize
        || references.len() > u32::MAX as usize
    {
        return None;
    }
    let reference_bytes = references
        .len()
        .checked_mul(ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES)?;
    let capacity = 15usize
        .checked_add(base.len())?
        .checked_add(reference_bytes)?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&[
        WAL_BINARY_TAG,
        WAL_BINARY_VERSION,
        OP_SEQUENCE_REFERENCED_TRANSACTION,
    ]);
    out.extend_from_slice(&(base.len() as u64).to_le_bytes());
    out.extend_from_slice(base);
    out.extend_from_slice(&(references.len() as u32).to_le_bytes());
    for reference in references {
        let mut encoded = [0; ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
        encode_sequence_value_reference_into_exact(reference, &mut encoded)
            .expect("validated sequence-reference closure encodes every member");
        out.extend_from_slice(&encoded);
    }
    Some(out)
}

pub(crate) fn decode_sequence_referenced_transaction(
    payload: &[u8],
) -> Result<BinaryTransactionRecord, EngineError> {
    let mut decoder = SequenceDecoder::new(payload);
    decoder.header(OP_SEQUENCE_REFERENCED_TRANSACTION)?;
    let base_len = usize::try_from(decoder.u64()?)
        .map_err(|_| decoder.fail("base transaction length overflow"))?;
    let base = decoder.take(base_len)?.to_vec();
    let reference_count =
        usize::try_from(decoder.u32()?).map_err(|_| decoder.fail("reference count overflow"))?;
    let reference_bytes = reference_count
        .checked_mul(ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES)
        .ok_or_else(|| decoder.fail("reference byte length overflow"))?;
    if decoder.remaining() != reference_bytes {
        return Err(decoder.fail("reference count does not match remaining bytes"));
    }
    let mut references = Vec::new();
    references
        .try_reserve_exact(reference_count)
        .map_err(|_| decoder.fail("reference allocation exceeds capacity"))?;
    for _ in 0..reference_count {
        references.push(decode_sequence_value_reference_exact(
            decoder.take(ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES)?,
        )?);
    }
    decoder.finish()?;
    if !valid_sequence_value_reference_closure(&references) {
        return Err(decoder.fail("noncanonical sequence transition references"));
    }
    let BinaryWalRecord::Transaction(mut transaction) = decode_binary_record(&base)? else {
        return Err(decoder.fail("wrapped payload is not a transaction"));
    };
    if !transaction.sequence_value_references.is_empty() {
        return Err(decoder.fail("nested sequence-reference transaction"));
    }
    if !references
        .iter()
        .filter(|reference| reference.default_expression)
        .all(|reference| {
            transaction.operation_order.is_empty()
                || transaction
                    .operation_order
                    .get(reference.statement_ordinal as usize)
                    .is_some_and(|operation| {
                        matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                    })
        })
    {
        return Err(decoder.fail("sequence reference does not name an INSERT operation"));
    }
    transaction.sequence_value_references = references;
    Ok(transaction)
}

pub(crate) fn validate_sequence_envelope_transaction_id(
    payload: &[u8],
    outer_txn_id: TxnId,
) -> Result<(), EngineError> {
    if payload.get(0..2) != Some(&[WAL_BINARY_TAG, WAL_BINARY_VERSION]) {
        return Ok(());
    }
    match payload.get(2) {
        Some(&OP_SEQUENCE_VALUE_TRANSITION) => {
            let record = decode_sequence_value_transition(payload)?;
            if record.transition_txn_id != outer_txn_id {
                return Err(EngineError::Durability(format!(
                    "sequence transition identity {} does not match canonical transaction identity {outer_txn_id}",
                    record.transition_txn_id
                )));
            }
        }
        Some(&OP_SEQUENCE_REFERENCED_TRANSACTION) => {
            let transaction = decode_sequence_referenced_transaction(payload)?;
            if transaction
                .sequence_value_references
                .iter()
                .any(|reference| {
                    reference.parent_txn_id != outer_txn_id
                        || reference.transition_txn_id == outer_txn_id
                })
            {
                return Err(EngineError::Durability(
                    "sequence-reference parent does not match canonical transaction identity"
                        .to_string(),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Encode one sequence-value reference into its exact opcode-21 subcodec layout.
///
/// This is allocation-free and deliberately does not own ordering or duplicate checks across a
/// transaction; use [`valid_sequence_value_reference_closure`] for that enclosing invariant.
pub(crate) fn encode_sequence_value_reference_into_exact(
    reference: &BinarySequenceValueReference,
    out: &mut [u8; ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES],
) -> Result<(), EngineError> {
    if !valid_sequence_value_reference(reference) {
        return Err(EngineError::Durability(
            "noncanonical standalone sequence-value reference".to_string(),
        ));
    }
    out[0..8].copy_from_slice(&reference.transition_txn_id.to_le_bytes());
    out[8..16].copy_from_slice(&reference.parent_txn_id.to_le_bytes());
    out[16..20].copy_from_slice(&reference.statement_ordinal.to_le_bytes());
    out[20..24].copy_from_slice(&reference.expression_ordinal.to_le_bytes());
    out[24..28].copy_from_slice(&reference.sequence_oid.to_le_bytes());
    out[28..36].copy_from_slice(&reference.returned_value.to_le_bytes());
    out[36..68].copy_from_slice(&reference.input_digest);
    out[68..72].copy_from_slice(&reference.table_oid.to_le_bytes());
    out[72..76].copy_from_slice(&reference.column_id.to_le_bytes());
    out[76..84].copy_from_slice(&reference.row_id.to_le_bytes());
    out[84] = u8::from(reference.final_value_overwritten);
    out[85] = u8::from(reference.default_expression);
    Ok(())
}

/// Decode one sequence-value reference only when `bytes` is the exact standalone width.
///
/// The decoder rejects truncation, surplus, invalid booleans, and every noncanonical scalar
/// binding before a caller can place the value into a transaction closure.
pub(crate) fn decode_sequence_value_reference_exact(
    bytes: &[u8],
) -> Result<BinarySequenceValueReference, EngineError> {
    if bytes.len() != ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES {
        return Err(EngineError::Durability(format!(
            "malformed binary sequence record: sequence-value reference length {} does not match exact width {ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES}",
            bytes.len()
        )));
    }
    let mut decoder = SequenceDecoder::new(bytes);
    let reference = BinarySequenceValueReference {
        transition_txn_id: decoder.u64()?,
        parent_txn_id: decoder.u64()?,
        statement_ordinal: decoder.u32()?,
        expression_ordinal: decoder.u32()?,
        sequence_oid: decoder.u32()?,
        returned_value: decoder.i64()?,
        input_digest: decoder.digest()?,
        table_oid: decoder.u32()?,
        column_id: decoder.u32()?,
        staging_row_ordinal: 0,
        row_id: decoder.u64()?,
        final_value_overwritten: decoder.boolean()?,
        default_expression: decoder.boolean()?,
    };
    decoder.finish()?;
    if !valid_sequence_value_reference(&reference) {
        return Err(decoder.fail("noncanonical sequence transition reference"));
    }
    Ok(reference)
}

fn valid_sequence_value_reference(reference: &BinarySequenceValueReference) -> bool {
    reference.transition_txn_id != 0
        && reference.parent_txn_id != 0
        && reference.sequence_oid != 0
        && reference.input_digest != [0; 32]
        && reference.staging_row_ordinal == 0
        && if reference.default_expression {
            reference.table_oid != 0 && reference.column_id != 0 && reference.row_id != 0
        } else {
            reference.table_oid == 0
                && reference.column_id == 0
                && reference.row_id == 0
                && !reference.final_value_overwritten
        }
}

/// Validate the reference slice as one transaction-owned sequence closure.
///
/// Transition and statement order are monotonic; an expression identity is unique per parent
/// transaction/statement; and one materialized default may bind each table/row/column at most
/// once.  Different expression ordinals on the same row remain distinct when they bind distinct
/// columns, so a multi-default INSERT row is representable without admitting duplicate bindings.
pub(crate) fn valid_sequence_value_reference_closure(
    references: &[BinarySequenceValueReference],
) -> bool {
    if references.is_empty() {
        return false;
    }
    let mut prior_transition_id = None;
    let mut prior_statement_ordinal = None;
    let mut expressions = BTreeSet::new();
    let mut default_bindings = BTreeSet::new();
    for reference in references {
        if !valid_sequence_value_reference(reference)
            || !expressions.insert((
                reference.parent_txn_id,
                reference.statement_ordinal,
                reference.expression_ordinal,
            ))
            || (reference.default_expression
                && !default_bindings.insert((
                    reference.table_oid,
                    reference.row_id,
                    reference.column_id,
                )))
            || prior_transition_id.is_some_and(|prior| prior >= reference.transition_txn_id)
            || prior_statement_ordinal.is_some_and(|prior| prior > reference.statement_ordinal)
        {
            return false;
        }
        prior_transition_id = Some(reference.transition_txn_id);
        prior_statement_ordinal = Some(reference.statement_ordinal);
    }
    true
}

fn put_string(out: &mut Vec<u8>, value: &str) -> Option<()> {
    let len = u16::try_from(value.len()).ok()?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Some(())
}

struct SequenceDecoder<'a> {
    payload: &'a [u8],
    at: usize,
}

impl<'a> SequenceDecoder<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self { payload, at: 0 }
    }

    fn fail(&self, what: &str) -> EngineError {
        EngineError::Durability(format!("malformed binary sequence record: {what}"))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .at
            .checked_add(len)
            .ok_or_else(|| self.fail("length overflow"))?;
        let value = self
            .payload
            .get(self.at..end)
            .ok_or_else(|| self.fail("truncated"))?;
        self.at = end;
        Ok(value)
    }

    fn header(&mut self, operation: u8) -> Result<(), EngineError> {
        if self.byte()? != WAL_BINARY_TAG
            || self.byte()? != WAL_BINARY_VERSION
            || self.byte()? != operation
        {
            return Err(self.fail("header mismatch"));
        }
        Ok(())
    }

    fn byte(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn boolean(&mut self) -> Result<bool, EngineError> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(self.fail("invalid boolean")),
        }
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn i64(&mut self) -> Result<i64, EngineError> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn digest(&mut self) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
        Ok(self.take(32)?.try_into().expect("digest width"))
    }

    fn string(&mut self) -> Result<String, EngineError> {
        let len = self.u16()? as usize;
        let value =
            std::str::from_utf8(self.take(len)?).map_err(|_| self.fail("non-UTF-8 name"))?;
        Ok(value.to_string())
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }

    fn finish(&self) -> Result<(), EngineError> {
        if self.at == self.payload.len() {
            Ok(())
        } else {
            Err(self.fail("trailing bytes"))
        }
    }

    fn remaining(&self) -> usize {
        self.payload.len() - self.at
    }
}
