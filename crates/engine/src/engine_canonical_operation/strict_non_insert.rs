//! Strict current-codec semantic ownership for codec-5 section S3.
//!
//! This is deliberately inert: no live route, recovery path, or operation-version gate calls
//! it yet.  Its sole job is to turn one S3-declared current engine-operation body into a
//! move-owned semantic model only after the exact current encoding has been reconstructed and
//! compared byte-for-byte.  It is intentionally narrower than the historical recovery decoder:
//! codec 2 admits only direct by-key DELETE/UPDATE, codec 4 has the typed allowlist below, and
//! codecs 1 and 3 remain translator-only compatibility input.  This owner never reparses SQL.

use super::*;

/// A decoded current non-INSERT engine operation retained by future codec-5 S3 parsing.
///
/// The model is deliberately non-`Clone`.  An S3 reader may inspect its fragment kind and codec
/// before moving its one semantic authority onward to a future typed replay owner; it cannot
/// manufacture a second independently decoded operation from cached metadata.
pub(crate) struct CurrentNonInsertCanonicalOperation {
    fragment_kind: gpu_db_wal::CanonicalFragmentKind,
    body_digest: gpu_db_wal::CanonicalDigest,
    statement_digest: gpu_db_wal::CanonicalDigest,
    semantic_class: CurrentNonInsertSemanticClass,
    /// Exact presence fact distilled from a current typed UPDATE/DELETE.  It deliberately
    /// carries no projection shape or raw command: S6 needs only presence, while S7 will own
    /// the eventual typed projection-resolution closure.
    has_returning: bool,
    semantic: CurrentNonInsertCanonicalOperationSemantic,
}

/// Closed S3 semantic categories for current non-INSERT operations.
///
/// The category is evidence derived during exact decode.  It has no constructor carrying an
/// executable command or raw body, so later S3 owners can select their sealed replay path without
/// receiving a second semantic authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CurrentNonInsertSemanticClass {
    Update,
    Delete,
    Catalog,
    TableReset,
    TableRewrite,
    Sequence,
    KeyValueMutation,
}

impl CurrentNonInsertSemanticClass {
    fn fragment_kind(self) -> gpu_db_wal::CanonicalFragmentKind {
        match self {
            Self::Update | Self::Delete | Self::KeyValueMutation => {
                gpu_db_wal::CanonicalFragmentKind::RowMutation
            }
            Self::Catalog => gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            Self::TableReset => gpu_db_wal::CanonicalFragmentKind::TableReset,
            Self::TableRewrite => gpu_db_wal::CanonicalFragmentKind::TableRewrite,
            Self::Sequence => gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
        }
    }
}

/// Copy-only facts retained from the exact current-codec operation body.
///
/// For codec 2, the historical direct by-key operation's body and statement identities are the
/// same exact body digest.  For codec 4, request/body identity remains that digest while
/// statement identity is the canonical typed transaction-statement digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CurrentNonInsertCanonicalOperationFacts {
    body_digest: gpu_db_wal::CanonicalDigest,
    statement_digest: gpu_db_wal::CanonicalDigest,
    fragment_kind: gpu_db_wal::CanonicalFragmentKind,
    operation_codec: u8,
    semantic_class: CurrentNonInsertSemanticClass,
    has_returning: bool,
}

impl CurrentNonInsertCanonicalOperationFacts {
    pub(crate) fn body_digest(self) -> gpu_db_wal::CanonicalDigest {
        self.body_digest
    }

    pub(crate) fn request_digest(self) -> gpu_db_wal::CanonicalDigest {
        self.body_digest
    }

    pub(crate) fn statement_digest(self) -> gpu_db_wal::CanonicalDigest {
        self.statement_digest
    }

    pub(crate) fn fragment_kind(self) -> gpu_db_wal::CanonicalFragmentKind {
        self.fragment_kind
    }

    pub(crate) fn operation_codec(self) -> u8 {
        self.operation_codec
    }

    pub(crate) fn semantic_class(self) -> CurrentNonInsertSemanticClass {
        self.semantic_class
    }

    /// Whether this exact current operation declares a typed UPDATE/DELETE `RETURNING` list.
    /// Resolved-binary by-key operations and every non-DML S3 class have no such list.
    pub(crate) fn has_returning(self) -> bool {
        self.has_returning
    }
}

/// Borrowed input facts for the only typed S3 commands that create a sequence transition.
///
/// These facts let the sequence-binding owner resolve a stable OID and recompute its input digest
/// without receiving the underlying typed command, an owned raw body, or a semantic constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CurrentNonInsertExplicitSequenceFacts<'a> {
    source_name: &'a str,
    operation: BinarySequenceValueOperation,
    requested_set_value: Option<i64>,
}

impl CurrentNonInsertExplicitSequenceFacts<'_> {
    pub(crate) fn source_name(&self) -> &str {
        self.source_name
    }

    pub(crate) fn operation(self) -> BinarySequenceValueOperation {
        self.operation
    }

    pub(crate) fn requested_set_value(self) -> Option<i64> {
        self.requested_set_value
    }
}

/// The decoded, executable semantic payload behind [`CurrentNonInsertCanonicalOperation`].
///
/// This is not a body digest or a raw-byte carrier.  Each case owns the existing decoded engine
/// model and can be re-encoded only through its current binary or typed-command codec.
enum CurrentNonInsertCanonicalOperationSemantic {
    ResolvedBinary(BinaryWalRecord),
    TypedCommandV2(Command),
}

impl CurrentNonInsertCanonicalOperation {
    pub(crate) fn fragment_kind(&self) -> gpu_db_wal::CanonicalFragmentKind {
        self.fragment_kind
    }

    pub(crate) fn operation_codec(&self) -> u8 {
        match self.semantic {
            CurrentNonInsertCanonicalOperationSemantic::ResolvedBinary(_) => {
                ENGINE_OPERATION_CODEC_RESOLVED_BINARY
            }
            CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(_) => {
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2
            }
        }
    }

    pub(crate) fn facts(&self) -> CurrentNonInsertCanonicalOperationFacts {
        CurrentNonInsertCanonicalOperationFacts {
            body_digest: self.body_digest,
            statement_digest: self.statement_digest,
            fragment_kind: self.fragment_kind,
            operation_codec: self.operation_codec(),
            semantic_class: self.semantic_class,
            has_returning: self.has_returning,
        }
    }

    /// Expose the minimum borrowed facts needed to bind an explicit `nextval` or `setval` to a
    /// stable sequence identity.  Default expressions and all non-sequence statements are
    /// intentionally absent: they have different ownership in the typed INSERT lifecycle.
    pub(crate) fn explicit_sequence_facts(
        &self,
    ) -> Option<CurrentNonInsertExplicitSequenceFacts<'_>> {
        match &self.semantic {
            CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(
                Command::SequenceNextVal(nextval),
            ) => Some(CurrentNonInsertExplicitSequenceFacts {
                source_name: nextval.name.as_str(),
                operation: BinarySequenceValueOperation::NextVal,
                requested_set_value: None,
            }),
            CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(
                Command::SequenceSetVal(setval),
            ) => Some(CurrentNonInsertExplicitSequenceFacts {
                source_name: setval.name.as_str(),
                operation: BinarySequenceValueOperation::SetVal {
                    is_called: setval.is_called,
                },
                requested_set_value: Some(setval.value),
            }),
            CurrentNonInsertCanonicalOperationSemantic::ResolvedBinary(_)
            | CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(_) => None,
        }
    }

    /// Reconstruct the exact `GPUDBOP1` body from the retained semantic model.
    ///
    /// Decode has already compared this reconstruction to durable bytes, so a caller receives
    /// byte-identical current-codec framing without retaining a second raw operation body.
    pub(crate) fn reencode_operation_body(&self) -> Result<Vec<u8>, EngineError> {
        let (codec, payload) = match &self.semantic {
            CurrentNonInsertCanonicalOperationSemantic::ResolvedBinary(record) => (
                ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
                reencode_current_non_insert_binary(record)?,
            ),
            CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(command) => (
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                serde_json::to_vec(command).map_err(|error| {
                    EngineError::Durability(format!(
                        "current typed-command-v2 re-encode failed: {error}"
                    ))
                })?,
            ),
        };
        encode_operation_body(codec, &payload)
    }
}

/// Decode one S3-declared current non-INSERT engine operation.
///
/// S3 owns statement/family ordering.  This primitive owns the leaf's declared fragment kind,
/// engine-operation codec, exact body length/digest, and `GPUDBOP1` header.  Only the current
/// resolved-binary codec (2) and current typed-command-v2 codec (4) may enter codec-5 semantics.
pub(crate) fn decode_current_non_insert_canonical_operation(
    declared_fragment_kind: gpu_db_wal::CanonicalFragmentKind,
    declared_operation_codec: u8,
    declared_body_len: u32,
    declared_body_digest: gpu_db_wal::CanonicalDigest,
    body: &[u8],
) -> Result<CurrentNonInsertCanonicalOperation, EngineError> {
    if u32::try_from(body.len()).ok() != Some(declared_body_len) {
        return Err(durability(
            "S3 operation body length does not match its declared length",
        ));
    }
    if gpu_db_wal::canonical_request_digest(body) != declared_body_digest {
        return Err(durability(
            "S3 operation body digest does not match its declared digest",
        ));
    }
    let payload = parse_current_operation_body_header(body, declared_operation_codec)?;
    let (semantic_class, statement_digest, has_returning, semantic) = match declared_operation_codec
    {
        ENGINE_OPERATION_CODEC_RESOLVED_BINARY => {
            let record = decode_binary_record(payload)?;
            validate_current_non_insert_binary(&record)?;
            let semantic_class = current_non_insert_binary_semantic_class(&record)?;
            if semantic_class.fragment_kind() != declared_fragment_kind {
                return Err(durability(
                    "S3 resolved-binary operation fragment kind does not match its semantics",
                ));
            }
            let reencoded = reencode_current_non_insert_binary(&record)?;
            if reencoded != payload {
                return Err(durability(
                    "S3 resolved-binary operation is not the exact current canonical encoding",
                ));
            }
            (
                semantic_class,
                declared_body_digest,
                false,
                CurrentNonInsertCanonicalOperationSemantic::ResolvedBinary(record),
            )
        }
        ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2 => {
            let command: Command = serde_json::from_slice(payload).map_err(|error| {
                EngineError::Durability(format!(
                    "S3 typed-command-v2 operation decode failed: {error}"
                ))
            })?;
            let reencoded = serde_json::to_vec(&command).map_err(|error| {
                EngineError::Durability(format!(
                    "S3 typed-command-v2 operation re-encode failed: {error}"
                ))
            })?;
            if reencoded != payload {
                return Err(durability(
                    "S3 typed-command-v2 operation is not canonical JSON",
                ));
            }
            let Some(semantic_class) = current_typed_non_insert_semantic_class(&command) else {
                return Err(durability(
                    "S3 typed-command-v2 operation is not in the current non-INSERT allowlist",
                ));
            };
            if semantic_class.fragment_kind() != declared_fragment_kind {
                return Err(durability(
                    "S3 typed-command-v2 fragment kind does not match its command",
                ));
            }
            let statement_digest = transaction_statement_digest(&command).map_err(|error| {
                EngineError::Durability(format!(
                    "S3 typed-command-v2 statement digest failed: {error}"
                ))
            })?;
            let has_returning = typed_non_insert_has_returning(&command);
            (
                semantic_class,
                statement_digest,
                has_returning,
                CurrentNonInsertCanonicalOperationSemantic::TypedCommandV2(command),
            )
        }
        _ => {
            return Err(durability(
                "S3 operation uses a legacy, opaque, or unknown engine-operation codec",
            ));
        }
    };
    Ok(CurrentNonInsertCanonicalOperation {
        fragment_kind: declared_fragment_kind,
        body_digest: declared_body_digest,
        statement_digest,
        semantic_class,
        has_returning,
        semantic,
    })
}

/// Distill the only S3 outcome fact that is currently meaningful before S7: whether a typed
/// UPDATE/DELETE declared `RETURNING`.  The caller retains no projection or command access.
fn typed_non_insert_has_returning(command: &Command) -> bool {
    match command {
        Command::Update(update) => !update.returning.is_empty(),
        Command::Delete(delete) => !delete.returning.is_empty(),
        _ => false,
    }
}

fn parse_current_operation_body_header(
    body: &[u8],
    declared_operation_codec: u8,
) -> Result<&[u8], EngineError> {
    if !matches!(
        declared_operation_codec,
        ENGINE_OPERATION_CODEC_RESOLVED_BINARY | ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2
    ) {
        return Err(durability(
            "S3 operation uses a legacy, opaque, or unknown engine-operation codec",
        ));
    }
    if body.len() < ENGINE_OPERATION_BODY_PREFIX_BYTES
        || &body[..ENGINE_OPERATION_MAGIC.len()] != ENGINE_OPERATION_MAGIC
    {
        return Err(durability(
            "S3 operation has an invalid GPUDBOP1 header magic",
        ));
    }
    let body_codec = body[ENGINE_OPERATION_MAGIC.len()];
    if body_codec != declared_operation_codec {
        return Err(durability(
            "S3 operation codec does not match its GPUDBOP1 header",
        ));
    }
    let reserved_start = ENGINE_OPERATION_MAGIC.len() + 1;
    if body[reserved_start..reserved_start + 3] != [0; 3] {
        return Err(durability(
            "S3 operation GPUDBOP1 reserved bytes are non-zero",
        ));
    }
    let len_start = ENGINE_OPERATION_MAGIC.len() + 4;
    let payload_len = u64::from_le_bytes(
        body[len_start..len_start + std::mem::size_of::<u64>()]
            .try_into()
            .expect("operation header length was bounds-checked"),
    );
    let payload = &body[ENGINE_OPERATION_BODY_PREFIX_BYTES..];
    if payload_len != u64::try_from(payload.len()).unwrap_or(u64::MAX) {
        return Err(durability(
            "S3 operation GPUDBOP1 payload length is inconsistent or has trailing bytes",
        ));
    }
    Ok(payload)
}

fn reencode_current_non_insert_binary(record: &BinaryWalRecord) -> Result<Vec<u8>, EngineError> {
    match record {
        BinaryWalRecord::DeleteByKey(record) => {
            reencode_binary_delete_by_key(record).ok_or_else(|| {
                durability("S3 current binary DELETE cannot be re-encoded in its exact codec")
            })
        }
        BinaryWalRecord::UpdateByKey(record) => {
            reencode_binary_update_by_key(record).ok_or_else(|| {
                durability("S3 current binary UPDATE cannot be re-encoded in its exact codec")
            })
        }
        BinaryWalRecord::Insert(_)
        | BinaryWalRecord::Transaction(_)
        | BinaryWalRecord::SequenceValueTransition(_) => Err(durability(
            "S3 resolved-binary codec admits only direct current DELETE and UPDATE records",
        )),
    }
}

fn validate_current_non_insert_binary(record: &BinaryWalRecord) -> Result<(), EngineError> {
    match record {
        BinaryWalRecord::DeleteByKey(_) | BinaryWalRecord::UpdateByKey(_) => Ok(()),
        BinaryWalRecord::Insert(_)
        | BinaryWalRecord::Transaction(_)
        | BinaryWalRecord::SequenceValueTransition(_) => Err(durability(
            "S3 resolved-binary codec admits only direct current DELETE and UPDATE records",
        )),
    }
}

fn current_non_insert_binary_semantic_class(
    record: &BinaryWalRecord,
) -> Result<CurrentNonInsertSemanticClass, EngineError> {
    match record {
        BinaryWalRecord::DeleteByKey(_) => Ok(CurrentNonInsertSemanticClass::Delete),
        BinaryWalRecord::UpdateByKey(_) => Ok(CurrentNonInsertSemanticClass::Update),
        BinaryWalRecord::Insert(_)
        | BinaryWalRecord::Transaction(_)
        | BinaryWalRecord::SequenceValueTransition(_) => Err(durability(
            "S3 resolved-binary codec admits only direct current DELETE and UPDATE records",
        )),
    }
}

/// The current typed-command-v2 non-INSERT allowlist is deliberately exhaustive.  It admits one
/// typed statement only; transaction controls, read/session commands, historical INSERT input,
/// and sequence restart remain outside S3 until their transaction/overlay closure is represented
/// by the later codec-5 sections.
fn current_typed_non_insert_semantic_class(
    command: &Command,
) -> Option<CurrentNonInsertSemanticClass> {
    match command {
        Command::SetKv { .. } | Command::DeleteKv { .. } => {
            Some(CurrentNonInsertSemanticClass::KeyValueMutation)
        }
        Command::Delete(_) => Some(CurrentNonInsertSemanticClass::Delete),
        Command::Update(_) => Some(CurrentNonInsertSemanticClass::Update),
        Command::TruncateTable(_) => Some(CurrentNonInsertSemanticClass::TableReset),
        Command::RefreshMaterializedView(_) => Some(CurrentNonInsertSemanticClass::TableRewrite),
        Command::SequenceNextVal(_) | Command::SequenceSetVal(_) => {
            Some(CurrentNonInsertSemanticClass::Sequence)
        }
        Command::CreateSchema(_)
        | Command::DropSchema(_)
        | Command::CreateDatabase(_)
        | Command::DropDatabase(_)
        | Command::RenameDatabase(_)
        | Command::CreateTablespace(_)
        | Command::DropTablespace(_)
        | Command::RenameTablespace(_)
        | Command::CreateTable(_)
        | Command::AddPrimaryKey(_)
        | Command::AddUniqueConstraint(_)
        | Command::AddCheckConstraint(_)
        | Command::AddForeignKey(_)
        | Command::AddColumn(_)
        | Command::RenameTable(_)
        | Command::RenameColumn(_)
        | Command::RenameConstraint(_)
        | Command::DropColumn(_)
        | Command::DropConstraint(_)
        | Command::CreateIndex(_)
        | Command::RenameIndex(_)
        | Command::CreateView(_)
        | Command::RenameView(_)
        | Command::CreateMaterializedView(_)
        | Command::RenameMaterializedView(_)
        | Command::CreateFunction(_)
        | Command::RenameFunction(_)
        | Command::DropFunction(_)
        | Command::CreateExtension(_)
        | Command::DropExtension(_)
        | Command::CreateSequence(_)
        | Command::CreateDomain(_)
        | Command::RenameSequence(_)
        | Command::DropSequence(_)
        | Command::DropDomain(_)
        | Command::CreatePublication(_)
        | Command::DropPublication(_)
        | Command::CreateSubscription(_)
        | Command::DropSubscription(_)
        | Command::CreateRole(_)
        | Command::DropRole(_)
        | Command::RenameRole(_)
        | Command::GrantTable(_)
        | Command::RevokeTable(_)
        | Command::GrantDatabase(_)
        | Command::RevokeDatabase(_)
        | Command::GrantTablespace(_)
        | Command::RevokeTablespace(_)
        | Command::GrantFunction(_)
        | Command::RevokeFunction(_)
        | Command::GrantSchema(_)
        | Command::RevokeSchema(_)
        | Command::GrantDefaultTablePrivileges(_)
        | Command::RevokeDefaultTablePrivileges(_)
        | Command::DropTable(_)
        | Command::DropIndex(_)
        | Command::DropMaterializedView(_)
        | Command::DropView(_)
        | Command::AlterColumnDefault(_)
        | Command::CommentOn(_)
        | Command::AlterRoleLogin(_) => Some(CurrentNonInsertSemanticClass::Catalog),
        Command::Begin { .. }
        | Command::Commit { .. }
        | Command::Rollback { .. }
        | Command::Flush
        | Command::ResetAll
        | Command::SetRole { .. }
        | Command::GetKv { .. }
        | Command::SequenceCurrVal(_)
        | Command::Insert(_)
        | Command::Select(_)
        | Command::SelectLiteral(_)
        | Command::ShowTransactionIsolation
        | Command::SessionControl { .. }
        | Command::PreparedCatalog(_)
        | Command::SequenceRestart(_)
        | Command::SelectFunction(_) => None,
    }
}

fn durability(message: &str) -> EngineError {
    EngineError::Durability(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn operation_body(codec: u8, payload: &[u8]) -> Vec<u8> {
        encode_operation_body(codec, payload).unwrap()
    }

    fn decode(
        kind: gpu_db_wal::CanonicalFragmentKind,
        codec: u8,
        body: &[u8],
    ) -> Result<CurrentNonInsertCanonicalOperation, EngineError> {
        decode_current_non_insert_canonical_operation(
            kind,
            codec,
            u32::try_from(body.len()).unwrap(),
            gpu_db_wal::canonical_request_digest(body),
            body,
        )
    }

    fn sequence_transition() -> BinarySequenceValueTransitionRecord {
        let operation = BinarySequenceValueOperation::NextVal;
        let parent_request_digest = [0x81; 32];
        BinarySequenceValueTransitionRecord {
            transition_txn_id: 101,
            parent_txn_id: 100,
            parent_autocommit: false,
            statement_ordinal: 2,
            expression_ordinal: 3,
            parent_request_digest,
            input_digest: sequence_value_input_digest(SequenceValueInput {
                parent_txn_id: 100,
                parent_autocommit: false,
                statement_ordinal: 2,
                expression_ordinal: 3,
                parent_request_digest,
                source_name: "s3_sequence",
                operation,
                set_value: None,
            }),
            sequence_oid: 41,
            source_name: "s3_sequence".to_string(),
            effective_name: "s3_sequence".to_string(),
            published_name: "s3_sequence".to_string(),
            base_catalog_generation: 7,
            prior_last_value: 9,
            prior_is_called: true,
            new_last_value: 10,
            new_is_called: true,
            returned_value: 10,
            private_descriptor_digest: None,
            operation,
        }
    }

    fn sequence_catalog_record(names: &[&str]) -> BinaryTransactionRecord {
        let catalog_commands = names
            .iter()
            .enumerate()
            .map(|(ordinal, name)| BinaryTransactionCatalogCommand {
                ordinal: u32::try_from(ordinal).unwrap(),
                command: parse_command(&format!("CREATE SEQUENCE {name}")).unwrap(),
            })
            .collect::<Vec<_>>();
        let sequence_lifecycle_operations = names
            .iter()
            .enumerate()
            .map(|(ordinal, name)| {
                let oid = 41_u32 + u32::try_from(ordinal).unwrap();
                BinaryTransactionSequenceLifecycleOperationIdentity {
                    command_index: u32::try_from(ordinal).unwrap(),
                    ordinal: u32::try_from(ordinal).unwrap(),
                    targets: vec![BinaryTransactionSequenceLifecycleTargetIdentity {
                        before_name: (*name).to_string(),
                        target_before: None,
                        dependencies_before: BTreeMap::new(),
                        after_name: Some((*name).to_string()),
                        target_after: Some(BinaryCatalogRelationIdentity {
                            kind: BinaryCatalogRelationKind::Sequence,
                            oid,
                            digest: [u8::try_from(ordinal + 1).unwrap(); 32],
                        }),
                        dependencies_after: BTreeMap::new(),
                    }],
                }
            })
            .collect::<Vec<_>>();
        let operation_order = (0..names.len())
            .map(|ordinal| BinaryTransactionOperationIdentity::Catalog {
                command_index: u32::try_from(ordinal).unwrap(),
            })
            .collect::<Vec<_>>();
        let statement_digests = catalog_commands
            .iter()
            .map(|operation| transaction_statement_digest(&operation.command).unwrap())
            .collect::<Vec<_>>();
        BinaryTransactionRecord {
            catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
            allocator_high_water: 0,
            catalog_commands,
            created_table_identities: BTreeMap::new(),
            created_table_index_identities: BTreeMap::new(),
            catalog_output: Some(BinaryTransactionCatalogOutput {
                relational_next_oid: 41 + u32::try_from(names.len()).unwrap(),
                relational_next_column_id: 1,
                created_sequence_oids: BTreeMap::new(),
            }),
            view_operations: Vec::new(),
            view_lifecycle_operations: Vec::new(),
            index_lifecycle_operations: Vec::new(),
            sequence_lifecycle_operations,
            sequence_reset_operations: Vec::new(),
            sequence_advances_by_oid: BTreeMap::new(),
            operation_order,
            statement_digests,
            sequence_input_oids: BTreeMap::new(),
            sequence_value_references: Vec::new(),
            table_resets: Vec::new(),
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::new(),
            mutations: Vec::new(),
        }
    }

    fn isolated_reset_record() -> BinaryTransactionRecord {
        BinaryTransactionRecord {
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water: 11,
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
            table_resets: vec![BinaryTransactionTableReset {
                ordinal: 0,
                table: "s3_reset".to_string(),
                table_oid: 42,
                schema_digest: [1; 32],
                source_commit_seq: 6,
                before_digest: [2; 32],
                expected_rows: 7,
                after_empty_digest: [3; 32],
                dependency_identities: BTreeMap::from([("s3_reset".to_string(), 42)]),
            }],
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::new(),
            mutations: Vec::new(),
        }
    }

    fn nested_insert_transaction() -> BinaryTransactionRecord {
        BinaryTransactionRecord {
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water: 2,
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
            mutations: vec![BinaryTransactionMutation::Insert {
                table: "s3_nested_insert".to_string(),
                row_id: 1,
                row_encoded: "i:1".to_string(),
            }],
        }
    }

    #[test]
    fn strict_s3_current_operation_reencodes_non_insert_binary_shapes_exactly() {
        let update = encode_binary_update_by_key(
            "s3_update",
            "id",
            7,
            9,
            &[SqlValue::Int4(7), SqlValue::Text("current".to_string())],
        )
        .unwrap();
        let delete = encode_binary_delete_by_key("s3_delete", "id", 7).unwrap();

        for (kind, payload) in [
            (gpu_db_wal::CanonicalFragmentKind::RowMutation, update),
            (gpu_db_wal::CanonicalFragmentKind::RowMutation, delete),
        ] {
            let body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &payload);
            let decoded = decode(kind, ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &body).unwrap();
            assert_eq!(decoded.fragment_kind(), kind);
            assert_eq!(
                decoded.operation_codec(),
                ENGINE_OPERATION_CODEC_RESOLVED_BINARY
            );
            assert_eq!(decoded.reencode_operation_body().unwrap(), body);
        }
    }

    #[test]
    fn strict_s3_current_operation_reencodes_non_insert_typed_shapes_exactly() {
        for (sql, kind) in [
            (
                "UPDATE s3_t SET id = 8 WHERE id = 7",
                gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ),
            (
                "DELETE FROM s3_t WHERE id = 7",
                gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ),
            (
                "CREATE SEQUENCE s3_catalog",
                gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            ),
            (
                "TRUNCATE s3_t",
                gpu_db_wal::CanonicalFragmentKind::TableReset,
            ),
            (
                "SELECT nextval('s3_sequence')",
                gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            ),
        ] {
            let command = parse_command(sql).unwrap();
            let payload = serde_json::to_vec(&command).unwrap();
            let body = operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &payload);
            let decoded = decode(kind, ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &body).unwrap();
            assert_eq!(decoded.fragment_kind(), kind);
            assert_eq!(
                decoded.operation_codec(),
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2
            );
            assert_eq!(decoded.reencode_operation_body().unwrap(), body);
        }
    }

    #[test]
    fn strict_s3_current_operation_facts_bind_every_class_and_digest_formula() {
        let binary_operations = [
            (
                encode_binary_update_by_key("s3_facts_update", "id", 7, 9, &[SqlValue::Int4(9)])
                    .unwrap(),
                CurrentNonInsertSemanticClass::Update,
            ),
            (
                encode_binary_delete_by_key("s3_facts_delete", "id", 7).unwrap(),
                CurrentNonInsertSemanticClass::Delete,
            ),
        ];
        for (payload, expected_class) in binary_operations {
            let body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &payload);
            let expected_body_digest = gpu_db_wal::canonical_request_digest(&body);
            let decoded = decode(
                gpu_db_wal::CanonicalFragmentKind::RowMutation,
                ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
                &body,
            )
            .unwrap();
            let facts = decoded.facts();
            assert_eq!(facts.body_digest(), expected_body_digest);
            assert_eq!(facts.request_digest(), expected_body_digest);
            assert_eq!(facts.statement_digest(), expected_body_digest);
            assert_eq!(
                facts.fragment_kind(),
                gpu_db_wal::CanonicalFragmentKind::RowMutation
            );
            assert_eq!(
                facts.operation_codec(),
                ENGINE_OPERATION_CODEC_RESOLVED_BINARY
            );
            assert_eq!(facts.semantic_class(), expected_class);
        }

        let typed_operations = [
            (
                parse_command("UPDATE s3_facts SET id = 8 WHERE id = 7").unwrap(),
                CurrentNonInsertSemanticClass::Update,
            ),
            (
                parse_command("DELETE FROM s3_facts WHERE id = 7").unwrap(),
                CurrentNonInsertSemanticClass::Delete,
            ),
            (
                Command::SetKv {
                    key: "s3_facts_key".to_string(),
                    value: "value".to_string(),
                },
                CurrentNonInsertSemanticClass::KeyValueMutation,
            ),
            (
                parse_command("CREATE SEQUENCE s3_facts_sequence").unwrap(),
                CurrentNonInsertSemanticClass::Catalog,
            ),
            (
                parse_command("TRUNCATE s3_facts").unwrap(),
                CurrentNonInsertSemanticClass::TableReset,
            ),
            (
                parse_command("REFRESH MATERIALIZED VIEW s3_facts_view").unwrap(),
                CurrentNonInsertSemanticClass::TableRewrite,
            ),
            (
                parse_command("SELECT nextval('s3_facts_sequence')").unwrap(),
                CurrentNonInsertSemanticClass::Sequence,
            ),
        ];
        for (command, expected_class) in typed_operations {
            let payload = serde_json::to_vec(&command).unwrap();
            let body = operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &payload);
            let expected_body_digest = gpu_db_wal::canonical_request_digest(&body);
            let expected_statement_digest = transaction_statement_digest(&command).unwrap();
            let decoded = decode(
                expected_class.fragment_kind(),
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                &body,
            )
            .unwrap();
            let facts = decoded.facts();
            assert_eq!(facts.body_digest(), expected_body_digest);
            assert_eq!(facts.request_digest(), expected_body_digest);
            assert_eq!(facts.statement_digest(), expected_statement_digest);
            assert_eq!(facts.fragment_kind(), expected_class.fragment_kind());
            assert_eq!(
                facts.operation_codec(),
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2
            );
            assert_eq!(facts.semantic_class(), expected_class);
        }
    }

    #[test]
    fn strict_s3_facts_distill_only_typed_update_delete_returning_presence() {
        let resolved = operation_body(
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &encode_binary_update_by_key("s3_returning_binary", "id", 7, 9, &[SqlValue::Int4(9)])
                .unwrap(),
        );
        assert!(!decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &resolved,
        )
        .unwrap()
        .facts()
        .has_returning());

        for (sql, expected) in [
            ("UPDATE s3_returning SET id = 8 WHERE id = 7", false),
            ("DELETE FROM s3_returning WHERE id = 7", false),
            (
                "UPDATE s3_returning SET id = 8 WHERE id = 7 RETURNING id",
                true,
            ),
            ("DELETE FROM s3_returning WHERE id = 7 RETURNING id", true),
        ] {
            let command = parse_command(sql).unwrap();
            let body = operation_body(
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                &serde_json::to_vec(&command).unwrap(),
            );
            assert_eq!(
                decode(
                    gpu_db_wal::CanonicalFragmentKind::RowMutation,
                    ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                    &body,
                )
                .unwrap()
                .facts()
                .has_returning(),
                expected,
                "{sql}"
            );
        }
    }

    #[test]
    fn strict_s3_explicit_sequence_facts_preserve_only_typed_nextval_and_setval_inputs() {
        let nextval = parse_command("SELECT nextval('s3_facts_nextval')").unwrap();
        let nextval_body = operation_body(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &serde_json::to_vec(&nextval).unwrap(),
        );
        let nextval = decode(
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &nextval_body,
        )
        .unwrap();
        let nextval_facts = nextval.explicit_sequence_facts().unwrap();
        assert_eq!(nextval_facts.source_name(), "s3_facts_nextval");
        assert_eq!(
            nextval_facts.operation(),
            BinarySequenceValueOperation::NextVal
        );
        assert_eq!(nextval_facts.requested_set_value(), None);

        let setval = parse_command("SELECT setval('s3_facts_setval', 73, false)").unwrap();
        let setval_body = operation_body(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &serde_json::to_vec(&setval).unwrap(),
        );
        let setval = decode(
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &setval_body,
        )
        .unwrap();
        let setval_facts = setval.explicit_sequence_facts().unwrap();
        assert_eq!(setval_facts.source_name(), "s3_facts_setval");
        assert_eq!(
            setval_facts.operation(),
            BinarySequenceValueOperation::SetVal { is_called: false }
        );
        assert_eq!(setval_facts.requested_set_value(), Some(73));

        let update = parse_command("UPDATE s3_facts SET id = 8 WHERE id = 7").unwrap();
        let update_body = operation_body(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &serde_json::to_vec(&update).unwrap(),
        );
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &update_body,
        )
        .unwrap()
        .explicit_sequence_facts()
        .is_none());
    }

    #[test]
    fn strict_s3_rejects_bad_declared_body_metadata_and_engine_header() {
        let command = parse_command("DELETE FROM s3_t WHERE id = 7").unwrap();
        let payload = serde_json::to_vec(&command).unwrap();
        let body = operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &payload);
        let digest = gpu_db_wal::canonical_request_digest(&body);
        let kind = gpu_db_wal::CanonicalFragmentKind::RowMutation;

        assert!(decode_current_non_insert_canonical_operation(
            kind,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            u32::try_from(body.len() - 1).unwrap(),
            digest,
            &body,
        )
        .is_err());
        assert!(decode_current_non_insert_canonical_operation(
            kind,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            u32::try_from(body.len()).unwrap(),
            [0; 32],
            &body,
        )
        .is_err());

        let mut bad_magic = body.clone();
        bad_magic[0] ^= 1;
        assert!(decode(kind, ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &bad_magic).is_err());
        let mut bad_reserved = body.clone();
        bad_reserved[9] = 1;
        assert!(decode(kind, ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &bad_reserved).is_err());
        let mut bad_length = body.clone();
        bad_length[12..20].copy_from_slice(&(payload.len() as u64 + 1).to_le_bytes());
        assert!(decode(kind, ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &bad_length).is_err());
        let mut trailing = body;
        trailing.push(0);
        assert!(decode(kind, ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &trailing).is_err());
    }

    #[test]
    fn strict_s3_rejects_legacy_unknown_and_mismatched_operation_codecs() {
        let command = parse_command("DELETE FROM s3_t WHERE id = 7").unwrap();
        let payload = serde_json::to_vec(&command).unwrap();
        let kind = gpu_db_wal::CanonicalFragmentKind::RowMutation;
        for codec in [1, 3, 99] {
            let body = operation_body(codec, &payload);
            assert!(decode(kind, codec, &body).is_err());
        }
        let body = operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &payload);
        assert!(decode(kind, ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &body).is_err());
    }

    #[test]
    fn strict_s3_rejects_alternate_json_insert_and_fragment_kind_mismatches() {
        let delete = parse_command("DELETE FROM s3_t WHERE id = 7").unwrap();
        let canonical = serde_json::to_vec(&delete).unwrap();
        let alternate = [b" \n".as_slice(), canonical.as_slice()].concat();
        let body = operation_body(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, &alternate);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &body,
        )
        .is_err());

        let insert = parse_command("INSERT INTO s3_t (id) VALUES (7)").unwrap();
        let insert_body = operation_body(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &serde_json::to_vec(&insert).unwrap(),
        );
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &insert_body,
        )
        .is_err());

        let reset = parse_command("TRUNCATE s3_t").unwrap();
        let reset_body = operation_body(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &serde_json::to_vec(&reset).unwrap(),
        );
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            &reset_body,
        )
        .is_err());

        let delete = encode_binary_delete_by_key("s3_kind", "id", 7).unwrap();
        let delete_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &delete);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &delete_body,
        )
        .is_err());
    }

    #[test]
    fn strict_s3_rejects_typed_control_read_and_deferred_lifecycle_commands() {
        let commands = [
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "FLUSH",
            "RESET ALL",
            "SET ROLE NONE",
            "SET search_path = pg_catalog, public",
            "SHOW TRANSACTION ISOLATION LEVEL",
            "SELECT 1",
            "SELECT currval('s3_sequence')",
            "ALTER SEQUENCE s3_sequence RESTART",
        ];
        for sql in commands {
            let command = parse_command(sql).unwrap();
            let body = operation_body(
                ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                &serde_json::to_vec(&command).unwrap(),
            );
            assert!(
                decode(
                    gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
                    ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
                    &body,
                )
                .is_err(),
                "{sql}"
            );
        }
    }

    #[test]
    fn strict_s3_rejects_top_level_nested_and_non_direct_binary_residuals() {
        let top_level =
            try_encode_binary_insert("s3_top_insert", &[(1, &[SqlValue::Int4(1)])]).unwrap();
        let top_level_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &top_level);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &top_level_body,
        )
        .is_err());

        let nested = try_encode_binary_transaction(&nested_insert_transaction()).unwrap();
        let nested_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &nested);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::RowMutation,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &nested_body,
        )
        .is_err());

        let sequence = encode_sequence_value_transition(&sequence_transition()).unwrap();
        let sequence_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &sequence);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &sequence_body,
        )
        .is_err());

        let reset = try_encode_binary_transaction(&isolated_reset_record()).unwrap();
        let reset_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &reset);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::TableReset,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &reset_body,
        )
        .is_err());

        let multiple =
            try_encode_binary_transaction(&sequence_catalog_record(&["s3_multi_a", "s3_multi_b"]))
                .unwrap();
        let multiple_body = operation_body(ENGINE_OPERATION_CODEC_RESOLVED_BINARY, &multiple);
        assert!(decode(
            gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
            &multiple_body,
        )
        .is_err());
    }

    #[test]
    fn strict_s3_semantic_model_has_no_public_constructor_escape() {
        let source = include_str!("strict_non_insert.rs");
        assert!(source.contains("pub(crate) struct CurrentNonInsertCanonicalOperation"));
        let crate_public_enum = [
            "pub(crate)",
            " enum CurrentNonInsertCanonicalOperationSemantic",
        ]
        .concat();
        let public_enum = ["pub", " enum CurrentNonInsertCanonicalOperationSemantic"].concat();
        let consuming_escape = ["into", "_semantic"].concat();
        let crate_public_constructor = ["pub(crate) fn ", "new"].concat();
        let public_constructor = ["pub fn ", "new"].concat();
        let raw_body_field = ["body", ": Vec<u8>"].concat();
        assert!(!source.contains(&crate_public_enum));
        assert!(!source.contains(&public_enum));
        assert!(!source.contains(&consuming_escape));
        assert!(!source.contains(&crate_public_constructor));
        assert!(!source.contains(&public_constructor));
        assert!(!source.contains(&raw_body_field));
    }
}
