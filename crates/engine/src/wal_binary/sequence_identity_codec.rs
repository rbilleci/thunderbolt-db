//! Typed sequence lifecycle identities for ordered transaction WAL.
//!
//! Opcodes 18/19 are strict supersets of the accepted index-lifecycle layout. They add one
//! complete stable sequence preimage/dependency/postimage closure per CREATE/RESTART/RENAME/DROP
//! command while preserving every byte of opcodes 4--17.

use super::*;

pub(super) const OP_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION: u8 = 18;
pub(super) const OP_IDENTITY_ORDERED_CATALOG_SEQUENCE_LIFECYCLE_TRANSACTION: u8 = 19;

const CATALOG_RELATION_TABLE: u8 = 1;
const CATALOG_RELATION_SEQUENCE: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinarySequenceColumnDependencyIdentity {
    pub(crate) table: BinaryCatalogRelationIdentity,
    pub(crate) column_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionSequenceLifecycleTargetIdentity {
    pub(crate) before_name: String,
    pub(crate) target_before: Option<BinaryCatalogRelationIdentity>,
    pub(crate) dependencies_before: BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
    pub(crate) after_name: Option<String>,
    pub(crate) target_after: Option<BinaryCatalogRelationIdentity>,
    pub(crate) dependencies_after: BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionSequenceLifecycleOperationIdentity {
    pub(crate) command_index: u32,
    pub(crate) ordinal: u32,
    pub(crate) targets: Vec<BinaryTransactionSequenceLifecycleTargetIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionSequenceResetOperationIdentity {
    pub(crate) ordinal: u32,
    pub(crate) table: String,
    pub(crate) targets: Vec<BinaryTransactionSequenceLifecycleTargetIdentity>,
}

pub(crate) fn command_is_sequence_lifecycle(command: &Command) -> bool {
    matches!(
        command,
        Command::CreateSequence(_)
            | Command::SequenceRestart(_)
            | Command::RenameSequence(_)
            | Command::DropSequence(_)
    )
}

pub(crate) fn generated_sequence_output_from_inputs(
    commands: &[BinaryTransactionCatalogCommand],
    sequence_input_oids: &BTreeMap<(u32, String), u32>,
) -> Option<BTreeMap<String, u32>> {
    let mut generated = BTreeMap::new();
    for operation in commands {
        let Command::CreateTable(create) = &operation.command else {
            continue;
        };
        for sequence in create
            .columns
            .iter()
            .filter_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal {
                    sequence,
                    create_if_missing: true,
                }) => Some(sequence),
                _ => None,
            })
        {
            let oid = sequence_input_oids
                .get(&(operation.ordinal, sequence.clone()))
                .copied()?;
            if generated.insert(sequence.clone(), oid).is_some() {
                return None;
            }
        }
    }
    Some(generated)
}

pub(crate) fn generated_sequence_names(
    commands: &[BinaryTransactionCatalogCommand],
) -> Option<BTreeSet<&str>> {
    let mut generated = BTreeSet::new();
    for sequence in commands
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
    {
        if !generated.insert(sequence) {
            return None;
        }
    }
    Some(generated)
}

pub(crate) fn generated_sequence_output_matches_inputs(
    commands: &[BinaryTransactionCatalogCommand],
    sequence_input_oids: &BTreeMap<(u32, String), u32>,
    output: &BinaryTransactionCatalogOutput,
) -> bool {
    generated_sequence_output_from_inputs(commands, sequence_input_oids)
        .is_some_and(|generated| generated == output.created_sequence_oids)
}

fn valid_catalog_identity(
    identity: &BinaryCatalogRelationIdentity,
    kind: BinaryCatalogRelationKind,
) -> bool {
    identity.kind == kind && identity.oid != 0 && identity.digest != [0; 32]
}

fn valid_dependencies(
    dependencies: &BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> bool {
    dependencies.len() <= u32::MAX as usize
        && dependencies.iter().all(|(name, dependency)| {
            !name.is_empty()
                && name.len() <= u16::MAX as usize
                && dependency.column_id != 0
                && valid_catalog_identity(&dependency.table, BinaryCatalogRelationKind::Table)
        })
}

fn same_dependency_bindings(
    before: &BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
    after: &BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> bool {
    before.len() == after.len()
        && before.iter().all(|(name, dependency)| {
            after.get(name).is_some_and(|next| {
                dependency.column_id == next.column_id && dependency.table.oid == next.table.oid
            })
        })
}

fn valid_target(identity: &BinaryTransactionSequenceLifecycleTargetIdentity) -> bool {
    !identity.before_name.is_empty()
        && identity.before_name.len() <= u16::MAX as usize
        && identity.target_before.as_ref().is_none_or(|target| {
            valid_catalog_identity(target, BinaryCatalogRelationKind::Sequence)
        })
        && valid_dependencies(&identity.dependencies_before)
        && identity.after_name.is_some() == identity.target_after.is_some()
        && identity
            .after_name
            .as_ref()
            .is_none_or(|name| !name.is_empty() && name.len() <= u16::MAX as usize)
        && identity.target_after.as_ref().is_none_or(|target| {
            valid_catalog_identity(target, BinaryCatalogRelationKind::Sequence)
        })
        && valid_dependencies(&identity.dependencies_after)
}

pub(crate) fn valid_sequence_lifecycle_operation_identity(
    command: &Command,
    identity: &BinaryTransactionSequenceLifecycleOperationIdentity,
) -> bool {
    if identity.targets.is_empty()
        || identity.targets.len() > u32::MAX as usize
        || !identity.targets.iter().all(valid_target)
    {
        return false;
    }
    match command {
        Command::CreateSequence(create) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == create.name
                && target.target_before.is_none()
                && target.dependencies_before.is_empty()
                && target.after_name.as_deref() == Some(create.name.as_str())
                && target.target_after.is_some()
                && target.dependencies_after.is_empty()
        }
        Command::RenameSequence(rename) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == rename.old_name
                && target.after_name.as_deref() == Some(rename.new_name.as_str())
                && target
                    .target_before
                    .as_ref()
                    .zip(target.target_after.as_ref())
                    .is_some_and(|(before, after)| before.oid == after.oid)
                && same_dependency_bindings(&target.dependencies_before, &target.dependencies_after)
        }
        Command::SequenceRestart(restart) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == restart.name
                && target.after_name.as_deref() == Some(restart.name.as_str())
                && target
                    .target_before
                    .as_ref()
                    .zip(target.target_after.as_ref())
                    .is_some_and(|(before, after)| before.oid == after.oid)
                && same_dependency_bindings(&target.dependencies_before, &target.dependencies_after)
        }
        Command::DropSequence(drop) => {
            if identity.targets.len() != drop.names.len() {
                return false;
            }
            let mut names = BTreeSet::new();
            identity
                .targets
                .iter()
                .zip(&drop.names)
                .all(|(target, name)| {
                    names.insert(name.as_str())
                        && target.before_name == *name
                        && target.after_name.is_none()
                        && target.target_after.is_none()
                        && target.dependencies_after.is_empty()
                        && match &target.target_before {
                            Some(_) => target.dependencies_before.is_empty(),
                            None => drop.if_exists && target.dependencies_before.is_empty(),
                        }
                })
        }
        _ => false,
    }
}

pub(crate) fn valid_sequence_reset_operation_identity(
    identity: &BinaryTransactionSequenceResetOperationIdentity,
) -> bool {
    if identity.table.is_empty()
        || identity.table.len() > u16::MAX as usize
        || identity.targets.len() > u32::MAX as usize
    {
        return false;
    }
    let mut names = BTreeSet::new();
    let mut oids = BTreeSet::new();
    identity.targets.iter().all(|target| {
        valid_target(target)
            && names.insert(target.before_name.as_str())
            && target.after_name.as_deref() == Some(target.before_name.as_str())
            && target
                .target_before
                .as_ref()
                .zip(target.target_after.as_ref())
                .is_some_and(|(before, after)| before.oid == after.oid && oids.insert(before.oid))
            && same_dependency_bindings(&target.dependencies_before, &target.dependencies_after)
    })
}

fn encode_string(out: &mut Vec<u8>, value: &str) -> Option<()> {
    out.extend_from_slice(&u16::try_from(value.len()).ok()?.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Some(())
}

fn encode_identity(out: &mut Vec<u8>, identity: &BinaryCatalogRelationIdentity) -> Option<()> {
    out.push(match identity.kind {
        BinaryCatalogRelationKind::Table => CATALOG_RELATION_TABLE,
        BinaryCatalogRelationKind::Sequence => CATALOG_RELATION_SEQUENCE,
        BinaryCatalogRelationKind::View => return None,
    });
    out.extend_from_slice(&identity.oid.to_le_bytes());
    out.extend_from_slice(&identity.digest);
    Some(())
}

fn encode_optional_identity(
    out: &mut Vec<u8>,
    identity: Option<&BinaryCatalogRelationIdentity>,
) -> Option<()> {
    match identity {
        Some(identity) => {
            out.push(1);
            encode_identity(out, identity)?;
        }
        None => out.push(0),
    }
    Some(())
}

fn encode_dependencies(
    out: &mut Vec<u8>,
    dependencies: &BTreeMap<String, BinarySequenceColumnDependencyIdentity>,
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(dependencies.len()).ok()?.to_le_bytes());
    for (name, dependency) in dependencies {
        encode_string(out, name)?;
        encode_identity(out, &dependency.table)?;
        out.extend_from_slice(&dependency.column_id.to_le_bytes());
    }
    Some(())
}

pub(super) fn encode_sequence_lifecycle_operations(
    out: &mut Vec<u8>,
    identities: &[BinaryTransactionSequenceLifecycleOperationIdentity],
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(identities.len()).ok()?.to_le_bytes());
    for identity in identities {
        out.extend_from_slice(&identity.command_index.to_le_bytes());
        out.extend_from_slice(&identity.ordinal.to_le_bytes());
        out.extend_from_slice(&u32::try_from(identity.targets.len()).ok()?.to_le_bytes());
        for target in &identity.targets {
            encode_string(out, &target.before_name)?;
            encode_optional_identity(out, target.target_before.as_ref())?;
            encode_dependencies(out, &target.dependencies_before)?;
            match (&target.after_name, &target.target_after) {
                (Some(name), Some(after)) => {
                    out.push(1);
                    encode_string(out, name)?;
                    encode_identity(out, after)?;
                }
                (None, None) => out.push(0),
                _ => return None,
            }
            encode_dependencies(out, &target.dependencies_after)?;
        }
    }
    Some(())
}

pub(super) fn encode_sequence_reset_operations(
    out: &mut Vec<u8>,
    identities: &[BinaryTransactionSequenceResetOperationIdentity],
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(identities.len()).ok()?.to_le_bytes());
    let mut prior_ordinal = None;
    for identity in identities {
        if !valid_sequence_reset_operation_identity(identity)
            || prior_ordinal.is_some_and(|prior| identity.ordinal <= prior)
        {
            return None;
        }
        prior_ordinal = Some(identity.ordinal);
        out.extend_from_slice(&identity.ordinal.to_le_bytes());
        encode_string(out, &identity.table)?;
        out.extend_from_slice(&u32::try_from(identity.targets.len()).ok()?.to_le_bytes());
        for target in &identity.targets {
            encode_string(out, &target.before_name)?;
            encode_optional_identity(out, target.target_before.as_ref())?;
            encode_dependencies(out, &target.dependencies_before)?;
            match (&target.after_name, &target.target_after) {
                (Some(name), Some(after)) => {
                    out.push(1);
                    encode_string(out, name)?;
                    encode_identity(out, after)?;
                }
                (None, None) => out.push(0),
                _ => return None,
            }
            encode_dependencies(out, &target.dependencies_after)?;
        }
    }
    Some(())
}

fn decode_string<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
    what: &str,
) -> Result<String, EngineError> {
    let len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
    std::str::from_utf8(take(len)?)
        .map(str::to_string)
        .map_err(|_| fail(what))
}

fn decode_identity<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryCatalogRelationIdentity, EngineError> {
    let kind = match take(1)?[0] {
        CATALOG_RELATION_TABLE => BinaryCatalogRelationKind::Table,
        CATALOG_RELATION_SEQUENCE => BinaryCatalogRelationKind::Sequence,
        _ => return Err(fail("invalid sequence lifecycle relation kind")),
    };
    let identity = BinaryCatalogRelationIdentity {
        kind,
        oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
        digest: take(32)?.try_into().expect("32 bytes"),
    };
    if identity.oid == 0 || identity.digest == [0; 32] {
        return Err(fail("empty sequence lifecycle identity"));
    }
    Ok(identity)
}

fn decode_optional_identity<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<Option<BinaryCatalogRelationIdentity>, EngineError> {
    match take(1)?[0] {
        0 => Ok(None),
        1 => decode_identity(take, fail).map(Some),
        _ => Err(fail("invalid optional sequence lifecycle identity flag")),
    }
}

fn decode_dependencies<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BTreeMap<String, BinarySequenceColumnDependencyIdentity>, EngineError> {
    let count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut dependencies = BTreeMap::new();
    for _ in 0..count {
        let name = decode_string(take, fail, "non-utf8 sequence dependency name")?;
        let table = decode_identity(take, fail)?;
        if table.kind != BinaryCatalogRelationKind::Table {
            return Err(fail("sequence dependency is not a table"));
        }
        let column_id = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        if column_id == 0
            || dependencies
                .insert(
                    name,
                    BinarySequenceColumnDependencyIdentity { table, column_id },
                )
                .is_some()
        {
            return Err(fail("duplicate or empty sequence dependency identity"));
        }
    }
    Ok(dependencies)
}

pub(super) fn decode_sequence_lifecycle_operation<'a>(
    command_index: u32,
    ordinal: u32,
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryTransactionSequenceLifecycleOperationIdentity, EngineError> {
    let count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    if count == 0 {
        return Err(fail("sequence lifecycle identity has no targets"));
    }
    let mut targets = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let before_name = decode_string(take, fail, "non-utf8 sequence target name")?;
        let target_before = decode_optional_identity(take, fail)?;
        let dependencies_before = decode_dependencies(take, fail)?;
        let (after_name, target_after) = match take(1)?[0] {
            0 => (None, None),
            1 => (
                Some(decode_string(
                    take,
                    fail,
                    "non-utf8 sequence postimage name",
                )?),
                Some(decode_identity(take, fail)?),
            ),
            _ => return Err(fail("invalid sequence postimage flag")),
        };
        let dependencies_after = decode_dependencies(take, fail)?;
        targets.push(BinaryTransactionSequenceLifecycleTargetIdentity {
            before_name,
            target_before,
            dependencies_before,
            after_name,
            target_after,
            dependencies_after,
        });
    }
    Ok(BinaryTransactionSequenceLifecycleOperationIdentity {
        command_index,
        ordinal,
        targets,
    })
}

pub(super) fn decode_sequence_reset_operations<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<Vec<BinaryTransactionSequenceResetOperationIdentity>, EngineError> {
    let count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut identities = Vec::with_capacity(count.min(1024));
    let mut prior_ordinal = None;
    for _ in 0..count {
        let ordinal = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
        if prior_ordinal.is_some_and(|prior| ordinal <= prior) {
            return Err(fail(
                "sequence reset identities are not in canonical ordinal order",
            ));
        }
        prior_ordinal = Some(ordinal);
        let table = decode_string(take, fail, "non-utf8 sequence reset table")?;
        let target_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        let mut targets = Vec::with_capacity(target_count.min(1024));
        for _ in 0..target_count {
            let before_name = decode_string(take, fail, "non-utf8 sequence reset target")?;
            let target_before = decode_optional_identity(take, fail)?;
            let dependencies_before = decode_dependencies(take, fail)?;
            let (after_name, target_after) = match take(1)?[0] {
                0 => (None, None),
                1 => (
                    Some(decode_string(
                        take,
                        fail,
                        "non-utf8 sequence reset postimage name",
                    )?),
                    Some(decode_identity(take, fail)?),
                ),
                _ => return Err(fail("invalid sequence reset postimage flag")),
            };
            let dependencies_after = decode_dependencies(take, fail)?;
            targets.push(BinaryTransactionSequenceLifecycleTargetIdentity {
                before_name,
                target_before,
                dependencies_before,
                after_name,
                target_after,
                dependencies_after,
            });
        }
        let identity = BinaryTransactionSequenceResetOperationIdentity {
            ordinal,
            table,
            targets,
        };
        if !valid_sequence_reset_operation_identity(&identity) {
            return Err(fail("invalid sequence reset identity closure"));
        }
        identities.push(identity);
    }
    Ok(identities)
}
