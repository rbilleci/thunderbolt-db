//! Typed stored-view identity codecs for ordered transaction WAL.
//!
//! Opcodes 12/13 retain the accepted CREATE/CREATE OR REPLACE layout. The lifecycle layout used
//! by additive opcodes 14/15 covers CREATE, RENAME, and multi-target DROP without weakening the
//! old decoder or teaching replay to infer identity from SQL text.

use super::*;

const CATALOG_RELATION_TABLE: u8 = 1;
const CATALOG_RELATION_VIEW: u8 = 2;

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

/// Accepted opcode-12/13 identity layout for CREATE VIEW / CREATE OR REPLACE VIEW.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionViewOperationIdentity {
    pub(crate) command_index: u32,
    pub(crate) ordinal: u32,
    pub(crate) target_before: Option<BinaryCatalogRelationIdentity>,
    pub(crate) dependencies: BTreeMap<String, BinaryCatalogRelationIdentity>,
    pub(crate) target_after: BinaryCatalogRelationIdentity,
}

/// One named target transition inside a stored-view lifecycle command.
///
/// `after_name == None` is a typed absence postcondition (DROP). A missing `target_before` with
/// no dependencies is permitted only for `DROP VIEW IF EXISTS`; command-specific validation
/// enforces that distinction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionViewLifecycleTargetIdentity {
    pub(crate) before_name: String,
    pub(crate) target_before: Option<BinaryCatalogRelationIdentity>,
    pub(crate) dependencies: BTreeMap<String, BinaryCatalogRelationIdentity>,
    pub(crate) after_name: Option<String>,
    pub(crate) target_after: Option<BinaryCatalogRelationIdentity>,
}

/// Per-command identity closure for CREATE/RENAME/DROP VIEW. A vector is required because DROP
/// may name multiple targets and because one transaction may transition the same binding more
/// than once at different statement ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionViewLifecycleOperationIdentity {
    pub(crate) command_index: u32,
    pub(crate) ordinal: u32,
    pub(crate) targets: Vec<BinaryTransactionViewLifecycleTargetIdentity>,
}

pub(crate) fn command_is_view_lifecycle(command: &Command) -> bool {
    matches!(
        command,
        Command::CreateView(_) | Command::RenameView(_) | Command::DropView(_)
    )
}

pub(super) fn command_requires_view_lifecycle_opcode(command: &Command) -> bool {
    matches!(command, Command::RenameView(_) | Command::DropView(_))
}

fn valid_catalog_identity(identity: &BinaryCatalogRelationIdentity) -> bool {
    identity.oid != 0 && identity.digest != [0; 32]
}

pub(super) fn valid_legacy_view_operation_identity(
    identity: &BinaryTransactionViewOperationIdentity,
) -> bool {
    identity.target_before.as_ref().is_none_or(|before| {
        before.kind == BinaryCatalogRelationKind::View && valid_catalog_identity(before)
    }) && identity.target_after.kind == BinaryCatalogRelationKind::View
        && valid_catalog_identity(&identity.target_after)
        && !identity.dependencies.is_empty()
        && valid_dependencies(&identity.dependencies)
}

fn valid_dependencies(dependencies: &BTreeMap<String, BinaryCatalogRelationIdentity>) -> bool {
    dependencies.len() <= u32::MAX as usize
        && dependencies.iter().all(|(name, dependency)| {
            !name.is_empty()
                && name.len() <= u16::MAX as usize
                && valid_catalog_identity(dependency)
        })
}

fn valid_lifecycle_target(identity: &BinaryTransactionViewLifecycleTargetIdentity) -> bool {
    !identity.before_name.is_empty()
        && identity.before_name.len() <= u16::MAX as usize
        && identity.target_before.as_ref().is_none_or(|before| {
            before.kind == BinaryCatalogRelationKind::View && valid_catalog_identity(before)
        })
        && valid_dependencies(&identity.dependencies)
        && identity.after_name.is_some() == identity.target_after.is_some()
        && identity
            .after_name
            .as_ref()
            .is_none_or(|name| !name.is_empty() && name.len() <= u16::MAX as usize)
        && identity.target_after.as_ref().is_none_or(|after| {
            after.kind == BinaryCatalogRelationKind::View && valid_catalog_identity(after)
        })
}

pub(crate) fn valid_view_lifecycle_operation_identity(
    command: &Command,
    identity: &BinaryTransactionViewLifecycleOperationIdentity,
) -> bool {
    if identity.targets.is_empty()
        || identity.targets.len() > u32::MAX as usize
        || !identity.targets.iter().all(valid_lifecycle_target)
    {
        return false;
    }
    match command {
        Command::CreateView(create) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == create.name
                && target.after_name.as_deref() == Some(create.name.as_str())
                && target.target_after.is_some()
                && !target.dependencies.is_empty()
                && (create.or_replace || target.target_before.is_none())
        }
        Command::RenameView(rename) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == rename.old_name
                && target.after_name.as_deref() == Some(rename.new_name.as_str())
                && !target.dependencies.is_empty()
                && target
                    .target_before
                    .as_ref()
                    .zip(target.target_after.as_ref())
                    .is_some_and(|(before, after)| before.oid == after.oid)
        }
        Command::DropView(drop) => {
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
                        && match &target.target_before {
                            Some(_) => !target.dependencies.is_empty(),
                            None => drop.if_exists && target.dependencies.is_empty(),
                        }
                })
        }
        _ => false,
    }
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

fn encode_string(out: &mut Vec<u8>, value: &str) -> Option<()> {
    let len = u16::try_from(value.len()).ok()?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Some(())
}

fn encode_dependencies(
    out: &mut Vec<u8>,
    dependencies: &BTreeMap<String, BinaryCatalogRelationIdentity>,
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(dependencies.len()).ok()?.to_le_bytes());
    for (name, dependency) in dependencies {
        encode_string(out, name)?;
        encode_catalog_identity(out, dependency);
    }
    Some(())
}

pub(super) fn encode_legacy_view_operations(
    out: &mut Vec<u8>,
    identities: &[BinaryTransactionViewOperationIdentity],
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(identities.len()).ok()?.to_le_bytes());
    for identity in identities {
        out.extend_from_slice(&identity.command_index.to_le_bytes());
        out.extend_from_slice(&identity.ordinal.to_le_bytes());
        encode_optional_catalog_identity(out, identity.target_before.as_ref());
        encode_dependencies(out, &identity.dependencies)?;
        encode_catalog_identity(out, &identity.target_after);
    }
    Some(())
}

pub(super) fn encode_view_lifecycle_operations(
    out: &mut Vec<u8>,
    identities: &[BinaryTransactionViewLifecycleOperationIdentity],
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(identities.len()).ok()?.to_le_bytes());
    for identity in identities {
        out.extend_from_slice(&identity.command_index.to_le_bytes());
        out.extend_from_slice(&identity.ordinal.to_le_bytes());
        out.extend_from_slice(&u32::try_from(identity.targets.len()).ok()?.to_le_bytes());
        for target in &identity.targets {
            encode_string(out, &target.before_name)?;
            encode_optional_catalog_identity(out, target.target_before.as_ref());
            encode_dependencies(out, &target.dependencies)?;
            match (&target.after_name, &target.target_after) {
                (Some(name), Some(after)) => {
                    out.push(1);
                    encode_string(out, name)?;
                    encode_catalog_identity(out, after);
                }
                (None, None) => out.push(0),
                _ => return None,
            }
        }
    }
    Some(())
}

fn decode_catalog_identity<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryCatalogRelationIdentity, EngineError> {
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

fn decode_optional_catalog_identity<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<Option<BinaryCatalogRelationIdentity>, EngineError> {
    match take(1)?[0] {
        0 => Ok(None),
        1 => decode_catalog_identity(take, fail).map(Some),
        _ => Err(fail("invalid optional catalog identity flag")),
    }
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

fn decode_dependencies<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BTreeMap<String, BinaryCatalogRelationIdentity>, EngineError> {
    let count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    let mut dependencies = BTreeMap::new();
    let mut prior_name = None;
    for _ in 0..count {
        let name = decode_string(take, fail, "non-utf8 view dependency name")?;
        let dependency = decode_catalog_identity(take, fail)?;
        if name.is_empty()
            || prior_name.as_ref().is_some_and(|prior| prior >= &name)
            || dependencies.insert(name.clone(), dependency).is_some()
        {
            return Err(fail("non-canonical view dependency identity"));
        }
        prior_name = Some(name);
    }
    Ok(dependencies)
}

pub(super) fn decode_legacy_view_operation<'a>(
    command_index: u32,
    ordinal: u32,
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryTransactionViewOperationIdentity, EngineError> {
    let identity = BinaryTransactionViewOperationIdentity {
        command_index,
        ordinal,
        target_before: decode_optional_catalog_identity(take, fail)?,
        dependencies: decode_dependencies(take, fail)?,
        target_after: decode_catalog_identity(take, fail)?,
    };
    if !valid_legacy_view_operation_identity(&identity) {
        return Err(fail("invalid CREATE VIEW identity closure"));
    }
    Ok(identity)
}

pub(super) fn decode_view_lifecycle_operation<'a>(
    command_index: u32,
    ordinal: u32,
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryTransactionViewLifecycleOperationIdentity, EngineError> {
    let target_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    if target_count == 0 {
        return Err(fail("view lifecycle identity has no targets"));
    }
    let mut targets = Vec::with_capacity(target_count.min(1024));
    for _ in 0..target_count {
        let before_name = decode_string(take, fail, "non-utf8 view target name")?;
        let target_before = decode_optional_catalog_identity(take, fail)?;
        let dependencies = decode_dependencies(take, fail)?;
        let (after_name, target_after) = match take(1)?[0] {
            0 => (None, None),
            1 => (
                Some(decode_string(
                    take,
                    fail,
                    "non-utf8 view target postimage name",
                )?),
                Some(decode_catalog_identity(take, fail)?),
            ),
            _ => return Err(fail("invalid view target postimage flag")),
        };
        targets.push(BinaryTransactionViewLifecycleTargetIdentity {
            before_name,
            target_before,
            dependencies,
            after_name,
            target_after,
        });
    }
    Ok(BinaryTransactionViewLifecycleOperationIdentity {
        command_index,
        ordinal,
        targets,
    })
}

pub(crate) fn lifecycle_from_legacy_create(
    create: &CreateView,
    identity: &BinaryTransactionViewOperationIdentity,
) -> BinaryTransactionViewLifecycleOperationIdentity {
    BinaryTransactionViewLifecycleOperationIdentity {
        command_index: identity.command_index,
        ordinal: identity.ordinal,
        targets: vec![BinaryTransactionViewLifecycleTargetIdentity {
            before_name: create.name.clone(),
            target_before: identity.target_before.clone(),
            dependencies: identity.dependencies.clone(),
            after_name: Some(create.name.clone()),
            target_after: Some(identity.target_after.clone()),
        }],
    }
}

pub(crate) fn legacy_from_lifecycle_create(
    create: &CreateView,
    identity: &BinaryTransactionViewLifecycleOperationIdentity,
) -> Option<BinaryTransactionViewOperationIdentity> {
    if !valid_view_lifecycle_operation_identity(&Command::CreateView(create.clone()), identity) {
        return None;
    }
    let target = identity.targets.first()?;
    Some(BinaryTransactionViewOperationIdentity {
        command_index: identity.command_index,
        ordinal: identity.ordinal,
        target_before: target.target_before.clone(),
        dependencies: target.dependencies.clone(),
        target_after: target.target_after.clone()?,
    })
}
