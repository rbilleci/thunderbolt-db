//! Typed index identity codecs for ordered transaction WAL.
//!
//! Additive opcodes 16/17 bind every CREATE/RENAME/multi-target DROP INDEX target to a
//! catalog-stable index OID, its owning table identity, exact pre/post shape, and typed absence.

use super::*;

const CATALOG_RELATION_TABLE: u8 = 1;
/// Additive PRODUCT-001 records preserve every earlier layout while binding stable index
/// identities, exact owning-table transitions, and the shared `pg_class.oid` allocator proof.
pub(super) const OP_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION: u8 = 16;
pub(super) const OP_IDENTITY_ORDERED_CATALOG_INDEX_LIFECYCLE_TRANSACTION: u8 = 17;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryCatalogIndexIdentity {
    pub(crate) oid: u32,
    pub(crate) table_oid: u32,
    pub(crate) digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionIndexLifecycleTargetIdentity {
    pub(crate) before_name: String,
    pub(crate) owner_name: Option<String>,
    pub(crate) table_before: Option<BinaryCatalogRelationIdentity>,
    pub(crate) index_before: Option<BinaryCatalogIndexIdentity>,
    pub(crate) after_name: Option<String>,
    pub(crate) table_after: Option<BinaryCatalogRelationIdentity>,
    pub(crate) index_after: Option<BinaryCatalogIndexIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryTransactionIndexLifecycleOperationIdentity {
    pub(crate) command_index: u32,
    pub(crate) ordinal: u32,
    pub(crate) targets: Vec<BinaryTransactionIndexLifecycleTargetIdentity>,
}

pub(crate) fn command_is_index_lifecycle(command: &Command) -> bool {
    matches!(
        command,
        Command::CreateIndex(_) | Command::RenameIndex(_) | Command::DropIndex(_)
    )
}

/// Select the additive index-aware ordered-catalog envelope. Besides explicit lifecycle
/// statements, transactional CREATE TABLE needs this envelope when it allocates implicit
/// primary/unique index OIDs, so the allocator postimage is durable and replay-verifiable.
pub(crate) fn command_requires_index_catalog_opcode(command: &Command) -> bool {
    command_is_index_lifecycle(command)
        || matches!(
            command,
            Command::CreateTable(create)
                if create.primary_key.is_some() || !create.unique_constraints.is_empty()
        )
        || matches!(
            command,
            Command::AddPrimaryKey(_) | Command::AddUniqueConstraint(_)
        )
}

fn valid_relation(identity: &BinaryCatalogRelationIdentity) -> bool {
    identity.kind == BinaryCatalogRelationKind::Table
        && identity.oid != 0
        && identity.digest != [0; 32]
}

fn valid_index(identity: &BinaryCatalogIndexIdentity) -> bool {
    identity.oid != 0 && identity.table_oid != 0 && identity.digest != [0; 32]
}

fn valid_before(
    table: Option<&BinaryCatalogRelationIdentity>,
    index: Option<&BinaryCatalogIndexIdentity>,
) -> bool {
    match (table, index) {
        (Some(table), Some(index)) => {
            valid_relation(table) && valid_index(index) && table.oid == index.table_oid
        }
        (Some(table), None) => valid_relation(table),
        (None, None) => true,
        (None, Some(_)) => false,
    }
}

fn valid_pair(
    table: Option<&BinaryCatalogRelationIdentity>,
    index: Option<&BinaryCatalogIndexIdentity>,
) -> bool {
    table.is_some()
        && index.is_some()
        && table.zip(index).is_some_and(|(table, index)| {
            valid_relation(table) && valid_index(index) && table.oid == index.table_oid
        })
}

fn valid_target(target: &BinaryTransactionIndexLifecycleTargetIdentity) -> bool {
    !target.before_name.is_empty()
        && target.before_name.len() <= u16::MAX as usize
        && target.owner_name.is_some()
            == (target.table_before.is_some() || target.table_after.is_some())
        && target
            .owner_name
            .as_ref()
            .is_none_or(|name| !name.is_empty() && name.len() <= u16::MAX as usize)
        && valid_before(target.table_before.as_ref(), target.index_before.as_ref())
        && target
            .after_name
            .as_ref()
            .is_none_or(|name| !name.is_empty() && name.len() <= u16::MAX as usize)
        && match (
            target.after_name.as_ref(),
            target.table_after.as_ref(),
            target.index_after.as_ref(),
        ) {
            (Some(_), Some(table), Some(index)) => valid_pair(Some(table), Some(index)),
            (None, Some(table), None) => valid_relation(table),
            (None, None, None) => true,
            _ => false,
        }
}

pub(crate) fn valid_index_lifecycle_operation_identity(
    command: &Command,
    identity: &BinaryTransactionIndexLifecycleOperationIdentity,
) -> bool {
    if identity.targets.is_empty()
        || identity.targets.len() > u32::MAX as usize
        || !identity.targets.iter().all(valid_target)
    {
        return false;
    }
    match command {
        Command::CreateIndex(create) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == create.name
                && target.owner_name.as_deref() == Some(create.table.as_str())
                && target.table_before.is_some()
                && target.index_before.is_none()
                && target.after_name.as_deref() == Some(create.name.as_str())
                && target
                    .table_before
                    .as_ref()
                    .zip(target.table_after.as_ref())
                    .is_some_and(|(before, after)| before.oid == after.oid)
                && target.index_after.is_some()
        }
        Command::RenameIndex(rename) => {
            let [target] = identity.targets.as_slice() else {
                return false;
            };
            target.before_name == rename.old_name
                && target.after_name.as_deref() == Some(rename.new_name.as_str())
                && target
                    .table_before
                    .as_ref()
                    .zip(target.table_after.as_ref())
                    .is_some_and(|(before, after)| before.oid == after.oid)
                && target
                    .index_before
                    .as_ref()
                    .zip(target.index_after.as_ref())
                    .is_some_and(|(before, after)| {
                        before.oid == after.oid && before.table_oid == after.table_oid
                    })
        }
        Command::DropIndex(drop) => {
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
                        && target.index_after.is_none()
                        && match (
                            target.table_before.as_ref(),
                            target.index_before.as_ref(),
                            target.table_after.as_ref(),
                        ) {
                            (Some(before), Some(_), Some(after)) => before.oid == after.oid,
                            (None, None, None) => drop.if_exists,
                            _ => false,
                        }
                })
        }
        _ => false,
    }
}

fn encode_relation(out: &mut Vec<u8>, identity: &BinaryCatalogRelationIdentity) {
    out.push(CATALOG_RELATION_TABLE);
    out.extend_from_slice(&identity.oid.to_le_bytes());
    out.extend_from_slice(&identity.digest);
}

fn encode_index(out: &mut Vec<u8>, identity: &BinaryCatalogIndexIdentity) {
    out.extend_from_slice(&identity.oid.to_le_bytes());
    out.extend_from_slice(&identity.table_oid.to_le_bytes());
    out.extend_from_slice(&identity.digest);
}

fn encode_optional_pair(
    out: &mut Vec<u8>,
    table: Option<&BinaryCatalogRelationIdentity>,
    index: Option<&BinaryCatalogIndexIdentity>,
) -> Option<()> {
    match (table, index) {
        (Some(table), Some(index)) => {
            out.push(1);
            encode_relation(out, table);
            encode_index(out, index);
        }
        (Some(table), None) => {
            out.push(2);
            encode_relation(out, table);
        }
        (None, None) => out.push(0),
        _ => return None,
    }
    Some(())
}

fn encode_string(out: &mut Vec<u8>, value: &str) -> Option<()> {
    out.extend_from_slice(&u16::try_from(value.len()).ok()?.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Some(())
}

pub(super) fn encode_index_lifecycle_operations(
    out: &mut Vec<u8>,
    identities: &[BinaryTransactionIndexLifecycleOperationIdentity],
) -> Option<()> {
    out.extend_from_slice(&u32::try_from(identities.len()).ok()?.to_le_bytes());
    for identity in identities {
        out.extend_from_slice(&identity.command_index.to_le_bytes());
        out.extend_from_slice(&identity.ordinal.to_le_bytes());
        out.extend_from_slice(&u32::try_from(identity.targets.len()).ok()?.to_le_bytes());
        for target in &identity.targets {
            encode_string(out, &target.before_name)?;
            match &target.owner_name {
                Some(name) => {
                    out.push(1);
                    encode_string(out, name)?;
                }
                None => out.push(0),
            }
            encode_optional_pair(
                out,
                target.table_before.as_ref(),
                target.index_before.as_ref(),
            )?;
            match (&target.after_name, &target.table_after, &target.index_after) {
                (Some(name), Some(table), Some(index)) => {
                    out.push(1);
                    encode_string(out, name)?;
                    encode_relation(out, table);
                    encode_index(out, index);
                }
                (None, Some(table), None) => {
                    out.push(2);
                    encode_relation(out, table);
                }
                (None, None, None) => out.push(0),
                _ => return None,
            }
        }
    }
    Some(())
}

fn decode_relation<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryCatalogRelationIdentity, EngineError> {
    if take(1)?[0] != CATALOG_RELATION_TABLE {
        return Err(fail("index owner identity is not a table"));
    }
    let identity = BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Table,
        oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
        digest: take(32)?.try_into().expect("32 bytes"),
    };
    valid_relation(&identity)
        .then_some(identity)
        .ok_or_else(|| fail("empty index owner identity"))
}

fn decode_index<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryCatalogIndexIdentity, EngineError> {
    let identity = BinaryCatalogIndexIdentity {
        oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
        table_oid: u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")),
        digest: take(32)?.try_into().expect("32 bytes"),
    };
    valid_index(&identity)
        .then_some(identity)
        .ok_or_else(|| fail("empty index identity"))
}

fn decode_pair<'a>(
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<
    (
        Option<BinaryCatalogRelationIdentity>,
        Option<BinaryCatalogIndexIdentity>,
    ),
    EngineError,
> {
    match take(1)?[0] {
        0 => Ok((None, None)),
        1 => {
            let table = decode_relation(take, fail)?;
            let index = decode_index(take, fail)?;
            if table.oid != index.table_oid {
                return Err(fail("index identity changed owning table"));
            }
            Ok((Some(table), Some(index)))
        }
        2 => Ok((Some(decode_relation(take, fail)?), None)),
        _ => Err(fail("invalid optional index identity flag")),
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

pub(super) fn decode_index_lifecycle_operation<'a>(
    command_index: u32,
    ordinal: u32,
    take: &mut impl FnMut(usize) -> Result<&'a [u8], EngineError>,
    fail: &impl Fn(&str) -> EngineError,
) -> Result<BinaryTransactionIndexLifecycleOperationIdentity, EngineError> {
    let target_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
    if target_count == 0 {
        return Err(fail("index lifecycle identity has no targets"));
    }
    let mut targets = Vec::with_capacity(target_count.min(1024));
    for _ in 0..target_count {
        let before_name = decode_string(take, fail, "non-utf8 index target name")?;
        let owner_name = match take(1)?[0] {
            0 => None,
            1 => Some(decode_string(take, fail, "non-utf8 index owner name")?),
            _ => return Err(fail("invalid index owner name flag")),
        };
        let (table_before, index_before) = decode_pair(take, fail)?;
        let (after_name, table_after, index_after) = match take(1)?[0] {
            0 => (None, None, None),
            1 => {
                let name = decode_string(take, fail, "non-utf8 index postimage name")?;
                let table = decode_relation(take, fail)?;
                let index = decode_index(take, fail)?;
                if table.oid != index.table_oid {
                    return Err(fail("index postimage changed owning table"));
                }
                (Some(name), Some(table), Some(index))
            }
            2 => (None, Some(decode_relation(take, fail)?), None),
            _ => return Err(fail("invalid index postimage flag")),
        };
        targets.push(BinaryTransactionIndexLifecycleTargetIdentity {
            before_name,
            owner_name,
            table_before,
            index_before,
            after_name,
            table_after,
            index_after,
        });
    }
    Ok(BinaryTransactionIndexLifecycleOperationIdentity {
        command_index,
        ordinal,
        targets,
    })
}
