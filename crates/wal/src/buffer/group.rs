//! Bounded, reusable scatter ownership for one WAL durability group.
//!
//! The descriptor arena is deliberately separate from `WalBuffer`'s logical history: moving a
//! claimed exact record here transfers its one immutable serialized owner into the I/O job, while
//! legacy records retain only a fixed header and their existing payload `Arc`.  No record-byte
//! staging buffer is permitted in this leaf.

use super::typed_exact::ExactTypedWireRecord;
use super::{DurableIdentityBinding, WalBuffer, WalDurableCore};
use crate::{CanonicalIdentity, EngineError, WalRecord, WAL_RECORD_HEADER_LEN};
use gpu_db_types::{DurabilityBackend, DurabilityFault, DurabilityStage};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use std::cell::RefCell;

#[cfg(unix)]
use gpu_db_write_conveyor::{
    FuaFrameFault, FuaFrameFaultStage, FuaScatterSource, FUA_FRAME_FAULT_NO_FRAME,
};

#[cfg(test)]
type SerialWriteHook = Box<dyn FnMut(&[libc::iovec], u64) -> SerialIoTestWrite>;

#[cfg(test)]
type SerialSyncHook = Box<dyn FnMut() -> SerialIoTestSync>;

/// A canonical record is bounded by the 64 MiB envelope plus the format's 1 MiB packed framing
/// allowance and its fixed outer storage header.  Legacy records must fit the same live group
/// bound; their compatibility route otherwise fails before any group handoff.
pub(crate) const MAX_WAL_GROUP_WIRE_BYTES: usize = 65 * 1024 * 1024 + WAL_RECORD_HEADER_LEN;
pub(crate) const MAX_WAL_GROUP_RECORDS: usize = 1_024;
const GROUP_DESCRIPTOR_SLOTS: usize = 2;
const MAX_WRITE_IOVECS: usize = 128;
const SERIAL_SEGMENT_ID: u64 = 0;

/// Test-only syscall seam.  Production always invokes positional `pwritev` and `sync_data`; the
/// hook runs on the committing thread and makes partial writes, EINTR, zero, and sync failures
/// deterministic without weakening the live durability route.
#[cfg(test)]
pub(super) struct SerialIoTestHooks {
    pub(super) write: SerialWriteHook,
    pub(super) sync: SerialSyncHook,
    pub(super) before_fixed_fault_wake: Option<Box<dyn FnMut(DurabilityFault)>>,
}

/// A test syscall action carries only fixed scalar state so allocation coverage can exercise the
/// same hook dispatch as the serial post-handoff path.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) enum SerialIoTestWrite {
    Written(usize),
    Errno(i32),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) enum SerialIoTestSync {
    Complete,
    Errno(i32),
}

#[cfg(test)]
thread_local! {
    static SERIAL_IO_TEST_HOOKS: RefCell<Option<SerialIoTestHooks>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[must_use]
pub(super) struct SerialIoTestHookGuard;

#[cfg(test)]
impl Drop for SerialIoTestHookGuard {
    fn drop(&mut self) {
        SERIAL_IO_TEST_HOOKS.with(|hooks| {
            hooks
                .borrow_mut()
                .take()
                .expect("serial IO test hook guard lost its hook");
        });
    }
}

#[cfg(test)]
pub(super) fn install_serial_io_hooks_for_test(hooks: SerialIoTestHooks) -> SerialIoTestHookGuard {
    SERIAL_IO_TEST_HOOKS.with(|current| {
        assert!(
            current.borrow().is_none(),
            "nested serial IO test hooks are not supported"
        );
        *current.borrow_mut() = Some(hooks);
    });
    SerialIoTestHookGuard
}

/// A preflight-selected inclusive/exclusive logical record range.  It contains no byte owner and
/// can therefore be computed before an exact owner leaves its rollbackable sidecar.
#[derive(Debug, Clone, Copy)]
pub(super) struct WalGroupPrefix {
    pub(super) first_record: usize,
    pub(super) target_records: usize,
    pub(super) wire_bytes: usize,
    pub(super) exact_records: usize,
}

impl WalGroupPrefix {
    pub(super) fn group_size(self) -> usize {
        self.target_records - self.first_record
    }
}

#[derive(Debug)]
pub(super) struct TypedExactPendingCredit {
    credits: Arc<Mutex<TypedExactCreditState>>,
    bytes: usize,
    active: bool,
}

impl TypedExactPendingCredit {
    fn release(&mut self) {
        if !self.active {
            return;
        }
        let mut state = self
            .credits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.records = state
            .records
            .checked_sub(1)
            .expect("typed exact group-credit record underflow");
        state.bytes = state
            .bytes
            .checked_sub(self.bytes)
            .expect("typed exact group-credit byte underflow");
        self.active = false;
    }
}

impl Drop for TypedExactPendingCredit {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug, Default)]
struct TypedExactCreditState {
    records: usize,
    bytes: usize,
}

#[derive(Debug)]
enum GroupSlotState {
    Idle(Vec<GroupEntry>),
    Busy,
    /// Keep the permanently provisioned descriptor backing even after fail-stop poison.  The
    /// next recovery-created buffer gets a fresh arena, while this wedged buffer never turns its
    /// bounded slot capacity into an unaccounted deallocation/reallocation cycle.
    Poisoned {
        entries: Vec<GroupEntry>,
        fault: DurabilityFault,
    },
}

#[derive(Debug)]
struct GroupSlot {
    state: Mutex<GroupSlotState>,
}

impl GroupSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(GroupSlotState::Idle(Vec::with_capacity(
                MAX_WAL_GROUP_RECORDS,
            ))),
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Result<Option<PreparedWalGroup>, EngineError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match std::mem::replace(&mut *state, GroupSlotState::Busy) {
            GroupSlotState::Idle(entries) => Ok(Some(PreparedWalGroup {
                slot: Arc::clone(self),
                entries: Some(entries),
                wire_bytes: 0,
                first_record: 0,
                target_records: 0,
                contains_exact: false,
            })),
            GroupSlotState::Busy => {
                *state = GroupSlotState::Busy;
                Ok(None)
            }
            GroupSlotState::Poisoned { entries, fault } => {
                *state = GroupSlotState::Poisoned { entries, fault };
                Err(EngineError::DurabilityFault(fault))
            }
        }
    }
}

/// Two permanently allocated descriptors plus the scalar exact-admission credits.  Two groups
/// are intentionally available: a serial write may be in flight while the next bounded group is
/// forming, so a claimed typed owner never discovers descriptor or byte exhaustion after WAL.
#[derive(Debug)]
pub(super) struct WalPreparedGroupArena {
    slots: [Arc<GroupSlot>; GROUP_DESCRIPTOR_SLOTS],
    typed_exact_credits: Arc<Mutex<TypedExactCreditState>>,
}

impl Default for WalPreparedGroupArena {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Arc::new(GroupSlot::new())),
            typed_exact_credits: Arc::new(Mutex::new(TypedExactCreditState::default())),
        }
    }
}

impl WalPreparedGroupArena {
    pub(super) fn reserve_typed_exact_credit(
        &self,
        bytes: usize,
    ) -> Result<TypedExactPendingCredit, EngineError> {
        if bytes > MAX_WAL_GROUP_WIRE_BYTES {
            return Err(EngineError::ProposalFailed(format!(
                "typed exact WAL record of {bytes} bytes exceeds the configured {MAX_WAL_GROUP_WIRE_BYTES}-byte group bound"
            )));
        }
        let mut state = self
            .typed_exact_credits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let max_records = MAX_WAL_GROUP_RECORDS
            .checked_mul(GROUP_DESCRIPTOR_SLOTS)
            .expect("configured WAL group-record bound overflow");
        let max_bytes = MAX_WAL_GROUP_WIRE_BYTES
            .checked_mul(GROUP_DESCRIPTOR_SLOTS)
            .expect("configured WAL group-byte bound overflow");
        let next_records = state.records.checked_add(1).ok_or_else(|| {
            EngineError::ProposalFailed("typed exact WAL group-record credit overflow".to_string())
        })?;
        let next_bytes = state.bytes.checked_add(bytes).ok_or_else(|| {
            EngineError::ProposalFailed("typed exact WAL group-byte credit overflow".to_string())
        })?;
        if next_records > max_records || next_bytes > max_bytes {
            return Err(EngineError::ProposalFailed(format!(
                "typed exact WAL group credit is busy (pending records={}/{}, bytes={}/{})",
                state.records, state.bytes, max_records, max_bytes,
            )));
        }
        state.records = next_records;
        state.bytes = next_bytes;
        Ok(TypedExactPendingCredit {
            credits: Arc::clone(&self.typed_exact_credits),
            bytes,
            active: true,
        })
    }

    pub(super) fn try_acquire(&self) -> Result<Option<PreparedWalGroup>, EngineError> {
        for slot in &self.slots {
            if let Some(group) = slot.try_acquire()? {
                return Ok(Some(group));
            }
        }
        Ok(None)
    }
}

impl WalBuffer {
    /// Compute a bounded group prefix before moving an exact owner.  A prefix containing exact
    /// owners stops before later legacy appends: those bytes were not part of the exact
    /// reservation's lineage/extent proof and belong to the next group.
    pub(super) fn select_prepared_group_prefix(
        &self,
        first_record: usize,
        serial_free_bytes: Option<usize>,
    ) -> Result<Option<WalGroupPrefix>, EngineError> {
        if first_record >= self.records.len() {
            return Ok(None);
        }
        let mut target_records = first_record;
        let mut wire_bytes = 0usize;
        let mut includes_exact = false;
        while target_records < self.records.len()
            && target_records - first_record < MAX_WAL_GROUP_RECORDS
        {
            let record = &self.records[target_records];
            let exact = self.exact_wire_for_group(target_records, record)?;
            if includes_exact && exact.is_none() {
                break;
            }
            let record_bytes = match exact {
                Some(exact) => exact.serialized_len(),
                None => WAL_RECORD_HEADER_LEN
                    .checked_add(record.payload.len())
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "WAL legacy record byte length overflows group accounting".to_string(),
                        )
                    })?,
            };
            let next_bytes = wire_bytes.checked_add(record_bytes).ok_or_else(|| {
                EngineError::Durability("WAL group byte accounting overflows".to_string())
            })?;
            let exceeds_extent = serial_free_bytes.is_some_and(|free| next_bytes > free);
            if next_bytes > MAX_WAL_GROUP_WIRE_BYTES || exceeds_extent {
                if target_records == first_record {
                    return Err(EngineError::Durability(format!(
                        "WAL record of {record_bytes} bytes exceeds its configured group or preallocated serial extent bound"
                    )));
                }
                break;
            }
            wire_bytes = next_bytes;
            includes_exact |= exact.is_some();
            target_records += 1;
        }
        let exact_records = self.exact_wire_prefix_count(target_records)?;
        Ok(Some(WalGroupPrefix {
            first_record,
            target_records,
            wire_bytes,
            exact_records,
        }))
    }

    /// Infallibly transfer the selected claimed exact owners into a permanent scatter descriptor.
    /// `None` is bounded backpressure, not a post-claim capacity failure.
    pub(super) fn handoff_prepared_group(
        &mut self,
        prefix: WalGroupPrefix,
    ) -> Result<Option<PreparedWalGroup>, EngineError> {
        let Some(mut group) = self.prepared_group_arena.try_acquire()? else {
            return Ok(None);
        };
        let exact_count = self.exact_wire_prefix_count(prefix.target_records)?;
        if exact_count != prefix.exact_records {
            return Err(EngineError::Durability(
                "WAL exact-owner prefix drifted after group preflight".to_string(),
            ));
        }
        let records = &self.records;
        let wires = &mut self.exact_typed_wire_records;
        let mut exact = wires.drain(..exact_count);
        for (index, record) in records
            .iter()
            .enumerate()
            .take(prefix.target_records)
            .skip(prefix.first_record)
        {
            if exact
                .as_slice()
                .first()
                .is_some_and(|wire| wire.record_index() == index)
            {
                let mut wire = exact.next().expect("checked exact WAL group owner");
                wire.mark_in_flight();
                group.push_exact(wire);
            } else {
                group.push_legacy(record);
            }
        }
        debug_assert!(
            exact.next().is_none(),
            "group prefix left exact owner behind"
        );
        drop(exact);
        group.finish_formation(prefix);
        if prefix.exact_records != 0 {
            self.typed_exact_handoff_cursor =
                self.typed_exact_handoff_cursor.max(prefix.target_records);
        }
        Ok(Some(group))
    }

    /// Count the current logical tail from its immutable owners.  An in-flight exact owner has
    /// left the sparse sidecar, but `24 + packed payload` is its exact serialized byte length.
    fn current_tail_wire_bytes_for_admission(
        &self,
        first_record: usize,
    ) -> Result<u64, EngineError> {
        let mut bytes = 0u64;
        for index in first_record..self.records.len() {
            let record = &self.records[index];
            let record_bytes = match self
                .exact_typed_wire_records
                .binary_search_by_key(&index, |wire| wire.record_index())
            {
                Ok(wire) => self.exact_typed_wire_records[wire].serialized_len(),
                Err(_) => WAL_RECORD_HEADER_LEN
                    .checked_add(record.payload.len())
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "WAL admission legacy record bytes overflow".to_string(),
                        )
                    })?,
            };
            bytes = bytes.checked_add(record_bytes as u64).ok_or_else(|| {
                EngineError::Durability("WAL admission tail bytes overflow".to_string())
            })?;
        }
        Ok(bytes)
    }

    fn validate_pending_legacy_group_geometry(
        &self,
        first_record: usize,
    ) -> Result<(), EngineError> {
        for index in first_record..self.records.len() {
            if self
                .exact_typed_wire_records
                .binary_search_by_key(&index, |wire| wire.record_index())
                .is_ok()
            {
                continue;
            }
            let record = &self.records[index];
            let bytes = WAL_RECORD_HEADER_LEN
                .checked_add(record.payload.len())
                .ok_or_else(|| {
                    EngineError::Durability(
                        "pending legacy WAL record byte length overflows group accounting"
                            .to_string(),
                    )
                })?;
            if bytes > MAX_WAL_GROUP_WIRE_BYTES {
                return Err(EngineError::ProposalFailed(format!(
                    "typed exact WAL admission is blocked by a pre-existing {bytes}-byte legacy record beyond the configured {MAX_WAL_GROUP_WIRE_BYTES}-byte group bound"
                )));
            }
        }
        Ok(())
    }

    fn preverify_typed_exact_lineage(
        &mut self,
        core: &super::WalDurableCore,
        exact_identity: CanonicalIdentity,
    ) -> Result<DurableIdentityBinding, EngineError> {
        self.durable_identity_binding.verify_through(
            &core.segment_path,
            &self.records,
            self.records.len(),
        )?;
        let prior = self.durable_identity_binding.clone();
        match prior.identity {
            Some(identity) if identity != exact_identity => {
                return Err(EngineError::Durability(
                    "typed exact WAL identity diverges from the preverified logical lineage"
                        .to_string(),
                ));
            }
            Some(identity) => {
                crate::identity::require_durable_identity(&core.segment_path, identity)?
            }
            None => crate::identity::check_or_install_durable_identity(
                &core.segment_path,
                exact_identity,
            )?,
        }
        Ok(prior)
    }

    /// Exact serial admission avoids the fdatasync wait in the common case.  It accounts for the
    /// full logical unflushed tail (including an in-flight group) and waits/recomputes only when
    /// the prospective extent genuinely needs to grow.
    pub(super) fn preflight_typed_exact_serial_admission(
        &mut self,
        expected_serialized_len: usize,
        exact_identity: CanonicalIdentity,
    ) -> Result<DurableIdentityBinding, EngineError> {
        let Some(core) = self.durable.clone() else {
            return Ok(self.durable_identity_binding.clone());
        };
        let mut state = core.lock_state();
        if let Some(fault) = core.fixed_fault() {
            return Err(EngineError::DurabilityFault(fault));
        }
        if let Some(reason) = state.poisoned.clone() {
            return Err(core.poisoned_error(&reason));
        }
        core.ensure_created(&mut state)?;
        let first = state.flushed_records;
        self.validate_pending_legacy_group_geometry(first)?;
        let pending = self.current_tail_wire_bytes_for_admission(first)?;
        let prospective_end = state
            .durable_bytes
            .checked_add(pending)
            .and_then(|end| end.checked_add(expected_serialized_len as u64))
            .ok_or_else(|| {
                EngineError::Durability(
                    "typed exact WAL prospective tail overflows u64".to_string(),
                )
            })?;
        if prospective_end > state.prealloc_bytes {
            drop(state);
            state = core.lock_state_idle();
            if let Some(fault) = core.fixed_fault() {
                return Err(EngineError::DurabilityFault(fault));
            }
            if let Some(reason) = state.poisoned.clone() {
                return Err(core.poisoned_error(&reason));
            }
            let first = state.flushed_records;
            self.validate_pending_legacy_group_geometry(first)?;
            let pending = self.current_tail_wire_bytes_for_admission(first)?;
            let prospective_end = state
                .durable_bytes
                .checked_add(pending)
                .and_then(|end| end.checked_add(expected_serialized_len as u64))
                .ok_or_else(|| {
                    EngineError::Durability(
                        "typed exact WAL prospective tail overflows after serial wait".to_string(),
                    )
                })?;
            core.ensure_preallocated_through(&mut state, prospective_end)?;
        }
        drop(state);
        self.preverify_typed_exact_lineage(&core, exact_identity)
    }

    #[cfg(unix)]
    pub(super) fn encode_fua_legacy_wire_record_into(
        &self,
        output: &mut Vec<u8>,
        index: usize,
    ) -> Result<(), EngineError> {
        let record = self.records.get(index).ok_or_else(|| {
            EngineError::Durability("FUA WAL record index exceeded logical frontier".to_string())
        })?;
        if self
            .exact_typed_wire_records
            .binary_search_by_key(&index, |wire| wire.record_index())
            .is_ok()
        {
            return Err(EngineError::Durability(
                "typed exact WAL owner reached the unsupported FUA frame encoder".to_string(),
            ));
        }
        crate::encode_record_into(output, record)
    }
}

#[derive(Debug)]
enum GroupEntry {
    Exact(ExactTypedWireRecord),
    Legacy {
        header: [u8; WAL_RECORD_HEADER_LEN],
        payload: Arc<[u8]>,
    },
}

impl GroupEntry {
    fn region(&self, part: usize) -> Option<&[u8]> {
        match self {
            Self::Exact(record) => (part == 0).then(|| record.serialized_bytes()),
            Self::Legacy { header, payload } => match part {
                0 => Some(header),
                1 => Some(payload),
                _ => None,
            },
        }
    }

    fn parts(&self) -> usize {
        match self {
            Self::Exact(_) => 1,
            // An empty legacy payload has no iovec and no physical cursor region.  Reporting
            // only its header keeps every scatter cursor over bytes that can actually be
            // offered or consumed.
            Self::Legacy { payload, .. } => 1 + usize::from(!payload.is_empty()),
        }
    }
}

// BEGIN SERIAL_EXACT_POST_HANDOFF_IO_NO_ALLOC
#[derive(Debug, Clone, Copy)]
enum SerialWriteOutcome {
    Written(usize),
    Interrupted,
    OffsetOverflow,
    Errno(i32),
}

#[derive(Debug, Clone, Copy)]
enum SerialSyncOutcome {
    Complete,
    Interrupted,
    Errno(i32),
}

fn pwritev_once(file: &std::fs::File, iovecs: &[libc::iovec], offset: u64) -> SerialWriteOutcome {
    #[cfg(test)]
    if let Some(result) = SERIAL_IO_TEST_HOOKS.with(|hooks| {
        hooks
            .borrow_mut()
            .as_mut()
            .map(|hook| (hook.write)(iovecs, offset))
    }) {
        return match result {
            SerialIoTestWrite::Written(written) => SerialWriteOutcome::Written(written),
            SerialIoTestWrite::Errno(raw_os_error) if raw_os_error == libc::EINTR => {
                SerialWriteOutcome::Interrupted
            }
            SerialIoTestWrite::Errno(raw_os_error) => SerialWriteOutcome::Errno(raw_os_error),
        };
    }
    let Ok(offset) = i64::try_from(offset) else {
        return SerialWriteOutcome::OffsetOverflow;
    };
    // SAFETY: the caller supplies initialized iovecs borrowing live immutable group owners for
    // the duration of this synchronous positional syscall.
    let written = unsafe {
        libc::pwritev(
            file.as_raw_fd(),
            iovecs.as_ptr(),
            iovecs.len() as libc::c_int,
            offset as libc::off_t,
        )
    };
    if written < 0 {
        let raw_os_error = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if raw_os_error == libc::EINTR {
            SerialWriteOutcome::Interrupted
        } else {
            SerialWriteOutcome::Errno(raw_os_error)
        }
    } else {
        SerialWriteOutcome::Written(written as usize)
    }
}

fn sync_data_once(file: &std::fs::File) -> SerialSyncOutcome {
    #[cfg(test)]
    if let Some(result) =
        SERIAL_IO_TEST_HOOKS.with(|hooks| hooks.borrow_mut().as_mut().map(|hook| (hook.sync)()))
    {
        return match result {
            SerialIoTestSync::Complete => SerialSyncOutcome::Complete,
            SerialIoTestSync::Errno(raw_os_error) if raw_os_error == libc::EINTR => {
                SerialSyncOutcome::Interrupted
            }
            SerialIoTestSync::Errno(raw_os_error) => SerialSyncOutcome::Errno(raw_os_error),
        };
    }
    match file.sync_data() {
        Ok(()) => SerialSyncOutcome::Complete,
        Err(error) => {
            let raw_os_error = error.raw_os_error().unwrap_or(0);
            if raw_os_error == libc::EINTR {
                SerialSyncOutcome::Interrupted
            } else {
                SerialSyncOutcome::Errno(raw_os_error)
            }
        }
    }
}

fn signal_fixed_fault_before_wake(_fault: DurabilityFault) {
    #[cfg(test)]
    SERIAL_IO_TEST_HOOKS.with(|hooks| {
        if let Some(hook) = hooks.borrow_mut().as_mut() {
            if let Some(before_wake) = hook.before_fixed_fault_wake.as_mut() {
                before_wake(_fault);
            }
        }
    });
}

fn sync_data_retry_interrupted(
    file: &std::fs::File,
    group: &PreparedWalGroup,
) -> Result<(), DurabilityFault> {
    loop {
        match sync_data_once(file) {
            SerialSyncOutcome::Complete => return Ok(()),
            SerialSyncOutcome::Interrupted => continue,
            SerialSyncOutcome::Errno(raw_os_error) => {
                return Err(group.fault(DurabilityStage::SyncData, Some(raw_os_error)));
            }
        }
    }
}
// END SERIAL_EXACT_POST_HANDOFF_IO_NO_ALLOC

/// One formed group.  The `Drop` implementation is deliberately fail-closed: after an exact
/// owner is handed off there is no return-to-tentative/re-encode path.
#[derive(Debug)]
pub(crate) struct PreparedWalGroup {
    slot: Arc<GroupSlot>,
    entries: Option<Vec<GroupEntry>>,
    wire_bytes: usize,
    first_record: usize,
    target_records: usize,
    contains_exact: bool,
}

impl PreparedWalGroup {
    pub(super) fn push_exact(&mut self, record: ExactTypedWireRecord) {
        let entries = self.entries.as_mut().expect("prepared WAL group is live");
        assert!(
            entries.len() < MAX_WAL_GROUP_RECORDS,
            "preflight allowed a WAL group beyond its descriptor capacity"
        );
        self.wire_bytes = self
            .wire_bytes
            .checked_add(record.serialized_len())
            .expect("preflight allowed a WAL group byte overflow");
        entries.push(GroupEntry::Exact(record));
    }

    pub(super) fn push_legacy(&mut self, record: &WalRecord) {
        let entries = self.entries.as_mut().expect("prepared WAL group is live");
        assert!(
            entries.len() < MAX_WAL_GROUP_RECORDS,
            "preflight allowed a WAL group beyond its descriptor capacity"
        );
        let mut header = [0; WAL_RECORD_HEADER_LEN];
        crate::encode_record_header_into(&mut header, record);
        self.wire_bytes = self
            .wire_bytes
            .checked_add(WAL_RECORD_HEADER_LEN + record.payload.len())
            .expect("preflight allowed a WAL group byte overflow");
        entries.push(GroupEntry::Legacy {
            header,
            payload: Arc::clone(&record.payload),
        });
    }

    pub(super) fn finish_formation(&mut self, prefix: WalGroupPrefix) {
        let entries = self.entries.as_ref().expect("prepared WAL group is live");
        assert_eq!(entries.len(), prefix.group_size());
        assert_eq!(self.wire_bytes, prefix.wire_bytes);
        self.first_record = prefix.first_record;
        self.target_records = prefix.target_records;
        self.contains_exact = prefix.exact_records != 0;
    }

    pub(crate) fn wire_bytes(&self) -> usize {
        self.wire_bytes
    }

    pub(super) fn contains_exact(&self) -> bool {
        self.contains_exact
    }

    #[cfg(unix)]
    pub(crate) fn contains_only_exact(&self) -> bool {
        self.contains_exact
            && self.entries.as_ref().is_some_and(|entries| {
                entries
                    .iter()
                    .all(|entry| matches!(entry, GroupEntry::Exact(_)))
            })
    }

    #[cfg(unix)]
    pub(crate) fn first_record(&self) -> usize {
        self.first_record
    }

    #[cfg(unix)]
    pub(crate) fn target_records(&self) -> usize {
        self.target_records
    }

    fn fault(&self, stage: DurabilityStage, raw_os_error: Option<i32>) -> DurabilityFault {
        DurabilityFault::new(
            DurabilityBackend::SerialWal,
            stage,
            raw_os_error,
            SERIAL_SEGMENT_ID,
            self.first_record as u64,
        )
    }

    #[cfg(test)]
    pub(super) fn first_exact_serialized_ptr_for_test(&self) -> Option<*const u8> {
        self.entries.as_ref()?.iter().find_map(|entry| match entry {
            GroupEntry::Exact(exact) => Some(exact.serialized_bytes().as_ptr()),
            GroupEntry::Legacy { .. } => None,
        })
    }

    /// Positional scatter write with an exact partial-write cursor.  The temporary iovec array is
    /// fixed-size stack storage; no group-byte buffer or post-claim allocator sits on this path.
    // BEGIN SERIAL_EXACT_POST_HANDOFF_NO_ALLOC
    pub(super) fn write_all_vectored_at(
        &self,
        file: &std::fs::File,
        offset: u64,
    ) -> Result<(), DurabilityFault> {
        let entries = self.entries.as_ref().expect("prepared WAL group is live");
        let mut entry = 0usize;
        let mut part = 0usize;
        let mut part_offset = 0usize;
        let mut file_offset = offset;
        let mut remaining = self.wire_bytes;
        while remaining != 0 {
            let mut iovecs: [MaybeUninit<libc::iovec>; MAX_WRITE_IOVECS] =
                [const { MaybeUninit::uninit() }; MAX_WRITE_IOVECS];
            let mut count = 0usize;
            let mut cursor_entry = entry;
            let mut cursor_part = part;
            let mut cursor_offset = part_offset;
            while cursor_entry < entries.len() && count < MAX_WRITE_IOVECS {
                let Some(current) = entries.get(cursor_entry) else {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                };
                let Some(bytes) = current.region(cursor_part) else {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                };
                let start = if cursor_entry == entry && cursor_part == part {
                    cursor_offset
                } else {
                    0
                };
                if start < bytes.len() {
                    iovecs[count].write(libc::iovec {
                        iov_base: bytes[start..].as_ptr() as *mut libc::c_void,
                        iov_len: bytes.len() - start,
                    });
                    count += 1;
                }
                cursor_part += 1;
                if cursor_part == current.parts() {
                    cursor_entry += 1;
                    cursor_part = 0;
                    cursor_offset = 0;
                }
            }
            if count == 0 {
                return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
            }
            // SAFETY: `count` is exactly the initialized iovec prefix assembled above.
            let iovecs =
                unsafe { std::slice::from_raw_parts(iovecs.as_ptr() as *const libc::iovec, count) };
            let written = match pwritev_once(file, iovecs, file_offset) {
                SerialWriteOutcome::Written(written) => written,
                SerialWriteOutcome::Interrupted => continue,
                SerialWriteOutcome::OffsetOverflow => {
                    return Err(self.fault(
                        DurabilityStage::PositionalWriteOverflow,
                        Some(libc::EOVERFLOW),
                    ));
                }
                SerialWriteOutcome::Errno(raw_os_error) => {
                    return Err(self.fault(DurabilityStage::PositionalWrite, Some(raw_os_error)));
                }
            };
            if written == 0 {
                return Err(self.fault(DurabilityStage::PositionalWriteZero, None));
            }
            let mut advanced = written;
            if advanced > remaining {
                return Err(self.fault(DurabilityStage::PositionalWriteOverflow, None));
            }
            remaining -= advanced;
            let Some(next_file_offset) = file_offset.checked_add(advanced as u64) else {
                return Err(self.fault(DurabilityStage::PositionalWriteOverflow, None));
            };
            file_offset = next_file_offset;
            while advanced != 0 {
                let Some(current) = entries.get(entry) else {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                };
                let Some(bytes) = current.region(part) else {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                };
                if part_offset > bytes.len() {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                }
                let available = bytes.len() - part_offset;
                if available == 0 {
                    return Err(self.fault(DurabilityStage::PositionalWriteInvariant, None));
                }
                let step = available.min(advanced);
                part_offset += step;
                advanced -= step;
                if part_offset == bytes.len() {
                    part += 1;
                    part_offset = 0;
                    if part == entries[entry].parts() {
                        entry += 1;
                        part = 0;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn finish_success(mut self) {
        let mut entries = self.entries.take().expect("prepared WAL group is live");
        entries.clear();
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = GroupSlotState::Idle(entries);
    }

    pub(crate) fn poison(mut self, fault: DurabilityFault) {
        let mut entries = self.entries.take().expect("prepared WAL group is live");
        entries.clear();
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = GroupSlotState::Poisoned { entries, fault };
    }
}

/// The exact FUA route stages directly from the permanent group descriptor.  Its cursor crosses
/// immutable exact-record regions without concatenating or re-encoding logical WAL bytes.
#[cfg(unix)]
impl FuaScatterSource for PreparedWalGroup {
    fn len(&self) -> usize {
        self.wire_bytes
    }

    fn copy_into(
        &mut self,
        mut source_offset: usize,
        mut destination: &mut [u8],
    ) -> Result<(), FuaFrameFault> {
        let Some(source_end) = source_offset.checked_add(destination.len()) else {
            return Err(FuaFrameFault::new(
                FuaFrameFaultStage::ScatterSource,
                None,
                0,
                FUA_FRAME_FAULT_NO_FRAME,
            ));
        };
        if source_end > self.wire_bytes {
            return Err(FuaFrameFault::new(
                FuaFrameFaultStage::ScatterSource,
                None,
                0,
                FUA_FRAME_FAULT_NO_FRAME,
            ));
        }
        let entries = self.entries.as_ref().ok_or_else(|| {
            FuaFrameFault::new(
                FuaFrameFaultStage::ScatterSource,
                None,
                0,
                FUA_FRAME_FAULT_NO_FRAME,
            )
        })?;
        for entry in entries {
            for part in 0..entry.parts() {
                let bytes = entry.region(part).ok_or_else(|| {
                    FuaFrameFault::new(
                        FuaFrameFaultStage::ScatterSource,
                        None,
                        0,
                        FUA_FRAME_FAULT_NO_FRAME,
                    )
                })?;
                if source_offset >= bytes.len() {
                    source_offset -= bytes.len();
                    continue;
                }
                let available = bytes.len() - source_offset;
                let copied = available.min(destination.len());
                destination[..copied]
                    .copy_from_slice(&bytes[source_offset..source_offset + copied]);
                destination = &mut destination[copied..];
                source_offset = 0;
                if destination.is_empty() {
                    return Ok(());
                }
            }
        }
        Err(FuaFrameFault::new(
            FuaFrameFaultStage::ScatterSource,
            None,
            0,
            FUA_FRAME_FAULT_NO_FRAME,
        ))
    }
}

impl Drop for PreparedWalGroup {
    fn drop(&mut self) {
        if self.entries.is_none() {
            return;
        }
        let mut entries = self.entries.take().expect("prepared WAL group is live");
        entries.clear();
        let mut state = self
            .slot
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = GroupSlotState::Poisoned {
            entries,
            fault: self.fault(DurabilityStage::DescriptorPoison, None),
        };
    }
}

fn serial_exact_fault(
    core: Arc<WalDurableCore>,
    group: PreparedWalGroup,
    fault: DurabilityFault,
) -> EngineError {
    let first_fault = core.install_fixed_fault(fault);
    signal_fixed_fault_before_wake(first_fault);
    group.poison(first_fault);
    let mut state = core.lock_state();
    state.io_in_flight = false;
    drop(state);
    core.cv.notify_all();
    EngineError::DurabilityFault(first_fault)
}

fn serial_exact_post_handoff(
    core: Arc<WalDurableCore>,
    file: Arc<std::fs::File>,
    offset: u64,
    group: PreparedWalGroup,
    target_records: usize,
    group_size: usize,
) -> Result<usize, EngineError> {
    let group_bytes = group.wire_bytes();
    if let Err(fault) = group
        .write_all_vectored_at(&file, offset)
        .and_then(|_| sync_data_retry_interrupted(&file, &group))
    {
        return Err(serial_exact_fault(core, group, fault));
    }
    let mut state = core.lock_state();
    let Some(durable_end) = offset.checked_add(group_bytes as u64) else {
        let fault = group.fault(DurabilityStage::FrontierDrift, None);
        drop(state);
        return Err(serial_exact_fault(core, group, fault));
    };
    if state.durable_bytes != offset || durable_end > state.prealloc_bytes {
        let fault = group.fault(DurabilityStage::FrontierDrift, None);
        drop(state);
        return Err(serial_exact_fault(core, group, fault));
    }
    state.durable_bytes = durable_end;
    WalDurableCore::note_group(&mut state, group_size, target_records);
    state.io_in_flight = false;
    group.finish_success();
    drop(state);
    core.cv.notify_all();
    Ok(target_records)
}

fn serial_exact_abandon(core: Arc<WalDurableCore>, group: PreparedWalGroup) {
    let fault = group.fault(DurabilityStage::Abandoned, None);
    let first_fault = core.install_fixed_fault(fault);
    signal_fixed_fault_before_wake(first_fault);
    group.poison(first_fault);
    let mut state = core.lock_state();
    state.io_in_flight = false;
    drop(state);
    core.cv.notify_all();
}
// END SERIAL_EXACT_POST_HANDOFF_NO_ALLOC

/// A fully formed serial group. Exact groups enter the fixed post-handoff path; compatibility
/// groups retain their legacy string diagnostics outside that allocation-free region.
pub(super) struct SerialFlushJob {
    core: Arc<WalDurableCore>,
    file: Arc<std::fs::File>,
    offset: u64,
    group: PreparedWalGroup,
    target_records: usize,
    group_size: usize,
}

impl SerialFlushJob {
    pub(super) fn new(
        core: Arc<WalDurableCore>,
        file: Arc<std::fs::File>,
        offset: u64,
        target_records: usize,
        group_size: usize,
        group: PreparedWalGroup,
    ) -> Self {
        Self {
            core,
            file,
            offset,
            group,
            target_records,
            group_size,
        }
    }

    pub(super) fn commit(self) -> Result<usize, EngineError> {
        let Self {
            core,
            file,
            offset,
            group,
            target_records,
            group_size,
        } = self;
        if group.contains_exact() {
            return serial_exact_post_handoff(
                core,
                file,
                offset,
                group,
                target_records,
                group_size,
            );
        }
        serial_legacy_post_handoff(core, file, offset, group, target_records, group_size)
    }

    pub(super) fn abandon(self) {
        if self.group.contains_exact() {
            serial_exact_abandon(self.core, self.group);
        } else {
            serial_legacy_abandon(self.core, self.group);
        }
    }

    #[cfg(test)]
    pub(super) fn first_exact_serialized_ptr_for_test(&self) -> Option<*const u8> {
        self.group.first_exact_serialized_ptr_for_test()
    }
}

fn serial_legacy_post_handoff(
    core: Arc<WalDurableCore>,
    file: Arc<std::fs::File>,
    offset: u64,
    group: PreparedWalGroup,
    target_records: usize,
    group_size: usize,
) -> Result<usize, EngineError> {
    let group_bytes = group.wire_bytes();
    let io_result = group
        .write_all_vectored_at(&file, offset)
        .and_then(|_| sync_data_retry_interrupted(&file, &group));
    let mut state = core.lock_state();
    state.io_in_flight = false;
    let outcome = match io_result {
        Ok(()) => match offset.checked_add(group_bytes as u64) {
            Some(durable_end)
                if state.durable_bytes == offset && durable_end <= state.prealloc_bytes =>
            {
                state.durable_bytes = durable_end;
                WalDurableCore::note_group(&mut state, group_size, target_records);
                group.finish_success();
                Ok(target_records)
            }
            _ => {
                let fault = group.fault(DurabilityStage::FrontierDrift, None);
                group.poison(fault);
                state.poisoned = Some(format!("serial WAL group frontier drifted ({fault})"));
                Err(EngineError::Durability(format!(
                    "serial WAL group frontier drifted for {}: {fault}",
                    core.segment_path.display()
                )))
            }
        },
        Err(fault) => {
            group.poison(fault);
            state.poisoned = Some(format!("group flush failed ({fault})"));
            Err(EngineError::Durability(format!(
                "failed to flush WAL segment group {}: {fault}",
                core.segment_path.display()
            )))
        }
    };
    drop(state);
    core.cv.notify_all();
    outcome
}

fn serial_legacy_abandon(core: Arc<WalDurableCore>, group: PreparedWalGroup) {
    let fault = group.fault(DurabilityStage::Abandoned, None);
    group.poison(fault);
    let mut state = core.lock_state();
    state.io_in_flight = false;
    state.poisoned = Some("group flush abandoned mid-IO".to_string());
    drop(state);
    core.cv.notify_all();
}

#[cfg(test)]
mod group_tests;
