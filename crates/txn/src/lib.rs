use std::collections::BTreeMap;

use gpu_db_types::TxnId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Txn {
    pub id: TxnId,
    pub state: TxnState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnError {
    NotFound(TxnId),
    NotActive(TxnId),
    AlreadyExists(TxnId),
    IdExhausted,
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "transaction {id} not found"),
            Self::NotActive(id) => write!(f, "transaction {id} is not active"),
            Self::AlreadyExists(id) => write!(f, "transaction {id} already exists"),
            Self::IdExhausted => write!(f, "transaction id space exhausted"),
        }
    }
}

impl std::error::Error for TxnError {}

#[derive(Debug, Default)]
pub struct TxnManager {
    next_id: TxnId,
    states: BTreeMap<TxnId, TxnState>,
    active_count: usize,
}

impl TxnManager {
    pub fn begin(&mut self) -> Result<Txn, TxnError> {
        let Some(id) = self.next_id.checked_add(1) else {
            return Err(TxnError::IdExhausted);
        };
        self.next_id = id;
        self.states.insert(id, TxnState::Active);
        self.active_count = self.active_count.saturating_add(1);
        Ok(Txn {
            id,
            state: TxnState::Active,
        })
    }

    pub fn begin_with_id(&mut self, id: TxnId) -> Result<Txn, TxnError> {
        if self.states.contains_key(&id) {
            return Err(TxnError::AlreadyExists(id));
        }
        self.next_id = self.next_id.max(id);
        self.states.insert(id, TxnState::Active);
        self.active_count = self.active_count.saturating_add(1);
        Ok(Txn {
            id,
            state: TxnState::Active,
        })
    }

    pub fn state(&self, id: TxnId) -> Option<TxnState> {
        self.states.get(&id).copied()
    }

    pub fn commit(&mut self, id: TxnId) -> Result<Txn, TxnError> {
        self.transition_terminal(id, TxnState::Committed)
    }

    pub fn rollback(&mut self, id: TxnId) -> Result<Txn, TxnError> {
        self.transition_terminal(id, TxnState::Aborted)
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn oldest_active_txn_id(&self) -> Option<TxnId> {
        self.states
            .iter()
            .find_map(|(&id, state)| matches!(state, TxnState::Active).then_some(id))
    }

    pub fn newest_active_txn_id(&self) -> Option<TxnId> {
        self.states
            .iter()
            .rev()
            .find_map(|(&id, state)| matches!(state, TxnState::Active).then_some(id))
    }

    pub fn active_txn_ids(&self) -> Vec<TxnId> {
        self.states
            .iter()
            .filter_map(|(&id, state)| matches!(state, TxnState::Active).then_some(id))
            .collect()
    }

    fn transition_terminal(&mut self, id: TxnId, to: TxnState) -> Result<Txn, TxnError> {
        let Some(state) = self.states.get_mut(&id) else {
            return Err(TxnError::NotFound(id));
        };
        if !matches!(*state, TxnState::Active) {
            return Err(TxnError::NotActive(id));
        }
        *state = to;
        self.active_count = self.active_count.saturating_sub(1);
        Ok(Txn { id, state: to })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_assigns_monotonic_ids_and_tracks_active() {
        let mut tm = TxnManager::default();

        let a = tm.begin().unwrap();
        let b = tm.begin().unwrap();

        assert_eq!(a.id, 1);
        assert_eq!(b.id, 2);
        assert_eq!(tm.state(a.id), Some(TxnState::Active));
        assert_eq!(tm.state(b.id), Some(TxnState::Active));
        assert_eq!(tm.active_count(), 2);
    }

    #[test]
    fn commit_transitions_to_terminal_and_decrements_active() {
        let mut tm = TxnManager::default();
        let t = tm.begin().unwrap();

        let committed = tm.commit(t.id).unwrap();

        assert_eq!(committed.state, TxnState::Committed);
        assert_eq!(tm.state(t.id), Some(TxnState::Committed));
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn rollback_transitions_to_terminal_and_decrements_active() {
        let mut tm = TxnManager::default();
        let t = tm.begin().unwrap();

        let rolled_back = tm.rollback(t.id).unwrap();

        assert_eq!(rolled_back.state, TxnState::Aborted);
        assert_eq!(tm.state(t.id), Some(TxnState::Aborted));
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn cannot_transition_missing_or_non_active_txn() {
        let mut tm = TxnManager::default();
        assert_eq!(tm.commit(99), Err(TxnError::NotFound(99)));

        let t = tm.begin().unwrap();
        tm.commit(t.id).unwrap();

        assert_eq!(tm.commit(t.id), Err(TxnError::NotActive(t.id)));
        assert_eq!(tm.rollback(t.id), Err(TxnError::NotActive(t.id)));
    }

    #[test]
    fn begin_with_id_uses_caller_id_and_advances_allocator_floor() {
        let mut tm = TxnManager::default();

        let opened = tm.begin_with_id(42).unwrap();
        assert_eq!(opened.id, 42);
        assert_eq!(tm.state(42), Some(TxnState::Active));

        let next = tm.begin().unwrap();
        assert_eq!(next.id, 43);
    }

    #[test]
    fn begin_with_id_rejects_duplicate_active_id() {
        let mut tm = TxnManager::default();

        tm.begin_with_id(7).unwrap();
        assert_eq!(tm.begin_with_id(7), Err(TxnError::AlreadyExists(7)));
    }

    #[test]
    fn begin_with_id_rejects_duplicate_terminal_id() {
        let mut tm = TxnManager::default();

        tm.begin_with_id(9).unwrap();
        tm.commit(9).unwrap();

        assert_eq!(tm.begin_with_id(9), Err(TxnError::AlreadyExists(9)));
    }

    #[test]
    fn begin_reports_exhaustion_after_u64_max_floor() {
        let mut tm = TxnManager::default();
        tm.begin_with_id(u64::MAX).unwrap();
        tm.commit(u64::MAX).unwrap();

        assert_eq!(tm.begin(), Err(TxnError::IdExhausted));
    }

    #[test]
    fn active_count_stays_consistent_across_error_paths() {
        let mut tm = TxnManager::default();
        let t = tm.begin().unwrap();
        assert_eq!(tm.active_count(), 1);

        assert_eq!(tm.commit(999), Err(TxnError::NotFound(999)));
        assert_eq!(tm.active_count(), 1);

        tm.commit(t.id).unwrap();
        assert_eq!(tm.active_count(), 0);

        assert_eq!(tm.rollback(t.id), Err(TxnError::NotActive(t.id)));
        assert_eq!(tm.active_count(), 0);

        assert_eq!(tm.begin_with_id(t.id), Err(TxnError::AlreadyExists(t.id)));
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn active_txn_helpers_track_oldest_newest_and_ids() {
        let mut tm = TxnManager::default();
        assert_eq!(tm.oldest_active_txn_id(), None);
        assert_eq!(tm.newest_active_txn_id(), None);
        assert!(tm.active_txn_ids().is_empty());

        tm.begin_with_id(7).unwrap();
        tm.begin_with_id(3).unwrap();
        tm.begin_with_id(11).unwrap();
        tm.commit(7).unwrap();

        assert_eq!(tm.oldest_active_txn_id(), Some(3));
        assert_eq!(tm.newest_active_txn_id(), Some(11));
        assert_eq!(tm.active_txn_ids(), vec![3, 11]);

        tm.rollback(3).unwrap();
        assert_eq!(tm.oldest_active_txn_id(), Some(11));
        assert_eq!(tm.newest_active_txn_id(), Some(11));
        assert_eq!(tm.active_txn_ids(), vec![11]);
    }
}
