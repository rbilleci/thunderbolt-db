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
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "transaction {id} not found"),
            Self::NotActive(id) => write!(f, "transaction {id} is not active"),
        }
    }
}

impl std::error::Error for TxnError {}

#[derive(Debug, Default)]
pub struct TxnManager {
    next_id: TxnId,
    states: BTreeMap<TxnId, TxnState>,
}

impl TxnManager {
    pub fn begin(&mut self) -> Txn {
        self.next_id += 1;
        let id = self.next_id;
        self.states.insert(id, TxnState::Active);
        Txn {
            id,
            state: TxnState::Active,
        }
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
        self.states
            .values()
            .filter(|state| matches!(state, TxnState::Active))
            .count()
    }

    fn transition_terminal(&mut self, id: TxnId, to: TxnState) -> Result<Txn, TxnError> {
        let Some(state) = self.states.get_mut(&id) else {
            return Err(TxnError::NotFound(id));
        };
        if !matches!(*state, TxnState::Active) {
            return Err(TxnError::NotActive(id));
        }
        *state = to;
        Ok(Txn { id, state: to })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_assigns_monotonic_ids_and_tracks_active() {
        let mut tm = TxnManager::default();

        let a = tm.begin();
        let b = tm.begin();

        assert_eq!(a.id, 1);
        assert_eq!(b.id, 2);
        assert_eq!(tm.state(a.id), Some(TxnState::Active));
        assert_eq!(tm.state(b.id), Some(TxnState::Active));
        assert_eq!(tm.active_count(), 2);
    }

    #[test]
    fn commit_transitions_to_terminal_and_decrements_active() {
        let mut tm = TxnManager::default();
        let t = tm.begin();

        let committed = tm.commit(t.id).unwrap();

        assert_eq!(committed.state, TxnState::Committed);
        assert_eq!(tm.state(t.id), Some(TxnState::Committed));
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn rollback_transitions_to_terminal_and_decrements_active() {
        let mut tm = TxnManager::default();
        let t = tm.begin();

        let rolled_back = tm.rollback(t.id).unwrap();

        assert_eq!(rolled_back.state, TxnState::Aborted);
        assert_eq!(tm.state(t.id), Some(TxnState::Aborted));
        assert_eq!(tm.active_count(), 0);
    }

    #[test]
    fn cannot_transition_missing_or_non_active_txn() {
        let mut tm = TxnManager::default();
        assert_eq!(tm.commit(99), Err(TxnError::NotFound(99)));

        let t = tm.begin();
        tm.commit(t.id).unwrap();

        assert_eq!(tm.commit(t.id), Err(TxnError::NotActive(t.id)));
        assert_eq!(tm.rollback(t.id), Err(TxnError::NotActive(t.id)));
    }
}
