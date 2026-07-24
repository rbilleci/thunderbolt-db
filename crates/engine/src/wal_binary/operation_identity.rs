//! Stable statement identities for the ordered transaction WAL envelope.

use super::BinaryTransactionMutation;

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
                Self::Insert {
                    table: operation_table
                },
                BinaryTransactionMutation::Insert {
                    table: mutation_table,
                    ..
                }
            ) | (
                Self::Update {
                    table: operation_table
                },
                BinaryTransactionMutation::Update {
                    table: mutation_table,
                    ..
                }
            ) | (
                Self::Delete {
                    table: operation_table
                },
                BinaryTransactionMutation::Delete {
                    table: mutation_table,
                    ..
                }
            ) if operation_table == mutation_table
        )
    }
}
