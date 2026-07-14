use super::{
    compose_resolved_mvcc_rows, InMemoryTupleStore, MvccLabeledValueChainBranch, MvccReadSource,
    MvccSourceProvenance, MvccValueChainBranchFanIn, MvccValueChainPlan, MvccValueChainTerminal,
    ResolvedMvccRow, StorageError, StorageVisibility, TupleStore, TupleVersion,
};

pub(crate) fn resolve_follow_value_chain_from_seed(
    store: &InMemoryTupleStore,
    seed: &TupleVersion,
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
    branch_label: Option<&str>,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut current = seed.clone();
    let mut provenance_path = vec![seed.clone()];
    let mut previous = None;
    for _ in 0..plan.value_key_hops {
        previous = Some(current.clone());
        let Some(next) = store.tuple_fetch_by_key(&current.value, visibility)? else {
            return Ok(Vec::new());
        };
        current = next;
        provenance_path.push(current.clone());
    }

    let terminal_input_index = match plan.terminal {
        MvccValueChainTerminal::CurrentRow => {
            if plan.value_key_hops == 0 {
                0
            } else {
                provenance_path.len().saturating_sub(2)
            }
        }
        MvccValueChainTerminal::CurrentValuePrefixes => provenance_path.len().saturating_sub(1),
    };

    let provenance_tuple = match provenance {
        MvccSourceProvenance::Seed => seed.clone(),
        MvccSourceProvenance::TerminalInput => match plan.terminal {
            MvccValueChainTerminal::CurrentRow => previous.unwrap_or_else(|| seed.clone()),
            MvccValueChainTerminal::CurrentValuePrefixes => current.clone(),
        },
    };
    let source_key = provenance_tuple.key.clone();

    let mut rows = Vec::new();
    match plan.terminal {
        MvccValueChainTerminal::CurrentRow => rows.push(ResolvedMvccRow {
            branch_label: branch_label.map(ToOwned::to_owned),
            source_key: Some(source_key.clone()),
            source_tuple: Some(provenance_tuple.clone()),
            provenance_path: Some(provenance_path.clone()),
            terminal_input_index: Some(terminal_input_index),
            tuple: current,
        }),
        MvccValueChainTerminal::CurrentValuePrefixes => {
            let mut cursor = store.seq_scan_open(visibility)?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&current.value) {
                    rows.push(ResolvedMvccRow {
                        branch_label: branch_label.map(ToOwned::to_owned),
                        source_key: Some(source_key.clone()),
                        source_tuple: Some(provenance_tuple.clone()),
                        provenance_path: Some(provenance_path.clone()),
                        terminal_input_index: Some(terminal_input_index),
                        tuple,
                    });
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn resolve_follow_value_chain(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    plan: MvccValueChainPlan,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        rows.extend(resolve_follow_value_chain_from_seed(
            store, &seed, visibility, plan, provenance, None,
        )?);
    }
    Ok(rows)
}

pub(crate) fn resolve_follow_value_chain_branches(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    plans: &[MvccValueChainPlan],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for plan in plans {
                    rows.extend(resolve_follow_value_chain_from_seed(
                        store, &seed, visibility, *plan, provenance, None,
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for plan in plans {
                    let branch_rows = resolve_follow_value_chain_from_seed(
                        store, &seed, visibility, *plan, provenance, None,
                    )?;
                    if !branch_rows.is_empty() {
                        rows.extend(branch_rows);
                        break;
                    }
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn resolve_follow_value_chain_labeled_branches(
    store: &InMemoryTupleStore,
    keys: &[String],
    visibility: StorageVisibility,
    branches: &[MvccLabeledValueChainBranch],
    fan_in: MvccValueChainBranchFanIn,
    provenance: MvccSourceProvenance,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    let mut rows = Vec::new();
    for key in keys {
        let Some(seed) = store.tuple_fetch_by_key(key, visibility)? else {
            continue;
        };
        match fan_in {
            MvccValueChainBranchFanIn::AllBranches => {
                for branch in branches {
                    rows.extend(resolve_follow_value_chain_from_seed(
                        store,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
                    )?);
                }
            }
            MvccValueChainBranchFanIn::FirstNonEmptyBranch => {
                for branch in branches {
                    let branch_rows = resolve_follow_value_chain_from_seed(
                        store,
                        &seed,
                        visibility,
                        branch.plan,
                        provenance,
                        Some(branch.label.as_str()),
                    )?;
                    if !branch_rows.is_empty() {
                        rows.extend(branch_rows);
                        break;
                    }
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn resolve_mvcc_source(
    store: &InMemoryTupleStore,
    source: &MvccReadSource,
    visibility: StorageVisibility,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    match source {
        MvccReadSource::FullScan => {
            let mut cursor = store.seq_scan_open(visibility)?;
            let mut rows = Vec::new();
            while let Some(tuple) = cursor.next() {
                rows.push(ResolvedMvccRow {
                    branch_label: None,
                    source_key: None,
                    source_tuple: None,
                    provenance_path: None,
                    terminal_input_index: None,
                    tuple,
                });
            }
            Ok(rows)
        }
        MvccReadSource::KeyLookup { key } => Ok(store
            .tuple_fetch_by_key(key, visibility)?
            .into_iter()
            .map(|tuple| ResolvedMvccRow {
                branch_label: None,
                source_key: None,
                source_tuple: None,
                provenance_path: None,
                terminal_input_index: None,
                tuple,
            })
            .collect()),
        MvccReadSource::KeyBatchLookup { keys } => {
            let mut rows = Vec::new();
            for key in keys {
                if let Some(tuple) = store.tuple_fetch_by_key(key, visibility)? {
                    rows.push(ResolvedMvccRow {
                        branch_label: None,
                        source_key: None,
                        source_tuple: None,
                        provenance_path: None,
                        terminal_input_index: None,
                        tuple,
                    });
                }
            }
            Ok(rows)
        }
        MvccReadSource::Concat { sources } => {
            let mut rows = Vec::new();
            for source in sources {
                rows.extend(resolve_mvcc_source(store, source, visibility)?);
            }
            Ok(rows)
        }
        MvccReadSource::ConcatDistinct { sources }
        | MvccReadSource::IntersectDistinct { sources }
        | MvccReadSource::IntersectAll { sources }
        | MvccReadSource::ExceptDistinct { sources }
        | MvccReadSource::ExceptAll { sources }
        | MvccReadSource::SymmetricDifferenceDistinct { sources }
        | MvccReadSource::SymmetricDifferenceAll { sources } => {
            let resolved_sources = sources
                .iter()
                .map(|source| resolve_mvcc_source(store, source, visibility))
                .collect::<Result<Vec<_>, StorageError>>()?;
            Ok(compose_resolved_mvcc_rows(source, resolved_sources))
        }
        MvccReadSource::FollowValueChain {
            keys,
            plan,
            provenance,
        } => resolve_follow_value_chain(store, keys, visibility, *plan, *provenance),
        MvccReadSource::FollowValueChainBranches {
            keys,
            plans,
            fan_in,
            provenance,
        } => resolve_follow_value_chain_branches(
            store,
            keys,
            visibility,
            plans,
            *fan_in,
            *provenance,
        ),
        MvccReadSource::FollowValueChainLabeledBranches {
            keys,
            branches,
            fan_in,
            provenance,
        } => resolve_follow_value_chain_labeled_branches(
            store,
            keys,
            visibility,
            branches,
            *fan_in,
            *provenance,
        ),
        MvccReadSource::FollowValueKeyRefs { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 0,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 1,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyRefs { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentRow,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyPrefixes { keys } => resolve_follow_value_chain(
            store,
            keys,
            visibility,
            MvccValueChainPlan {
                value_key_hops: 2,
                terminal: MvccValueChainTerminal::CurrentValuePrefixes,
            },
            MvccSourceProvenance::Seed,
        ),
        MvccReadSource::FollowValueKeyRefValueKeyRefPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 3,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefs { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 4,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
        MvccReadSource::FollowValueKeyRefValueKeyRefValueKeyRefPrefixes { keys } => {
            resolve_follow_value_chain(
                store,
                keys,
                visibility,
                MvccValueChainPlan {
                    value_key_hops: 5,
                    terminal: MvccValueChainTerminal::CurrentValuePrefixes,
                },
                MvccSourceProvenance::Seed,
            )
        }
    }
}

pub(crate) fn resolve_mvcc_all_versions(
    store: &InMemoryTupleStore,
    visibility: StorageVisibility,
) -> Result<Vec<ResolvedMvccRow>, StorageError> {
    if visibility.read_txn_id == 0 {
        return Err(StorageError::InvalidVisibility);
    }

    Ok(store
        .all_versions()
        .into_iter()
        .map(|tuple| ResolvedMvccRow {
            branch_label: None,
            source_key: None,
            source_tuple: None,
            provenance_path: None,
            terminal_input_index: None,
            tuple,
        })
        .collect())
}
