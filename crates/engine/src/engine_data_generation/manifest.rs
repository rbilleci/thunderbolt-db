//! Immutable manifests and the fixed-height persistent radix maps they own.
//!
//! Roots are supplied by completed GPU work. The host checks only the fixed path, depth, and
//! count commitments while retaining/relinking unchanged `Arc` children.

use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

use super::digest::{
    CurrentRowLeafRoot, DataGeneration, IndexEntryLeafRoot, IndexGeneration, IndexMapRoot,
    IndexRoot, IndexShapeRoot, RowMapRoot, StableIndexId, StableRowId, StableTableId,
    StableTransactionId, TableMapRoot, TableRoot,
};
use super::DataGenerationError;

const RADIX_DEPTH: u8 = 64;

#[cfg(test)]
std::thread_local! {
    static RADIX_NODE_VISITS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn record_hot_path_node_visit() {
    RADIX_NODE_VISITS.with(|visits| visits.set(visits.get().saturating_add(1)));
}

#[cfg(test)]
pub(super) fn reset_hot_path_node_visits_for_test() {
    RADIX_NODE_VISITS.with(|visits| visits.set(0));
}

#[cfg(test)]
pub(super) fn hot_path_node_visits_for_test() -> usize {
    RADIX_NODE_VISITS.with(Cell::get)
}

pub(super) trait RadixKey: Copy + Eq {
    fn bits(self) -> u64;
}

impl RadixKey for StableRowId {
    fn bits(self) -> u64 {
        self.get()
    }
}

impl RadixKey for StableTableId {
    fn bits(self) -> u64 {
        self.get()
    }
}

impl RadixKey for StableTransactionId {
    fn bits(self) -> u64 {
        self.get()
    }
}

pub(super) trait RadixLeafValue: Clone {
    fn same_commitment(&self, other: &Self) -> bool;
}

impl RadixLeafValue for CurrentRowLeafRoot {
    fn same_commitment(&self, other: &Self) -> bool {
        self == other
    }
}

impl RadixLeafValue for IndexEntryLeafRoot {
    fn same_commitment(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Clone, Debug)]
pub(super) struct GpuRadixPathNode<R> {
    pub(super) depth: u8,
    pub(super) subtree_count: u64,
    pub(super) root: R,
}

/// One completed GPU substitution path, ordered from the root (depth zero) through depth 63.
/// `leaf_root` is either the changed leaf or the depth-64 canonical empty root.
#[derive(Clone, Debug)]
pub(super) struct GpuRadixPathCompletion<R> {
    pub(super) leaf_root: R,
    pub(super) nodes: Vec<GpuRadixPathNode<R>>,
}

impl<R: Copy> GpuRadixPathCompletion<R> {
    pub(super) fn validate_shape(&self) -> Result<(), DataGenerationError> {
        if self.nodes.len() != usize::from(RADIX_DEPTH) {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "radix path length",
            ));
        }
        for (depth, node) in self.nodes.iter().enumerate() {
            if node.depth != depth as u8 {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "radix path depth",
                ));
            }
        }
        Ok(())
    }
}

/// One completed canonical empty root, explicitly bound to its radix depth by the GPU completion
/// adapter. The host validates that the submitted sequence is exactly depth zero through 64.
#[derive(Clone, Debug)]
struct GpuRadixEmptyRoot<R> {
    depth: u8,
    root: R,
}

/// The 65 canonical empty roots for one map domain. Production construction is deliberately
/// private to this module so a future GPU-completion adapter is the sole source; tests use the
/// cfg(test) synthetic constructor below.
#[derive(Clone, Debug)]
pub(super) struct GpuRadixEmptyRoots<R> {
    roots: Vec<GpuRadixEmptyRoot<R>>,
}

impl<R: Copy + Eq> GpuRadixEmptyRoots<R> {
    /// The sealed GPU root-completion handoff is the only production route that can create this
    /// depth-bound set.  Tests retain their explicit synthetic constructor below.
    pub(super) fn from_verified_depths(roots: Vec<(u8, R)>) -> Result<Self, DataGenerationError> {
        let result = Self {
            roots: roots
                .into_iter()
                .map(|(depth, root)| GpuRadixEmptyRoot { depth, root })
                .collect(),
        };
        result.validate_shape()?;
        Ok(result)
    }

    pub(super) fn validate_shape(&self) -> Result<(), DataGenerationError> {
        if self.roots.len() != usize::from(RADIX_DEPTH) + 1 {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "empty radix roots",
            ));
        }
        for (position, completed) in self.roots.iter().enumerate() {
            if completed.depth != position as u8 {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "empty radix root depth",
                ));
            }
            if self.roots[..position]
                .iter()
                .any(|earlier| earlier.root == completed.root)
            {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "duplicate empty radix root",
                ));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn synthetic_for_test(roots: Vec<(u8, R)>) -> Self {
        Self {
            roots: roots
                .into_iter()
                .map(|(depth, root)| GpuRadixEmptyRoot { depth, root })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
enum FixedRadixNode<K, R, V> {
    Empty {
        depth: u8,
        root: R,
    },
    Leaf {
        key: K,
        root: R,
        value: V,
    },
    Branch {
        depth: u8,
        root: R,
        subtree_count: u64,
        left: Arc<FixedRadixNode<K, R, V>>,
        right: Arc<FixedRadixNode<K, R, V>>,
    },
}

impl<K: RadixKey, R: Copy + Eq, V: RadixLeafValue> FixedRadixNode<K, R, V> {
    fn root(&self) -> R {
        match self {
            Self::Empty { root, .. } | Self::Leaf { root, .. } | Self::Branch { root, .. } => *root,
        }
    }

    fn count(&self) -> u64 {
        match self {
            Self::Empty { .. } => 0,
            Self::Leaf { .. } => 1,
            Self::Branch { subtree_count, .. } => *subtree_count,
        }
    }

    fn validate(
        &self,
        expected_depth: u8,
        prefix: u64,
        empty_roots: &[R],
    ) -> Result<u64, DataGenerationError> {
        #[cfg(test)]
        record_hot_path_node_visit();
        match self {
            Self::Empty { depth, root } => {
                if *depth != expected_depth || *root != empty_roots[usize::from(expected_depth)] {
                    return Err(DataGenerationError::Invalid("radix empty-node depth"));
                }
                Ok(0)
            }
            Self::Leaf { key, .. } => {
                if expected_depth != RADIX_DEPTH || key.bits() != prefix {
                    return Err(DataGenerationError::Invalid("radix leaf path"));
                }
                Ok(1)
            }
            Self::Branch {
                depth,
                subtree_count,
                left,
                right,
                ..
            } => {
                if *depth != expected_depth || expected_depth >= RADIX_DEPTH {
                    return Err(DataGenerationError::Invalid("radix branch depth"));
                }
                let left_count = left.validate(expected_depth + 1, prefix, empty_roots)?;
                let right_prefix = prefix | (1_u64 << (63 - expected_depth));
                let right_count = right.validate(expected_depth + 1, right_prefix, empty_roots)?;
                let checked = left_count
                    .checked_add(right_count)
                    .ok_or(DataGenerationError::CountOverflow)?;
                if checked == 0 || checked != *subtree_count {
                    return Err(DataGenerationError::Invalid("radix branch count"));
                }
                Ok(checked)
            }
        }
    }

    fn get(&self, key: K, depth: u8) -> Option<&V> {
        #[cfg(test)]
        record_hot_path_node_visit();
        match self {
            Self::Empty { .. } => None,
            Self::Leaf {
                key: leaf_key,
                value,
                ..
            } => (*leaf_key == key).then_some(value),
            Self::Branch { left, right, .. } => {
                let bit = (key.bits() >> (63 - depth)) & 1;
                if bit == 0 {
                    left.get(key, depth + 1)
                } else {
                    right.get(key, depth + 1)
                }
            }
        }
    }

    fn validate_values(
        &self,
        validate: &mut impl FnMut(K, &V) -> Result<(), DataGenerationError>,
    ) -> Result<(), DataGenerationError> {
        #[cfg(test)]
        record_hot_path_node_visit();
        match self {
            Self::Empty { .. } => Ok(()),
            Self::Leaf { key, value, .. } => validate(*key, value),
            Self::Branch { left, right, .. } => {
                left.validate_values(validate)?;
                right.validate_values(validate)
            }
        }
    }

    fn substitute(
        &self,
        key: K,
        depth: u8,
        after: &Option<V>,
        completion: &GpuRadixPathCompletion<R>,
        empty_roots: &[R],
    ) -> Result<Arc<Self>, DataGenerationError> {
        #[cfg(test)]
        record_hot_path_node_visit();
        if depth == RADIX_DEPTH {
            return Ok(match after {
                Some(value) => Arc::new(Self::Leaf {
                    key,
                    root: completion.leaf_root,
                    value: value.clone(),
                }),
                None => {
                    if completion.leaf_root != empty_roots[usize::from(RADIX_DEPTH)] {
                        return Err(DataGenerationError::GpuCompletionMismatch(
                            "empty radix leaf root",
                        ));
                    }
                    Arc::new(Self::Empty {
                        depth,
                        root: completion.leaf_root,
                    })
                }
            });
        }

        let (old_left, old_right) = match self {
            Self::Empty { .. } => (
                Arc::new(Self::Empty {
                    depth: depth + 1,
                    root: empty_roots[usize::from(depth + 1)],
                }),
                Arc::new(Self::Empty {
                    depth: depth + 1,
                    root: empty_roots[usize::from(depth + 1)],
                }),
            ),
            Self::Branch { left, right, .. } => (Arc::clone(left), Arc::clone(right)),
            Self::Leaf { .. } => {
                return Err(DataGenerationError::Invalid("radix leaf before depth 64"));
            }
        };
        let bit = (key.bits() >> (63 - depth)) & 1;
        let (left, right) = if bit == 0 {
            (
                old_left.substitute(key, depth + 1, after, completion, empty_roots)?,
                old_right,
            )
        } else {
            (
                old_left,
                old_right.substitute(key, depth + 1, after, completion, empty_roots)?,
            )
        };
        let subtree_count = left
            .count()
            .checked_add(right.count())
            .ok_or(DataGenerationError::CountOverflow)?;
        let completion_node = &completion.nodes[usize::from(depth)];
        if completion_node.subtree_count != subtree_count {
            return Err(DataGenerationError::GpuCompletionMismatch(
                "radix path count",
            ));
        }
        if subtree_count == 0 {
            if completion_node.root != empty_roots[usize::from(depth)] {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "empty radix node root",
                ));
            }
            Ok(Arc::new(Self::Empty {
                depth,
                root: completion_node.root,
            }))
        } else {
            Ok(Arc::new(Self::Branch {
                depth,
                root: completion_node.root,
                subtree_count,
                left,
                right,
            }))
        }
    }
}

/// A persistent, uncompressed 64-bit MSB-first radix tree. It performs no hash computation.
#[derive(Clone, Debug)]
pub(super) struct FixedRadixMap<K, R, V> {
    root: Arc<FixedRadixNode<K, R, V>>,
    empty_roots: Arc<[R]>,
}

impl<K: RadixKey, R: Copy + Eq, V: RadixLeafValue> FixedRadixMap<K, R, V> {
    pub(super) fn empty(empty_roots: GpuRadixEmptyRoots<R>) -> Result<Self, DataGenerationError> {
        empty_roots.validate_shape()?;
        let empty_roots: Arc<[R]> = empty_roots
            .roots
            .into_iter()
            .map(|completed| completed.root)
            .collect::<Vec<_>>()
            .into();
        Ok(Self {
            root: Arc::new(FixedRadixNode::Empty {
                depth: 0,
                root: empty_roots[0],
            }),
            empty_roots,
        })
    }

    pub(super) fn root(&self) -> R {
        self.root.root()
    }

    pub(super) fn count(&self) -> u64 {
        self.root.count()
    }

    pub(super) fn get(&self, key: K) -> Option<&V> {
        self.root.get(key, 0)
    }

    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        self.root.validate(0, 0, &self.empty_roots).map(|_| ())
    }

    pub(super) fn validate_values(
        &self,
        validate: impl FnMut(K, &V) -> Result<(), DataGenerationError>,
    ) -> Result<(), DataGenerationError> {
        let mut validate = validate;
        self.root.validate_values(&mut validate)
    }

    pub(super) fn substitute(
        &self,
        key: K,
        expected_before: Option<&V>,
        after: Option<V>,
        completion: &GpuRadixPathCompletion<R>,
    ) -> Result<Self, DataGenerationError> {
        completion.validate_shape()?;
        match (self.get(key), expected_before) {
            (None, None) => {}
            (Some(actual), Some(expected)) if actual.same_commitment(expected) => {}
            _ => return Err(DataGenerationError::PredecessorMismatch("radix leaf")),
        }
        let root = self
            .root
            .substitute(key, 0, &after, completion, &self.empty_roots)?;
        Ok(Self {
            root,
            empty_roots: Arc::clone(&self.empty_roots),
        })
    }

    #[cfg(test)]
    pub(super) fn root_arc_address_for_test(&self) -> usize {
        Arc::as_ptr(&self.root) as usize
    }

    #[cfg(test)]
    pub(super) fn corrupt_first_empty_root_for_test(
        &self,
        depth: u8,
        replacement_root: R,
    ) -> Result<Self, DataGenerationError> {
        fn replace<K: RadixKey, R: Copy + Eq, V: RadixLeafValue>(
            node: &Arc<FixedRadixNode<K, R, V>>,
            target_depth: u8,
            replacement_root: R,
            replaced: &mut bool,
        ) -> Arc<FixedRadixNode<K, R, V>> {
            match node.as_ref() {
                FixedRadixNode::Empty { depth, root } if !*replaced && *depth == target_depth => {
                    *replaced = true;
                    Arc::new(FixedRadixNode::Empty {
                        depth: *depth,
                        root: replacement_root,
                    })
                }
                FixedRadixNode::Empty { depth, root } => Arc::new(FixedRadixNode::Empty {
                    depth: *depth,
                    root: *root,
                }),
                FixedRadixNode::Leaf { key, root, value } => Arc::new(FixedRadixNode::Leaf {
                    key: *key,
                    root: *root,
                    value: value.clone(),
                }),
                FixedRadixNode::Branch {
                    depth,
                    root,
                    subtree_count,
                    left,
                    right,
                } => {
                    let left = replace(left, target_depth, replacement_root, replaced);
                    let right = replace(right, target_depth, replacement_root, replaced);
                    Arc::new(FixedRadixNode::Branch {
                        depth: *depth,
                        root: *root,
                        subtree_count: *subtree_count,
                        left,
                        right,
                    })
                }
            }
        }

        let mut replaced = false;
        let root = replace(&self.root, depth, replacement_root, &mut replaced);
        if !replaced {
            return Err(DataGenerationError::Missing("test empty radix node"));
        }
        Ok(Self {
            root,
            empty_roots: Arc::clone(&self.empty_roots),
        })
    }
}

pub(super) type RowMap = FixedRadixMap<StableRowId, RowMapRoot, CurrentRowLeafRoot>;
pub(super) type IndexMap = FixedRadixMap<StableRowId, IndexMapRoot, IndexEntryLeafRoot>;

#[derive(Clone, Debug)]
pub(super) struct IndexManifest {
    pub(super) owner_table_id: StableTableId,
    pub(super) index_id: StableIndexId,
    pub(super) generation: IndexGeneration,
    pub(super) shape_root: IndexShapeRoot,
    pub(super) entries: IndexMap,
    pub(super) root: IndexRoot,
}

impl IndexManifest {
    pub(super) fn validate(
        &self,
        owner_table_id: StableTableId,
    ) -> Result<(), DataGenerationError> {
        if self.owner_table_id != owner_table_id {
            return Err(DataGenerationError::Invalid("index owner"));
        }
        self.entries.validate()
    }
}

#[derive(Clone, Debug)]
pub(super) struct TableManifest {
    pub(super) table_id: StableTableId,
    pub(super) data_generation: DataGeneration,
    pub(super) rows: RowMap,
    pub(super) indexes: Vec<Arc<IndexManifest>>,
    pub(super) root: TableRoot,
}

impl TableManifest {
    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        self.rows.validate()?;
        let mut previous = None;
        for index in &self.indexes {
            if previous.is_some_and(|id| id >= index.index_id) {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "table index manifest",
                ));
            }
            index.validate(self.table_id)?;
            previous = Some(index.index_id);
        }
        Ok(())
    }

    pub(super) fn index(&self, index_id: StableIndexId) -> Option<&IndexManifest> {
        self.indexes
            .binary_search_by_key(&index_id, |index| index.index_id)
            .ok()
            .map(|position| self.indexes[position].as_ref())
    }
}

#[derive(Clone, Debug)]
pub(super) struct TableMapLeaf {
    pub(super) manifest: Arc<TableManifest>,
}

impl RadixLeafValue for TableMapLeaf {
    fn same_commitment(&self, other: &Self) -> bool {
        self.manifest.table_id == other.manifest.table_id
            && self.manifest.root == other.manifest.root
    }
}

pub(super) type TableMap = FixedRadixMap<StableTableId, TableMapRoot, TableMapLeaf>;

impl TableMap {
    pub(super) fn table(&self, table_id: StableTableId) -> Option<&Arc<TableManifest>> {
        self.get(table_id).map(|leaf| &leaf.manifest)
    }

    pub(super) fn validate_manifest_closure(&self) -> Result<(), DataGenerationError> {
        self.validate_values(|table_id, leaf| {
            if table_id != leaf.manifest.table_id {
                return Err(DataGenerationError::Invalid("table-map leaf identity"));
            }
            leaf.manifest.validate()
        })
    }
}
