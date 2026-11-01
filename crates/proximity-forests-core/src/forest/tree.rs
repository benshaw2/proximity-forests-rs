//! Compact fitted-tree representation and iterative traversal primitives.
use std::sync::Arc;

use crate::{DistanceConfig, NodeMembership};

use super::{FinishedExemplars, LeafPayload, OwnedObservationStore};

/// Stable index of one fitted tree node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(usize);

impl NodeId {
    /// Creates a node index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the underlying index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Training-sample index used by fitted exemplars.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ObservationIndex(usize);

impl ObservationIndex {
    /// Creates an observation index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the underlying index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Stable index into one fitted tree's model-owned exemplar store.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExemplarId(u32);

impl ExemplarId {
    /// Creates a durable exemplar identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Creates a compact identifier from a platform-sized index.
    #[must_use]
    pub fn try_from_index(value: usize) -> Option<Self> {
        u32::try_from(value).ok().map(Self)
    }

    /// Returns the compact underlying identifier.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Returns the identifier as a platform-sized index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One exemplar branch stored by an internal node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExemplarBranch {
    exemplar: ExemplarId,
    child: NodeId,
}

impl ExemplarBranch {
    /// Creates one exemplar-to-child branch.
    #[must_use]
    pub const fn new(exemplar: ExemplarId, child: NodeId) -> Self {
        Self { exemplar, child }
    }

    /// Returns the durable tree-local exemplar identifier.
    #[must_use]
    pub const fn exemplar(self) -> ExemplarId {
        self.exemplar
    }

    /// Returns the child reached by this branch.
    #[must_use]
    pub const fn child(self) -> NodeId {
        self.child
    }

    /// Replaces the exemplar identifier while preserving the child edge.
    pub(crate) const fn set_exemplar(&mut self, exemplar: ExemplarId) {
        self.exemplar = exemplar;
    }
}

/// Fitted internal-node routing state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitNode {
    distance: DistanceConfig,
    selected_dimensions: Vec<usize>,
    branches: Vec<ExemplarBranch>,
}

impl SplitNode {
    /// Creates a fitted internal node.
    #[must_use]
    pub fn new(
        distance: DistanceConfig,
        selected_dimensions: Vec<usize>,
        branches: Vec<ExemplarBranch>,
    ) -> Self {
        Self {
            distance,
            selected_dimensions,
            branches,
        }
    }

    /// Returns the fitted distance configuration.
    #[must_use]
    pub const fn distance(&self) -> &DistanceConfig {
        &self.distance
    }

    /// Returns selected dimensions, or an empty slice when all dimensions apply.
    #[must_use]
    pub fn selected_dimensions(&self) -> &[usize] {
        &self.selected_dimensions
    }

    /// Returns exemplar branches in routing order.
    #[must_use]
    pub fn branches(&self) -> &[ExemplarBranch] {
        &self.branches
    }

    /// Returns mutable exemplar branches for checked model compaction.
    pub(crate) fn branches_mut(&mut self) -> &mut [ExemplarBranch] {
        &mut self.branches
    }
}

/// Compact task-aware fitted leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode {
    occurrence_count: usize,
    distinct_count: usize,
    payload: LeafPayload,
}

impl LeafNode {
    /// Builds a fitted leaf from membership and task-specific prediction state.
    #[must_use]
    pub fn new(membership: &NodeMembership, payload: LeafPayload) -> Self {
        Self {
            occurrence_count: membership.occurrence_count(),
            distinct_count: membership.distinct_count(),
            payload,
        }
    }

    /// Builds a task-independent placeholder leaf.
    #[must_use]
    pub fn placeholder(membership: &NodeMembership) -> Self {
        Self::new(membership, LeafPayload::Empty)
    }

    /// Reconstructs a fitted leaf from validated persisted state.
    pub(crate) fn from_parts(
        occurrence_count: usize,
        distinct_count: usize,
        payload: LeafPayload,
    ) -> crate::CoreResult<Self> {
        if occurrence_count == 0 {
            return Err(crate::CoreError::invalid_data(
                "a fitted leaf requires at least one occurrence",
            ));
        }
        if distinct_count == 0 || distinct_count > occurrence_count {
            return Err(crate::CoreError::invalid_data(
                "leaf distinct count must be positive and no greater than occurrence count",
            ));
        }
        Ok(Self {
            occurrence_count,
            distinct_count,
            payload,
        })
    }

    /// Returns the number of bootstrap occurrences reaching the leaf.
    #[must_use]
    pub const fn occurrence_count(&self) -> usize {
        self.occurrence_count
    }

    /// Returns the number of distinct observations reaching the leaf.
    #[must_use]
    pub const fn distinct_count(&self) -> usize {
        self.distinct_count
    }

    /// Returns task-specific prediction state.
    #[must_use]
    pub const fn payload(&self) -> &LeafPayload {
        &self.payload
    }
}

/// One fitted tree node.
#[derive(Clone, Debug, PartialEq)]
pub enum TreeNode {
    /// Terminal node.
    Leaf(LeafNode),
    /// Internal exemplar split.
    Split(SplitNode),
}

/// Compact arena-backed fitted tree.
///
/// Nodes refer to children by stable indices, avoiding recursive ownership and
/// enabling iterative training, traversal, serialization, and future parallel
/// construction of independent trees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProximityTree {
    nodes: Vec<TreeNode>,
    exemplars: Option<Arc<OwnedObservationStore>>,
    exemplar_training_indices: Option<Arc<[usize]>>,
}

impl ProximityTree {
    /// Creates an empty tree arena.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            exemplars: None,
            exemplar_training_indices: None,
        }
    }

    /// Attaches immutable tree-local exemplar values and their training identities.
    pub fn set_exemplars(&mut self, exemplars: FinishedExemplars) {
        let (store, training_indices) = exemplars.into_parts();
        self.exemplars = Some(Arc::new(store));
        self.exemplar_training_indices = Some(training_indices.into());
    }
    /// Returns original training indices in local exemplar-ID order.
    #[must_use]
    pub fn exemplar_training_indices(&self) -> Option<&[usize]> {
        self.exemplar_training_indices.as_deref()
    }

    /// Returns tree-local model-owned exemplar values.
    #[must_use]
    pub fn exemplars(&self) -> Option<&OwnedObservationStore> {
        self.exemplars.as_deref()
    }

    /// Reports whether two fitted trees reference the identical exemplar pool.
    #[must_use]
    pub(crate) fn shares_exemplar_pool_with(&self, other: &Self) -> bool {
        match (&self.exemplars, &other.exemplars) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, None) => true,
            _ => false,
        }
    }

    /// Reconstructs a fitted tree from a validated node arena and exemplar store.
    pub(crate) fn from_parts(
        nodes: Vec<TreeNode>,
        exemplars: OwnedObservationStore,
    ) -> crate::CoreResult<Self> {
        if nodes.is_empty() {
            return Err(crate::CoreError::invalid_data(
                "a fitted tree requires at least one node",
            ));
        }
        Ok(Self {
            nodes,
            exemplars: Some(Arc::new(exemplars)),
            // Format v1 did not preserve provenance. Restored v1 trees remain
            // usable, but are upgraded by value during the serialization slice.
            exemplar_training_indices: None,
        })
    }

    /// Reconstructs a fitted tree against one forest-owned exemplar pool.
    pub(crate) fn from_shared_parts(
        nodes: Vec<TreeNode>,
        exemplars: Arc<OwnedObservationStore>,
    ) -> crate::CoreResult<Self> {
        if nodes.is_empty() {
            return Err(crate::CoreError::invalid_data(
                "a fitted tree requires at least one node",
            ));
        }
        Ok(Self {
            nodes,
            exemplars: Some(exemplars),
            exemplar_training_indices: None,
        })
    }

    /// Returns the number of fitted nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Reports whether no nodes have been fitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the root index when the tree is nonempty.
    #[must_use]
    pub fn root(&self) -> Option<NodeId> {
        (!self.nodes.is_empty()).then_some(NodeId::new(0))
    }

    /// Appends a fitted node and returns its stable index.
    pub fn push(&mut self, node: TreeNode) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Returns one fitted node.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&TreeNode> {
        self.nodes.get(id.get())
    }

    /// Replaces a reserved or previously fitted node.
    pub fn replace(&mut self, id: NodeId, node: TreeNode) -> Option<TreeNode> {
        self.nodes
            .get_mut(id.get())
            .map(|destination| std::mem::replace(destination, node))
    }

    /// Returns all nodes in arena order.
    #[must_use]
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Attaches one forest-owned pool shared by all fitted trees.
    pub(crate) fn replace_shared_exemplars(
        &mut self,
        exemplars: Arc<OwnedObservationStore>,
        training_indices: Arc<[usize]>,
    ) {
        self.exemplars = Some(exemplars);
        self.exemplar_training_indices = Some(training_indices);
    }

    /// Rewrites every branch exemplar through a complete old-to-new ID map.
    ///
    /// # Errors
    ///
    /// Returns an error when a branch references an unavailable local ID or a
    /// remapped identifier cannot be represented by the compact `u32` ID type.
    pub(crate) fn remap_exemplars(&mut self, old_to_new: &[usize]) -> crate::CoreResult<()> {
        for node in &mut self.nodes {
            let TreeNode::Split(split) = node else {
                continue;
            };
            for branch in split.branches_mut() {
                let old = branch.exemplar().index();
                let new = old_to_new.get(old).copied().ok_or_else(|| {
                    crate::CoreError::invalid_data(format!(
                        "tree branch exemplar {old} is outside the compaction map"
                    ))
                })?;
                let id = ExemplarId::try_from_index(new).ok_or_else(|| {
                    crate::CoreError::invalid_data(
                        "forest exemplar pool contains too many observations for u32 IDs",
                    )
                })?;
                branch.set_exemplar(id);
            }
        }
        Ok(())
    }
}

/// LIFO work item for iterative tree construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeWork {
    node: NodeId,
    depth: usize,
    membership: NodeMembership,
}

impl NodeWork {
    /// Creates one pending node-build item.
    #[must_use]
    pub const fn new(node: NodeId, depth: usize, membership: NodeMembership) -> Self {
        Self {
            node,
            depth,
            membership,
        }
    }

    /// Returns the reserved fitted-node index.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Returns the node depth.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns node membership.
    #[must_use]
    pub const fn membership(&self) -> &NodeMembership {
        &self.membership
    }

    /// Consumes the work item and returns membership.
    #[must_use]
    pub fn into_membership(self) -> NodeMembership {
        self.membership
    }
}

/// Explicit LIFO stack for iterative tree construction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TreeBuildStack {
    work: Vec<NodeWork>,
}

impl TreeBuildStack {
    /// Creates an empty build stack.
    #[must_use]
    pub const fn new() -> Self {
        Self { work: Vec::new() }
    }

    /// Pushes pending node work.
    pub fn push(&mut self, work: NodeWork) {
        self.work.push(work);
    }

    /// Pops the next node to build.
    pub fn pop(&mut self) -> Option<NodeWork> {
        self.work.pop()
    }

    /// Reports whether no work remains.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.work.is_empty()
    }

    /// Returns the number of pending nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.work.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{LeafNode, NodeWork, ProximityTree, TreeBuildStack, TreeNode};
    use crate::{NodeMembership, SampleId};

    #[test]
    fn arena_indices_are_stable() {
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1))]).expect("valid membership");
        let mut tree = ProximityTree::new();
        let root = tree.push(TreeNode::Leaf(LeafNode::placeholder(&membership)));
        let child = tree.push(TreeNode::Leaf(LeafNode::placeholder(&membership)));
        assert_eq!(root.get(), 0);
        assert_eq!(child.get(), 1);
        assert_eq!(tree.root(), Some(root));
    }

    #[test]
    fn exemplar_ids_can_be_checked_and_remapped_for_forest_pooling() {
        use super::{ExemplarBranch, ExemplarId, SplitNode};
        use crate::DistanceConfig;

        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1)), Some(SampleId::new(2))])
                .expect("valid membership");
        let mut tree = ProximityTree::new();
        let root = tree.push(TreeNode::Leaf(LeafNode::placeholder(&membership)));
        let child = tree.push(TreeNode::Leaf(LeafNode::placeholder(&membership)));
        tree.replace(
            root,
            TreeNode::Split(SplitNode::new(
                DistanceConfig::SquaredEuclidean,
                Vec::new(),
                vec![ExemplarBranch::new(ExemplarId::new(1), child)],
            )),
        );
        tree.remap_exemplars(&[9, 4]).expect("valid remap");
        let TreeNode::Split(split) = &tree.nodes()[0] else {
            panic!("split")
        };
        assert_eq!(split.branches()[0].exemplar().index(), 4);
        assert!(tree.remap_exemplars(&[0]).is_err());
    }

    #[test]
    fn build_work_is_lifo_and_nonrecursive() {
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(4))]).expect("valid membership");
        let mut stack = TreeBuildStack::new();
        stack.push(NodeWork::new(super::NodeId::new(0), 0, membership.clone()));
        stack.push(NodeWork::new(super::NodeId::new(1), 1, membership));
        assert_eq!(stack.pop().expect("work").node().get(), 1);
        assert_eq!(stack.pop().expect("work").node().get(), 0);
        assert!(stack.is_empty());
    }
}
