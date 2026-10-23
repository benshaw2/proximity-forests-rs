//! Compact fitted-tree representation and iterative traversal primitives.

use crate::{DistanceConfig, NodeMembership};

use super::LeafPayload;

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

/// One exemplar branch stored by an internal node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExemplarBranch {
    exemplar: ObservationIndex,
    child: NodeId,
}

impl ExemplarBranch {
    /// Creates one exemplar-to-child branch.
    #[must_use]
    pub const fn new(exemplar: ObservationIndex, child: NodeId) -> Self {
        Self { exemplar, child }
    }

    /// Returns the exemplar's training-observation index.
    #[must_use]
    pub const fn exemplar(self) -> ObservationIndex {
        self.exemplar
    }

    /// Returns the child reached by this branch.
    #[must_use]
    pub const fn child(self) -> NodeId {
        self.child
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
}

impl ProximityTree {
    /// Creates an empty tree arena.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
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
