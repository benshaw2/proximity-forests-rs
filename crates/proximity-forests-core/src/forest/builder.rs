//! Iterative arena-backed proximity-tree construction.

use crate::{CoreError, CoreResult, ForestConfig, SeedMixer};

use super::{
    ExemplarBranch, LeafNode, LeafPayload, NodeId, NodeMembership, ObservationIndex, ProximityTree,
    SelectedCandidate, SplitNode, TreeNode, TreeWorkspace, build_child_memberships,
    stopping_reason,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct BuildWork {
    node: NodeId,
    depth: usize,
    path: u64,
    membership: NodeMembership,
    observations: Vec<ObservationIndex>,
}

/// Builds one tree iteratively with caller-supplied split and leaf logic.
///
/// `select` owns task-specific candidate generation, routing, and scoring.
/// `leaf` builds persistent prediction state whenever a node becomes terminal.
///
/// # Errors
///
/// Returns an error for invalid mappings, callback failures, malformed winning
/// candidates, or child-construction failures.
pub fn build_tree_iterative<Select, MakeLeaf>(
    config: &ForestConfig,
    tree_index: usize,
    root_membership: NodeMembership,
    root_observations: Vec<ObservationIndex>,
    workspace: &mut TreeWorkspace,
    mut select: Select,
    mut leaf: MakeLeaf,
) -> CoreResult<ProximityTree>
where
    Select: FnMut(
        u64,
        usize,
        u64,
        &NodeMembership,
        &[ObservationIndex],
        &mut TreeWorkspace,
    ) -> CoreResult<Option<SelectedCandidate>>,
    MakeLeaf: FnMut(&NodeMembership, &[ObservationIndex]) -> CoreResult<LeafPayload>,
{
    if root_observations.len() != root_membership.distinct_count() {
        return Err(CoreError::shape_mismatch(format!(
            "received {} root observations for {} distinct membership entries",
            root_observations.len(),
            root_membership.distinct_count()
        )));
    }

    let tree_seed = SeedMixer::tree(config.seed, tree_index);
    let mut tree = ProximityTree::new();
    let root = tree.push(TreeNode::Leaf(LeafNode::placeholder(&root_membership)));
    let mut stack = vec![BuildWork {
        node: root,
        depth: 0,
        path: 0,
        membership: root_membership,
        observations: root_observations,
    }];

    while let Some(work) = stack.pop() {
        if stopping_reason(config, work.depth, &work.membership).is_some() {
            write_leaf(&mut tree, &work, &mut leaf)?;
            continue;
        }

        let selected = select(
            tree_seed,
            work.depth,
            work.path,
            &work.membership,
            &work.observations,
            workspace,
        )?;
        let Some(selected) = selected else {
            write_leaf(&mut tree, &work, &mut leaf)?;
            continue;
        };

        let (candidate, _score, partition) = selected.into_parts();
        if candidate.branch_count() != partition.branch_count() {
            return Err(CoreError::shape_mismatch(
                "winning candidate and partition have different branch counts",
            ));
        }
        let children = build_child_memberships(&work.membership, &work.observations, &partition)?;
        let (distance, selected_dimensions, exemplars) = candidate.into_parts();
        let mut branches = Vec::with_capacity(children.len());
        let mut pending = Vec::with_capacity(children.len());
        for (branch, child) in children.into_iter().enumerate() {
            let (membership, observations) = child.into_parts();
            let child_id = tree.push(TreeNode::Leaf(LeafNode::placeholder(&membership)));
            branches.push(ExemplarBranch::new(exemplars[branch], child_id));
            pending.push(BuildWork {
                node: child_id,
                depth: work.depth + 1,
                path: SeedMixer::child_path(work.path, branch),
                membership,
                observations,
            });
        }
        tree.replace(
            work.node,
            TreeNode::Split(SplitNode::new(distance, selected_dimensions, branches)),
        )
        .ok_or_else(|| CoreError::invalid_data("reserved tree node was unavailable"))?;

        for child in pending.into_iter().rev() {
            stack.push(child);
        }
    }
    Ok(tree)
}

fn write_leaf<MakeLeaf>(
    tree: &mut ProximityTree,
    work: &BuildWork,
    leaf: &mut MakeLeaf,
) -> CoreResult<()>
where
    MakeLeaf: FnMut(&NodeMembership, &[ObservationIndex]) -> CoreResult<LeafPayload>,
{
    let payload = leaf(&work.membership, &work.observations)?;
    tree.replace(
        work.node,
        TreeNode::Leaf(LeafNode::new(&work.membership, payload)),
    )
    .ok_or_else(|| CoreError::invalid_data("reserved tree node was unavailable"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::build_tree_iterative;
    use crate::forest::{LeafPayload, ObservationIndex, TreeNode, TreeWorkspace};
    use crate::{ForestConfig, NodeMembership, SampleId};

    #[test]
    fn no_selected_candidate_produces_one_leaf() {
        let config = ForestConfig::new(NonZeroUsize::MIN, NonZeroUsize::MIN, 8);
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1)), Some(SampleId::new(2))])
                .expect("membership");
        let tree = build_tree_iterative(
            &config,
            0,
            membership,
            vec![ObservationIndex::new(0), ObservationIndex::new(1)],
            &mut TreeWorkspace::new(),
            |_tree_seed, _depth, _path, _membership, _observations, _workspace| Ok(None),
            |_membership, _observations| Ok(LeafPayload::Empty),
        )
        .expect("tree");
        assert_eq!(tree.len(), 1);
        assert!(matches!(tree.nodes()[0], TreeNode::Leaf(_)));
    }
}
