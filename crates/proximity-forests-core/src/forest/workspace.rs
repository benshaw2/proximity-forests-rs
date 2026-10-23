//! Reusable worker-local buffers for tree candidate evaluation.

use crate::{CoreError, CoreResult, DistanceScratch};

use super::{GroupedPartition, NodeMembership, ObservationIndex};

/// Reusable worker-local training buffers.
///
/// One workspace should be owned by one active tree-building worker. It is not
/// shared across threads, which avoids synchronization in candidate routing.
#[derive(Clone, Debug, Default)]
pub struct TreeWorkspace {
    candidate_exemplars: Vec<ObservationIndex>,
    selected_dimensions: Vec<usize>,
    distinct_to_branch: Vec<usize>,
    branch_counts: Vec<usize>,
    branch_offsets: Vec<usize>,
    partition_order: Vec<usize>,
    write_offsets: Vec<usize>,
    distance: DistanceScratch,
}

impl TreeWorkspace {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            candidate_exemplars: Vec::new(),
            selected_dimensions: Vec::new(),
            distinct_to_branch: Vec::new(),
            branch_counts: Vec::new(),
            branch_offsets: Vec::new(),
            partition_order: Vec::new(),
            write_offsets: Vec::new(),
            distance: DistanceScratch::new(),
        }
    }

    /// Returns reusable distance scratch.
    pub const fn distance_scratch(&mut self) -> &mut DistanceScratch {
        &mut self.distance
    }

    /// Clears and returns candidate-exemplar storage.
    pub fn candidate_exemplars(&mut self) -> &mut Vec<ObservationIndex> {
        self.candidate_exemplars.clear();
        &mut self.candidate_exemplars
    }

    /// Clears and returns selected-dimension storage.
    pub fn selected_dimensions(&mut self) -> &mut Vec<usize> {
        self.selected_dimensions.clear();
        &mut self.selected_dimensions
    }

    /// Resizes assignment storage for one branch per distinct observation.
    pub fn assignments(&mut self, distinct_count: usize) -> &mut [usize] {
        self.distinct_to_branch.resize(distinct_count, 0);
        &mut self.distinct_to_branch
    }

    /// Groups current distinct assignments by branch without allocating new
    /// storage after capacities have been established.
    ///
    /// # Errors
    ///
    /// Returns an error for zero branches, assignment/membership mismatch,
    /// out-of-range branches, or count overflow.
    pub fn group_assignments<'a>(
        &'a mut self,
        membership: &NodeMembership,
        branch_count: usize,
    ) -> CoreResult<GroupedPartition<'a>> {
        if branch_count == 0 {
            return Err(CoreError::invalid_data(
                "partition grouping requires at least one branch",
            ));
        }
        if self.distinct_to_branch.len() != membership.distinct_count() {
            return Err(CoreError::shape_mismatch(format!(
                "workspace has {} assignments for {} distinct observations",
                self.distinct_to_branch.len(),
                membership.distinct_count()
            )));
        }

        self.branch_counts.resize(branch_count, 0);
        self.branch_counts.fill(0);
        for &branch in &self.distinct_to_branch {
            let count = self.branch_counts.get_mut(branch).ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "branch assignment {branch} is outside branch count {branch_count}"
                ))
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| CoreError::invalid_data("branch distinct count overflowed usize"))?;
        }

        self.branch_offsets.resize(branch_count + 1, 0);
        self.branch_offsets[0] = 0;
        for branch in 0..branch_count {
            self.branch_offsets[branch + 1] = self.branch_offsets[branch]
                .checked_add(self.branch_counts[branch])
                .ok_or_else(|| CoreError::invalid_data("partition offset overflowed usize"))?;
        }

        self.partition_order.resize(membership.distinct_count(), 0);
        self.write_offsets.clear();
        self.write_offsets
            .extend_from_slice(&self.branch_offsets[..branch_count]);
        for (distinct, &branch) in self.distinct_to_branch.iter().enumerate() {
            let destination = self.write_offsets[branch];
            self.partition_order[destination] = distinct;
            self.write_offsets[branch] += 1;
        }

        GroupedPartition::new(&self.partition_order, &self.branch_offsets)
    }

    /// Sets one distinct-observation branch assignment.
    ///
    /// # Errors
    ///
    /// Returns an error when the distinct index is outside prepared storage.
    pub fn set_assignment(&mut self, distinct: usize, branch: usize) -> CoreResult<()> {
        let prepared_count = self.distinct_to_branch.len();
        let destination = self.distinct_to_branch.get_mut(distinct).ok_or_else(|| {
            CoreError::invalid_data(format!(
                "distinct assignment index {distinct} is outside prepared count {prepared_count}"
            ))
        })?;
        *destination = branch;
        Ok(())
    }

    /// Returns current distinct assignments.
    #[must_use]
    pub fn current_assignments(&self) -> &[usize] {
        &self.distinct_to_branch
    }
}

#[cfg(test)]
mod tests {
    use super::TreeWorkspace;
    use crate::{NodeMembership, SampleId};

    #[test]
    fn groups_distinct_observations_by_branch() {
        let membership = NodeMembership::from_occurrences(&[
            Some(SampleId::new(1)),
            Some(SampleId::new(2)),
            Some(SampleId::new(3)),
        ])
        .expect("membership");
        let mut workspace = TreeWorkspace::new();
        workspace.assignments(3).copy_from_slice(&[1, 0, 1]);
        let grouped = workspace
            .group_assignments(&membership, 2)
            .expect("grouped partition");
        assert_eq!(grouped.branch(0), Some(&[1][..]));
        assert_eq!(grouped.branch(1), Some(&[0, 2][..]));
    }
}
