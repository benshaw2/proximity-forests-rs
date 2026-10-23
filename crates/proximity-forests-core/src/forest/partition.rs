//! Compact candidate partitions over distinct node observations.

use crate::{CoreError, CoreResult, NodeMembership};

/// Compact branch assignment for one candidate split.
///
/// Distances are evaluated once per distinct observation. Bootstrap occurrence
/// counts are recovered through `NodeMembership::multiplicities`, so scoring
/// remains equivalent to routing every occurrence separately.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidatePartition {
    branch_count: usize,
    distinct_to_branch: Vec<usize>,
    branch_distinct_counts: Vec<usize>,
    branch_occurrence_counts: Vec<usize>,
}

impl CandidatePartition {
    /// Builds a compact partition from one branch per distinct observation.
    ///
    /// # Errors
    ///
    /// Returns an error for zero branches, mismatched assignment length, an
    /// invalid branch, or count overflow.
    pub fn from_assignments(
        membership: &NodeMembership,
        branch_count: usize,
        distinct_to_branch: Vec<usize>,
    ) -> CoreResult<Self> {
        if branch_count == 0 {
            return Err(CoreError::invalid_data(
                "a candidate partition requires at least one branch",
            ));
        }
        if distinct_to_branch.len() != membership.distinct_count() {
            return Err(CoreError::shape_mismatch(format!(
                "received {} branch assignments for {} distinct observations",
                distinct_to_branch.len(),
                membership.distinct_count()
            )));
        }

        let mut branch_distinct_counts = vec![0usize; branch_count];
        let mut branch_occurrence_counts = vec![0usize; branch_count];
        for (distinct, &branch) in distinct_to_branch.iter().enumerate() {
            if branch >= branch_count {
                return Err(CoreError::invalid_data(format!(
                    "branch assignment {branch} is outside branch count {branch_count}"
                )));
            }
            branch_distinct_counts[branch] = branch_distinct_counts[branch]
                .checked_add(1)
                .ok_or_else(|| CoreError::invalid_data("branch distinct count overflowed usize"))?;
            branch_occurrence_counts[branch] = branch_occurrence_counts[branch]
                .checked_add(membership.multiplicities()[distinct])
                .ok_or_else(|| {
                    CoreError::invalid_data("branch occurrence count overflowed usize")
                })?;
        }

        Ok(Self {
            branch_count,
            distinct_to_branch,
            branch_distinct_counts,
            branch_occurrence_counts,
        })
    }

    /// Returns the number of branches.
    #[must_use]
    pub const fn branch_count(&self) -> usize {
        self.branch_count
    }

    /// Returns one branch assignment per distinct observation.
    #[must_use]
    pub fn assignments(&self) -> &[usize] {
        &self.distinct_to_branch
    }

    /// Returns distinct-observation counts by branch.
    #[must_use]
    pub fn distinct_counts(&self) -> &[usize] {
        &self.branch_distinct_counts
    }

    /// Returns bootstrap occurrence counts by branch.
    #[must_use]
    pub fn occurrence_counts(&self) -> &[usize] {
        &self.branch_occurrence_counts
    }

    /// Reports whether every branch receives at least one occurrence.
    #[must_use]
    pub fn has_no_empty_branches(&self) -> bool {
        self.branch_occurrence_counts.iter().all(|&count| count > 0)
    }

    /// Returns total routed occurrence count.
    #[must_use]
    pub fn occurrence_count(&self) -> usize {
        self.branch_occurrence_counts.iter().sum()
    }

    /// Returns the assigned branch for one distinct observation.
    #[must_use]
    pub fn branch_of(&self, distinct: usize) -> Option<usize> {
        self.distinct_to_branch.get(distinct).copied()
    }
}

/// Borrowed branch-grouped view over distinct indices.
#[derive(Clone, Copy, Debug)]
pub struct GroupedPartition<'a> {
    order: &'a [usize],
    offsets: &'a [usize],
}

impl<'a> GroupedPartition<'a> {
    /// Creates a grouped view.
    ///
    /// # Errors
    ///
    /// Returns an error unless offsets begin at zero, are monotone, and end at
    /// the order length.
    pub fn new(order: &'a [usize], offsets: &'a [usize]) -> CoreResult<Self> {
        if offsets.is_empty()
            || offsets[0] != 0
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || offsets[offsets.len() - 1] != order.len()
        {
            return Err(CoreError::invalid_data(
                "grouped partition offsets are inconsistent with order storage",
            ));
        }
        Ok(Self { order, offsets })
    }

    /// Returns the number of branches.
    #[must_use]
    pub fn branch_count(self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns distinct indices assigned to a branch.
    #[must_use]
    pub fn branch(self, branch: usize) -> Option<&'a [usize]> {
        let start = *self.offsets.get(branch)?;
        let end = *self.offsets.get(branch + 1)?;
        self.order.get(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::CandidatePartition;
    use crate::{NodeMembership, SampleId};

    #[test]
    fn occurrence_counts_respect_bootstrap_multiplicity() {
        let membership = NodeMembership::from_occurrences(&[
            Some(SampleId::new(3)),
            Some(SampleId::new(3)),
            Some(SampleId::new(8)),
        ])
        .expect("membership");
        let partition =
            CandidatePartition::from_assignments(&membership, 2, vec![1, 0]).expect("partition");
        assert_eq!(partition.distinct_counts(), &[1, 1]);
        assert_eq!(partition.occurrence_counts(), &[1, 2]);
        assert!(partition.has_no_empty_branches());
    }
}
