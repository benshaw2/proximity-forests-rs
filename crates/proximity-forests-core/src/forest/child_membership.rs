//! Child membership construction for a winning candidate partition.

use crate::{CoreError, CoreResult};

use super::{CandidatePartition, NodeMembership, ObservationIndex};

/// Membership and dataset-row mapping for one child node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildMembership {
    membership: NodeMembership,
    observations: Vec<ObservationIndex>,
}

impl ChildMembership {
    /// Returns compact child membership.
    #[must_use]
    pub const fn membership(&self) -> &NodeMembership {
        &self.membership
    }

    /// Returns one dataset row per distinct child observation.
    #[must_use]
    pub fn observations(&self) -> &[ObservationIndex] {
        &self.observations
    }

    /// Consumes the child mapping.
    #[must_use]
    pub fn into_parts(self) -> (NodeMembership, Vec<ObservationIndex>) {
        (self.membership, self.observations)
    }
}

/// Builds child memberships only after a candidate has won.
///
/// Stable identities remain deduplicated. Identity-free occurrences remain
/// conservatively distinct, matching `NodeMembership::from_occurrences`.
///
/// # Errors
///
/// Returns an error for mapping mismatches or invalid partition branches.
pub fn build_child_memberships(
    parent: &NodeMembership,
    parent_observations: &[ObservationIndex],
    partition: &CandidatePartition,
) -> CoreResult<Vec<ChildMembership>> {
    if parent_observations.len() != parent.distinct_count()
        || partition.assignments().len() != parent.distinct_count()
    {
        return Err(CoreError::shape_mismatch(
            "parent membership, observations, and partition must have equal distinct counts",
        ));
    }

    let mut occurrence_ids = vec![Vec::new(); partition.branch_count()];
    let mut occurrence_observations = vec![Vec::new(); partition.branch_count()];

    for &distinct in parent.occurrence_to_distinct() {
        let branch = partition
            .branch_of(distinct)
            .ok_or_else(|| CoreError::invalid_data("parent distinct assignment was unavailable"))?;
        occurrence_ids[branch].push(parent.sample_ids()[distinct]);
        occurrence_observations[branch].push(parent_observations[distinct]);
    }

    let mut children = Vec::with_capacity(partition.branch_count());
    for branch in 0..partition.branch_count() {
        let membership = NodeMembership::from_occurrences(&occurrence_ids[branch])?;
        let observations = membership
            .representative_occurrences()
            .iter()
            .map(|&occurrence| occurrence_observations[branch][occurrence])
            .collect();
        children.push(ChildMembership {
            membership,
            observations,
        });
    }
    Ok(children)
}

#[cfg(test)]
mod tests {
    use super::build_child_memberships;
    use super::{CandidatePartition, ObservationIndex};
    use crate::{NodeMembership, SampleId};

    #[test]
    fn preserves_multiplicity_in_children() {
        let parent = NodeMembership::from_occurrences(&[
            Some(SampleId::new(1)),
            Some(SampleId::new(1)),
            Some(SampleId::new(2)),
        ])
        .expect("parent");
        let observations = [ObservationIndex::new(4), ObservationIndex::new(9)];
        let partition =
            CandidatePartition::from_assignments(&parent, 2, vec![1, 0]).expect("partition");
        let children =
            build_child_memberships(&parent, &observations, &partition).expect("children");
        assert_eq!(children[0].membership().occurrence_count(), 1);
        assert_eq!(children[1].membership().occurrence_count(), 2);
        assert_eq!(children[1].observations(), &[ObservationIndex::new(4)]);
    }
}
