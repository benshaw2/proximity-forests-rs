//! Compact node membership and bootstrap multiplicity metadata.

use std::collections::HashMap;

use crate::{CoreError, CoreResult};

/// Stable identity of one training observation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SampleId(u64);

impl SampleId {
    /// Creates a sample identity from its stable integer representation.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying stable integer representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Deduplicated observations and occurrence mappings for one node.
///
/// Stable duplicate identities share one distinct entry, allowing distances to
/// be evaluated once. An occurrence without a stable identity conservatively
/// receives its own distinct entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeMembership {
    occurrence_to_distinct: Vec<usize>,
    representative_occurrences: Vec<usize>,
    multiplicities: Vec<usize>,
    sample_ids: Vec<Option<SampleId>>,
    all_identities_stable: bool,
}

impl NodeMembership {
    /// Builds compact membership from one identity per bootstrap occurrence.
    ///
    /// # Errors
    ///
    /// Returns an error if a multiplicity counter overflows `usize`.
    pub fn from_occurrences(occurrence_ids: &[Option<SampleId>]) -> CoreResult<Self> {
        let mut stable_to_distinct = HashMap::new();
        let mut occurrence_to_distinct = Vec::with_capacity(occurrence_ids.len());
        let mut representative_occurrences = Vec::new();
        let mut multiplicities = Vec::new();
        let mut sample_ids = Vec::new();
        let mut all_identities_stable = true;

        for (occurrence, &sample_id) in occurrence_ids.iter().enumerate() {
            let distinct = if let Some(id) = sample_id {
                if let Some(&existing) = stable_to_distinct.get(&id) {
                    existing
                } else {
                    let next = representative_occurrences.len();
                    stable_to_distinct.insert(id, next);
                    representative_occurrences.push(occurrence);
                    multiplicities.push(0usize);
                    sample_ids.push(Some(id));
                    next
                }
            } else {
                all_identities_stable = false;
                let next = representative_occurrences.len();
                representative_occurrences.push(occurrence);
                multiplicities.push(0usize);
                sample_ids.push(None);
                next
            };

            multiplicities[distinct] = multiplicities[distinct]
                .checked_add(1)
                .ok_or_else(|| CoreError::invalid_data("node multiplicity overflowed usize"))?;
            occurrence_to_distinct.push(distinct);
        }

        Ok(Self {
            occurrence_to_distinct,
            representative_occurrences,
            multiplicities,
            sample_ids,
            all_identities_stable,
        })
    }

    /// Returns the number of bootstrap occurrences at the node.
    #[must_use]
    pub fn occurrence_count(&self) -> usize {
        self.occurrence_to_distinct.len()
    }

    /// Returns the number of observations requiring distinct distance work.
    #[must_use]
    pub fn distinct_count(&self) -> usize {
        self.representative_occurrences.len()
    }

    /// Maps each occurrence to its compact distinct-observation index.
    #[must_use]
    pub fn occurrence_to_distinct(&self) -> &[usize] {
        &self.occurrence_to_distinct
    }

    /// Returns one representative occurrence position per distinct entry.
    #[must_use]
    pub fn representative_occurrences(&self) -> &[usize] {
        &self.representative_occurrences
    }

    /// Returns bootstrap multiplicity for each distinct entry.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }

    /// Returns a stable identity, when available, for each distinct entry.
    #[must_use]
    pub fn sample_ids(&self) -> &[Option<SampleId>] {
        &self.sample_ids
    }

    /// Reports whether every occurrence supplied a stable identity.
    #[must_use]
    pub const fn all_identities_stable(&self) -> bool {
        self.all_identities_stable
    }
}

#[cfg(test)]
mod tests {
    use super::{NodeMembership, SampleId};

    #[test]
    fn stable_duplicates_are_deduplicated() {
        let ids = [
            Some(SampleId::new(7)),
            Some(SampleId::new(4)),
            Some(SampleId::new(7)),
            Some(SampleId::new(9)),
            Some(SampleId::new(7)),
            Some(SampleId::new(4)),
        ];
        let membership = NodeMembership::from_occurrences(&ids).expect("valid membership");
        assert_eq!(membership.occurrence_count(), 6);
        assert_eq!(membership.distinct_count(), 3);
        assert_eq!(membership.multiplicities(), &[3, 2, 1]);
        assert_eq!(membership.occurrence_to_distinct(), &[0, 1, 0, 2, 0, 1]);
        assert!(membership.all_identities_stable());
    }

    #[test]
    fn missing_identities_are_kept_distinct() {
        let ids = [None, None, Some(SampleId::new(2)), Some(SampleId::new(2))];
        let membership = NodeMembership::from_occurrences(&ids).expect("valid membership");
        assert_eq!(membership.distinct_count(), 3);
        assert_eq!(membership.multiplicities(), &[1, 1, 2]);
        assert!(!membership.all_identities_stable());
    }
}
