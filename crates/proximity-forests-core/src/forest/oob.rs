//! Compact out-of-bag membership derived from persisted per-tree samples.
//!
//! [`TreeSample`] remains the sole persisted source of truth. This module
//! transposes tree-to-sample OOB lists into a cache-friendly sample-to-tree
//! index for scoring without duplicating serialized state.

use crate::{CoreError, CoreResult};

use super::TreeSample;

/// Compressed sample-to-OOB-tree membership.
///
/// `offsets[s]..offsets[s + 1]` selects the stable ascending tree indices for
/// which training sample `s` was out of bag. Empty rows represent zero votes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OobMembership {
    sample_count: usize,
    tree_count: usize,
    offsets: Box<[usize]>,
    tree_indices: Box<[usize]>,
}

impl OobMembership {
    /// Builds a deterministic transpose of per-tree OOB positions.
    pub fn from_tree_samples(sample_count: usize, samples: &[TreeSample]) -> CoreResult<Self> {
        if sample_count == 0 {
            return Err(CoreError::invalid_data(
                "OOB membership requires at least one training sample",
            ));
        }
        if samples.is_empty() {
            return Err(CoreError::invalid_data(
                "OOB membership requires at least one fitted tree",
            ));
        }

        let mut counts = vec![0usize; sample_count];
        for sample in samples {
            if sample.multiplicities().len() != sample_count {
                return Err(CoreError::shape_mismatch(
                    "tree-sample metadata does not match the training sample count",
                ));
            }
            let mut previous = None;
            for &position in sample.out_of_bag_positions() {
                if position >= sample_count {
                    return Err(CoreError::invalid_data(
                        "OOB position is outside the training collection",
                    ));
                }
                if previous.is_some_and(|value| position <= value) {
                    return Err(CoreError::invalid_data(
                        "OOB positions must be strictly increasing",
                    ));
                }
                counts[position] = counts[position]
                    .checked_add(1)
                    .ok_or_else(|| CoreError::invalid_data("OOB vote count overflowed usize"))?;
                previous = Some(position);
            }
        }

        let mut offsets = Vec::with_capacity(sample_count + 1);
        offsets.push(0usize);
        for count in counts {
            let next = offsets
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(count)
                .ok_or_else(|| CoreError::invalid_data("OOB membership size overflowed usize"))?;
            offsets.push(next);
        }
        let total = offsets.last().copied().unwrap_or(0);
        let mut tree_indices = vec![0usize; total];
        let mut cursors = offsets[..sample_count].to_vec();
        for (tree_index, sample) in samples.iter().enumerate() {
            for &position in sample.out_of_bag_positions() {
                let destination = cursors[position];
                tree_indices[destination] = tree_index;
                cursors[position] += 1;
            }
        }

        Ok(Self {
            sample_count,
            tree_count: samples.len(),
            offsets: offsets.into_boxed_slice(),
            tree_indices: tree_indices.into_boxed_slice(),
        })
    }

    /// Returns the number of training samples represented by the index.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the number of fitted trees represented by the index.
    #[must_use]
    pub const fn tree_count(&self) -> usize {
        self.tree_count
    }

    /// Returns stable ascending OOB tree indices for one training sample.
    #[must_use]
    pub fn trees_for_sample(&self, sample: usize) -> Option<&[usize]> {
        let start = *self.offsets.get(sample)?;
        let end = *self.offsets.get(sample + 1)?;
        self.tree_indices.get(start..end)
    }

    /// Returns one OOB vote count per training sample.
    #[must_use]
    pub fn vote_counts(&self) -> Vec<usize> {
        self.offsets
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect()
    }

    /// Returns how many training samples receive no OOB votes.
    #[must_use]
    pub fn zero_vote_count(&self) -> usize {
        self.offsets
            .windows(2)
            .filter(|pair| pair[0] == pair[1])
            .count()
    }

    /// Iterates `(sample_index, oob_tree_indices)` in sample order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = (usize, &[usize])> {
        (0..self.sample_count).map(|sample| {
            let start = self.offsets[sample];
            let end = self.offsets[sample + 1];
            (sample, &self.tree_indices[start..end])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::OobMembership;
    use crate::forest::TreeSample;

    #[test]
    fn transposes_tree_samples_in_stable_order() {
        let samples = vec![
            TreeSample::from_parts(vec![0, 2], vec![1, 0, 1, 0], vec![1, 3]).unwrap(),
            TreeSample::from_parts(vec![1, 2], vec![0, 1, 1, 0], vec![0, 3]).unwrap(),
            TreeSample::from_parts(vec![0, 1, 3], vec![1, 1, 0, 1], vec![2]).unwrap(),
        ];
        let membership = OobMembership::from_tree_samples(4, &samples).unwrap();
        assert_eq!(membership.sample_count(), 4);
        assert_eq!(membership.tree_count(), 3);
        assert_eq!(membership.trees_for_sample(0), Some(&[1][..]));
        assert_eq!(membership.trees_for_sample(1), Some(&[0][..]));
        assert_eq!(membership.trees_for_sample(2), Some(&[2][..]));
        assert_eq!(membership.trees_for_sample(3), Some(&[0, 1][..]));
        assert_eq!(membership.vote_counts(), vec![1, 1, 1, 2]);
        assert_eq!(membership.zero_vote_count(), 0);
    }

    #[test]
    fn full_sampling_preserves_zero_vote_rows() {
        let samples = vec![
            TreeSample::identity(3).unwrap(),
            TreeSample::identity(3).unwrap(),
        ];
        let membership = OobMembership::from_tree_samples(3, &samples).unwrap();
        assert_eq!(membership.vote_counts(), vec![0, 0, 0]);
        assert_eq!(membership.zero_vote_count(), 3);
        assert!(membership.rows().all(|(_, trees)| trees.is_empty()));
    }

    #[test]
    fn rejects_metadata_with_the_wrong_population() {
        let sample = TreeSample::identity(3).unwrap();
        assert!(OobMembership::from_tree_samples(4, &[sample]).is_err());
    }
}
