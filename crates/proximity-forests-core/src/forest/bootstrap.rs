//! Bootstrap sampling for tree construction.

use crate::{CoreError, CoreResult, ModelRng};

/// Bootstrap draw, multiplicity, and out-of-bag information for one tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BootstrapSample {
    sampled_positions: Vec<usize>,
    multiplicities: Vec<usize>,
    out_of_bag_positions: Vec<usize>,
}

impl BootstrapSample {
    /// Draws `sample_count` positions with replacement from `0..sample_count`.
    pub fn draw(sample_count: usize, rng: &mut ModelRng) -> CoreResult<Self> {
        if sample_count == 0 {
            return Err(CoreError::invalid_data(
                "bootstrap sampling requires at least one observation",
            ));
        }

        let mut sampled_positions = Vec::with_capacity(sample_count);
        let mut multiplicities = vec![0usize; sample_count];
        for _ in 0..sample_count {
            let position = rng.index(sample_count)?;
            sampled_positions.push(position);
            multiplicities[position] = multiplicities[position]
                .checked_add(1)
                .ok_or_else(|| CoreError::invalid_data("bootstrap multiplicity overflow"))?;
        }
        let out_of_bag_positions = multiplicities
            .iter()
            .enumerate()
            .filter_map(|(position, &count)| (count == 0).then_some(position))
            .collect();

        Ok(Self {
            sampled_positions,
            multiplicities,
            out_of_bag_positions,
        })
    }

    /// Creates the no-bootstrap identity sample.
    #[must_use]
    pub fn identity(sample_count: usize) -> Self {
        Self {
            sampled_positions: (0..sample_count).collect(),
            multiplicities: vec![1; sample_count],
            out_of_bag_positions: Vec::new(),
        }
    }

    /// Returns the tree-local sampled positions, including repeated draws.
    #[must_use]
    pub fn sampled_positions(&self) -> &[usize] {
        &self.sampled_positions
    }

    /// Returns one multiplicity for every original training position.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }

    /// Returns original training positions absent from the bootstrap sample.
    #[must_use]
    pub fn out_of_bag_positions(&self) -> &[usize] {
        &self.out_of_bag_positions
    }
}

#[cfg(test)]
mod tests {
    use super::BootstrapSample;
    use crate::ModelRng;

    #[test]
    fn bootstrap_counts_sum_to_training_size() {
        let sample =
            BootstrapSample::draw(100, &mut ModelRng::from_seed(9)).expect("valid bootstrap");
        assert_eq!(sample.sampled_positions().len(), 100);
        assert_eq!(sample.multiplicities().iter().sum::<usize>(), 100);
        for &position in sample.out_of_bag_positions() {
            assert_eq!(sample.multiplicities()[position], 0);
        }
    }

    #[test]
    fn identity_sample_has_no_out_of_bag_rows() {
        let sample = BootstrapSample::identity(4);
        assert_eq!(sample.sampled_positions(), &[0, 1, 2, 3]);
        assert_eq!(sample.multiplicities(), &[1, 1, 1, 1]);
        assert!(sample.out_of_bag_positions().is_empty());
    }
}
