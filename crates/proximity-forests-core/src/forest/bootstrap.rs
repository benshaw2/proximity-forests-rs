//! Per-tree observation samples, multiplicities, and out-of-bag membership.

use crate::{CoreError, CoreResult, ModelRng};

/// Selected training observations and their multiplicities for one tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeSample {
    sampled_positions: Vec<usize>,
    multiplicities: Vec<usize>,
    out_of_bag_positions: Vec<usize>,
}

impl TreeSample {
    /// Draws `draw_count` positions with replacement from `0..population`.
    pub fn bootstrap(population: usize, draw_count: usize, rng: &mut ModelRng) -> CoreResult<Self> {
        Self::validate_nonempty(population, draw_count)?;
        let mut sampled_positions = Vec::with_capacity(draw_count);
        let mut multiplicities = vec![0usize; population];
        for _ in 0..draw_count {
            let position = rng.index(population)?;
            sampled_positions.push(position);
            multiplicities[position] = multiplicities[position]
                .checked_add(1)
                .ok_or_else(|| CoreError::invalid_data("tree-sample multiplicity overflow"))?;
        }
        Self::from_valid_draws(sampled_positions, multiplicities)
    }

    /// Selects `sample_count` distinct positions from `0..population`.
    pub fn subsample(
        population: usize,
        sample_count: usize,
        rng: &mut ModelRng,
    ) -> CoreResult<Self> {
        Self::validate_nonempty(population, sample_count)?;
        if sample_count > population {
            return Err(CoreError::invalid_configuration(
                "sample_size",
                "sampling without replacement cannot exceed the training population",
            ));
        }
        let mut positions: Vec<_> = (0..population).collect();
        for selected in 0..sample_count {
            let offset = rng.index(population - selected)?;
            positions.swap(selected, selected + offset);
        }
        positions.truncate(sample_count);
        let mut multiplicities = vec![0usize; population];
        for &position in &positions {
            multiplicities[position] = 1;
        }
        Self::from_valid_draws(positions, multiplicities)
    }

    /// Creates the full-data identity sample.
    pub fn identity(population: usize) -> CoreResult<Self> {
        if population == 0 {
            return Err(CoreError::invalid_data(
                "tree sampling requires at least one observation",
            ));
        }
        Ok(Self {
            sampled_positions: (0..population).collect(),
            multiplicities: vec![1; population],
            out_of_bag_positions: Vec::new(),
        })
    }

    /// Reconstructs a tree sample from persisted ownership components.
    pub(crate) fn from_parts(
        sampled_positions: Vec<usize>,
        multiplicities: Vec<usize>,
        out_of_bag_positions: Vec<usize>,
    ) -> CoreResult<Self> {
        if multiplicities.is_empty() || sampled_positions.is_empty() {
            return Err(CoreError::invalid_data(
                "persisted tree-sample metadata must be nonempty",
            ));
        }
        let mut reconstructed = vec![0usize; multiplicities.len()];
        for &position in &sampled_positions {
            let count = reconstructed.get_mut(position).ok_or_else(|| {
                CoreError::invalid_data("a persisted draw is outside the training collection")
            })?;
            *count = count.checked_add(1).ok_or_else(|| {
                CoreError::invalid_data("persisted tree-sample multiplicity overflow")
            })?;
        }
        if reconstructed != multiplicities {
            return Err(CoreError::invalid_data(
                "persisted multiplicities do not match sampled positions",
            ));
        }
        let expected = Self::oob_from(&multiplicities);
        if out_of_bag_positions != expected {
            return Err(CoreError::invalid_data(
                "persisted out-of-bag positions do not match multiplicities",
            ));
        }
        Ok(Self {
            sampled_positions,
            multiplicities,
            out_of_bag_positions,
        })
    }

    fn validate_nonempty(population: usize, sample_count: usize) -> CoreResult<()> {
        if population == 0 || sample_count == 0 {
            return Err(CoreError::invalid_data(
                "tree sampling requires a nonempty population and sample",
            ));
        }
        Ok(())
    }

    fn from_valid_draws(
        sampled_positions: Vec<usize>,
        multiplicities: Vec<usize>,
    ) -> CoreResult<Self> {
        let out_of_bag_positions = Self::oob_from(&multiplicities);
        Self::from_parts(sampled_positions, multiplicities, out_of_bag_positions)
    }

    fn oob_from(multiplicities: &[usize]) -> Vec<usize> {
        multiplicities
            .iter()
            .enumerate()
            .filter_map(|(position, &count)| (count == 0).then_some(position))
            .collect()
    }

    /// Returns sampled training positions in draw order.
    #[must_use]
    pub fn sampled_positions(&self) -> &[usize] {
        &self.sampled_positions
    }
    /// Returns one selection multiplicity per original training position.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }
    /// Returns original positions absent from this tree sample.
    #[must_use]
    pub fn out_of_bag_positions(&self) -> &[usize] {
        &self.out_of_bag_positions
    }
    /// Consumes the sample and returns positions, multiplicities, and OOB positions.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        (
            self.sampled_positions,
            self.multiplicities,
            self.out_of_bag_positions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::TreeSample;
    use crate::ModelRng;

    #[test]
    fn configurable_bootstrap_counts_match_draw_count() {
        let sample = TreeSample::bootstrap(100, 37, &mut ModelRng::from_seed(9)).unwrap();
        assert_eq!(sample.sampled_positions().len(), 37);
        assert_eq!(sample.multiplicities().iter().sum::<usize>(), 37);
    }

    #[test]
    fn fixed_subsample_is_distinct_and_has_expected_oob_complement() {
        let sample = TreeSample::subsample(10, 4, &mut ModelRng::from_seed(11)).unwrap();
        let mut selected = sample.sampled_positions().to_vec();
        selected.sort_unstable();
        selected.dedup();
        assert_eq!(selected.len(), 4);
        assert_eq!(sample.out_of_bag_positions().len(), 6);
        assert!(sample.multiplicities().iter().all(|&count| count <= 1));
    }

    #[test]
    fn sampling_is_repeatable_for_one_seed() {
        let a = TreeSample::subsample(20, 7, &mut ModelRng::from_seed(17)).unwrap();
        let b = TreeSample::subsample(20, 7, &mut ModelRng::from_seed(17)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn identity_sample_has_no_out_of_bag_rows() {
        let sample = TreeSample::identity(4).unwrap();
        assert_eq!(sample.sampled_positions(), &[0, 1, 2, 3]);
        assert_eq!(sample.multiplicities(), &[1, 1, 1, 1]);
        assert!(sample.out_of_bag_positions().is_empty());
    }

    #[test]
    fn persisted_parts_allow_samples_smaller_than_population() {
        assert!(TreeSample::from_parts(vec![0, 2], vec![1, 0, 1, 0], vec![1, 3]).is_ok());
        assert!(TreeSample::from_parts(vec![0, 2], vec![1, 1, 0, 0], vec![2, 3]).is_err());
    }
}
