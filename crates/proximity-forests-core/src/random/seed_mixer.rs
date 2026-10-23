//! Scheduling-independent derivation of deterministic 64-bit seeds.

use super::RandomPurpose;

/// Mixes stable model identities into independent deterministic seeds.
///
/// The mixer is independent of the random-number generator used by sampling.
/// Forest, tree, node, candidate, observation, and purpose identifiers can be
/// combined before initializing an RNG. This prevents parallel task scheduling
/// from changing stochastic choices.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SeedMixer;

impl SeedMixer {
    /// `SplitMix64` additive constant used to separate ordered components.
    const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    /// Applies the `SplitMix64` finalizer to one 64-bit value.
    #[must_use]
    pub const fn mix64(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Derives a seed from a base seed and ordered discriminator values.
    ///
    /// An empty component list returns `base_seed`. The order of components is
    /// significant, and callers should use stable semantic identities rather
    /// than counters whose values depend on worker scheduling.
    #[must_use]
    pub fn derive(base_seed: u64, components: &[u64]) -> u64 {
        components
            .iter()
            .enumerate()
            .fold(base_seed, |state, (position, component)| {
                let separated =
                    component.wrapping_add(Self::GOLDEN_GAMMA.wrapping_mul(position as u64 + 1));
                Self::mix64(state ^ Self::mix64(separated))
            })
    }

    /// Derives the seed for one tree.
    #[must_use]
    pub fn tree(forest_seed: u64, tree_index: usize) -> u64 {
        Self::derive(forest_seed, &[tree_index as u64])
    }

    /// Derives a purpose-specific seed at one tree node.
    #[must_use]
    pub fn node_purpose(tree_seed: u64, node_path: u64, purpose: RandomPurpose) -> u64 {
        Self::derive(tree_seed, &[node_path, purpose.discriminator()])
    }

    /// Derives a seed for one split candidate.
    #[must_use]
    pub fn candidate(node_seed: u64, candidate_index: usize) -> u64 {
        Self::derive(node_seed, &[candidate_index as u64])
    }

    /// Derives a seed for one observation identity.
    #[must_use]
    pub fn observation(candidate_seed: u64, sample_id: u64) -> u64 {
        Self::derive(candidate_seed, &[sample_id])
    }

    /// Derives a stable child-path identity from a parent path and branch.
    #[must_use]
    pub fn child_path(parent_path: u64, branch_index: usize) -> u64 {
        Self::derive(parent_path, &[branch_index as u64])
    }
}

#[cfg(test)]
mod tests {
    use super::SeedMixer;
    use crate::RandomPurpose;

    #[test]
    fn mix64_has_stable_known_values() {
        assert_eq!(SeedMixer::mix64(0), 0);
        assert_eq!(SeedMixer::mix64(1), 0x5692_161d_100b_05e5);
        assert_eq!(SeedMixer::mix64(u64::MAX), 0xb4d0_55fc_f2cb_bd7b);
    }

    #[test]
    fn derivation_is_repeatable_and_ordered() {
        assert_eq!(
            SeedMixer::derive(42, &[3, 7, 11]),
            SeedMixer::derive(42, &[3, 7, 11])
        );
        assert_ne!(
            SeedMixer::derive(42, &[3, 7, 11]),
            SeedMixer::derive(42, &[11, 7, 3])
        );
    }

    #[test]
    fn empty_components_preserve_the_base_seed() {
        assert_eq!(SeedMixer::derive(42, &[]), 42);
    }

    #[test]
    fn semantic_purposes_produce_distinct_streams() {
        let tree = SeedMixer::tree(4, 2);
        assert_ne!(
            SeedMixer::node_purpose(tree, 1, RandomPurpose::Dimensions),
            SeedMixer::node_purpose(tree, 1, RandomPurpose::CandidateExemplars)
        );
    }
}
