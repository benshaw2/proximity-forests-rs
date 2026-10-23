//! Small deterministic pseudo-random generator for model-internal sampling.

use crate::{CoreError, CoreResult};

/// Deterministic SplitMix64 generator used after semantic seed derivation.
///
/// This generator is fast, small, reproducible, and independent of Java's RNG.
/// It is not cryptographically secure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelRng {
    state: u64,
}

impl ModelRng {
    /// Creates a generator from one already-derived seed.
    #[must_use]
    pub const fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Samples uniformly from the half-open interval `[0, 1)`.
    #[must_use]
    pub fn unit_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Returns the next uniformly distributed 64-bit value.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Samples uniformly from `0..upper` without modulo bias.
    pub fn index(&mut self, upper: usize) -> CoreResult<usize> {
        if upper == 0 {
            return Err(CoreError::invalid_configuration(
                "sampling_upper_bound",
                "sampling requires a positive exclusive upper bound",
            ));
        }

        let upper = u64::try_from(upper).map_err(|_| {
            CoreError::invalid_configuration(
                "sampling_upper_bound",
                "sampling upper bound does not fit in u64",
            )
        })?;
        let rejection_zone = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= rejection_zone {
                return usize::try_from(value % upper)
                    .map_err(|_| CoreError::invalid_data("sampled index does not fit in usize"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelRng;

    #[test]
    fn sequence_is_repeatable() {
        let mut first = ModelRng::from_seed(42);
        let mut second = ModelRng::from_seed(42);
        for _ in 0..16 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn bounded_indices_stay_in_range() {
        let mut rng = ModelRng::from_seed(7);
        for _ in 0..1_000 {
            assert!(rng.index(13).expect("positive bound") < 13);
        }
    }

    #[test]
    fn zero_bound_is_rejected() {
        assert!(ModelRng::from_seed(1).index(0).is_err());
    }
}
