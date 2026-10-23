//! Allocation-free closest-exemplar selection with deterministic ties.

use crate::{CoreError, CoreResult, DistanceScratch, ModelRng, TiePolicy};

/// Closest exemplar branch and its exact winning distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestExemplar {
    /// Winning exemplar/branch index.
    pub index: usize,
    /// Exact minimum distance.
    pub distance: f64,
}

/// Selects the closest exemplar from precomputed distances.
///
/// `First` performs one scan with no scratch writes. `Random` uses reservoir
/// sampling over exact ties, so it also avoids storing tied indices while
/// choosing every tied exemplar with equal probability. NaN and negative
/// distances are rejected; positive infinity is permitted.
///
/// # Errors
///
/// Returns an error for empty input, NaN, negative distance, or a missing RNG
/// when randomized tie resolution is requested.
pub fn closest_exemplar(
    distances: &[f64],
    tie_policy: TiePolicy,
    rng: Option<&mut ModelRng>,
    _scratch: &mut DistanceScratch,
) -> CoreResult<ClosestExemplar> {
    if distances.is_empty() {
        return Err(CoreError::invalid_data(
            "closest-exemplar selection requires at least one distance",
        ));
    }

    match tie_policy {
        TiePolicy::First => closest_first(distances),
        TiePolicy::Random => {
            let rng = rng.ok_or_else(|| {
                CoreError::invalid_configuration(
                    "tie_policy",
                    "random tie resolution requires a random generator",
                )
            })?;
            closest_random(distances, rng)
        }
    }
}

#[inline]
fn closest_first(distances: &[f64]) -> CoreResult<ClosestExemplar> {
    validate_distance(distances[0])?;
    let mut winner = 0usize;
    let mut best = distances[0];

    for (index, &distance) in distances.iter().enumerate().skip(1) {
        validate_distance(distance)?;
        if distance < best {
            best = distance;
            winner = index;
        }
    }

    Ok(ClosestExemplar {
        index: winner,
        distance: best,
    })
}

fn closest_random(distances: &[f64], rng: &mut ModelRng) -> CoreResult<ClosestExemplar> {
    validate_distance(distances[0])?;
    let mut winner = 0usize;
    let mut best = distances[0];
    let mut tie_count = 1usize;

    for (index, &distance) in distances.iter().enumerate().skip(1) {
        validate_distance(distance)?;
        if distance < best {
            best = distance;
            winner = index;
            tie_count = 1;
        } else if distance == best {
            tie_count += 1;
            // Reservoir sampling: replace the current winner with probability
            // 1 / tie_count, yielding a uniform choice over all exact ties.
            if rng.index(tie_count)? == 0 {
                winner = index;
            }
        }
    }

    Ok(ClosestExemplar {
        index: winner,
        distance: best,
    })
}

#[inline]
fn validate_distance(distance: f64) -> CoreResult<()> {
    if distance.is_nan() || distance < 0.0 {
        Err(CoreError::invalid_data(
            "exemplar distances must be nonnegative and not NaN",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::closest_exemplar;
    use crate::{DistanceScratch, ModelRng, TiePolicy};

    #[test]
    fn first_policy_chooses_lowest_tied_index() {
        let result = closest_exemplar(
            &[3.0, 1.0, 1.0],
            TiePolicy::First,
            None,
            &mut DistanceScratch::new(),
        )
        .expect("valid distances");
        assert_eq!(result.index, 1);
        assert_eq!(result.distance, 1.0);
    }

    #[test]
    fn random_policy_is_repeatable_for_one_seed() {
        let mut first_rng = ModelRng::from_seed(9);
        let mut second_rng = ModelRng::from_seed(9);
        let first = closest_exemplar(
            &[1.0, 1.0, 1.0],
            TiePolicy::Random,
            Some(&mut first_rng),
            &mut DistanceScratch::new(),
        )
        .expect("valid distances");
        let second = closest_exemplar(
            &[1.0, 1.0, 1.0],
            TiePolicy::Random,
            Some(&mut second_rng),
            &mut DistanceScratch::new(),
        )
        .expect("valid distances");
        assert_eq!(first, second);
    }

    #[test]
    fn positive_infinity_is_a_valid_exact_tie() {
        let result = closest_exemplar(
            &[f64::INFINITY, f64::INFINITY],
            TiePolicy::First,
            None,
            &mut DistanceScratch::new(),
        )
        .expect("valid unavailable routes");
        assert_eq!(result.index, 0);
        assert_eq!(result.distance, f64::INFINITY);
    }

    #[test]
    fn invalid_distance_is_rejected_even_after_a_winner_exists() {
        assert!(
            closest_exemplar(
                &[0.0, f64::NAN],
                TiePolicy::First,
                None,
                &mut DistanceScratch::new(),
            )
            .is_err()
        );
    }
}
