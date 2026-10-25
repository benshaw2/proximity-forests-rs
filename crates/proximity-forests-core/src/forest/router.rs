//! Fused query-to-exemplar routing with current-best cutoff propagation.

use crate::{
    CoreError, CoreResult, DistanceCutoff, DistanceEvaluator, DistanceScratch, ModelRng,
    MultivariateSeriesView, NumericVectorView, TiePolicy,
};

use super::{CandidateSplit, ObservationIndex};

/// Winning candidate branch and exact distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteResult {
    branch: usize,
    distance: f64,
}

impl RouteResult {
    /// Returns the winning branch index.
    #[must_use]
    pub const fn branch(self) -> usize {
        self.branch
    }

    /// Returns the winning distance.
    #[must_use]
    pub const fn distance(self) -> f64 {
        self.distance
    }
}

/// Routes one vector observation to the nearest candidate exemplar.
///
/// Distances are evaluated and consumed immediately. Every comparison after the
/// first receives the current best distance as a cutoff, so compatible kernels
/// can abandon without constructing a temporary distance vector.
///
/// # Errors
///
/// Returns an error for missing exemplars, invalid distances, or missing RNG
/// state when randomized tie handling is requested. Positive infinity denotes
/// an unavailable comparison. If every comparison is unavailable, all branches
/// form an exact tie resolved by the configured tie policy.
pub fn route_vector(
    query: NumericVectorView<'_>,
    observations: &[NumericVectorView<'_>],
    candidate: &CandidateSplit,
    tie_policy: TiePolicy,
    rng: Option<&mut ModelRng>,
    scratch: &mut DistanceScratch,
) -> CoreResult<RouteResult> {
    let evaluator = DistanceEvaluator::new(candidate.distance().clone());
    let selected = selected(candidate);
    route(
        candidate.exemplars(),
        tie_policy,
        rng,
        |exemplar, cutoff, scratch| {
            let exemplar = observations.get(exemplar.get()).copied().ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "exemplar observation {} is outside vector observation count {}",
                    exemplar.get(),
                    observations.len()
                ))
            })?;
            evaluator.evaluate_vector(query, exemplar, selected, cutoff, scratch)
        },
        scratch,
    )
}

/// Routes one multivariate observation to the nearest candidate exemplar.
///
/// # Errors
///
/// Returns an error for missing exemplars, invalid distances, or missing RNG
/// state when randomized tie handling is requested.
pub fn route_multivariate(
    query: MultivariateSeriesView<'_>,
    observations: &[MultivariateSeriesView<'_>],
    candidate: &CandidateSplit,
    tie_policy: TiePolicy,
    rng: Option<&mut ModelRng>,
    scratch: &mut DistanceScratch,
) -> CoreResult<RouteResult> {
    let evaluator = DistanceEvaluator::new(candidate.distance().clone());
    let selected = selected(candidate);
    route(
        candidate.exemplars(),
        tie_policy,
        rng,
        |exemplar, cutoff, scratch| {
            let exemplar = observations.get(exemplar.get()).copied().ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "exemplar observation {} is outside multivariate observation count {}",
                    exemplar.get(),
                    observations.len()
                ))
            })?;
            evaluator.evaluate_multivariate(query, exemplar, selected, cutoff, scratch)
        },
        scratch,
    )
}

fn route<Evaluate>(
    exemplars: &[ObservationIndex],
    tie_policy: TiePolicy,
    mut rng: Option<&mut ModelRng>,
    mut evaluate: Evaluate,
    scratch: &mut DistanceScratch,
) -> CoreResult<RouteResult>
where
    Evaluate: FnMut(ObservationIndex, DistanceCutoff, &mut DistanceScratch) -> CoreResult<f64>,
{
    let mut best = f64::INFINITY;
    let mut winner = 0usize;
    let mut tie_count = 0usize;

    for (branch, &exemplar) in exemplars.iter().enumerate() {
        let cutoff = if best.is_finite() {
            DistanceCutoff::finite(best)?
        } else {
            DistanceCutoff::None
        };
        let distance = evaluate(exemplar, cutoff, scratch)?;
        validate_distance(distance)?;
        if distance < best {
            best = distance;
            winner = branch;
            tie_count = 1;
        } else if distance == best {
            tie_count += 1;
            if tie_policy == TiePolicy::Random {
                let rng = rng.as_deref_mut().ok_or_else(|| {
                    CoreError::invalid_configuration(
                        "tie_policy",
                        "random routing ties require a random generator",
                    )
                })?;
                if rng.index(tie_count)? == 0 {
                    winner = branch;
                }
            }
        }
    }

    Ok(RouteResult {
        branch: winner,
        distance: best,
    })
}

#[inline]
fn selected(candidate: &CandidateSplit) -> Option<&[usize]> {
    (!candidate.selected_dimensions().is_empty()).then_some(candidate.selected_dimensions())
}

#[inline]
fn validate_distance(distance: f64) -> CoreResult<()> {
    if distance.is_nan() || distance < 0.0 {
        Err(CoreError::invalid_data(
            "routing distances must be nonnegative and not NaN",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::route_vector;
    use super::{CandidateSplit, ObservationIndex};
    use crate::{DistanceConfig, DistanceScratch, NumericVectorView, TiePolicy};

    #[test]
    fn routes_without_materializing_distances() {
        let values = [[0.0_f64], [2.0], [5.0]];
        let observations: Vec<_> = values
            .iter()
            .map(|values| NumericVectorView::from_f64(values))
            .collect();
        let candidate = CandidateSplit::new(
            DistanceConfig::SquaredEuclidean,
            Vec::new(),
            vec![ObservationIndex::new(0), ObservationIndex::new(2)],
        )
        .expect("candidate");
        let result = route_vector(
            observations[1],
            &observations,
            &candidate,
            TiePolicy::First,
            None,
            &mut DistanceScratch::new(),
        )
        .expect("route");
        assert_eq!(result.branch(), 0);
        assert_eq!(result.distance(), 4.0);
    }
    #[test]
    fn all_unavailable_routes_follow_the_tie_policy() {
        use crate::NoOverlapPolicy;
        let values = [[f64::NAN], [f64::NAN]];
        let observations: Vec<_> = values
            .iter()
            .map(|values| NumericVectorView::from_f64(values))
            .collect();
        let candidate = CandidateSplit::new(
            DistanceConfig::NaNSquaredEuclidean {
                no_overlap: NoOverlapPolicy::Infinite,
            },
            Vec::new(),
            vec![ObservationIndex::new(0), ObservationIndex::new(1)],
        )
        .expect("candidate");
        let first = route_vector(
            observations[0],
            &observations,
            &candidate,
            TiePolicy::First,
            None,
            &mut DistanceScratch::new(),
        )
        .expect("unavailable route");
        assert_eq!(first.branch(), 0);
        assert_eq!(first.distance(), f64::INFINITY);

        let mut first_rng = crate::ModelRng::from_seed(17);
        let mut second_rng = crate::ModelRng::from_seed(17);
        let random_first = route_vector(
            observations[0],
            &observations,
            &candidate,
            TiePolicy::Random,
            Some(&mut first_rng),
            &mut DistanceScratch::new(),
        )
        .expect("random unavailable route");
        let random_second = route_vector(
            observations[0],
            &observations,
            &candidate,
            TiePolicy::Random,
            Some(&mut second_rng),
            &mut DistanceScratch::new(),
        )
        .expect("repeat random unavailable route");
        assert_eq!(random_first, random_second);
        assert_eq!(random_first.distance(), f64::INFINITY);
    }
}
