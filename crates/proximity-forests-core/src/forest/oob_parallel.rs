//! Bounded parallel execution for independent OOB sample rows.
//!
//! The caller supplies the existing [`ExecutionContext`], so OOB work enters
//! the same bounded Rayon pool used by fitting and inference. No nested pool is
//! created. Rayon work stealing balances uneven per-sample vote counts, while
//! indexed collection preserves deterministic sample order and error order.

use crate::CoreResult;

use super::work_estimate::oob_thread_count;
use super::{ExecutionContext, OobMembership, parallel_indexed_init_in};

/// Creates the bounded execution context for one OOB calculation.
///
/// The compact membership index supplies the exact number of routed tree votes,
/// allowing small workloads to avoid Rayon pool construction entirely.
pub fn oob_execution_context(
    membership: &OobMembership,
    requested_workers: usize,
) -> CoreResult<ExecutionContext> {
    let total_votes = membership
        .rows()
        .map(|(_, trees)| trees.len())
        .sum::<usize>();
    ExecutionContext::new(oob_thread_count(
        membership.sample_count(),
        total_votes,
        requested_workers,
    ))
}

/// Executes one independent operation for every training sample's OOB tree row.
///
/// Task-local state is initialized once per sequential run or Rayon job and is
/// reused across samples handled by that job. This is intended for reusable
/// [`crate::DistanceScratch`] and small accumulation buffers.
///
/// Results retain ascending sample order. Zero-vote samples are passed to
/// `work` with an empty tree slice so the numerical layer can assign its own
/// explicit `NaN` or warning semantics.
pub fn parallel_oob_rows_in<T, State, Init, Work>(
    context: &ExecutionContext,
    membership: &OobMembership,
    init: Init,
    work: Work,
) -> CoreResult<Vec<T>>
where
    T: Send,
    State: Send,
    Init: Fn() -> State + Sync + Send,
    Work: Fn(&mut State, usize, &[usize]) -> CoreResult<T> + Sync + Send,
{
    parallel_indexed_init_in(context, membership.sample_count(), init, |state, sample| {
        let trees = membership.trees_for_sample(sample).ok_or_else(|| {
            crate::CoreError::invalid_data("OOB sample index is outside the membership index")
        })?;
        work(state, sample, trees)
    })
}

#[cfg(test)]
mod tests {
    use super::{oob_execution_context, parallel_oob_rows_in};
    use crate::forest::{ExecutionContext, OobMembership, ParallelPolicy, TreeSample};

    fn membership() -> OobMembership {
        OobMembership::from_tree_samples(
            5,
            &[
                TreeSample::from_parts(vec![0, 2, 4], vec![1, 0, 1, 0, 1], vec![1, 3]).unwrap(),
                TreeSample::from_parts(vec![1, 3, 4], vec![0, 1, 0, 1, 1], vec![0, 2]).unwrap(),
                TreeSample::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1, 0], vec![4]).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn operation_context_avoids_small_pools_and_enables_large_work() {
        let small = membership();
        let small_context = oob_execution_context(&small, 4).unwrap();
        assert!(!small_context.is_parallel());

        let population = 128;
        let samples = (0..128)
            .map(|tree| {
                let oob = (0..population)
                    .filter(|sample| (sample + tree) % 2 == 0)
                    .collect::<Vec<_>>();
                let mut multiplicities = vec![1; population];
                for &sample in &oob {
                    multiplicities[sample] = 0;
                }
                let sampled = (0..population)
                    .filter(|&sample| multiplicities[sample] > 0)
                    .collect::<Vec<_>>();
                TreeSample::from_parts(sampled, multiplicities, oob).unwrap()
            })
            .collect::<Vec<_>>();
        let large = OobMembership::from_tree_samples(population, &samples).unwrap();
        let large_context = oob_execution_context(&large, 4).unwrap();
        assert!(large_context.is_parallel());
        assert_eq!(large_context.budget().resolved(), 4);
    }

    #[test]
    fn sequential_and_parallel_rows_are_exactly_equal() {
        let membership = membership();
        let sequential = ExecutionContext::with_policy(1, ParallelPolicy::sequential()).unwrap();
        let parallel = ExecutionContext::new(4).unwrap();
        let work = |state: &mut Vec<usize>, sample: usize, trees: &[usize]| {
            state.clear();
            state.extend_from_slice(trees);
            Ok((sample, state.clone()))
        };
        let left = parallel_oob_rows_in(&sequential, &membership, Vec::new, work).unwrap();
        let right = parallel_oob_rows_in(&parallel, &membership, Vec::new, work).unwrap();
        assert_eq!(left, right);
        assert_eq!(right[0], (0, vec![1]));
        assert_eq!(right[4], (4, vec![2]));
    }

    #[test]
    fn zero_vote_rows_are_retained_in_sample_order() {
        let membership = OobMembership::from_tree_samples(
            3,
            &[
                TreeSample::identity(3).unwrap(),
                TreeSample::identity(3).unwrap(),
            ],
        )
        .unwrap();
        let context = ExecutionContext::new(4).unwrap();
        let rows = parallel_oob_rows_in(
            &context,
            &membership,
            || (),
            |_, sample, trees| Ok((sample, trees.len())),
        )
        .unwrap();
        assert_eq!(rows, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn errors_remain_in_lowest_sample_order() {
        let membership = membership();
        let context = ExecutionContext::new(4).unwrap();
        let error = parallel_oob_rows_in(
            &context,
            &membership,
            || (),
            |_, sample, _| {
                if sample == 1 || sample == 4 {
                    return Err(crate::CoreError::invalid_data(format!("sample {sample}")));
                }
                Ok(sample)
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("sample 1"));
    }
}
