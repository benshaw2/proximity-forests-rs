//! Bounded scheduling context for deterministic indexed work.

use crate::{CoreError, CoreResult};

#[cfg(feature = "parallel")]
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Requested worker budget for one native operation.
///
/// A value of zero requests the hardware thread count. The resolved value is
/// computed once when an [`ExecutionContext`] is created and is never queried
/// from inside distance, routing, scoring, or tree-traversal hot loops.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkerBudget {
    requested: usize,
    resolved: usize,
}

impl WorkerBudget {
    /// Resolves an external `n_jobs`-style worker request.
    #[must_use]
    pub fn resolve(requested: usize) -> Self {
        let resolved = if requested == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            requested
        }
        .max(1);
        Self {
            requested,
            resolved,
        }
    }

    /// Returns the caller-supplied value; zero denotes automatic selection.
    #[must_use]
    pub const fn requested(self) -> usize {
        self.requested
    }

    /// Returns the fixed positive worker limit used by the operation.
    #[must_use]
    pub const fn resolved(self) -> usize {
        self.resolved
    }

    /// Reports whether the operation must execute sequentially.
    #[must_use]
    pub const fn is_sequential(self) -> bool {
        self.resolved == 1
    }
}

/// Internal controls for which scheduling levels may consume a worker budget.
///
/// P1 deliberately enables only the coarse levels already present in the
/// released implementation. Later slices will add cost-gated candidate work and
/// query-tree tiling without changing the execution-context contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParallelPolicy {
    tree_fitting: bool,
    query_batches: bool,
    candidate_splits: bool,
    query_tree_tiles: bool,
}

impl ParallelPolicy {
    /// Current automatic policy: preserve released tree/query scheduling only.
    #[must_use]
    pub const fn automatic() -> Self {
        Self {
            tree_fitting: true,
            query_batches: true,
            candidate_splits: true,
            query_tree_tiles: true,
        }
    }

    /// Completely sequential policy, useful for tests and future public control.
    #[must_use]
    pub const fn sequential() -> Self {
        Self {
            tree_fitting: false,
            query_batches: false,
            candidate_splits: false,
            query_tree_tiles: false,
        }
    }

    /// Reports whether independent trees may be scheduled concurrently.
    #[must_use]
    pub const fn tree_fitting(self) -> bool {
        self.tree_fitting
    }

    /// Reports whether independent query rows may be scheduled concurrently.
    #[must_use]
    pub const fn query_batches(self) -> bool {
        self.query_batches
    }

    /// Reports whether split candidates may be scheduled concurrently.
    #[must_use]
    pub const fn candidate_splits(self) -> bool {
        self.candidate_splits
    }

    /// Reports whether query-tree tiles may be scheduled concurrently.
    #[must_use]
    pub const fn query_tree_tiles(self) -> bool {
        self.query_tree_tiles
    }
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self::automatic()
    }
}

/// Operation-scoped bounded executor.
///
/// The sequential representation owns no pool and calls work directly. Parallel
/// contexts own one fixed-size Rayon pool. Nested scheduling in later slices will
/// install work into this same pool, preventing oversubscription from independent
/// native thread groups.
#[derive(Debug)]
pub struct ExecutionContext {
    budget: WorkerBudget,
    policy: ParallelPolicy,
    #[cfg(feature = "parallel")]
    pool: Option<ThreadPool>,
}

impl ExecutionContext {
    /// Creates an automatic-policy context from an external worker request.
    ///
    /// # Errors
    ///
    /// Returns an error if Rayon cannot construct the bounded worker pool.
    pub fn new(thread_count: usize) -> CoreResult<Self> {
        Self::with_policy(thread_count, ParallelPolicy::automatic())
    }

    /// Creates a context with an explicit internal scheduling policy.
    ///
    /// # Errors
    ///
    /// Returns an error if Rayon cannot construct the bounded worker pool.
    pub fn with_policy(thread_count: usize, policy: ParallelPolicy) -> CoreResult<Self> {
        let budget = WorkerBudget::resolve(thread_count);
        #[cfg(feature = "parallel")]
        let pool = if budget.is_sequential() || policy == ParallelPolicy::sequential() {
            None
        } else {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(budget.resolved())
                    .thread_name(|index| format!("proximity-forests-{index}"))
                    .build()
                    .map_err(|error| {
                        CoreError::invalid_configuration(
                            "n_jobs",
                            format!("could not construct the bounded worker pool: {error}"),
                        )
                    })?,
            )
        };
        Ok(Self {
            budget,
            policy,
            #[cfg(feature = "parallel")]
            pool,
        })
    }

    /// Returns the operation's fixed worker budget.
    #[must_use]
    pub const fn budget(&self) -> WorkerBudget {
        self.budget
    }

    /// Returns the internal scheduling policy.
    #[must_use]
    pub const fn policy(&self) -> ParallelPolicy {
        self.policy
    }

    /// Reports whether this context owns a parallel executor.
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        #[cfg(feature = "parallel")]
        {
            self.pool.is_some()
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Runs one closure within this context's bounded pool.
    ///
    /// This is the sole bridge to the scheduling backend. Project modules should
    /// depend on `ExecutionContext`, not Rayon types.
    pub(crate) fn install<R>(&self, operation: impl FnOnce() -> R + Send) -> R
    where
        R: Send,
    {
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            return pool.install(operation);
        }
        operation()
    }
}

/// Executes indexed work using a newly created operation context.
///
/// This compatibility entry point preserves the released API while P2 migrates
/// fit and inference boundaries to create one context and pass it through.
/// Results and errors are returned in ascending stable index order.
pub fn parallel_indexed<T, Work>(
    item_count: usize,
    thread_count: usize,
    work: Work,
) -> CoreResult<Vec<T>>
where
    T: Send,
    Work: Fn(usize) -> CoreResult<T> + Sync + Send,
{
    let context = ExecutionContext::new(thread_count)?;
    parallel_indexed_in(&context, item_count, &work)
}

/// Executes indexed work inside an existing bounded execution context.
///
/// The sequential path is a direct iterator with no task allocation, locks,
/// atomics, or pool entry. Parallel collection through Rayon preserves indexed
/// iterator order, so deterministic reduction and ascending error selection are
/// maintained.
pub fn parallel_indexed_in<T, Work>(
    context: &ExecutionContext,
    item_count: usize,
    work: &Work,
) -> CoreResult<Vec<T>>
where
    T: Send,
    Work: Fn(usize) -> CoreResult<T> + Sync + Send,
{
    if item_count == 0 {
        return Ok(Vec::new());
    }
    if !context.is_parallel() || item_count == 1 {
        return (0..item_count).map(work).collect();
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        return context.install(|| {
            (0..item_count)
                .into_par_iter()
                .map(work)
                .collect::<CoreResult<Vec<_>>>()
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        (0..item_count).map(work).collect()
    }
}

/// Executes indexed work with reusable task-local mutable state.
///
/// The initializer is invoked only for sequential execution or for Rayon jobs,
/// never once per indexed item. Items handled by one job reuse the same state.
/// This is intended for scratch-heavy nested work such as candidate evaluation,
/// where sharing one mutable workspace across candidates is invalid but allocating
/// a fresh workspace for every candidate would be unnecessarily expensive.
///
/// Results and errors retain ascending index order. The sequential path creates
/// exactly one state value and executes a direct loop.
pub fn parallel_indexed_init_in<T, State, Init, Work>(
    context: &ExecutionContext,
    item_count: usize,
    init: Init,
    work: Work,
) -> CoreResult<Vec<T>>
where
    T: Send,
    State: Send,
    Init: Fn() -> State + Sync + Send,
    Work: Fn(&mut State, usize) -> CoreResult<T> + Sync + Send,
{
    if item_count == 0 {
        return Ok(Vec::new());
    }
    if !context.is_parallel() || item_count == 1 {
        let mut state = init();
        return (0..item_count)
            .map(|index| work(&mut state, index))
            .collect();
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        return context.install(|| {
            (0..item_count)
                .into_par_iter()
                .map_init(&init, |state, index| work(state, index))
                .collect::<CoreResult<Vec<_>>>()
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut state = init();
        (0..item_count)
            .map(|index| work(&mut state, index))
            .collect()
    }
}

/// Shared immutable value retained for existing scoped closures.
#[derive(Debug)]
pub struct Shared<T>(std::sync::Arc<T>);

impl<T> Shared<T> {
    /// Wraps a value in atomic shared ownership.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self(std::sync::Arc::new(value))
    }

    /// Returns the shared value.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(std::sync::Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionContext, ParallelPolicy, WorkerBudget, parallel_indexed, parallel_indexed_in,
        parallel_indexed_init_in,
    };
    use crate::CoreError;

    #[test]
    fn resolves_explicit_and_automatic_worker_budgets_once() {
        assert_eq!(WorkerBudget::resolve(4).resolved(), 4);
        assert!(
            !WorkerBudget::resolve(0).is_sequential() || WorkerBudget::resolve(0).resolved() == 1
        );
    }

    #[test]
    fn automatic_policy_enables_candidate_work() {
        let policy = ParallelPolicy::automatic();
        assert!(policy.tree_fitting());
        assert!(policy.query_batches());
        assert!(policy.candidate_splits());
        assert!(policy.query_tree_tiles());
    }

    #[test]
    fn sequential_policy_never_constructs_parallel_execution() {
        let context = ExecutionContext::with_policy(8, ParallelPolicy::sequential())
            .expect("sequential context");
        assert!(!context.is_parallel());
        assert_eq!(context.budget().resolved(), 8);
    }

    #[test]
    fn preserves_index_order() {
        let values = parallel_indexed(32, 4, |index| Ok(index * index)).expect("parallel work");
        assert_eq!(values[0], 0);
        assert_eq!(values[7], 49);
        assert_eq!(values[31], 961);
    }

    #[test]
    fn existing_context_can_be_reused_without_building_another_pool() {
        let context = ExecutionContext::new(4).expect("context");
        let first = parallel_indexed_in(&context, 16, &|index| Ok(index + 1)).expect("first");
        let second = parallel_indexed_in(&context, 16, &|index| Ok(index + 2)).expect("second");
        assert_eq!(first[15], 16);
        assert_eq!(second[15], 17);
    }

    #[test]
    fn one_context_supports_multiple_ordered_phases() {
        let context = ExecutionContext::new(4).expect("context");
        let squares =
            parallel_indexed_in(&context, 64, &|index| Ok(index * index)).expect("squares");
        let shifted = parallel_indexed_in(&context, squares.len(), &|index| Ok(squares[index] + 7))
            .expect("shifted");
        assert_eq!(shifted[0], 7);
        assert_eq!(shifted[17], 296);
        assert_eq!(shifted[63], 3976);
    }

    #[test]
    fn task_local_state_preserves_index_order() {
        let context = ExecutionContext::new(4).expect("context");
        let values = parallel_indexed_init_in(&context, 128, Vec::<usize>::new, |state, index| {
            state.push(index);
            Ok(index * 3)
        })
        .expect("stateful indexed work");
        assert_eq!(values, (0..128).map(|index| index * 3).collect::<Vec<_>>());
    }

    #[test]
    fn sequential_state_is_initialized_once_and_reused() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let context = ExecutionContext::new(1).expect("context");
        let initializations = AtomicUsize::new(0);
        let values = parallel_indexed_init_in(
            &context,
            16,
            || {
                initializations.fetch_add(1, Ordering::Relaxed);
                0_usize
            },
            |state, index| {
                *state += 1;
                Ok(index + *state)
            },
        )
        .expect("sequential stateful work");
        assert_eq!(initializations.load(Ordering::Relaxed), 1);
        assert_eq!(values[0], 1);
        assert_eq!(values[15], 31);
    }

    #[test]
    fn stateful_indexed_errors_remain_ordered() {
        let context = ExecutionContext::new(4).expect("context");
        let error = parallel_indexed_init_in(
            &context,
            32,
            || (),
            |_state, index| {
                if index == 5 || index == 19 {
                    Err(CoreError::invalid_data(format!("stateful failure {index}")))
                } else {
                    Ok(index)
                }
            },
        )
        .expect_err("expected indexed error");
        assert!(error.to_string().contains("stateful failure 5"));
    }

    #[test]
    fn returns_the_lowest_failing_index() {
        let error = parallel_indexed(32, 4, |index| {
            if index == 3 || index == 11 {
                Err(CoreError::invalid_data(format!("failed {index}")))
            } else {
                Ok(index)
            }
        })
        .expect_err("expected indexed failure");
        assert!(error.to_string().contains("failed 3"));
    }
}
