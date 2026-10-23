//! Scheduling-independent indexed parallel execution.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::{CoreError, CoreResult};

/// Executes indexed work in parallel and returns results in ascending index order.
///
/// The callback receives stable indices rather than scheduling-dependent worker
/// counters. This preserves deterministic seed derivation when each operation
/// derives randomness from its index. `thread_count == 0` selects the available
/// hardware parallelism. A count of one executes without spawning threads.
///
/// # Errors
///
/// Returns the first error in ascending result-index order after all workers
/// have stopped or completed.
pub fn parallel_indexed<T, Work>(
    item_count: usize,
    thread_count: usize,
    work: Work,
) -> CoreResult<Vec<T>>
where
    T: Send,
    Work: Fn(usize) -> CoreResult<T> + Sync,
{
    if item_count == 0 {
        return Ok(Vec::new());
    }
    let requested = if thread_count == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        thread_count
    };
    let workers = requested.max(1).min(item_count);
    if workers == 1 {
        return (0..item_count).map(work).collect();
    }

    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<CoreResult<T>>>> =
        (0..item_count).map(|_| Mutex::new(None)).collect();
    let failed = AtomicUsize::new(usize::MAX);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let work = &work;
            let next = &next;
            let slots = &slots;
            let failed = &failed;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= item_count || index > failed.load(Ordering::Acquire) {
                        break;
                    }
                    let result = work(index);
                    if result.is_err() {
                        failed.fetch_min(index, Ordering::AcqRel);
                    }
                    if let Ok(mut slot) = slots[index].lock() {
                        *slot = Some(result);
                    } else {
                        failed.fetch_min(index, Ordering::AcqRel);
                        break;
                    }
                }
            });
        }
    });

    let mut values = Vec::with_capacity(item_count);
    for (index, slot) in slots.into_iter().enumerate() {
        let result = slot
            .into_inner()
            .map_err(|_| CoreError::invalid_data("parallel result lock was poisoned"))?
            .ok_or_else(|| {
                CoreError::invalid_data(format!(
                    "parallel operation did not produce result index {index}"
                ))
            })?;
        values.push(result?);
    }
    Ok(values)
}

/// Shared immutable value convenient for scoped parallel closures.
#[derive(Debug)]
pub struct Shared<T>(Arc<T>);

impl<T> Shared<T> {
    /// Wraps a value in atomic shared ownership.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self(Arc::new(value))
    }

    /// Returns the shared value.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.0
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::parallel_indexed;

    #[test]
    fn preserves_index_order() {
        let values = parallel_indexed(32, 4, |index| Ok(index * index)).expect("parallel work");
        assert_eq!(values[0], 0);
        assert_eq!(values[7], 49);
        assert_eq!(values[31], 961);
    }
}
