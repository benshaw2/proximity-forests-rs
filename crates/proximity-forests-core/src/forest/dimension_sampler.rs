//! Sampling of distinct node dimensions.

use std::collections::HashSet;

use crate::{CoreError, CoreResult, DimensionSelection, ModelRng};

/// Samples sorted distinct dimension indices for one node.
pub fn sample_dimensions(
    available: usize,
    selection: DimensionSelection,
    rng: &mut ModelRng,
) -> CoreResult<Vec<usize>> {
    let count = selection.selected_count(available)?;
    if count == available {
        return Ok((0..available).collect());
    }

    let mut selected = if count.saturating_mul(4) < available {
        floyd_sample(available, count, rng)?
    } else {
        partial_fisher_yates(available, count, rng)?
    };
    selected.sort_unstable();
    Ok(selected)
}

fn partial_fisher_yates(
    available: usize,
    count: usize,
    rng: &mut ModelRng,
) -> CoreResult<Vec<usize>> {
    let mut values: Vec<usize> = (0..available).collect();
    for index in 0..count {
        let swap_with = index + rng.index(available - index)?;
        values.swap(index, swap_with);
    }
    values.truncate(count);
    Ok(values)
}

fn floyd_sample(available: usize, count: usize, rng: &mut ModelRng) -> CoreResult<Vec<usize>> {
    if count > available {
        return Err(CoreError::invalid_configuration(
            "dimension_selection",
            "cannot select more dimensions than are available",
        ));
    }
    let mut selected = HashSet::with_capacity(count);
    for current in (available - count)..available {
        let candidate = rng.index(current + 1)?;
        if !selected.insert(candidate) {
            selected.insert(current);
        }
    }
    Ok(selected.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::sample_dimensions;
    use crate::{DimensionSelection, ModelRng};

    #[test]
    fn samples_sorted_distinct_dimensions() {
        let result = sample_dimensions(
            100,
            DimensionSelection::FixedCount(7),
            &mut ModelRng::from_seed(3),
        )
        .expect("valid selection");
        assert_eq!(result.len(), 7);
        assert!(result.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(result.iter().all(|&index| index < 100));
    }

    #[test]
    fn all_dimensions_are_returned_without_sampling() {
        let result = sample_dimensions(4, DimensionSelection::All, &mut ModelRng::from_seed(3))
            .expect("valid selection");
        assert_eq!(result, vec![0, 1, 2, 3]);
    }
}
