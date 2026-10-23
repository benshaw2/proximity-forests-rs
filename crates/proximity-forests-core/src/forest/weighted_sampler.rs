//! Weighted sampling without replacement for exemplar selection.

use crate::{CoreError, CoreResult, ModelRng};

/// Samples distinct indices without replacement in proportion to integer weights.
pub fn sample_weighted_without_replacement(
    weights: &[usize],
    count: usize,
    rng: &mut ModelRng,
) -> CoreResult<Vec<usize>> {
    let positive_count = weights.iter().filter(|&&weight| weight > 0).count();
    if count > positive_count {
        return Err(CoreError::invalid_configuration(
            "weighted_sample_count",
            format!(
                "requested {count} selections but only {positive_count} entries have positive weight"
            ),
        ));
    }

    let mut tree = FenwickTree::new(weights)?;
    let mut selected = Vec::with_capacity(count);
    for _ in 0..count {
        let total = tree.total();
        if total == 0 {
            return Err(CoreError::invalid_data(
                "weighted sampler exhausted positive mass unexpectedly",
            ));
        }
        let draw = rng.index(total)?;
        let index = tree.find_by_order(draw);
        let weight = tree.value(index);
        selected.push(index);
        tree.remove(index, weight);
    }
    Ok(selected)
}

#[derive(Clone, Debug)]
struct FenwickTree {
    tree: Vec<usize>,
    values: Vec<usize>,
}

impl FenwickTree {
    fn new(weights: &[usize]) -> CoreResult<Self> {
        let mut result = Self {
            tree: vec![0; weights.len() + 1],
            values: weights.to_vec(),
        };
        for (index, &weight) in weights.iter().enumerate() {
            result.add(index, weight)?;
        }
        Ok(result)
    }

    fn total(&self) -> usize {
        self.prefix_sum(self.values.len())
    }

    fn value(&self, index: usize) -> usize {
        self.values[index]
    }

    fn add(&mut self, index: usize, amount: usize) -> CoreResult<()> {
        let mut cursor = index + 1;
        while cursor < self.tree.len() {
            self.tree[cursor] = self.tree[cursor]
                .checked_add(amount)
                .ok_or_else(|| CoreError::invalid_data("weighted mass overflowed usize"))?;
            cursor += cursor & cursor.wrapping_neg();
        }
        Ok(())
    }

    fn remove(&mut self, index: usize, amount: usize) {
        self.values[index] -= amount;
        let mut cursor = index + 1;
        while cursor < self.tree.len() {
            self.tree[cursor] -= amount;
            cursor += cursor & cursor.wrapping_neg();
        }
    }

    fn prefix_sum(&self, end: usize) -> usize {
        let mut cursor = end;
        let mut sum = 0usize;
        while cursor > 0 {
            sum += self.tree[cursor];
            cursor &= cursor - 1;
        }
        sum
    }

    fn find_by_order(&self, order: usize) -> usize {
        let mut index = 0usize;
        let mut accumulated = 0usize;
        let mut step = highest_power_of_two_at_most(self.values.len());
        while step > 0 {
            let next = index + step;
            if next < self.tree.len() && accumulated.saturating_add(self.tree[next]) <= order {
                accumulated += self.tree[next];
                index = next;
            }
            step >>= 1;
        }
        index
    }
}

fn highest_power_of_two_at_most(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::sample_weighted_without_replacement;
    use crate::ModelRng;

    #[test]
    fn selects_only_positive_distinct_indices() {
        let result =
            sample_weighted_without_replacement(&[0, 3, 1, 0, 2], 3, &mut ModelRng::from_seed(19))
                .expect("valid weighted sample");
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&index| [1, 2, 4].contains(&index)));
        let mut sorted = result.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn rejects_excessive_sample_count() {
        assert!(
            sample_weighted_without_replacement(&[0, 1], 2, &mut ModelRng::from_seed(1)).is_err()
        );
    }
}
