//! Isolation split-quality primitives.

use crate::{CoreError, CoreResult};

/// Returns the smallest nonempty child size.
///
/// Lower values are preferred because they indicate that a candidate split
/// isolates at least one small group. Empty children are rejected before
/// scoring rather than silently ignored.
pub fn isolation_min_child(child_sizes: &[usize]) -> CoreResult<f64> {
    if child_sizes.is_empty() {
        return Err(CoreError::invalid_data(
            "isolation scoring requires at least one child",
        ));
    }
    if child_sizes.contains(&0) {
        return Err(CoreError::invalid_data(
            "isolation scoring does not accept empty children",
        ));
    }

    Ok(*child_sizes.iter().min().expect("nonempty checked above") as f64)
}

#[cfg(test)]
mod tests {
    use super::isolation_min_child;

    #[test]
    fn returns_smallest_child() {
        assert_eq!(isolation_min_child(&[8, 1, 4]), Ok(1.0));
    }

    #[test]
    fn rejects_empty_inputs_and_children() {
        assert!(isolation_min_child(&[]).is_err());
        assert!(isolation_min_child(&[3, 0, 2]).is_err());
    }
}
