//! Centralized task-independent tree stopping rules.

use crate::ForestConfig;

use super::NodeMembership;

/// Reason an iterative builder should emit a leaf without evaluating splits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StopReason {
    /// The configured maximum depth has been reached.
    MaximumDepth,
    /// Too few bootstrap occurrences reach the node.
    MinimumSplitSize,
    /// At most one distinct observation reaches the node.
    InsufficientDistinctObservations,
}

/// Evaluates task-independent stopping rules in deterministic priority order.
#[must_use]
pub fn stopping_reason(
    config: &ForestConfig,
    depth: usize,
    membership: &NodeMembership,
) -> Option<StopReason> {
    if config.maximum_depth.is_some_and(|maximum| depth >= maximum) {
        return Some(StopReason::MaximumDepth);
    }
    if membership.occurrence_count() < config.minimum_split_size.get() {
        return Some(StopReason::MinimumSplitSize);
    }
    if membership.distinct_count() <= 1 {
        return Some(StopReason::InsufficientDistinctObservations);
    }
    None
}

/// Reports whether task-independent rules allow candidate evaluation.
#[must_use]
pub fn can_split(config: &ForestConfig, depth: usize, membership: &NodeMembership) -> bool {
    stopping_reason(config, depth, membership).is_none()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{StopReason, stopping_reason};
    use crate::{ForestConfig, NodeMembership, SampleId};

    #[test]
    fn maximum_depth_has_priority() {
        let mut config = ForestConfig::new(NonZeroUsize::MIN, NonZeroUsize::MIN, 0);
        config.maximum_depth = Some(2);
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1)), Some(SampleId::new(2))])
                .expect("membership");
        assert_eq!(
            stopping_reason(&config, 2, &membership),
            Some(StopReason::MaximumDepth)
        );
    }

    #[test]
    fn one_distinct_observation_stops() {
        let config = ForestConfig::new(NonZeroUsize::MIN, NonZeroUsize::MIN, 0);
        let membership =
            NodeMembership::from_occurrences(&[Some(SampleId::new(1)), Some(SampleId::new(1))])
                .expect("membership");
        assert_eq!(
            stopping_reason(&config, 0, &membership),
            Some(StopReason::InsufficientDistinctObservations)
        );
    }
}
