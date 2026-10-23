//! Parallel forest orchestration with model-owned observations.

use crate::{
    ClassificationTargets, CoreError, CoreResult, DistanceConfig, ForestConfig, IsolationCriterion,
    ModelRng, MultivariateSeriesView, NumericVectorView, RandomPurpose, RegressionTargets,
    SampleId, SeedMixer, SupervisedCriterion,
};

use super::{
    ClassificationForest, FittedForest, IsolationForest, NodeMembership, ObservationIndex,
    OwnedMultivariateStore, OwnedObservationStore, OwnedVectorStore, RegressionForest,
    TreeWorkspace, build_multivariate_classification_tree, build_multivariate_isolation_tree,
    build_multivariate_regression_tree, build_vector_classification_tree,
    build_vector_isolation_tree, build_vector_regression_tree, parallel_indexed,
};

/// Fits a vector-layout classification forest in parallel across trees.
pub fn fit_vector_classification_forest(
    config: ForestConfig,
    observations: &[NumericVectorView<'_>],
    targets: &ClassificationTargets,
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<ClassificationForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let store = OwnedVectorStore::from_views(observations)?;
    let trees = {
        let views = store.views();
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_vector_classification_tree(
                &config,
                tree_index,
                &views,
                targets,
                membership,
                root_observations,
                distance_catalog,
                available_dimensions,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Vector(store)),
        targets.output_count(),
    ))
}

/// Fits a multivariate classification forest in parallel across trees.
pub fn fit_multivariate_classification_forest(
    config: ForestConfig,
    observations: &[MultivariateSeriesView<'_>],
    targets: &ClassificationTargets,
    distance_catalog: &[DistanceConfig],
    criterion: SupervisedCriterion,
) -> CoreResult<ClassificationForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let store = OwnedMultivariateStore::from_views(observations)?;
    let dimensions = store.dimension_count();
    let trees = {
        let views = store.views()?;
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_multivariate_classification_tree(
                &config,
                tree_index,
                &views,
                targets,
                membership,
                root_observations,
                distance_catalog,
                dimensions,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Multivariate(store)),
        targets.output_count(),
    ))
}

/// Fits a vector-layout regression forest in parallel across trees.
pub fn fit_vector_regression_forest(
    config: ForestConfig,
    observations: &[NumericVectorView<'_>],
    targets: &RegressionTargets,
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<RegressionForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let store = OwnedVectorStore::from_views(observations)?;
    let trees = {
        let views = store.views();
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_vector_regression_tree(
                &config,
                tree_index,
                &views,
                targets,
                membership,
                root_observations,
                distance_catalog,
                available_dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Vector(store)),
        targets.output_count(),
    ))
}

/// Fits a multivariate regression forest in parallel across trees.
pub fn fit_multivariate_regression_forest(
    config: ForestConfig,
    observations: &[MultivariateSeriesView<'_>],
    targets: &RegressionTargets,
    distance_catalog: &[DistanceConfig],
    exemplar_count: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<RegressionForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let store = OwnedMultivariateStore::from_views(observations)?;
    let dimensions = store.dimension_count();
    let trees = {
        let views = store.views()?;
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_multivariate_regression_tree(
                &config,
                tree_index,
                &views,
                targets,
                membership,
                root_observations,
                distance_catalog,
                dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Multivariate(store)),
        targets.output_count(),
    ))
}

/// Fits a vector-layout isolation forest in parallel across trees.
pub fn fit_vector_isolation_forest(
    config: ForestConfig,
    observations: &[NumericVectorView<'_>],
    distance_catalog: &[DistanceConfig],
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
) -> CoreResult<IsolationForest> {
    config.validate()?;
    let store = OwnedVectorStore::from_views(observations)?;
    let normalization_size = observations.len();
    let trees = {
        let views = store.views();
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_vector_isolation_tree(
                &config,
                tree_index,
                &views,
                membership,
                root_observations,
                distance_catalog,
                available_dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Vector(store)),
        normalization_size,
    ))
}

/// Fits a multivariate isolation forest in parallel across trees.
pub fn fit_multivariate_isolation_forest(
    config: ForestConfig,
    observations: &[MultivariateSeriesView<'_>],
    distance_catalog: &[DistanceConfig],
    exemplar_count: usize,
    criterion: IsolationCriterion,
) -> CoreResult<IsolationForest> {
    config.validate()?;
    let store = OwnedMultivariateStore::from_views(observations)?;
    let dimensions = store.dimension_count();
    let normalization_size = observations.len();
    let trees = {
        let views = store.views()?;
        parallel_indexed(config.tree_count.get(), 0, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_multivariate_isolation_tree(
                &config,
                tree_index,
                &views,
                membership,
                root_observations,
                distance_catalog,
                dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(config, trees, OwnedObservationStore::Multivariate(store)),
        normalization_size,
    ))
}

fn root_membership(
    config: &ForestConfig,
    tree_index: usize,
    sample_count: usize,
) -> CoreResult<(NodeMembership, Vec<ObservationIndex>)> {
    if sample_count == 0 {
        return Err(CoreError::invalid_data(
            "forest fitting requires at least one observation",
        ));
    }
    let mut occurrence_ids = Vec::with_capacity(sample_count);
    if config.bootstrap {
        let tree_seed = SeedMixer::tree(config.seed, tree_index);
        let seed = SeedMixer::node_purpose(tree_seed, 0, RandomPurpose::Bootstrap);
        let mut rng = ModelRng::from_seed(seed);
        for _ in 0..sample_count {
            occurrence_ids.push(Some(SampleId::new(rng.index(sample_count)? as u64)));
        }
    } else {
        occurrence_ids.extend((0..sample_count).map(|sample| Some(SampleId::new(sample as u64))));
    }
    let membership = NodeMembership::from_occurrences(&occurrence_ids)?;
    let observations = membership
        .sample_ids()
        .iter()
        .map(|sample_id| {
            sample_id
                .map(|id| ObservationIndex::new(id.get() as usize))
                .ok_or_else(|| CoreError::invalid_data("root sample identity was unavailable"))
        })
        .collect::<CoreResult<Vec<_>>>()?;
    Ok((membership, observations))
}

fn validate_counts(observations: usize, targets: usize) -> CoreResult<()> {
    if observations == targets {
        Ok(())
    } else {
        Err(CoreError::shape_mismatch(format!(
            "received {observations} observations and {targets} target rows"
        )))
    }
}
