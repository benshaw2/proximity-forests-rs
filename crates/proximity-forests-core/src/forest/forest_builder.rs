//! Parallel forest orchestration with model-owned observations.

use crate::{
    ClassificationTargets, CoreError, CoreResult, DistanceChoice, DistanceSamplingContext,
    ForestConfig, IsolationCriterion, ModelRng, MultivariateSeriesView, NumericVectorView,
    RandomPurpose, RegressionTargets, SampleId, SeedMixer, SupervisedCriterion,
};

use super::{
    BootstrapSample, ClassificationForest, DataRetention, FittedForest, IsolationForest,
    NodeMembership, ObservationIndex, OwnedMultivariateStore, OwnedObservationStore,
    OwnedVectorStore, RegressionForest, RetainedTrainingData, TrainingMetadata, TreeWorkspace,
    build_multivariate_classification_tree, build_multivariate_isolation_tree,
    build_multivariate_regression_tree, build_vector_classification_tree,
    build_vector_isolation_tree, build_vector_regression_tree, parallel_indexed,
};

/// Fits a vector-layout classification forest in parallel across trees.
pub fn fit_vector_classification_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[NumericVectorView<'_>],
    targets: &ClassificationTargets,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<ClassificationForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
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
                distance_sampling,
                available_dimensions,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        targets.output_count(),
    ))
}

/// Fits a multivariate classification forest in parallel across trees.
pub fn fit_multivariate_classification_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[MultivariateSeriesView<'_>],
    targets: &ClassificationTargets,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    criterion: SupervisedCriterion,
) -> CoreResult<ClassificationForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let dimensions = observations
        .first()
        .ok_or_else(|| CoreError::invalid_data("forest fitting requires at least one observation"))?
        .dimension_count();
    if observations
        .iter()
        .any(|observation| observation.dimension_count() != dimensions)
    {
        return Err(CoreError::shape_mismatch(
            "multivariate observations must have one common channel count",
        ));
    }
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
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
                distance_sampling,
                dimensions,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        targets.output_count(),
    ))
}

/// Fits a vector-layout regression forest in parallel across trees.
pub fn fit_vector_regression_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[NumericVectorView<'_>],
    targets: &RegressionTargets,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<RegressionForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
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
                distance_sampling,
                available_dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        targets.output_count(),
    ))
}

/// Fits a multivariate regression forest in parallel across trees.
pub fn fit_multivariate_regression_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[MultivariateSeriesView<'_>],
    targets: &RegressionTargets,
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    exemplar_count: usize,
    criterion: SupervisedCriterion,
) -> CoreResult<RegressionForest> {
    validate_counts(observations.len(), targets.sample_count())?;
    config.validate()?;
    let dimensions = observations
        .first()
        .ok_or_else(|| CoreError::invalid_data("forest fitting requires at least one observation"))?
        .dimension_count();
    if observations
        .iter()
        .any(|observation| observation.dimension_count() != dimensions)
    {
        return Err(CoreError::shape_mismatch(
            "multivariate observations must have one common channel count",
        ));
    }
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
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
                distance_sampling,
                dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        targets.output_count(),
    ))
}

/// Fits a vector-layout isolation forest in parallel across trees.
pub fn fit_vector_isolation_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[NumericVectorView<'_>],
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    available_dimensions: usize,
    exemplar_count: usize,
    criterion: IsolationCriterion,
) -> CoreResult<IsolationForest> {
    config.validate()?;
    let normalization_size = observations.len();
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_vector_isolation_tree(
                &config,
                tree_index,
                &views,
                membership,
                root_observations,
                distance_catalog,
                distance_sampling,
                available_dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        normalization_size,
    ))
}

/// Fits a multivariate isolation forest in parallel across trees.
pub fn fit_multivariate_isolation_forest(
    config: ForestConfig,
    thread_count: usize,
    observations: &[MultivariateSeriesView<'_>],
    distance_catalog: &[DistanceChoice],
    distance_sampling: DistanceSamplingContext,
    exemplar_count: usize,
    criterion: IsolationCriterion,
) -> CoreResult<IsolationForest> {
    config.validate()?;
    let dimensions = observations
        .first()
        .ok_or_else(|| CoreError::invalid_data("forest fitting requires at least one observation"))?
        .dimension_count();
    if observations
        .iter()
        .any(|observation| observation.dimension_count() != dimensions)
    {
        return Err(CoreError::shape_mismatch(
            "multivariate observations must have one common channel count",
        ));
    }
    let normalization_size = observations.len();
    let trees = {
        let views = observations;
        parallel_indexed(config.tree_count.get(), thread_count, |tree_index| {
            let (membership, root_observations) =
                root_membership(&config, tree_index, views.len())?;
            build_multivariate_isolation_tree(
                &config,
                tree_index,
                &views,
                membership,
                root_observations,
                distance_catalog,
                distance_sampling,
                dimensions,
                exemplar_count,
                criterion,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            training_metadata(&config, observations.len())?,
        ),
        normalization_size,
    ))
}

fn retain_vector(
    config: &ForestConfig,
    observations: &[NumericVectorView<'_>],
) -> CoreResult<RetainedTrainingData> {
    match config.data_retention {
        DataRetention::Copy => Ok(RetainedTrainingData::Owned(OwnedObservationStore::Vector(
            OwnedVectorStore::from_views(observations)?,
        ))),
        DataRetention::Exemplars => Ok(RetainedTrainingData::None),
        DataRetention::Reference => Ok(RetainedTrainingData::Reference(
            config.external_training_data.clone().ok_or_else(|| {
                CoreError::invalid_configuration(
                    "external_training_data",
                    "reference retention requires an external training-data descriptor",
                )
            })?,
        )),
    }
}

fn retain_multivariate(
    config: &ForestConfig,
    observations: &[MultivariateSeriesView<'_>],
) -> CoreResult<RetainedTrainingData> {
    match config.data_retention {
        DataRetention::Copy => Ok(RetainedTrainingData::Owned(
            OwnedObservationStore::Multivariate(OwnedMultivariateStore::from_views(observations)?),
        )),
        DataRetention::Exemplars => Ok(RetainedTrainingData::None),
        DataRetention::Reference => Ok(RetainedTrainingData::Reference(
            config.external_training_data.clone().ok_or_else(|| {
                CoreError::invalid_configuration(
                    "external_training_data",
                    "reference retention requires an external training-data descriptor",
                )
            })?,
        )),
    }
}

fn training_metadata(config: &ForestConfig, sample_count: usize) -> CoreResult<TrainingMetadata> {
    let samples = (0..config.tree_count.get())
        .map(|tree_index| bootstrap_sample(config, tree_index, sample_count))
        .collect::<CoreResult<Vec<_>>>()?;
    Ok(TrainingMetadata::new(sample_count, samples))
}

fn bootstrap_sample(
    config: &ForestConfig,
    tree_index: usize,
    sample_count: usize,
) -> CoreResult<BootstrapSample> {
    if config.bootstrap {
        let tree_seed = SeedMixer::tree(config.seed, tree_index);
        let seed = SeedMixer::node_purpose(tree_seed, 0, RandomPurpose::Bootstrap);
        BootstrapSample::draw(sample_count, &mut ModelRng::from_seed(seed))
    } else {
        Ok(BootstrapSample::identity(sample_count))
    }
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
