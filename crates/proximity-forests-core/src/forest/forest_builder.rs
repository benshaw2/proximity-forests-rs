//! Parallel forest orchestration with model-owned observations.

use crate::{
    ClassificationTargets, CoreError, CoreResult, DimensionSelection, DimensionSelectionSupport,
    DistanceChoice, DistanceSamplingContext, ForestConfig, IsolationCriterion, ModelRng,
    MultivariateSeriesView, NumericVectorView, ObservationSampling, RandomPurpose,
    RegressionTargets, SampleId, SeedMixer, SupervisedCriterion,
};

use super::{
    ClassificationForest, DataRetention, ExecutionContext, FittedForest, IsolationForest,
    NodeMembership, ObservationIndex, OwnedMultivariateStore, OwnedObservationStore,
    OwnedVectorStore, RegressionForest, RetainedTrainingData, TrainingMetadata, TreeSample,
    TreeWorkspace, build_multivariate_classification_tree, build_multivariate_isolation_tree,
    build_multivariate_regression_tree, build_vector_classification_tree,
    build_vector_isolation_tree, build_vector_regression_tree, parallel_indexed_in,
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::VectorFeatures,
    )?;
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::MultivariateChannels,
    )?;
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
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(ClassificationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::VectorFeatures,
    )?;
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::MultivariateChannels,
    )?;
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
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(RegressionForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::VectorFeatures,
    )?;
    let normalization_size = config
        .observation_sampling
        .resolved_size(observations.len())?;
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_vector(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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
    validate_dimension_selection_compatibility(
        config.dimension_selection,
        distance_catalog,
        DimensionSelectionSupport::MultivariateChannels,
    )?;
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
    let normalization_size = config
        .observation_sampling
        .resolved_size(observations.len())?;
    let tree_samples = build_tree_samples(&config, observations.len())?;
    let context = ExecutionContext::new(thread_count)?;
    let trees = {
        let views = observations;
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            let (membership, root_observations) = root_membership(&tree_samples[tree_index])?;
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
                &context,
                &mut TreeWorkspace::new(),
            )
        })?
    };
    Ok(IsolationForest::new(
        FittedForest::new(
            config.clone(),
            trees,
            retain_multivariate(&config, observations)?,
            TrainingMetadata::new(observations.len(), tree_samples),
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

fn build_tree_samples(config: &ForestConfig, population: usize) -> CoreResult<Vec<TreeSample>> {
    (0..config.tree_count.get())
        .map(|tree_index| tree_sample(config, tree_index, population))
        .collect()
}

fn tree_sample(
    config: &ForestConfig,
    tree_index: usize,
    population: usize,
) -> CoreResult<TreeSample> {
    let size = config.observation_sampling.resolved_size(population)?;
    match config.observation_sampling {
        ObservationSampling::Full => TreeSample::identity(population),
        ObservationSampling::Bootstrap(_) => {
            let mut rng = sampling_rng(config, tree_index);
            TreeSample::bootstrap(population, size, &mut rng)
        }
        ObservationSampling::Subsample(_) => {
            let mut rng = sampling_rng(config, tree_index);
            TreeSample::subsample(population, size, &mut rng)
        }
    }
}

fn sampling_rng(config: &ForestConfig, tree_index: usize) -> ModelRng {
    let tree_seed = SeedMixer::tree(config.seed, tree_index);
    let seed = SeedMixer::node_purpose(tree_seed, 0, RandomPurpose::Bootstrap);
    ModelRng::from_seed(seed)
}

fn root_membership(sample: &TreeSample) -> CoreResult<(NodeMembership, Vec<ObservationIndex>)> {
    let occurrence_ids = sample
        .sampled_positions()
        .iter()
        .map(|&position| Some(SampleId::new(position as u64)))
        .collect::<Vec<_>>();
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

fn validate_dimension_selection_compatibility(
    selection: DimensionSelection,
    catalog: &[DistanceChoice],
    required: DimensionSelectionSupport,
) -> CoreResult<()> {
    if selection == DimensionSelection::All {
        return Ok(());
    }
    for choice in catalog {
        let support = choice.dimension_selection_support();
        if support != required {
            let layout = match required {
                DimensionSelectionSupport::VectorFeatures => "vector-layout features",
                DimensionSelectionSupport::MultivariateChannels => "multivariate channels",
                DimensionSelectionSupport::None => "dimensions",
            };
            return Err(CoreError::invalid_configuration(
                "dimension_selection",
                format!(
                    "distance choice {choice:?} does not support selecting {layout}; \
                     use all dimensions or choose a compatible distance"
                ),
            ));
        }
    }
    Ok(())
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

#[cfg(test)]
mod sampling_tests {
    use super::{build_tree_samples, tree_sample};
    use crate::{ForestConfig, ObservationSampling, SampleSize};
    use std::num::NonZeroUsize;

    use super::super::{ExecutionContext, TreeSample, parallel_indexed_in};

    const POPULATION: usize = 48;
    const TREES: usize = 12;

    fn config(policy: ObservationSampling) -> ForestConfig {
        let mut config = ForestConfig::new(
            NonZeroUsize::new(TREES).expect("positive tree count"),
            NonZeroUsize::new(3).expect("positive candidate count"),
            0x5eed_cafe,
        );
        config.observation_sampling = policy;
        config
    }

    fn parallel_samples(config: &ForestConfig, workers: usize) -> Vec<TreeSample> {
        let context = ExecutionContext::new(workers).expect("valid execution context");
        parallel_indexed_in(&context, config.tree_count.get(), &|tree_index| {
            tree_sample(config, tree_index, POPULATION)
        })
        .expect("parallel sample generation")
    }

    #[test]
    fn each_stochastic_tree_uses_its_tree_index_seed() {
        for policy in [
            ObservationSampling::Bootstrap(SampleSize::Fixed(NonZeroUsize::new(17).unwrap())),
            ObservationSampling::Subsample(SampleSize::Fixed(NonZeroUsize::new(17).unwrap())),
        ] {
            let samples = build_tree_samples(&config(policy), POPULATION).unwrap();
            assert_eq!(samples.len(), TREES);
            assert!(
                samples.windows(2).any(|pair| pair[0] != pair[1]),
                "stochastic policies must not reuse one common sample for every tree"
            );
        }
    }

    #[test]
    fn repeated_fits_reproduce_all_tree_samples() {
        for policy in [
            ObservationSampling::Full,
            ObservationSampling::Bootstrap(SampleSize::All),
            ObservationSampling::Bootstrap(SampleSize::Proportion(0.5)),
            ObservationSampling::Subsample(SampleSize::Fixed(NonZeroUsize::new(17).unwrap())),
            ObservationSampling::Subsample(SampleSize::Proportion(0.4)),
        ] {
            let config = config(policy);
            let first = build_tree_samples(&config, POPULATION).unwrap();
            let second = build_tree_samples(&config, POPULATION).unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn sequential_and_parallel_sample_generation_match() {
        for policy in [
            ObservationSampling::Full,
            ObservationSampling::Bootstrap(SampleSize::All),
            ObservationSampling::Bootstrap(SampleSize::Fixed(NonZeroUsize::new(23).unwrap())),
            ObservationSampling::Subsample(SampleSize::Fixed(NonZeroUsize::new(17).unwrap())),
            ObservationSampling::Subsample(SampleSize::Proportion(0.4)),
        ] {
            let config = config(policy);
            let sequential = build_tree_samples(&config, POPULATION).unwrap();
            let one_worker = parallel_samples(&config, 1);
            let four_workers = parallel_samples(&config, 4);
            assert_eq!(sequential, one_worker);
            assert_eq!(sequential, four_workers);
        }
    }

    #[test]
    fn policy_invariants_hold_for_every_tree() {
        let bootstrap = build_tree_samples(
            &config(ObservationSampling::Bootstrap(SampleSize::Fixed(
                NonZeroUsize::new(61).unwrap(),
            ))),
            POPULATION,
        )
        .unwrap();
        for sample in bootstrap {
            assert_eq!(sample.sampled_positions().len(), 61);
            assert_eq!(sample.multiplicities().len(), POPULATION);
            assert_eq!(sample.multiplicities().iter().sum::<usize>(), 61);
        }

        let subsample = build_tree_samples(
            &config(ObservationSampling::Subsample(SampleSize::Fixed(
                NonZeroUsize::new(17).unwrap(),
            ))),
            POPULATION,
        )
        .unwrap();
        for sample in subsample {
            assert_eq!(sample.sampled_positions().len(), 17);
            assert_eq!(sample.out_of_bag_positions().len(), POPULATION - 17);
            assert!(sample.multiplicities().iter().all(|&count| count <= 1));
        }
    }

    #[test]
    fn full_policy_is_the_only_intentionally_identical_per_tree_sample() {
        let samples = build_tree_samples(&config(ObservationSampling::Full), POPULATION).unwrap();
        assert!(samples.windows(2).all(|pair| pair[0] == pair[1]));
        for sample in samples {
            assert_eq!(sample.sampled_positions().len(), POPULATION);
            assert!(sample.out_of_bag_positions().is_empty());
        }
    }
}

#[cfg(test)]
mod dimension_selection_compatibility_tests {
    use super::validate_dimension_selection_compatibility;
    use crate::{
        DimensionSelection, DimensionSelectionSupport, DistanceChoice, DistanceConfig,
        SampledDistanceFamily,
    };

    #[test]
    fn all_dimensions_preserves_every_existing_catalog() {
        let catalog = [DistanceChoice::Sampled(SampledDistanceFamily::Dtw)];
        assert!(
            validate_dimension_selection_compatibility(
                DimensionSelection::All,
                &catalog,
                DimensionSelectionSupport::VectorFeatures,
            )
            .is_ok()
        );
    }

    #[test]
    fn vector_temporal_fixed_and_sampled_choices_are_rejected() {
        for catalog in [
            vec![DistanceChoice::Fixed(DistanceConfig::Dtw(
                Default::default(),
            ))],
            vec![DistanceChoice::Sampled(SampledDistanceFamily::DtwCv)],
        ] {
            let error = validate_dimension_selection_compatibility(
                DimensionSelection::Sqrt,
                &catalog,
                DimensionSelectionSupport::VectorFeatures,
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("does not support selecting vector-layout features")
            );
        }
    }

    #[test]
    fn vector_tabular_choices_allow_feature_subsets() {
        let catalog = [
            DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean),
            DistanceChoice::Fixed(DistanceConfig::Manhattan),
        ];
        assert!(
            validate_dimension_selection_compatibility(
                DimensionSelection::FixedCount(2),
                &catalog,
                DimensionSelectionSupport::VectorFeatures,
            )
            .is_ok()
        );
    }

    #[test]
    fn multivariate_independent_and_dependent_choices_allow_channel_subsets() {
        let catalog = [
            DistanceChoice::Sampled(SampledDistanceFamily::DtwIndependent),
            DistanceChoice::Sampled(SampledDistanceFamily::DtwCvDependent),
        ];
        assert!(
            validate_dimension_selection_compatibility(
                DimensionSelection::Proportion(0.5),
                &catalog,
                DimensionSelectionSupport::MultivariateChannels,
            )
            .is_ok()
        );
    }

    #[test]
    fn wrong_layout_semantics_are_rejected_even_if_selection_is_supported_elsewhere() {
        let vector_catalog = [DistanceChoice::Fixed(DistanceConfig::SquaredEuclidean)];
        assert!(
            validate_dimension_selection_compatibility(
                DimensionSelection::Sqrt,
                &vector_catalog,
                DimensionSelectionSupport::MultivariateChannels,
            )
            .is_err()
        );
    }
}
