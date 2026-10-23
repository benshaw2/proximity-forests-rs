//! Forest configuration and model implementation.

mod apply;
mod batch_predict;
mod bootstrap;
mod builder;
mod candidate;
mod candidate_generator;
mod candidate_router;
mod candidate_selector;
mod child_membership;
mod class_exemplars;
mod classification_builder;
mod classification_predict;
mod config;
mod dimension_sampler;
mod dimension_selection;
mod exemplar_store;
mod forest_builder;
mod forest_predict;
mod isolation_builder;
mod isolation_predict;
mod leaf_payload;
mod model;
mod node_membership;
mod parallel;
mod partition;
mod predict;
mod regression_builder;
mod regression_predict;
mod router;
mod scorer;
mod stopping;
mod task_selector;
mod tree;
mod weighted_sampler;
mod workspace;

pub use apply::{
    AppliedLeafId, ForestApplyResult, ForestApplyWithDepths, apply_multivariate_many,
    apply_multivariate_many_with_depths, apply_vector_many, apply_vector_many_with_depths,
};
pub use batch_predict::{
    anomaly_scores_multivariate_many, anomaly_scores_vector_many,
    predict_multivariate_classification_many, predict_multivariate_regression_many,
    predict_vector_classification_many, predict_vector_regression_many,
    score_multivariate_isolation_many, score_vector_isolation_many,
};
pub use bootstrap::BootstrapSample;
pub use builder::build_tree_iterative;
pub use candidate::{CandidateScore, CandidateSplit, ScorePreference, ScoredCandidate};
pub use candidate_generator::{CandidateIdentity, generate_candidate};
pub use candidate_router::{partition_multivariate, partition_vectors};
pub use candidate_selector::{CandidateSelector, SelectedCandidate};
pub use child_membership::{ChildMembership, build_child_memberships};
pub use class_exemplars::{
    classification_node_is_pure, classification_output_is_pure, sample_class_exemplars,
};
pub use classification_builder::{
    build_multivariate_classification_tree, build_vector_classification_tree,
};
pub use classification_predict::{
    ClassificationPrediction, predict_multivariate_classification, predict_vector_classification,
};
pub use config::{
    ClassificationConfig, ForestConfig, IsolationConfig, RegressionConfig, TiePolicy,
};
pub use dimension_sampler::sample_dimensions;
pub use dimension_selection::DimensionSelection;
pub use exemplar_store::{
    OwnedMultivariateStore, OwnedObservationStore, OwnedSeries, OwnedVectorStore,
};
pub use forest_builder::{
    fit_multivariate_classification_forest, fit_multivariate_isolation_forest,
    fit_multivariate_regression_forest, fit_vector_classification_forest,
    fit_vector_isolation_forest, fit_vector_regression_forest,
};
pub use forest_predict::{
    ForestClassificationPrediction, predict_forest_multivariate_classification,
    predict_forest_multivariate_isolation, predict_forest_multivariate_regression,
    predict_forest_vector_classification, predict_forest_vector_isolation,
    predict_forest_vector_regression,
};
pub use isolation_builder::{build_multivariate_isolation_tree, build_vector_isolation_tree};
pub use isolation_predict::{
    IsolationPrediction, expected_path_length, predict_multivariate_isolation,
    predict_vector_isolation,
};
pub use leaf_payload::{ClassificationLeaf, LeafPayload, RegressionLeaf};
pub use model::{ClassificationForest, FittedForest, IsolationForest, RegressionForest};
pub use node_membership::{NodeMembership, SampleId};
pub use parallel::{Shared, parallel_indexed};
pub use partition::{CandidatePartition, GroupedPartition};
pub use predict::{LeafVisit, traverse_multivariate, traverse_vector};
pub use regression_builder::{build_multivariate_regression_tree, build_vector_regression_tree};
pub use regression_predict::{predict_multivariate_regression, predict_vector_regression};
pub use router::{RouteResult, route_multivariate, route_vector};
pub use scorer::{score_classification, score_isolation, score_regression};
pub use stopping::{StopReason, can_split, stopping_reason};
pub use task_selector::{
    CandidateSelectionContext, select_multivariate_classification, select_multivariate_isolation,
    select_multivariate_regression, select_vector_classification, select_vector_isolation,
    select_vector_regression,
};
pub use tree::{
    ExemplarBranch, LeafNode, NodeId, NodeWork, ObservationIndex, ProximityTree, SplitNode,
    TreeBuildStack, TreeNode,
};
pub use weighted_sampler::sample_weighted_without_replacement;
pub use workspace::TreeWorkspace;
