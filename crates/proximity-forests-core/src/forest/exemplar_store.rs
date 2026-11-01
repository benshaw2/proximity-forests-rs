//! Model-owned observation storage used by fitted exemplar references.

use std::collections::BTreeMap;

use crate::{CoreError, CoreResult, MultivariateSeriesView, NumericVectorView, ScalarKind};

/// Owned vector observations preserving their original floating-point dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedVectorStore {
    /// Single-precision observations, including variable lengths.
    F32(Vec<Vec<f32>>),
    /// Double-precision observations, including variable lengths.
    F64(Vec<Vec<f64>>),
}

impl OwnedVectorStore {
    /// Copies borrowed observations into model-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an error for empty input or mixed dtypes.
    pub fn from_views(observations: &[NumericVectorView<'_>]) -> CoreResult<Self> {
        let first = observations.first().ok_or_else(|| {
            CoreError::invalid_data("an owned vector store requires at least one observation")
        })?;
        match first.scalar_kind() {
            ScalarKind::F32 => observations
                .iter()
                .map(|observation| match observation {
                    NumericVectorView::F32(values) => Ok(values.to_vec()),
                    NumericVectorView::F64(_) => Err(CoreError::shape_mismatch(
                        "an owned vector store requires one common dtype",
                    )),
                })
                .collect::<CoreResult<Vec<_>>>()
                .map(Self::F32),
            ScalarKind::F64 => observations
                .iter()
                .map(|observation| match observation {
                    NumericVectorView::F64(values) => Ok(values.to_vec()),
                    NumericVectorView::F32(_) => Err(CoreError::shape_mismatch(
                        "an owned vector store requires one common dtype",
                    )),
                })
                .collect::<CoreResult<Vec<_>>>()
                .map(Self::F64),
        }
    }

    /// Returns the number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Reports whether no observations are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns one borrowed observation.
    #[must_use]
    pub fn view(&self, index: usize) -> Option<NumericVectorView<'_>> {
        match self {
            Self::F32(values) => values
                .get(index)
                .map(|values| NumericVectorView::F32(values)),
            Self::F64(values) => values
                .get(index)
                .map(|values| NumericVectorView::F64(values)),
        }
    }

    /// Returns borrowed views for all observations.
    #[must_use]
    pub fn views(&self) -> Vec<NumericVectorView<'_>> {
        (0..self.len())
            .filter_map(|index| self.view(index))
            .collect()
    }
}

/// One owned rectangular multivariate observation.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSeries<T> {
    values: Vec<T>,
    dimension_count: usize,
    time_count: usize,
}

impl<T> OwnedSeries<T> {
    /// Reconstructs one rectangular series from validated persisted components.
    pub(crate) fn from_parts(
        values: Vec<T>,
        dimension_count: usize,
        time_count: usize,
    ) -> CoreResult<Self> {
        if dimension_count == 0 || time_count == 0 {
            return Err(CoreError::invalid_data(
                "an owned series requires positive dimension and time counts",
            ));
        }
        let expected = dimension_count
            .checked_mul(time_count)
            .ok_or_else(|| CoreError::invalid_data("owned series shape overflows usize"))?;
        if values.len() != expected {
            return Err(CoreError::shape_mismatch(
                "owned series values do not match dimension and time counts",
            ));
        }
        Ok(Self {
            values,
            dimension_count,
            time_count,
        })
    }

    /// Returns the contiguous dimension-major values.
    pub(crate) fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the number of dimensions.
    pub(crate) const fn dimension_count(&self) -> usize {
        self.dimension_count
    }

    /// Returns the number of time points per dimension.
    pub(crate) const fn time_count(&self) -> usize {
        self.time_count
    }
}

/// Owned multivariate observations in dimension-major contiguous layout.
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedMultivariateStore {
    /// Single-precision observations.
    F32(Vec<OwnedSeries<f32>>),
    /// Double-precision observations.
    F64(Vec<OwnedSeries<f64>>),
}

impl OwnedMultivariateStore {
    /// Copies logical series values into model-owned contiguous storage.
    ///
    /// Strided input is compacted once during fitting. Observations may have
    /// different time lengths but must share dtype and dimension count.
    pub fn from_views(observations: &[MultivariateSeriesView<'_>]) -> CoreResult<Self> {
        let first = observations.first().ok_or_else(|| {
            CoreError::invalid_data("an owned series store requires at least one observation")
        })?;
        let dimensions = first.dimension_count();
        match first.scalar_kind() {
            ScalarKind::F32 => {
                let mut owned = Vec::with_capacity(observations.len());
                for &observation in observations {
                    validate_series(observation, ScalarKind::F32, dimensions)?;
                    let mut values = Vec::with_capacity(
                        observation.dimension_count() * observation.time_count(),
                    );
                    for dimension in 0..dimensions {
                        for time in 0..observation.time_count() {
                            values.push(observation.get_f64(dimension, time).ok_or_else(|| {
                                CoreError::invalid_data("validated series value was unavailable")
                            })? as f32);
                        }
                    }
                    owned.push(OwnedSeries {
                        values,
                        dimension_count: dimensions,
                        time_count: observation.time_count(),
                    });
                }
                Ok(Self::F32(owned))
            }
            ScalarKind::F64 => {
                let mut owned = Vec::with_capacity(observations.len());
                for &observation in observations {
                    validate_series(observation, ScalarKind::F64, dimensions)?;
                    let mut values = Vec::with_capacity(
                        observation.dimension_count() * observation.time_count(),
                    );
                    for dimension in 0..dimensions {
                        for time in 0..observation.time_count() {
                            values.push(observation.get_f64(dimension, time).ok_or_else(|| {
                                CoreError::invalid_data("validated series value was unavailable")
                            })?);
                        }
                    }
                    owned.push(OwnedSeries {
                        values,
                        dimension_count: dimensions,
                        time_count: observation.time_count(),
                    });
                }
                Ok(Self::F64(owned))
            }
        }
    }

    /// Returns the number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Reports whether no observations are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the common dimension count.
    #[must_use]
    pub fn dimension_count(&self) -> usize {
        match self {
            Self::F32(values) => values.first().map_or(0, |series| series.dimension_count),
            Self::F64(values) => values.first().map_or(0, |series| series.dimension_count),
        }
    }

    /// Returns one borrowed series view.
    pub fn view(&self, index: usize) -> CoreResult<Option<MultivariateSeriesView<'_>>> {
        match self {
            Self::F32(values) => values
                .get(index)
                .map(|series| {
                    MultivariateSeriesView::contiguous_f32(
                        &series.values,
                        series.dimension_count,
                        series.time_count,
                    )
                })
                .transpose(),
            Self::F64(values) => values
                .get(index)
                .map(|series| {
                    MultivariateSeriesView::contiguous_f64(
                        &series.values,
                        series.dimension_count,
                        series.time_count,
                    )
                })
                .transpose(),
        }
    }

    /// Returns borrowed views for all observations.
    pub fn views(&self) -> CoreResult<Vec<MultivariateSeriesView<'_>>> {
        (0..self.len())
            .map(|index| {
                self.view(index)?
                    .ok_or_else(|| CoreError::invalid_data("owned series index was unavailable"))
            })
            .collect()
    }
}

/// One completed tree-local exemplar arena plus original training identities.
#[derive(Clone, Debug, PartialEq)]
pub struct FinishedExemplars {
    store: OwnedObservationStore,
    training_indices: Vec<usize>,
}
impl FinishedExemplars {
    /// Creates a checked completed arena.
    pub fn new(store: OwnedObservationStore, training_indices: Vec<usize>) -> CoreResult<Self> {
        let value_count = match &store {
            OwnedObservationStore::Vector(values) => values.len(),
            OwnedObservationStore::Multivariate(values) => values.len(),
        };
        if value_count != training_indices.len() {
            return Err(CoreError::shape_mismatch(
                "exemplar values and training-index provenance must have equal lengths",
            ));
        }
        Ok(Self {
            store,
            training_indices,
        })
    }
    /// Returns model-owned exemplar values.
    #[must_use]
    pub const fn store(&self) -> &OwnedObservationStore {
        &self.store
    }
    /// Returns original training indices in exemplar-ID order.
    #[must_use]
    pub fn training_indices(&self) -> &[usize] {
        &self.training_indices
    }
    /// Consumes the completed arena.
    #[must_use]
    pub fn into_parts(self) -> (OwnedObservationStore, Vec<usize>) {
        (self.store, self.training_indices)
    }
}

/// Identity-interning vector exemplar arena used while committing fitted splits.
///
/// Repeated winning uses of the same training observation reuse one durable ID.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorExemplarBuilder {
    values: OwnedVectorStore,
    training_indices: BTreeMap<usize, super::ExemplarId>,
}
impl VectorExemplarBuilder {
    /// Creates an empty arena for one scalar kind.
    #[must_use]
    pub fn new(scalar_kind: ScalarKind) -> Self {
        Self {
            values: match scalar_kind {
                ScalarKind::F32 => OwnedVectorStore::F32(Vec::new()),
                ScalarKind::F64 => OwnedVectorStore::F64(Vec::new()),
            },
            training_indices: BTreeMap::new(),
        }
    }
    /// Returns the number of distinct committed exemplars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.training_indices.len()
    }
    /// Reports whether no exemplars have been committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.training_indices.is_empty()
    }
    /// Interns one selected training observation and returns its durable ID.
    pub fn intern(
        &mut self,
        training_index: usize,
        observation: NumericVectorView<'_>,
    ) -> CoreResult<super::ExemplarId> {
        if let Some(&id) = self.training_indices.get(&training_index) {
            return Ok(id);
        }
        let id =
            super::ExemplarId::try_from_index(self.training_indices.len()).ok_or_else(|| {
                CoreError::invalid_data("a fitted tree contains too many exemplars for u32 IDs")
            })?;
        match (&mut self.values, observation) {
            (OwnedVectorStore::F32(values), NumericVectorView::F32(row)) => {
                values.push(row.to_vec())
            }
            (OwnedVectorStore::F64(values), NumericVectorView::F64(row)) => {
                values.push(row.to_vec())
            }
            _ => {
                return Err(CoreError::shape_mismatch(
                    "committed vector exemplars must share one scalar kind",
                ));
            }
        }
        self.training_indices.insert(training_index, id);
        Ok(id)
    }
    /// Finishes the arena while preserving original training identities.
    pub fn finish(self) -> CoreResult<FinishedExemplars> {
        let mut indexed = self.training_indices.into_iter().collect::<Vec<_>>();
        indexed.sort_by_key(|(_, id)| id.index());
        FinishedExemplars::new(
            OwnedObservationStore::Vector(self.values),
            indexed.into_iter().map(|(index, _)| index).collect(),
        )
    }
}
/// Identity-interning multivariate exemplar arena used while committing fitted splits.
#[derive(Clone, Debug, PartialEq)]
pub struct MultivariateExemplarBuilder {
    values: OwnedMultivariateStore,
    dimension_count: usize,
    training_indices: BTreeMap<usize, super::ExemplarId>,
}
impl MultivariateExemplarBuilder {
    /// Creates an empty arena for one scalar kind and channel count.
    #[must_use]
    pub fn new(scalar_kind: ScalarKind, dimension_count: usize) -> Self {
        Self {
            values: match scalar_kind {
                ScalarKind::F32 => OwnedMultivariateStore::F32(Vec::new()),
                ScalarKind::F64 => OwnedMultivariateStore::F64(Vec::new()),
            },
            dimension_count,
            training_indices: BTreeMap::new(),
        }
    }
    /// Returns the number of distinct committed exemplars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.training_indices.len()
    }
    /// Reports whether no exemplars have been committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.training_indices.is_empty()
    }
    /// Interns one selected training observation and returns its durable ID.
    pub fn intern(
        &mut self,
        training_index: usize,
        observation: MultivariateSeriesView<'_>,
    ) -> CoreResult<super::ExemplarId> {
        if let Some(&id) = self.training_indices.get(&training_index) {
            return Ok(id);
        }
        validate_series(observation, observation.scalar_kind(), self.dimension_count)?;
        let id =
            super::ExemplarId::try_from_index(self.training_indices.len()).ok_or_else(|| {
                CoreError::invalid_data("a fitted tree contains too many exemplars for u32 IDs")
            })?;
        let value_count = observation.dimension_count() * observation.time_count();
        match &mut self.values {
            OwnedMultivariateStore::F32(values) if observation.scalar_kind() == ScalarKind::F32 => {
                let mut owned = Vec::with_capacity(value_count);
                for dimension in 0..observation.dimension_count() {
                    for time in 0..observation.time_count() {
                        owned.push(observation.get_f64(dimension, time).ok_or_else(|| {
                            CoreError::invalid_data("validated series value was unavailable")
                        })? as f32);
                    }
                }
                values.push(OwnedSeries {
                    values: owned,
                    dimension_count: observation.dimension_count(),
                    time_count: observation.time_count(),
                });
            }
            OwnedMultivariateStore::F64(values) if observation.scalar_kind() == ScalarKind::F64 => {
                let mut owned = Vec::with_capacity(value_count);
                for dimension in 0..observation.dimension_count() {
                    for time in 0..observation.time_count() {
                        owned.push(observation.get_f64(dimension, time).ok_or_else(|| {
                            CoreError::invalid_data("validated series value was unavailable")
                        })?);
                    }
                }
                values.push(OwnedSeries {
                    values: owned,
                    dimension_count: observation.dimension_count(),
                    time_count: observation.time_count(),
                });
            }
            _ => {
                return Err(CoreError::shape_mismatch(
                    "committed multivariate exemplars must share one scalar kind",
                ));
            }
        }
        self.training_indices.insert(training_index, id);
        Ok(id)
    }
    /// Finishes the arena while preserving original training identities.
    pub fn finish(self) -> CoreResult<FinishedExemplars> {
        let mut indexed = self.training_indices.into_iter().collect::<Vec<_>>();
        indexed.sort_by_key(|(_, id)| id.index());
        FinishedExemplars::new(
            OwnedObservationStore::Multivariate(self.values),
            indexed.into_iter().map(|(index, _)| index).collect(),
        )
    }
}
/// Task-independent model-owned observation repository.
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedObservationStore {
    /// Vector, tabular-row, or univariate-series observations.
    Vector(OwnedVectorStore),
    /// Multivariate time-series observations.
    Multivariate(OwnedMultivariateStore),
}

fn validate_series(
    observation: MultivariateSeriesView<'_>,
    scalar_kind: ScalarKind,
    dimensions: usize,
) -> CoreResult<()> {
    if observation.scalar_kind() != scalar_kind || observation.dimension_count() != dimensions {
        return Err(CoreError::shape_mismatch(
            "owned series require one common dtype and dimension count",
        ));
    }
    Ok(())
}
