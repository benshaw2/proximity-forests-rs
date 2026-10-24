//! Model-owned observation storage used by fitted exemplar references.

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

/// Append-only vector exemplar arena used while committing fitted splits.
///
/// The arena preserves the fitting collection's scalar kind and copies only
/// observations selected by winning candidates. Rejected candidates never
/// allocate durable exemplar storage.
#[derive(Clone, Debug, PartialEq)]
pub enum VectorExemplarBuilder {
    /// Single-precision exemplar values.
    F32(Vec<Vec<f32>>),
    /// Double-precision exemplar values.
    F64(Vec<Vec<f64>>),
}

impl VectorExemplarBuilder {
    /// Creates an empty arena for one scalar kind.
    #[must_use]
    pub const fn new(scalar_kind: ScalarKind) -> Self {
        match scalar_kind {
            ScalarKind::F32 => Self::F32(Vec::new()),
            ScalarKind::F64 => Self::F64(Vec::new()),
        }
    }

    /// Returns the number of committed exemplars.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Reports whether no exemplars have been committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies one selected observation and returns its durable exemplar ID.
    pub fn push(&mut self, observation: NumericVectorView<'_>) -> CoreResult<super::ExemplarId> {
        let id = super::ExemplarId::try_from_index(self.len()).ok_or_else(|| {
            CoreError::invalid_data("a fitted tree contains too many exemplars for u32 IDs")
        })?;
        match (self, observation) {
            (Self::F32(values), NumericVectorView::F32(observation)) => {
                values.push(observation.to_vec());
            }
            (Self::F64(values), NumericVectorView::F64(observation)) => {
                values.push(observation.to_vec());
            }
            _ => {
                return Err(CoreError::shape_mismatch(
                    "committed vector exemplars must share one scalar kind",
                ));
            }
        }
        Ok(id)
    }

    /// Finishes the arena as immutable model-owned storage.
    #[must_use]
    pub fn finish(self) -> OwnedVectorStore {
        match self {
            Self::F32(values) => OwnedVectorStore::F32(values),
            Self::F64(values) => OwnedVectorStore::F64(values),
        }
    }
}

/// Append-only multivariate exemplar arena used while committing fitted splits.
#[derive(Clone, Debug, PartialEq)]
pub enum MultivariateExemplarBuilder {
    /// Single-precision exemplar values.
    F32 {
        /// Shared channel count.
        dimension_count: usize,
        /// Committed observations.
        values: Vec<OwnedSeries<f32>>,
    },
    /// Double-precision exemplar values.
    F64 {
        /// Shared channel count.
        dimension_count: usize,
        /// Committed observations.
        values: Vec<OwnedSeries<f64>>,
    },
}

impl MultivariateExemplarBuilder {
    /// Creates an empty arena for one scalar kind and channel count.
    #[must_use]
    pub const fn new(scalar_kind: ScalarKind, dimension_count: usize) -> Self {
        match scalar_kind {
            ScalarKind::F32 => Self::F32 {
                dimension_count,
                values: Vec::new(),
            },
            ScalarKind::F64 => Self::F64 {
                dimension_count,
                values: Vec::new(),
            },
        }
    }

    /// Returns the number of committed exemplars.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32 { values, .. } => values.len(),
            Self::F64 { values, .. } => values.len(),
        }
    }

    /// Reports whether no exemplars have been committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies one selected multivariate observation into contiguous storage.
    pub fn push(
        &mut self,
        observation: MultivariateSeriesView<'_>,
    ) -> CoreResult<super::ExemplarId> {
        let id = super::ExemplarId::try_from_index(self.len()).ok_or_else(|| {
            CoreError::invalid_data("a fitted tree contains too many exemplars for u32 IDs")
        })?;
        let expected_dimensions = match self {
            Self::F32 {
                dimension_count, ..
            }
            | Self::F64 {
                dimension_count, ..
            } => *dimension_count,
        };
        validate_series(observation, observation.scalar_kind(), expected_dimensions)?;
        let value_count = observation.dimension_count() * observation.time_count();
        match self {
            Self::F32 { values, .. } if observation.scalar_kind() == ScalarKind::F32 => {
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
            Self::F64 { values, .. } if observation.scalar_kind() == ScalarKind::F64 => {
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
        Ok(id)
    }

    /// Finishes the arena as immutable model-owned storage.
    #[must_use]
    pub fn finish(self) -> OwnedMultivariateStore {
        match self {
            Self::F32 { values, .. } => OwnedMultivariateStore::F32(values),
            Self::F64 { values, .. } => OwnedMultivariateStore::F64(values),
        }
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
