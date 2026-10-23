//! Logical observation layouts accepted by the modeling core.

/// Logical role of a one-dimensional numeric observation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VectorRole {
    /// A row of tabular features.
    Tabular,
    /// One univariate time series.
    UnivariateSeries,
}

/// Logical layout of one observation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DataLayout {
    /// One vector interpreted as tabular features.
    Tabular,
    /// One vector interpreted as an ordered univariate series.
    UnivariateSeries,
    /// A dimension-major multivariate series with shape `dimension x time`.
    MultivariateSeries,
}
