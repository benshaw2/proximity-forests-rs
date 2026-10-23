//! Borrowed numeric observation views and logical data layouts.

mod layout;
mod numeric;
mod series;
mod time_axis;

pub use layout::{DataLayout, VectorRole};
pub use numeric::NumericVectorView;
pub use series::MultivariateSeriesView;
pub use time_axis::TimeAxisView;
