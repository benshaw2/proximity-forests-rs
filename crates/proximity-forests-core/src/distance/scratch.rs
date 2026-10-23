//! Reusable worker-local scratch storage for distance operations.

/// Reusable buffers owned by one worker.
#[derive(Clone, Debug, Default)]
pub struct DistanceScratch {
    row_a: Vec<f64>,
    row_b: Vec<f64>,
    integer_row_a: Vec<usize>,
    integer_row_b: Vec<usize>,
    comparable_row_a: Vec<bool>,
    comparable_row_b: Vec<bool>,
    transform_a: Vec<f64>,
    transform_b: Vec<f64>,
    weights: Vec<f64>,
}

impl DistanceScratch {
    /// Creates empty scratch storage.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            row_a: Vec::new(),
            row_b: Vec::new(),
            integer_row_a: Vec::new(),
            integer_row_b: Vec::new(),
            comparable_row_a: Vec::new(),
            comparable_row_b: Vec::new(),
            transform_a: Vec::new(),
            transform_b: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Resizes and initializes both floating-point rolling rows.
    pub(crate) fn rolling_rows(&mut self, length: usize, value: f64) -> (&mut [f64], &mut [f64]) {
        self.row_a.resize(length, value);
        self.row_b.resize(length, value);
        self.row_a.fill(value);
        self.row_b.fill(value);
        (&mut self.row_a, &mut self.row_b)
    }

    /// Resizes and zeroes both integer rolling rows.
    pub(crate) fn integer_rolling_rows(&mut self, length: usize) -> (&mut [usize], &mut [usize]) {
        self.integer_row_a.resize(length, 0);
        self.integer_row_b.resize(length, 0);
        self.integer_row_a.fill(0);
        self.integer_row_b.fill(0);
        (&mut self.integer_row_a, &mut self.integer_row_b)
    }

    /// Resizes and initializes cost and comparability rows for DTW-AROW.
    pub(crate) fn arow_rows(
        &mut self,
        length: usize,
    ) -> (&mut [f64], &mut [f64], &mut [bool], &mut [bool]) {
        self.row_a.resize(length, 0.0);
        self.row_b.resize(length, f64::INFINITY);
        self.comparable_row_a.resize(length, false);
        self.comparable_row_b.resize(length, false);
        self.row_a.fill(0.0);
        self.row_b.fill(f64::INFINITY);
        self.comparable_row_a.fill(false);
        self.comparable_row_b.fill(false);
        (
            &mut self.row_a,
            &mut self.row_b,
            &mut self.comparable_row_a,
            &mut self.comparable_row_b,
        )
    }

    /// Splits mutable access between transform buffers and rolling rows.
    pub(crate) fn transform_and_rows(
        &mut self,
        first_length: usize,
        second_length: usize,
        row_length: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        self.transform_a.resize(first_length, 0.0);
        self.transform_b.resize(second_length, 0.0);
        self.row_a.resize(row_length, f64::INFINITY);
        self.row_b.resize(row_length, f64::INFINITY);
        self.row_a.fill(f64::INFINITY);
        self.row_b.fill(f64::INFINITY);
        (
            &mut self.transform_a,
            &mut self.transform_b,
            &mut self.row_a,
            &mut self.row_b,
        )
    }
    /// Resizes weight storage and rolling rows for weighted DTW.
    pub(crate) fn weighted_rows(
        &mut self,
        weight_length: usize,
        row_length: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64]) {
        self.weights.resize(weight_length, 0.0);
        self.row_a.resize(row_length, f64::INFINITY);
        self.row_b.resize(row_length, f64::INFINITY);
        self.row_a.fill(f64::INFINITY);
        self.row_b.fill(f64::INFINITY);
        (&mut self.weights, &mut self.row_a, &mut self.row_b)
    }
}
