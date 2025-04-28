extern crate openblas_src;

use itertools::Itertools;
use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr, s, stack};
use ndarray_linalg::Solve;
use std::f32::EPSILON;
use std::f32::consts::PI;

struct CoherentPointDriftTransform {
    target_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    source_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    lambda: f32,
    beta: f32,
    transformed_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    variance: f32,
    tolerance: f32,
    w: f32,
    max_iterations: u32,
    change_in_variance: f32,
    probability_of_match: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    W: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    history: Vec<String>, // Contains all the transformed_points matrices for all iterations.
    debug: bool,          // Whether or not to take a history.
}

impl CoherentPointDriftTransform {
    pub fn new(
        target_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        source_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        lambda: f32,
        beta: f32,
        w: Option<f32>,
        tolerance: Option<f32>,
        max_iterations: Option<u32>,
        debug: Option<bool>,
    ) -> CoherentPointDriftTransform {
        let num_target_points: usize = target_points.dim().0;
        let dimensions: usize = target_points.dim().1;
        let num_source_points: usize = source_points.dim().0;
        let initial_variance: f32 = {
            let sum_sq_dists =
                compute_squared_euclidean_distance(&target_points, &source_points).sum();
            let denominator: f32 =
                dimensions as f32 * num_target_points as f32 * num_source_points as f32;
            sum_sq_dists / denominator
        };
        CoherentPointDriftTransform {
            target_points: target_points,
            source_points: source_points.clone(),
            lambda: lambda,
            beta: beta,
            transformed_points: source_points,
            variance: initial_variance,
            tolerance: tolerance.unwrap_or(0.001),
            w: w.unwrap_or(0.0),
            max_iterations: max_iterations.unwrap_or(100),
            change_in_variance: f32::MAX,
            probability_of_match: Array::zeros((num_source_points, num_target_points)),
            W: Array::zeros((num_source_points, dimensions)),
            history: Vec::new(),
            debug: debug.unwrap_or(false),
        }
    }

    pub fn register(&mut self) {
        let gaussian_kernel = compute_gaussian_kernel(&self.source_points, &self.source_points, self.beta);
        self.transformed_points = transform_point_cloud(&self.source_points, &gaussian_kernel, &self.W);
        let mut iteration = 0;
        while iteration < self.max_iterations && self.change_in_variance > self.tolerance {
            if self.debug {
                self.history.push(format!(
                    "\"{}\": {}",
                    iteration,
                    array_to_string(&self.transformed_points)
                ));
            }
            self.expectation();
            self.maximization();
            iteration += 1;
        }
    }

    fn expectation(&mut self) {
        let mut P =
            compute_squared_euclidean_distance(&self.target_points, &self.transformed_points);
        P = (-P / (2_f32 * self.variance)).exp();
        let c = {
            let num_target_points: usize = self.target_points.dim().0;
            let dimensions: usize = self.target_points.dim().1;
            let num_source_points: usize = self.source_points.dim().0;
            let left = (2.0 * PI * self.variance).powf((dimensions as f32) / 2.0);
            let right =
                self.w / (1.0 - self.w) * (num_source_points as f32) / (num_target_points as f32);
            left * right
        };
        let mut den = P.sum_axis(Axis(0));
        den = den.mapv(|v| if v == 0.0 { f32::EPSILON + c } else { v + c });

        self.probability_of_match = P / den;
    }

    fn maximization(&mut self) {
        let sum_of_probability_rows = self.probability_of_match.sum_axis(Axis(1));
        // Mathematically speaking, sum_of_probability_columns should always be
        // a vector of approximately ones (most runs the whole vector is within 1e-5 of 1.0).
        // However, I don't want to change the logic of the original author, so it stays.
        // TODO: Test whether this is necessary.
        let sum_of_probability_columns = self.probability_of_match.sum_axis(Axis(0));
        let PX = self.probability_of_match.dot(&self.target_points);
        let gaussian_kernel= compute_gaussian_kernel(&self.source_points, &self.source_points, self.beta);

        self.W = update_transform(
            &self.source_points,
            &sum_of_probability_rows,
            &PX,
            &gaussian_kernel,
            self.lambda,
            self.variance,
        );
        self.transformed_points = transform_point_cloud(&self.source_points, &gaussian_kernel, &self.W);
        (self.variance, self.change_in_variance) = update_variance(
            &self.target_points,
            &self.transformed_points,
            &sum_of_probability_rows,
            &sum_of_probability_columns,
            &PX,
            self.variance,
            self.tolerance,
        );
    }
}

/// Computes the squared euclidean distance between all vectors in A and B.
fn compute_squared_euclidean_distance(
    matrix_a: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    matrix_b: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let matrix_a_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> = Array::from_shape_vec(
        (1, matrix_a.dim().0, matrix_a.dim().1),
        matrix_a.clone().into_raw_vec_and_offset().0,
    )
    .unwrap();
    let matrix_b_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> = Array::from_shape_vec(
        (matrix_b.dim().0, 1, matrix_b.dim().1),
        matrix_b.clone().into_raw_vec_and_offset().0,
    )
    .unwrap();
    (matrix_a_3d - matrix_b_3d).powi(2).sum_axis(Axis(2))
}

/// Computes the gaussian kernel for CPD.
fn compute_gaussian_kernel(
    matrix_a: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    matrix_b: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    beta: f32,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let sum_sq_dists = compute_squared_euclidean_distance(matrix_a, matrix_b);
    (-sum_sq_dists / (2.0 * beta.powi(2))).exp()
}

/// Computes the solution for a matrix equation AX = B.
///
/// Goes column by column through B to find the vector that
/// solves that equation and then contatenates all the solution
/// vectors as columns in the resulting matrix.
fn solve_matrices(
    matrix_a: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    matrix_b: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let num_cols = matrix_b.dim().1;
    let mut solutions: Vec<_> = Vec::new();
    for column_ix in 0..num_cols {
        let col = matrix_b.slice(s![.., column_ix]).to_owned();
        let solution = matrix_a.solve_into(col).unwrap();
        solutions.push(solution);
    }
    let solutions = solutions.iter().map(|x| x.view()).collect::<Vec<_>>();
    stack(Axis(1), &solutions[..]).unwrap()
}

fn transform_point_cloud(
    source_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    gaussian_kernel: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    W: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    source_points + gaussian_kernel.dot(W)
}

fn update_transform(
    source_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    sum_of_probability_rows: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    gaussian_kernel: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    lambda: f32,
    variance: f32,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let A = {
        let num_source_points: usize = source_points.dim().0;
        let left_term = Array::from_diag(sum_of_probability_rows).dot(gaussian_kernel);
        let right_term = lambda * variance * Array::eye(num_source_points);
        left_term + right_term
    };
    let B = PX - Array::from_diag(sum_of_probability_rows).dot(source_points);
    solve_matrices(&A, &B)
}

fn update_variance(
    target_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    transformed_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    sum_of_probability_rows: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    sum_of_probability_columns: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    variance: f32,
    tolerance: f32,
) -> (f32, f32) {
    let previous_variance = variance;
    let xPx = sum_of_probability_columns
        .t()
        .dot(&target_points.powi(2).sum_axis(Axis(1)));
    let yPy = sum_of_probability_rows
        .t()
        .dot(&transformed_points.powi(2).sum_axis(Axis(1)));
    let trPXY = (transformed_points * PX).sum();
    let dimensions = target_points.dim().1;
    let mut new_variance =
        (xPx - 2.0 * trPXY + yPy) / (sum_of_probability_rows.sum() * dimensions as f32);
    if new_variance <= 0.0 {
        new_variance = tolerance / 10.0;
    }
    let change_in_variance = (new_variance - previous_variance).abs();
    (new_variance, change_in_variance)
}

/// A helper function for converting a 2d array into a string representation.
fn array_to_string(arr: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>) -> String {
    let mut arr_str = String::from("[");
    let arr = arr.clone();
    let chunks = arr.into_iter().chunks(2);
    let mut p_str: Vec<String> = Vec::new();
    for chunk in &chunks {
        let p: Vec<f32> = chunk.collect::<Vec<f32>>();
        p_str.push(String::from(format!(
            "{{\"x\": {}, \"y\": {}}}",
            p[0], p[1]
        )));
    }
    let p_str: String = p_str.join(", ");
    arr_str.push_str(&p_str);

    arr_str.push_str("]");
    arr_str
}
