extern crate openblas_src;

use crate::annotations::point::Point;
use itertools::Itertools;
use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr, s, stack};
use ndarray_linalg::Solve;
use std::f32::EPSILON;
use std::f32::consts::PI;

pub struct CoherentPointDriftTransform {
    /// The points to try to move the source towards.
    target_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    /// The points to move towards the target points. May contain outliers or
    /// missing points.
    source_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    /// The tradeoff between the goodness of maximum likelihood fit and regularlization.
    lambda: f32,
    /// The width of the smoothing Gaussian filter.
    beta: f32,
    /// The source points after they have been moved by the cpd algorithm.
    transformed_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    /// The variance of the Gaussian mixture model.
    variance: f32,
    /// A parameter that can end the iteration process early if the change in variance
    /// is less than the tolerance.
    tolerance: f32,
    /// The weight of the uniform distribution. Must be between 0 and 1.
    /// See the coherent point drift paper for more details.
    weight_of_uniform_dist: f32,
    /// The maximum number of iterations to perform.
    max_iterations: u32,
    /// The change in variance between the previous iteration and the current one.
    change_in_variance: f32,
    /// An MxN matrix containing the probability that a point from the target_points
    /// set matches with a point from the source_points set.
    probability_of_match: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    /// A matrix which, when linearly combined with the Gaussian kernel, contains
    /// the optimal displacement field to align the source points to the target.
    w_coefs: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    /// A vector of json formatted lists containing the transformed_points at all
    /// iterations. Use with caution, and set max_iterations low to start.
    pub history: Vec<String>,
    /// Whether or not to record the history of the transformed points.
    debug: bool,
}

impl CoherentPointDriftTransform {
    pub fn new(
        target_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        source_points: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        lambda: f32,
        beta: f32,
        weight_of_uniform_dist: Option<f32>,
        tolerance: Option<f32>,
        max_iterations: Option<u32>,
        debug: Option<bool>,
    ) -> CoherentPointDriftTransform {
        let num_target_points: usize = target_points.dim().0;
        let dimensions: usize = target_points.dim().1;
        let num_source_points: usize = source_points.dim().0;
        let initial_variance: f32 = {
            let sum_sq_dists = compute_squared_distance(&target_points, &source_points).sum();
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
            weight_of_uniform_dist: weight_of_uniform_dist.unwrap_or(0.0),
            max_iterations: max_iterations.unwrap_or(100),
            change_in_variance: f32::MAX,
            probability_of_match: Array::zeros((num_source_points, num_target_points)),
            w_coefs: Array::zeros((num_source_points, dimensions)),
            history: Vec::new(),
            debug: debug.unwrap_or(false),
        }
    }

    pub fn from_point_vectors(
        target_points: Vec<Point>,
        source_points: Vec<Point>,
        lambda: f32,
        beta: f32,
        weight_of_uniform_dist: Option<f32>,
        tolerance: Option<f32>,
        max_iterations: Option<u32>,
        debug: Option<bool>,
    ) -> CoherentPointDriftTransform {
        let target_point_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = {
            let mut flattened_point_vec = Vec::new();
            for p in target_points.iter() {
                flattened_point_vec.push(p.x);
                flattened_point_vec.push(p.y);
            }
            Array::from_shape_vec((target_points.len(), 2), flattened_point_vec).unwrap()
        };
        let source_point_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = {
            let mut flattened_point_vec = Vec::new();
            for p in source_points.iter() {
                flattened_point_vec.push(p.x);
                flattened_point_vec.push(p.y);
            }
            Array::from_shape_vec((source_points.len(), 2), flattened_point_vec).unwrap()
        };
        CoherentPointDriftTransform::new(
            target_point_array,
            source_point_array,
            lambda,
            beta,
            weight_of_uniform_dist,
            tolerance,
            max_iterations,
            debug,
        )
    }

    pub fn register(&mut self) {
        let gaussian_kernel =
            compute_gaussian_kernel(&self.source_points, &self.source_points, self.beta);
        self.transformed_points =
            compute_transformed_point_cloud(&self.source_points, &gaussian_kernel, &self.w_coefs);
        let mut iteration = 0;
        while iteration < self.max_iterations && self.change_in_variance > self.tolerance {
            if self.debug {
                self.history.push(format!(
                    "\"{}\": {}",
                    iteration,
                    array_to_json_string(&self.transformed_points)
                ));
            }
            self.expectation();
            self.maximization();
            iteration += 1;
        }
    }

    fn expectation(&mut self) {
        let mut new_probabilities =
            compute_squared_distance(&self.target_points, &self.transformed_points);
        new_probabilities = (-new_probabilities / (2_f32 * self.variance)).exp();
        let c = {
            let num_target_points: usize = self.target_points.dim().0;
            let dimensions: usize = self.target_points.dim().1;
            let num_source_points: usize = self.source_points.dim().0;
            let left = (2.0 * PI * self.variance).powf((dimensions as f32) / 2.0);
            let right = self.weight_of_uniform_dist / (1.0 - self.weight_of_uniform_dist)
                * (num_source_points as f32)
                / (num_target_points as f32);
            left * right
        };
        let mut den = new_probabilities.sum_axis(Axis(0));
        den = den.mapv(|v| if v == 0.0 { f32::EPSILON + c } else { v + c });

        self.probability_of_match = new_probabilities / den;
    }

    fn maximization(&mut self) {
        let sum_of_probability_rows = self.probability_of_match.sum_axis(Axis(1));
        // Mathematically speaking, sum_of_probability_columns should always be
        // a vector of approximately ones (most runs the whole vector is within 1e-5 of 1.0).
        // However, I don't want to change the logic of the original author, so it stays.
        // TODO: Test whether this is necessary.
        let sum_of_probability_columns = self.probability_of_match.sum_axis(Axis(0));
        let PX = self.probability_of_match.dot(&self.target_points);
        let gaussian_kernel =
            compute_gaussian_kernel(&self.source_points, &self.source_points, self.beta);

        self.w_coefs = compute_updated_transform(
            &self.source_points,
            &sum_of_probability_rows,
            &PX,
            &gaussian_kernel,
            self.lambda,
            self.variance,
        );
        self.transformed_points =
            compute_transformed_point_cloud(&self.source_points, &gaussian_kernel, &self.w_coefs);
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

    /// Uses the probability_of_match matrix to get a highest-likelihood match
    /// between the source and the target points.
    ///
    /// Returns a vector of tuples of 2 indices, the first belonging to a point
    /// in the source set and the second in the target set.
    pub fn generate_matching(&self) -> Vec<(usize, usize)> {
        fn generate_matching_inner(
            probs: Vec<((usize, usize), &f32)>,
            mut matches: Vec<(usize, usize)>,
        ) -> Vec<(usize, usize)> {
            if probs == vec![] {
                return matches;
            }
            let (max_index, max_value) = probs
                .iter()
                .max_by(|&&(_, &a), &&(_, &b)| {
                    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            matches.push(*max_index);
            let probs: Vec<((usize, usize), &f32)> = probs
                .clone()
                .into_iter()
                .filter(|((row, col), _)| *row != max_index.0 && *col != max_index.1)
                .collect::<Vec<((usize, usize), &f32)>>();
            generate_matching_inner(probs, matches)
        }
        let indexed_probabilities: Vec<((usize, usize), &f32)> = self
            .probability_of_match
            .indexed_iter()
            .collect::<Vec<((usize, usize), &f32)>>();
        let matching: Vec<(usize, usize)> =
            generate_matching_inner(indexed_probabilities, Vec::new());
        println!("{:?}", matching);
        matching
    }
}

/// Computes the squared euclidean distance between all vectors in A and B.
fn compute_squared_distance(
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
    let sum_sq_dists = compute_squared_distance(matrix_a, matrix_b);
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

fn compute_transformed_point_cloud(
    source_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    gaussian_kernel: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    w_coefs: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    source_points + gaussian_kernel.dot(w_coefs)
}

fn compute_updated_transform(
    source_points: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    sum_of_probability_rows: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    gaussian_kernel: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    lambda: f32,
    variance: f32,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let matrix_a = {
        let num_source_points: usize = source_points.dim().0;
        let left_term = Array::from_diag(sum_of_probability_rows).dot(gaussian_kernel);
        let right_term = lambda * variance * Array::eye(num_source_points);
        left_term + right_term
    };
    let matrix_b = PX - Array::from_diag(sum_of_probability_rows).dot(source_points);
    solve_matrices(&matrix_a, &matrix_b)
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
///
/// Used for debugging CoherentPointDriftTransform. When debug is set to true,
/// the transformed point cloud is dumped to a json formatted string using
/// this function.
fn array_to_json_string(array: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>) -> String {
    let mut array_str = String::from("[");
    let array = array.clone();
    let chunks = array.into_iter().chunks(2);
    let mut point_str: Vec<String> = Vec::new();
    for chunk in &chunks {
        let point: Vec<f32> = chunk.collect::<Vec<f32>>();
        point_str.push(String::from(format!(
            "{{\"x\": {}, \"y\": {}}}",
            point[0], point[1]
        )));
    }
    let point_str: String = point_str.join(", ");
    array_str.push_str(&point_str);

    array_str.push_str("]");
    array_str
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching() {
        let small_delta: f32 = 0.2;
        let source_points = Array::from_shape_vec(
            (3, 2),
            vec![
                1.0 - small_delta,
                0.0 + small_delta,
                0.5 + small_delta,
                0.5 - small_delta,
                0.0 + small_delta,
                0.0 - small_delta,
            ],
        )
        .unwrap();
        let target_points =
            Array::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.5]).unwrap();
        let mut cpd_transform = CoherentPointDriftTransform::new(
            target_points,
            source_points,
            0.01,
            20.0,
            Some(0.0),
            None,
            Some(100),
            Some(true),
        );
        cpd_transform.register();
        let matches = cpd_transform.generate_matching();
        let true_matches = vec![(2, 0), (0, 1), (1, 2)];
        assert_eq!(matches, true_matches)
    }

    #[test]
    fn test_matching_extra_point() {
        let small_delta: f32 = 0.2;
        let source_points = Array::from_shape_vec(
            (4, 2),
            vec![
                1.0 - small_delta,
                0.0 + small_delta,
                0.5 + small_delta,
                0.5 - small_delta,
                0.0 + small_delta,
                0.0 - small_delta,
                3.0 + small_delta,
                3.5 - small_delta,
            ],
        )
        .unwrap();
        let target_points =
            Array::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.5]).unwrap();
        let mut cpd_transform = CoherentPointDriftTransform::new(
            target_points,
            source_points,
            0.01,
            20.0,
            Some(0.0),
            None,
            Some(100),
            Some(true),
        );
        cpd_transform.register();
        let matches = cpd_transform.generate_matching();
        let true_matches = vec![(2, 0), (0, 1), (1, 2)];
        assert_eq!(matches, true_matches)
    }

    #[test]
    fn test_solve_matrices() {
        let mat_1: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                0.0,
                1.0,
                4.0,
                3.0
            ]
        ).unwrap();
        let mat_2: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                1.0,
                2.0,
                5.0,
                4.0
            ]
        ).unwrap();
        let true_soln:  ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                0.5,
                -0.5,
                1.0,
                2.0
            ]
        ).unwrap(); 
        assert_eq!(true_soln, solve_matrices(&mat_1, &mat_2))
    }

    #[test]
    fn test_compute_squared_distance() {
        let mat_1: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                0.0,
                1.0,
                4.0,
                3.0
            ]
        ).unwrap();
        let mat_2: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                1.0,
                2.0,
                5.0,
                4.0
            ]
        ).unwrap();
        let true_sq_dist: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                2.0,
                10.0,
                34.0,
                2.0,
            ]
        ).unwrap();
        let dists = compute_squared_distance(&mat_1, &mat_2);
        assert_eq!(true_sq_dist, dists)
    }

    #[test]
    fn test_compute_variance() {
        let mat_1: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                0.0,
                1.0,
                4.0,
                3.0
            ]
        ).unwrap();
        let mat_2: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::from_shape_vec(
            (2, 2),
            vec![
                1.0,
                2.0,
                5.0,
                4.0
            ]
        ).unwrap();
        let true_variance: f32 = 6.0;
        let computed_variance: f32 = {
            let sum_sq_dists = compute_squared_distance(&mat_1, &mat_2).sum();
            let denominator: f32 = 2_f32 * 2_f32 * 2_f32;
            sum_sq_dists / denominator
        };
        assert_eq!(true_variance, computed_variance)
    }
}
