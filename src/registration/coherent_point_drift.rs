extern crate openblas_src;

use itertools::Itertools;
use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr, s, stack};
use ndarray_linalg::Solve;
use std::f32::EPSILON;
use std::f32::consts::PI;

struct CoherentPointDriftTransform {
    X: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    Y: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    lambda: f32,
    beta: f32,
    TY: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    variance: f32,
    tolerance: f32,
    w: f32,
    max_iterations: u32,
    change_in_variance: f32,
    probability_of_match: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    W: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    history: Vec<String>, // Contains all the TY matrices for all iterations for debugging.
    debug: bool,          // Whether or not to take a history.
}

impl CoherentPointDriftTransform {
    pub fn new(
        X: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        Y: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        lambda: f32,
        beta: f32,
        w: Option<f32>,
        tolerance: Option<f32>,
        max_iterations: Option<u32>,
        debug: Option<bool>,
    ) -> CoherentPointDriftTransform {
        let num_target_points: usize = X.dim().0;
        let dimensions: usize = X.dim().1;
        let num_source_points: usize = Y.dim().0;
        let initial_variance: f32 = {
            let sum_sq_dists = compute_squared_euclidean_distance(&X, &Y).sum();
            let denominator: f32 = 
                (dimensions as f32 * num_target_points as f32 * num_source_points as f32);
            sum_sq_dists / denominator
        };
        CoherentPointDriftTransform {
            X: X.clone(),
            Y: Y.clone(),
            lambda: lambda,
            beta: beta,
            TY: Y.clone(),
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
        let G = gaussian_kernel(&self.Y, &self.Y, self.beta);
        self.TY = transform_point_cloud(&self.Y, &G, &self.W);
        let mut iteration = 0;
        while iteration < self.max_iterations && self.change_in_variance > self.tolerance {
            if self.debug {
                self.history
                    .push(format!("\"{}\": {}", iteration, array_to_string(&self.TY)));
            }
            self.expectation();
            self.maximization();
            iteration += 1;
        }
    }

    fn expectation(&mut self) {
        let mut P = compute_squared_euclidean_distance(&self.X, &self.TY);
        P = (-P / (2_f32 * self.variance)).exp();
        let c = {
            let num_target_points: usize = self.X.dim().0;
            let dimensions: usize = self.X.dim().1;
            let num_source_points: usize = self.Y.dim().0;
            let left = (2.0 * PI * self.variance).powf((dimensions as f32) / 2.0);
            let right =
                self.w / (1.0 - self.w) * (num_source_points as f32) / (num_target_points as f32);
            left * right
        };
        let mut den = P.sum_axis(Axis(0));
        den = den.mapv(|v| if v == 0.0 { f32::EPSILON + c } else { v + c });

        self.probability_of_match = P.clone() / den;
    }

    fn maximization(&mut self) {
        let P1 = self.probability_of_match.clone().sum_axis(Axis(1));
        let Pt1 = self.probability_of_match.clone().sum_axis(Axis(0));
        let PX = self.probability_of_match.clone().dot(&self.X);
        let G = gaussian_kernel(&self.Y, &self.Y, self.beta);
        
        self.W = update_transform(&self.Y, &P1, &PX, &G, self.lambda, self.variance);
        self.TY = transform_point_cloud(&self.Y, &G, &self.W);
        self.update_variance(&P1, &Pt1, &PX);
    }

    fn update_variance(
        &mut self,
        P1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        Pt1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ) {
        let qprev = self.variance;
        let xPx = Pt1
            .clone()
            .t()
            .dot(&self.X.clone().powi(2).sum_axis(Axis(1)));
        let yPy = P1
            .clone()
            .t()
            .dot(&self.TY.clone().powi(2).sum_axis(Axis(1)));
        let trPXY = (self.TY.clone() * PX.clone()).sum();
        let dimensions = self.X.dim().1;
        self.variance = (xPx - 2.0 * trPXY + yPy) / (P1.clone().sum() * dimensions as f32);
        if self.variance <= 0.0 {
            self.variance = self.tolerance / 10.0;
        }
        self.change_in_variance = (self.variance - qprev).abs();
    }
}

/// Computes the squared euclidean distance between all vectors in A and B.
fn compute_squared_euclidean_distance(
    A: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    B: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let A_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> =
        Array::from_shape_vec((1, A.dim().0, A.dim().1), A.clone().into_raw_vec()).unwrap();
    let B_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> =
        Array::from_shape_vec((B.dim().0, 1, B.dim().1), B.clone().into_raw_vec()).unwrap();
    (A_3d - B_3d).powi(2).sum_axis(Axis(2))
}

/// Computes the gaussian kernel for CPD.
fn gaussian_kernel(
    A: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    B: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    beta: f32,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let sum_sq_dists = compute_squared_euclidean_distance(A, B);
    (-sum_sq_dists / (2.0 * beta.powi(2))).exp()
}

/// Computes the solution for a matrix equation AX = B.
///
/// Goes column by column through B to find the vector that
/// solves that equation and then contatenates all the solution
/// vectors as columns in the resulting matrix.
fn solve_matrices(
    A: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    B: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let A = A.clone();
    let B = B.clone();
    let num_cols = B.dim().1;
    let mut solutions: Vec<_> = Vec::new();
    for c in 0..num_cols {
        let col = B.slice(s![.., c]).to_owned();
        let soln = A.solve_into(col).unwrap();
        solutions.push(soln);
    }
    let solutions = solutions.iter().map(|x| x.view()).collect::<Vec<_>>();
    stack(Axis(1), &solutions[..]).unwrap()
}

fn transform_point_cloud(
    Y: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    G: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    W: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    Y + G.dot(W)
}

fn update_transform(
    Y: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    P1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    G: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    lambda: f32,
    variance: f32
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let A = {
        let num_source_points: usize = Y.dim().0;
        let left_term = Array::from_diag(&P1.clone()).dot(&G.clone());
        let right_term = lambda * variance * Array::eye(num_source_points.clone());
        left_term + right_term
    };
    let B = PX.clone() - Array::from_diag(&P1.clone()).dot(&Y.clone());
    solve_matrices(&A, &B)
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
