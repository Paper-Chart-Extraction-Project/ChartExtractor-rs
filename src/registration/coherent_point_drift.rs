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
    sigma2: f32,
    num_target_points: usize,
    num_source_points: usize,
    dimensions: usize,
    tolerance: f32,
    w: f32,
    max_iterations: u32,
    iteration: u32,
    diff: f32,
    q: f32,
    P: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    W: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    num_eig: u32,
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
        num_eig: Option<u32>,
        debug: Option<bool>,
    ) -> CoherentPointDriftTransform {
        let num_target_points: usize = X.dim().0;
        let dimensions: usize = X.dim().1;
        let num_source_points: usize = Y.dim().0;
        CoherentPointDriftTransform {
            X: X.clone(),
            Y: Y.clone(),
            lambda: lambda,
            beta: beta,
            TY: Y.clone(),
            sigma2: initialize_sigma2(&X, &Y),
            num_target_points: num_target_points,
            num_source_points: num_source_points,
            dimensions: dimensions,
            tolerance: tolerance.unwrap_or(0.001),
            w: w.unwrap_or(0.0),
            max_iterations: max_iterations.unwrap_or(100),
            iteration: 0,
            diff: f32::MAX,
            q: f32::MAX,
            P: Array::zeros((num_source_points, num_target_points)),
            W: Array::zeros((num_source_points, dimensions)),
            num_eig: num_eig.unwrap_or(100),
            history: Vec::new(),
            debug: debug.unwrap_or(false),
        }
    }

    pub fn register(&mut self) {
        self.transform_point_cloud(&gaussian_kernel(&self.Y, &self.Y, self.beta));
        while self.iteration < self.max_iterations && self.diff > self.tolerance {
            if self.debug {
                self.history.push(format!(
                    "\"{}\": {}",
                    self.iteration,
                    array_to_string(&self.TY)
                ));
            }
            self.iterate();
        }
    }

    fn iterate(&mut self) {
        self.expectation();
        self.maximization();
        self.iteration += 1;
    }

    fn expectation(&mut self) {
        let mut P = ((compute_diff(&self.X, &self.TY)).powi(2)).sum_axis(Axis(2));
        P = (-P / (2_f32 * self.sigma2)).exp();
        let c = {
            let left = (2.0 * PI * self.sigma2).powf((self.dimensions as f32) / 2.0);
            let right = self.w / (1.0 - self.w) * (self.num_source_points as f32)
                / (self.num_target_points as f32);
            left * right
        };
        let mut den = P.sum_axis(Axis(0));
        den = den.mapv(|v| if v == 0.0 { f32::EPSILON + c } else { v + c });

        self.P = P.clone() / den;
    }

    fn maximization(&mut self) {
        let P1 = self.P.clone().sum_axis(Axis(1));
        let Pt1 = self.P.clone().sum_axis(Axis(0));
        let PX = self.P.clone().dot(&self.X);
        let G = gaussian_kernel(&self.Y, &self.Y, self.beta);

        self.update_transform(&P1, &PX, &G);
        self.transform_point_cloud(&G);
        self.update_variance(&P1, &Pt1, &PX);
    }

    fn update_transform(
        &mut self,
        P1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
        G: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ) {
        let A = {
            let left_term = Array::from_diag(&P1.clone()).dot(&G.clone());
            let right_term = self.lambda * self.sigma2 * Array::eye(self.num_source_points.clone());
            left_term + right_term
        };
        let B = PX.clone() - Array::from_diag(&P1.clone()).dot(&self.Y.clone());
        self.W = solve_matrices(&A, &B);
    }
    
    fn transform_point_cloud(
        &mut self,
        G: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ) {
        self.TY = self.Y.clone() + G.clone().dot(&self.W.clone());
    }

    fn update_variance(
        &mut self,
        P1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        Pt1: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        PX: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ) {
        let qprev = self.sigma2;
        self.q = f32::MAX;
        let xPx = Pt1
            .clone()
            .t()
            .dot(&self.X.clone().powi(2).sum_axis(Axis(1)));
        let yPy = P1
            .clone()
            .t()
            .dot(&self.TY.clone().powi(2).sum_axis(Axis(1)));
        let trPXY = (self.TY.clone() * PX.clone()).sum();

        self.sigma2 = (xPx - 2.0 * trPXY + yPy) / (P1.clone().sum() * self.dimensions as f32);
        if self.sigma2 <= 0.0 {
            self.sigma2 = self.tolerance / 10.0;
        }
        self.diff = (self.sigma2 - qprev).abs();
    }
}

// Non cpd functions.
fn compute_diff(
    X: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    Y: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> {
    let X_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> =
        Array::from_shape_vec((1, X.dim().0, X.dim().1), X.clone().into_raw_vec()).unwrap();
    let Y_3d: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> =
        Array::from_shape_vec((Y.dim().0, 1, Y.dim().1), Y.clone().into_raw_vec()).unwrap();
    X_3d - Y_3d
}

/// Computes the gaussian kernel for CPD.
fn gaussian_kernel(
    X: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    Y: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    beta: f32,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let X = X.clone();
    let Y = Y.clone();
    let diff = compute_diff(&X, &Y).powi(2).sum_axis(Axis(2));
    (-diff / (2.0 * beta.powi(2))).exp()
}

/// Initializes sigma squared.
fn initialize_sigma2(
    X: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    Y: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> f32 {
    let num_target_points = X.dim().0 as f32;
    let dimensions = X.dim().1 as f32;
    let num_source_points = Y.dim().0 as f32;
    let diff = compute_diff(X, Y);
    let err = diff.powi(2);
    err.sum() / (dimensions * num_target_points * num_source_points)
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
