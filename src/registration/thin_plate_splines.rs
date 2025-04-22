extern crate openblas_src;

use crate::annotations::point::Point;
use ndarray::{Array, ArrayBase, Axis, concatenate, Dim, OwnedRepr, stack};
use ndarray_linalg::Solve;

pub struct TpsTransform {
    pub source: Vec<Point>,
    pub destination: Vec<Point>
}

impl TpsTransform {

    pub fn transform_point(&self, p: Point) -> Point {
        let l_matrix = self._create_l_matrix();
        let w_matrix = self._solve_for_w_matrix();
        let mut kernel_vec = vec![];
        for dest_point in self.destination.iter() {
            kernel_vec.push(self._kernel(*dest_point, p));
        }
        kernel_vec.push(1.0);
        kernel_vec.push(p.x);
        kernel_vec.push(p.y);
        let kernel_vec = Array::from_shape_vec((1, kernel_vec.len()), kernel_vec).unwrap();
        let out = kernel_vec.dot(&w_matrix);
        let new_x = out.index_axis(Axis(1), 0).to_vec()[0];
        let new_y = out.index_axis(Axis(1), 1).to_vec()[0];
        println!("{:?}, {:?}", new_x, new_y);
        Point{ x: new_x, y: new_y }
    }

    fn _create_l_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        let k_matrix = self._create_k_matrix();
        let p_matrix = self._create_p_matrix();
        let p_transpose = self._create_p_matrix().reversed_axes();
        let o_matrix = self._create_o_matrix();
        let l_matrix = concatenate(
            Axis(0),
            &[
                concatenate(Axis(1), &[k_matrix.view(), p_matrix.view()]).unwrap().view(),
                concatenate(Axis(1), &[p_transpose.view(), o_matrix.view()]).unwrap().view()
            ]
        ).unwrap();
        l_matrix
    }

    fn _create_k_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        let mut k_matrix: Vec<f32> = Vec::new();
        for source_point in self.source.iter() {
            for dest_point in self.destination.iter() {
                if source_point == dest_point {
                    k_matrix.push(0_f32);
                } else {
                    k_matrix.push(self._kernel(source_point.clone(), dest_point.clone()));
                }
            }
        }
        Array::from_shape_vec((self.source.len(), self.source.len()), k_matrix).unwrap()
    }

    fn _create_p_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        let mut p_values: Vec<f32> = Vec::new();
        for point in self.source.iter() {
            p_values.push(1_f32);
            p_values.push(point.x);
            p_values.push(point.y);
        }
        Array::from_shape_vec((self.source.len(), 3), p_values).unwrap()
    }

    fn _create_o_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        Array::zeros((3, 3))   
    }

    fn _create_b_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        let mut b_values = Vec::new();
        for point in self.destination.iter() {
            b_values.push(point.x);
            b_values.push(point.y);
        }
        let ZEROS_TO_ADD = 6;
        for _ in 0..ZEROS_TO_ADD {
            b_values.push(0_f32);
        }
        Array::from_shape_vec((self.destination.len() + 3, 2), b_values).unwrap()
    }

    fn _euclidean_distance(&self, p1: Point, p2: Point) -> f32 {
        ((p1.x-p2.x).powi(2) + (p1.y-p2.y).powi(2)).sqrt()
    }
    
    fn _kernel(&self, p1: Point, p2: Point) -> f32 {
        let dist = self._euclidean_distance(p1, p2);
        dist.powi(2) * dist.ln()
    }

    fn _solve_for_w_matrix(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        let l_matrix = self._create_l_matrix();
        let b_matrix = self._create_b_matrix();
        let col_0 = b_matrix.column(0).to_owned();
        let col_1 = b_matrix.column(1).to_owned();
        let w_matrix_col_0 = l_matrix.solve(&col_0).unwrap();
        let w_matrix_col_1 = l_matrix.solve(&col_1).unwrap();
        stack(Axis(1), &[w_matrix_col_0.view(), w_matrix_col_1.view()]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_testing_transform() -> TpsTransform {
        let source: Vec<Point> = vec![
            Point { x: 0_f32, y: 0_f32 },
            Point { x: 2_f32, y: 0_f32 },
            Point { x: 0_f32, y: 2_f32 },
            Point { x: 2_f32, y: 2_f32 }
        ];
        let destination: Vec<Point> = vec![
            Point { x: 0_f32, y: 0_f32 },
            Point { x: 2_f32, y: 0_f32 },
            Point { x: 0.5_f32, y: 2_f32 },
            Point { x: 1.5_f32, y: 2_f32 }
        ];
        TpsTransform { source, destination }
    }
    
    #[test]
    fn create_o_matrix() {
        let test_transf = create_testing_transform();
        let true_o_matrix = Array::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).unwrap();
        assert!(test_transf._create_o_matrix().eq(&true_o_matrix));
    }

    #[test]
    fn create_p_matrix() {
        let test_transf = create_testing_transform();
        let true_p_matrix = Array::from_shape_vec(
            (4, 3),
            vec![
                1_f32, 0_f32, 0_f32, 1_f32, 2_f32, 0_f32, 1_f32, 0_f32, 2_f32, 1_f32, 2_f32, 2_f32
            ]
        ).unwrap();
        assert!(test_transf._create_p_matrix().eq(&true_p_matrix))
    }

    #[test]
    fn create_p_transpose_matrix() {
        let test_transf = create_testing_transform();
        let true_p_matrix = Array::from_shape_vec(
            (3, 4),
            vec![
                1_f32, 1_f32, 1_f32, 1_f32, 0_f32, 2_f32, 0_f32, 2_f32, 0_f32, 0_f32, 2_f32, 2_f32
            ]
        ).unwrap();
        assert!(test_transf._create_p_matrix().reversed_axes().eq(&true_p_matrix))
    }

    #[test]
    fn create_k_matrix() {
        let test_transf = create_testing_transform();
        let true_k_matrix = Array::from_shape_vec(
            (4, 4),
            vec![
                0_f32,
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 0_f32, y: 0_f32 }),
                0_f32,
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 0_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 0_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
            ]
        ).unwrap();
        assert!(test_transf._create_k_matrix().eq(&true_k_matrix))
    }

    #[test]
    fn create_l_matrix() {
        let test_transf = create_testing_transform();
        let true_l_matrix = Array::from_shape_vec(
            (7, 7),
            vec![
                // row 0
                0_f32,
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 0_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                1_f32,
                0_f32,
                0_f32,
                // row 1
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 0_f32, y: 0_f32 }),
                0_f32,
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 0_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                1_f32,
                2_f32,
                0_f32,
                // row 2
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 0_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 0_f32, y: 2_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                1_f32,
                0_f32,
                2_f32,
                // row 3
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 0_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 2_f32, y: 0_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 0.5_f32, y: 2_f32 }),
                test_transf._kernel(Point { x: 2_f32, y: 2_f32 }, Point { x: 1.5_f32, y: 2_f32 }),
                1_f32,
                2_f32,
                2_f32,
                // row 4
                1_f32,
                1_f32,
                1_f32,
                1_f32,
                0_f32,
                0_f32,
                0_f32,
                // row 5
                0_f32,
                2_f32,
                0_f32,
                2_f32,
                0_f32,
                0_f32,
                0_f32,
                // row 6
                0_f32,
                0_f32,
                2_f32,
                2_f32,
                0_f32,
                0_f32,
                0_f32,
            ]
        ).unwrap();
        assert!(test_transf._create_l_matrix().eq(&true_l_matrix));
    }

    #[test]
    fn create_b_matrix() {
        let test_transf = create_testing_transform();
        let true_b_matrix = Array::from_shape_vec(
            (7, 2),
            vec![
                0_f32,
                0_f32,
                2_f32,
                0_f32,
                0.5_f32,
                2_f32,
                1.5_f32,
                2_f32,
                0_f32,
                0_f32,
                0_f32,
                0_f32,
                0_f32,
                0_f32
            ]
        ).unwrap();
        assert!(test_transf._create_b_matrix().eq(&true_b_matrix));
    }

    #[test]
    fn solve_for_w_matrix() {
        let test_transf = create_testing_transform();
        let w_matrix = test_transf._solve_for_w_matrix();
        let l_matrix = test_transf._create_l_matrix();
        let b_matrix = test_transf._create_b_matrix();
        println!("{:?}", w_matrix);
        println!("{:?}", l_matrix.dot(&w_matrix));
        assert!(l_matrix.dot(&w_matrix).abs_diff_eq(&b_matrix, 0.0001));
    }

    #[test]
    fn test_tranform_point() {
        let test_transf = create_testing_transform();
        let transformed_point = test_transf.transform_point(Point { x: 2.0, y: 2.0 });
        let true_transformed_point = Point{ x: 1.5, y: 2.0 };
        assert_eq!(transformed_point, true_transformed_point);
    }
}
