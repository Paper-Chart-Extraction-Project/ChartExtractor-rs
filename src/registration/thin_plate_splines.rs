extern crate openblas_src;

use crate::annotations::point::Point;
use crate::annotations::bounding_box::{BoundingBox, BoundingBoxError, BoundingBoxGeometry};
use crate::annotations::bounding_box_with_keypoint::BoundingBoxWithKeypoint;
use ndarray::{Array, ArrayBase, Axis, Dim, OwnedRepr, concatenate, stack};
use ndarray_linalg::Solve;
use std::cmp::{Ordering, max_by, min_by};
use std::iter::zip;

pub struct TpsTransform {
    source: Vec<Point>,
    destination: Vec<Point>,
    w_matrix: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
}

impl TpsTransform {
    pub fn new(source: Vec<Point>, destination: Vec<Point>) -> TpsTransform {
        let w_matrix = solve_for_w_matrix(&source, &destination); // Cached for performance.
        TpsTransform {
            source,
            destination,
            w_matrix,
        }
    }

    pub fn transform_point(&self, p: Point) -> Point {
        let mut kernel_vec = vec![];
        for dest_point in self.destination.iter() {
            kernel_vec.push(kernel(dest_point, &p));
        }
        kernel_vec.push(1.0);
        kernel_vec.push(p.x);
        kernel_vec.push(p.y);
        let kernel_vec = Array::from_shape_vec((1, kernel_vec.len()), kernel_vec).unwrap();
        let out = kernel_vec.dot(&self.w_matrix);
        let new_x = out.index_axis(Axis(1), 0).to_vec()[0];
        let new_y = out.index_axis(Axis(1), 1).to_vec()[0];
        Point { x: new_x, y: new_y }
    }

    pub fn transform_bounding_box(
        &self,
        bbox: BoundingBox
    ) -> Result<BoundingBox, BoundingBoxError> {
        let category: String = bbox.category().clone();
        let (new_left, new_top, new_right, new_bottom) = self.transform_box(bbox);
        BoundingBox::new(new_left, new_top, new_right, new_bottom, category)
    }

    pub fn transform_bounding_box_with_keypoint(
        &self,
        bbox_with_kp: BoundingBoxWithKeypoint,
    ) -> Result<BoundingBoxWithKeypoint, BoundingBoxError> {
        let category: String = bbox_with_kp.category().clone();
        let keypoint_x: f32 = bbox_with_kp.get_keypoint_x();
        let keypoint_y: f32 = bbox_with_kp.get_keypoint_y();
        let keypoint: Point = Point { x: keypoint_x, y: keypoint_y };
        let new_keypoint: Point = self.transform_point(keypoint);
        let (new_left, new_top, new_right, new_bottom) = self.transform_box(bbox_with_kp);
        BoundingBoxWithKeypoint::new(
            new_left,
            new_top,
            new_right,
            new_bottom,
            new_keypoint.x,
            new_keypoint.y,
            category,
        )
    }

    fn transform_box<T: BoundingBoxGeometry> (&self, bbox: T) -> (f32, f32, f32, f32) {
        let left_top: Point = Point { x: bbox.left(), y: bbox.top() };
        let left_bottom: Point = Point { x: bbox.left(), y: bbox.bottom() };
        let right_top: Point = Point { x: bbox.right(), y: bbox.top() };
        let right_bottom: Point = Point { x: bbox.right(), y: bbox.bottom() };

        let transformed_left_top: Point = self.transform_point(left_top);
        let transformed_left_bottom: Point = self.transform_point(left_bottom);
        let transformed_right_top: Point = self.transform_point(right_top);
        let transformed_right_bottom: Point = self.transform_point(right_bottom);
        
        let f32_cmp = |a: &f32, b: &f32| a.total_cmp(b);
        let new_left: f32 = min_by(left_top.x, left_bottom.x, f32_cmp);
        let new_top: f32 = min_by(left_top.y, right_top.y, f32_cmp);
        let new_right: f32 = max_by(right_top.x, right_bottom.x, f32_cmp);
        let new_bottom: f32 = max_by(left_bottom.y, right_bottom.y, f32_cmp);

        (new_left, new_top, new_right, new_bottom)
    }
}

fn create_l_matrix(
    source: &[Point],
    destination: &[Point],
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let k_matrix = create_k_matrix(source, destination);
    let p_matrix = create_p_matrix(source);
    let p_transpose = p_matrix.clone().reversed_axes();
    let o_matrix = create_o_matrix();
    let l_matrix = concatenate(
        Axis(0),
        &[
            concatenate(Axis(1), &[k_matrix.view(), p_matrix.view()])
                .unwrap()
                .view(),
            concatenate(Axis(1), &[p_transpose.view(), o_matrix.view()])
                .unwrap()
                .view(),
        ],
    )
    .unwrap();
    l_matrix
}

fn create_k_matrix(
    source: &[Point],
    destination: &[Point],
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let mut k_matrix: Vec<f32> = Vec::new();
    for source_point in source.iter() {
        for dest_point in destination.iter() {
            if source_point == dest_point {
                k_matrix.push(0_f32);
            } else {
                k_matrix.push(kernel(source_point, dest_point));
            }
        }
    }
    Array::from_shape_vec((source.len(), source.len()), k_matrix).unwrap()
}

fn create_p_matrix(source: &[Point]) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let mut p_values: Vec<f32> = Vec::new();
    for point in source.iter() {
        p_values.push(1_f32);
        p_values.push(point.x);
        p_values.push(point.y);
    }
    Array::from_shape_vec((source.len(), 3), p_values).unwrap()
}

fn create_o_matrix() -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    Array::zeros((3, 3))
}

fn kernel(p1: &Point, p2: &Point) -> f32 {
    let dist = euclidean_distance(&p1, &p2);
    match dist {
        0.0 => 0.0,
        _ => dist.powi(2) * dist.ln(),
    }
}

fn euclidean_distance(p1: &Point, p2: &Point) -> f32 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

fn create_b_matrix(destination: &[Point]) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let mut b_values = Vec::new();
    for point in destination.iter() {
        b_values.push(point.x);
        b_values.push(point.y);
    }
    let zeros_to_add = 6;
    for _ in 0..zeros_to_add {
        b_values.push(0_f32);
    }
    Array::from_shape_vec((destination.len() + 3, 2), b_values).unwrap()
}

fn solve_for_w_matrix(
    source: &[Point],
    destination: &[Point],
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    let l_matrix = create_l_matrix(source, destination);
    let b_matrix = create_b_matrix(destination);
    let col_0 = b_matrix.column(0).to_owned();
    let col_1 = b_matrix.column(1).to_owned();
    let w_matrix_col_0 = l_matrix.solve(&col_0).unwrap();
    let w_matrix_col_1 = l_matrix.solve(&col_1).unwrap();
    stack(Axis(1), &[w_matrix_col_0.view(), w_matrix_col_1.view()]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_testing_transform() -> TpsTransform {
        let source: Vec<Point> = vec![
            Point { x: 0_f32, y: 0_f32 },
            Point { x: 2_f32, y: 0_f32 },
            Point { x: 0_f32, y: 2_f32 },
            Point { x: 2_f32, y: 2_f32 },
        ];
        let destination: Vec<Point> = vec![
            Point { x: 0_f32, y: 0_f32 },
            Point { x: 2_f32, y: 0_f32 },
            Point {
                x: 0.5_f32,
                y: 2_f32,
            },
            Point {
                x: 1.5_f32,
                y: 2_f32,
            },
        ];
        TpsTransform::new(source, destination)
    }

    #[test]
    fn test_create_o_matrix() {
        let true_o_matrix =
            Array::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap();
        let z = create_o_matrix();
        assert!(z == true_o_matrix);
    }

    #[test]
    fn test_create_p_matrix() {
        let test_transf = create_testing_transform();
        let true_p_matrix = Array::from_shape_vec(
            (4, 3),
            vec![
                1_f32, 0_f32, 0_f32, 1_f32, 2_f32, 0_f32, 1_f32, 0_f32, 2_f32, 1_f32, 2_f32, 2_f32,
            ],
        )
        .unwrap();
        assert!(create_p_matrix(&test_transf.source).eq(&true_p_matrix))
    }

    #[test]
    fn test_create_p_transpose_matrix() {
        let test_transf = create_testing_transform();
        let true_p_matrix = Array::from_shape_vec(
            (3, 4),
            vec![
                1_f32, 1_f32, 1_f32, 1_f32, 0_f32, 2_f32, 0_f32, 2_f32, 0_f32, 0_f32, 2_f32, 2_f32,
            ],
        )
        .unwrap();
        assert!(
            create_p_matrix(&test_transf.source)
                .reversed_axes()
                .eq(&true_p_matrix)
        )
    }

    #[test]
    fn test_create_k_matrix() {
        let test_transf = create_testing_transform();
        let true_k_matrix = Array::from_shape_vec(
            (4, 4),
            vec![
                0_f32,
                kernel(&Point { x: 0_f32, y: 0_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 0_f32, y: 0_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 0_f32, y: 0_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(&Point { x: 2_f32, y: 0_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                0_f32,
                kernel(
                    &Point { x: 2_f32, y: 0_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 2_f32, y: 0_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(&Point { x: 0_f32, y: 2_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                kernel(&Point { x: 0_f32, y: 2_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 0_f32, y: 2_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 0_f32, y: 2_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(&Point { x: 2_f32, y: 2_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                kernel(&Point { x: 2_f32, y: 2_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 2_f32, y: 2_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 2_f32, y: 2_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
            ],
        )
        .unwrap();
        assert!(create_k_matrix(&test_transf.source, &test_transf.destination) == true_k_matrix)
    }

    #[test]
    fn test_create_l_matrix() {
        let test_transf = create_testing_transform();
        let true_l_matrix = Array::from_shape_vec(
            (7, 7),
            vec![
                // row 0
                0_f32,
                kernel(&Point { x: 0_f32, y: 0_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 0_f32, y: 0_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 0_f32, y: 0_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                1_f32,
                0_f32,
                0_f32,
                // row 1
                kernel(&Point { x: 2_f32, y: 0_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                0_f32,
                kernel(
                    &Point { x: 2_f32, y: 0_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 2_f32, y: 0_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                1_f32,
                2_f32,
                0_f32,
                // row 2
                kernel(&Point { x: 0_f32, y: 2_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                kernel(&Point { x: 0_f32, y: 2_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 0_f32, y: 2_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 0_f32, y: 2_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
                1_f32,
                0_f32,
                2_f32,
                // row 3
                kernel(&Point { x: 2_f32, y: 2_f32 }, &Point { x: 0_f32, y: 0_f32 }),
                kernel(&Point { x: 2_f32, y: 2_f32 }, &Point { x: 2_f32, y: 0_f32 }),
                kernel(
                    &Point { x: 2_f32, y: 2_f32 },
                    &Point {
                        x: 0.5_f32,
                        y: 2_f32,
                    },
                ),
                kernel(
                    &Point { x: 2_f32, y: 2_f32 },
                    &Point {
                        x: 1.5_f32,
                        y: 2_f32,
                    },
                ),
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
            ],
        )
        .unwrap();
        assert!(create_l_matrix(&test_transf.source, &test_transf.destination).eq(&true_l_matrix));
    }

    #[test]
    fn test_create_b_matrix() {
        let test_transf = create_testing_transform();
        let true_b_matrix = Array::from_shape_vec(
            (7, 2),
            vec![
                0_f32, 0_f32, 2_f32, 0_f32, 0.5_f32, 2_f32, 1.5_f32, 2_f32, 0_f32, 0_f32, 0_f32,
                0_f32, 0_f32, 0_f32,
            ],
        )
        .unwrap();
        assert!(create_b_matrix(&test_transf.destination).eq(&true_b_matrix));
    }

    #[test]
    fn test_solve_for_w_matrix() {
        let test_transf = create_testing_transform();
        let w_matrix = solve_for_w_matrix(&test_transf.source, &test_transf.destination);
        let l_matrix = create_l_matrix(&test_transf.source, &test_transf.destination);
        let b_matrix = create_b_matrix(&test_transf.destination);
        assert!(l_matrix.dot(&w_matrix).abs_diff_eq(&b_matrix, 0.0001));
    }

    #[test]
    fn test_tranform_point() {
        let test_transf = create_testing_transform();
        let transformed_point = test_transf.transform_point(Point { x: 2.0, y: 2.0 });
        let true_transformed_point = Point { x: 1.5, y: 2.0 };
        let src_points = test_transf.source.clone();
        let dst_points = test_transf.destination.clone();
        for (src_point, dst_point) in zip(src_points, dst_points) {
            let transformed_point = test_transf.transform_point(src_point);
            assert!((transformed_point.x - dst_point.x) < 0.00001)
        }
    }
}
