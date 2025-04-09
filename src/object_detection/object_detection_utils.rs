use crate::annotations::bounding_box::{BoundingBox, BoundingBoxGeometry};
use crate::annotations::detection::Detection;
use crate::image_utils::tiling::{OverlapProportion, TilingError, tile_image};
use crate::object_detection::object_detection_model::ObjectDetectionModel;
use ndarray::{ArrayBase, Dim, OwnedRepr, ViewRepr};
use std::fs::File;
use std::fmt::Display;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Reads a file with the class names into a vector so that the number ids
/// which come directly from the ORT inference session can be given meaning.
pub fn read_classes_txt_file(filepath: &Path) -> io::Result<Vec<String>> {
    BufReader::new(File::open(filepath)?).lines().collect()
}

/// Non maxmimum suppression is a way of removing duplicate detections.
pub fn non_maximum_suppression<T: BoundingBoxGeometry + Display>(
    mut detections: Vec<Detection<T>>,
    iou_threshold: f32,
) -> Vec<Detection<T>> {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut detections_to_remove: Vec<bool> = vec![false; detections.len()];
    for (current_index, current_det) in detections.iter().enumerate() {
        for (other_index, other_det) in detections[current_index + 1..].iter().enumerate() {
            if detections_to_remove[current_index + other_index + 1] {
                continue;
            }
            if current_det.annotation.category() != other_det.annotation.category() {
                continue;
            }
            let iou = current_det
                .annotation
                .intersection_over_union(&other_det.annotation);
            if iou > iou_threshold {
                detections_to_remove[current_index + other_index + 1] = true;
            }
        }
    }
    let mut drop_iter = detections_to_remove.iter();
    detections.retain(|_| !drop_iter.next().unwrap());
    detections
}

/// Predicts small objects on an image using image tiling.
///
/// Tiles an image, predicts on each tile, then corrects the detection's coordinates and
/// applies NMS to them.
pub fn tile_and_predict<T: BoundingBoxGeometry + Display, U: ObjectDetectionModel<T>>(
    model: &U,
    image_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    tile_size: u32,
    overlap_proportion: OverlapProportion,
    confidence: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<Detection<T>>, TilingError> {
    let tiles: Vec<Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>>> =
        tile_image(&image_array, tile_size, overlap_proportion)?;
    let stride: u32 = (tile_size * overlap_proportion.numerator) / overlap_proportion.denominator;
    let mut detections: Vec<Detection<T>> = Vec::new();
    for (row_ix, row_of_tiles) in tiles.iter().enumerate() {
        for (col_ix, tile) in row_of_tiles.iter().enumerate() {
            let preds = model.run_inference(*tile, confidence);
            for mut pred in preds {
                let x_correction = ((col_ix as u32) * stride) as f32;
                let y_correction = ((row_ix as u32) * stride) as f32;
                *pred.annotation.left_mut() += x_correction;
                *pred.annotation.top_mut() += y_correction;
                *pred.annotation.right_mut() += x_correction;
                *pred.annotation.bottom_mut() += y_correction;
                detections.push(pred);
            }
        }
    }
    detections = non_maximum_suppression(detections, nms_iou_threshold);
    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_no_overlap() {
        let dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 1_f32, 1_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
            Detection {
                annotation: BoundingBox::new(2_f32, 2_f32, 3_f32, 3_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
        ];
        let nms_result = non_maximum_suppression(dets, 0.5_f32);
        let true_dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 1_f32, 1_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
            Detection {
                annotation: BoundingBox::new(2_f32, 2_f32, 3_f32, 3_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
        ];
        assert_eq!(true_dets, nms_result);
    }

    #[test]
    fn nms_standard_usage() {
        let dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 4_f32, 4_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 5_f32, 5_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.55_f32,
            },
            Detection {
                annotation: BoundingBox::new(6_f32, 6_f32, 10_f32, 10_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.75_f32,
            },
        ];
        let nms_result = non_maximum_suppression(dets, 0.5_f32);
        let true_dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(6_f32, 6_f32, 10_f32, 10_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.75_f32,
            },
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 4_f32, 4_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
        ];
        assert_eq!(true_dets, nms_result);
    }

    #[test]
    fn nms_overlap_but_different_classes() {
        let dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 4.5_f32, 4.5_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
            Detection {
                annotation: BoundingBox::new(
                    0_f32,
                    0_f32,
                    5_f32,
                    5_f32,
                    "test_different_class".to_string(),
                )
                .unwrap(),
                confidence: 0.55_f32,
            },
            Detection {
                annotation: BoundingBox::new(0.5_f32, 0.5_f32, 4_f32, 4_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.8_f32,
            },
            Detection {
                annotation: BoundingBox::new(6_f32, 6_f32, 10_f32, 10_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.75_f32,
            },
        ];
        let nms_result = non_maximum_suppression(dets, 0.5_f32);
        let true_dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0.5_f32, 0.5_f32, 4_f32, 4_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.8_f32,
            },
            Detection {
                annotation: BoundingBox::new(6_f32, 6_f32, 10_f32, 10_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.75_f32,
            },
            Detection {
                annotation: BoundingBox::new(
                    0_f32,
                    0_f32,
                    5_f32,
                    5_f32,
                    "test_different_class".to_string(),
                )
                .unwrap(),
                confidence: 0.55_f32,
            },
        ];
        assert_eq!(true_dets, nms_result);
    }
}
