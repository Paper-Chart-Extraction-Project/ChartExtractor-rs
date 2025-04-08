use crate::annotations::bounding_box::{BoundingBox, BoundingBoxGeometry};
use crate::annotations::detection::Detection;
use crate::image_utils::tiling::{OverlapProportion, TilingError, tile_image};
use ndarray::{ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// An onnxruntime inference session.
pub struct OrtInferenceSession {
    session: Session,
}

impl OrtInferenceSession {
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }
}

pub struct Yolov11 {
    ort_session: OrtInferenceSession,
    class_names: Vec<String>,
    input_width: usize,
    input_height: usize,
    model_name: String,
}

impl Yolov11 {
    pub fn new(
        model_path: &Path,
        class_names: Vec<String>,
        input_width: usize,
        input_height: usize,
        model_name: String,
    ) -> ort::Result<Self> {
        let ort_session = OrtInferenceSession::new(model_path)?;
        Ok(Yolov11 {
            ort_session,
            class_names,
            input_width,
            input_height,
            model_name,
        })
    }

    /// This method does not take an array directly, but rather a view into an array.
    /// For our use case, we will be doing a lot of cropping images by making views
    /// into ndarray objects. The purpose of this is to be able to do tiled detection
    /// on an image without making copies.
    ///
    /// image.slice(s![.., .., start_row..end_row, start_col..end_col])
    /// where image is an ndarray with dimensions (1, 3, image_width, image_height).
    ///
    /// If you want to reuse this and skip the view making process, changing ViewRepr<f32>
    /// to OwnedRepr<f32> will likely work.
    pub fn run_inference(
        &self,
        input_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>,
        confidence: f32,
    ) -> Vec<Detection<BoundingBox>> {
        let outputs: SessionOutputs = self
            .ort_session
            .session
            .run(inputs!["images" => input_array].unwrap())
            .unwrap();
        let output = outputs["output0"].try_extract_tensor::<f32>().unwrap();
        let output = output.t();

        let mut detections: Vec<Detection<BoundingBox>> = Vec::new();
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4) // skips bounding box coords.
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < confidence {
                continue;
            }
            let label = match self.class_names.get(class_id) {
                Some(v) => v,
                None => &class_id.to_string(),
            };
            let x = row[0];
            let y = row[1];
            let w = row[2];
            let h = row[3];
            let bbox = BoundingBox::new(
                x - (w / 2.0),
                y - (h / 2.0),
                x + (w / 2.0),
                y + (y / 2.0),
                label.to_string(),
            );
            detections.push(Detection {
                annotation: bbox.unwrap(),
                confidence: prob,
            });
        }
        detections = non_maximum_suppression(detections, 0.5_f32);
        detections
    }
}

pub fn read_classes_txt_file(filepath: &Path) -> io::Result<Vec<String>> {
    BufReader::new(File::open(filepath)?).lines().collect()
}

/// Non maxmimum suppression is a way of removing duplicate detections.
pub fn non_maximum_suppression<T: BoundingBoxGeometry + std::fmt::Display>(
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
pub fn tile_and_predict(
    model: &Yolov11,
    image_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    tile_size: u32,
    overlap_proportion: OverlapProportion,
    confidence: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<Detection<BoundingBox>>, TilingError> {
    let tiles: Vec<Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>>> =
        tile_image(&image_array, tile_size, overlap_proportion)?;
    let stride = tile_size / (overlap_proportion as u32);
    let mut detections: Vec<Detection<BoundingBox>> = Vec::new();
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

    fn nms_standard_usage() {
        let dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 3_f32, 3_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.6_f32,
            },
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 5_f32, 5_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.55_f32,
            },
            Detection {
                annotation: BoundingBox::new(2_f32, 2_f32, 4_f32, 4_f32, "test".to_string())
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
                annotation: BoundingBox::new(2_f32, 2_f32, 4_f32, 4_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.8_f32,
            },
            Detection {
                annotation: BoundingBox::new(6_f32, 6_f32, 10_f32, 10_f32, "test".to_string())
                    .unwrap(),
                confidence: 0.75_f32,
            },
        ];
        assert_eq!(true_dets, nms_result);
    }

    fn nms_overlap_but_different_classes() {
        let dets: Vec<Detection<BoundingBox>> = vec![
            Detection {
                annotation: BoundingBox::new(0_f32, 0_f32, 3_f32, 3_f32, "test".to_string())
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
                annotation: BoundingBox::new(2_f32, 2_f32, 4_f32, 4_f32, "test".to_string())
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
                annotation: BoundingBox::new(2_f32, 2_f32, 4_f32, 4_f32, "test".to_string())
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
