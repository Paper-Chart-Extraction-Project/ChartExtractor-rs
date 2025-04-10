use crate::annotations::bounding_box::BoundingBoxGeometry;
use crate::annotations::bounding_box_with_keypoint::BoundingBoxWithKeypoint;
use crate::annotations::detection::Detection;
use crate::object_detection::object_detection_model::ObjectDetectionModel;
use crate::object_detection::ort_inference_session::OrtInferenceSession;
use ndarray::{ArrayBase, Axis, Dim, ViewRepr};
use ort::{inputs, session::SessionOutputs};
use std::fmt::Display;
use std::path::Path;

pub struct Yolov11PoseEstimation {
    ort_session: OrtInferenceSession,
    class_names: Vec<String>,
    input_width: usize,
    input_height: usize,
    model_name: String,
}

impl Yolov11PoseEstimation {
    pub fn new(
        model_path: &Path,
        class_names: Vec<String>,
        input_width: usize,
        input_height: usize,
        model_name: String,
    ) -> ort::Result<Self> {
        let ort_session = OrtInferenceSession::new(model_path)?;
        Ok(Yolov11PoseEstimation {
            ort_session,
            class_names,
            input_width,
            input_height,
            model_name,
        })
    }
}

impl ObjectDetectionModel<BoundingBoxWithKeypoint> for Yolov11PoseEstimation {
    fn run_inference(
        &self,
        input_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>,
        confidence: f32,
    ) -> Vec<Detection<BoundingBoxWithKeypoint>> {
        let outputs: SessionOutputs = self
            .ort_session
            .session
            .run(inputs!["images" => input_array].unwrap())
            .unwrap();
        let output = outputs["output0"].try_extract_tensor::<f32>().unwrap();
        let output = output.t();
        let mut detections: Vec<Detection<BoundingBoxWithKeypoint>> = Vec::new();
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<f32> = row.iter().copied().collect();
            println!("Row: {:?}", row);
            let class_id = 0;
            let prob = row[4];

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
            let kpx = row[5];
            let kpy = row[6];
            let _ = row[7]; //Keypoint probability.

            let bbox_wkp = BoundingBoxWithKeypoint::new(
                x - (w / 2.0),
                y - (h / 2.0),
                x + (w / 2.0),
                y + (h / 2.0),
                kpx,
                kpy,
                label.to_string(),
            );
            detections.push(Detection {
                annotation: bbox_wkp.unwrap(),
                confidence: prob,
            });
        }
        detections
    }
}
