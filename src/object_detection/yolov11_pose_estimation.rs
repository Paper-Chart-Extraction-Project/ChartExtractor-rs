use crate::annotations::bounding_box::{BoundingBox, BoundingBoxGeometry};
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
