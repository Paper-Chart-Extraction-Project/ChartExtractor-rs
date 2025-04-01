use crate::annotations::bounding_box::BoundingBox;
use crate::annotations::detection::Detection;
use ndarray::{Array4, Axis, s};
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;
use std::path::Path;

/// An onnxruntime inference session.
pub struct OrtInferenceSession {
    session: Session
}

impl OrtInferenceSession {
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self{session})
    }
}

pub struct Yolov11 {
    ort_session: OrtInferenceSession,
    class_names: Vec<String>,
    input_width: usize,
    input_height: usize,
    model_name: String
}

impl Yolov11 {
    pub fn new(
        model_path: &Path,
        class_names: Vec<String>,
        input_width: usize,
        input_height: usize,
        model_name: String
    ) -> ort::Result<Self> {
        let ort_session = OrtInferenceSession::new(model_path)?;
        Ok(Yolov11{ort_session, class_names, input_width, input_height, model_name})
    }

    pub fn run_inference(
        &self, input_array: Array4<f32>, confidence: f32
    ) -> Vec<Detection<BoundingBox>> {
        let mut detections: Vec<Detection<BoundingBox>> = Vec::new();
        let outputs: SessionOutputs = self.ort_session.session.run(
            inputs!["images" => TensorRef::from_array_view(&input_array)?]
        )?;
        let output = outputs.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4) // skips bounding box coords.
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 {row} else {accum})
                .unwrap();
            if prob < 0.5 {
                continue;
            }
            let label = self.class_names[class_id];
            let x = row[0];
            let y = row[1];
            let w = row[2];
            let h = row[3];
            let bbox = BoundingBox::new(
                x - (w/2.0),
                y - (h/2.0),
                x + (w/2.0),
                y + (y/2.0),
                label
            );
            detections.push(
                Detection {
                    annotation: bbox.unwrap(),
                    confidence: prob
                }
            );
        }
        detections
    }
}
