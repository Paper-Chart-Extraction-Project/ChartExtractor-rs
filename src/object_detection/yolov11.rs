use crate::annotations::bounding_box::BoundingBox;
use crate::annotations::detection::Detection;
use ndarray::{Array4, ArrayBase, Axis, Dim, s, ViewRepr};
use ort::inputs;
use ort::session::{Session, SessionOutputs};
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
        &self, input_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>, confidence: f32
    ) -> Vec<Detection<BoundingBox>> {
        let outputs: SessionOutputs = self.ort_session.session.run(
            inputs!["images" => input_array].unwrap()
        ).unwrap();
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
                .reduce(|accum, row| if row.1 > accum.1 {row} else {accum})
                .unwrap();
            if prob < confidence {
                continue;
            }
            let label = match self.class_names.get(class_id) {
                Some(v) => v,
                None => &class_id.to_string()
            };
            let x = row[0];
            let y = row[1];
            let w = row[2];
            let h = row[3];
            let bbox = BoundingBox::new(x - (w/2.0), y - (h/2.0), x + (w/2.0), y + (y/2.0), label.to_string());
            detections.push(Detection {annotation: bbox.unwrap(), confidence: prob});
        }
        detections
    }
}
