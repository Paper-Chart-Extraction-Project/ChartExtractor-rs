use crate::annotations::bounding_box::BoundingBox;
use crate::annotations::detection::Detection;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use ort::session::{Session, SessionInputs, SessionInputValue, SessionOutputs};
use ort::session::builder::SessionBuilder;
use ort::value::{Tensor, Value};
use std::borrow::Cow;
use std::path::Path;
use std::time::Instant;

/// An onnxruntime inference session.
pub struct OrtInferenceSession {
    session: Session
}

impl OrtInferenceSession {
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session = SessionBuilder::new()?.commit_from_file(model_path)?;
        Ok(Self{session})
    }

    pub fn run_inference(
        &self,
        input_image: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
        time_inference: Option<bool>
    ) -> ort::Result<SessionOutputs> {
        match time_inference {
            Some(t) => t,
            None => false
        }
        if time_inference {
            let time_pre_compute = Instant::now();
        }
        let shape = input_image.shape().to_vec();
        let raw_data = input_image.as_slice().unwrap().to_vec();
        let input_tensor = Tensor::from_array((shape, raw_data.into_boxed_slice()))?;
        let inputs: Vec<<Cow<str>, SessionInputValue)> = vec![
            (Cow::Borrowed("images"), input_values)
        ];
        let outputs: SessionOutputs = self.session.run(SessionInputs::from(inputs))?;
        if time_inference {
            let time_post_compute = Instant::now();
            println!("Inference Time: {:#?}", time_post_compute - time_pre_compute);
        }

        Ok(outputs)
    }
}

pub struct Yolov11 {
    ort_session: OrtInferenceSession,
    input_size: (u32, u32),
    use_nms: bool,
    model_name: String
}

impl Yolov11 {
    pub fn new(
        model_path: &Path,
        input_size: (u32, u32),
        use_nms: bool,
        model_name: String
    ) -> ort::Result<Self> {
        let session = OrtInferenceSession::new(model_path)?;
        Ok(Yolov11{session, input_size, use_nms, model_name})
    }

    pub fn run_inference(&self, input_tensor: Array4<f32>) -> Vec<Detection> {
        let outputs = self
            .session
            .run_inference(input_tensor)
            .expect("Inference failed");
        let output = outputs["output0"]
            .try_exact_tensor::<f32>()
            .expect("Failed to extract tensor")
            .into_owned();
        let mut boxes = Vec::new();
        println!("Output shape: {:?}", output.shape());

        let original_output_shape = output.shape();
        let reshaped_output = output
            .to_shape((original_output_shape[1], original_output_shape[2]))
            .expect("Failed to reshape the output.");
        for i in 0..reshaped_output.shape()[1] {
            let x = reshaped_output[[0, i]];
            let y = reshaped_output[[1, i]];
            let w = reshaped_output[[2, i]];
            let h = reshaped_output[[3, i]];

            let mut max_class_prob = 0.0;
            let mut max_class_id = 0;

            for j in 4..reshaped_output.shape()[0] {
                if reshaped_output[[j, i]] > max_class_prob {
                    max_class_prob = reshaped_output[[j, i]];
                    max_class_id = j - 4;
                }
            }

            if max_class_prob > 0.25 {
                let bbox = BoundingBox::new(
                    left: (x - w/2.0) as f64,
                    top: (y - h/2.0) as f64,
                    right: (x + w/2.0) as f64,
                    bottom: (y + h/2.0) as f64,
                    category: String::from(max_class_id)
                );
                let detection = Detection {
                    annotation: bbox,
                    confidence: max_class_prob as f64
                };
                boxes.push(detection);
            }
        }
        boxes
    }
}
