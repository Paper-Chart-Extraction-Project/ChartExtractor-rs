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

