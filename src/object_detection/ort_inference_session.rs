use ort::session::Session;
use std::path::Path;

/// An onnxruntime inference session.
///
/// All of the object detection classes in this project are just wrappers
/// around an ONNX inference session that handles running the model on
/// hardware.
pub struct OrtInferenceSession {
    pub session: Session,
}

impl OrtInferenceSession {
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }
}
