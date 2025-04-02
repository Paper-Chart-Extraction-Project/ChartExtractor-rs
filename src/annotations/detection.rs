use crate::annotations::bounding_box::BoundingBoxGeometry;
use std::fmt;

/// A detection is what is produced as output from an object detection model.
///
/// A detection is any annotation combined with a confidence score: a probability value that
/// encodes the model's belief that the detection is true.
#[derive(Debug)]
pub struct Detection<T: BoundingBoxGeometry + fmt::Display> {
    pub annotation: T,
    pub confidence: f32,
}

impl<T: BoundingBoxGeometry + fmt::Display> fmt::Display for Detection<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Detection {{ annotation: {}, confidence: {} }}",
            self.annotation, self.confidence
        )
    }
}
