use crate::annotations::bounding_box::{BoundingBoxGeometry};

/// A detection is what is produced as output from an object detection model.
///
/// A detection is any annotation combined with a confidence score: a probability value that
/// encodes the model's belief that the detection is true.
#[derive(Debug)]
pub struct Detection<T: BoundingBoxGeometry> {
    pub annotation: T,
    pub confidence: f64
}
