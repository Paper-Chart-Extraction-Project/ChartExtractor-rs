use crate::annotations::bounding_box::BoundingBoxGeometry;
use crate::annotations::detection::Detection;
use ndarray::{ArrayBase, Dim, ViewRepr};
use std::fmt::Display;

/// Defines a trait that all object detection models must follow.
pub trait ObjectDetectionModel<T: BoundingBoxGeometry + Display> {
    /// run_inference does not take an array directly, but rather a view into an array.
    /// For our use case, we will be doing a lot of cropping images by making views
    /// into ndarray objects. This is so we are able to do tiled detection on an image
    /// without making copies.
    ///
    /// image.slice(s![.., .., start_row..end_row, start_col..end_col])
    /// where image is an ndarray with dimensions (1, 3, image_width, image_height).
    ///
    /// If you want to reuse this and skip the view making process, changing ViewRepr<f32>
    /// to OwnedRepr<f32> will likely work.
    fn run_inference(
        &self,
        input_array: ArrayBase<ViewRepr<&f32>, Dim<[usize; 4]>>,
        confidence: f32,
    ) -> Vec<Detection<T>>;
}
