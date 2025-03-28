use annotations::bounding_box::{BoundingBox};
use annotations::point::{Point};

/// A struct representing a BoundingBox + Keypoint annotation.
///
/// Pose estimation models use a standard detection model as their base, and add on functionality
/// to place keypoints into the frame as well. Therefore, the output of pose models is both a
/// bounding box as well as a list of points relating to the "pose" of the object. For this project
/// we only have pose models that predict a single keypoint.
#[Derive(Debug)]
pub struct BoundingBoxWithKeypoint {
    bounding_box: BoundingBox,
    keypoint: Point
}

impl BoundingBoxWithKeypoint {
    pub fn new(
        left: f64,
        top: f64,
        right: f64,
        bottom: f64,
        keypoint_x: f64,
        keypoint_y: f64,
        category: String
    ) -> Result<BoundingBoxWithKeypoint, String> {}

    pub fn left() -> f64 {}
    pub fn top() -> f64 {}
    pub fn right() -> f64 {}
    pub fn bottom() -> f64 {}
    pub fn category() -> f64 {}
}
