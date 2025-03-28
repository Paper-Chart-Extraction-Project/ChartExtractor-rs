use crate::annotations::bounding_box::{BoundingBox};
use crate::annotations::point::{Point};

/// A struct representing a BoundingBox + Keypoint annotation.
///
/// Pose estimation models use a standard detection model as their base, and add on functionality
/// to place keypoints into the frame as well. Therefore, the output of pose models is both a
/// bounding box as well as a list of points relating to the "pose" of the object. For this project
/// we only have pose models that predict a single keypoint.
#[derive(Debug)]
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
    ) -> Result<BoundingBoxWithKeypoint, String> {
        let bbox: Result<BoundingBox, String> = BoundingBox::new(
            left, top, right, bottom, category
        );
        let bounding_box: BoundingBox = match bbox {
            Ok(b) => b,
            Err(s) => return Err(s)
        };
        let keypoint: Point = Point {x: keypoint_x, y: keypoint_y};
        Ok(BoundingBoxWithKeypoint {bounding_box, keypoint})
    }

    pub fn left(&self) -> f64 {
        self.bounding_box.left()
    }
    pub fn top(&self) -> f64 {
        self.bounding_box.top()
    }
    pub fn right(&self) -> f64 {
        self.bounding_box.right()
    }
    pub fn bottom(&self) -> f64 {
        self.bounding_box.bottom()
    }
    pub fn category(&self) -> &str {
        self.bounding_box.category()
    }
}
