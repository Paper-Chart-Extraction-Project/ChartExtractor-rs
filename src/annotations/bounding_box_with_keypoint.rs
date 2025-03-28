use crate::annotations::bounding_box::{BoundingBox, BoundingBoxError, BoundingBoxGeometry};
use crate::annotations::point::Point;
use std::fmt;

/// A struct representing a BoundingBox + Keypoint annotation.
///
/// Pose estimation models use a standard detection model as their base, and add on functionality
/// to place keypoints into the frame as well. Therefore, the output of pose models is both a
/// bounding box as well as a list of points relating to the "pose" of the object. For this project
/// we only have pose models that predict a single keypoint.
#[derive(Debug)]
pub struct BoundingBoxWithKeypoint {
    bounding_box: BoundingBox,
    keypoint: Point,
}

impl BoundingBoxWithKeypoint {
    pub fn new(
        left: f64,
        top: f64,
        right: f64,
        bottom: f64,
        keypoint_x: f64,
        keypoint_y: f64,
        category: String,
    ) -> Result<BoundingBoxWithKeypoint, BoundingBoxError> {
        Ok(BoundingBoxWithKeypoint {
            bounding_box: BoundingBox::new(left, top, right, bottom, category)?,
            keypoint: Point {
                x: keypoint_x,
                y: keypoint_y,
            },
        })
    }
}

impl fmt::Display for BoundingBoxWithKeypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoundingBoxWithKeypoint {{ bounding_box: {}, keypoint: {}}}",
            self.bounding_box, self.keypoint
        )
    }
}

impl BoundingBoxGeometry for BoundingBoxWithKeypoint {
    fn left(&self) -> f64 {
        self.bounding_box.left()
    }

    fn top(&self) -> f64 {
        self.bounding_box.top()
    }

    fn right(&self) -> f64 {
        self.bounding_box.right()
    }

    fn bottom(&self) -> f64 {
        self.bounding_box.bottom()
    }

    fn category(&self) -> &str {
        self.bounding_box.category()
    }

    fn area(&self) -> f64 {
        self.bounding_box.area()
    }

    fn center(&self) -> (f64, f64) {
        self.bounding_box.center()
    }

    fn as_xyxy(&self) -> (f64, f64, f64, f64) {
        self.bounding_box.as_xyxy()
    }
}
