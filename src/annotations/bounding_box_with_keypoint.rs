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
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        keypoint_x: f32,
        keypoint_y: f32,
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
    fn left(&self) -> f32 {
        self.bounding_box.left()
    }

    fn top(&self) -> f32 {
        self.bounding_box.top()
    }

    fn right(&self) -> f32 {
        self.bounding_box.right()
    }

    fn bottom(&self) -> f32 {
        self.bounding_box.bottom()
    }

    fn category(&self) -> &str {
        self.bounding_box.category()
    }

    fn area(&self) -> f32 {
        self.bounding_box.area()
    }

    fn center(&self) -> (f32, f32) {
        self.bounding_box.center()
    }

    fn as_xyxy(&self) -> (f32, f32, f32, f32) {
        self.bounding_box.as_xyxy()
    }

    fn intersection_area<T: BoundingBoxGeometry> (&self, other: &T) -> f32 {
        self.intersection_area(other)
    }

    fn union_area<T: BoundingBoxGeometry> (&self, other: &T) -> f32 {
        self.union_area(other)
    }

    fn intersection_over_union<T: BoundingBoxGeometry> (&self, other: &T) -> f32 {
        self.intersection_over_union(other)
    }
}
