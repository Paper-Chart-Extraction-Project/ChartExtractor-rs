use crate::annotations::bounding_box::{BoundingBox, BoundingBoxError, BoundingBoxGeometry};
use crate::annotations::point::Point;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A struct representing a BoundingBox + Keypoint annotation.
///
/// Pose estimation models use a standard detection model as their base, and add on functionality
/// to place keypoints into the frame as well. Therefore, the output of pose models is both a
/// bounding box as well as a list of points relating to the "pose" of the object. For this project
/// we only have pose models that predict a single keypoint.
#[derive(Debug, Deserialize, Serialize)]
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

impl BoundingBoxWithKeypoint {
    pub fn get_keypoint_x(&self) -> f32 {
        self.keypoint.x
    }
    pub fn get_keypoint_y(&self) -> f32 {
        self.keypoint.y
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
    fn category(&self) -> &String {
        self.bounding_box.category()
    }

    fn left_mut(&mut self) -> &mut f32 {
        self.bounding_box.left_mut()
    }
    fn top_mut(&mut self) -> &mut f32 {
        self.bounding_box.top_mut()
    }
    fn right_mut(&mut self) -> &mut f32 {
        self.bounding_box.right_mut()
    }
    fn bottom_mut(&mut self) -> &mut f32 {
        self.bounding_box.bottom_mut()
    }
    fn category_mut(&mut self) -> &mut String {
        self.bounding_box.category_mut()
    }

    fn area(&self) -> f32 {
        self.bounding_box.area()
    }

    fn center(&self) -> Point {
        self.bounding_box.center()
    }

    fn as_xyxy(&self) -> (f32, f32, f32, f32) {
        self.bounding_box.as_xyxy()
    }

    fn intersection_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        self.bounding_box.intersection_area(other)
    }

    fn union_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        self.bounding_box.union_area(other)
    }

    fn intersection_over_union<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        self.bounding_box.intersection_over_union(other)
    }
}
