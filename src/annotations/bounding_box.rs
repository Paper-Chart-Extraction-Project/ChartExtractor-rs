use crate::annotations::point::Point;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug, PartialEq)]
pub enum BoundingBoxError {
    InvalidLeftRight { left: f32, right: f32 },
    InvalidTopBottom { top: f32, bottom: f32 },
}

impl fmt::Display for BoundingBoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundingBoxError::InvalidLeftRight { left, right } => {
                write!(
                    f,
                    "Failed to create BoundingBox, left ({}) > right ({}).",
                    left, right
                )
            }
            BoundingBoxError::InvalidTopBottom { top, bottom } => {
                write!(
                    f,
                    "Failed to create BoundingBox, top ({}) > bottom ({}).",
                    top, bottom
                )
            }
        }
    }
}

impl std::error::Error for BoundingBoxError {}

/// A struct representing a bounding box.
///
/// A bounding box is a rectangle used to annotate objects in images for training deep object
/// detection models. An ideal bounding box is the smallest box that totally contains the
/// object within the image. Bounding boxes are composed of a rectangle and a category denoting
/// what object it is. When an object detection model runs, it will output bounding boxes as its
/// output along with a probability encoding its confidence in that box+category.
///
/// This project uses the standard convention of the left side of the image being x=0 and the top
/// of the image being y=0.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct BoundingBox {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    category: String,
}

impl BoundingBox {
    /// Checks if a box has valid parameters before constructing.
    pub fn new(
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        category: String,
    ) -> Result<Self, BoundingBoxError> {
        if left > right {
            Err(BoundingBoxError::InvalidLeftRight { left, right })
        } else if top > bottom {
            Err(BoundingBoxError::InvalidTopBottom { top, bottom })
        } else {
            Ok(BoundingBox {
                left,
                top,
                right,
                bottom,
                category,
            })
        }
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BoundingBox {{ left: {}, top: {}, right: {}, bottom: {}, category: {}}}",
            self.left(),
            self.top(),
            self.right(),
            self.bottom(),
            self.category()
        )
    }
}

/// A trait providing methods for computing attributes about boxes.
///
/// Any annotation that uses a bounding box as its base has "box like geometry", and therefore can
/// be passed in to useful functions like non-maximum suppression and intersection over union.
pub trait BoundingBoxGeometry {
    fn left(&self) -> f32;
    fn top(&self) -> f32;
    fn right(&self) -> f32;
    fn bottom(&self) -> f32;
    fn category(&self) -> &String;
    fn left_mut(&mut self) -> &mut f32;
    fn top_mut(&mut self) -> &mut f32;
    fn right_mut(&mut self) -> &mut f32;
    fn bottom_mut(&mut self) -> &mut f32;
    fn category_mut(&mut self) -> &mut String;
    fn area(&self) -> f32;
    fn center(&self) -> Point;
    fn as_xyxy(&self) -> (f32, f32, f32, f32);
    fn intersection_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32;
    fn union_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32;
    fn intersection_over_union<T: BoundingBoxGeometry>(&self, other: &T) -> f32;
}

impl BoundingBoxGeometry for BoundingBox {
    fn left(&self) -> f32 {
        self.left
    }
    fn top(&self) -> f32 {
        self.top
    }
    fn right(&self) -> f32 {
        self.right
    }
    fn bottom(&self) -> f32 {
        self.bottom
    }
    fn category(&self) -> &String {
        &self.category
    }

    fn left_mut(&mut self) -> &mut f32 {
        &mut self.left
    }
    fn top_mut(&mut self) -> &mut f32 {
        &mut self.top
    }
    fn right_mut(&mut self) -> &mut f32 {
        &mut self.right
    }
    fn bottom_mut(&mut self) -> &mut f32 {
        &mut self.bottom
    }
    fn category_mut(&mut self) -> &mut String {
        &mut self.category
    }

    fn area(&self) -> f32 {
        (self.right() - self.left()) * (self.bottom() - self.top())
    }

    fn center(&self) -> Point {
        Point {
            x: 0.5_f32 * (self.right() + self.left()),
            y: 0.5_f32 * (self.bottom() + self.top()),
        }
    }

    fn as_xyxy(&self) -> (f32, f32, f32, f32) {
        (self.left(), self.top(), self.right(), self.bottom())
    }

    fn intersection_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        let intersection_left = self.left().max(other.left());
        let intersection_top = self.top().max(other.top());
        let intersection_right = self.right().min(other.right());
        let intersection_bottom = self.bottom().min(other.bottom());

        let intersection_box = BoundingBox::new(
            intersection_left,
            intersection_top,
            intersection_right,
            intersection_bottom,
            String::from(""),
        );
        match intersection_box {
            Ok(b) => {
                return b.area();
            }
            Err(_) => {
                return 0_f32;
            }
        }
    }

    fn union_area<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        let intersection_area = self.intersection_area(other);
        self.area() + other.area() - intersection_area
    }

    fn intersection_over_union<T: BoundingBoxGeometry>(&self, other: &T) -> f32 {
        let intersection_area = self.intersection_area(other);
        let union_area = self.union_area(other);
        if union_area == 0_f32 {
            panic!("Both bounding boxes are degenerate union area is 0, so IOU is undefined.");
        }
        intersection_area / union_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_bbox() {
        let left = 0_f32;
        let top = 0_f32;
        let right = 1_f32;
        let bottom = 1_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test"));
        assert!(bbox.is_ok());
    }

    #[test]
    fn valid_degenerate_bbox() {
        // A degenerate polygon typically refers to one that is valid, but has 0 area.
        let left = 0_f32;
        let top = 0_f32;
        let right = 0_f32;
        let bottom = 100_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test"));
        assert!(bbox.is_ok());
    }

    #[test]
    fn invalid_left_right_bbox() {
        let left = 1_f32;
        let top = 0_f32;
        let right = 0_f32;
        let bottom = 1_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test"));
        assert_eq!(
            bbox,
            Err(BoundingBoxError::InvalidLeftRight { left, right })
        )
    }

    #[test]
    fn invalid_top_bottom_bbox() {
        let left = 0_f32;
        let top = 1_f32;
        let right = 1_f32;
        let bottom = 0_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test"));
        assert_eq!(
            bbox,
            Err(BoundingBoxError::InvalidTopBottom { top, bottom })
        )
    }

    #[test]
    fn area() {
        let left = 0_f32;
        let top = 0_f32;
        let right = 3_f32;
        let bottom = 2_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test")).unwrap();
        assert_eq!(bbox.area(), 6_f32);
    }

    #[test]
    fn degenerate_area() {
        let left = 0_f32;
        let top = 0_f32;
        let right = 0_f32;
        let bottom = 100_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test")).unwrap();
        assert_eq!(bbox.area(), 0_f32);
    }

    #[test]
    fn center() {
        let left = 0_f32;
        let top = 0_f32;
        let right = 2_f32;
        let bottom = 1_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test")).unwrap();
        assert_eq!(
            bbox.center(),
            Point {
                x: 1_f32,
                y: 0.5_f32
            }
        );
    }

    #[test]
    fn as_xyxy() {
        let left = 1_f32;
        let top = 0_f32;
        let right = 2_f32;
        let bottom = 1_f32;
        let bbox = BoundingBox::new(left, top, right, bottom, String::from("test")).unwrap();
        assert_eq!(bbox.as_xyxy(), (1_f32, 0_f32, 2_f32, 1_f32));
    }

    #[test]
    fn intersection_area_no_overlap() {
        let left_0 = 0_f32;
        let top_0 = 0_f32;
        let right_0 = 2_f32;
        let bottom_0 = 2_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 3_f32;
        let top_1 = 3_f32;
        let right_1 = 5_f32;
        let bottom_1 = 7_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_area(&bbox_1), 0_f32)
    }

    #[test]
    fn intersection_area_corners_overlap() {
        let left_0 = 1_f32;
        let top_0 = 3_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 1_f32;
        let right_1 = 5_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_area(&bbox_1), 1_f32);
        assert_eq!(bbox_1.intersection_area(&bbox_0), 1_f32);
    }

    #[test]
    fn intersection_area_nested() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 3_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_area(&bbox_1), 2_f32);
        assert_eq!(bbox_1.intersection_area(&bbox_0), 2_f32);
    }

    #[test]
    fn intersection_area_nested_degenerate() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 2_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_area(&bbox_1), 0_f32);
        assert_eq!(bbox_1.intersection_area(&bbox_0), 0_f32);
    }

    #[test]
    fn union_area_no_overlap() {
        let left_0 = 0_f32;
        let top_0 = 0_f32;
        let right_0 = 2_f32;
        let bottom_0 = 2_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 3_f32;
        let top_1 = 3_f32;
        let right_1 = 5_f32;
        let bottom_1 = 7_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.union_area(&bbox_1), 12_f32);
        assert_eq!(bbox_1.union_area(&bbox_0), 12_f32);
    }

    #[test]
    fn union_area_corners_overlap() {
        let left_0 = 1_f32;
        let top_0 = 3_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 1_f32;
        let right_1 = 5_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.union_area(&bbox_1), 12_f32);
        assert_eq!(bbox_1.union_area(&bbox_0), 12_f32);
    }

    #[test]
    fn union_area_nested() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 3_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.union_area(&bbox_1), 8_f32);
        assert_eq!(bbox_1.union_area(&bbox_0), 8_f32);
    }

    #[test]
    fn union_area_no_overlap_degenerate() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 5_f32;
        let top_1 = 4_f32;
        let right_1 = 5_f32;
        let bottom_1 = 8_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.union_area(&bbox_1), 8_f32);
        assert_eq!(bbox_1.union_area(&bbox_0), 8_f32);
    }

    #[test]
    fn iou_no_overlap() {
        let left_0 = 0_f32;
        let top_0 = 0_f32;
        let right_0 = 2_f32;
        let bottom_0 = 2_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 3_f32;
        let top_1 = 3_f32;
        let right_1 = 5_f32;
        let bottom_1 = 7_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_over_union(&bbox_1), 0_f32);
        assert_eq!(bbox_1.intersection_over_union(&bbox_0), 0_f32);
    }

    #[test]
    fn iou_corners_overlap() {
        let left_0 = 1_f32;
        let top_0 = 3_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 1_f32;
        let right_1 = 5_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_over_union(&bbox_1), 1_f32 / 12_f32);
        assert_eq!(bbox_1.intersection_over_union(&bbox_0), 1_f32 / 12_f32);
    }

    #[test]
    fn iou_nested() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 3_f32;
        let bottom_1 = 4_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_over_union(&bbox_1), 1_f32 / 4_f32);
        assert_eq!(bbox_1.intersection_over_union(&bbox_0), 1_f32 / 4_f32);
    }

    #[test]
    fn iou_one_box_degenerate() {
        let left_0 = 1_f32;
        let top_0 = 1_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 2_f32;
        let bottom_1 = 6_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_over_union(&bbox_1), 0_f32);
        assert_eq!(bbox_1.intersection_over_union(&bbox_0), 0_f32);
    }

    #[test]
    #[should_panic(
        expected = "Both bounding boxes are degenerate union area is 0, so IOU is undefined."
    )]
    fn iou_both_box_degenerate() {
        let left_0 = 1_f32;
        let top_0 = 5_f32;
        let right_0 = 3_f32;
        let bottom_0 = 5_f32;
        let bbox_0 =
            BoundingBox::new(left_0, top_0, right_0, bottom_0, String::from("test")).unwrap();

        let left_1 = 2_f32;
        let top_1 = 2_f32;
        let right_1 = 2_f32;
        let bottom_1 = 6_f32;
        let bbox_1 =
            BoundingBox::new(left_1, top_1, right_1, bottom_1, String::from("test")).unwrap();

        assert_eq!(bbox_0.intersection_over_union(&bbox_1), 0_f32);
        assert_eq!(bbox_1.intersection_over_union(&bbox_0), 0_f32);
    }
}
