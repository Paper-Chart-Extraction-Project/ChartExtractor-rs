use std::fmt;

/// A set of custom errors for more informative error handling.
#[derive(Debug)]
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
#[derive(Debug)]
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
            self.left, self.top, self.right, self.bottom, self.category
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
    fn category(&self) -> &str;
    fn area(&self) -> f32;
    fn center(&self) -> (f32, f32);
    fn as_xyxy(&self) -> (f32, f32, f32, f32);
    fn intersection_area<T: BoundingBoxGeometry> (&self, other: &T);
    fn union_area<T: BoundingBoxGeometry> (&self, other: &T);
    fn intersection_over_union(&self) -> f32;
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

    fn category(&self) -> &str {
        &self.category
    }

    fn area(&self) -> f32 {
        (self.right - self.left) * (self.bottom - self.top)
    }

    fn center(&self) -> (f32, f32) {
        (
            0.5_f32 * (self.right - self.left),
            0.5_f32 * (self.bottom - self.top),
        )
    }

    fn as_xyxy(&self) -> (f32, f32, f32, f32) {
        (self.left, self.top, self.right, self.bottom)
    }
}
