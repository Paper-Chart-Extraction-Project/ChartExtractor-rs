/// A struct representing a bounding box.
/// 
/// A bounding box is a rectangle used to annotate objects in images for training deep object
/// detection models. An ideal bounding box is the smallest box that totally contains the
/// object within the image. Bounding boxes are composed of a rectangle and a category denoting
/// what object it is.
///
/// This project uses the standard convention of the left side of the image being x=0 and the top
/// of the image being y=0.
#[derive(Debug)]
pub struct BoundingBox {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
    category: String
}

impl BoundingBox {
    /// Checks if a box has valid parameters before constructing.
    pub fn new(
        left: f64, top: f64, right: f64, bottom: f64, category: String
    ) -> Result<Self, String> {
        if left > right {
            Err(format!(
                "Failed to create BoundingBox, value for left > value for right ({} > {}).",
                left,
                right
            ))
        } else if top > bottom {
            Err(format!(
                "Failed to create BoundingBox, value for top > value for bottom ({} > {}).",
                top,
                bottom
            ))
        } else {
            Ok(BoundingBox {left, top, right, bottom, category})
        }
    }

    pub fn left(&self) -> f64 {
        self.left
    }

    pub fn top(&self) -> f64 {
        self.top
    }

    pub fn right(&self) -> f64 {
        self.right
    }

    pub fn bottom(&self) -> f64 {
        self.bottom
    }

    pub fn category(&self) -> String {
        self.category
    }
}
